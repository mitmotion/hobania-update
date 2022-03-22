use crate::{comp::Controller, util::Dir};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use specs::Component;
use specs_idvs::IdvStorage;
use std::{collections::VecDeque, time::Duration};

pub type ControlCommands = VecDeque<ControlCommand>;

/// Controller messages are real-time-combat relevant.
/// Controller contains all kind of inputs a entity can change
/// Remote controlled entities (e.g. clients) have a latency
/// They can predict the future locally and send us their TIMED controls
/// Other things, like chunk requests or build a weapon are done one other
/// channel via other methods.
#[derive(Debug)]
pub struct RemoteController {
    ///sorted, deduplicated
    commands: ControlCommands,
    existing_commands: HashSet<u64>,
    max_hold: Duration,
    avg_latency: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlCommand {
    id: u64,
    /// *Time* when Controller should be applied
    action_time: Duration,
    /// *ContinuousMonotonicTime* when this msg was first send to remote
    first_send_monotonic_time: Option<Duration>,
    /// action_time - *Time* when this msg was first send to remote
    first_send_simulate_ahead_time: Option<f64>,
    /// *ContinuousMonotonicTime* when this msg was first acked by remote
    first_acked_monotonic_time: Option<Duration>,
    msg: Controller,
}

impl RemoteController {
    /// delete old and outdated( older than server_time) commands
    pub fn maintain(&mut self, server_time: Option<Duration>) {
        self.commands.make_contiguous();
        if let Some(last) = self.commands.back() {
            let min_allowed_age = last.action_time;

            while let Some(first) = self.commands.front() {
                if first.action_time + self.max_hold < min_allowed_age
                    || matches!(server_time, Some(x) if first.action_time < x)
                {
                    self.commands
                        .pop_front()
                        .map(|c| self.existing_commands.remove(&c.id));
                } else {
                    break;
                }
            }
        }
    }

    pub fn push(&mut self, command: ControlCommand) -> Option<u64> {
        let id = command.id;
        //check if command fits on the end.
        if self.existing_commands.contains(&id) {
            return None; // element already existed
        }
        match self
            .commands
            .binary_search_by_key(&command.action_time, |e| e.action_time)
        {
            Ok(_) => None,
            Err(i) => {
                self.existing_commands.insert(id);
                self.commands.insert(i, command);

                Some(id)
            },
        }
    }

    pub fn append(&mut self, commands: ControlCommands) -> HashSet<u64> {
        let mut result = HashSet::new();
        for command in commands {
            let id = command.id;
            if !self.existing_commands.contains(&id) {
                // TODO: improve algorithm to move binary search out of loop
                if let Err(i) = self
                    .commands
                    .binary_search_by_key(&command.action_time, |e| e.action_time)
                {
                    result.insert(id);
                    self.existing_commands.insert(id);
                    self.commands.insert(i, command);
                }
            }
        }
        result
    }

    /// remote confirmed arrival of commands. (not that they actually were used)
    /// we need to hold them, (especially with high lag) as we might revert to
    /// server-time.
    pub fn acked(
        &mut self,
        ids: HashSet<u64>,
        monotonic_time: Duration,
        highest_ahead_command: f64,
    ) {
        // calculating avg_latency.
        //  - server returns the time the furthest away command has left till it becomes
        //    active. As the server time is constant and we stored this value locally,
        //    we use it to calculate the first ahead_command for the first command in
        //    this batch.
        //  - we try to keep the remote_ahead in a specific window to allow for
        //    retransmittion and remote tick time

        // find time it took from client -> server
        let high_filter = self
            .commands
            .iter()
            .filter(|c| ids.contains(&c.id) && c.first_acked_monotonic_time.is_none());
        if let Some(highest) = high_filter.max_by_key(|c| c.action_time) {
            let remote_time = highest.action_time.as_secs_f64() - highest_ahead_command;
            //let latency = highest.first_send_simulate_ahead_time.unwrap_or_default() -
            // remote_time;
            common_base::plot!("prob_remote_time", remote_time);
            //common_base::plot!("prob_avg_latency", latency);
            //let low_filter = self.commands.iter().filter(|c| ids.contains(&c.id) &&
            // c.first_send_monotonic_time == highest.first_send_monotonic_time);
            let low_filter = self
                .commands
                .iter()
                .filter(|c| ids.contains(&c.id) && c.first_acked_monotonic_time.is_none());
            if let Some(lowest) = low_filter.min_by_key(|c| c.action_time) {
                let low_high_diff = highest.action_time - lowest.action_time;
                // if this is 50ms, it means the lowest command arrived at server 50 before it
                // was used
                common_base::plot!("highest_ahead_command", highest_ahead_command);
                let ahead_time = highest_ahead_command - low_high_diff.as_secs_f64();
                common_base::plot!("low_high_diff", low_high_diff.as_secs_f64());
                common_base::plot!("ahead_time", ahead_time);
                const LOWER_END: f64 = 0.18;
                const UPPER_END: f64 = 0.22;
                //if ahead_time > 2.0 {
                let len = self.commands.len();
                tracing::error!(
                    ?ahead_time,
                    ?ids,
                    ?highest_ahead_command,
                    ?highest,
                    ?lowest,
                    ?len,
                    "bigger2"
                );
                //}
                if ahead_time < LOWER_END {
                    self.avg_latency += Duration::from_millis(10);
                }
                if ahead_time > UPPER_END && self.avg_latency > Duration::from_millis(3) {
                    self.avg_latency -= Duration::from_millis(3);
                }
            }
        }

        for c in &mut self.commands {
            c.first_acked_monotonic_time = Some(monotonic_time);
        }

        /*
        let mut lowest_monotonic_time = None;
        for c in &mut self.commands {
            let new_acked = c.first_acked_monotonic_time.is_none() && ids.contains(&c.id);
            if new_acked {
                c.first_acked_monotonic_time = Some(monotonic_time);
                if c.first_send_monotonic_time.map_or(false, |mt| {
                    mt < lowest_monotonic_time.unwrap_or(monotonic_time)
                }) {
                    lowest_monotonic_time = c.first_send_monotonic_time;
                }
            }
        }
        if let Some(lowest_monotonic_time) = lowest_monotonic_time {
            if let Some(latency) = monotonic_time.checked_sub(lowest_monotonic_time) {
                common_base::plot!("latency", latency.as_secs_f64());
                self.avg_latency = ((99 * self.avg_latency + latency) / 100).max(latency);
            } else {
                warn!(
                    "latency is negative, this should never be the case as this value is not \
                     synced and monotonic!"
                );
            }
        }*/
        common_base::plot!("avg_latency", self.avg_latency.as_secs_f64());
    }

    /// prepare commands for sending
    /// only include commands, that were not yet acked!
    pub fn prepare_commands(&mut self, monotonic_time: Duration) -> ControlCommands {
        let simulate_ahead = self.simulate_ahead().as_secs_f64();
        self.commands
            .iter_mut()
            .filter(|c| c.first_acked_monotonic_time.is_none())
            .map(|c| {
                c.first_send_monotonic_time.get_or_insert(monotonic_time);
                c.first_send_simulate_ahead_time
                    .get_or_insert(simulate_ahead);
                &*c
            })
            .cloned()
            .collect()
    }

    pub fn commands(&self) -> &ControlCommands { &self.commands }

    /// Creates a single Controller event from all existing Controllers in the
    /// time range. E.g. when client with 60Hz sends: STAY JUMP STAY
    /// and server runs at 30Hz it could oversee the JUMP Control otherwise
    /// At the cost of a command to may be executed for longer
    /// on the server the client (1/30 instead 1/60)
    pub fn compress(&self, start: Duration, dt: Duration) -> Option<Controller> {
        // Assume we have 1 event every sec: 0,1,2,3,4
        // compressing   1s - 3s   should lead to index 1-2
        // compressing   1s - 2s   should lead to index 1 only
        // compressing   1s - 1.2s should lead to index 1 only
        // compressing 0.8s - 1.2s should lead to index 0-1
        let start_i = match self
            .commands
            .binary_search_by_key(&start, |e| e.action_time)
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        let end_exclusive_i = match self
            .commands
            .binary_search_by_key(&(start + dt), |e| e.action_time)
        {
            Ok(i) => i,
            Err(i) => i,
        };
        if self.commands.is_empty() || end_exclusive_i == start_i {
            return None;
        }
        let mut result = Controller::default();
        let mut look_dir = result.inputs.look_dir.to_vec();
        //if self.commands[start_i].source_time
        // Inputs are averaged over all elements by time
        // Queued Inputs are just added
        // Events and Actions are not included when the frame isn't fully inserted, must
        // not be duplicated
        let mut last_start = start;
        for i in start_i..end_exclusive_i {
            let e = &self.commands[i];
            let local_start = e.action_time.max(last_start);
            let local_end = if let Some(e) = self.commands.get(i + 1) {
                e.action_time.min(start + dt)
            } else {
                start + dt
            };
            let local_dur = local_end - local_start;
            result.inputs.move_dir += e.msg.inputs.move_dir * local_dur.as_secs_f32();
            result.inputs.move_z += e.msg.inputs.move_z * local_dur.as_secs_f32();
            look_dir = look_dir + e.msg.inputs.look_dir.to_vec() * local_dur.as_secs_f32();
            //TODO: manually combine 70% up and 30% down to UP
            result.inputs.climb = result.inputs.climb.or(e.msg.inputs.climb);
            result.inputs.break_block_pos = result
                .inputs
                .break_block_pos
                .or(e.msg.inputs.break_block_pos);
            result.inputs.strafing = result.inputs.strafing || e.msg.inputs.strafing;
            // we apply events from all that are started here.
            // if we only are part of 1 tick here we would assume that it was already
            // covered before
            if i != start_i || e.action_time >= start {
                result.actions.append(&mut e.msg.actions.clone());
                result.events.append(&mut e.msg.events.clone());
                result
                    .queued_inputs
                    .append(&mut e.msg.queued_inputs.clone());
            }
            last_start = local_start;
        }
        if result.actions.iter().any(|e| {
            if let crate::comp::ControlAction::StartInput {
                input,
                target_entity: _,
                select_pos: _,
            } = e
            {
                input == &crate::comp::InputKind::Jump
            } else {
                false
            }
        }) {
            tracing::error!("jump detencted");
        }
        result.inputs.move_dir /= dt.as_secs_f32();
        result.inputs.move_z /= dt.as_secs_f32();
        result.inputs.look_dir = Dir::new(look_dir.normalized());

        Some(result)
    }

    /// the average latency is 300ms by default and will be adjusted over time
    /// with every Ack the server sends its local tick time at the Ack
    /// so we can calculate how much time was spent for 1 way from
    /// server->client and assume that this is also true for client->server
    /// latency
    pub fn avg_latency(&self) -> Duration { self.avg_latency }

    pub fn simulate_ahead(&self) -> Duration {
        const FIXED_OFFSET: Duration = Duration::from_millis(0);
        self.avg_latency() + FIXED_OFFSET
    }
}

impl Default for RemoteController {
    fn default() -> Self {
        Self {
            commands: VecDeque::new(),
            existing_commands: HashSet::new(),
            max_hold: Duration::from_secs(5),
            avg_latency: Duration::from_millis(50),
        }
    }
}

impl Component for RemoteController {
    type Storage = IdvStorage<Self>;
}

#[derive(Default)]
pub struct CommandGenerator {
    id: u64,
}

impl CommandGenerator {
    pub fn gen(&mut self, action_time: Duration, msg: Controller) -> ControlCommand {
        self.id += 1;
        ControlCommand {
            action_time,
            first_send_monotonic_time: None,
            first_send_simulate_ahead_time: None,
            first_acked_monotonic_time: None,
            id: self.id,
            msg,
        }
    }
}

impl ControlCommand {
    pub fn msg(&self) -> &Controller { &self.msg }

    pub fn source_time(&self) -> Duration { self.action_time }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        comp,
        comp::{Climb, ControllerInputs},
    };
    use vek::{Vec2, Vec3};

    const INCREASE: Duration = Duration::from_millis(33);

    fn generate_control_cmds(count: usize) -> ControlCommands {
        let mut result = VecDeque::new();
        let mut generator = CommandGenerator::default();
        let mut time = Duration::new(0, 0);
        for i in 0..count {
            let mut msg = Controller::default();
            msg.inputs.move_dir = Vec2::new(i as f32 / 5.0, i as f32 / 10.0);
            msg.inputs.move_z = i as f32;
            if i == 7 {
                msg.inputs.break_block_pos = Some(Vec3::new(2.0, 3.0, 4.0));
            }
            if i > 6 {
                msg.inputs.climb = Some(comp::Climb::Up);
            }
            if i == 4 {
                msg.inputs.strafing = true;
            }
            if i < 5 {
                msg.events.push(comp::ControlEvent::EnableLantern);
            };
            if i >= 5 && i < 10 {
                msg.events.push(comp::ControlEvent::DisableLantern);
            }
            if i == 1 {
                msg.actions.push(comp::ControlAction::Dance);
            }
            if i == 9 {
                msg.actions.push(comp::ControlAction::Wield);
            }
            if i < 5 {
                msg.queued_inputs
                    .insert(comp::InputKind::Jump, comp::InputAttr {
                        select_pos: None,
                        target_entity: None,
                    });
            }
            if i >= 10 && i < 15 {
                msg.queued_inputs
                    .insert(comp::InputKind::Primary, comp::InputAttr {
                        select_pos: None,
                        target_entity: None,
                    });
            }
            let cc = generator.gen(time, msg);
            result.push_back(cc);
            time += INCREASE;
        }
        result
    }

    #[test]
    fn resend_data() {
        let data = generate_control_cmds(5);
        let mut list = RemoteController::default();
        assert_eq!(list.push(data[0].clone()), Some(1));
        assert_eq!(list.commands.len(), 1);

        assert_eq!(list.push(data[0].clone()), None);
        assert_eq!(list.push(data[1].clone()), Some(2));
        assert_eq!(list.commands.len(), 2);

        assert_eq!(list.push(data[0].clone()), None);
        assert_eq!(list.push(data[1].clone()), None);
        assert_eq!(list.push(data[2].clone()), Some(3));
        assert_eq!(list.commands.len(), 3);

        assert_eq!(list.push(data[1].clone()), None);
        assert_eq!(list.push(data[2].clone()), None);
        assert_eq!(list.push(data[3].clone()), Some(4));
        assert_eq!(list.commands.len(), 4);

        assert_eq!(list.push(data[2].clone()), None);
        assert_eq!(list.push(data[3].clone()), None);
        assert_eq!(list.push(data[4].clone()), Some(5));
        assert_eq!(list.commands.len(), 5);
    }

    #[test]
    fn auto_evict() {
        let data = generate_control_cmds(6);
        let mut list = RemoteController::default();
        list.max_hold = Duration::from_millis(100);
        assert_eq!(list.push(data[0].clone()), Some(1));
        assert_eq!(list.push(data[1].clone()), Some(2));
        assert_eq!(list.push(data[2].clone()), Some(3));
        assert_eq!(list.commands.len(), 3);
        assert_eq!(list.push(data[3].clone()), Some(4));
        assert_eq!(list.commands.len(), 4);
        assert_eq!(list.commands[0].id, 1);
        assert_eq!(list.push(data[4].clone()), Some(5));
        assert_eq!(list.commands.len(), 5);
        list.maintain(None);
        assert_eq!(list.commands.len(), 4);
        assert_eq!(list.commands[0].id, 2);
        assert_eq!(list.push(data[5].clone()), Some(6));
        assert_eq!(list.commands.len(), 5);
        list.maintain(None);
        assert_eq!(list.commands.len(), 4);
        assert_eq!(list.commands[0].id, 3);
        assert_eq!(list.commands[1].id, 4);
        assert_eq!(list.commands[2].id, 5);
        assert_eq!(list.commands[3].id, 6);
    }

    #[test]
    fn acked() {
        let data = generate_control_cmds(7);
        let mut list = RemoteController::default();
        assert_eq!(list.push(data[0].clone()), Some(1));
        assert_eq!(list.push(data[1].clone()), Some(2));
        assert_eq!(list.push(data[2].clone()), Some(3));
        assert_eq!(list.push(data[3].clone()), Some(4));
        assert_eq!(list.push(data[4].clone()), Some(5));
        let mut to_export = list.commands().iter().map(|e| e.id).collect::<HashSet<_>>();
        // damange one entry
        to_export.remove(&3);
        list.acked(to_export, Duration::from_secs(6));
        assert_eq!(list.push(data[5].clone()), Some(6));
        assert_eq!(list.push(data[6].clone()), Some(7));
        println!("asd{:?}", &list);

        let to_export = list.commands().clone();
        assert_eq!(to_export.len(), 3);
        assert_eq!(to_export[0].id, 3);
        assert_eq!(to_export[1].id, 6);
        assert_eq!(to_export[2].id, 7);
    }

    #[test]
    fn append() {
        let data = generate_control_cmds(5);
        let mut list = RemoteController::default();
        let set = list.append(data);
        assert_eq!(set.len(), 5);
        assert!(!set.contains(&0));
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(set.contains(&4));
        assert!(set.contains(&5));
        assert!(!set.contains(&6));
    }

    #[test]
    fn compress_basic() {
        let data = generate_control_cmds(5);
        let mut list = RemoteController::default();
        list.append(data);
        let compressed = list.compress(0 * INCREASE, 1 * INCREASE).unwrap();
        assert_eq!(compressed, Controller {
            inputs: comp::ControllerInputs {
                break_block_pos: None,
                climb: None,
                move_dir: Vec2::new(0.0, 0.0),
                move_z: 0.,
                look_dir: Dir::new(Vec3::new(0.0, 1.0, 0.0)),
                strafing: false,
            },
            queued_inputs: vec!((comp::InputKind::Jump, comp::InputAttr {
                select_pos: None,
                target_entity: None
            }))
            .into_iter()
            .collect(),
            events: vec!(comp::ControlEvent::EnableLantern),
            actions: vec!(),
        });
    }

    #[test]
    fn compress_avg_input_full() {
        let data = generate_control_cmds(4);
        let mut list = RemoteController::default();
        list.append(data);
        let compressed = list.compress(0 * INCREASE, 2 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.1, 0.05),
            move_z: 0.5,
            ..ControllerInputs::default()
        });
        let compressed = list.compress(0 * INCREASE, 3 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.2, 0.1),
            move_z: 1.0,
            ..ControllerInputs::default()
        });
        let compressed = list.compress(1 * INCREASE, 3 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.4, 0.2),
            move_z: 2.0,
            ..ControllerInputs::default()
        });
    }

    #[test]
    fn compress_last_input_afterwards() {
        let data = generate_control_cmds(4);
        let mut list = RemoteController::default();
        list.append(data);
        let compressed = list.compress(10 * INCREASE, 2 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.6, 0.3),
            move_z: 3.0,
            ..ControllerInputs::default()
        });
    }

    #[test]
    fn compress_avg_input_partial() {
        let data = generate_control_cmds(4);
        let mut list = RemoteController::default();
        list.append(data);
        const HALF: Duration = Duration::from_nanos(INCREASE.as_nanos() as u64 / 2);
        let compressed = list.compress(HALF, INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.1, 0.05),
            move_z: 0.5,
            ..ControllerInputs::default()
        });
        let compressed = list.compress(HALF, 3 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.3, 0.15),
            move_z: 1.5,
            ..ControllerInputs::default()
        });
        let compressed = list.compress(INCREASE + HALF, 2 * INCREASE).unwrap();
        assert_eq!(compressed.inputs, ControllerInputs {
            move_dir: Vec2::new(0.4, 0.2),
            move_z: 2.0,
            ..ControllerInputs::default()
        });
    }

    #[test]
    fn compress_avg_input_missing() {
        let mut data = generate_control_cmds(7);
        let mut list = RemoteController::default();
        data.pop_front();
        data.pop_front();
        data.pop_front();
        list.append(data);
        let compressed = list.compress(0 * INCREASE, 2 * INCREASE);
        assert_eq!(compressed, None);
        //let compressed = list.compress(10 * INCREASE, 2 * INCREASE);
        //assert_eq!(compressed, None);
    }

    #[test]
    fn compress_other_input() {
        let data = generate_control_cmds(10);
        let mut list = RemoteController::default();
        list.append(data);
        let compressed = list.compress(5 * INCREASE, 5 * INCREASE).unwrap();
        assert_eq!(
            compressed.inputs.break_block_pos,
            Some(Vec3::new(2.0, 3.0, 4.0))
        );
        assert_eq!(compressed.inputs.climb, Some(Climb::Up));
        assert_eq!(compressed.inputs.move_dir, Vec2::new(1.4, 0.7));
        let move_z = (compressed.inputs.move_z - 7.0).abs();
        println!("move_z: {}", move_z);
        assert!(move_z < 0.0001);
        assert_eq!(compressed.inputs.strafing, false);
    }

    //
}
