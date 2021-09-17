use crate::{comp::Controller, util::Dir};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use specs::Component;
use specs_idvs::IdvStorage;
use std::{collections::VecDeque, time::Duration};
use vek::Vec3;

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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ControlCommand {
    id: u64,
    source_time: Duration,
    msg: Controller,
}

impl RemoteController {
    /// delte old commands
    pub fn maintain(&mut self) {
        self.commands.make_contiguous();
        if let Some(last) = self.commands.back() {
            let min_allowed_age = last.source_time;

            while let Some(first) = self.commands.front() {
                if first.source_time + self.max_hold < min_allowed_age {
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
            .binary_search_by_key(&command.source_time, |e| e.source_time)
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
                    .binary_search_by_key(&command.source_time, |e| e.source_time)
                {
                    result.insert(id);
                    self.existing_commands.insert(id);
                    self.commands.insert(i, command);
                }
            }
        }
        result
    }

    /// arrived at remote and no longer need to be hold locally
    pub fn acked(&mut self, ids: HashSet<u64>) { self.commands.retain(|c| !ids.contains(&c.id)); }

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
            .binary_search_by_key(&start, |e| e.source_time)
        {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };
        let end_exclusive_i = match self
            .commands
            .binary_search_by_key(&(start + dt), |e| e.source_time)
        {
            Ok(i) => i,
            Err(i) => i,
        };
        println!("start_i {:?}", start_i);
        println!("end_exclusive_i {:?}", end_exclusive_i);

        if self.commands.is_empty() || end_exclusive_i == start_i {
            return None;
        }
        let mut result = Controller::default();
        let mut look_dir = Vec3::zero();
        //if self.commands[start_i].source_time
        // Inputs are averaged over all elements by time
        // Queued Inputs are just added
        // Events and Actions are not included when the frame isn't fully inserted, must
        // not be duplicated
        let mut last_start = start;
        for i in start_i..end_exclusive_i {
            let e = &self.commands[i];
            let local_start = e.source_time.max(last_start);
            let local_end = if let Some(e) = self.commands.get(i + 1) {
                e.source_time.min(start + dt)
            } else {
                start + dt
            };
            let local_dur = local_end - local_start;
            println!("local_start {:?}, local_end {:?}", local_start, local_end);
            println!("source_time {:?}", e.source_time);
            result.inputs.move_dir += e.msg.inputs.move_dir * local_dur.as_secs_f32();
            result.inputs.move_z += e.msg.inputs.move_z * local_dur.as_secs_f32();
            look_dir = result.inputs.look_dir.to_vec()
                + e.msg.inputs.look_dir.to_vec() * local_dur.as_secs_f32();
            //TODO: manually combine 70% up and 30% down to UP
            result.inputs.climb = result.inputs.climb.or(e.msg.inputs.climb);
            result.inputs.break_block_pos = result
                .inputs
                .break_block_pos
                .or(e.msg.inputs.break_block_pos);
            result.inputs.strafing = result.inputs.strafing || e.msg.inputs.strafing;
            last_start = local_start;
        }
        result.inputs.move_dir /= dt.as_secs_f32();
        result.inputs.move_z /= dt.as_secs_f32();
        result.inputs.look_dir = Dir::new(look_dir.normalized());

        Some(result)
    }
}

impl Default for RemoteController {
    fn default() -> Self {
        Self {
            commands: VecDeque::new(),
            existing_commands: HashSet::new(),
            max_hold: Duration::from_secs(1),
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
    pub fn gen(&mut self, time: Duration, msg: Controller) -> ControlCommand {
        self.id += 1;
        ControlCommand {
            source_time: time,
            id: self.id,
            msg,
        }
    }
}

impl ControlCommand {
    pub fn msg(&self) -> &Controller { &self.msg }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        comp,
        comp::{Climb, ControllerInputs},
    };
    use std::collections::BTreeMap;
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
        list.maintain();
        assert_eq!(list.commands.len(), 4);
        assert_eq!(list.commands[0].id, 2);
        assert_eq!(list.push(data[5].clone()), Some(6));
        assert_eq!(list.commands.len(), 5);
        list.maintain();
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
        list.acked(to_export);
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
            queued_inputs: BTreeMap::new(),
            events: Vec::new(),
            actions: Vec::new(),
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
