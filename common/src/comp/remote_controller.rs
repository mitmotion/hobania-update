use crate::comp::Controller;
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
        let min_allowed_age = self.commands.back().unwrap().source_time;

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

    pub fn push(&mut self, command: ControlCommand) -> Option<u64> {
        let id = command.id;
        //check if command fits on the end.
        if self.existing_commands.contains(&id) {
            return None; // element already existed
        }
        self.existing_commands.insert(id);
        self.commands.push_back(command);

        Some(id)
    }

    pub fn append(&mut self, commands: ControlCommands) -> HashSet<u64> {
        let mut result = HashSet::new();
        for command in commands {
            let id = command.id;
            if !self.existing_commands.contains(&id) {
                result.insert(id);
                self.existing_commands.insert(id);
                self.commands.push_back(command);
            }
        }
        result
    }

    pub fn current(&self, time: Duration) -> Option<&ControlCommand> {
        //TODO: actually sort it!
        let mut lowest = None;
        let mut lowest_cmd = None;
        for c in &self.commands {
            if c.source_time >= time {
                let diff = c.source_time - time;
                if match lowest {
                    None => true,
                    Some(lowest) => diff < lowest,
                } {
                    lowest = Some(diff);
                    lowest_cmd = Some(c);
                }
            }
        }
        lowest_cmd
    }

    /// arrived at remote and no longer need to be hold locally
    pub fn acked(&mut self, ids: HashSet<u64>) { self.commands.retain(|c| !ids.contains(&c.id)); }

    pub fn commands(&self) -> &ControlCommands { &self.commands }
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

    fn generate_control_cmds(count: usize) -> ControlCommands {
        let mut result = VecDeque::new();
        let mut generator = CommandGenerator::default();
        let mut time = Duration::new(0, 0);
        const INCREASE: Duration = Duration::from_millis(33);
        for _ in 0..count {
            let msg = Controller::default();
            let cc = generator.gen(time, msg);
            result.push_back(cc);
            time += INCREASE;
        }
        result
    }

    #[test]
    fn test_resend_data() {
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
    fn test_auto_evict() {
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
    fn test_acked() {
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
    fn test_append() {
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
}
