use bitflags::bitflags;
use num_traits::FromPrimitive;
use specs::{hibitset::DrainableBitSet, BitSet, Component};

#[derive(Clone, Copy, PartialEq, num_derive::FromPrimitive)]
#[repr(u8)]
pub enum Action {
    Read = 1,
    Write = 2,
}

bitflags! {
    struct Actions: u8 {
        const READ = Action::Read as u8;
        const WRITE = Action::Write as u8;
        const READ_WRITE = Self::READ.bits | Self::WRITE.bits;
    }
}

const OBJ_MULTIPLIER: u32 = 16;

#[derive(Clone, Copy, PartialEq, num_derive::FromPrimitive)]
#[repr(u16)]
pub enum Object {
    ChatGlobal = 0x0001,
    ChatWorld = 0x0002,
    ChatRegion = 0x0003,
    ChatLocal = 0x0004,
    CommandTeleport = 0x0005,
    CommandKick = 0x0006,
    CommandBan = 0x0007,
    CommandMute = 0x0008,
    BuildCreateBlock = 0x0009,
    BuildCreateSprite = 0x000A,
    BuildDestroy = 0x000B,
}

impl Actions {
    pub const fn allows(&self, action: Action) -> bool {
        match action {
            Action::Read => self.contains(Actions::READ),
            Action::Write => self.contains(Actions::WRITE),
        }
    }
}
impl Object {
    const fn get_valid_actions(&self) -> Actions {
        match self {
            Self::ChatGlobal => Actions::READ_WRITE,
            Self::ChatWorld => Actions::READ_WRITE,
            Self::ChatRegion => Actions::READ_WRITE,
            Self::ChatLocal => Actions::READ_WRITE,
            Self::CommandTeleport => Actions::WRITE,
            Self::CommandKick => Actions::WRITE,
            Self::CommandBan => Actions::WRITE,
            Self::CommandMute => Actions::WRITE,
            Self::BuildCreateBlock => Actions::WRITE,
            Self::BuildCreateSprite => Actions::WRITE,
            Self::BuildDestroy => Actions::WRITE,
        }
    }
}

pub struct ObjectAction {
    obj: Object,
    act: Action,
}

impl ObjectAction {
    pub fn new(obj: Object, act: Action) -> Self { Self { obj, act } }

    const fn id(&self) -> u32 { self.obj as u32 * OBJ_MULTIPLIER + self.act as u32 }

    pub fn from_id(id: u32) -> Option<Self> {
        let act = FromPrimitive::from_u32(id.rem_euclid(OBJ_MULTIPLIER));
        let obj = FromPrimitive::from_u32(id.div_euclid(OBJ_MULTIPLIER));
        if let (Some(act), Some(obj)) = (act, obj) {
            Some(Self::new(obj, act))
        } else {
            None
        }
    }

    const fn valid(&self) -> bool { self.obj.get_valid_actions().allows(self.act) }
}

#[derive(Clone, Debug, Default)]
pub struct RuleSet {
    allowed: BitSet,
}

impl Component for RuleSet {
    type Storage = specs::VecStorage<Self>;
}

impl RuleSet {
    pub fn is_allowed(&self, object_action: ObjectAction) -> bool {
        self.allowed.contains(object_action.id())
    }

    pub fn add(&mut self, object_action: ObjectAction) {
        if object_action.valid() {
            self.allowed.add(object_action.id());
        }
    }

    pub fn remove(&mut self, object_action: ObjectAction) {
        self.allowed.remove(object_action.id());
    }

    pub fn append(&mut self, mut other: RuleSet) {
        for o in other.allowed.drain() {
            self.allowed.add(o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_max_action() {
        assert!(OBJ_MULTIPLIER >= Actions::all().bits as u32 + 1);
    }

    #[test]
    fn ser_and_deser() {
        let oa = ObjectAction::new(Object::ChatGlobal, Action::Write);
        let i = oa.id();
        let oa2 = ObjectAction::from_id(i).unwrap();
        assert_eq!(oa.id(), oa2.id());
    }

    #[test]
    fn check_permissions() {
        let mut ruleset = RuleSet::default();
        ruleset.add(ObjectAction::new(Object::ChatGlobal, Action::Read));
        ruleset.add(ObjectAction::new(Object::ChatRegion, Action::Read));
        ruleset.add(ObjectAction::new(Object::ChatGlobal, Action::Write));
        let oa = ObjectAction::new(Object::ChatGlobal, Action::Write);
        let oa2 = ObjectAction::new(Object::CommandBan, Action::Write);
        let oa3 = ObjectAction::new(Object::ChatGlobal, Action::Read);
        let oa4 = ObjectAction::new(Object::ChatRegion, Action::Write);
        assert!(ruleset.is_allowed(oa));
        assert!(!ruleset.is_allowed(oa2));
        assert!(ruleset.is_allowed(oa3));
        assert!(!ruleset.is_allowed(oa4));
    }
}
