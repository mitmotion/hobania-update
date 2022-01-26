use crate::{
    comp::{
        ability,
        inventory::slot::{EquipSlot, InvSlotId, Slot},
        invite::{InviteKind, InviteResponse},
        BuffKind,
    },
    trade::{TradeAction, TradeId},
    uid::Uid,
    util::Dir,
};
use serde::{Deserialize, Serialize};
use specs::Component;
use specs_idvs::IdvStorage;
use std::collections::BTreeMap;
use vek::*;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InventoryEvent {
    Pickup(Uid),
    Swap(InvSlotId, InvSlotId),
    SplitSwap(InvSlotId, InvSlotId),
    Drop(InvSlotId),
    SplitDrop(InvSlotId),
    Sort,
    CraftRecipe {
        craft_event: CraftEvent,
        craft_sprite: Option<Vec3<i32>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum InventoryAction {
    Swap(EquipSlot, Slot),
    Drop(EquipSlot),
    Use(Slot),
    Sort,
    Collect(Vec3<i32>),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InventoryManip {
    Pickup(Uid),
    Collect(Vec3<i32>),
    Use(Slot),
    Swap(Slot, Slot),
    SplitSwap(Slot, Slot),
    Drop(Slot),
    SplitDrop(Slot),
    Sort,
    CraftRecipe {
        craft_event: CraftEvent,
        craft_sprite: Option<Vec3<i32>>,
    },
    SwapEquippedWeapons,
}

impl From<InventoryAction> for InventoryManip {
    fn from(inv_action: InventoryAction) -> Self {
        match inv_action {
            InventoryAction::Use(slot) => Self::Use(slot),
            InventoryAction::Swap(equip, slot) => Self::Swap(Slot::Equip(equip), slot),
            InventoryAction::Drop(equip) => Self::Drop(Slot::Equip(equip)),
            InventoryAction::Sort => Self::Sort,
            InventoryAction::Collect(collect) => Self::Collect(collect),
        }
    }
}

impl From<InventoryEvent> for InventoryManip {
    fn from(inv_event: InventoryEvent) -> Self {
        match inv_event {
            InventoryEvent::Pickup(pickup) => Self::Pickup(pickup),
            InventoryEvent::Swap(inv1, inv2) => {
                Self::Swap(Slot::Inventory(inv1), Slot::Inventory(inv2))
            },
            InventoryEvent::SplitSwap(inv1, inv2) => {
                Self::SplitSwap(Slot::Inventory(inv1), Slot::Inventory(inv2))
            },
            InventoryEvent::Drop(inv) => Self::Drop(Slot::Inventory(inv)),
            InventoryEvent::SplitDrop(inv) => Self::SplitDrop(Slot::Inventory(inv)),
            InventoryEvent::Sort => Self::Sort,
            InventoryEvent::CraftRecipe {
                craft_event,
                craft_sprite,
            } => Self::CraftRecipe {
                craft_event,
                craft_sprite,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CraftEvent {
    Simple {
        recipe: String,
        slots: Vec<(u32, InvSlotId)>,
    },
    Salvage(InvSlotId),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GroupManip {
    Leave,
    Kick(Uid),
    AssignLeader(Uid),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UtteranceKind {
    Calm,
    Angry,
    Surprised,
    Hurt,
    Greeting,
    Scream,
    /* Death,
     * TODO: Wait for more post-death features (i.e. animiations) before implementing death
     * sounds */
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlEvent {
    //ToggleLantern,
    EnableLantern,
    DisableLantern,
    Interact(Uid),
    InitiateInvite(Uid, InviteKind),
    InviteResponse(InviteResponse),
    PerformTradeAction(TradeId, TradeAction),
    Mount(Uid),
    Unmount,
    InventoryEvent(InventoryEvent),
    GroupManip(GroupManip),
    RemoveBuff(BuffKind),
    Respawn,
    Utterance(UtteranceKind),
    ChangeAbility {
        slot: usize,
        auxiliary_key: ability::AuxiliaryKey,
        new_ability: ability::AuxiliaryAbility,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ControlAction {
    SwapEquippedWeapons,
    InventoryAction(InventoryAction),
    Wield,
    GlideWield,
    Unwield,
    Sit,
    Dance,
    Sneak,
    Stand,
    Talk,
    StartInput {
        input: InputKind,
        target_entity: Option<Uid>,
        // Some inputs need a selected position, such as mining
        select_pos: Option<Vec3<f32>>,
    },
    CancelInput(InputKind),
}

impl ControlAction {
    pub fn basic_input(input: InputKind) -> Self {
        ControlAction::StartInput {
            input,
            target_entity: None,
            select_pos: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Eq, Ord, PartialOrd)]
#[repr(u32)]
pub enum InputKind {
    Primary = 0,
    Secondary = 1,
    Block = 2,
    Ability(usize) = 3,
    Roll = 4,
    Jump = 5,
    Fly = 6,
}

impl InputKind {
    pub fn is_ability(self) -> bool {
        matches!(
            self,
            Self::Primary | Self::Secondary | Self::Ability(_) | Self::Block
        )
    }
}

impl From<InputKind> for Option<ability::AbilityInput> {
    fn from(input: InputKind) -> Option<ability::AbilityInput> {
        use ability::AbilityInput;
        match input {
            InputKind::Primary => Some(AbilityInput::Primary),
            InputKind::Secondary => Some(AbilityInput::Secondary),
            InputKind::Roll => Some(AbilityInput::Movement),
            InputKind::Ability(index) => Some(AbilityInput::Auxiliary(index)),
            InputKind::Jump | InputKind::Fly | InputKind::Block => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InputAttr {
    pub select_pos: Option<Vec3<f32>>,
    pub target_entity: Option<Uid>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Climb {
    Up,
    Down,
    Hold,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ControllerInputs {
    pub climb: Option<Climb>,
    pub move_dir: Vec2<f32>,
    pub move_z: f32, /* z axis (not combined with move_dir because they may have independent
                      * limits) */
    pub look_dir: Dir,
    pub break_block_pos: Option<Vec3<f32>>,
    /// Attempt to enable strafing.
    /// Currently, setting this to false will *not* disable strafing during a
    /// wielding character state.
    pub strafing: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Controller {
    pub inputs: ControllerInputs,
    pub queued_inputs: BTreeMap<InputKind, InputAttr>,
    // TODO: consider SmallVec
    pub events: Vec<ControlEvent>,
    pub actions: Vec<ControlAction>,
}

impl ControllerInputs {
    /// Sanitize inputs to avoid clients sending bad data.
    pub fn sanitize(&mut self) {
        self.move_dir = if self.move_dir.map(|e| e.is_finite()).reduce_and() {
            self.move_dir / self.move_dir.magnitude().max(1.0)
        } else {
            Vec2::zero()
        };
        self.move_z = if self.move_z.is_finite() {
            self.move_z.clamped(-1.0, 1.0)
        } else {
            0.0
        };
    }

    /// Updates Controller inputs with new version received from the client
    pub fn update_with_new(&mut self, new: Self) {
        self.climb = new.climb;
        self.move_dir = new.move_dir;
        self.move_z = new.move_z;
        self.look_dir = new.look_dir;
        self.break_block_pos = new.break_block_pos;
    }
}

impl Controller {
    /// Sets all inputs to default
    pub fn reset(&mut self) {
        self.inputs = Default::default();
        self.queued_inputs = Default::default();
    }

    pub fn clear_events(&mut self) { self.events.clear(); }

    pub fn push_event(&mut self, event: ControlEvent) { self.events.push(event); }

    pub fn push_action(&mut self, action: ControlAction) { self.actions.push(action); }

    pub fn push_basic_input(&mut self, input: InputKind) {
        self.push_action(ControlAction::basic_input(input));
    }
}

impl Component for Controller {
    type Storage = IdvStorage<Self>;
}
