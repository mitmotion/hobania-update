use super::utils::*;
use crate::{
    comp::{slot::EquipSlot, Body, CharacterState, InventoryAction, Ori, StateUpdate},
    glider::Glider,
    states::{
        behavior::{CharacterBehavior, JoinData},
        glide,
    },
};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data(pub Glider);

impl Data {
    pub fn new(body: &Body, ori: &Ori) -> Self {
        Self(Glider::new(
            body.dimensions().z * 3.0,
            body.dimensions().z / 3.0,
            *ori,
        ))
    }
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_orientation(data, &mut update, 1.0);
        handle_move(data, &mut update, 1.0);
        handle_jump(data, &mut update, 1.0);
        handle_dodge_input(data, &mut update);
        handle_wield(data, &mut update);

        let mut glider = self.0;
        glider.ori = glider.ori.slerped_towards(
            Ori::from(data.inputs.look_dir)
                .yawed_towards(data.ori.look_dir())
                .pitched_up(inline_tweak::tweak!(0.7)),
            inline_tweak::tweak!(3.0) * data.dt.0,
        );

        // If not on the ground while wielding glider enter gliding state
        update.character = if data.physics.on_ground.is_none() {
            CharacterState::Glide(glide::Data::new(glider))
        } else if data
            .physics
            .in_liquid()
            .map(|depth| depth > 0.5)
            .unwrap_or(false)
        {
            CharacterState::Idle
        } else if data.inventory.equipped(EquipSlot::Glider).is_none() {
            CharacterState::Idle
        } else {
            CharacterState::GlideWield(Self(glider))
        };

        update
    }

    fn sit(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        attempt_sit(data, &mut update);
        update
    }

    fn dance(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        attempt_dance(data, &mut update);
        update
    }

    fn sneak(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        attempt_sneak(data, &mut update);
        update
    }

    fn unwield(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        update.character = CharacterState::Idle;
        update
    }

    fn manipulate_loadout(&self, data: &JoinData, inv_action: InventoryAction) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        handle_manipulate_loadout(&data, &mut update, inv_action);
        update
    }
}
