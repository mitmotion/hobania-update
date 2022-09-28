use super::utils::*;
use crate::{
    comp::{character_state::OutputEvents, CharacterState, StateUpdate, MovementKind},
    states::{
        behavior::{CharacterBehavior, JoinData},
        idle,
    },
};
use serde::{Deserialize, Serialize};
use vek::Vec2;

const WALLRUN_ANTIGRAV: f32 = crate::consts::GRAVITY * 0.5;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub struct Data;

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_wield(data, &mut update);
        handle_jump(data, output_events, &mut update, 1.0);
        handle_climb(data, &mut update);

        {
            let lift = WALLRUN_ANTIGRAV
            * (Vec2::<f32>::from(data.vel.0).magnitude() * 0.075)
                .min(1.0)
                .max(0.2);
            update.movement = update.movement.with_movement(MovementKind::SlowFall { lift });
        }

        // fall off wall, hit ground, or enter water
        // TODO: Rugged way to determine when state change occurs and we need to leave
        // this state
        if data.physics.on_wall.is_none()
            || data.physics.on_ground.is_some()
            || data.physics.in_liquid().is_some()
        {
            update.character = CharacterState::Idle(idle::Data::default());
        }

        update
    }
}
