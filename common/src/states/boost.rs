use crate::{
    comp::{character_state::OutputEvents, CharacterState, StateUpdate, MovementKind},
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::*,
        wielding,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use vek::*;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    pub movement_duration: Duration,
    pub only_up: bool,
    pub speed: f32,
    pub max_exit_velocity: f32,
    pub ability_info: AbilityInfo,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// Struct containing data that does not change over the course of the
    /// character state
    pub static_data: StaticData,
    /// Timer for each stage
    pub timer: Duration,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_move(data, &mut update, 1.0);

        if self.timer < self.static_data.movement_duration {
            // Movement
            let dir = if self.static_data.only_up {
                Vec3::unit_z()
            } else {
                *data.inputs.look_dir
            };
            update.movement = update.movement.with_movement(MovementKind::Boost { dir, accel: self.static_data.speed });
            update.character = CharacterState::Boost(Data {
                timer: tick_attack_or_default(data, self.timer, None),
                ..*self
            });
        } else {
            // Done
            if input_is_pressed(data, self.static_data.ability_info.input) {
                reset_state(self, data, output_events, &mut update);
            } else {
                let speed = data.vel.0.magnitude().min(self.static_data.max_exit_velocity);
                update.movement = update.movement.with_movement(MovementKind::ChangeSpeed { speed });
                update.character = CharacterState::Wielding(wielding::Data { is_sneaking: false });
            }
        }

        update
    }
}

fn reset_state(
    data: &Data,
    join: &JoinData,
    output_events: &mut OutputEvents,
    update: &mut StateUpdate,
) {
    handle_input(
        join,
        output_events,
        update,
        data.static_data.ability_info.input,
    );
}
