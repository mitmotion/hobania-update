use super::utils::*;
use crate::{
    comp::{character_state::OutputEvents, CharacterState, InputKind, StateUpdate},
    states::{
        behavior::{CharacterBehavior, JoinData},
        wielding,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// How long until state should deal damage
    pub buildup_duration: Duration,
    /// How long the state has until exiting
    pub recover_duration: Duration,
    /// Max angle (45.0 will give you a 90.0 angle window)
    pub max_angle: f32,
    /// What percentage incoming damage is reduced by
    pub block_strength: f32,
    /// What key is used to press ability
    pub ability_info: AbilityInfo,
    /// Energy consumed to initiate the block
    pub energy_cost: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// Struct containing data that does not change over the course of the
    /// character state
    pub static_data: StaticData,
    /// Timer for each stage
    pub timer: Duration,
    /// What section the character stage is in
    pub stage_section: StageSection,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, _: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_orientation(data, &mut update, 1.0, None);
        handle_move(data, &mut update, 0.4);

        match self.stage_section {
            StageSection::Buildup => {
                if self.timer < self.static_data.buildup_duration {
                    // Build up
                    update.character = CharacterState::BasicBlock(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transitions to swing section of stage
                    update.character = CharacterState::BasicBlock(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Action,
                        ..*self
                    });
                }
            },
            StageSection::Action => {
                if input_is_pressed(data, InputKind::Block) {
                    // Block
                    update.character = CharacterState::BasicBlock(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transitions to recover section of stage
                    update.character = CharacterState::BasicBlock(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Recover,
                        ..*self
                    });
                }
            },
            StageSection::Recover => {
                if self.timer < self.static_data.recover_duration {
                    // Recovery
                    update.character = CharacterState::BasicBlock(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Done
                    update.character =
                        CharacterState::Wielding(wielding::Data { is_sneaking: false });
                }
            },
            _ => {
                // If it somehow ends up in an incorrect stage section
                update.character = CharacterState::Wielding(wielding::Data { is_sneaking: false });
            },
        }

        // At end of state logic so an interrupt isn't overwritten
        handle_dodge_interrupt(data, &mut update, None);

        update
    }
}
