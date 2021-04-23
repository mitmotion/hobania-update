use crate::{
    comp::{CharacterState, HealthSource, StateUpdate, HealthChange},
    event::ServerEvent,
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::*,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use specs::saveload::MarkerAllocator;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// How long the state builds up for
    pub buildup_duration: Duration,
    /// How long the state recovers for
    pub recover_duration: Duration,
    /// What the max range of the heal is
    pub max_range: f32,
    /// Miscellaneous information about the ability
    pub ability_info: AbilityInfo,
    /// Heal
    pub heal: f32,
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
    fn behavior(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        match self.stage_section {
            StageSection::Buildup => {
                if self.timer < self.static_data.buildup_duration {
                    // Build up
                    update.character = CharacterState::TargetedEffect(Data {
                        timer: self
                            .timer
                            .checked_add(Duration::from_secs_f32(data.dt.0))
                            .unwrap_or_default(),
                        ..*self
                    });
                } else {
                    // Heals a target   
                    if let Some(input_attr) = self.static_data.ability_info.input_attr {
                        if let Some(target) = input_attr.target_entity {
                            if let Some(target) = data.uid_allocator.retrieve_entity_internal(target.into()) {
                                update.server_events.push_front(ServerEvent::HealthChange {
                                    entity: target,
                                    change: HealthChange {
                                        amount: self.static_data.heal as i32,
                                        cause: HealthSource::Heal { by: Some(*data.uid) },
                                    }});
                                update.server_events.push_front(ServerEvent::ComboChange {
                                    entity: data.entity,
                                    change: -1,
                                })
                            }
                        }
                    }
                    // Transitions to recover section of stage
                    update.character = CharacterState::TargetedEffect(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Recover,
                        ..*self
                    });
                }
            },
            StageSection::Recover => {
                if self.timer < self.static_data.recover_duration {
                    // Recovery
                    update.character = CharacterState::TargetedEffect(Data {
                        timer: self
                            .timer
                            .checked_add(Duration::from_secs_f32(data.dt.0))
                            .unwrap_or_default(),
                        ..*self
                    });
                } else {
                    // Done
                    update.character = CharacterState::Wielding;
                }
            },
            _ => {
                // If it somehow ends up in an incorrect stage section
                update.character = CharacterState::Wielding;
            },
        }

        update
    }
}   
