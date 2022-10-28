use crate::{
    comp::{
        buff::{BuffChange, BuffKind},
        character_state::OutputEvents,
        CharacterState, InputKind, StateUpdate,
    },
    event::ServerEvent,
    states::{
        behavior::{CharacterBehavior, JoinData},
        idle,
        utils::*,
        wielding,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// How long until state should roll
    pub buildup_duration: Duration,
    /// How long state is rolling for
    pub movement_duration: Duration,
    /// How long it takes to recover from roll
    pub recover_duration: Duration,
    /// Affects the speed and distance of the roll
    pub roll_strength: f32,
    /// Affects whether you are immune to melee attacks while rolling
    pub immune_melee: bool,
    /// Information about the ability
    pub ability_info: AbilityInfo,
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
    /// Had weapon
    pub was_wielded: bool,
    /// Is sneaking, true if previous state was also considered sneaking
    pub is_sneaking: bool,
    /// Was in state with combo
    pub was_combo: Option<(InputKind, u32)>,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        // You should not be able to strafe while rolling
        update.should_strafe = false;

        // Smooth orientation
        handle_orientation(data, &mut update, 2.5, None);

        match self.stage_section {
            StageSection::Buildup => {
                handle_move(data, &mut update, 1.0);
                if self.timer < self.static_data.buildup_duration {
                    // Build up
                    update.character = CharacterState::Roll(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Remove burning effect if active
                    output_events.emit_server(ServerEvent::Buff {
                        entity: data.entity,
                        buff_change: BuffChange::RemoveByKind(BuffKind::Burning),
                    });
                    // Transitions to movement section of stage
                    update.character = CharacterState::Roll(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Movement,
                        ..*self
                    });
                }
            },
            StageSection::Movement => {
                // Update velocity
                handle_forced_movement(
                    data,
                    &mut update,
                    ForcedMovement::Forward(
                        self.static_data.roll_strength
                            * ((1.0
                                - self.timer.as_secs_f32()
                                    / self.static_data.movement_duration.as_secs_f32())
                                / 2.0
                                + 0.25),
                    ),
                );

                if self.timer < self.static_data.movement_duration {
                    // Movement
                    update.character = CharacterState::Roll(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transition to recover section of stage
                    update.character = CharacterState::Roll(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Recover,
                        ..*self
                    });
                }
            },
            StageSection::Recover => {
                handle_move(data, &mut update, 1.0);
                // Allows for jumps to interrupt recovery in roll
                if self.timer < self.static_data.recover_duration
                    && !handle_jump(data, output_events, &mut update, 1.5)
                {
                    // Recover
                    update.character = CharacterState::Roll(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Done
                    if let Some((input, stage)) = self.was_combo {
                        if input_is_pressed(data, input) {
                            handle_input(data, output_events, &mut update, input);
                            // If other states are introduced that progress through stages, add them
                            // here
                            if let CharacterState::ComboMelee(c) = &mut update.character {
                                c.stage = stage;
                            }
                        } else {
                            update.character = CharacterState::Wielding(wielding::Data {
                                is_sneaking: self.is_sneaking,
                            });
                        }
                    } else {
                        update.character = if self.was_wielded {
                            CharacterState::Wielding(wielding::Data {
                                is_sneaking: self.is_sneaking,
                            })
                        } else {
                            CharacterState::Idle(idle::Data {
                                is_sneaking: self.is_sneaking,
                                footwear: None,
                            })
                        }
                    }
                }
            },
            _ => {
                // If it somehow ends up in an incorrect stage section
                update.character = CharacterState::Idle(idle::Data {
                    is_sneaking: self.is_sneaking,
                    footwear: None,
                });
            },
        }

        update
    }
}
