use crate::{
    comp::{character_state::OutputEvents, CharacterState, Melee, MeleeConstructor, StateUpdate},
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::*,
        wielding,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// Rate of energy drain
    pub energy_drain: f32,
    /// How quickly dasher moves forward
    pub forward_speed: f32,
    /// Whether the state can charge through enemies and do a second hit
    pub charge_through: bool,
    /// How long until state should deal damage
    pub buildup_duration: Duration,
    /// How long the state charges for until it reaches max damage
    pub charge_duration: Duration,
    /// Duration of state spent in swing
    pub swing_duration: Duration,
    /// How long the state has until exiting
    pub recover_duration: Duration,
    /// Used to construct the Melee attack
    pub melee_constructor: MeleeConstructor,
    /// How fast can you turn during charge
    pub ori_modifier: f32,
    /// Whether the state can be interrupted by other abilities
    pub is_interruptible: bool,
    /// What key is used to press ability
    pub ability_info: AbilityInfo,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// Struct containing data that does not change over the course of the
    /// character state
    pub static_data: StaticData,
    /// Whether the charge should last a default amount of time or until the
    /// mouse is released
    pub auto_charge: bool,
    /// Timer for each stage
    pub timer: Duration,
    /// What section the character stage is in
    pub stage_section: StageSection,
    /// Whether the state should attempt attacking again
    pub exhausted: bool,
    /// Time that charge should end (used for charge through)
    pub charge_end_timer: Duration,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, _: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_move(data, &mut update, 0.1);

        match self.stage_section {
            StageSection::Buildup => {
                if self.timer < self.static_data.buildup_duration {
                    handle_orientation(data, &mut update, 1.0, None);
                    // Build up
                    update.character = CharacterState::DashMelee(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transitions to charge section of stage
                    update.character = CharacterState::DashMelee(Data {
                        auto_charge: !input_is_pressed(data, self.static_data.ability_info.input),
                        timer: Duration::default(),
                        stage_section: StageSection::Charge,
                        ..*self
                    });
                }
            },
            StageSection::Charge => {
                if self.timer < self.charge_end_timer
                    && (input_is_pressed(data, self.static_data.ability_info.input)
                        || (self.auto_charge && self.timer < self.static_data.charge_duration))
                    && update.energy.current() > 0.0
                {
                    // Forward movement
                    let charge_frac = (self.timer.as_secs_f32()
                        / self.static_data.charge_duration.as_secs_f32())
                    .min(1.0);

                    handle_orientation(data, &mut update, self.static_data.ori_modifier, None);
                    handle_forced_movement(data, &mut update, ForcedMovement::Forward {
                        strength: self.static_data.forward_speed * charge_frac.sqrt(),
                    });

                    // This logic basically just decides if a charge should end,
                    // and prevents the character state spamming attacks
                    // while checking if it has hit something.
                    if !self.exhausted {
                        // Hit attempt
                        let crit_data = get_crit_data(data, self.static_data.ability_info);
                        let buff_strength = get_buff_strength(data, self.static_data.ability_info);

                        data.updater.insert(
                            data.entity,
                            self.static_data
                                .melee_constructor
                                .handle_scaling(charge_frac)
                                .create_melee(crit_data, buff_strength),
                        );

                        update.character = CharacterState::DashMelee(Data {
                            timer: tick_attack_or_default(data, self.timer, None),
                            exhausted: true,
                            ..*self
                        })
                    } else if let Some(melee) = data.melee_attack {
                        if !melee.applied {
                            // If melee attack has not applied, just tick duration
                            update.character = CharacterState::DashMelee(Data {
                                timer: tick_attack_or_default(data, self.timer, None),
                                ..*self
                            });
                        } else if melee.hit_count == 0 {
                            // If melee attack has applied, but not hit anything, remove exhausted
                            // so it can attack again
                            update.character = CharacterState::DashMelee(Data {
                                timer: tick_attack_or_default(data, self.timer, None),
                                exhausted: false,
                                ..*self
                            });
                        } else if self.static_data.charge_through {
                            // If can charge through, set charge_end_timer to stop after a little
                            // more time
                            let charge_end_timer =
                                if self.charge_end_timer != self.static_data.charge_duration {
                                    self.charge_end_timer
                                } else {
                                    self.timer
                                        .checked_add(Duration::from_secs_f32(
                                            0.2 * self.static_data.melee_constructor.range
                                                / self.static_data.forward_speed,
                                        ))
                                        .unwrap_or(self.static_data.charge_duration)
                                        .min(self.static_data.charge_duration)
                                };
                            update.character = CharacterState::DashMelee(Data {
                                timer: tick_attack_or_default(data, self.timer, None),
                                charge_end_timer,
                                ..*self
                            });
                        } else {
                            // Stop charging now and go to swing stage section
                            update.character = CharacterState::DashMelee(Data {
                                timer: Duration::default(),
                                stage_section: StageSection::Action,
                                exhausted: false,
                                ..*self
                            });
                        }
                    } else {
                        // If melee attack has not applied, just tick duration
                        update.character = CharacterState::DashMelee(Data {
                            timer: tick_attack_or_default(data, self.timer, None),
                            exhausted: false,
                            ..*self
                        });
                    }

                    // Consumes energy if there's enough left and charge has not stopped
                    update
                        .energy
                        .change_by(-self.static_data.energy_drain * data.dt.0);
                } else {
                    // Transitions to swing section of stage
                    update.character = CharacterState::DashMelee(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Action,
                        exhausted: false,
                        ..*self
                    });
                }
            },
            StageSection::Action => {
                if self.static_data.charge_through && !self.exhausted {
                    // If can charge through and not exhausted, do one more melee attack

                    // Assumes charge got to charge_end_timer for damage calculations
                    let charge_frac = (self.charge_end_timer.as_secs_f32()
                        / self.static_data.charge_duration.as_secs_f32())
                    .min(1.0);

                    let crit_data = get_crit_data(data, self.static_data.ability_info);
                    let buff_strength = get_buff_strength(data, self.static_data.ability_info);

                    data.updater.insert(
                        data.entity,
                        self.static_data
                            .melee_constructor
                            .handle_scaling(charge_frac)
                            .create_melee(crit_data, buff_strength),
                    );

                    update.character = CharacterState::DashMelee(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        exhausted: true,
                        ..*self
                    })
                } else if self.timer < self.static_data.swing_duration {
                    // Swings
                    update.character = CharacterState::DashMelee(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transitions to recover section of stage
                    update.character = CharacterState::DashMelee(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Recover,
                        ..*self
                    });
                }
            },
            StageSection::Recover => {
                if self.timer < self.static_data.recover_duration {
                    // Recover
                    update.character = CharacterState::DashMelee(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Done
                    update.character =
                        CharacterState::Wielding(wielding::Data { is_sneaking: false });
                    // Make sure attack component is removed
                    data.updater.remove::<Melee>(data.entity);
                }
            },
            _ => {
                // If it somehow ends up in an incorrect stage section
                update.character = CharacterState::Wielding(wielding::Data { is_sneaking: false });
                // Make sure attack component is removed
                data.updater.remove::<Melee>(data.entity);
            },
        }

        // At end of state logic so an interrupt isn't overwritten
        if !input_is_pressed(data, self.static_data.ability_info.input) {
            handle_state_interrupt(data, &mut update, self.static_data.is_interruptible);
        }

        update
    }
}
