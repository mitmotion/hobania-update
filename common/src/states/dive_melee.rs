use crate::{
    comp::{character_state::OutputEvents, CharacterState, Melee, MeleeConstructor, StateUpdate},
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::{StageSection, *},
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// How long the state is moving
    pub movement_duration: Duration,
    /// How long the weapon swings
    pub swing_duration: Duration,
    /// How long the state has until exiting
    pub recover_duration: Duration,
    /// The minimum vertical speed the state needed
    pub vertical_speed: f32,
    /// Used to construct the Melee attack
    pub melee_constructor: MeleeConstructor,
    /// What key is used to press ability
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
    /// Whether the attack can deal more damage
    pub exhausted: bool,
    /// The maximum negative vertical velocity achieved during the state
    pub max_vertical_speed: f32,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, _output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        if let CharacterState::DiveMelee(c) = &mut update.character {
            c.max_vertical_speed = c.max_vertical_speed.max(-data.vel.0.z);
        }

        match self.stage_section {
            StageSection::Movement => {
                if data.physics.on_ground.is_some() {
                    // Transitions to swing portion of state upon hitting ground
                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = Duration::default();
                        c.stage_section = StageSection::Action;
                    }
                } else if self.timer < self.static_data.movement_duration {
                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = self
                            .timer
                            .checked_add(Duration::from_secs_f32(data.dt.0))
                            .unwrap_or_default();
                    }
                } else {
                    // In character state for too long, leaving in case somethign went wrong
                    end_ability(data, &mut update);
                }
            },
            StageSection::Action => {
                if !self.exhausted {
                    // Attack
                    let crit_data = get_crit_data(data, self.static_data.ability_info);
                    let buff_strength = get_buff_strength(data, self.static_data.ability_info);
                    let scaling = self.max_vertical_speed / self.static_data.vertical_speed;
                    // TODO: Remove when server authoritative physics
                    let scaling = scaling.max(2.0);

                    data.updater.insert(
                        data.entity,
                        self.static_data
                            .melee_constructor
                            .handle_scaling(scaling)
                            .create_melee(crit_data, buff_strength),
                    );

                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                        c.exhausted = true;
                    }
                } else if self.timer < self.static_data.swing_duration {
                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else {
                    // Transitions to recover portion
                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = Duration::default();
                        c.stage_section = StageSection::Recover;
                    }
                }
            },
            StageSection::Recover => {
                handle_orientation(data, &mut update, 0.5, None);
                handle_move(data, &mut update, 0.3);

                if self.timer < self.static_data.recover_duration {
                    // Complete recovery delay before finishing state
                    if let CharacterState::DiveMelee(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else {
                    // Done
                    end_ability(data, &mut update);
                    // Make sure attack component is removed
                    data.updater.remove::<Melee>(data.entity);
                }
            },
            _ => {
                // If it somehow ends up in an incorrect stage section
                end_ability(data, &mut update);
                // Make sure attack component is removed
                data.updater.remove::<Melee>(data.entity);
            },
        }

        update
    }
}
