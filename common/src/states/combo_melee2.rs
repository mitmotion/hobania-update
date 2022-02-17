use crate::{
    comp::{
        character_state::OutputEvents, tool::Stats, CharacterState, InputKind, Melee,
        MeleeConstructor, StateUpdate,
    },
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::*,
        wielding,
    },
};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Strike<T> {
    /// Used to construct the Melee attack
    pub melee_constructor: MeleeConstructor,
    /// Initial buildup duration of stage (how long until state can deal damage)
    pub buildup_duration: T,
    /// Duration of stage spent in swing (controls animation stuff, and can also
    /// be used to handle movement separately to buildup)
    pub swing_duration: T,
    /// At what fraction of the swing duration to apply the melee "hit"
    pub hit_timing: f32,
    /// Initial recover duration of stage (how long until character exits state)
    pub recover_duration: T,
    /// How much forward movement there is in the swing portion of the stage
    pub forward_movement: f32,
    /// Adjusts turning rate during the attack
    pub ori_modifier: f32,
}

impl Strike<f32> {
    pub fn to_duration(self) -> Strike<Duration> {
        Strike::<Duration> {
            melee_constructor: self.melee_constructor,
            buildup_duration: Duration::from_secs_f32(self.buildup_duration),
            swing_duration: Duration::from_secs_f32(self.swing_duration),
            hit_timing: self.hit_timing,
            recover_duration: Duration::from_secs_f32(self.recover_duration),
            forward_movement: self.forward_movement,
            ori_modifier: self.ori_modifier,
        }
    }

    #[must_use]
    pub fn adjusted_by_stats(self, stats: Stats) -> Self {
        Self {
            melee_constructor: self.melee_constructor.adjusted_by_stats(stats),
            buildup_duration: self.buildup_duration / stats.speed,
            swing_duration: self.swing_duration / stats.speed,
            hit_timing: self.hit_timing,
            recover_duration: self.recover_duration / stats.speed,
            forward_movement: self.forward_movement,
            ori_modifier: self.ori_modifier,
        }
    }
}

// TODO: Completely rewrite this with skill tree rework. Don't bother converting
// to melee constructor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Separated out to condense update portions of character state
pub struct StaticData {
    /// Data for each stage
    pub strikes: Vec<Strike<Duration>>,
    /// What key is used to press ability
    pub ability_info: AbilityInfo,
}
/// A sequence of attacks that can incrementally become faster and more
/// damaging.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// Struct containing data that does not change over the course of the
    /// character state
    pub static_data: StaticData,
    /// Whether the attack was executed already
    pub exhausted: bool,
    /// Whether the strike should skip recover
    pub skip_recover: bool,
    /// Timer for each stage
    pub timer: Duration,
    /// Checks what section a strike is in, if a strike is currently being
    /// performed
    pub stage_section: Option<StageSection>,
    /// Index of the strike that is currently in progress, or if not in a strike
    /// currently the next strike that will occur
    pub completed_strikes: usize,
}

pub const STANCE_TIME: Duration = Duration::from_secs(5);

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        handle_orientation(data, &mut update, 1.0, None);
        handle_move(data, &mut update, 0.7);
        handle_dodge_interrupt(data, &mut update, Some(InputKind::Primary));

        let strike_data =
            self.static_data.strikes[self.completed_strikes % self.static_data.strikes.len()];

        match self.stage_section {
            Some(StageSection::Buildup) => {
                if self.timer < strike_data.buildup_duration {
                    // Build up
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else {
                    // Transitions to swing section of stage
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = Duration::default();
                        c.stage_section = Some(StageSection::Action);
                    }
                }
            },
            Some(StageSection::Action) => {
                if input_is_pressed(data, InputKind::Primary) {
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.skip_recover = true;
                    }
                }
                if self.timer.as_secs_f32()
                    > strike_data.hit_timing * strike_data.swing_duration.as_secs_f32()
                    && !self.exhausted
                {
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                        c.exhausted = true;
                    }

                    let crit_data = get_crit_data(data, self.static_data.ability_info);
                    let buff_strength = get_buff_strength(data, self.static_data.ability_info);

                    data.updater.insert(
                        data.entity,
                        strike_data
                            .melee_constructor
                            .create_melee(crit_data, buff_strength),
                    );
                } else if self.timer < strike_data.swing_duration {
                    // Swings
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else if self.skip_recover {
                    next_strike(&mut update)
                } else {
                    // Transitions to recover section of stage
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = Duration::default();
                        c.stage_section = Some(StageSection::Recover);
                    }
                }
            },
            Some(StageSection::Recover) => {
                if self.timer < strike_data.recover_duration {
                    // Recovery
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else {
                    // Done
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = Duration::default();
                        c.stage_section = None;
                    }
                }
            },
            Some(_) => {
                // If it somehow ends up in an incorrect stage section
                update.character = CharacterState::Wielding(wielding::Data { is_sneaking: false });
                // Make sure attack component is removed
                data.updater.remove::<Melee>(data.entity);
            },
            None => {
                if self.timer < STANCE_TIME {
                    if let CharacterState::ComboMelee2(c) = &mut update.character {
                        c.timer = tick_attack_or_default(data, self.timer, None);
                    }
                } else {
                    // Done
                    update.character =
                        CharacterState::Wielding(wielding::Data { is_sneaking: false });
                    // Make sure melee component is removed
                    data.updater.remove::<Melee>(data.entity);
                }

                handle_climb(data, &mut update);
                handle_jump(data, output_events, &mut update, 1.0);

                if input_is_pressed(data, InputKind::Primary) {
                    next_strike(&mut update)
                } else {
                    attempt_input(data, output_events, &mut update);
                }
            },
        }

        update
    }
}

fn next_strike(update: &mut StateUpdate) {
    if let CharacterState::ComboMelee2(c) = &mut update.character {
        c.exhausted = false;
        c.skip_recover = false;
        c.timer = Duration::default();
        c.stage_section = Some(StageSection::Buildup);
        c.completed_strikes += 1;
    }
}
