use crate::{
    comp::{character_state::OutputEvents, CharacterState, StateUpdate},
    event::ServerEvent,
    spiral::Spiral2d,
    states::{
        behavior::{CharacterBehavior, JoinData},
        utils::*,
        wielding,
    },
    terrain::{Block, SpriteKind},
    vol::ReadVol,
};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use vek::*;

/// Separated out to condense update portions of character state
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StaticData {
    /// How long the state builds up for
    pub buildup_duration: Duration,
    /// How long the state is casting for
    pub cast_duration: Duration,
    /// How long the state recovers for
    pub recover_duration: Duration,
    /// What kind of sprite is created by this state
    pub sprite: SpriteKind,
    /// Range that sprites are created relative to the summonner
    pub summon_distance: (f32, f32),
    /// Chance that sprite is not created on a particular square
    pub sparseness: f64,
    /// Miscellaneous information about the ability
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
    /// What radius of sprites have already been summoned
    pub achieved_radius: i32,
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData, output_events: &mut OutputEvents) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        match self.stage_section {
            StageSection::Buildup => {
                if self.timer < self.static_data.buildup_duration {
                    // Build up
                    update.character = CharacterState::SpriteSummon(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        ..*self
                    });
                } else {
                    // Transitions to recover section of stage
                    update.character = CharacterState::SpriteSummon(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Action,
                        ..*self
                    });
                }
            },
            StageSection::Action => {
                if self.timer < self.static_data.cast_duration {
                    let timer_frac =
                        self.timer.as_secs_f32() / self.static_data.cast_duration.as_secs_f32();

                    // Determines distance from summoner sprites should be created. Goes outward
                    // with time.
                    let summon_distance = timer_frac
                        * (self.static_data.summon_distance.1 - self.static_data.summon_distance.0)
                        + self.static_data.summon_distance.0;
                    let summon_distance = summon_distance.round() as i32;

                    // Only summons sprites if summon distance is greater than achieved radius
                    for radius in self.achieved_radius..=summon_distance {
                        // 1 added to make range correct, too lazy to add 1 to both variables above
                        let radius = radius + 1;
                        // Creates a spiral iterator for the newly achieved radius
                        let spiral = Spiral2d::with_edge_radius(radius);
                        for point in spiral {
                            // If square is not sparse, generate sprite
                            if !thread_rng().gen_bool(self.static_data.sparseness) {
                                // The coordinates of where the sprite is created
                                let sprite_pos = Vec3::new(
                                    data.pos.0.x.floor() as i32 + point.x,
                                    data.pos.0.y.floor() as i32 + point.y,
                                    data.pos.0.z.floor() as i32,
                                );

                                // Check for collision in z up to 10 blocks up or down
                                let obstacle_z = data
                                    .terrain
                                    .ray(
                                        sprite_pos.map(|x| x as f32 + 0.5) + Vec3::unit_z() * 10.0,
                                        sprite_pos.map(|x| x as f32 + 0.5) - Vec3::unit_z() * 10.0,
                                    )
                                    .until(|b| {
                                        // Until reaching a solid block that is not the created
                                        // sprite
                                        Block::is_solid(b)
                                            && b.get_sprite() != Some(self.static_data.sprite)
                                    })
                                    .cast()
                                    .0;

                                // z height relative to caster
                                let z = sprite_pos.z + (10.5 - obstacle_z).ceil() as i32;

                                // Location sprite will be created
                                let sprite_pos =
                                    Vec3::new(sprite_pos.x as i32, sprite_pos.y as i32, z);
                                // Layers of sprites
                                let layers = match self.static_data.sprite {
                                    SpriteKind::SeaUrchin => 2,
                                    _ => 1,
                                };
                                for i in 0..layers {
                                    // Send server event to create sprite
                                    output_events.emit_server(ServerEvent::CreateSprite {
                                        pos: Vec3::new(sprite_pos.x as i32, sprite_pos.y, z + i),
                                        sprite: self.static_data.sprite,
                                    });
                                }
                            }
                        }
                    }

                    update.character = CharacterState::SpriteSummon(Data {
                        timer: tick_attack_or_default(data, self.timer, None),
                        achieved_radius: summon_distance,
                        ..*self
                    });
                } else {
                    // Transitions to recover section of stage
                    update.character = CharacterState::SpriteSummon(Data {
                        timer: Duration::default(),
                        stage_section: StageSection::Recover,
                        ..*self
                    });
                }
            },
            StageSection::Recover => {
                if self.timer < self.static_data.recover_duration {
                    // Recovery
                    update.character = CharacterState::SpriteSummon(Data {
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

        update
    }
}
