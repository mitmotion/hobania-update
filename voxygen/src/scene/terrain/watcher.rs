use crate::hud::CraftingTab;
use common::terrain::{BlockKind, SpriteKind, TerrainChunk, TERRAIN_CHUNK_BLOCKS_LG};
use common_base::span;
use rand::prelude::*;
use vek::*;

#[derive(Copy, Clone, Debug)]
pub enum Interaction {
    Collect,
    Craft(CraftingTab),
    Mine,
}

pub enum FireplaceType {
    House,
    Workshop, // this also includes witch hut
}

pub struct SmokerProperties {
    pub position: Vec3<i32>,
    pub kind: FireplaceType,
}

impl SmokerProperties {
    fn new(position: Vec3<i32>, kind: FireplaceType) -> Self { Self { position, kind } }
}

#[derive(Default)]
pub struct BlocksOfInterest {
    pub leaves: Vec<Vec3<i32>>,
    pub drip: Vec<Vec3<i32>>,
    pub grass: Vec<Vec3<i32>>,
    pub slow_river: Vec<Vec3<i32>>,
    pub fast_river: Vec<Vec3<i32>>,
    pub fires: Vec<Vec3<i32>>,
    pub smokers: Vec<SmokerProperties>,
    pub beehives: Vec<Vec3<i32>>,
    pub reeds: Vec<Vec3<i32>>,
    pub fireflies: Vec<Vec3<i32>>,
    pub flowers: Vec<Vec3<i32>>,
    pub fire_bowls: Vec<Vec3<i32>>,
    pub snow: Vec<Vec3<i32>>,
    //This is so crickets stay in place and don't randomly change sounds
    pub cricket1: Vec<Vec3<i32>>,
    pub cricket2: Vec<Vec3<i32>>,
    pub cricket3: Vec<Vec3<i32>>,
    pub frogs: Vec<Vec3<i32>>,
    // Note: these are only needed for chunks within the iteraction range so this is a potential
    // area for optimization
    pub interactables: Vec<(Vec3<i32>, Interaction)>,
    pub lights: Vec<(Vec3<i32>, u8)>,
    // needed for biome specific smoke variations
    pub temperature: f32,
    pub humidity: f32,
}

impl BlocksOfInterest {
    pub fn from_chunk(chunk: &TerrainChunk) -> (Self, Vec<(Vec3<i32>, (SpriteKind, Option<u8>))>) {
        span!(_guard, "from_chunk", "BlocksOfInterest::from_chunk");
        let mut leaves = Vec::new();
        let mut drip = Vec::new();
        let mut grass = Vec::new();
        let mut slow_river = Vec::new();
        let mut fast_river = Vec::new();
        let mut fires = Vec::new();
        let mut smokers = Vec::new();
        let mut beehives = Vec::new();
        let mut reeds = Vec::new();
        let mut fireflies = Vec::new();
        let mut flowers = Vec::new();
        let mut interactables = Vec::new();
        let mut lights = Vec::new();
        // Lights that can be omitted at random if we have too many and need to cull
        // some of them
        let mut minor_lights = Vec::new();
        let mut fire_bowls = Vec::new();
        let mut snow = Vec::new();
        let mut cricket1 = Vec::new();
        let mut cricket2 = Vec::new();
        let mut cricket3 = Vec::new();
        let mut frogs = Vec::new();
        let mut sprite_kinds = Vec::new();

        let mut rng = SmallRng::from_seed(thread_rng().gen());

        let river_speed_sq = chunk.meta().river_velocity().magnitude_squared();
        let z_offset = chunk.get_min_z();

        const LEAF_BITS: u32 = 4;
        const DRIP_BITS: u32 = 3;
        const SNOW_BITS: u32 = 4;
        const LAVA_BITS: u32 = 2;
        const GRASS_BITS: u32 = 4;
        const CRICKET_BITS: u32 = 13;
        const FROG_BITS: u32 = 4;

        const LEAF_MASK: u64 = (1 << LEAF_BITS) - 1;
        const DRIP_MASK: u64 = (1 << DRIP_BITS) - 1;
        const SNOW_MASK: u64 = (1 << SNOW_BITS) - 1;
        const LAVA_MASK: u64 = (1 << LAVA_BITS) - 1;
        // NOTE: Grass and cricket bits are merged together to save a call to the rng.
        const CRICKET_MASK: u64 = ((1 << CRICKET_BITS) - 1);
        const GRASS_MASK: u64 = ((1 << GRASS_BITS) - 1) << CRICKET_BITS;
        const FROG_MASK: u64 = (1 << FROG_BITS) - 1;

        // NOTE: Z chunk total height cannot exceed 2^14, so x+y+z fits in 24 bits.  Therefore we
        // know -1 is never a valid height, so it's okay to use as a representative of "no river".
        let mut river_data = [-1i16; (1 << TERRAIN_CHUNK_BLOCKS_LG * 2)];

        let river = if river_speed_sq > 0.9_f32.powi(2) {
            Some(&mut fast_river)
        } else if river_speed_sq > 0.3_f32.powi(2) {
            Some(&mut slow_river)
        } else {
            None
        };

        chunk.iter_changed().for_each(|(index, block)| {
            // FIXME: Before merge, make this properly generic.
            #[inline(always)]
            fn make_pos_raw(index: u32) -> Vec3<i32> {
                let grp = index >> 6;
                let rel = index & 63;
                let x = ((grp & 7) << 2) | (rel & 3);
                let y = (((grp >> 3) & 7) << 2) | ((rel >> 2) & 3);
                let z = ((grp >> 6) << 2) | (rel >> 4);
                Vec3::new(x as i32, y as i32, z as i32)
            };
            let make_pos = #[inline(always)] |index: u32| {
                let mut pos = make_pos_raw(index);
                pos.z += z_offset;
                pos
            };

            let mut do_sprite = |sprite: SpriteKind, rng: &mut SmallRng| {
                let pos = make_pos(index);
                match sprite {
                    SpriteKind::Ember => {
                        fires.push(pos);
                        smokers.push(SmokerProperties::new(pos, FireplaceType::House));
                    },
                    // Offset positions to account for block height.
                    // TODO: Is this a good idea?
                    SpriteKind::StreetLamp => fire_bowls.push(pos + Vec3::unit_z() * 2),
                    SpriteKind::FireBowlGround => fire_bowls.push(pos + Vec3::unit_z()),
                    SpriteKind::StreetLampTall => fire_bowls.push(pos + Vec3::unit_z() * 4),
                    SpriteKind::WallSconce => fire_bowls.push(pos + Vec3::unit_z()),
                    SpriteKind::Beehive => beehives.push(pos),
                    SpriteKind::CrystalHigh => fireflies.push(pos),
                    SpriteKind::Reed => {
                        reeds.push(pos);
                        fireflies.push(pos);
                        if rng.next_u64() & FROG_MASK == 0 {
                            frogs.push(pos);
                        }
                    },
                    SpriteKind::CaveMushroom => fireflies.push(pos),
                    SpriteKind::PinkFlower => flowers.push(pos),
                    SpriteKind::PurpleFlower => flowers.push(pos),
                    SpriteKind::RedFlower => flowers.push(pos),
                    SpriteKind::WhiteFlower => flowers.push(pos),
                    SpriteKind::YellowFlower => flowers.push(pos),
                    SpriteKind::Sunflower => flowers.push(pos),
                    SpriteKind::CraftingBench => {
                        interactables.push((pos, Interaction::Craft(CraftingTab::All)))
                    },
                    SpriteKind::SmokeDummy => {
                        smokers.push(SmokerProperties::new(pos, FireplaceType::Workshop));
                    },
                    SpriteKind::Forge => interactables
                        .push((pos, Interaction::Craft(CraftingTab::ProcessedMaterial))),
                    SpriteKind::TanningRack => interactables
                        .push((pos, Interaction::Craft(CraftingTab::ProcessedMaterial))),
                    SpriteKind::SpinningWheel => {
                        interactables.push((pos, Interaction::Craft(CraftingTab::All)))
                    },
                    SpriteKind::Loom => {
                        interactables.push((pos, Interaction::Craft(CraftingTab::All)))
                    },
                    SpriteKind::Cauldron => {
                        fires.push(pos);
                        interactables.push((pos, Interaction::Craft(CraftingTab::Potion)))
                    },
                    SpriteKind::Anvil => {
                        interactables.push((pos, Interaction::Craft(CraftingTab::Weapon)))
                    },
                    SpriteKind::CookingPot => {
                        fires.push(pos);
                        interactables.push((pos, Interaction::Craft(CraftingTab::Food)))
                    },
                    SpriteKind::DismantlingBench => {
                        fires.push(pos);
                        interactables.push((pos, Interaction::Craft(CraftingTab::Dismantle)))
                    },
                    _ => {},
                }
                sprite_kinds.push((pos, (sprite, sprite.has_ori().then(|| block.get_ori_raw()))));
                if sprite.is_collectible() {
                    interactables.push((pos, Interaction::Collect));
                }
                sprite.get_glow()
            };
            let mut has_sprite = false;
            let glow = match block.kind() {
                kind @ BlockKind::Leaves => {
                    if rng.next_u64() & LEAF_MASK == 0 { leaves.push(make_pos(index)); }
                    kind.get_glow_raw()
                },
                kind @ BlockKind::WeakRock => {
                    if rng.next_u64() & DRIP_MASK == 0 { drip.push(make_pos(index)); }
                    kind.get_glow_raw()
                },
                kind @ BlockKind::Grass => {
                    let bits = rng.next_u64();
                    if bits & GRASS_MASK == 0 {
                        grass.push(make_pos(index));
                    }
                    match bits & CRICKET_MASK {
                        1 => cricket1.push(make_pos(index)),
                        2 => cricket2.push(make_pos(index)),
                        3 => cricket3.push(make_pos(index)),
                        _ => {},
                    }
                    kind.get_glow_raw()
                },
                // Assign a river speed to water blocks depending on river velocity
                kind @ BlockKind::Water => {
                    if river.is_some() {
                        // Remember only the top river blocks.  Since we always go from low to high z, this assignment
                        // can be unconditional.
                        let mut pos = make_pos_raw(index);
                        river_data[((pos.y << TERRAIN_CHUNK_BLOCKS_LG) | pos.x) as usize] = pos.z as i16;
                    }
                    if let Some(sprite) = block.get_sprite_raw() {
                        has_sprite = true;
                        do_sprite(sprite, &mut rng)
                    } else {
                        kind.get_glow_raw()
                    }
                }
                kind @ BlockKind::Lava => {
                    if rng.next_u64() & LAVA_MASK == 0 { fires.push(make_pos(index) + Vec3::unit_z()) }
                    kind.get_glow_raw()
                },
                kind @ BlockKind::Snow | kind @ BlockKind::Ice => {
                    if rng.next_u64() & SNOW_MASK == 0 { snow.push(make_pos(index)); }
                    kind.get_glow_raw()
                },
                kind @ BlockKind::Air => if let Some(sprite) = block.get_sprite_raw() {
                    has_sprite = true;
                    do_sprite(sprite, &mut rng)
                } else {
                    kind.get_glow_raw()
                },
                kind @ BlockKind::Rock | kind @ BlockKind::GlowingRock | kind @ BlockKind::GlowingWeakRock |
                kind @ BlockKind::GlowingMushroom |
                kind @ BlockKind::Earth | kind @ BlockKind::Sand | kind @ BlockKind::Wood | kind @ BlockKind::Misc => {
                    kind.get_glow_raw()
                }
            };

            if let Some(glow) = glow {
                let pos = make_pos(index);
                // Currently, we count filled blocks as 'minor' lights, and sprites as
                // non-minor.
                if has_sprite {
                    minor_lights.push((pos, glow));
                } else {
                    lights.push((pos, glow));
                }
            }
        });

        // Convert river grid to vector.
        const X_MASK: usize = (1 << TERRAIN_CHUNK_BLOCKS_LG) - 1;

        if let Some(river) = river {
            river.extend(
                river_data.into_iter().enumerate()
                // Avoid blocks with no water
                .filter(|&(_, river_block)| river_block != -1)
                .map(|(index, river_block)|
                     Vec3::new(
                        (index & X_MASK) as i32,
                        (index >> TERRAIN_CHUNK_BLOCKS_LG) as i32,
                        z_offset + i32::from(river_block),
                    ))
            );
        }

        // TODO: Come up with a better way to prune many light sources: grouping them
        // into larger lights with k-means clustering, perhaps?
        const MAX_MINOR_LIGHTS: usize = 64;
        lights.extend(
            minor_lights
                .choose_multiple(&mut rng, MAX_MINOR_LIGHTS)
                .copied(),
        );

        (Self {
            leaves,
            drip,
            grass,
            slow_river,
            fast_river,
            fires,
            smokers,
            beehives,
            reeds,
            fireflies,
            flowers,
            fire_bowls,
            snow,
            cricket1,
            cricket2,
            cricket3,
            frogs,
            interactables,
            lights,
            temperature: chunk.meta().temp(),
            humidity: chunk.meta().humidity(),
        }, sprite_kinds)
    }
}
