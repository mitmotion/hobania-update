use crate::{column::ColumnSample, sim::SimChunk, util::{RandomPerm, Sampler, StructureGen2d}, Canvas, CONFIG};
use common::terrain::{Block, BlockKind, SpriteKind};
use noise::NoiseFn;
use rand::prelude::*;
use std::f32;
use vek::*;

fn close(x: f32, tgt: f32, falloff: f32) -> f32 {
    (1.0 - (x - tgt).abs() / falloff).max(0.0).powf(0.125)
}

const MUSH_FACT: f32 = 1.0e-4; // To balance things around the mushroom spawning rate
const GRASS_FACT: f32 = 1.0e-3; // To balance things around the grass spawning rate
const DEPTH_WATER_NORM: f32 = 15.0; // Water depth at which regular underwater sprites start spawning
pub fn apply_scatter_to(canvas: &mut Canvas, rng: &mut impl Rng) {
    enum WaterMode {
        Underwater,
        Floating,
        Ground,
    }
    use WaterMode::*;

    use SpriteKind::*;

    struct ScatterConfig {
        kind: SpriteKind,
        water_mode: WaterMode,
        /// (base_density_proportion, wavelen, threshold)
        patch: Option<(f32, u32, f32)>,
        permit: fn(BlockKind) -> bool,
        /// (chunk, col) -> density
        f: fn(&SimChunk, &ColumnSample) -> (f32,),
    }

    // TODO: Add back all sprites we had before
    let scatter: &[ScatterConfig] = &[
        // Flowers
        ScatterConfig {
            kind: BlueFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 256, 0.25)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.7).min(close(
                        col.humidity,
                        CONFIG.jungle_hum,
                        0.4,
                    )) * col.tree_density
                        * MUSH_FACT
                        * 256.0,
                )
            },
        },
        ScatterConfig {
            kind: PinkFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.1)),
            f: |_, col| {
                (
                    close(col.temp, 0.0, 0.7).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: PurpleFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.1)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.7).min(close(
                        col.humidity,
                        CONFIG.jungle_hum,
                        0.4,
                    )) * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: RedFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.1)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.tropical_temp, 0.7).min(close(
                        col.humidity,
                        CONFIG.jungle_hum,
                        0.4,
                    )) * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: WhiteFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.1)),
            f: |_, col| {
                (
                    close(col.temp, 0.0, 0.7).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: YellowFlower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.1)),
            f: |_, col| {
                (
                    close(col.temp, 0.0, 0.7).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: Cotton,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 256, 0.25)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.7).min(close(
                        col.humidity,
                        CONFIG.jungle_hum,
                        0.4,
                    )) * col.tree_density
                        * MUSH_FACT
                        * 75.0,
                )
            },
        },
        ScatterConfig {
            kind: Sunflower,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.15)),
            f: |_, col| {
                (
                    close(col.temp, 0.0, 0.7).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        ScatterConfig {
            kind: WildFlax,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.15)),
            f: |_, col| {
                (
                    close(col.temp, 0.0, 0.7).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 600.0,
                )
            },
        },
        // Herbs and Spices
        ScatterConfig {
            kind: LingonBerry,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.jungle_hum, 0.5))
                        * MUSH_FACT
                        * 2.5,
                )
            },
        },
        ScatterConfig {
            kind: LeafyPlant,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.jungle_hum, 0.3))
                        * GRASS_FACT
                        * 4.0,
                )
            },
        },
        ScatterConfig {
            kind: JungleLeafyPlant,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.15, 64, 0.2)),
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * GRASS_FACT
                        * 32.0,
                )
            },
        },
        ScatterConfig {
            kind: Fern,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 64, 0.2)),
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.forest_hum, 0.5))
                        * GRASS_FACT
                        * 0.25,
                )
            },
        },
        ScatterConfig {
            kind: JungleFern,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 84, 0.35)),
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 200.0,
                )
            },
        },
        ScatterConfig {
            kind: Blueberry,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.5).min(close(
                        col.humidity,
                        CONFIG.forest_hum,
                        0.5,
                    )) * MUSH_FACT
                        * 0.3,
                )
            },
        },
        ScatterConfig {
            kind: Pumpkin,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 512, 0.05)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.5).min(close(
                        col.humidity,
                        CONFIG.forest_hum,
                        0.5,
                    )) * MUSH_FACT
                        * 500.0,
                )
            },
        },
        // Collectable Objects
        // Only spawn twigs in temperate forests
        ScatterConfig {
            kind: Twigs,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    (col.tree_density * 1.25 - 0.25).powf(0.5).max(0.0) * 0.75e-3,
                )
            },
        },
        // Only spawn logs in temperate forests (arbitrarily set to ~20% twig density)
        ScatterConfig {
            kind: Wood,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    (col.tree_density * 1.25 - 0.25).powf(0.5).max(0.0) * 0.15e-3,
                )
            },
        },
        ScatterConfig {
            kind: Stones,
            water_mode: Ground,
            permit: |b| {
                matches!(
                    b,
                    BlockKind::Earth | BlockKind::Grass | BlockKind::Rock | BlockKind::Sand
                )
            },
            patch: None,
            f: |chunk, _| ((chunk.rockiness - 0.5).max(0.025) * 1.0e-3,),
        },
        ScatterConfig {
            kind: Copper,
            water_mode: Ground,
            permit: |b| {
                matches!(
                    b,
                    BlockKind::Earth | BlockKind::Grass | BlockKind::Rock | BlockKind::Sand
                )
            },
            patch: None,
            f: |chunk, _| ((chunk.rockiness - 0.5).max(0.0) * 1.5e-3,),
        },
        ScatterConfig {
            kind: Tin,
            water_mode: Ground,
            permit: |b| {
                matches!(
                    b,
                    BlockKind::Earth | BlockKind::Grass | BlockKind::Rock | BlockKind::Sand
                )
            },
            patch: None,
            f: |chunk, _| ((chunk.rockiness - 0.5).max(0.0) * 1.5e-3,),
        },
        // Don't spawn Mushrooms in snowy regions
        ScatterConfig {
            kind: Mushroom,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.forest_hum, 0.35))
                        * MUSH_FACT,
                )
            },
        },
        // Grass
        ScatterConfig {
            kind: ShortGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.3, 64, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, 0.2, 0.75).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * GRASS_FACT
                        * 150.0,
                )
            },
        },
        ScatterConfig {
            kind: MediumGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.3, 64, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, 0.2, 0.6).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * GRASS_FACT
                        * 120.0,
                )
            },
        },
        ScatterConfig {
            kind: LongGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.1, 48, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.35).min(close(col.humidity, CONFIG.jungle_hum, 0.3))
                        * GRASS_FACT
                        * 150.0,
                )
            },
        },
        ScatterConfig {
            kind: JungleRedGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 128, 0.25)),
            f: |_, col| {
                (
                    close(col.temp, 0.3, 0.4).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * col.tree_density
                        * MUSH_FACT
                        * 350.0,
                )
            },
        },
        // Jungle Sprites
        // (LongGrass, Ground, |c, col| {
        //     (
        //         close(col.temp, CONFIG.tropical_temp, 0.4).min(close(
        //             col.humidity,
        //             CONFIG.jungle_hum,
        //             0.6,
        //         )) * 0.08,
        //         Some((0.0, 60, 5.0)),
        //     )
        // }),
        /*(WheatGreen, Ground, |c, col| {
            (
                close(col.temp, 0.4, 0.2).min(close(col.humidity, CONFIG.forest_hum, 0.1))
                    * MUSH_FACT
                    * 0.001,
                None,
            )
        }),*/
        ScatterConfig {
            kind: GrassSnow,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 48, 0.2)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.snow_temp - 0.2, 0.4).min(close(
                        col.humidity,
                        CONFIG.forest_hum,
                        0.5,
                    )) * GRASS_FACT
                        * 100.0,
                )
            },
        },
        ScatterConfig {
            kind: Moonbell,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 48, 0.2)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.snow_temp - 0.2, 0.4).min(close(
                        col.humidity,
                        CONFIG.forest_hum,
                        0.5,
                    )) * 0.003,
                )
            },
        },
        // Savanna Plants
        ScatterConfig {
            kind: SavannaGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.15, 64, 0.2)),
            f: |_, col| {
                (
                    {
                        let savanna = close(col.temp, 1.0, 0.4) * close(col.humidity, 0.2, 0.25);
                        let desert = close(col.temp, 1.0, 0.25) * close(col.humidity, 0.0, 0.1);
                        (savanna - desert * 5.0).max(0.0) * GRASS_FACT * 250.0
                    },
                )
            },
        },
        ScatterConfig {
            kind: TallSavannaGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.1, 48, 0.2)),
            f: |_, col| {
                (
                    {
                        let savanna = close(col.temp, 1.0, 0.4) * close(col.humidity, 0.2, 0.25);
                        let desert = close(col.temp, 1.0, 0.25) * close(col.humidity, 0.0, 0.1);
                        (savanna - desert * 5.0).max(0.0) * GRASS_FACT * 150.0
                    },
                )
            },
        },
        ScatterConfig {
            kind: RedSavannaGrass,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.15, 48, 0.25)),
            f: |_, col| {
                (
                    {
                        let savanna = close(col.temp, 1.0, 0.4) * close(col.humidity, 0.2, 0.25);
                        let desert = close(col.temp, 1.0, 0.25) * close(col.humidity, 0.0, 0.1);
                        (savanna - desert * 5.0).max(0.0) * GRASS_FACT * 120.0
                    },
                )
            },
        },
        ScatterConfig {
            kind: SavannaBush,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.1, 96, 0.15)),
            f: |_, col| {
                (
                    {
                        let savanna = close(col.temp, 1.0, 0.4) * close(col.humidity, 0.2, 0.25);
                        let desert = close(col.temp, 1.0, 0.25) * close(col.humidity, 0.0, 0.1);
                        (savanna - desert * 5.0).max(0.0) * GRASS_FACT * 40.0
                    },
                )
            },
        },
        // Desert Plants
        ScatterConfig {
            kind: DeadBush,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.95).min(close(col.humidity, 0.0, 0.3)) * MUSH_FACT * 7.5,
                )
            },
        },
        ScatterConfig {
            kind: Pyrebloom,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.95).min(close(col.humidity, 0.0, 0.3))
                        * MUSH_FACT
                        * 0.35,
                )
            },
        },
        ScatterConfig {
            kind: LargeCactus,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.25).min(close(col.humidity, 0.0, 0.1)) * MUSH_FACT * 1.5,
                )
            },
        },
        ScatterConfig {
            kind: RoundCactus,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.25).min(close(col.humidity, 0.0, 0.1)) * MUSH_FACT * 2.5,
                )
            },
        },
        ScatterConfig {
            kind: ShortCactus,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.25).min(close(col.humidity, 0.0, 0.1)) * MUSH_FACT * 2.5,
                )
            },
        },
        ScatterConfig {
            kind: MedFlatCactus,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.25).min(close(col.humidity, 0.0, 0.1)) * MUSH_FACT * 2.5,
                )
            },
        },
        ScatterConfig {
            kind: ShortFlatCactus,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: None,
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.25).min(close(col.humidity, 0.0, 0.1)) * MUSH_FACT * 2.5,
                )
            },
        },
        // Underwater chests
        ScatterConfig {
            kind: ChestBuried,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth | BlockKind::Sand),
            patch: None,
            f: |_, col| {
                (
                    MUSH_FACT
                        * 1.0e-6
                        * if col.alt < col.water_level - DEPTH_WATER_NORM + 30.0 {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Underwater mud piles
        ScatterConfig {
            kind: Mud,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: None,
            f: |_, col| {
                (
                    MUSH_FACT
                        * 1.0e-3
                        * if col.alt < col.water_level - DEPTH_WATER_NORM {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Underwater grass
        ScatterConfig {
            kind: GrassBlue,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 100, 0.15)),
            f: |_, col| {
                (
                    MUSH_FACT
                        * 250.0
                        * if col.alt < col.water_level - DEPTH_WATER_NORM {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // seagrass
        ScatterConfig {
            kind: Seagrass,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 150, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.8)
                        * MUSH_FACT
                        * 300.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 18.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // seagrass, coastal patches
        ScatterConfig {
            kind: Seagrass,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 150, 0.4)),
            f: |_, col| {
                (
                    MUSH_FACT
                        * 600.0
                        * if col.water_level <= CONFIG.sea_level
                            && (col.water_level - col.alt) < 3.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // scattered seaweed (temperate species)
        ScatterConfig {
            kind: SeaweedTemperate,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 500, 0.75)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.8)
                        * MUSH_FACT
                        * 50.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 11.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // scattered seaweed (tropical species)
        ScatterConfig {
            kind: SeaweedTropical,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.0, 500, 0.75)),
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.95)
                        * MUSH_FACT
                        * 50.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 11.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Caulerpa lentillifera algae patch
        ScatterConfig {
            kind: SeaGrapes,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 100, 0.15)),
            f: |_, col| {
                (
                    MUSH_FACT
                        * 250.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 10.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Caulerpa prolifera algae patch
        ScatterConfig {
            kind: WavyAlgae,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 100, 0.15)),
            f: |_, col| {
                (
                    MUSH_FACT
                        * 250.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 10.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Mermaids' fan algae patch
        ScatterConfig {
            kind: MermaidsFan,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 50, 0.10)),
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.95)
                        * MUSH_FACT
                        * 500.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 10.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Sea anemones
        ScatterConfig {
            kind: SeaAnemone,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 100, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.8)
                        * MUSH_FACT
                        * 125.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM - 9.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Giant Kelp
        ScatterConfig {
            kind: GiantKelp,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 200, 0.4)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.8)
                        * MUSH_FACT
                        * 220.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM - 9.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Bull Kelp
        ScatterConfig {
            kind: BullKelp,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 75, 0.3)),
            f: |_, col| {
                (
                    close(col.temp, CONFIG.temperate_temp, 0.7)
                        * MUSH_FACT
                        * 300.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 3.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Stony Corals
        ScatterConfig {
            kind: StonyCoral,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 120, 0.4)),
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.9)
                        * MUSH_FACT
                        * 160.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 10.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Soft Corals
        ScatterConfig {
            kind: SoftCoral,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: Some((0.0, 120, 0.4)),
            f: |_, col| {
                (
                    close(col.temp, 1.0, 0.9)
                        * MUSH_FACT
                        * 120.0
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 10.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        // Seashells
        ScatterConfig {
            kind: Seashells,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: None,
            f: |c, col| {
                (
                    (c.rockiness - 0.5).max(0.0)
                        * 1.0e-3
                        * if col.water_level <= CONFIG.sea_level
                            && col.alt < col.water_level - DEPTH_WATER_NORM + 20.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        ScatterConfig {
            kind: Stones,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Earth),
            patch: None,
            f: |c, col| {
                (
                    (c.rockiness - 0.5).max(0.0)
                        * 1.0e-3
                        * if col.alt < col.water_level - DEPTH_WATER_NORM {
                            1.0
                        } else {
                            0.0
                        },
                )
            },
        },
        //River-related scatter
        ScatterConfig {
            kind: LillyPads,
            water_mode: Floating,
            permit: |_| true,
            patch: Some((0.0, 128, 0.35)),
            f: |_, col| {
                (
                    close(col.temp, 0.2, 0.6).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * GRASS_FACT
                        * 100.0
                        * ((col.alt - CONFIG.sea_level) / 12.0).clamped(0.0, 1.0)
                        * col
                            .water_dist
                            .map_or(0.0, |d| 1.0 / (1.0 + (d.abs() * 0.4).powi(2))),
                )
            },
        },
        ScatterConfig {
            kind: Reed,
            water_mode: Underwater,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.2, 128, 0.5)),
            f: |_, col| {
                (
                    close(col.temp, 0.2, 0.6).min(close(col.humidity, CONFIG.jungle_hum, 0.4))
                        * GRASS_FACT
                        * 100.0
                        * ((col.alt - CONFIG.sea_level) / 12.0).clamped(0.0, 1.0)
                        * col
                            .water_dist
                            .map_or(0.0, |d| 1.0 / (1.0 + (d.abs() * 0.40).powi(2))),
                )
            },
        },
        ScatterConfig {
            kind: Reed,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.2, 128, 0.5)),
            f: |_, col| {
                (
                    close(col.humidity, CONFIG.jungle_hum, 0.9)
                        * col
                            .water_dist
                            .map(|wd| Lerp::lerp(0.2, 0.0, (wd / 8.0).clamped(0.0, 1.0)))
                            .unwrap_or(0.0)
                        * ((col.alt - CONFIG.sea_level) / 12.0).clamped(0.0, 1.0),
                )
            },
        },
        ScatterConfig {
            kind: Bamboo,
            water_mode: Ground,
            permit: |b| matches!(b, BlockKind::Grass),
            patch: Some((0.2, 128, 0.5)),
            f: |_, col| {
                (
                    0.014
                        * close(col.humidity, CONFIG.jungle_hum, 0.9)
                        * col
                            .water_dist
                            .map(|wd| Lerp::lerp(0.2, 0.0, (wd / 8.0).clamped(0.0, 1.0)))
                            .unwrap_or(0.0)
                        * ((col.alt - CONFIG.sea_level) / 12.0).clamped(0.0, 1.0),
                )
            },
        },
    ];

    let canvas_area = canvas.area();
    let canvas_size = canvas.column_grid.size() - 1;
    if canvas_size.reduce_partial_min() < 0 {
        return;
    }
    let corner_cols = [
        canvas.column_grid.get(Vec2::new(0, 0)).expect("Definitely in bounds"),
        canvas.column_grid.get(Vec2::new(canvas_size.x, 0)).expect("Definitely in bounds"),
        canvas.column_grid.get(Vec2::new(0, canvas_size.y)).expect("Definitely in bounds"),
        canvas.column_grid.get(canvas_size).expect("Definitely in bounds"),
    ];
    let chunk = canvas.chunk();
    scatter.iter().enumerate().for_each(
        |(
            i,
            config,
        )| {
            // NOTE: We know the number of sprite kinds can't exceed u32::MAX, because that's
            // the maximum amount of information contained in a Block (in practice, for many
            // reasons we don't have anywhere close to that many unique sprite kinds).  Since
            // we don't have more scatter configs than sprite kinds (currently at least),
            // this cast from usize to u32 must therefore always be valid.
            let i = i as u32;

            let corner_densities = corner_cols.map(|col| (config.f)(chunk, col).0);
            // NOTE: Hack to try to rule out sprites in the chunk by just looking at the corners.
            if corner_densities == [0.0; 4] {
                return;
            }

            fn draw_sprites(
                canvas: &mut Canvas,
                rng: &mut impl Rng,
                ScatterConfig {
                    kind,
                    water_mode,
                    permit,
                    ..
                }: &ScatterConfig,
                corner_densities: &[f32; 4],
                aabr: Aabr<i32>,
                mut f: impl FnMut(&ColumnSample) -> f32,
                mut filter: impl FnMut(Vec2<i32>) -> bool,
            ) {

    canvas.foreach_col_area(aabr, /*Aabr { min: canvas.wpos(), max: canvas.wpos() + 1 }, */|canvas, wpos2d, col| {
        let underwater = col.water_level.floor() > col.alt;

        let /*kind*/(kind, water_mode) = /* scatter.iter().enumerate().find_map(
            |(
                i,
                ScatterConfig {
                    kind,
                    water_mode,
                    permit,
                    f,
                },
            )| */{
                let block_kind = canvas
                    .get(Vec3::new(wpos2d.x, wpos2d.y, col.alt as i32))
                    .kind();
                if !permit(block_kind) {
                    return;
                }
                if !filter(wpos2d) {
                    return;
                }
                let density = f(col);
                /* let density = patch
                    .map(|(base_density_prop, wavelen, threshold)| {
                        if canvas
                            .index()
                            .noise
                            .scatter_nz
                            .get(
                                wpos2d
                                    .map(|e| e as f64 / wavelen as f64 + i as f64 * 43.0)
                                    .into_array(),
                            )
                            .abs()
                            > 1.0 - threshold as f64
                        {
                            density
                        } else {
                            density * base_density_prop
                        }
                    })
                    .unwrap_or(density); */
                if density > 0.0
                    && rng.gen::<f32>() < density //RandomField::new(i as u32).chance(Vec3::new(wpos2d.x, wpos2d.y, 0), density)
                    && matches!(&water_mode, Underwater | Floating) == underwater
                {
                    (*kind, water_mode)
                } else {
                    return;
                }
            }/*,
        )*/;

        /*if let Some((kind, water_mode)) = kind */{
            let (alt, is_under): (_, fn(Block) -> bool) = match water_mode {
                Ground | Underwater => (col.alt as i32, |block| block.is_solid()),
                Floating => (col.water_level as i32, |block| !block.is_air()),
            };

            // Find the intersection between ground and air, if there is one near the
            // Ground
            if let Some(solid_end) = (-4..8)
                .find(|z| is_under(canvas.get(Vec3::new(wpos2d.x, wpos2d.y, alt + z))))
                .and_then(|solid_start| {
                    (1..8)
                        .map(|z| solid_start + z)
                        .find(|z| !is_under(canvas.get(Vec3::new(wpos2d.x, wpos2d.y, alt + z))))
                })
            {
                canvas.map(Vec3::new(wpos2d.x, wpos2d.y, alt + solid_end), |block| {
                    block.with_sprite(kind)
                });
            }
        }
    });
            }

            let base_density_prop = if let Some((base_density_prop, wavelen, threshold)) = config.patch {
                // Compute GenStructure2D for this sprite kind, and iterate over each patch, with:
                //
                // seed = i
                //
                // FIXME: Justify conversion (threshold limits?)
                // spread = (sizei / threshold)
                // NOTE: Safe conversion because it was a positive i32.
                let spread = (wavelen as f32 / 32.0/* / threshold*/) as u32;
                // freq = spread * 2
                let freq = spread << 1;
                let scatter_gen = StructureGen2d::new(i, freq, spread);
                /* let perm = RandomPerm::new(seed); */
                let scatter_nz = /*RandomPerm::new(seed)*/&canvas.index().noise.scatter_nz;

                /* let probability = threshold * threshold; */
                scatter_gen
                    .iter(canvas_area.min, canvas_area.max)
                    // NOTE: Rejection sample against threshold to determine whether to draw this
                    // patch at all.
                    .filter(|&(wpos, _)| {
                        let wposf = wpos.as_::<f64>() / wavelen as f64 + i as f64 * 43.0;
                        scatter_nz.get(wposf) > 1.0 - threshold
                    })
                    // TODO: Consider checking against the threshold to decide whether to draw
                    // the scatter field at all.
                    .for_each(|(wpos, seed)| {
                        /* let size_factor = RandomPerm::new(i).get_f32(1); */
                        // size = wavelen / 16
                        //
                        // FIXME: Justify conversion (maybe limit spread to a u16?).
                        let sizei = (wavelen as f32 / 24.0/* * size_factor*/) as i32;
                        // FIXME: Justify no overflow (maybe limit spread to a u16?).
                        let size2 = sizei * sizei;

                        // Sample within the aabb surrounding the structure.
                        let mut aabr = Aabr {
                            min: wpos - sizei,
                            max: wpos + sizei,
                        };
                        aabr.intersect(canvas_area);
                        /* let scatter_nz = /*RandomField::new(seed)*/
                                &canvas
                                    .index()
                                    .noise
                                    .scatter_nz; */
                        draw_sprites(
                            canvas, rng, config, &corner_densities, aabr,
                            |col| (config.f)(chunk, col).0,
                            |pos| {
                                let dist2 = pos.distance_squared(wpos);
                                dist2 < size2/* && /*scatter_nz.chance(Vec3::new(pos.x, pos.y, 0), threshold)*/
                                /*canvas
                                    .index()
                                    .noise
                                    .*/scatter_nz
                                    .get(
                                        /*wpos2d*/pos
                                            .map(|e| e as f64 / wavelen as f64 + i as f64 * 43.0)
                                            .into_array(),
                                    )
                                    .abs()
                                    > 1.0 - threshold as f64*/
                            },
                        );
                    });
                // Return the non-scatter density multiplier
                base_density_prop
            } else {
                // There is no GenStructure2D to further restrict the sprite's bounds, so we sample
                // over the whole chunk using a 1.0 density multiplier.
                1.0
            };
            if base_density_prop == 0.0 {
                // Sprite doesn't get drawn outside of its patches.
                return;
            }
            // TODO: Draw sprites at the base density over the whole chunk.
            //
            // NOTE: This could be very expensive, would we consider just not doing this?
            /* let scatter_nz = RandomField::new(i); */
            draw_sprites(
                canvas, rng, config, &corner_densities, canvas_area,
                |col| base_density_prop * (config.f)(chunk, col).0,
                |pos| /*scatter_nz.chance(Vec3::new(pos.x, pos.y, 0), threshold)*/true,
            );
    });

}
