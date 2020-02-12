mod natural;

use crate::{
    column::{ColumnGen, ColumnSample},
    generator::{Generator, TownGen},
    util::{RandomField, Sampler, SmallCache},
    CONFIG,
};
use common::{
    terrain::{structure::StructureBlock, Block, BlockKind, Structure},
    util::pair_4_to_combination,
    vol::{ReadVol, Vox},
};
use std::ops::{Add, Div, Mul, Neg};
use vek::*;

pub struct BlockGen<'a> {
    pub column_cache: SmallCache<Option<ColumnSample<'a>>>,
    pub column_gen: ColumnGen<'a>,
}

impl<'a> BlockGen<'a> {
    pub fn new(column_gen: ColumnGen<'a>) -> Self {
        Self {
            column_cache: SmallCache::default(),
            column_gen,
        }
    }

    pub fn sample_column<'b>(
        column_gen: &ColumnGen<'a>,
        cache: &'b mut SmallCache<Option<ColumnSample<'a>>>,
        wpos: Vec2<i32>,
    ) -> Option<&'b ColumnSample<'a>> {
        cache
            .get(Vec2::from(wpos), |wpos| column_gen.get(wpos))
            .as_ref()
    }

    fn get_cliff_height(
        column_gen: &ColumnGen<'a>,
        cache: &mut SmallCache<Option<ColumnSample<'a>>>,
        wpos: Vec2<f32>,
        close_cliffs: &[(Vec2<i32>, u32); 9],
        cliff_hill: f32,
        tolerance: f32,
    ) -> f32 {
        close_cliffs.iter().fold(
            0.0f32,
            |max_height, (cliff_pos, seed)| match Self::sample_column(
                column_gen,
                cache,
                Vec2::from(*cliff_pos),
            ) {
                Some(cliff_sample)
                    if cliff_sample.is_cliffs
                        && cliff_sample.spawn_rate > 0.5
                        && cliff_sample.spawn_rules.cliffs =>
                {
                    let cliff_pos3d = Vec3::from(*cliff_pos);

                    // Conservative range of height: [15.70, 49.33]
                    let height = (RandomField::new(seed + 1).get(cliff_pos3d) % 64) as f32
                        // [0, 63] / (1 + 3 * [0.12, 1.32]) + 3 =
                        // [0, 63] / (1 + [0.36, 3.96]) + 3 =
                        // [0, 63] / [1.36, 4.96] + 3 =
                        // [0, 63] / [1.36, 4.96] + 3 =
                        // (height min) [0, 0] + 3 = [3, 3]
                        // (height max) [12.70, 46.33] + 3 = [15.70, 49.33]
                        / (1.0 + 3.0 * cliff_sample.chaos)
                        + 3.0;
                    // Conservative range of radius: [8, 47]
                    let radius = RandomField::new(seed + 2).get(cliff_pos3d) % 48 + 8;

                    max_height.max(
                        if cliff_pos.map(|e| e as f32).distance_squared(wpos)
                            < (radius as f32 + tolerance).powf(2.0)
                        {
                            cliff_sample.alt + height * (1.0 - cliff_sample.chaos) + cliff_hill
                        } else {
                            0.0
                        },
                    )
                }
                _ => max_height,
            },
        )
    }

    pub fn get_z_cache(&mut self, wpos: Vec2<i32>) -> Option<ZCache<'a>> {
        let BlockGen {
            column_cache,
            column_gen,
        } = self;

        // Main sample
        let sample = column_gen.get(wpos)?;

        // Tree samples
        let mut structures = [None, None, None, None, None, None, None, None, None];
        sample
            .close_structures
            .iter()
            .zip(structures.iter_mut())
            .for_each(|(close_structure, structure)| {
                if let Some(st) = *close_structure {
                    let st_sample =
                        Self::sample_column(column_gen, column_cache, Vec2::from(st.pos));
                    if let Some(st_sample) = st_sample {
                        let st_sample = st_sample.clone();
                        let st_info = match st.meta {
                            None => natural::structure_gen(
                                column_gen,
                                column_cache,
                                st.pos,
                                st.seed,
                                &st_sample,
                            ),
                            Some(meta) => Some(StructureInfo {
                                pos: Vec3::from(st.pos) + Vec3::unit_z() * st_sample.alt as i32,
                                seed: st.seed,
                                meta,
                            }),
                        };
                        if let Some(st_info) = st_info {
                            *structure = Some((st_info, st_sample));
                        }
                    }
                }
            });

        Some(ZCache {
            wpos,
            sample,
            structures,
        })
    }

    pub fn get_with_z_cache(
        &mut self,
        wpos: Vec3<i32>,
        z_cache: Option<&ZCache>,
        only_structures: bool,
    ) -> Option<Block> {
        let BlockGen {
            column_cache,
            column_gen,
        } = self;
        let world = column_gen.sim;

        let sample = &z_cache?.sample;
        let &ColumnSample {
            alt,
            basement,
            chaos,
            water_level,
            water_packed,
            warp_factor,
            surface_color,
            sub_surface_color,
            //tree_density,
            //forest_kind,
            //close_structures,
            cave_xy,
            cave_alt,
            marble,
            marble_small,
            rock,
            //cliffs,
            cliff_hill,
            close_cliffs,
            temp,
            humidity,
            chunk,
            stone_col,
            ..
        } = sample;

        let structures = &z_cache?.structures;

        let wposf = wpos.map(|e| e as f64);

        let water_height = water_level;
        let water_height_min = water_height.ceil();
        let (block, height) = if !only_structures {
            let (_definitely_underground, height, on_cliff, basement_height) =
                if
                /* alt <= water_level || */
                (wposf.z as f32) < alt - 64.0 * chaos {
                    // Shortcut warping
                    (true, alt, false, basement)
                } else {
                    // Apply warping
                    let warp = world
                        .gen_ctx
                        .warp_nz
                        .get(wposf.div(24.0))
                        .max(-1.0)
                        .min(1.0)
                        /* .mul(0.5)
                        .add(0.5) */
                        .mul((chaos - 0.1).max(0.0).min(1.0).powf(2.0))
                        .mul(((wposf.z as f32) - water_height_min).abs().min(16.0));
                    let warp = Lerp::lerp(0.0, warp, warp_factor);

                    let surface_height = alt + warp;

                    let (height, on_cliff) = if (wposf.z as f32) < alt + warp - 10.0 {
                        // Shortcut cliffs
                        (surface_height, false)
                    } else {
                        let turb = Vec2::new(
                            world.gen_ctx.fast_turb_x_nz.get(wposf.div(25.0)) as f32,
                            world.gen_ctx.fast_turb_y_nz.get(wposf.div(25.0)) as f32,
                        ) * 8.0;

                        let wpos_turb = Vec2::from(wpos).map(|e: i32| e as f32) + turb;
                        let cliff_height = Self::get_cliff_height(
                            column_gen,
                            column_cache,
                            wpos_turb,
                            &close_cliffs,
                            cliff_hill,
                            0.0,
                        );

                        (
                            surface_height.max(cliff_height),
                            cliff_height > surface_height + 16.0,
                        )
                    };

                    (
                        false,
                        height,
                        on_cliff,
                        basement + height - alt,
                        /*(if water_level <= alt {
                            water_level + warp
                        } else {
                            water_level
                        })*/
                    )
                };
            let height_min = if alt <= water_height {
                height.floor()
            } else {
                // height.ceil()
                height.ceil()
                // height.ceil()
            };

            // Sample blocks

            // let stone_col = Rgb::new(195, 187, 201);

            // let dirt_col = Rgb::new(79, 67, 60);

            let _air = Block::empty();
            // let stone = Block::new(2, stone_col);
            // let surface_stone = Block::new(1, Rgb::new(200, 220, 255));
            // let dirt = Block::new(1, dirt_col);
            // let sand = Block::new(1, Rgb::new(180, 150, 50));
            // let warm_stone = Block::new(1, Rgb::new(165, 165, 130));

            // let water = Block::new(BlockKind::Water, Rgb::new(60, 90, 190));

            let grass_depth = (1.5 + 2.0 * chaos).min(height - basement_height);
            let block = if (wposf.z as f32) < /*height*/height_min - grass_depth {
                let col = Lerp::lerp(
                    sub_surface_color,
                    stone_col.map(|e| e as f32 / 255.0),
                    (/*height*/height_min - grass_depth - wposf.z as f32) * 0.15,
                )
                .map(|e| (e * 255.0) as u8);

                // Underground
                if (wposf.z as f32) > alt - 32.0 * chaos {
                    Some(Block::new(BlockKind::Normal, col))
                } else {
                    Some(Block::new(BlockKind::Dense, col))
                }
            } else if /*(wposf.z as f32) < height*/(wposf.z as f32) <= height_min {
                let col = Lerp::lerp(
                    sub_surface_color,
                    surface_color,
                    (wposf.z as f32 - (/*height*/height_min - grass_depth))
                        .div(grass_depth)
                        .powf(0.5),
                );
                // Surface
                Some(Block::new(
                    BlockKind::Normal,
                    col.map(|e| (e * 255.0) as u8),
                ))
            } else if /*(wposf.z as f32) < height + 0.9*/
                (wposf.z as f32) <= height_min
                && temp < CONFIG.desert_temp
                // && (wposf.z as f32 > water_height + 3.0)
                && (wposf.z as f32 > water_height_min + 3.0)
                && marble > 0.6
                && marble_small > 0.55
                && (marble * 3173.7).fract() < 0.6
                && humidity > CONFIG.desert_hum
            {
                let treasures = [BlockKind::Chest, BlockKind::Velorite];

                let flowers = [
                    BlockKind::BlueFlower,
                    BlockKind::PinkFlower,
                    BlockKind::PurpleFlower,
                    BlockKind::RedFlower,
                    BlockKind::WhiteFlower,
                    BlockKind::YellowFlower,
                    BlockKind::Sunflower,
                    BlockKind::Mushroom,
                    BlockKind::LeafyPlant,
                    BlockKind::Blueberry,
                    BlockKind::LingonBerry,
                    BlockKind::Fern,
                ];
                let grasses = [
                    BlockKind::LongGrass,
                    BlockKind::MediumGrass,
                    BlockKind::ShortGrass,
                ];

                Some(Block::new(
                    if on_cliff && (height * 1271.0).fract() < 0.015 {
                        treasures[(height * 731.3) as usize % treasures.len()]
                    } else if (height * 1271.0).fract() < 0.1 {
                        flowers[(height * 0.2) as usize % flowers.len()]
                    } else {
                        grasses[(height * 103.3) as usize % grasses.len()]
                    },
                    Rgb::broadcast(0),
                ))
            } else if /*(wposf.z as f32) < height + 0.9*/
                (wposf.z as f32) <= height_min
                && wposf.z as f32 > water_height_min
                && temp > CONFIG.desert_temp
                && (marble * 4423.5).fract() < 0.0005
            {
                let large_cacti = [
                    BlockKind::LargeCactus,
                    BlockKind::MedFlatCactus,
                    BlockKind::Welwitch,
                ];

                let small_cacti = [
                    BlockKind::BarrelCactus,
                    BlockKind::RoundCactus,
                    BlockKind::ShortCactus,
                    BlockKind::ShortFlatCactus,
                    BlockKind::DeadBush,
                ];

                Some(Block::new(
                    if (height * 1271.0).fract() < 0.5 {
                        large_cacti[(height * 0.2) as usize % large_cacti.len()]
                    } else {
                        small_cacti[(height * 0.3) as usize % small_cacti.len()]
                    },
                    Rgb::broadcast(0),
                ))
            } else {
                None
            }
            .or_else(|| {
                // Rocks
                if (height + 2.5 - wposf.z as f32).div(7.5).abs().powf(2.0) < rock {
                    let field0 = RandomField::new(world.seed + 0);
                    let field1 = RandomField::new(world.seed + 1);
                    let field2 = RandomField::new(world.seed + 2);

                    Some(Block::new(
                        BlockKind::Normal,
                        stone_col
                            - Rgb::new(
                                field0.get(wpos) as u8 % 16,
                                field1.get(wpos) as u8 % 16,
                                field2.get(wpos) as u8 % 16,
                            ),
                    ))
                } else {
                    None
                }
            })
            .and_then(|block| {
                // Caves
                // Underground
                let cave = cave_xy.powf(2.0)
                    * (wposf.z as f32 - cave_alt)
                        .div(40.0)
                        .powf(4.0)
                        .neg()
                        .add(1.0)
                    > 0.9993;

                if cave /*&& wposf.z as f32 > water_height + 3.0 */&& wposf.z as f32 > water_height_min {
                    None
                } else {
                    Some(block)
                }
            })
            .or_else(|| {
                // Water
                if (wposf.z as f32) /*< water_height + 1.0*/<= water_height_min {
                    // Idea: the "voxel" we use for water can be partially full.  If it is
                    // partially full, it has a meaningful offset from the bottom of the
                    // usual voxel.
                    //
                    // We compute it as follows:
                    //
                    // - if wposf.z >= water_height, fill in up to wposf.z - water_height.
                    // - if wposf.z <
                    //
                    // water_packed is almost there, but we need to include a subvoxel height.
                    // TODO: We are only using 19 bits so far, figure out what to do with the
                    // other 5 bits.
                    // NOTE: Probably breaks a bit if water_height < 1.0, so hopefully this doesn't
                    // happen.
                    // let min_sub_height = (water_height as f64).sub(31.0 / 32.0).max(31.0 / 32.0);
                    // IDEA: If wpos.z is in [water_height - 31 / 32, water_height], use
                    // water_height - wpos.z.
                    /*let sub_height = if water_height as f64 - 1.0 >= wpos.z {

                    }wpos.z as f64
                    let sub_height = (water_height as f64).sub(31.0 / 32.0).max(wposf.z).max(0.0).fract(); */
                    let max_sub_height = (water_height as f64)/*.sub(1.0 / 16.0)*/.max(1.0);
                    // (wposf.z as f64).sub(31.0 / 32.0).max(1.0);
                    let sub_top = max_sub_height.max(wposf.z /*.min(max_sub_height)*/).fract();
                    // let sub_offset = min_sub_height.max(height/*.max(min_sub_height)*/).fract();
                    // sub_height is reinterpreted such that encoded 0-31 means from 1/32 to 1.
                    let sub_bottom = if wposf.z as f64 - 1.0 >= height as f64 {
                        0.0
                        /*1.0 - /*max_sub_height*/sub_height*/
                    } else {
                        wposf.z - height as f64
                    };
                    let encoded_sub_top = sub_top.mul(16.0) as u8;
                    let encoded_sub_bottom = sub_bottom.mul(16.0) as u8;
                    // If it returns an error, it probably means we gave two of the same
                    // thing--generally speaking, we interpret that as meaning this should be a
                    // full water block.
                    let encoded_sub_b_t =
                        pair_4_to_combination(encoded_sub_bottom, encoded_sub_top)
                        .unwrap_or(120) as u32;
                    let water_packed =
                        water_packed | /*(encoded_sub_height << 16) | (encoded_sub_offset << 20)*/
                                       (encoded_sub_b_t << 17);
                    // let water = Rgb::new(60, 90, 190);
                    let water = Block::new(
                        BlockKind::Water,
                        Rgb::new(
                            ((water_packed & 0xFF0000) >> 16) as u8,
                            ((water_packed & 0xFF00) >> 8) as u8,
                            (water_packed & 0xFF) as u8,
                        ),
                    );
                    Some(water)
                } else {
                    None
                }
            });

            (block, height)
        } else {
            (None, sample.alt)
        };

        // Structures (like towns)
        let block = chunk
            .structures
            .town
            .as_ref()
            .and_then(|town| TownGen.get((town, wpos, sample, height)))
            .or(block);

        let block = structures
            .iter()
            .find_map(|st| {
                let (st, st_sample) = st.as_ref()?;
                st.get(wpos, st_sample)
            })
            .or(block);

        Some(block.unwrap_or(Block::empty()))
    }
}

pub struct ZCache<'a> {
    wpos: Vec2<i32>,
    sample: ColumnSample<'a>,
    structures: [Option<(StructureInfo, ColumnSample<'a>)>; 9],
}

impl<'a> ZCache<'a> {
    pub fn get_z_limits(&self, block_gen: &mut BlockGen) -> (f32, f32, f32) {
        let cave_depth =
            if self.sample.cave_xy.abs() > 0.9 && self.sample.water_level <= self.sample.alt {
                (self.sample.alt - self.sample.cave_alt + 8.0).max(0.0)
            } else {
                0.0
            };

        let min = self.sample.alt - (self.sample.chaos.min(1.0) * 16.0 + cave_depth);
        let min = min - 4.0;

        let cliff = BlockGen::get_cliff_height(
            &mut block_gen.column_gen,
            &mut block_gen.column_cache,
            self.wpos.map(|e| e as f32),
            &self.sample.close_cliffs,
            self.sample.cliff_hill,
            32.0,
        );

        let rocks = if self.sample.rock > 0.0 { 12.0 } else { 0.0 };

        let warp = self.sample.chaos * 32.0;

        let (structure_min, structure_max) = self
            .structures
            .iter()
            .filter_map(|st| st.as_ref())
            .fold((0.0f32, 0.0f32), |(min, max), (st_info, _st_sample)| {
                let bounds = st_info.get_bounds();
                let st_area = Aabr {
                    min: Vec2::from(bounds.min),
                    max: Vec2::from(bounds.max),
                };

                if st_area.contains_point(self.wpos - st_info.pos) {
                    (min.min(bounds.min.z as f32), max.max(bounds.max.z as f32))
                } else {
                    (min, max)
                }
            });

        let ground_max = (self.sample.alt + warp + rocks).max(cliff) + 3.0;

        let min = min + structure_min;
        let max = (ground_max + structure_max).max(self.sample.water_level + 3.0);

        // Structures
        let (min, max) = self
            .sample
            .chunk
            .structures
            .town
            .as_ref()
            .map(|town| {
                let (town_min, town_max) = TownGen.get_z_limits(town, self.wpos, &self.sample);
                (town_min.min(min), town_max.max(max))
            })
            .unwrap_or((min, max));

        let structures_only_min_z = ground_max.max(self.sample.water_level + 3.0);

        (min, structures_only_min_z, max)
    }
}

#[derive(Copy, Clone)]
pub enum StructureMeta {
    Pyramid {
        height: i32,
    },
    Volume {
        units: (Vec2<i32>, Vec2<i32>),
        volume: &'static Structure,
    },
}

pub struct StructureInfo {
    pos: Vec3<i32>,
    seed: u32,
    meta: StructureMeta,
}

impl StructureInfo {
    fn get_bounds(&self) -> Aabb<i32> {
        match self.meta {
            StructureMeta::Pyramid { height } => {
                let base = 40;
                Aabb {
                    min: Vec3::new(-base - height, -base - height, -base),
                    max: Vec3::new(base + height, base + height, height),
                }
            },
            StructureMeta::Volume { units, volume } => {
                let bounds = volume.get_bounds();

                (Aabb {
                    min: Vec3::from(units.0 * bounds.min.x + units.1 * bounds.min.y)
                        + Vec3::unit_z() * bounds.min.z,
                    max: Vec3::from(units.0 * bounds.max.x + units.1 * bounds.max.y)
                        + Vec3::unit_z() * bounds.max.z,
                })
                .made_valid()
            },
        }
    }

    fn get(&self, wpos: Vec3<i32>, sample: &ColumnSample) -> Option<Block> {
        match self.meta {
            StructureMeta::Pyramid { height } => {
                if wpos.z - self.pos.z
                    < height
                        - Vec2::from(wpos - self.pos)
                            .map(|e: i32| (e.abs() / 2) * 2)
                            .reduce_max()
                {
                    Some(Block::new(BlockKind::Dense, Rgb::new(203, 170, 146)))
                } else {
                    None
                }
            },
            StructureMeta::Volume { units, volume } => {
                let rpos = wpos - self.pos;
                let block_pos = Vec3::unit_z() * rpos.z
                    + Vec3::from(units.0) * rpos.x
                    + Vec3::from(units.1) * rpos.y;

                volume
                    .get((block_pos * 128) / 128) // Scaling
                    .ok()
                    .and_then(|b| {
                        block_from_structure(
                            *b,
                            volume.default_kind(),
                            block_pos,
                            self.pos.into(),
                            self.seed,
                            sample,
                        )
                    })
            },
        }
    }
}

pub fn block_from_structure(
    sblock: StructureBlock,
    default_kind: BlockKind,
    pos: Vec3<i32>,
    structure_pos: Vec2<i32>,
    structure_seed: u32,
    _sample: &ColumnSample,
) -> Option<Block> {
    let field = RandomField::new(structure_seed + 0);

    let lerp = 0.5
        + ((field.get(Vec3::from(structure_pos)) % 256) as f32 / 256.0 - 0.5) * 0.85
        + ((field.get(Vec3::from(pos)) % 256) as f32 / 256.0 - 0.5) * 0.15;

    match sblock {
        StructureBlock::None => None,
        StructureBlock::TemperateLeaves => Some(Block::new(
            BlockKind::Leaves,
            Lerp::lerp(
                Rgb::new(0.0, 132.0, 94.0),
                Rgb::new(142.0, 181.0, 0.0),
                lerp,
            )
            .map(|e| e as u8),
        )),
        StructureBlock::PineLeaves => Some(Block::new(
            BlockKind::Leaves,
            Lerp::lerp(Rgb::new(0.0, 60.0, 50.0), Rgb::new(30.0, 100.0, 10.0), lerp)
                .map(|e| e as u8),
        )),
        StructureBlock::PalmLeavesInner => Some(Block::new(
            BlockKind::Leaves,
            Lerp::lerp(
                Rgb::new(61.0, 166.0, 43.0),
                Rgb::new(29.0, 130.0, 32.0),
                lerp,
            )
            .map(|e| e as u8),
        )),
        StructureBlock::PalmLeavesOuter => Some(Block::new(
            BlockKind::Leaves,
            Lerp::lerp(
                Rgb::new(62.0, 171.0, 38.0),
                Rgb::new(45.0, 171.0, 65.0),
                lerp,
            )
            .map(|e| e as u8),
        )),
        StructureBlock::Water => {
            // Encode water block RGB.
            Some(Block::new(BlockKind::Water, Rgb::new(100, 150, 255)))
        },
        StructureBlock::GreenSludge => Some(Block::new(BlockKind::Water, Rgb::new(30, 126, 23))),
        StructureBlock::Acacia => Some(Block::new(
            BlockKind::Normal,
            Lerp::lerp(
                Rgb::new(15.0, 126.0, 50.0),
                Rgb::new(30.0, 180.0, 10.0),
                lerp,
            )
            .map(|e| e as u8),
        )),
        StructureBlock::Fruit => Some(Block::new(BlockKind::Apple, Rgb::new(194, 30, 37))),
        StructureBlock::Chest => Some(if structure_seed % 10 < 7 {
            Block::empty()
        } else {
            Block::new(BlockKind::Chest, Rgb::new(0, 0, 0))
        }),
        StructureBlock::Liana => Some(Block::new(
            BlockKind::Liana,
            Lerp::lerp(
                Rgb::new(0.0, 125.0, 107.0),
                Rgb::new(0.0, 155.0, 129.0),
                lerp,
            )
            .map(|e| e as u8),
        )),
        StructureBlock::Mangrove => Some(Block::new(
            BlockKind::Normal,
            Lerp::lerp(Rgb::new(32.0, 56.0, 22.0), Rgb::new(57.0, 69.0, 27.0), lerp)
                .map(|e| e as u8),
        )),
        StructureBlock::Hollow => Some(Block::empty()),
        StructureBlock::Normal(color) => {
            Some(Block::new(default_kind, color)).filter(|block| !block.is_empty())
        },
    }
}
