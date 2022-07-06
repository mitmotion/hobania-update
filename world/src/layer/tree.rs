use crate::{
    all::*,
    block::block_from_structure,
    column::ColumnGen,
    site2::{self, PrimitiveTransform},
    util::{gen_cache::StructureGenCache, RandomPerm, Sampler, UnitChooser},
    Canvas, ColumnSample,
};
use common::{
    assets::AssetHandle,
    calendar::{Calendar, CalendarEvent},
    terrain::{
        structure::{Structure, StructureBlock, StructuresGroup},
        Block, BlockKind, SpriteKind,
    },
    vol::ReadVol,
};
use lazy_static::lazy_static;
use probability::{
    prelude::{source, Gaussian, Inverse},
};
use rand::{distributions::Uniform, prelude::*};
use std::{f32, ops::Range};
use vek::*;

lazy_static! {
    static ref OAK_STUMPS: AssetHandle<StructuresGroup> = Structure::load_group("trees.oak_stumps");
    static ref PALMS: AssetHandle<StructuresGroup> = Structure::load_group("trees.palms");
    static ref FRUIT_TREES: AssetHandle<StructuresGroup> =
        Structure::load_group("trees.fruit_trees");
    static ref BIRCHES: AssetHandle<StructuresGroup> = Structure::load_group("trees.birch");
    static ref SWAMP_TREES: AssetHandle<StructuresGroup> =
        Structure::load_group("trees.swamp_trees");
}

static MODEL_RAND: RandomPerm = RandomPerm::new(0xDB21C052);
static UNIT_CHOOSER: UnitChooser = UnitChooser::new(0x700F4EC7);
static QUIRKY_RAND: RandomPerm = RandomPerm::new(0xA634460F);

// Ensure that it's valid to place a tree here
pub fn tree_valid_at(col: &ColumnSample, seed: u32) -> bool {
    if col.alt < col.water_level
        || col.spawn_rate < 0.9
        || col.water_dist.map(|d| d < 8.0).unwrap_or(false)
        || col.path.map(|(d, _, _, _)| d < 12.0).unwrap_or(false)
    {
        return false;
    }

    if ((seed.wrapping_mul(13)) & 0xFF) as f32 / 256.0 > col.tree_density {
        return false;
    }

    true
}

// Inlined from
//
// https://docs.rs/probability/latest/src/probability/distribution/binomial.rs.html#393
//
// Approximate normal of binomial distribution.
fn approximate_by_normal(p: f64, np: f64, v: f64, u: f64) -> f64 {
    let w = Gaussian::new(0.0, 1.0).inverse(u);
    let w2 = w * w;
    let w3 = w2 * w;
    let w4 = w3 * w;
    let w5 = w4 * w;
    let w6 = w5 * w;
    let sd = v.sqrt();
    let sd_em1 = sd.recip();
    let sd_em2 = v.recip();
    let sd_em3 = sd_em1 * sd_em2;
    let sd_em4 = sd_em2 * sd_em2;
    let p2 = p * p;
    let p3 = p2 * p;
    let p4 = p2 * p2;

    np +
    sd * w +
    (p + 1.0) / 3.0 -
    (2.0 * p - 1.0) * w2 / 6.0 +
    sd_em1 * w3 * (2.0 * p2 - 2.0 * p - 1.0) / 72.0 -
    w * (7.0 * p2 - 7.0 * p + 1.0) / 36.0 +
    sd_em2 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (3.0 * w4 + 7.0 * w2 - 16.0 / 1620.0) +
    sd_em3 * (
        w5 * (4.0 * p4 - 8.0 * p3 - 48.0 * p2 + 52.0 * p - 23.0) / 17280.0 +
        w3 * (256.0 * p4 - 512.0 * p3 - 147.0 * p2 + 403.0 * p - 137.0) / 38880.0 -
        w * (433.0 * p4 - 866.0 * p3 - 921.0 * p2 + 1354.0 * p - 671.0) / 38880.0
    ) +
    sd_em4 * (
        w6 * (2.0 * p - 1.0) * (p2 - p + 1.0) * (p2 - p + 19.0) / 34020.0 +
        w4 * (2.0 * p - 1.0) * (9.0 * p4 - 18.0 * p3 - 35.0 * p2 + 44.0 * p - 25.0) / 15120.0 +
        w2 * (2.0 * p - 1.0) * (
                923.0 * p4 - 1846.0 * p3 + 5271.0 * p2 - 4348.0 * p + 5189.0
        ) / 408240.0 -
        4.0 * (2.0 * p - 1.0) * (p + 1.0) * (p - 2.0) * (23.0 * p2 - 23.0 * p + 2.0) / 25515.0
    )
    // + O(v.powf(-2.5)), with probabilty of 1 - 2e-9
}

pub fn apply_trees_to(
    canvas: &mut Canvas,
    dynamic_rng: &mut impl Rng,
    /* calendar: Option<&Calendar>, */
) {
    // TODO: Get rid of this
    #[allow(clippy::large_enum_variant)]
    enum TreeModel {
        Structure(&'static Structure),
        Procedural(ProceduralTree, StructureBlock),
    }

    struct Tree {
        pos: Vec3<i32>,
        model: TreeModel,
        seed: u32,
        units: (Vec2<i32>, Vec2<i32>),
        lights: bool,
    }

    let info = canvas.info();
    let calendar = info.calendar();
    /* let mut tree_cache = StructureGenCache::new(info.chunks().gen_ctx.structure_gen.clone()); */

    // Get all the trees in range.
    let render_area = Aabr {
        min: info.wpos(),
        max: info.wpos() + Vec2::from(info.area().size().map(|e| e as i32)),
    };

    let mut arena = bumpalo::Bump::new();

    /*canvas.foreach_col(|canvas, wpos2d, col| {*/
    info.chunks()
        .get_area_trees(render_area.min + 15, render_area.max + 16)
        .filter_map(|attr| {
            info.col_or_gen(attr.pos)
                .filter(|col| tree_valid_at(col, attr.seed))
                .zip(Some(attr))
        })
        .for_each(|(col, attr)| {
            let seed = attr.seed;
        /* let trees = tree_cache.get(wpos2d, |wpos, seed| {
            let forest_kind = *info
                .chunks()
                .make_forest_lottery(wpos)
                .choose_seeded(seed)
                .as_ref()?;

            let col = ColumnGen::new(info.chunks()).get((wpos, info.index(), calendar))?;

            if !tree_valid_at(&col, seed) {
                return None;
            } */

            let scale = 1.0;
            let inhabited = false;
            let tree = /*Some(*/Tree {
                pos: Vec3::new(/*wpos.x*/attr.pos.x, /*wpos.y*/attr.pos.y, col.alt as i32),
                model: 'model: {
                    let models: AssetHandle<_> = match attr.forest_kind {
                        ForestKind::Oak if QUIRKY_RAND.chance(seed + 1, 1.0 / 16.0) => *OAK_STUMPS,
                        ForestKind::Oak if QUIRKY_RAND.chance(seed + 2, 1.0 / 20.0) => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::apple(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::TemperateLeaves,
                            );
                        },
                        ForestKind::Palm => *PALMS,
                        ForestKind::Acacia => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::acacia(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::Acacia,
                            );
                        },
                        ForestKind::Baobab => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::baobab(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::Baobab,
                            );
                        },
                        ForestKind::Oak => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::oak(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::TemperateLeaves,
                            );
                        },
                        ForestKind::Chestnut => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::chestnut(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::Chestnut,
                            );
                        },
                        ForestKind::Pine => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::pine(&mut RandomPerm::new(seed), scale, calendar),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::PineLeaves,
                            );
                        },
                        ForestKind::Cedar => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::cedar(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::PineLeaves,
                            );
                        },
                        ForestKind::Birch => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::birch(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::TemperateLeaves,
                            );
                        },
                        ForestKind::Frostpine => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::frostpine(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::FrostpineLeaves,
                            );
                        },

                        ForestKind::Mangrove => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::jungle(&mut RandomPerm::new(seed), scale),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::Mangrove,
                            );
                        },
                        ForestKind::Swamp => *SWAMP_TREES,
                        ForestKind::Giant => {
                            break 'model TreeModel::Procedural(
                                ProceduralTree::generate(
                                    TreeConfig::giant(&mut RandomPerm::new(seed), scale, inhabited),
                                    &mut RandomPerm::new(seed),
                                ),
                                StructureBlock::TemperateLeaves,
                            );
                        },
                    };

                    let models = models./*read*/get();
                    TreeModel::Structure(
                        &models
                            [(MODEL_RAND.get(seed.wrapping_mul(17)) / 13) as usize % models.len()]
                        /*.clone(),*/
                    )
                },
                seed,
                units: UNIT_CHOOSER.get(seed),
                lights: inhabited,
            }/*)*/;
        /* }); */

            /* criterion::black_box(tree); */
        /*for tree in trees {*/
            /* 
    arena: &'b bumpalo::Bump,
    canvas_info: CanvasInfo<'c>,
    render_area: Aabr<i32>,
    filler: /*impl FnOnce(&'b mut Canvas<'a>) -> &'b mut F*/&'b mut F,
    render: Render, */
            let mut wpos = tree.pos;
            let (bounds, hanging_sprites) = match &tree.model {
                TreeModel::Structure(s) => {
                    site2::render_collect(
                        &arena,
                        info,
                        render_area,
                        canvas,
                        |painter, filler| {
                            painter
                                .prefab(s)
                                .translate(/*totem_pos*/tree.pos)
                                .fill(filler.prefab(s, tree.pos, tree.seed), filler);
                        },
                    );
                    (s.get_bounds(), [(0.0004, SpriteKind::Beehive)].as_ref())
                },
                &TreeModel::Procedural(ref t, leaf_block) => {
                    let bounds = t.get_bounds().map(|e| e as i32);
                    let trunk_block = t.config.trunk_block;
                    let leaf_vertical_scale = t.config.leaf_vertical_scale.recip();
                    let branch_child_radius_lerp = t.config.branch_child_radius_lerp;

                    // NOTE: Technically block_from_structure isn't correct here, because it could
                    // lerp with position; in practice, it almost never does, and most of the other
                    // expensive parameters are unused.
                    /* let trunk_block = if let Some(block) = block_from_structure(
                        info.index(),
                        trunk_block,
                        wpos,
                        tree.pos.xy(),
                        tree.seed,
                        &col,
                        Block::air,
                        calendar,
                    ) {
                        block
                    } else {
                        return;
                    };
                    let leaf_block = if let Some(block) = block_from_structure(
                        info.index(),
                        leaf_block,
                        wpos,
                        tree.pos.xy(),
                        tree.seed,
                        &col,
                        Block::air,
                        calendar,
                    ) {
                        block
                    } else {
                        return;
                    }; */

                    site2::render_collect(
                        &arena,
                        info,
                        render_area,
                        canvas,
                        |painter, filler| {
                            let trunk_block = filler.block_from_structure(
                                trunk_block,
                                tree.pos.xy(),
                                tree.seed,
                                &col,
                            );
                            let leaf_block = filler.block_from_structure(
                                leaf_block,
                                tree.pos.xy(),
                                tree.seed,
                                &col,
                            );
                            t.walk(|branch, parent| {
                                let aabr = Aabr {
                                    min: wpos.xy() + branch.get_aabb().min.xy().as_(),
                                    max: wpos.xy() + branch.get_aabb().max.xy().as_(),
                                };
                                if aabr.collides_with_aabr(filler.render_aabr().as_()) {
                                    let start =
                                        wpos.as_::<f32>() + branch.get_line().start/*.as_()*//* - 0.5*/;
                                    let end =
                                        wpos.as_::<f32>() + branch.get_line().end/*.as_()*//* - 0.5*/;
                                    let wood_radius = branch.get_wood_radius();
                                    let leaf_radius = branch.get_leaf_radius();
                                    let parent_wood_radius = if branch_child_radius_lerp {
                                        parent.get_wood_radius()
                                    } else {
                                        wood_radius
                                    };
                                    let leaf_eats_wood = leaf_radius > wood_radius;
                                    let leaf_eats_parent_wood = leaf_radius > parent_wood_radius;
                                    if !leaf_eats_wood || !leaf_eats_parent_wood {
                                        // Render the trunk, since it's not swallowed by its leaf.
                                        painter
                                            .line_two_radius(
                                                start,
                                                end,
                                                parent_wood_radius,
                                                wood_radius,
                                                1.0,
                                            )
                                            .fill(/*filler.block(trunk_block)*/trunk_block, filler);
                                    }
                                    if leaf_eats_wood || leaf_eats_parent_wood {
                                        // Render the leaf, since it's not *completely* swallowed
                                        // by the trunk.
                                        painter
                                            .line_two_radius(
                                                start,
                                                end,
                                                leaf_radius,
                                                leaf_radius,
                                                leaf_vertical_scale,
                                            )
                                            .fill(/*filler.block(leaf_block)*/leaf_block, filler);
                                    }
                                    true
                                } else {
                                    false
                                }
                            });
                            // Draw the roots.
                            t.roots.iter().for_each(|root| {
                                painter
                                    .line(
                                        wpos/*.as_::<f32>()*/ + root.line.start.as_()/* - 0.5*/,
                                        wpos/*.as_::<f32>()*/ + root.line.end.as_()/* - 0.5*/,
                                        root.radius,
                                    )
                                    .fill(/*filler.block(leaf_block)*/trunk_block, filler);
                            });
                        },
                    );
                    (bounds, t.config.hanging_sprites)
                },
            };
            arena.reset();

            // Subtract 1 from our max z, since we know there can be no hanging sprites at the top
            // of the tree.
            wpos.z -= 1;
            let mut aabb = Aabb {
                min: tree.pos + bounds.min.as_(),
                max: wpos + bounds.max.as_(),
            };
            if !aabb.is_valid() {
                // Zero-size tree somehow?
                return;
            }
            let size = aabb.size();
            // NOTE: The chunk volume should always fit into 24 bits, so if it doesn't we can
            // exit early (as sprites not rendering will be the least of our worries).
            let volume: u32 = if let Ok(volume) = size.product().abs().try_into() {
                volume
            } else {
                tracing::warn!("Tree bounds were larger than our maximum chunk size...");
                return;
            };
            if volume == 0 {
                // One-layer tree, so we can't draw any sprites...
                return;
            }
            // NOTE: volume.x * volume.y fits in 10 bits, so it fits in a u32, hence can't overflow.
            let volume_x = size.w.abs() as u32;
            // NOTE: volume.z fits into 14 bits, so it fits into a u32, and it times volume_x fits
            // into 19 bits, so this can't overflow.
            let volume_xy = volume_x * size.h.abs() as u32;
            aabb.intersect(Aabb {
                min: render_area.min.with_z(aabb.min.z),
                max: render_area.max.with_z(aabb.max.z),
            });

            struct Source<T: ?Sized>(T);
            impl<'a, T: rand::Rng + ?Sized> source::Source for Source<T> {
                fn read_u64(&mut self) -> u64 {
                    self.0.next_u64()
                }
            }

            let mut source = Source(&mut *dynamic_rng);

            // Sprite rendering is performed by constructing random sequences of sample blocks,
            // then testing whether they are valid for placement in the tree.  This should
            // hopefully be a lot more efficient than actually iterating through the tree and
            // randomly sampling each time, since the sprite probability is tiny, assuming
            // that we can keep sampling from the distribution cheap enough (to be determined!) and
            // that the difference between overall volume and hangable surface area doesn't drop
            // too low (very likely).
            hanging_sprites.iter().for_each(|(chance, sprite)| {
                // Let B(volume, chance) be the binomial distribution over independent
                // tests of [volume] blocks, each of which has probability [chance].  We sample
                // uniformly at random from this distribution in order to get an expected number of
                // *successful* trials, without actually having to iterate over every block.
                // Hopefully (assuming sampling the distribution isn't too slow), this should be a
                // lot faster than *actually* iterating over that many blocks.
                //
                // NOTE: We assume u32 fits into usize, so this is legal; we should statically
                // assert this somewhere (we might already do so).
                /* println!("Generating samples: n={} p={}", volume, chance); */
                let num_samples = {
                    use source::Source;

                    let n = volume;
                    let p = (*chance).into();
                    let q = 1.0 - p;
                    let np = volume as f64 * p;
                    let npq = np * q;

                    let u = source.read_f64();
                     approximate_by_normal(p, np, npq, u).floor() as usize
                };
                // let num_samples = Binomial::new(volume as usize, (*chance).into()).sample(&mut source);
                // Now, we choose [num_samples] blocks uniformly at random from the AABB, and test
                // them to see if they're candidates for our sprite; if so, we place it there.
                /* println!("Generated samples: {}", num_samples); */
                (&mut source.0)
                    // NOTE: Cannot panic, as we checked earlier that volume > 0.
                    .sample_iter(Uniform::new(0, volume))
                    .take(num_samples)
                    // NOTE: We gamble here that division and modulus for the tree volume will
                    // still be faster than sampling three times from the RNG.
                    .map(|ipos| {
                        let z = ipos.div_euclid(volume_xy);
                        let ipos = ipos - z * volume_xy;
                        let y = ipos.div_euclid(volume_x);
                        let x = ipos - y * volume_x;
                        tree.pos + Vec3::new(x, y, z).as_()
                    })
                    .for_each(|mut wpos| {
                        let get_block = |wpos| {
                            let mut model_pos = /*Vec3::from(*/
                                (wpos - tree.pos)/*
                                    .xy()
                                    .map2(Vec2::new(tree.units.0, tree.units.1), |rpos, unit| {
                                        unit * rpos
                                    })
                                    .sum(),
                            ) + Vec3::unit_z() * (wpos.z - tree.pos.z)*/;

                            block_from_structure(
                                info.index(),
                                match &tree.model {
                                    TreeModel::Structure(s) => s.get(model_pos).ok().copied()?,
                                    TreeModel::Procedural(t, leaf_block) =>
                                        match t.is_branch_or_leaves_at(model_pos.map(|e| e as f32 + 0.5)) {
                                            (_, _, true, _) => {
                                                StructureBlock::Filled(BlockKind::Wood, Rgb::new(150, 98, 41))
                                            },
                                            (_, _, _, true) => StructureBlock::None,
                                            (true, _, _, _) => t.config.trunk_block,
                                            (_, true, _, _) => *leaf_block,
                                            _ => StructureBlock::None,
                                        },
                                },
                                wpos,
                                tree.pos.xy(),
                                tree.seed,
                                &col,
                                Block::air,
                                calendar,
                            )
                        };
                        // Hanging sprites can be placed in locations that are currently empty, and
                        // which have a tree block above them.  NOTE: The old code overwrote
                        // existing terrain as well, but we opt not to do this.
                        canvas.map(wpos, |block| {
                            if !block.is_filled() {
                                wpos.z += 1;
                                if get_block(wpos).map_or(false, |block| block.is_filled()) {
                                    /* println!("Success: {:?} = {:?}", wpos, sprite); */
                                    block.with_sprite(*sprite)
                                } else {
                                    block
                                }
                            } else {
                                block
                            }
                        })
                    });
            });

            /* let bounds = match &tree.model {
                TreeModel::Structure(s) => s.get_bounds(),
                TreeModel::Procedural(t, _) => t.get_bounds().map(|e| e as i32),
            };

            let rpos2d = (wpos2d - tree.pos.xy())
                .map2(Vec2::new(tree.units.0, tree.units.1), |p, unit| unit * p)
                .sum();
            if !Aabr::from(bounds).contains_point(rpos2d) {
                // Skip this column
                continue;
            }
            let hanging_sprites = match &tree.model {
                TreeModel::Structure(_) => [(0.0004, SpriteKind::Beehive)].as_ref(),
                TreeModel::Procedural(t, _) => t.config.hanging_sprites,
            };

            let mut is_top = true;
            let mut is_leaf_top = true;
            let mut last_block = Block::empty();
            for z in (bounds.min.z..bounds.max.z).rev() {
                let wpos = Vec3::new(wpos2d.x, wpos2d.y, tree.pos.z + z);
                let model_pos = Vec3::from(
                    (wpos - tree.pos)
                        .xy()
                        .map2(Vec2::new(tree.units.0, tree.units.1), |rpos, unit| {
                            unit * rpos
                        })
                        .sum(),
                ) + Vec3::unit_z() * (wpos.z - tree.pos.z);
                block_from_structure(
                    info.index(),
                    if let Some(block) = match &tree.model {
                        TreeModel::Structure(s) => s.get(model_pos).ok().copied(),
                        TreeModel::Procedural(t, leaf_block) => Some(
                            match t.is_branch_or_leaves_at(model_pos.map(|e| e as f32 + 0.5)) {
                                (_, _, true, _) => {
                                    StructureBlock::Filled(BlockKind::Wood, Rgb::new(150, 98, 41))
                                },
                                (_, _, _, true) => StructureBlock::None,
                                (true, _, _, _) => t.config.trunk_block,
                                (_, true, _, _) => *leaf_block,
                                _ => StructureBlock::None,
                            },
                        ),
                    } {
                        block
                    } else {
                        break;
                    },
                    wpos,
                    tree.pos.xy(),
                    tree.seed,
                    col,
                    Block::air,
                    calendar,
                )
                .map(|block| {
                    // Add lights to the tree
                    if tree.lights
                        && last_block.is_air()
                        && block.kind() == BlockKind::Wood
                        && dynamic_rng.gen_range(0..256) == 0
                    {
                        canvas.set(wpos + Vec3::unit_z(), Block::air(SpriteKind::Lantern));
                        // Add a snow covering to the block above under certain
                        // circumstances
                    } else if col.snow_cover
                        && ((block.kind() == BlockKind::Leaves && is_leaf_top)
                            || (is_top && block.is_filled()))
                    {
                        canvas.set(
                            wpos + Vec3::unit_z(),
                            Block::new(BlockKind::Snow, Rgb::new(210, 210, 255)),
                        );
                    }
                    canvas.set(wpos, block);
                    is_leaf_top = false;
                    is_top = false;
                    last_block = block;
                })
                .unwrap_or_else(|| {
                    // Hanging sprites
                    if last_block.is_filled() {
                        for (chance, sprite) in hanging_sprites {
                            if dynamic_rng.gen_bool(*chance as f64) {
                                canvas.map(wpos, |block| block.with_sprite(*sprite));
                            }
                        }
                    }

                    is_leaf_top = true;
                    last_block = Block::empty();
                });
            } */
        /*}*/
        })
    /*})*/;
}

/// A type that specifies the generation properties of a tree.
#[derive(Clone)]
pub struct TreeConfig {
    /// Length of trunk, also scales other branches.
    pub trunk_len: f32,
    /// Radius of trunk, also scales other branches.
    pub trunk_radius: f32,
    /// The scale that child branch lengths should be compared to their parents.
    pub branch_child_len: f32,
    /// The scale that child branch radii should be compared to their parents.
    pub branch_child_radius: f32,
    /// Whether the child of a branch has its radius lerped to its parent.
    pub branch_child_radius_lerp: bool,
    /// The range of radii that leaf-emitting branches might have.
    pub leaf_radius: Range<f32>,
    /// An additional leaf radius that may be scaled with proportion along the
    /// parent and `branch_len_bias`.
    pub leaf_radius_scaled: f32,
    /// 0 - 1 (0 = chaotic, 1 = straight).
    pub straightness: f32,
    /// Maximum number of branch layers (not including trunk).
    pub max_depth: usize,
    /// The number of branches that form from each branch.
    pub splits: Range<f32>,
    /// The range of proportions along a branch at which a split into another
    /// branch might occur. This value is clamped between 0 and 1, but a
    /// wider range may bias the results towards branch ends.
    pub split_range: Range<f32>,
    /// The bias applied to the length of branches based on the proportion along
    /// their parent that they eminate from. -1.0 = negative bias (branches
    /// at ends are longer, branches at the start are shorter) 0.0 = no bias
    /// (branches do not change their length with regard to parent branch
    /// proportion) 1.0 = positive bias (branches at ends are shorter,
    /// branches at the start are longer)
    pub branch_len_bias: f32,
    /// The scale of leaves in the vertical plane. Less than 1.0 implies a
    /// flattening of the leaves.
    pub leaf_vertical_scale: f32,
    /// How evenly spaced (vs random) sub-branches are along their parent.
    pub proportionality: f32,
    /// Whether the tree is inhabited (adds various features and effects)
    pub inhabited: bool,
    pub hanging_sprites: &'static [(f32, SpriteKind)],
    /// The colour of branches and the trunk.
    pub trunk_block: StructureBlock,
}

impl TreeConfig {
    pub fn oak(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>().powi(2) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 9.0 * scale,
            trunk_radius: 2.0 * scale,
            branch_child_len: 0.9,
            branch_child_radius: 0.75,
            branch_child_radius_lerp: true,
            leaf_radius: 2.5 * log_scale..3.25 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.45,
            max_depth: 4,
            splits: 2.25..3.25,
            split_range: 0.75..1.5,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 1.0,
            proportionality: 0.0,
            inhabited: false,
            hanging_sprites: &[(0.0002, SpriteKind::Apple), (0.00007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(90, 45, 15)),
        }
    }

    pub fn apple(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>().powi(2) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 3.0 * scale,
            trunk_radius: 1.5 * scale,
            branch_child_len: 0.9,
            branch_child_radius: 0.9,
            branch_child_radius_lerp: true,
            leaf_radius: 2.0 * log_scale..3.0 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.4,
            max_depth: 6,
            splits: 1.0..3.0,
            split_range: 0.5..2.0,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.7,
            proportionality: 0.0,
            inhabited: false,
            hanging_sprites: &[(0.03, SpriteKind::Apple), (0.007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(90, 45, 15)),
        }
    }

    pub fn frostpine(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>().powi(2) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 36.0 * scale,
            trunk_radius: 2.3 * scale,
            branch_child_len: 0.25 / scale,
            branch_child_radius: 0.0,
            branch_child_radius_lerp: false,
            leaf_radius: 1.3..2.2,
            leaf_radius_scaled: 0.4 * log_scale,
            straightness: 0.3,
            max_depth: 1,
            splits: 34.0 * scale..35.0 * scale,
            split_range: 0.1..1.2,
            branch_len_bias: 0.75,
            leaf_vertical_scale: 0.6,
            proportionality: 1.0,
            inhabited: false,
            hanging_sprites: &[(0.0001, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(79, 102, 105)),
        }
    }

    pub fn jungle(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>() * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 44.0 * scale,
            trunk_radius: 2.25 * scale,
            branch_child_len: 0.35,
            branch_child_radius: 0.5,
            branch_child_radius_lerp: true,
            leaf_radius: 10.0 * log_scale..11.5 * log_scale,
            leaf_radius_scaled: -8.0 * log_scale,
            straightness: 0.2,
            max_depth: 2,
            splits: 7.5..8.5,
            split_range: 0.2..1.25,
            branch_len_bias: 0.5,
            leaf_vertical_scale: 0.35,
            proportionality: 0.8,
            inhabited: false,
            hanging_sprites: &[(0.00007, SpriteKind::Beehive), (0.015, SpriteKind::Liana)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(118, 67, 42)),
        }
    }

    pub fn baobab(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.5 + rng.gen::<f32>().powi(4) * 1.0);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 24.0 * scale,
            trunk_radius: 7.0 * scale,
            branch_child_len: 0.55,
            branch_child_radius: 0.3,
            branch_child_radius_lerp: true,
            leaf_radius: 2.5 * log_scale..3.0 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.5,
            max_depth: 4,
            splits: 3.0..3.5,
            split_range: 0.95..1.0,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.2,
            proportionality: 1.0,
            inhabited: false,
            hanging_sprites: &[(0.00007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(125, 60, 6)),
        }
    }

    pub fn cedar(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>().powi(2) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 9.0 * scale,
            trunk_radius: 2.0 * scale,
            branch_child_len: 0.9,
            branch_child_radius: 0.75,
            branch_child_radius_lerp: true,
            leaf_radius: 4.0 * log_scale..5.0 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.4,
            max_depth: 4,
            splits: 1.75..2.0,
            split_range: 0.75..1.5,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.4,
            proportionality: 0.0,
            inhabited: false,
            hanging_sprites: &[(0.00007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(110, 68, 65)),
        }
    }

    pub fn birch(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.8 + rng.gen::<f32>().powi(2) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 24.0 * scale,
            trunk_radius: 1.2 * scale,
            branch_child_len: 0.4,
            branch_child_radius: 0.75,
            branch_child_radius_lerp: true,
            leaf_radius: 4.0 * log_scale..5.0 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.6,
            max_depth: 4,
            splits: 1.75..2.5,
            split_range: 0.6..1.2,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.5,
            proportionality: 0.0,
            inhabited: false,
            hanging_sprites: &[(0.00007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::BirchWood,
        }
    }

    pub fn acacia(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.9 + rng.gen::<f32>().powi(4) * 0.75);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 7.5 * scale,
            trunk_radius: 1.5 * scale,
            branch_child_len: 0.75,
            branch_child_radius: 0.75,
            branch_child_radius_lerp: true,
            leaf_radius: 4.5 * log_scale..5.5 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.4,
            max_depth: 5,
            splits: 1.75..2.25,
            split_range: 1.0..1.25,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.2,
            proportionality: 1.0,
            inhabited: false,
            hanging_sprites: &[(0.00005, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(150, 95, 65)),
        }
    }

    pub fn chestnut(rng: &mut impl Rng, scale: f32) -> Self {
        let scale = scale * (0.85 + rng.gen::<f32>().powi(4) * 0.3);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 13.0 * scale,
            trunk_radius: 1.65 * scale,
            branch_child_len: 0.75,
            branch_child_radius: 0.6,
            branch_child_radius_lerp: true,
            leaf_radius: 1.5 * log_scale..2.0 * log_scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.3,
            max_depth: 5,
            splits: 3.5..4.25,
            split_range: 0.5..1.25,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.65,
            proportionality: 0.5,
            inhabited: false,
            hanging_sprites: &[(0.00007, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(110, 42, 28)),
        }
    }

    pub fn pine(rng: &mut impl Rng, scale: f32, calendar: Option<&Calendar>) -> Self {
        let scale = scale * (1.0 + rng.gen::<f32>().powi(4) * 0.5);
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 32.0 * scale,
            trunk_radius: 1.25 * scale,
            branch_child_len: 0.3 / scale,
            branch_child_radius: 0.0,
            branch_child_radius_lerp: false,
            leaf_radius: 1.9..2.1,
            leaf_radius_scaled: 1.5 * log_scale,
            straightness: 0.0,
            max_depth: 1,
            splits: 34.0 * scale..35.0 * scale,
            split_range: 0.165..1.2,
            branch_len_bias: 0.75,
            leaf_vertical_scale: 0.3,
            proportionality: 1.0,
            inhabited: false,
            hanging_sprites: if calendar.map_or(false, |c| c.is_event(CalendarEvent::Christmas)) {
                &[(0.0001, SpriteKind::Beehive), (0.01, SpriteKind::Orb)]
            } else {
                &[(0.0001, SpriteKind::Beehive)]
            },
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(90, 35, 15)),
        }
    }

    pub fn giant(_rng: &mut impl Rng, scale: f32, inhabited: bool) -> Self {
        let log_scale = 1.0 + scale.log2().max(0.0);

        Self {
            trunk_len: 11.0 * scale,
            trunk_radius: 6.0 * scale,
            branch_child_len: 0.9,
            branch_child_radius: 0.75,
            branch_child_radius_lerp: true,
            leaf_radius: 2.5 * scale..3.75 * scale,
            leaf_radius_scaled: 0.0,
            straightness: 0.36,
            max_depth: (7.0 + log_scale) as usize,
            splits: 1.5..2.5,
            split_range: 1.0..1.1,
            branch_len_bias: 0.0,
            leaf_vertical_scale: 0.6,
            proportionality: 0.0,
            inhabited,
            hanging_sprites: &[(0.00025, SpriteKind::Apple), (0.00025, SpriteKind::Beehive)],
            trunk_block: StructureBlock::Filled(BlockKind::Wood, Rgb::new(110, 68, 22)),
        }
    }
}

// TODO: Rename this to `Tree` when the name conflict is gone
pub struct ProceduralTree {
    branches: Vec<Branch>,
    trunk_idx: usize,
    pub(crate) config: TreeConfig,
    roots: Vec<Root>,
    root_aabb: Aabb<f32>,
}

impl ProceduralTree {
    /// Generate a new tree using the given configuration and seed.
    pub fn generate(config: TreeConfig, rng: &mut impl Rng) -> Self {
        let mut this = Self {
            branches: Vec::new(),
            trunk_idx: 0, // Gets replaced later
            config: config.clone(),
            roots: Vec::new(),
            root_aabb: Aabb::new_empty(Vec3::zero()),
        };

        // Make the roots visible a little
        let trunk_origin = Vec3::unit_z() * (config.trunk_radius * 0.25 + 3.0);

        // Add the tree trunk (and sub-branches) recursively
        let (trunk_idx, _) = this.add_branch(
            &config,
            // Our trunk starts at the origin...
            trunk_origin,
            // ...and has a roughly upward direction
            Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 10.0).normalized(),
            config.trunk_len,
            config.trunk_radius,
            0,
            None,
            1.0,
            rng,
        );
        this.trunk_idx = trunk_idx;

        // Add roots
        let mut root_aabb = Aabb::new_empty(Vec3::zero());
        for _ in 0..4 {
            let dir =
                Vec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), -1.0).normalized();
            let len = config.trunk_len * 0.75;
            let radius = config.trunk_radius;
            let mut aabb = Aabb {
                min: trunk_origin,
                max: trunk_origin + dir * len,
            }
            .made_valid();
            aabb.min -= radius;
            aabb.max += radius;

            root_aabb.expand_to_contain(aabb);

            this.roots.push(Root {
                line: LineSegment3 {
                    start: trunk_origin,
                    end: trunk_origin + dir * 10.0,
                },
                radius,
            });
        }

        this.root_aabb = root_aabb;

        this
    }

    // Recursively add a branch (with sub-branches) to the tree's branch graph,
    // returning the index and AABB of the branch. This AABB gets propagated
    // down to the parent and is used later during sampling to cull the branches to
    // be sampled.
    fn add_branch(
        &mut self,
        config: &TreeConfig,
        start: Vec3<f32>,
        dir: Vec3<f32>,
        branch_len: f32,
        branch_radius: f32,
        depth: usize,
        sibling_idx: Option<usize>,
        proportion: f32,
        rng: &mut impl Rng,
    ) -> (usize, Aabb<f32>) {
        let end = start + dir * branch_len;
        let line = LineSegment3 { start, end };
        let wood_radius = branch_radius;
        let leaf_radius = if depth == config.max_depth {
            rng.gen_range(config.leaf_radius.clone())
                + config.leaf_radius_scaled
                    * Lerp::lerp(1.0, 1.0 - proportion, config.branch_len_bias.abs())
        } else {
            0.0
        };

        let has_stairs = config.inhabited
            && depth < config.max_depth
            && branch_radius > 6.5
            && start.xy().distance(end.xy()) < (start.z - end.z).abs() * 1.5;
        let bark_radius = if has_stairs { 5.0 } else { 0.0 } + wood_radius * 0.25;

        // The AABB that covers this branch, along with wood and leaves that eminate
        // from it
        let mut aabb = Aabb {
            min: Vec3::partial_min(start, end) - (wood_radius + bark_radius).max(leaf_radius),
            max: Vec3::partial_max(start, end) + (wood_radius + bark_radius).max(leaf_radius),
        };

        let mut child_idx = None;
        // Don't add child branches if we're already enough layers into the tree
        if depth < config.max_depth {
            let x_axis = dir
                .cross(Vec3::<f32>::zero().map(|_| rng.gen_range(-1.0..1.0)))
                .normalized();
            let y_axis = dir.cross(x_axis).normalized();
            let screw_shift = rng.gen_range(0.0..f32::consts::TAU);

            let splits = rng.gen_range(config.splits.clone()).round() as usize;
            for i in 0..splits {
                let proportion = i as f32 / (splits - 1) as f32;
                let dist = Lerp::lerp(rng.gen_range(0.0..1.0), proportion, config.proportionality);

                const PHI: f32 = 0.618;
                const RAD_PER_BRANCH: f32 = f32::consts::TAU * PHI;
                let screw = (screw_shift + i as f32 * RAD_PER_BRANCH).sin() * x_axis
                    + (screw_shift + i as f32 * RAD_PER_BRANCH).cos() * y_axis;

                // Choose a point close to the branch to act as the target direction for the
                // branch to grow in let split_factor =
                // rng.gen_range(config.split_range.start, config.split_range.end).clamped(0.0,
                // 1.0);
                let split_factor =
                    Lerp::lerp(config.split_range.start, config.split_range.end, dist);
                let tgt = Lerp::lerp_unclamped(start, end, split_factor)
                    + Lerp::lerp(
                        Vec3::<f32>::zero().map(|_| rng.gen_range(-1.0..1.0)),
                        screw,
                        config.proportionality,
                    );
                // Start the branch at the closest point to the target
                let branch_start = line.projected_point(tgt);
                // Now, interpolate between the target direction and the parent branch's
                // direction to find a direction
                let branch_dir =
                    Lerp::lerp_unclamped(tgt - branch_start, dir, config.straightness).normalized();

                let (branch_idx, branch_aabb) = self.add_branch(
                    config,
                    branch_start,
                    branch_dir,
                    branch_len
                        * config.branch_child_len
                        * (1.0
                            - (split_factor - 0.5)
                                * 2.0
                                * config.branch_len_bias.clamped(-1.0, 1.0)),
                    branch_radius * config.branch_child_radius,
                    depth + 1,
                    child_idx,
                    proportion,
                    rng,
                );
                child_idx = Some(branch_idx);
                // Parent branches AABBs include the AABBs of child branches to allow for
                // culling during sampling
                aabb.expand_to_contain(branch_aabb);
            }
        }

        let idx = self.branches.len(); // Compute the index that this branch is going to have
        self.branches.push(Branch {
            line,
            wood_radius,
            leaf_radius,
            leaf_vertical_scale: config.leaf_vertical_scale,
            aabb,
            sibling_idx,
            child_idx,
            has_stairs,
        });

        (idx, aabb)
    }

    /// Get the bounding box that covers the tree (all branches and leaves)
    pub fn get_bounds(&self) -> Aabb<f32> {
        self.branches[self.trunk_idx].aabb.union(self.root_aabb)
    }

    // Recursively search for branches or leaves by walking the tree's branch graph.
    fn walk_inner(
        &self,
        descend: &mut impl FnMut(&Branch, &Branch) -> bool,
        parent: &Branch,
        branch_idx: usize,
    ) {
        let branch = &self.branches[branch_idx];
        // Always probe the sibling branch, since it's not a child of the current
        // branch.
        let _branch_or_leaves = branch
            .sibling_idx
            .map(|idx| self.walk_inner(descend, parent, idx));

        // Only continue probing this sub-graph of the tree if the branch maches a
        // criteria (usually that it falls within the region we care about
        // sampling)
        if descend(branch, parent) {
            // Probe the children of this branch
            let _children = branch
                .child_idx
                .map(|idx| self.walk_inner(descend, branch, idx));
        }
    }

    /// Recursively walk the tree's branches, calling the current closure with
    /// the branch and its parent. If the closure returns `false`, recursion
    /// into the child branches is skipped.
    pub fn walk<F: FnMut(&Branch, &Branch) -> bool>(&self, mut f: F) {
        self.walk_inner(&mut f, &self.branches[self.trunk_idx], self.trunk_idx);
    }

    /// Determine whether there are either branches or leaves at the given
    /// position in the tree.
    #[inline(always)]
    pub fn is_branch_or_leaves_at(&self, pos: Vec3<f32>) -> (bool, bool, bool, bool) {
        let mut flags = Vec4::broadcast(false);
        self.walk(|branch, parent| {
            if branch.aabb.contains_point(pos) {
                flags |=
                    Vec4::<bool>::from(branch.is_branch_or_leaves_at(&self.config, pos, parent).0);
                true
            } else {
                false
            }
        });

        let (log, leaf, platform, air) = flags.into_tuple();

        let root = if self.root_aabb.contains_point(pos) {
            self.roots.iter().any(|root| {
                let p = root.line.projected_point(pos);
                let d2 = p.distance_squared(pos);
                d2 < root.radius.powi(2)
            })
        } else {
            false
        };
        (
            (log || root), /* & !air */
            leaf & !air,
            platform & !air,
            air,
        )
    }
}

// Branches are arranged in a graph shape. Each branch points to both its first
// child (if any) and also to the next branch in the list of child branches
// associated with the parent. This means that the entire tree is laid out in a
// walkable graph where each branch refers only to two other branches. As a
// result, walking the tree is simply a case of performing double recursion.
pub struct Branch {
    line: LineSegment3<f32>,
    wood_radius: f32,
    leaf_radius: f32,
    leaf_vertical_scale: f32,
    aabb: Aabb<f32>,

    sibling_idx: Option<usize>,
    child_idx: Option<usize>,

    has_stairs: bool,
}

impl Branch {
    /// Determine whether there are either branches or leaves at the given
    /// position in the branch.
    /// (branch, leaves, stairs, forced_air)
    pub fn is_branch_or_leaves_at(
        &self,
        config: &TreeConfig,
        pos: Vec3<f32>,
        parent: &Branch,
    ) -> ((bool, bool, bool, bool), f32) {
        // fn finvsqrt(x: f32) -> f32 {
        //     let y = f32::from_bits(0x5f375a86 - (x.to_bits() >> 1));
        //     y * (1.5 - ( x * 0.5 * y * y ))
        // }

        fn length_factor(line: LineSegment3<f32>, p: Vec3<f32>) -> f32 {
            let len_sq = line.start.distance_squared(line.end);
            if len_sq < 0.001 {
                0.0
            } else {
                (p - line.start).dot(line.end - line.start) / len_sq
            }
        }

        // fn smooth(a: f32, b: f32, k: f32) -> f32 {
        //     // let h = (0.5 + 0.5 * (b - a) / k).clamped(0.0, 1.0);
        //     // Lerp::lerp(b, a, h) - k * h * (1.0 - h)

        //     let h = (k-(a-b).abs()).max(0.0);
        //     a.min(b) - h * h * 0.25 / k
        // }

        let p = self.line.projected_point(pos);
        let d2 = p.distance_squared(pos);

        let length_factor = length_factor(self.line, pos);
        let wood_radius = if config.branch_child_radius_lerp {
            Lerp::lerp(parent.wood_radius, self.wood_radius, length_factor)
        } else {
            self.wood_radius
        };

        let mask = if d2 < wood_radius.powi(2) {
            (true, false, false, false) // Wood
        } else if {
            let diff = (p - pos) / Vec3::new(1.0, 1.0, self.leaf_vertical_scale);
            diff.magnitude_squared() < self.leaf_radius.powi(2)
        } {
            (false, true, false, false) // Leaves
        } else {
            let stair_width = 5.0;
            let stair_thickness = 2.0;
            let stair_space = 5.0;
            if self.has_stairs {
                let (platform, air) = if pos.z >= self.line.start.z.min(self.line.end.z) - 1.0
                    && pos.z
                        <= self.line.start.z.max(self.line.end.z) + stair_thickness + stair_space
                    && d2 < (wood_radius + stair_width).powi(2)
                {
                    let rpos = pos.xy() - p;
                    let stretch = 32.0;
                    let stair_section =
                        ((rpos.x as f32).atan2(rpos.y as f32) / (f32::consts::PI * 2.0) * stretch
                            + pos.z)
                            .rem_euclid(stretch);
                    (
                        stair_section < stair_thickness,
                        stair_section >= stair_thickness
                            && stair_section < stair_thickness + stair_space,
                    ) // Stairs
                } else {
                    (false, false)
                };

                let platform = platform
                    || (self.has_stairs
                        && self.wood_radius > 4.0
                        && !air
                        && d2 < (wood_radius + 10.0).powi(2)
                        && pos.z % 48.0 < stair_thickness);

                (false, false, platform, air)
            } else {
                (false, false, false, false)
            }
        };

        (mask, d2)
    }

    /// This returns an AABB of both the branch and all of the children of that
    /// branch
    pub fn get_aabb(&self) -> Aabb<f32> { self.aabb }

    pub fn get_line(&self) -> LineSegment3<f32> { self.line }

    pub fn get_wood_radius(&self) -> f32 { self.wood_radius }

    pub fn get_leaf_radius(&self) -> f32 { self.leaf_radius }
}

struct Root {
    line: LineSegment3<f32>,
    radius: f32,
}
