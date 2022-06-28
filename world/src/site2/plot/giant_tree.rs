use crate::{
    layer::tree::{ProceduralTree, TreeConfig},
    site::namegen::NameGen,
    site2::{Fill, Filler, FillFn, Painter, Site, Structure},
    util::FastNoise,
    Land, Sampler,
};
use common::{
    generation::EntityInfo,
    terrain::{Block, BlockKind},
};
use rand::Rng;
use vek::*;

pub struct GiantTree {
    name: String,
    wpos: Vec3<i32>,
    tree: ProceduralTree,
    seed: u32,
}

impl GiantTree {
    pub fn generate(site: &Site, center_tile: Vec2<i32>, land: &Land, rng: &mut impl Rng) -> Self {
        let wpos = site.tile_center_wpos(center_tile);
        Self {
            name: format!("Tree of {}", NameGen::location(rng).generate()),
            // Get the tree's altitude
            wpos: wpos.with_z(land.get_alt_approx(wpos) as i32),
            tree: {
                let config = TreeConfig::giant(rng, 4.0, true);
                ProceduralTree::generate(config, rng)
            },
            seed: rng.gen(),
        }
    }

    pub fn name(&self) -> &str { &self.name }

    pub fn radius(&self) -> f32 { 100.0 }

    pub fn tree(&self) -> &ProceduralTree { &self.tree }

    pub fn entity_at(
        &self,
        pos: Vec3<i32>,
        above_block: &Block,
        dynamic_rng: &mut impl Rng,
    ) -> Option<EntityInfo> {
        if above_block.kind() == BlockKind::Leaves && dynamic_rng.gen_bool(0.001) {
            let entity = EntityInfo::at(pos.as_());
            match dynamic_rng.gen_range(0..=4) {
                0 => {
                    Some(entity.with_asset_expect(
                        "common.entity.wild.aggressive.horn_beetle",
                        dynamic_rng,
                    ))
                },
                1 => {
                    Some(entity.with_asset_expect(
                        "common.entity.wild.aggressive.stag_beetle",
                        dynamic_rng,
                    ))
                },
                2 => Some(
                    entity.with_asset_expect("common.entity.wild.aggressive.deadwood", dynamic_rng),
                ),
                3 => Some(
                    entity.with_asset_expect("common.entity.wild.aggressive.maneater", dynamic_rng),
                ),
                4 => Some(
                    entity.with_asset_expect("common.entity.wild.peaceful.parrot", dynamic_rng),
                ),
                _ => None,
            }
        } else {
            None
        }
    }
}

impl<F: Filler> Structure<F> for GiantTree {
    fn render<'a>(&self, _site: &Site, _land: Land, painter: &Painter<'a>, filler: &mut FillFn<'a, '_, F>) {
        let bounds = self.tree.get_bounds().map(|e| e as i32);
        /* let trunk_block = self.tree.config.trunk_block; */
        /* let leaf_vertical_scale = self.tree.config.leaf_vertical_scale.recip();
        let branch_child_radius_lerp = self.tree.config.branch_child_radius_lerp; */
        let leaf_vertical_scale = /*t.config.leaf_vertical_scale*/0.6f32.recip()/*1.0*/;
        let branch_child_radius_lerp = true;

        /* let trunk_block = filler.block_from_structure(
            trunk_block,
            self.wpos.xy(),
            self.seed,
            &col,
        ); */
        let trunk_block = Block::new(BlockKind::Wood, Rgb::new(80, 32, 0));
        /* let leaf_block = filler.block_from_structure(
            leaf_block,
            self.wpos.xy(),
            self.seed,
            &col,
        ); */
        let fast_noise = FastNoise::new(self.seed);
        let dark = Rgb::new(10, 70, 50).map(|e| e as f32);
        let light = Rgb::new(80, 140, 10).map(|e| e as f32);
        let leaf_col = Lerp::lerp(
            dark,
            light,
            fast_noise.get((self.wpos.map(|e| e as f64) * 0.05) * 0.5 + 0.5),
        );
        let leaf_block = Block::new(BlockKind::Leaves, leaf_col.map(|e| e as u8));

        let trunk_block = filler.block(trunk_block);
        let leaf_block = filler.block(leaf_block);

        self.tree.walk(|branch, parent| {
            let aabr = Aabr {
                min: self.wpos.xy() + branch.get_aabb().min.xy().as_(),
                max: self.wpos.xy() + branch.get_aabb().max.xy().as_(),
            };
            if aabr.collides_with_aabr(filler.render_aabr().as_()) {
                let start =
                    self.wpos.as_::<f32>() + branch.get_line().start/*.as_()*//* - 0.5*/;
                let end =
                    self.wpos.as_::<f32>() + branch.get_line().end/*.as_()*//* - 0.5*/;
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
        /* // Draw the roots.
        t.roots.iter().for_each(|root| {
            painter
                .line(
                    wpos/*.as_::<f32>()*/ + root.line.start.as_()/* - 0.5*/,
                    wpos/*.as_::<f32>()*/ + root.line.end.as_()/* - 0.5*/,
                    root.radius,
                )
                .fill(/*filler.block(leaf_block)*/trunk_block, filler);
        }); */

        /* let fast_noise = FastNoise::new(self.seed);
        let dark = Rgb::new(10, 70, 50).map(|e| e as f32);
        let light = Rgb::new(80, 140, 10).map(|e| e as f32);
        let leaf_col = Lerp::lerp(
            dark,
            light,
            fast_noise.get((self.wpos.map(|e| e as f64) * 0.05) * 0.5 + 0.5),
        );
        let leaf_vertical_scale = /*t.config.leaf_vertical_scale*/0.6f32.recip()/*1.0*/;
        self.tree.walk(|branch, parent| {
            let aabr = Aabr {
                min: self.wpos.xy() + branch.get_aabb().min.xy().as_(),
                max: self.wpos.xy() + branch.get_aabb().max.xy().as_(),
            };
            if aabr.collides_with_aabr(filler.render_aabr().as_()) {
                painter
                    .line_two_radius(
                        self.wpos + branch.get_line().start.as_(),
                        self.wpos + branch.get_line().end.as_(),
                        parent.get_wood_radius(),
                        branch.get_wood_radius(),
                        1.0,
                    )
                    .fill(filler.block(Block::new(
                        BlockKind::Wood,
                        Rgb::new(80, 32, 0),
                    )), filler);
                if branch.get_leaf_radius() > branch.get_wood_radius() {
                    painter
                        .line_two_radius(
                            self.wpos + branch.get_line().start.as_(),
                            self.wpos + branch.get_line().end.as_(),
                            parent.get_leaf_radius(),
                            branch.get_leaf_radius(),
                            leaf_vertical_scale,
                        )
                        .fill(filler.block(Block::new(
                            BlockKind::Leaves,
                            leaf_col.map(|e| e as u8),
                        )), filler)
                }
                true
            } else {
                false
            }
        }); */
    }
}
