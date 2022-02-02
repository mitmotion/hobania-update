use crate::{
    layer::tree::{ProceduralTree, TreeConfig},
    site::namegen::NameGen,
    site2::{Fill, Painter, Site, Structure},
    util::FastNoise,
    Land, Sampler,
};
use common::terrain::{Block, BlockKind};
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
            // Find the tree's altitude
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
}

impl Structure for GiantTree {
    fn render(&self, _site: &Site, _land: &Land, painter: &Painter) {
        let fast_noise = FastNoise::new(self.seed);
        let dark = Rgb::new(10, 70, 50).map(|e| e as f32);
        let light = Rgb::new(80, 140, 10).map(|e| e as f32);
        let leaf_col = Lerp::lerp(
            dark,
            light,
            fast_noise.get((self.wpos.map(|e| e as f64) * 0.05) * 0.5 + 0.5),
        );
        self.tree.walk(|branch, _| {
            let aabr = Aabr {
                min: self.wpos.xy() + branch.get_aabb().min.xy().as_(),
                max: self.wpos.xy() + branch.get_aabb().max.xy().as_(),
            };
            if aabr.collides_with_aabr(painter.render_aabr().as_()) {
                // TODO : Migrate to using Painter#line() instead
                painter
                    .line(
                        self.wpos + branch.get_line().start.as_(),
                        self.wpos + branch.get_line().end.as_(),
                        branch.get_wood_radius(),
                    )
                    .fill(Fill::Block(Block::new(
                        BlockKind::Wood,
                        Rgb::new(80, 32, 0),
                    )));
                if branch.get_leaf_radius() > branch.get_wood_radius() {
                    painter
                        .line(
                            self.wpos + branch.get_line().start.as_(),
                            self.wpos + branch.get_line().end.as_(),
                            branch.get_leaf_radius(),
                        )
                        .fill(Fill::Block(Block::new(
                            BlockKind::Leaves,
                            leaf_col.map(|e| e as u8),
                        )))
                }
                true
            } else {
                false
            }
        });
    }
}
