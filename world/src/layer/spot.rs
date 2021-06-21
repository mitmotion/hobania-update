use crate::{
    sim::{SimChunk, WorldSim},
    util::seed_expan,
    Canvas,
};
use common::terrain::{Block, BlockKind, Structure};
use rand::prelude::*;
use rand_chacha::ChaChaRng;
use vek::*;

#[derive(Copy, Clone, Debug)]
pub enum Spot {
    Camp,
    Hideout,
}

impl Spot {
    pub fn generate(world: &mut WorldSim) {
        Self::generate_spots(
            Spot::Camp,
            world,
            10.0,
            |g, c| g < 0.25 && !c.near_cliffs() && !c.river.near_water(),
            false,
        );
        Self::generate_spots(
            Spot::Hideout,
            world,
            10.0,
            |g, c| g < 0.25 && !c.near_cliffs() && !c.river.near_water(),
            false,
        );
    }

    fn generate_spots(
        spot: Spot,
        world: &mut WorldSim,
        freq: f32, // Per sq km
        mut valid: impl FnMut(f32, &SimChunk) -> bool,
        trees: bool,
    ) {
        let world_size = world.get_size();
        for _ in 0..(world_size.product() as f32 / 32.0f32.powi(2) * freq).ceil() as u64 {
            let pos = world_size.map(|e| world.rng.gen_range(0..e as i32));
            if let Some((_, chunk)) = world
                .get_gradient_approx(pos)
                .zip(world.get_mut(pos))
                .filter(|(grad, chunk)| valid(*grad, chunk))
            {
                chunk.spot = Some(spot);
                if !trees {
                    chunk.tree_density = 0.0;
                }
            }
        }
    }
}

pub fn apply_spots_to(canvas: &mut Canvas, dynamic_rng: &mut impl Rng) {
    let nearby_spots = canvas.nearby_spots().collect::<Vec<_>>();

    for (spot_wpos, spot, seed) in nearby_spots.iter().copied() {
        let mut rng = ChaChaRng::from_seed(seed_expan::rng_state(seed));
        match spot {
            Spot::Camp => {
                canvas.foreach_col_area(
                    Aabr {
                        min: spot_wpos - 8,
                        max: spot_wpos + 8,
                    },
                    |canvas, wpos2d, col| {
                        if nearby_spots
                            .iter()
                            .any(|(wpos, _, _)| wpos.distance_squared(wpos2d) < 64)
                        {
                            for z in -8..32 {
                                canvas.set(
                                    wpos2d.with_z(col.alt as i32 + z),
                                    Block::new(BlockKind::Misc, Rgb::broadcast(255)),
                                );
                            }
                        }
                    },
                );
            },
            Spot::Hideout => {
                let structures = Structure::load_group("dungeon_entrances").read();
                let structure = structures.choose(&mut rng).unwrap();
                let origin = spot_wpos.with_z(canvas.land().get_alt_approx(spot_wpos) as i32);
                canvas.blit_structure(origin, &structure, seed);
            },
        }
    }
}
