#![allow(dead_code)]
pub mod fluid;
mod types;
use common::{terrain::TerrainChunkSize, vol::RectVolSize};
pub use fluid::step_fluid;
use types::{WindGrid, X_SIZE, Y_SIZE, Z_SIZE};
use vek::*;
use world::sim::WorldSim;

use common::{
    comp::{Pos, Vel},
    resources::DeltaTime,
};
//use common_state::State;

#[derive(Default)]
pub struct WindSim {
    grid: WindGrid,
    blocks_per_cell: Vec3<u32>,
}

impl WindSim {
    pub fn new(world_sim: &WorldSim) -> Self {
        Self {
            grid: WindGrid::default(),
            blocks_per_cell: cell_size_in_blocks(world_sim),
        }
    }

    /// Converts world positions, to 3D grid positions.
    /// Returns None if out of bounds, for example negative positions.
    pub fn world_to_grid(&self, pos: Pos) -> Option<Vec3<usize>> {
        if pos
            .0
            .map2(self.blocks_per_cell, |pi, si| {
                pi >= 0.0 && pi <= (pi / si as f32)
            })
            .reduce_and()
        {
            Some(
                pos.0
                    .map2(self.blocks_per_cell, |pi, si| pi as usize / si as usize),
            )
        } else {
            None
        }
    }

    pub fn tick(&mut self, sources: Vec<(Pos, Vel)>, dt: &DeltaTime) {
        for (pos, vel) in sources {
            self.grid
                .add_velocity_source(pos.0.map(|e| e as usize), vel.0)
        }
        step_fluid(
            &mut self.grid.density,
            &mut self.grid.x_vel,
            &mut self.grid.y_vel,
            &mut self.grid.z_vel,
            dt.0,
            0.1,
            true,
        );
    }
}

fn cell_size_in_blocks(world_sim: &WorldSim) -> Vec3<u32> {
    let world_chunks: Vec2<u32> = world_sim.get_size();
    let blocks_per_chunk: Vec2<u32> = TerrainChunkSize::RECT_SIZE;

    let world_blocks: Vec2<u32> = world_chunks.map2(blocks_per_chunk, |ai, bi| ai * bi);

    let grid_size = Vec3 {
        x: X_SIZE as u32,
        y: Y_SIZE as u32,
        z: Z_SIZE as u32,
    };
    let cell_size_xy: Vec2<u32> = world_blocks.map2(grid_size.xy(), |ai, bi| ai / bi as u32);
    Vec3 {
        x: cell_size_xy.x,
        y: cell_size_xy.y,
        z: 500,
    }
}

// pub fn init(state: &mut State) {
//     let mut grid = WindGrid {
//         x_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
//         y_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
//         z_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
//         density: vec![0_f32; SIZE_1D].into_boxed_slice(),
//     };
//
//     state.ecs_mut().insert(grid);
//
// }
