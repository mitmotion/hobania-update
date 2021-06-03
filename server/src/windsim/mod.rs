#![allow(dead_code)]
pub mod fluid;
mod wind_grid;
use common::{terrain::TerrainChunkSize, vol::RectVolSize};
pub use fluid::step_fluid;
use vek::*;
use wind_grid::{WindGrid, X_SIZE, Y_SIZE, Z_SIZE};

use common::{
    comp::{Pos, Vel},
    resources::DeltaTime,
};

pub const DEFAULT_POS: Vec3<usize> = Vec3 { x: 0, y: 0, z: 0 };
pub const GRID_SIZE: Vec3<usize> = Vec3 {
    x: X_SIZE,
    y: Y_SIZE,
    z: Z_SIZE,
};
pub const TPS: u32 = 5;

#[derive(Default)]
pub struct WindSim {
    grid: WindGrid,
    blocks_per_cell: Vec3<u32>,
    timer: f32,
}

impl WindSim {
    pub fn new(world_size: &Vec2<u32>) -> Self {
        Self {
            grid: WindGrid::default(),
            blocks_per_cell: cell_size_in_blocks(world_size),
            timer: std::f32::INFINITY,
        }
    }

    /// Converts world positions, to 3D grid positions.
    /// Returns None if out of bounds, for example negative positions.
    pub fn world_to_grid(&self, pos: Pos) -> Option<Vec3<usize>> {
        let p = |pi, ci, gi| pi >= 0.0 && ci <= gi;

        let cell_pos = pos
            .0
            .map2(self.blocks_per_cell, |pi, si| pi as usize / si as usize);
        if pos.0.map3(cell_pos, GRID_SIZE, p).reduce_and() {
            Some(cell_pos)
        } else {
            None
        }
    }

    // Takes  a world position and returns a world velocity
    pub fn get_velocity(&self, pos: Pos) -> Vec3<f32> {
        let cell_pos = self.world_to_grid(pos).unwrap_or(DEFAULT_POS);
        let cell_vel = self.grid.get_velocity(cell_pos);
        cell_vel.map2(self.blocks_per_cell, |vi, si| vi * si as f32)
    }

    pub fn add_velocity_source(&mut self, pos: Pos, vel: Vel) {
        let cell_pos = self.world_to_grid(pos).unwrap_or(DEFAULT_POS);
        let cell_vel = vel.0.map2(self.blocks_per_cell, |vi, si| vi / si as f32);
        self.grid.add_velocity_source(cell_pos, cell_vel)
    }

    // Abstraction for running the simulation
    pub fn tick(&mut self, sources: Vec<(Pos, Vel)>, dt: &DeltaTime) {
        self.timer += dt.0;
        let step = 1.0 / TPS as f32;
        if self.timer >= step {
            for (pos, vel) in sources {
                self.add_velocity_source(pos, vel);
            }
            step_fluid(
                &mut self.grid.density,
                &mut self.grid.x_vel,
                &mut self.grid.y_vel,
                &mut self.grid.z_vel,
                step,
                0.1,
                true,
            );
            if self.timer.is_infinite() {
                self.timer = 0.0;
            } else {
                self.timer -= step;
            }
        }
    }
}

fn cell_size_in_blocks(world_chunks: &Vec2<u32>) -> Vec3<u32> {
    // world_blocks = world_chunks / blocks_per_chunk
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
