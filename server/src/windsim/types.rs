#![allow(dead_code)]
use vek::*;

pub const X_SIZE: usize = 96;
pub const Y_SIZE: usize = 96;
pub const Z_SIZE: usize = 4;
pub const SIZE_1D: usize = (X_SIZE + 2) * (Y_SIZE + 2) * (Z_SIZE + 2);

#[derive(Clone, Debug, PartialEq)]
pub struct WindGrid {
    pub x_vel: Box<[f32]>,
    pub y_vel: Box<[f32]>,
    pub z_vel: Box<[f32]>,
    pub density: Box<[f32]>,
}
impl Default for WindGrid {
    fn default() -> Self {
        WindGrid {
            x_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
            y_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
            z_vel: vec![0_f32; SIZE_1D].into_boxed_slice(),
            density: vec![0_f32; SIZE_1D].into_boxed_slice(),
        }
    }
}
impl WindGrid {
    // Takes 3D grid position (not a world position)
    pub fn add_velocity_source(&mut self, pos: Vec3<usize>, vel: Vec3<f32>) {
        // Convert to 1D grid position
        let index = Self::get_index(pos.x, pos.y, pos.z);
        self.x_vel[index] = vel.x;
        self.y_vel[index] = vel.y;
        self.z_vel[index] = vel.z;
    }

    // Takes 3D grid position (not a world position)
    pub fn add_density_source(&mut self, pos: Vec3<usize>, dens: f32) {
        let index = Self::get_index(pos.x, pos.y, pos.z);
        self.density[index] = dens;
    }

    pub fn get_index(x: usize, y: usize, z: usize) -> usize {
        x + (X_SIZE + 2) * (y + (Y_SIZE + 2) * z)
    }

    // Takes 3D grid position (not a world position)
    pub fn get_velocity(&self, pos: Vec3<usize>) -> Vec3<f32> {
        let index = Self::get_index(pos.x, pos.y, pos.z);
        let x = self.x_vel[index] as f32;
        let y = self.y_vel[index] as f32;
        let z = self.y_vel[index] as f32;

        Vec3 { x, y, z }
    }
}
