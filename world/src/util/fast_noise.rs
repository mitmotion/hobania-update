use super::{RandomField, Sampler};
use std::{f32, ops::Add};
use vek::*;

pub struct FastNoise {
    noise: RandomField,
}

impl FastNoise {
    pub const fn new(seed: u32) -> Self {
        Self {
            noise: RandomField::new(seed),
        }
    }

    #[allow(clippy::excessive_precision)] // TODO: Pending review in #587
    fn noise_at(&self, pos: Vec3<i32>) -> f32 {
        (self.noise.get(pos) & 4095) as f32 * 0.000244140625
    }
}

impl Sampler<'static, '_> for FastNoise {
    type Index = Vec3<f64>;
    type Sample = f32;

    fn get(&self, pos: Self::Index) -> Self::Sample {
        // let align_pos = pos.map(|e| e.floor());
        // let near_pos = align_pos.map(|e| e as i32);

        // let v000 = self.noise_at(near_pos + Vec3::new(0, 0, 0));
        // let v100 = self.noise_at(near_pos + Vec3::new(1, 0, 0));
        // let v010 = self.noise_at(near_pos + Vec3::new(0, 1, 0));
        // let v110 = self.noise_at(near_pos + Vec3::new(1, 1, 0));
        // let v001 = self.noise_at(near_pos + Vec3::new(0, 0, 1));
        // let v101 = self.noise_at(near_pos + Vec3::new(1, 0, 1));
        // let v011 = self.noise_at(near_pos + Vec3::new(0, 1, 1));
        // let v111 = self.noise_at(near_pos + Vec3::new(1, 1, 1));

        // let factor = (pos - align_pos).map(|e| e as f32);

        // let v00 = v000 + factor.z * (v001 - v000);
        // let v10 = v010 + factor.z * (v011 - v010);
        // let v01 = v100 + factor.z * (v101 - v100);
        // let v11 = v110 + factor.z * (v111 - v110);

        // let v0 = v00 + factor.y * (v01 - v00);
        // let v1 = v10 + factor.y * (v11 - v10);

        // (v0 + factor.x * (v1 - v0)) * 2.0 - 1.0

        let near_pos = pos.map(|e| e.floor() as i32);

        let v000 = self.noise_at(near_pos + Vec3::new(0, 0, 0));
        let v100 = self.noise_at(near_pos + Vec3::new(1, 0, 0));
        let v010 = self.noise_at(near_pos + Vec3::new(0, 1, 0));
        let v110 = self.noise_at(near_pos + Vec3::new(1, 1, 0));
        let v001 = self.noise_at(near_pos + Vec3::new(0, 0, 1));
        let v101 = self.noise_at(near_pos + Vec3::new(1, 0, 1));
        let v011 = self.noise_at(near_pos + Vec3::new(0, 1, 1));
        let v111 = self.noise_at(near_pos + Vec3::new(1, 1, 1));

        let factor = pos.map(|e| {
            let f = e.fract().add(1.0).fract() as f32;
            f.powi(2) * (3.0 - 2.0 * f)
        });

        let x00 = v000 + factor.x * (v100 - v000);
        let x10 = v010 + factor.x * (v110 - v010);
        let x01 = v001 + factor.x * (v101 - v001);
        let x11 = v011 + factor.x * (v111 - v011);

        let y0 = x00 + factor.y * (x10 - x00);
        let y1 = x01 + factor.y * (x11 - x01);

        (y0 + factor.z * (y1 - y0)) * 2.0 - 1.0
    }
}

pub struct FastNoise2d {
    noise: RandomField,
}

impl FastNoise2d {
    pub const fn new(seed: u32) -> Self {
        Self {
            noise: RandomField::new(seed),
        }
    }

    #[allow(clippy::excessive_precision)] // TODO: Pending review in #587
    fn noise_at(&self, pos: Vec2<i32>) -> f32 {
        (self.noise.get(Vec3::new(pos.x, pos.y, 0)) & 4095) as f32 * 0.000244140625
    }
}

impl Sampler<'static, '_> for FastNoise2d {
    type Index = Vec2<f64>;
    type Sample = f32;

    fn get(&self, pos: Self::Index) -> Self::Sample {
        let near_pos = pos.map(|e| e.floor() as i32);

        let v00 = self.noise_at(near_pos + Vec2::new(0, 0));
        let v10 = self.noise_at(near_pos + Vec2::new(1, 0));
        let v01 = self.noise_at(near_pos + Vec2::new(0, 1));
        let v11 = self.noise_at(near_pos + Vec2::new(1, 1));

        let factor = pos.map(|e| {
            let f = e.fract().add(1.0).fract() as f32;
            f.powi(2) * (3.0 - 2.0 * f)
        });

        let v0 = v00 + factor.x * (v10 - v00);
        let v1 = v01 + factor.x * (v11 - v01);

        (v0 + factor.y * (v1 - v0)) * 2.0 - 1.0
    }
}
