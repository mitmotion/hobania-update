use crate::{
    comp::{
        fluid_dynamics::{Drag, Glide, WingShape},
        Ori,
    },
    util::Dir,
};
use inline_tweak::tweak;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use vek::*;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glider {
    /// The aspect ratio is the ratio of the span squared to actual planform
    /// area
    pub wing_shape: WingShape,
    pub planform_area: f32,
    pub ori: Ori,
}

impl Glider {
    /// A glider is modelled as an elliptical wing and has a span length
    /// (distance from wing tip to wing tip) and a chord length (distance from
    /// leading edge to trailing edge through its centre) measured in block
    /// units.
    ///
    ///  https://en.wikipedia.org/wiki/Elliptical_wing
    pub fn new(span_length: f32, chord_length: f32, ori: Ori) -> Self {
        let planform_area = PI * chord_length * span_length * 0.25;
        let aspect_ratio = span_length.powi(2) / planform_area;
        Self {
            wing_shape: WingShape::Elliptical { aspect_ratio },
            planform_area,
            ori,
        }
    }

    pub fn roll(&mut self, angle_right: f32) { self.ori = self.ori.rolled_right(angle_right); }

    pub fn pitch(&mut self, angle_up: f32) { self.ori = self.ori.pitched_up(angle_up); }

    pub fn yaw(&mut self, angle_right: f32) { self.ori = self.ori.yawed_right(angle_right); }

    pub fn slerp_roll_towards(&mut self, dir: Dir, s: f32) {
        self.ori = self.ori.slerped_towards(
            self.ori
                .rolled_towards(if self.ori.up().dot(dir.to_vec()) < tweak!(0.0) {
                    Quaternion::rotation_3d(PI, self.ori.right()) * dir
                } else {
                    dir
                }),
            s,
        );
    }

    pub fn slerp_pitch_towards(&mut self, dir: Dir, s: f32) {
        self.ori = self.ori.slerped_towards(self.ori.pitched_towards(dir), s);
    }

    pub fn slerp_yaw_towards(&mut self, dir: Dir, s: f32) {
        self.ori = self.ori.slerped_towards(self.ori.yawed_towards(dir), s);
    }
}

impl Drag for Glider {
    fn parasite_drag_coefficient(&self) -> f32 { self.planform_area * 0.004 }
}

impl Glide for Glider {
    fn wing_shape(&self) -> &WingShape { &self.wing_shape }

    fn is_gliding(&self) -> bool { true }

    fn planform_area(&self) -> f32 { self.planform_area }

    fn ori(&self) -> &Ori { &self.ori }
}
