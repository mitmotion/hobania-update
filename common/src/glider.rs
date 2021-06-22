use crate::{
    comp::{
        fluid_dynamics::{Drag, Glide, WingShape},
        Body, Ori, Pos,
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
    span_length: f32,
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
            span_length,
            ori,
        }
    }

    pub fn pos(&self, pilot: (&Pos, &Ori, &Body)) -> Pos {
        let height = pilot.2.dimensions().z;
        let cg = Vec3::unit_z() * height * tweak!(0.7);
        let pos_from_cg = *pilot.1.up() * height * tweak!(0.5);
        Pos(pilot.0.0 + cg + pos_from_cg)
    }

    pub fn wing_tips(&self, pilot: (&Pos, &Ori, &Body)) -> (Pos, Pos) {
        let right = tweak!(0.65) * *self.ori.right() * self.span_length / 2.0;
        let pos = self.pos(pilot).0;
        (Pos(pos - right), Pos(pos + right))
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
