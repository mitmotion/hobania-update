use crate::{
    comp::{Ori, Pos, Vel},
    consts::GRAVITY,
    resources::DeltaTime,
    util::Dir,
};
use serde::{Deserialize, Serialize};
use specs::{Component, DerefFlaggedStorage};
use std::ops::Mul;
use vek::*;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MovementState {
    kind: MovementKind,
    ori: OriUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MovementKind {
    Stationary,
    SlowFall {
        lift: f32,
    },
    Flight {
        lift: f32,
        dir: Option<Dir>,
        accel: f32,
    },
    Swim {
        dir: Option<Dir>,
        accel: f32,
    },
    Leap {
        dir: Option<Dir>,
        vertical: f32,
        forward: f32,
        progress: f32,
    },
    Ground {
        dir: Option<Dir>,
        accel: f32,
    },
    Climb {
        dir: Option<Dir>,
        accel: f32,
    },
    Teleport {
        pos: Vec3<f32>,
    },
    Boost {
        dir: Option<Dir>,
        accel: f32,
    },
    ChangeSpeed {
        speed: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum OriUpdate {
    New(Ori),
    Stationary,
}

impl MovementState {
    pub fn with_movement(mut self, kind: MovementKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_ori_update(mut self, ori: OriUpdate) -> Self {
        self.ori = ori;
        self
    }

    pub fn handle_movement(&self, dt: &DeltaTime, pos: &mut Pos, vel: &mut Vel, ori: &mut Ori) {
        match self.kind {
            MovementKind::Stationary => {},
            MovementKind::SlowFall { lift } => {
                vel.0.z += dt.0 * lift;
            },
            MovementKind::Flight { lift, dir, accel } => {
                let dir = dir.map(|d| d.xy()).unwrap_or_default();
                vel.0.z += dt.0 * lift;
                vel.0 += dir * accel * dt.0;
            },
            MovementKind::Swim { dir, accel } => {
                let dir = dir.map(|d| *d).unwrap_or_default();
                vel.0 += dir * accel * dt.0;
            },
            MovementKind::Leap {
                dir,
                vertical,
                forward,
                progress,
            } => {
                let dir = dir.map(|d| d.xy()).unwrap_or_default();
                let progress = progress.clamp(0.0, 1.0);
                // TODO: Make this += instead of =, will require changing magnitude of strengths
                // probably, and potentially other behavior too Multiplication
                // by 2 to make `progress` "average" 1.0
                vel.0 = dir.mul(forward).with_z(vertical * progress * 2.0);
            },
            MovementKind::Ground { dir, accel } => {
                let dir = dir.map(|d| d.xy()).unwrap_or_default();
                vel.0 += dir * accel * dt.0;
            },
            MovementKind::Climb { dir, accel } => {
                let dir = dir.map(|d| *d).unwrap_or_default();
                vel.0.z += GRAVITY * dt.0;
                vel.0 += dir * accel * dt.0;
            },
            MovementKind::Teleport { pos: new_pos } => pos.0 = new_pos,
            MovementKind::Boost { dir, accel } => {
                let dir = dir.map(|d| *d).unwrap_or_default();
                vel.0 += dir * accel * dt.0;
            },
            MovementKind::ChangeSpeed { speed } => {
                vel.0 = vel.0.try_normalized().unwrap_or_default() * speed;
            },
        }

        match self.ori {
            OriUpdate::Stationary => {},
            OriUpdate::New(new_ori) => *ori = new_ori,
        }
    }
}

impl Default for MovementState {
    fn default() -> Self {
        Self {
            kind: MovementKind::Stationary,
            ori: OriUpdate::Stationary,
        }
    }
}

impl Component for MovementState {
    type Storage = DerefFlaggedStorage<Self>;
}
