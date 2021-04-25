use super::utils::handle_climb;
use crate::{
    comp::{
        fluid_dynamics::angle_of_attack, inventory::slot::EquipSlot, CharacterState, Ori,
        StateUpdate,
    },
    states::behavior::{CharacterBehavior, JoinData},
    util::{Dir, Plane, Projection},
};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use vek::*;

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    /// The aspect ratio is the ratio of the span squared to actual planform
    /// area
    pub aspect_ratio: f32,
    pub planform_area: f32,
    pub ori: Ori,
}

impl Data {
    /// A glider is modelled as an elliptical wing and has a span length
    /// (distance from wing tip to wing tip) and a chord length (distance from
    /// leading edge to trailing edge through its centre) measured in block
    /// units.
    ///
    ///  https://en.wikipedia.org/wiki/Elliptical_wing
    pub fn new(span_length: f32, chord_length: f32, ori: Ori) -> Self {
        let planform_area = PI * chord_length * span_length * 0.25;
        Self {
            aspect_ratio: span_length.powi(2) / planform_area,
            planform_area,
            ori,
        }
    }
}

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        // If player is on ground, end glide
        if data.physics.on_ground && data.vel.0.magnitude_squared() < 25.0 {
            update.character = CharacterState::GlideWield;
            update
        } else if data
            .physics
            .in_liquid()
            .map(|depth| depth > 0.5)
            .unwrap_or(false)
            || data.inventory.equipped(EquipSlot::Glider).is_none()
        {
            update.character = CharacterState::Idle;
            update
        } else if handle_climb(&data, &mut update) {
            update
        } else {
            let slerp_s = {
                let angle = self.ori.look_dir().angle_between(*data.inputs.look_dir);
                let rate = 0.4 * PI / angle;
                (data.dt.0 * rate).min(0.1)
            };

            let ori = Some(data.inputs.move_dir)
                .filter(|mv_dir| !mv_dir.is_approx_zero())
                .or_else(|| data.inputs.look_dir.xy().try_normalized())
                .map(|mv_dir| Vec3::new(mv_dir.x, mv_dir.y, (data.inputs.look_dir.z + 0.3) * 2.0))
                .and_then(Dir::from_unnormalized)
                .and_then(|tgt_dir| {
                    data.physics
                        .in_fluid
                        .map(|fluid| fluid.relative_flow(data.vel))
                        .and_then(|air_flow| {
                            let flow_dir = Dir::from_unnormalized(air_flow.0)?;
                            let tgt_dir_ori = Ori::from(tgt_dir);
                            let tgt_dir_up = tgt_dir_ori.up();
                            let tgt_up = flow_dir.projected(&Plane::from(tgt_dir)).map(|d| {
                                let ddot = d.dot(*tgt_dir_up);
                                if ddot.is_sign_negative() {
                                    Quaternion::rotation_3d(PI, *tgt_dir_ori.right()) * d
                                } else {
                                    d
                                }
                                .slerped_to(tgt_dir_up, 0.25 * ddot.abs().powf(0.25))
                            })?;
                            let global_roll = tgt_dir_up.rotation_between(tgt_up);
                            Some(
                                tgt_dir_ori
                                    .prerotated(global_roll)
                                    .pitched_up(angle_of_attack(&tgt_dir_ori, &flow_dir)),
                            )
                        })
                })
                .map(|tgt_ori| self.ori.slerped_towards(tgt_ori, slerp_s))
                .unwrap_or_else(|| self.ori.slerped_towards(self.ori.uprighted(), slerp_s));

            update.character = CharacterState::Glide(Self { ori, ..*self });

            if let Some(char_ori) = ori.to_horizontal() {
                let slerp_s = {
                    let angle = ori.look_dir().angle_between(*data.inputs.look_dir);
                    let rate = data.body.base_ori_rate() * PI / angle;
                    (data.dt.0 * rate).min(0.1)
                };
                update.ori = update.ori.slerped_towards(char_ori, slerp_s);
            }
            update
        }
    }

    fn unwield(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        update.character = CharacterState::Idle;
        update
    }
}
