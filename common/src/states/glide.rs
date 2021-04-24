use super::utils::handle_climb;
use crate::{
    comp::{inventory::slot::EquipSlot, CharacterState, Ori, StateUpdate},
    states::behavior::{CharacterBehavior, JoinData},
    util::Dir,
};
use serde::{Deserialize, Serialize};
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
    pub fn new(span_length: f32, chord_length: f32, ori: Ori) -> Self {
        let planform_area = std::f32::consts::PI * chord_length * span_length * 0.25;
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
            let tgt_ori = Some(data.inputs.move_dir)
                .filter(|mv_dir| !mv_dir.is_approx_zero())
                .map(|mv_dir| {
                    Vec3::new(
                        mv_dir.x,
                        mv_dir.y,
                        Lerp::lerp_unclamped(
                            0.0,
                            data.inputs.look_dir.z + inline_tweak::tweak!(0.3),
                            mv_dir.magnitude_squared() * inline_tweak::tweak!(2.0),
                        ),
                    )
                })
                .and_then(Dir::from_unnormalized)
                .and_then(|tgt_dir| {
                    Dir::from_unnormalized(data.vel.0)
                        .and_then(|moving_dir| moving_dir.to_horizontal())
                        .map(|moving_dir| {
                            Ori::from(tgt_dir).rolled_right(
                                (1.0 - moving_dir.dot(*tgt_dir).max(0.0))
                                    * self.ori.right().dot(*tgt_dir).signum()
                                    * std::f32::consts::PI
                                    / 3.0,
                            )
                        })
                })
                .unwrap_or_else(|| self.ori.uprighted());

            let rate = {
                let angle = self.ori.look_dir().angle_between(*data.inputs.look_dir);
                0.4 * std::f32::consts::PI / angle
            };

            let ori = self
                .ori
                .slerped_towards(tgt_ori, (data.dt.0 * rate).min(0.1));
            update.character = CharacterState::Glide(Self { ori, ..*self });

            if let Some(char_ori) = ori.to_horizontal() {
                let rate = {
                    let angle = ori.look_dir().angle_between(*data.inputs.look_dir);
                    data.body.base_ori_rate() * std::f32::consts::PI / angle
                };
                update.ori = update
                    .ori
                    .slerped_towards(char_ori, (data.dt.0 * rate).min(0.1));
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
