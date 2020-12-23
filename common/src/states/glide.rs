use super::utils::handle_climb;
use crate::{
    comp::{CharacterState, StateUpdate},
    states::behavior::{CharacterBehavior, JoinData},
    util::Dir,
};
use serde::{Deserialize, Serialize};
use vek::Vec2;
use vek::Vec3;
// Gravity is 9.81 * 4, so this makes gravity equal to .15
const GLIDE_ANTIGRAV: f32 = crate::consts::GRAVITY * 0.90;
const HORIZ_CHANGE: f32 = 0.8;

//This is used to lower AIR_FRIC applied in common/sys/source/phys.rs
//If you want to fully counteract AIR_FRIC = 0.0125, use 1.0126582278 (1/1-AIR_FRIC) here
const FRIC_REDUCE: f32 = 1.01;

//, Eq, Hash
#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub struct Data;

impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        // If player is on ground, end glide
        if data.physics.on_ground {
            update.character = CharacterState::GlideWield;
            return update;
        }
        if data
            .physics
            .in_liquid
            .map(|depth| depth > 0.5)
            .unwrap_or(false)
        {
            update.character = CharacterState::Idle;
        }
        if data.loadout.glider.is_none() {
            update.character = CharacterState::Idle
        };
        // If there is a wall in front of character and they are trying to climb go to
        // climb
        handle_climb(&data, &mut update);

        // Determine orientation vector from movement direction vector
        //let ori_dir = Vec2::from(update.vel.0);
        //update.ori.0 = Dir::slerp_to_vec3(update.ori.0, ori_dir.into(), data.dt.0);

        //A single variable that will control both vertical and horizontal speed while gliding
        let magnitude: f32 = (update.vel.0.magnitude()
            - if data.inputs.look_dir.z.abs() < 0.08 {
            7.0 * data.dt.0
            } else {
                0.0
            })
            //Reduce the effects of AIR_FRIC
            * FRIC_REDUCE.powf(data.dt.0 * 60.0);

        let ori_dir = Vec2::from(update.vel.0);
        update.ori.0 = Dir::slerp_to_vec3(update.ori.0, ori_dir.into(), 0.5 * data.dt.0);

        let look_grav = crate::consts::GRAVITY * data.inputs.look_dir.z;

        /*
        Vec3::slerp(update.vel.0,
            Vec3::broadcast(magnitude) * look,
            0.9);
*/
/*
        if data.inputs.look_dir.z < 1.0 {
            update.vel.0 += (Vec3::broadcast(magnitude)
                * look
                - update.vel.0)
                * 0.9;

            if data.inputs.look_dir.z > 0.0
                && magnitude < 5.0 {
                update.vel.0.z *= -1.0 - 10.0 * data.dt.0;
            }
        }
        */

        if data.inputs.look_dir.z < 0.707 {
            update.vel.0.x += (magnitude
                * data.inputs.look_dir.x
                - update.vel.0.x)
                * HORIZ_CHANGE;

            update.vel.0.y += (magnitude
                * data.inputs.look_dir.y
                - update.vel.0.y)
                * HORIZ_CHANGE;

            if data.inputs.look_dir.z > 0.0
                && update.vel.0.z < 0.0
                && magnitude < 1.8
                    * look_grav {
                    update.vel.0.z += GLIDE_ANTIGRAV
                        * data.dt.0;

                    update.vel.0.x = update.vel.0.x.abs().min(look_grav) * update.vel.0.x.signum();
                    update.vel.0.y = update.vel.0.y.abs().min(look_grav) * update.vel.0.y.signum();
                    update.vel.0.z = update.vel.0.z.max(-look_grav);

            } else {
                update.vel.0.z += magnitude
                    * data.inputs.look_dir.z
                    - if data.inputs.look_dir.z.abs() < 0.08 {
                    GLIDE_ANTIGRAV * data.dt.0
                    } else {
                        0.0
                    }
                    - update.vel.0.z;
            };
        };
        update
    }

    fn unwield(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);
        update.character = CharacterState::Idle;
        update
    }
}
