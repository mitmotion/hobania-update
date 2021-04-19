use super::utils::handle_climb;
use crate::{
    comp::{CharacterState, Ori, StateUpdate},
    states::behavior::{CharacterBehavior, JoinData},
    util::Dir,
};
use serde::{Deserialize, Serialize};
use vek::*;

#[derive(Default, Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flappy {
    /// 0..1 wing angular speed from zero to max
    flap_speed: f32,
    /// PI/2..-PI/2 from angled up (dihedral) to angled down (anhedral)
    dihedral_angle: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Data {
    span_length: f32,
    chord_length: f32,
    // could be magic or some other propellant;
    // we don't want it necessarily mutually exclusive with wings
    max_thrust: Option<f32>,
    flappy: Option<Flappy>,
}

impl Data {
    pub fn new(span_length: f32, chord_length: f32, max_thrust: Option<f32>, flappy: bool) -> Self {
        Self {
            span_length,
            chord_length,
            max_thrust,
            flappy: flappy.then(|| Flappy::default()),
        }
    }
}

// This is all predicated on gliders being mounts. This is only for entities
// which can themselves fly without external assistance
impl CharacterBehavior for Data {
    fn behavior(&self, data: &JoinData) -> StateUpdate {
        let mut update = StateUpdate::from(data);

        // If on ground, land
        if (data.physics.on_ground && data.vel.0.magnitude_squared() < 25.0)
            || (data.physics.in_liquid().map_or(false, |depth| depth > 0.5))
        {
            update.character = CharacterState::Idle;
            update
        } else if handle_climb(&data, &mut update) {
            update
        } else {
            let efficiency = 1.0; // input?
            // let mut ori = glider.map(|g| g.ori).unwrap_or(update.ori);
            let fw_dir = data.ori.look_dir().to_horizontal();
            let tgt_ori = Some(data.inputs.move_dir)
                .filter(|mv_dir| !mv_dir.is_approx_zero())
                .map(|mv_dir| {
                    Vec3::new(
                        mv_dir.x,
                        mv_dir.y,
                        Lerp::lerp_unclamped(
                            0.0,
                            data.inputs.look_dir.z + inline_tweak::tweak!(0.3),
                            mv_dir.magnitude_squared() * inline_tweak::tweak!(2.5),
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
                                    * data.ori.right().dot(*tgt_dir).signum()
                                    * std::f32::consts::PI
                                    / 3.0,
                            )
                        })
                })
                .or_else(|| fw_dir.map(Ori::from))
                .unwrap_or_default();
            let rate = {
                let angle = data.ori.look_dir().angle_between(*data.inputs.look_dir);
                data.body.base_ori_rate() * efficiency * std::f32::consts::PI / angle
            };

            update.ori = data
                .ori
                .slerped_towards(tgt_ori, (data.dt.0 * rate).min(0.1));

            if let Some(max_thrust) = self.max_thrust {
                let accel = efficiency * max_thrust / data.mass.0;

                update.vel.0 += Vec3::broadcast(data.dt.0)
                    * accel
                    * if data.body.can_strafe() {
                        tgt_ori.look_vec()
                    } else {
                        let d = tgt_ori.look_vec();
                        d * update.ori.look_dir().dot(d)
                    };
            };

            // Elevation control - the ability to maintain altitude determines success
            // match data.body {
            //     // flappy flappy
            //     Body::Dragon(_) | Body::BirdMedium(_) | Body::BirdSmall(_) => {
            //         update.vel.0.z += data.dt.0 * accel * data.inputs.move_z.max(0.0);

            //         true
            //     },

            //     // floaty floaty
            //     Body::Ship(ship @ ship::Body::DefaultAirship) => {
            //         let regulate_density = |min: f32, max: f32, def: f32, rate: f32| ->
            // Density {             // Reset to default on no input
            //             let change = if data.inputs.move_z.abs() > std::f32::EPSILON {
            //                 -data.inputs.move_z
            //             } else {
            //                 (def - data.density.0).max(-1.0).min(1.0)
            //             };
            //             Density((update.density.0 + data.dt.0 * rate * change).clamp(min,
            // max))         };
            //         let def_density = ship.density().0;
            //         if data.physics.in_liquid().is_some() {
            //             let hull_density = ship.hull_density().0;
            //             update.density.0 =
            //                 regulate_density(def_density * 0.6, hull_density,
            // hull_density, 25.0).0;         } else {
            //             update.density.0 = regulate_density(
            //                 def_density * 0.5,
            //                 def_density * 1.5,
            //                 def_density,
            //                 0.5,
            //             )
            //             .0;
            //         };

            //         true
            //     },

            //     // oopsie woopsie
            //     _ => false,
            // }
            update
        }
    }
}
