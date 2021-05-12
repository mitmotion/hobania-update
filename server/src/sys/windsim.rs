#![allow(dead_code)]
use crate::windsim::WindSim;
use common::{
    comp::{Pos, Vel},
    resources::DeltaTime,
};
use common_ecs::{Job, Origin, Phase, System};
use specs::{Read, Write};
use vek::*;
use rand::Rng;

/// This system updates the wind grid for the entire map
#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    #[allow(clippy::type_complexity)]
    type SystemData = (Read<'a, DeltaTime>, Write<'a, WindSim>);

    const NAME: &'static str = "windsim";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(_job: &mut Job<Self>, (dt, mut windsim): Self::SystemData) {
        let mut rng = rand::thread_rng();
        let wind_sources: Vec<(Pos, Vel)> = (0..1000)
            .map(|y| {
                let r1 = rng.gen_range(0.0..std::f32::consts::PI / 2.0);
                let r2 = rng.gen_range(0.0..std::f32::consts::PI / 2.0);
                let r = (r1 + r2) / 2.0;
                (
                    Pos(Vec3 {
                        x: inline_tweak::tweak!(5000.0),
                        y: y as f32 * 32.0,
                        z: inline_tweak::tweak!(200.0),
                    }),
                    Vel(Vec3 {
                        x: inline_tweak::tweak!(30.0) * r.sin(),
                        y: inline_tweak::tweak!(30.0) * r.cos(),
                        z: inline_tweak::tweak!(30.0) * rng.gen_range(-0.25..0.25),
                    }),
                )
            })
            .collect::<Vec<_>>();
        windsim.tick(wind_sources, &dt);
    }
}
