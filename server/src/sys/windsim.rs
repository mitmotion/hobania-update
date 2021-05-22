#![allow(dead_code)]
use crate::windsim::{WindSim, MS_BETWEEN_TICKS};
use common::{
    comp::{Pos, Vel},
    resources::DeltaTime,
};
use common_ecs::{Job, Origin, Phase, System};
use rand::Rng;
use specs::{Read, Write};
use vek::*;

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
        // 1000 chunks
        let wind_sources: Vec<(Pos, Vel)> = (0..1000)
            .map(|vertical_chunk| {
                let r1 = rng.gen_range(0.0..std::f32::consts::PI / 2.0);
                let r2 = rng.gen_range(0.0..std::f32::consts::PI / 2.0);
                let r = (r1 + r2) / 2.0;
                (
                    Pos(Vec3 {
                        x: inline_tweak::tweak!(1000.0),
                        y: vertical_chunk as f32 * 32.0,
                        z: inline_tweak::tweak!(200.0),
                    }),
                    Vel(Vec3 {
                        x: inline_tweak::tweak!(300.0) * r.sin(),
                        y: inline_tweak::tweak!(300.0) * r.cos(),
                        z: inline_tweak::tweak!(10.0) * rng.gen_range(-0.25..0.25),
                    }),
                )
            })
            .collect::<Vec<_>>();
        // If MS_BETWEEN_TICKS is 1000 it runs the sim once per second
        if windsim.ms_since_update >= MS_BETWEEN_TICKS {
            windsim.tick(wind_sources, &DeltaTime((MS_BETWEEN_TICKS / 1000) as f32));
        } else {
            windsim.ms_since_update += (dt.0 * 1000.0) as u32;
        }
    }
}
