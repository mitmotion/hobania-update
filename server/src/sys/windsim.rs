#![allow(dead_code)]
use crate::windsim::WindSim;
use common::{
    comp::{Pos, Vel},
    resources::DeltaTime,
};
use common_ecs::{Job, Origin, Phase, System};
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
        let wind_sources: Vec<(Pos, Vel)> = vec![(
            Pos(Vec3 {
                x: 100.0,
                y: 100.0,
                z: 1.0,
            }),
            Vel(Vec3 {
                x: 100.0,
                y: 100.0,
                z: 100.0,
            }),
        )];
        windsim.tick(wind_sources, &dt);
    }
}
