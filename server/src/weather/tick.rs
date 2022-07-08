use common::{resources::TimeOfDay, weather::WeatherGrid};
use common_ecs::{Origin, Phase, System};
use specs::{Read, Write, WriteExpect};

use crate::sys::SysScheduler;

use super::sim::WeatherSim;

#[derive(Default)]
pub struct Sys;

impl<'a> System<'a> for Sys {
    type SystemData = (
        Read<'a, TimeOfDay>,
        WriteExpect<'a, WeatherSim>,
        WriteExpect<'a, WeatherGrid>,
        Write<'a, SysScheduler<Self>>,
    );

    const NAME: &'static str = "weather::tick";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(
        _job: &mut common_ecs::Job<Self>,
        (game_time, mut sim, mut grid, mut scheduler): Self::SystemData,
    ) {
        if scheduler.should_run() {
            if grid.size() != sim.size() {
                *grid = WeatherGrid::new(sim.size());
            }
            sim.tick(&game_time, &mut grid);
        }
    }
}
