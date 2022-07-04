use common::weather::{CHUNKS_PER_CELL, WEATHER_DT};
use common_ecs::{dispatch, System};
use common_state::State;
use specs::DispatcherBuilder;
use std::time::Duration;

use crate::sys::SysScheduler;

mod sim;
mod sync;
mod tick;

pub fn add_server_systems(dispatch_builder: &mut DispatcherBuilder) {
    dispatch::<tick::Sys>(dispatch_builder, &[]);
    dispatch::<sync::Sys>(dispatch_builder, &[&tick::Sys::sys_name()]);
}

pub fn init(state: &mut State, world: &world::World) {
    // How many chunks wide a weather cell is.
    // 16 here means that a weather cell is 16x16 chunks.
    let weather_size = world.sim().get_size() / CHUNKS_PER_CELL;
    let sim = sim::WeatherSim::new(weather_size, world);
    state.ecs_mut().insert(sim);
    // Tick weather every 2 seconds
    state
        .ecs_mut()
        .insert(SysScheduler::<tick::Sys>::every(Duration::from_secs_f32(
            WEATHER_DT,
        )));
    state
        .ecs_mut()
        .insert(SysScheduler::<sync::Sys>::every(Duration::from_secs_f32(
            WEATHER_DT,
        )));
}
