pub mod agent;
pub mod chunk_send;
pub mod chunk_serialize;
pub mod entity_sync;
pub mod invite_timeout;
pub mod loot;
pub mod metrics;
pub mod msg;
pub mod object;
pub mod persistence;
pub mod pets;
pub mod sentinel;
pub mod subscription;
pub mod terrain;
pub mod terrain_sync;
pub mod waypoint;
pub mod wiring;

use common_base::span;
use common_ecs::{dispatch, run_now, System};
use common_state::State;
use common_systems::{melee, projectile};
use specs::{DispatcherBuilder, WorldExt};
use std::{
    marker::PhantomData,
    sync::Arc,
    time::{Duration, Instant},
};

pub type PersistenceScheduler = SysScheduler<persistence::Sys>;

pub fn add_server_systems(dispatch_builder: &mut DispatcherBuilder) {
    dispatch::<melee::Sys>(dispatch_builder, &[&projectile::Sys::sys_name()]);
    //Note: server should not depend on interpolation system
    dispatch::<agent::Sys>(dispatch_builder, &[]);
    dispatch::<terrain::Sys>(dispatch_builder, &[&msg::terrain::Sys::sys_name()]);
    dispatch::<waypoint::Sys>(dispatch_builder, &[]);
    dispatch::<invite_timeout::Sys>(dispatch_builder, &[]);
    dispatch::<persistence::Sys>(dispatch_builder, &[]);
    dispatch::<object::Sys>(dispatch_builder, &[]);
    dispatch::<wiring::Sys>(dispatch_builder, &[]);
    // no dependency, as we only work once per sec anyway.
    dispatch::<chunk_serialize::Sys>(dispatch_builder, &[]);
    // don't depend on chunk_serialize, as we assume everything is done in a SlowJow
    dispatch::<chunk_send::Sys>(dispatch_builder, &[]);
}

pub fn run_sync_systems(state: &mut State) {
    span!(_guard, "run_sync_systems");

    // Create and run a dispatcher for ecs systems that synchronize state.
    span!(guard, "create dispatcher");
    let mut dispatch_builder =
        DispatcherBuilder::new().with_pool(Arc::clone(&state.thread_pool()));

    // Setup for entity sync
    dispatch::<sentinel::Sys>(&mut dispatch_builder, &[]);
    dispatch::<subscription::Sys>(&mut dispatch_builder, &[]);

    // Sync
    dispatch::<terrain_sync::Sys>(&mut dispatch_builder, &[]);
    dispatch::<entity_sync::Sys>(&mut dispatch_builder, &[&sentinel::Sys::sys_name(), &subscription::Sys::sys_name()]);

    // This dispatches all the systems in parallel.
    let mut dispatcher = dispatch_builder.build();
    drop(guard);

    let ecs = state.ecs_mut();
    span!(guard, "run systems");
    dispatcher.dispatch(ecs);
    drop(guard);

    span!(guard, "maintain ecs");
    ecs.maintain();
    drop(guard);
}

/// Used to schedule systems to run at an interval
pub struct SysScheduler<S> {
    interval: Duration,
    last_run: Instant,
    _phantom: PhantomData<S>,
}

impl<S> SysScheduler<S> {
    pub fn every(interval: Duration) -> Self {
        Self {
            interval,
            last_run: Instant::now(),
            _phantom: PhantomData,
        }
    }

    pub fn should_run(&mut self) -> bool {
        if self.last_run.elapsed() > self.interval {
            self.last_run = Instant::now();

            true
        } else {
            false
        }
    }
}

impl<S> Default for SysScheduler<S> {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            last_run: Instant::now(),
            _phantom: PhantomData,
        }
    }
}
