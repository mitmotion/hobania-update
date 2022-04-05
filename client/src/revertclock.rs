use common::{
    comp::{RemoteController, Vel},
    resources::{DeltaTime, MonotonicTime, PlayerEntity, Time, TimeOfDay},
};
use common_state::State;
use common_systems::add_rewind_systems;
use specs::WorldExt;
use std::time::Duration;
use vek::{ops::Clamp, Vec3};

#[derive(Default)]
pub(crate) struct RevertClock {
    reverted_time: Option<Duration>,
    time_syncs: usize,
}

impl RevertClock {
    pub(crate) fn reset(&mut self) {
        self.reverted_time = None;
        self.time_syncs = 0;
    }

    // The server might need up to 33ms to calcualte its tick. It will send this
    // time with it, so that we can substract it from
    pub(crate) fn sync(
        &mut self,
        old_client_time: f64,
        new_server_time: Time,
        _inter_tick_offset: Duration,
    ) {
        self.time_syncs += 1;
        // Even with a stable network, expect time to oscillate around the actual time
        // by SERVER_TICK (33.3ms)
        let diff = old_client_time - new_server_time.0;
        if diff > 0.0 {
            tracing::warn!(?old_client_time, ?diff, "Time was reverted by server");
            let rewind_time =
                self.reverted_time.unwrap_or_default() + Duration::from_secs_f64(diff);
            self.reverted_time = Some(rewind_time);
        } else {
            tracing::warn!(?old_client_time, ?diff, "Time was advanced by server");
        }
    }

    /// Rewind local changes after the server send some old state
    fn rewind_tick(simulate_ahead: Duration, state: &mut State) {
        common_base::plot!("simulate_ahead", simulate_ahead.as_secs_f64());

        // store changes
        let time_of_day = state.ecs().read_resource::<TimeOfDay>().0;
        let monotonic_time = state.ecs().read_resource::<MonotonicTime>().0;
        let delta_time = state.ecs().read_resource::<DeltaTime>().0;

        const MAX_INCREMENTS: usize = 100; // The maximum number of collision tests per tick
        const STEP_SEC: f64 = 0.04;
        let increments =
            ((simulate_ahead.as_secs_f64() / STEP_SEC).ceil() as usize).clamped(1, MAX_INCREMENTS);
        for _ in 0..increments {
            let partial = simulate_ahead / (increments as u32);
            state.tick(
                partial,
                |dispatch_builder| {
                    add_rewind_systems(dispatch_builder);
                },
                false,
            );
        }

        // rewind changes
        state.ecs().write_resource::<TimeOfDay>().0 = time_of_day;
        state.ecs().write_resource::<MonotonicTime>().0 = monotonic_time;
        state.ecs().write_resource::<DeltaTime>().0 = delta_time;
    }

    pub(crate) fn tick(&mut self, state: &mut State, dt: Duration) {
        let entity = state
            .ecs()
            .read_resource::<PlayerEntity>()
            .0
            .expect("Client::entity should always have PlayerEntity be Some");

        common_base::plot!("recived_time_sync", self.time_syncs as f64);

        let rewind_time = match self.reverted_time {
            Some(rewind_time) => rewind_time,
            None => return,
        };

        let simulate_ahead = state
            .ecs()
            .read_storage::<RemoteController>()
            .get(entity)
            .map(|rc| rc.simulate_ahead())
            .unwrap_or_default();
        // We substract `dt` here, as otherwise we
        // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms,
        // rewind_tick=130ms end_tick=100+130+60=290
        // Tick2: server_time=130ms dt=30ms ping=30 simulate_ahead=130ms,
        // rewind_tick=130ms end_tick=130+130+30=290
        // Tick3: server_time=160ms dt=60ms ping=30 simulate_ahead=130ms,
        // rewind_tick=130ms end_tick=160+130+60=350 with dt substraction
        // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms,
        // rewind_tick=70ms end_tick=100+70+60=230 Tick2: server_time=130ms
        // dt=30ms ping=30 simulate_ahead=130ms, rewind_tick=100ms
        // end_tick=130+100+30=260 Tick3: server_time=160ms dt=60ms ping=30
        // simulate_ahead=130ms, rewind_tick=70ms end_tick=160+70+60=290
        let simulate_ahead = simulate_ahead.max(dt) - dt;
        let simulate_ahead = simulate_ahead + Duration::from_secs_f64(1.0 / 30.0);
        /*
        let x = if rewind_time > simulate_ahead + Duration::from_millis(25) {
            common_base::plot!("xxx", 1.0);
            common_base::plot!("gggggg", simulate_ahead.as_secs_f64() - rewind_time.as_secs_f64() );
            simulate_ahead + Duration::from_millis(15)
        } else if rewind_time < simulate_ahead - Duration::from_millis(25) {
            common_base::plot!("xxx", -1.0);
            simulate_ahead - Duration::from_millis(15)
        } else {
            common_base::plot!("xxx", 0.0);
            simulate_ahead
        };
         */
        {
            let last_tick_time = state.ecs().read_resource::<Time>().0 as f64;
            common_base::plot!("tick_before", last_tick_time);

            let vel = state
                .ecs()
                .read_storage::<Vel>()
                .get(entity)
                .cloned()
                .unwrap_or(Vel(Vec3::zero()));

            common_base::plot!("vel_x_before1", vel.0.x as f64);
            common_base::plot!("vel_y_before1", vel.0.y as f64);
            common_base::plot!("vel_z_before1", vel.0.z as f64);
            let pos = state
                .ecs()
                .read_storage::<common::comp::Pos>()
                .get(entity)
                .cloned()
                .unwrap_or(common::comp::Pos(Vec3::zero()));

            common_base::plot!("pos_x_before1", pos.0.x as f64);
            common_base::plot!("pos_y_before1", pos.0.y as f64);
            common_base::plot!("pos_z_before1", pos.0.z as f64);
        }

        common_base::plot!("rewind_time", rewind_time.as_secs_f64());
        RevertClock::rewind_tick(simulate_ahead, state);

        {
            let time = state.ecs().read_resource::<Time>().0 as f64;
            common_base::plot!("tick_afterwards1", time);
            let vel = state
                .ecs()
                .read_storage::<Vel>()
                .get(entity)
                .cloned()
                .unwrap_or(Vel(Vec3::zero()));

            common_base::plot!("vel_x_after1", vel.0.x as f64);
            common_base::plot!("vel_y_after1", vel.0.y as f64);
            common_base::plot!("vel_z_after1", vel.0.z as f64);
            let pos = state
                .ecs()
                .read_storage::<common::comp::Pos>()
                .get(entity)
                .cloned()
                .unwrap_or(common::comp::Pos(Vec3::zero()));

            common_base::plot!("pos_x_after1", pos.0.x as f64);
            common_base::plot!("pos_y_after1", pos.0.y as f64);
            common_base::plot!("pos_z_after1", pos.0.z as f64);
        }

        /*
        let mut rewind_time = Duration::from_secs_f64(0.0);
        const WARP_PERCENT: f64 = 0.05; // make sure we end up not further than 5% from the estimated tick
        let mut target_smooth_dt = dt.as_secs_f64();
        common_base::plot!("recived_time_sync", 0.0);
        if let Some(reverted_time) = self.inter_tick_reverted_time {
            // At smooth_time we wouldn't notice any rewind
            let smooth_time = reverted_time + dt;
            common_base::plot!("recived_time_sync", 1.0);
            common_base::plot!("reverted_time", reverted_time.as_secs_f64());
            let time = self.state.ecs().read_resource::<Time>().0 as f64;
            let simulate_ahead = self
                .state
                .ecs()
                .read_storage::<RemoteController>()
                .get(entity)
                .map(|rc| rc.simulate_ahead())
                .unwrap_or_default();
            // We substract `dt` here, as otherwise we
            // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms, rewind_tick=130ms end_tick=100+130+60=290
            // Tick2: server_time=130ms dt=30ms ping=30 simulate_ahead=130ms, rewind_tick=130ms end_tick=130+130+30=290
            // Tick3: server_time=160ms dt=60ms ping=30 simulate_ahead=130ms, rewind_tick=130ms end_tick=160+130+60=350
            // with dt substraction
            // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms, rewind_tick=70ms end_tick=100+70+60=230
            // Tick2: server_time=130ms dt=30ms ping=30 simulate_ahead=130ms, rewind_tick=100ms end_tick=130+100+30=260
            // Tick3: server_time=160ms dt=60ms ping=30 simulate_ahead=130ms, rewind_tick=70ms end_tick=160+70+60=290
            let simulate_ahead = simulate_ahead.max(dt) - dt;
            // measurements lead to the effect that smooth_diff is == 0.0 when we add 2 server ticks here.
            let simulate_ahead = simulate_ahead + Duration::from_secs_f64(1.0 / 30.0);
            rewind_time = simulate_ahead;

            // Simulate_ahead still fluctionates because Server Tick != Client Tick, and we cant
            // control the phase in which the sync happens.
            // In order to dampen it, we calculate the target_smooth_dt and make sure to not derive
            // to much from it

            ERROR, muss man das hier swappen ?
            target_smooth_dt = reverted_time.as_secs_f64() - simulate_ahead.as_secs_f64();
            // we store it, and will apply it over the course of the next ticks
            self.rewind_fluctuation_budget = target_smooth_dt;
            common_base::plot!("new_rewind_fluctuation_budget", self.rewind_fluctuation_budget);
        }
        common_base::plot!("rewind_time", rewind_time.as_secs_f64());


        // we need to subtract the normal dt as we have a separate tick call for it.
        // make it positive as we wont allow direct go in past
        let corrected_target_smooth_dt = (target_smooth_dt - dt.as_secs_f64()).max(0.0);
        common_base::plot!("corrected_target_smooth_dt", corrected_target_smooth_dt);

        // apply our budget to rewind_time
        let mut smooth_rewind_time = rewind_time.as_secs_f64();
        smooth_rewind_time += self.rewind_fluctuation_budget;
        let xxx =  smooth_rewind_time.clamp( corrected_target_smooth_dt * (1.0-WARP_PERCENT), corrected_target_smooth_dt * (1.0+WARP_PERCENT));
        common_base::plot!("xxx", xxx);
        smooth_rewind_time = xxx;
        self.rewind_fluctuation_budget -= smooth_rewind_time - rewind_time.as_secs_f64();

        common_base::plot!("rewind_fluctuation_budget", self.rewind_fluctuation_budget);
        common_base::plot!("target_smooth_dt", target_smooth_dt);


        tracing::warn!(?smooth_rewind_time, ?dt, "simulating ahead again");
        self.state.rewind_tick(
            Duration::from_secs_f64(smooth_rewind_time),
            |dispatch_builder| {
                add_rewind_systems(dispatch_builder);
            },
            false,
        );
         */
    }
}
