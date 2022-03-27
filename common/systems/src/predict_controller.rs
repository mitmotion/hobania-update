use common::{
    comp::{Controller, RemoteController},
    resources::{DeltaTime, Time},
};
use common_ecs::{Job, Origin, Phase, System};
use specs::{
    shred::ResourceId, Entities, Join, Read, ReadStorage, SystemData, World, WriteStorage,
};
use std::time::Duration;

#[derive(SystemData)]
pub struct ReadData<'a> {
    time: Read<'a, Time>,
    dt: Read<'a, DeltaTime>,
    entities: Entities<'a>,
    remote_controllers: ReadStorage<'a, RemoteController>,
}

#[derive(Default)]
pub struct Sys;

impl<'a> System<'a> for Sys {
    type SystemData = (ReadData<'a>, WriteStorage<'a, Controller>);

    const NAME: &'static str = "predict_controller";
    const ORIGIN: Origin = Origin::Common;
    const PHASE: Phase = Phase::Create;

    fn run(_job: &mut Job<Self>, (read_data, mut controllers): Self::SystemData) {
        for (entity, _a) in (&read_data.entities, &read_data.remote_controllers).join() {
            let command_len = _a.commands().len();
            common_base::plot!("predict_available", command_len as f64);
            common_base::plot!("predict_time", read_data.time.0);
            let first = _a
                .commands()
                .front()
                .map_or(read_data.time.0 - 1.0, |c| c.source_time().as_secs_f64());
            let last = _a
                .commands()
                .back()
                .map_or(read_data.time.0 - 1.0, |c| c.source_time().as_secs_f64());
            common_base::plot!("predict_first", first - read_data.time.0);
            common_base::plot!("predict_last", last - read_data.time.0);

            let _ = controllers
                .entry(entity)
                .map(|e| e.or_insert_with(Default::default));
        }

        for (_, remote_controller, controller) in (
            &read_data.entities,
            &read_data.remote_controllers,
            &mut controllers,
        )
            .join()
        {
            let dt = Duration::from_secs_f32(read_data.dt.0);
            let time = Duration::from_secs_f64(read_data.time.0);
            // grab the most fitting entry;
            let r = remote_controller.compress(time, dt);
            // Do nothing when already populated and we dont have a new value
            if let Some(r) = r {
                *controller = r.clone();
                let mut action = 0.0;
                for a in r.actions {
                    if let common::comp::ControlAction::StartInput {
                        input,
                        target_entity,
                        select_pos,
                    } = a
                    {
                        if input == common::comp::InputKind::Jump {
                            action = 1.0;
                        }
                    }
                }
                common_base::plot!("action_contains_jump", action);
            }
        }
    }
}
