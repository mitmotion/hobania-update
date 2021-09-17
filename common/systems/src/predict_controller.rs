use common::{
    comp::{Controller, RemoteController},
    resources::{Time, DeltaTime,},
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
            let i = _a.commands().len();
            let time = Duration::from_secs_f64(read_data.time.0);
            tracing::warn!(?i, ?time, "foo");
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
                *controller = r.clone()
            }
        }
    }
}
