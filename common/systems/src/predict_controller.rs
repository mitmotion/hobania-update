use common::comp::{Controller, RemoteController};
use common_ecs::{Job, Origin, Phase, System};
use specs::{shred::ResourceId, Entities, Join, ReadStorage, SystemData, World, WriteStorage};
use std::time::Duration;

#[derive(SystemData)]
pub struct ReadData<'a> {
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
            tracing::warn!(?i, "foo");
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
            // grab the most fitting entry
            let time = Duration::from_millis(1000);
            let r = remote_controller.current(time).map(|c| c.msg());
            // Do nothing when already populated and we dont have a new value
            if let Some(r) = r {
                *controller = r.clone()
            }
        }
    }
}
