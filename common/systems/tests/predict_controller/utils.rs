use common::{
    comp::{
        inventory::item::MaterialStatManifest, ControlCommand, ControlCommands, Controller,
        RemoteController,
    },
    resources::{DeltaTime, GameMode, Time},
};
use common_ecs::dispatch;
use common_net::sync::WorldSyncExt;
use common_state::State;
use hashbrown::HashSet;
use specs::{Builder, Entity, WorldExt};
use std::{error::Error, time::Duration};
use veloren_common_systems::predict_controller;

pub const DTT: f64 = 1.0 / 10.0;
pub const DT: Duration = Duration::from_millis(100);

pub fn setup() -> State {
    let mut state = State::new(GameMode::Server);
    state
        .ecs_mut()
        .insert(MaterialStatManifest(hashbrown::HashMap::default()));
    state.ecs_mut().read_resource::<Time>();
    state.ecs_mut().read_resource::<DeltaTime>();

    state
}

pub fn tick(state: &mut State, dt: Duration) {
    state.tick(
        dt,
        |dispatch_builder| {
            dispatch::<predict_controller::Sys>(dispatch_builder, &[]);
        },
        false,
    );
}

pub fn push_remote(
    state: &mut State,
    entity: Entity,
    command: ControlCommand,
) -> Result<u64, Box<dyn Error>> {
    let mut storage = state.ecs_mut().write_storage::<RemoteController>();
    let remote_controller = storage.get_mut(entity).ok_or(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        "Storage does not contain Entity RemoteController",
    )))?;
    remote_controller
        .push(command)
        .ok_or(Box::new(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Command couldn't be pushed",
        )))
}

pub fn get_controller(state: &State, entity: Entity) -> Result<Controller, Box<dyn Error>> {
    let storage = state.ecs().read_storage::<Controller>();
    let controller = storage.get(entity).ok_or(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        "Storage does not contain Entity Controller",
    )))?;
    Ok(controller.clone())
}

#[allow(dead_code)]
pub fn push_remotes(state: &mut State, entity: Entity, commands: ControlCommands) -> HashSet<u64> {
    let mut storage = state.ecs_mut().write_storage::<RemoteController>();
    let remote_controller = storage.get_mut(entity).unwrap();
    remote_controller.append(commands)
}

pub fn create_player(state: &mut State) -> Entity {
    let remote_controller = RemoteController::default();
    let controller = Controller::default();

    state
        .ecs_mut()
        .create_entity_synced()
        .with(remote_controller)
        .with(controller)
        .build()
}
