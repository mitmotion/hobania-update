use crate::utils;
use approx::assert_relative_eq;
use common::{comp::Controller, resources::Time};
use specs::WorldExt;
use std::error::Error;
use utils::{DT, DTT};
use vek::{approx, Vec2, Vec3};
use veloren_common_systems::add_local_systems;

#[test]
fn simple_run() {
    let mut state = utils::setup();
    utils::create_player(&mut state);
    state.tick(
        DT,
        |dispatcher_builder| {
            add_local_systems(dispatcher_builder);
        },
        false,
    );
}

#[test]
fn emulate_dont_fall_outside_world() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    {
        let mut storage = state.ecs_mut().write_storage::<common::comp::Pos>();
        storage
            .insert(p1, common::comp::Pos(Vec3::new(1000.0, 1000.0, 265.0)))
            .unwrap();
    }

    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 1000.0);
    assert_relative_eq!(pos.0.y, 1000.0);
    assert_relative_eq!(pos.0.z, 265.0);
    assert_eq!(vel.0, vek::Vec3::zero());

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 1000.0);
    assert_relative_eq!(pos.0.y, 1000.0);
    assert_relative_eq!(pos.0.z, 265.0);
    assert_eq!(vel.0, vek::Vec3::zero());
    Ok(())
}

#[test]
fn emulate_fall() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.0);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(pos.0.z, 265.0);
    assert_eq!(vel.0, vek::Vec3::zero());

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.0);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(pos.0.z, 264.9975, epsilon = 0.001);
    assert_relative_eq!(vel.0.z, -0.25, epsilon = 0.001);

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.z, 264.9925, epsilon = 0.001);
    assert_relative_eq!(vel.0.z, -0.49969065, epsilon = 0.001);

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.z, 264.985, epsilon = 0.001);
    assert_relative_eq!(vel.0.z, -0.7493813, epsilon = 0.001);

    utils::tick(&mut state, DT * 7);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(state.ecs_mut().read_resource::<Time>().0, DTT * 10.0);
    assert_relative_eq!(pos.0.z, 264.8102, epsilon = 0.001);
    assert_relative_eq!(vel.0.z, -2.4969761, epsilon = 0.001);

    Ok(())
}

#[test]
fn emulate_fall_fast() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    utils::tick(&mut state, DT * 5);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.0);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(pos.0.z, 264.9375);
    assert_relative_eq!(vel.0.z, -1.25);

    utils::tick(&mut state, DT * 5);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(state.ecs_mut().read_resource::<Time>().0, DTT * 10.0);
    assert_relative_eq!(pos.0.z, 264.8126, epsilon = 0.001);
    assert_relative_eq!(vel.0.z, -2.4979765, epsilon = 0.001);

    Ok(())
}

#[test]
fn emulate_walk() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    for _ in 0..100 {
        utils::tick(&mut state, DT);
    }
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.z, 257.0);
    assert_eq!(vel.0, vek::Vec3::zero());

    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::set_control(&mut state, p1, actions)?;

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.09994);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(pos.0.z, 257.0);
    assert_relative_eq!(vel.0.x, 9.065385);
    assert_relative_eq!(vel.0.y, 0.0);
    assert_relative_eq!(vel.0.z, 0.0);

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.29037, epsilon = 0.001);
    assert_relative_eq!(vel.0.x, 17.273937, epsilon = 0.001);

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.562656, epsilon = 0.001);
    assert_relative_eq!(vel.0.x, 24.698797, epsilon = 0.001);

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.908909, epsilon = 0.001);
    assert_relative_eq!(vel.0.x, 31.40836, epsilon = 0.001);

    for _ in 0..7 {
        utils::tick(&mut state, DT);
    }
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 20.876211, epsilon = 0.001);
    assert_relative_eq!(vel.0.x, 63.128525, epsilon = 0.001);

    Ok(())
}

#[test]
fn emulate_walk_fast() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    for _ in 0..100 {
        utils::tick(&mut state, DT);
    }
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.z, 257.0);
    assert_eq!(vel.0, vek::Vec3::zero());

    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::set_control(&mut state, p1, actions)?;

    utils::tick(&mut state, DT);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.09994);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(pos.0.z, 257.0);
    assert_relative_eq!(vel.0.x, 9.065385);
    assert_relative_eq!(vel.0.y, 0.0);
    assert_relative_eq!(vel.0.z, 0.0);

    utils::tick(&mut state, DT * 2);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.679186, epsilon = 0.001);
    assert_relative_eq!(vel.0.x, 23.83063, epsilon = 0.001);
    Ok(())
}

#[test]
fn emulate_cant_run_during_fall() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::set_control(&mut state, p1, actions)?;

    utils::tick(&mut state, DT * 2);
    let (pos, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(pos.0.x, 16.0);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(vel.0.x, 0.0);
    assert_relative_eq!(vel.0.y, 0.0);

    utils::tick(&mut state, DT * 2);
    let (_, vel, _) = utils::get_pos(&mut state, p1)?;
    assert_relative_eq!(state.ecs_mut().read_resource::<Time>().0, DTT * 4.0);
    assert_relative_eq!(pos.0.x, 16.0);
    assert_relative_eq!(pos.0.y, 16.0);
    assert_relative_eq!(vel.0.x, 0.49995682, epsilon = 0.001);
    assert_relative_eq!(vel.0.y, 0.0, epsilon = 0.001);

    Ok(())
}
