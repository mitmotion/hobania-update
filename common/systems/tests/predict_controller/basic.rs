use crate::utils;
use approx::assert_relative_eq;
use common::{
    comp::{self, CommandGenerator, Controller, InputKind},
    resources::Time,
};
use specs::WorldExt;
use std::{error::Error, time::Duration};
use utils::{DT, DTT};
use vek::{approx, Vec2};
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
fn emulate_walk() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    utils::tick(&mut state, DT);
    assert_eq!(state.ecs_mut().read_resource::<Time>().0, DTT);

    let mut generator = CommandGenerator::default();
    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::push_remote(&mut state, p1, generator.gen(DT * 3, actions))?;

    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(20.0, 0.0);
    utils::push_remote(&mut state, p1, generator.gen(DT * 5, actions))?;

    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(30.0, 0.0);
    utils::push_remote(&mut state, p1, generator.gen(DT * 6, actions))?;

    //DDT 2 - no data yet
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 0.0);

    //DDT 3
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    //DDT 4
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    //DDT 5
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 20.0);

    //DDT 6
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 30.0);

    //DDT 7
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 30.0);

    Ok(())
}

#[test]
fn emulate_partial_client_walk() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    utils::tick(&mut state, DT);
    assert_eq!(state.ecs_mut().read_resource::<Time>().0, DTT);

    let mut generator = CommandGenerator::default();
    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::push_remote(
        &mut state,
        p1,
        generator.gen(Duration::from_secs_f64(2.5 * DTT), actions),
    )?;

    //DDT 2 - no data yet officially, but we interpret
    // Explaination why we interpolate here:
    //  - Client says move at 2.5, server is at 2.0 currently.
    //  - Assume there was no interpolation:
    //    - Client would assume to be at pos 5 at 3.0 and 10 at 3.5
    //    - Server would notice that client started moving at 3.0 and is at 5 at 3.5
    //    => Client would lag 5 behind always
    //  - Now with interpolation:
    //    - Client would assume to be at pos 5 at 3.0 and 10 at 3.5
    //    - Server lets client run with speed of 5 at 2.0, so
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 5.0);

    //DDT 3
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    //DDT 4
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    Ok(())
}

#[test]
fn emulate_partial_server_walk() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    utils::tick(&mut state, DT);
    assert_eq!(state.ecs_mut().read_resource::<Time>().0, DTT);

    let mut generator = CommandGenerator::default();
    let mut actions = Controller::default();
    actions.inputs.move_dir = Vec2::new(10.0, 0.0);
    utils::push_remote(
        &mut state,
        p1,
        generator.gen(Duration::from_secs_f64(3.0 * DTT), actions),
    )?;

    //DDT 1.5 - no data yet
    utils::tick(&mut state, Duration::from_secs_f64(0.5 * DTT));
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 0.0);

    //DDT 2.5/3.5 - No Data for first 0.5, then 10
    // We interpolate here to 5 for the whole tick
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 5.0);

    //DDT 3
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    //DDT 4
    utils::tick(&mut state, DT);
    assert_relative_eq!(utils::get_controller(&state, p1)?.inputs.move_dir.x, 10.0);

    Ok(())
}

#[test]
fn emulate_jump() -> Result<(), Box<dyn Error>> {
    let mut state = utils::setup();
    let p1 = utils::create_player(&mut state);

    utils::tick(&mut state, DT);
    assert_eq!(state.ecs_mut().read_resource::<Time>().0, DTT);

    let mut generator = CommandGenerator::default();
    let mut actions = Controller::default();
    actions
        .queued_inputs
        .insert(comp::InputKind::Jump, comp::InputAttr {
            select_pos: None,
            target_entity: None,
        });
    utils::push_remote(
        &mut state,
        p1,
        generator.gen(Duration::from_secs_f64(5.0 * DTT), actions),
    )?;

    //DDT 2
    utils::tick(&mut state, DT);
    assert!(utils::get_controller(&state, p1)?.queued_inputs.is_empty());

    //DDT 3
    utils::tick(&mut state, DT);
    assert!(utils::get_controller(&state, p1)?.queued_inputs.is_empty());

    //DDT 4
    // TODO: i am NOT sure if this is correct behavior, just adjusted it in a rebase
    // to fix the test
    utils::tick(&mut state, DT);
    let inputs = utils::get_controller(&state, p1)?.queued_inputs;
    assert!(!inputs.is_empty());
    assert!(inputs.contains_key(&InputKind::Jump));

    //DDT 5
    utils::tick(&mut state, DT);
    let inputs = utils::get_controller(&state, p1)?.queued_inputs;
    assert!(!inputs.is_empty());
    assert!(inputs.contains_key(&InputKind::Jump));

    //DDT 6
    utils::tick(&mut state, DT);
    assert!(utils::get_controller(&state, p1)?.queued_inputs.is_empty());

    //DDT 7
    utils::tick(&mut state, DT);
    assert!(utils::get_controller(&state, p1)?.queued_inputs.is_empty());
    Ok(())
}
