use veloren_plugin_rt::{
    api::{event, GameMode, *},
    *,
};

#[event_handler]
pub fn on_load(game: &Game, init: event::Init) {
    match init.mode() {
        GameMode::Server => log!("Hello, server!"),
        GameMode::Client => log!("Hello, client!"),
        GameMode::Singleplayer => log!("Hello, singleplayer!"),
    }
}

#[event_handler]
pub fn on_command_test(game: &Game, cmd: event::Command) -> Result<Vec<String>, String> {
    Ok(vec![
        format!(
            "Entity with uid {:?} named {} with {:?} sent command with args {:?}",
            cmd.entity().uid(),
            cmd.entity().get_name(),
            cmd.entity().get_health(),
            cmd.args(),
        )
        .into(),
    ])
}

#[global_state]
#[derive(Default)]
struct State {
    total_joined: u64,
}

#[event_handler]
pub fn on_join(
    game: &Game,
    player_join: event::PlayerJoin,
    state: &mut State,
) -> event::PlayerJoinResponse<'static> {
    state.total_joined += 1;
    if state.total_joined > 10 {
        event::PlayerJoinResponse::Reject {
            reason: "Too many people have joined!".into(),
        }
    } else {
        event::PlayerJoinResponse::Accept
    }
}
