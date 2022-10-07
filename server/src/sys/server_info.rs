use crate::{client::Client, query_server::ServerInfoResponse, Settings};
use common_ecs::{Job, Origin, Phase, System};
use specs::{Read, ReadStorage, WriteExpect};
use tokio::sync::mpsc::UnboundedReceiver;
use tracing::error;

/// Processes server info requests from the Query Server
#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    type SystemData = (
        ReadStorage<'a, Client>,
        Read<'a, Settings>,
        WriteExpect<'a, UnboundedReceiver<ServerInfoRequest>>,
    );

    const NAME: &'static str = "server_info";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(_job: &mut Job<Self>, (clients, settings, mut receiver): Self::SystemData) {
        let players_current = clients.count() as u16;
        let server_info = ServerInfoResponse {
            players_current,
            players_max: settings.max_players,
            git_hash: common::util::GIT_HASH.to_owned(),
            battle_mode: settings.gameplay.battle_mode.into(),
        };

        while let Ok(request) = receiver.try_recv() {
            if let Err(e) = request.response_sender.send(server_info.clone()) {
                error!(?e, "Failed to process System Info request!");
            }
        }
    }
}

pub struct ServerInfoRequest {
    pub response_sender: tokio::sync::oneshot::Sender<ServerInfoResponse>,
}
