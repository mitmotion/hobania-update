use crate::{Client, ClientType, ServerInfo};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use futures_util::future::FutureExt;
use network::{Network, Participant, Promises};
use std::{sync::Arc, time::Duration};
use tokio::{runtime::Runtime, select, sync::oneshot};
use tracing::{debug, error, trace, warn};

pub(crate) struct ServerInfoPacket {
    pub info: ServerInfo,
    pub time: f64,
}

pub(crate) type IncomingClient = Client;

pub(crate) struct ConnectionHandler {
    _network: Arc<Network>,
    thread_handle: Option<tokio::task::JoinHandle<()>>,
    pub client_receiver: Receiver<IncomingClient>,
    pub info_requester_receiver: Receiver<Sender<ServerInfoPacket>>,
    stop_sender: Option<oneshot::Sender<()>>,
}

/// Instead of waiting the main loop we are handling connections, especially
/// their slow network .await part on a different thread. We need to communicate
/// to the Server main thread sometimes though to get the current server_info
/// and time
impl ConnectionHandler {
    pub fn new(network: Network, runtime: &Runtime) -> Self {
        let network = Arc::new(network);
        let network_clone = Arc::clone(&network);
        let (stop_sender, stop_receiver) = oneshot::channel();

        let (client_sender, client_receiver) = unbounded::<IncomingClient>();
        let (info_requester_sender, info_requester_receiver) =
            bounded::<Sender<ServerInfoPacket>>(1);

        let thread_handle = Some(runtime.spawn(Self::work(
            network_clone,
            client_sender,
            info_requester_sender,
            stop_receiver,
        )));

        Self {
            _network: network,
            thread_handle,
            client_receiver,
            info_requester_receiver,
            stop_sender: Some(stop_sender),
        }
    }

    async fn work(
        network: Arc<Network>,
        client_sender: Sender<IncomingClient>,
        info_requester_sender: Sender<Sender<ServerInfoPacket>>,
        stop_receiver: oneshot::Receiver<()>,
    ) {
        let mut stop_receiver = stop_receiver.fuse();
        loop {
            let participant = match select!(
                _ = &mut stop_receiver => None,
                p = network.connected().fuse() => Some(p),
            ) {
                None => break,
                Some(Ok(p)) => p,
                Some(Err(e)) => {
                    error!(
                        ?e,
                        "Stopping Conection Handler, no new connections can be made to server now!"
                    );
                    break;
                },
            };

            let client_sender = client_sender.clone();
            let info_requester_sender = info_requester_sender.clone();

            match select!(
                _ = &mut stop_receiver => None,
                e = Self::init_participant(participant, client_sender, info_requester_sender).fuse() => Some(e),
            ) {
                None => break,
                Some(Ok(())) => (),
                Some(Err(e)) => warn!(?e, "drop new participant, because an error occurred"),
            }
        }
    }

    /// This code just generates some messages so that the client can assume
    /// deserialization This is completely random chosen with the goal to
    /// cover as much as possible and to stop xMAC94x being annoyed in
    /// discord by people that just have a old version
    fn dump_messages_so_client_can_validate_itself(
        mut validate_stream: network::Stream,
    ) -> Result<(), network::StreamError> {
        use common::{
            character::{Character, CharacterItem},
            comp,
            comp::{
                humanoid,
                inventory::Inventory,
                item::{tool::ToolKind, Reagent},
                skills::SkillGroupKind,
            },
            outcome::Outcome,
            terrain::{biome::BiomeKind, Block, BlockKind, TerrainChunk, TerrainChunkMeta},
            trade::{Good, PendingTrade, Trades},
            uid::Uid,
        };
        use common_net::{
            msg::{world_msg::EconomyInfo, EcsCompPacket, ServerGeneral},
            sync::CompSyncPackage,
        };
        use rand::SeedableRng;
        use std::collections::HashMap;
        use vek::*;

        // 1. Simulate Character
        let mut rng = rand::rngs::SmallRng::from_seed([42; 32]);
        let item = CharacterItem {
            character: Character {
                id: Some(1337),
                alias: "foobar".to_owned(),
            },
            body: comp::Body::Humanoid(humanoid::Body::random_with(
                &mut rng,
                &humanoid::Species::Undead,
            )),
            inventory: Inventory::new_empty(),
        };
        validate_stream.send(ServerGeneral::CharacterListUpdate(vec![item]))?;

        // 2. Simulate Outcomes
        let item1 = Outcome::Explosion {
            pos: Vec3::new(1.0, 2.0, 3.0),
            power: 4.0,
            radius: 5.0,
            is_attack: true,
            reagent: Some(Reagent::Blue),
        };
        let item2 = Outcome::SkillPointGain {
            uid: Uid::from(1337u64),
            pos: Vec3::new(2.0, 4.0, 6.0),
            skill_tree: SkillGroupKind::Weapon(ToolKind::Empty),
            total_points: 99,
        };
        let item3 = Outcome::BreakBlock {
            pos: Vec3::new(1, 2, 3),
            color: Some(Rgb::new(0u8, 8u8, 13u8)),
        };
        validate_stream.send(ServerGeneral::Outcomes(vec![item1, item2, item3]))?;

        // 3. Simulate Terrain
        let item = TerrainChunk::new(
            5,
            Block::new(BlockKind::Water, Rgb::zero()),
            Block::new(BlockKind::Air, Rgb::zero()),
            TerrainChunkMeta::void(),
        );
        validate_stream.send(ServerGeneral::TerrainChunkUpdate {
            key: Vec2::new(42, 1337),
            chunk: Ok(Box::new(item)),
        })?;

        // 4. Simulate Componity Sync
        let mut item = CompSyncPackage::<EcsCompPacket>::new();
        let uid = Uid::from(70);
        item.comp_inserted(uid, comp::Pos(Vec3::new(42.1337, 0.0, 0.0)));
        item.comp_inserted(uid, comp::Vel(Vec3::new(0.0, 42.1337, 0.0)));
        validate_stream.send(ServerGeneral::CompSync(item))?;

        // 5. Pending Trade
        let uid = Uid::from(70);
        validate_stream.send(ServerGeneral::UpdatePendingTrade(
            Trades::default().begin_trade(uid, uid),
            PendingTrade::new(uid, Uid::from(71)),
        ))?;

        // 6. Economy Info
        validate_stream.send(ServerGeneral::SiteEconomy(EconomyInfo {
            id: 99,
            population: 55,
            stock: vec![
                (Good::Wood, 50.0),
                (Good::Tools, 33.3),
                (Good::Coin, 9000.1),
            ]
            .into_iter()
            .collect::<HashMap<Good, f32>>(),
            labor_values: HashMap::new(),
            values: vec![
                (Good::RoadSecurity, 1.0),
                (Good::Terrain(BiomeKind::Forest), 1.0),
            ]
            .into_iter()
            .collect::<HashMap<Good, f32>>(),
            labors: Vec::new(),
            last_exports: HashMap::new(),
            resources: HashMap::new(),
        }))?;
        Ok(())
    }

    async fn init_participant(
        participant: Participant,
        client_sender: Sender<IncomingClient>,
        info_requester_sender: Sender<Sender<ServerInfoPacket>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        debug!("New Participant connected to the server");
        let (sender, receiver) = bounded(1);
        info_requester_sender.send(sender)?;

        let reliable = Promises::ORDERED | Promises::CONSISTENCY;
        let reliablec = reliable | Promises::COMPRESSED;

        let general_stream = participant.open(3, reliablec, 500).await?;
        let ping_stream = participant.open(2, reliable, 500).await?;
        let mut register_stream = participant.open(3, reliablec, 500).await?;
        let character_screen_stream = participant.open(3, reliablec, 500).await?;
        let in_game_stream = participant.open(3, reliablec, 100_000).await?;
        let terrain_stream = participant.open(4, reliablec, 20_000).await?;
        let validate_stream = participant.open(4, reliablec, 1_000_000).await?;

        let server_data = receiver.recv()?;

        register_stream.send(server_data.info)?;

        const TIMEOUT: Duration = Duration::from_secs(5);
        let client_type = match select!(
            _ = tokio::time::sleep(TIMEOUT).fuse() => None,
            t = register_stream.recv::<ClientType>().fuse() => Some(t),
        ) {
            None => {
                debug!("Timeout for incoming client elapsed, aborting connection");
                return Ok(());
            },
            Some(client_type) => client_type?,
        };

        if let Err(e) = Self::dump_messages_so_client_can_validate_itself(validate_stream) {
            trace!(?e, ?client_type, "a client dropped as he failed validation");
            return Err(e.into());
        };

        let client = Client::new(
            client_type,
            participant,
            server_data.time,
            general_stream,
            ping_stream,
            register_stream,
            character_screen_stream,
            in_game_stream,
            terrain_stream,
        );

        client_sender.send(client)?;
        Ok(())
    }
}

impl Drop for ConnectionHandler {
    fn drop(&mut self) {
        let _ = self.stop_sender.take().unwrap().send(());
        trace!("aborting ConnectionHandler");
        self.thread_handle.take().unwrap().abort();
        trace!("aborted ConnectionHandler!");
    }
}
