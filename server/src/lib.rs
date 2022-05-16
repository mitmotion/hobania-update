#![deny(unsafe_code)]
#![allow(clippy::option_map_unit_fn)]
#![deny(clippy::clone_on_ref_ptr)]
#![feature(
    box_patterns,
    label_break_value,
    bool_to_option,
    drain_filter,
    never_type,
    option_zip,
    unwrap_infallible,
    let_else
)]
#![cfg_attr(not(feature = "worldgen"), feature(const_panic))]

pub mod alias_validator;
mod character_creator;
pub mod chunk_generator;
mod chunk_serialize;
pub mod client;
pub mod cmd;
pub mod connection_handler;
mod data_dir;
pub mod error;
pub mod events;
pub mod input;
pub mod location;
pub mod lod;
pub mod login_provider;
pub mod metrics;
pub mod persistence;
mod pet;
pub mod presence;
pub mod rtsim;
pub mod settings;
pub mod state_ext;
pub mod sys;
#[cfg(feature = "persistent_world")]
pub mod terrain_persistence;
#[cfg(not(feature = "worldgen"))] mod test_world;
pub mod wiring;

// Reexports
pub use crate::{
    data_dir::DEFAULT_DATA_DIR_NAME,
    error::Error,
    events::Event,
    input::Input,
    settings::{CalendarMode, EditableSettings, Settings},
};

#[cfg(feature = "persistent_world")]
use crate::terrain_persistence::TerrainPersistence;
use crate::{
    alias_validator::AliasValidator,
    chunk_generator::ChunkGenerator,
    client::Client,
    cmd::ChatCommandExt,
    connection_handler::ConnectionHandler,
    data_dir::DataDir,
    location::Locations,
    login_provider::LoginProvider,
    persistence::PersistedComponents,
    presence::{Presence, RegionSubscription, RepositionOnChunkLoad},
    rtsim::RtSim,
    state_ext::StateExt,
    sys::sentinel::{DeletedEntities, TrackedStorages},
};
#[cfg(not(feature = "worldgen"))]
use common::grid::Grid;
use common::{
    assets::AssetExt,
    calendar::Calendar,
    character::CharacterId,
    cmd::ChatCommand,
    comp,
    event::{EventBus, ServerEvent},
    recipe::default_recipe_book,
    resources::{BattleMode, Time, TimeOfDay},
    rtsim::RtSimEntity,
    slowjob::SlowJobPool,
    terrain::{TerrainChunk, TerrainChunkSize},
    vol::RectRasterableVol,
};
use common_ecs::run_now;
use common_net::{
    msg::{
        ClientType, DisconnectReason, ServerGeneral, ServerInfo, ServerInit, ServerMsg, WorldMapMsg,
    },
    sync::WorldSyncExt,
};
use common_state::{BuildAreas, State};
use common_systems::add_local_systems;
use metrics::{EcsSystemMetrics, PhysicsMetrics, TickMetrics};
use network::{ListenAddr, Network, Pid};
use persistence::{
    character_loader::{CharacterLoader, CharacterLoaderResponseKind},
    character_updater::CharacterUpdater,
};
use prometheus::Registry;
use prometheus_hyper::Server as PrometheusServer;
use specs::{join::Join, Builder, Entity as EcsEntity, Entity, SystemData, WorldExt};
use std::{
    i32,
    ops::{Deref, DerefMut},
    sync::Arc,
    time::{Duration, Instant},
};
#[cfg(not(feature = "worldgen"))]
use test_world::{IndexOwned, World};
use tokio::{runtime::Runtime, sync::Notify};
use tracing::{debug, error, info, trace, warn};
use vek::*;

use crate::{
    persistence::{DatabaseSettings, SqlLogMode},
    sys::terrain,
};
use hashbrown::HashMap;
use std::sync::RwLock;

use crate::settings::Protocol;

#[cfg(feature = "plugins")]
use {
    common::uid::UidAllocator,
    common_state::plugin::{memory_manager::EcsWorld, PluginMgr},
};

use common::comp::Anchor;
#[cfg(feature = "worldgen")]
use world::{
    sim::{FileOpts, WorldOpts, DEFAULT_WORLD_MAP},
    IndexOwned, World,
};

#[derive(Copy, Clone)]
pub struct SpawnPoint(pub Vec3<f32>);

impl Default for SpawnPoint {
    fn default() -> Self { Self(Vec3::new(0.0, 0.0, 256.0)) }
}

// This is the minimum chunk range that is kept loaded around each player
// server-side. This is independent of the client's view distance and exists to
// avoid exploits such as small view distance chunk reloading and also to keep
// various mechanics working fluidly (i.e: not unloading nearby entities).
pub const MIN_VD: u32 = 6;

// Tick count used for throttling network updates
// Note this doesn't account for dt (so update rate changes with tick rate)
#[derive(Copy, Clone, Default)]
pub struct Tick(u64);

#[derive(Clone)]
pub struct HwStats {
    hardware_threads: u32,
    rayon_threads: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum DisconnectType {
    WithPersistence,
    WithoutPersistence,
}

// Start of Tick, used for metrics
#[derive(Copy, Clone)]
pub struct TickStart(Instant);

/// Store of BattleMode cooldowns for players while they go offline
#[derive(Clone, Default, Debug)]
pub struct BattleModeBuffer {
    map: HashMap<CharacterId, (BattleMode, Time)>,
}

impl BattleModeBuffer {
    pub fn push(&mut self, char_id: CharacterId, save: (BattleMode, Time)) {
        self.map.insert(char_id, save);
    }

    pub fn get(&self, char_id: &CharacterId) -> Option<&(BattleMode, Time)> {
        self.map.get(char_id)
    }

    pub fn pop(&mut self, char_id: &CharacterId) -> Option<(BattleMode, Time)> {
        self.map.remove(char_id)
    }
}

pub struct ChunkRequest {
    entity: EcsEntity,
    key: Vec2<i32>,
}

pub struct Server {
    state: State,
    world: Arc<World>,
    index: IndexOwned,
    map: WorldMapMsg,

    connection_handler: ConnectionHandler,

    runtime: Arc<Runtime>,

    metrics_shutdown: Arc<Notify>,
    database_settings: Arc<RwLock<DatabaseSettings>>,
    disconnect_all_clients_requested: bool,
}

impl Server {
    /// Create a new `Server`
    pub fn new(
        settings: Settings,
        editable_settings: EditableSettings,
        database_settings: DatabaseSettings,
        data_dir: &std::path::Path,
        runtime: Arc<Runtime>,
    ) -> Result<Self, Error> {
        info!("Server data dir is: {}", data_dir.display());
        if settings.auth_server_address.is_none() {
            info!("Authentication is disabled");
        }

        // Run pending DB migrations (if any)
        debug!("Running DB migrations...");
        persistence::run_migrations(&database_settings);

        let database_settings = Arc::new(RwLock::new(database_settings));

        let registry = Arc::new(Registry::new());
        let chunk_gen_metrics = metrics::ChunkGenMetrics::new(&registry).unwrap();
        let job_metrics = metrics::JobMetrics::new(&registry).unwrap();
        let network_request_metrics = metrics::NetworkRequestMetrics::new(&registry).unwrap();
        let player_metrics = metrics::PlayerMetrics::new(&registry).unwrap();
        let ecs_system_metrics = EcsSystemMetrics::new(&registry).unwrap();
        let tick_metrics = TickMetrics::new(&registry).unwrap();
        let physics_metrics = PhysicsMetrics::new(&registry).unwrap();
        let battlemode_buffer = BattleModeBuffer::default();

        let mut state = State::server();
        state.ecs_mut().insert(battlemode_buffer);
        state.ecs_mut().insert(settings.clone());
        state.ecs_mut().insert(editable_settings);
        state.ecs_mut().insert(DataDir {
            path: data_dir.to_owned(),
        });
        state.ecs_mut().insert(EventBus::<ServerEvent>::default());
        state.ecs_mut().insert(Vec::<ChunkRequest>::new());
        state
            .ecs_mut()
            .insert(EventBus::<chunk_serialize::ChunkSendEntry>::default());
        state.ecs_mut().insert(Locations::default());
        state.ecs_mut().insert(LoginProvider::new(
            settings.auth_server_address.clone(),
            Arc::clone(&runtime),
        ));
        state.ecs_mut().insert(HwStats {
            hardware_threads: num_cpus::get() as u32,
            rayon_threads: num_cpus::get() as u32,
        });
        state.ecs_mut().insert(Tick(0));
        state.ecs_mut().insert(TickStart(Instant::now()));
        state.ecs_mut().insert(job_metrics);
        state.ecs_mut().insert(network_request_metrics);
        state.ecs_mut().insert(player_metrics);
        state.ecs_mut().insert(ecs_system_metrics);
        state.ecs_mut().insert(tick_metrics);
        state.ecs_mut().insert(physics_metrics);
        if settings.experimental_terrain_persistence {
            #[cfg(feature = "persistent_world")]
            {
                warn!(
                    "Experimental terrain persistence support is enabled. This feature may break, \
                     be disabled, or otherwise change under your feet at *any time*. \
                     Additionally, it is expected to be replaced in the future *without* \
                     migration or warning. You have been warned."
                );
                state
                    .ecs_mut()
                    .insert(TerrainPersistence::new(data_dir.to_owned()));
            }
            #[cfg(not(feature = "persistent_world"))]
            error!(
                "Experimental terrain persistence support was requested, but the server was not \
                 compiled with the feature. Terrain modifications will *not* be persisted."
            );
        }
        {
            let pool = state.ecs_mut().write_resource::<SlowJobPool>();
            pool.configure("CHUNK_GENERATOR", |n| n / 2 + n / 4);
            pool.configure("CHUNK_SERIALIZER", |n| n / 2);
        }
        state
            .ecs_mut()
            .insert(ChunkGenerator::new(chunk_gen_metrics));
        {
            let (sender, receiver) =
                crossbeam_channel::bounded::<chunk_serialize::SerializedChunk>(10_000);
            state.ecs_mut().insert(sender);
            state.ecs_mut().insert(receiver);
        }

        state.ecs_mut().insert(CharacterUpdater::new(
            Arc::<RwLock<DatabaseSettings>>::clone(&database_settings),
        )?);

        let ability_map = comp::item::tool::AbilityMap::<comp::AbilityItem>::load_expect_cloned(
            "common.abilities.ability_set_manifest",
        );
        state.ecs_mut().insert(ability_map);

        let msm = comp::inventory::item::MaterialStatManifest::load().cloned();
        state.ecs_mut().insert(msm);

        state.ecs_mut().insert(CharacterLoader::new(
            Arc::<RwLock<DatabaseSettings>>::clone(&database_settings),
        )?);

        // System schedulers to control execution of systems
        state
            .ecs_mut()
            .insert(sys::PersistenceScheduler::every(Duration::from_secs(10)));

        // Server-only components
        state.ecs_mut().register::<RegionSubscription>();
        state.ecs_mut().register::<Client>();
        state.ecs_mut().register::<Presence>();
        state.ecs_mut().register::<wiring::WiringElement>();
        state.ecs_mut().register::<wiring::Circuit>();
        state.ecs_mut().register::<comp::Anchor>();
        state.ecs_mut().register::<comp::Pet>();
        state.ecs_mut().register::<login_provider::PendingLogin>();
        state.ecs_mut().register::<RepositionOnChunkLoad>();

        //Alias validator
        let banned_words_paths = &settings.banned_words_files;
        let mut banned_words = Vec::new();
        for path in banned_words_paths {
            let mut list = match std::fs::File::open(&path) {
                Ok(file) => match ron::de::from_reader(&file) {
                    Ok(vec) => vec,
                    Err(error) => {
                        tracing::warn!(?error, ?file, "Couldn't deserialize banned words file");
                        return Err(Error::Other(format!(
                            "Couldn't read banned words file \"{}\"",
                            path.to_string_lossy()
                        )));
                    },
                },
                Err(error) => {
                    tracing::warn!(?error, ?path, "Couldn't open banned words file");
                    return Err(Error::Other(format!(
                        "Couldn't open banned words file \"{}\". Error: {}",
                        path.to_string_lossy(),
                        error
                    )));
                },
            };
            banned_words.append(&mut list);
        }
        let banned_words_count = banned_words.len();
        tracing::debug!(?banned_words_count);
        tracing::trace!(?banned_words);
        state.ecs_mut().insert(AliasValidator::new(banned_words));

        #[cfg(feature = "worldgen")]
        let (world, index) = World::generate(
            settings.world_seed,
            WorldOpts {
                seed_elements: true,
                world_file: if let Some(ref opts) = settings.map_file {
                    opts.clone()
                } else {
                    // Load default map from assets.
                    FileOpts::LoadAsset(DEFAULT_WORLD_MAP.into())
                },
                calendar: Some(settings.calendar_mode.calendar_now()),
            },
            state.thread_pool(),
        );

        #[cfg(feature = "worldgen")]
        let map = world.get_map_data(index.as_index_ref(), state.thread_pool());

        #[cfg(not(feature = "worldgen"))]
        let (world, index) = World::generate(settings.world_seed);
        #[cfg(not(feature = "worldgen"))]
        let map = WorldMapMsg {
            dimensions_lg: Vec2::zero(),
            max_height: 1.0,
            rgba: Grid::new(Vec2::new(1, 1), 1),
            horizons: [(vec![0], vec![0]), (vec![0], vec![0])],
            sea_level: 0.0,
            alt: Grid::new(Vec2::new(1, 1), 1),
            sites: Vec::new(),
            pois: Vec::new(),
        };

        #[cfg(feature = "worldgen")]
        let spawn_point = SpawnPoint({
            let index = index.as_index_ref();
            // NOTE: all of these `.map(|e| e as [type])` calls should compile into no-ops,
            // but are needed to be explicit about casting (and to make the compiler stop
            // complaining)

            // Search for town defined by spawn_town server setting. If this fails, or is
            // None, set spawn to the nearest town to the centre of the world
            let spawn_chunk = match settings.spawn_town.as_ref().and_then(|spawn_town| {
                world.civs().sites().find(|site| {
                    site.site_tmp
                        .map_or(false, |id| index.sites[id].name() == spawn_town)
                })
            }) {
                Some(t) => t.center,
                None => {
                    let center_chunk = world.sim().map_size_lg().chunks().map(i32::from) / 2;
                    use world::civ::SiteKind;
                    world
                        .civs()
                        .sites()
                        .filter(|site| {
                            matches!(site.kind, SiteKind::Settlement | SiteKind::Refactor)
                        })
                        .map(|site| site.center)
                        .min_by_key(|site_pos| site_pos.distance_squared(center_chunk))
                        .unwrap_or(center_chunk)
                },
            };

            world.find_accessible_pos(index, TerrainChunkSize::center_wpos(spawn_chunk), false)
        });
        #[cfg(not(feature = "worldgen"))]
        let spawn_point = SpawnPoint::default();

        // Set the spawn point we calculated above
        state.ecs_mut().insert(spawn_point);

        // Insert a default AABB for the world
        // TODO: prevent this from being deleted
        {
            #[cfg(feature = "worldgen")]
            let size = world.sim().get_size();
            #[cfg(not(feature = "worldgen"))]
            let size = Vec2::new(40, 40);

            let world_size = size.map(|e| e as i32) * TerrainChunk::RECT_SIZE.map(|e| e as i32);
            let world_aabb = Aabb {
                min: Vec3::new(0, 0, -32768),
                max: Vec3::new(world_size.x, world_size.y, 32767),
            }
            .made_valid();

            state
                .ecs()
                .write_resource::<BuildAreas>()
                .insert("world".to_string(), world_aabb)
                .expect("The initial insert should always work.");
        }

        // Insert the world into the ECS (todo: Maybe not an Arc?)
        let world = Arc::new(world);
        state.ecs_mut().insert(Arc::clone(&world));
        state
            .ecs_mut()
            .insert(lod::Lod::from_world(&world, index.as_index_ref()));
        state.ecs_mut().insert(index.clone());

        // Set starting time for the server.
        state.ecs_mut().write_resource::<TimeOfDay>().0 = settings.start_time;

        // Register trackers
        sys::sentinel::UpdateTrackers::register(state.ecs_mut());

        state.ecs_mut().insert(DeletedEntities::default());

        let network = Network::new_with_registry(Pid::new(), &runtime, &registry);
        let metrics_shutdown = Arc::new(Notify::new());
        let metrics_shutdown_clone = Arc::clone(&metrics_shutdown);
        let addr = settings.metrics_address;
        runtime.spawn(async move {
            PrometheusServer::run(
                Arc::clone(&registry),
                addr,
                metrics_shutdown_clone.notified(),
            )
            .await
        });

        let mut printed_quic_warning = false;
        for protocol in &settings.gameserver_protocols {
            match protocol {
                Protocol::Tcp { address } => {
                    runtime.block_on(network.listen(ListenAddr::Tcp(*address)))?;
                },
                Protocol::Quic {
                    address,
                    cert_file_path,
                    key_file_path,
                } => {
                    use rustls_pemfile::Item;
                    use std::fs;

                    match || -> Result<_, Box<dyn std::error::Error>> {
                        let key = fs::read(&key_file_path)?;
                        let key = if key_file_path.extension().map_or(false, |x| x == "der") {
                            rustls::PrivateKey(key)
                        } else {
                            debug!("convert pem key to der");
                            let key = rustls_pemfile::read_all(&mut key.as_slice())?
                                .into_iter()
                                .find_map(|item| match item {
                                    Item::RSAKey(v) | Item::PKCS8Key(v) => Some(v),
                                    Item::ECKey(_) => None,
                                    Item::X509Certificate(_) => None,
                                    _ => None,
                                })
                                .ok_or("No valid pem key in file")?;
                            rustls::PrivateKey(key)
                        };
                        let cert_chain = fs::read(&cert_file_path)?;
                        let cert_chain = if cert_file_path.extension().map_or(false, |x| x == "der")
                        {
                            vec![rustls::Certificate(cert_chain)]
                        } else {
                            debug!("convert pem cert to der");
                            let certs = rustls_pemfile::certs(&mut cert_chain.as_slice())?;
                            certs.into_iter().map(rustls::Certificate).collect()
                        };
                        let server_config = quinn::ServerConfig::with_single_cert(cert_chain, key)?;
                        Ok(server_config)
                    }() {
                        Ok(server_config) => {
                            runtime.block_on(
                                network.listen(ListenAddr::Quic(*address, server_config.clone())),
                            )?;

                            if !printed_quic_warning {
                                warn!(
                                    "QUIC is enabled. This is experimental and not recommended in \
                                     production"
                                );
                                printed_quic_warning = true;
                            }
                        },
                        Err(e) => {
                            error!(
                                ?e,
                                "Failed to load the TLS certificate, running without QUIC {}",
                                *address
                            );
                        },
                    }
                },
            }
        }

        runtime.block_on(network.listen(ListenAddr::Mpsc(14004)))?;

        let connection_handler = ConnectionHandler::new(network, &runtime);

        // Initiate real-time world simulation
        #[cfg(feature = "worldgen")]
        rtsim::init(&mut state, &world, index.as_index_ref(), spawn_point);
        #[cfg(not(feature = "worldgen"))]
        rtsim::init(&mut state);

        let this = Self {
            state,
            world,
            index,
            map,

            connection_handler,
            runtime,

            metrics_shutdown,
            database_settings,
            disconnect_all_clients_requested: false,
        };

        debug!(?settings, "created veloren server with");

        let git_hash = *common::util::GIT_HASH;
        let git_date = common::util::GIT_DATE.clone();
        let git_time = *common::util::GIT_TIME;
        let version = common::util::DISPLAY_VERSION_LONG.clone();
        info!(?version, "Server version");
        debug!(?git_hash, ?git_date, ?git_time, "detailed Server version");

        Ok(this)
    }

    pub fn get_server_info(&self) -> ServerInfo {
        let settings = self.state.ecs().fetch::<Settings>();
        let editable_settings = self.state.ecs().fetch::<EditableSettings>();
        ServerInfo {
            name: settings.server_name.clone(),
            description: (&*editable_settings.server_description).clone(),
            git_hash: common::util::GIT_HASH.to_string(),
            git_date: common::util::GIT_DATE.to_string(),
            auth_provider: settings.auth_server_address.clone(),
        }
    }

    /// Get a reference to the server's settings
    pub fn settings(&self) -> impl Deref<Target = Settings> + '_ {
        self.state.ecs().fetch::<Settings>()
    }

    /// Get a mutable reference to the server's settings
    pub fn settings_mut(&self) -> impl DerefMut<Target = Settings> + '_ {
        self.state.ecs().fetch_mut::<Settings>()
    }

    /// Get a mutable reference to the server's editable settings
    pub fn editable_settings_mut(&self) -> impl DerefMut<Target = EditableSettings> + '_ {
        self.state.ecs().fetch_mut::<EditableSettings>()
    }

    /// Get a reference to the server's editable settings
    pub fn editable_settings(&self) -> impl Deref<Target = EditableSettings> + '_ {
        self.state.ecs().fetch::<EditableSettings>()
    }

    /// Get path to the directory that the server info into
    pub fn data_dir(&self) -> impl Deref<Target = DataDir> + '_ {
        self.state.ecs().fetch::<DataDir>()
    }

    /// Get a reference to the server's game state.
    pub fn state(&self) -> &State { &self.state }

    /// Get a mutable reference to the server's game state.
    pub fn state_mut(&mut self) -> &mut State { &mut self.state }

    /// Get a reference to the server's world.
    pub fn world(&self) -> &World { &self.world }

    /// Get a reference to the server's world map.
    pub fn map(&self) -> &WorldMapMsg { &self.map }

    /// Execute a single server tick, handle input and update the game state by
    /// the given duration.
    pub fn tick(&mut self, _input: Input, dt: Duration) -> Result<Vec<Event>, Error> {
        self.state.ecs().write_resource::<Tick>().0 += 1;
        self.state.ecs().write_resource::<TickStart>().0 = Instant::now();

        // Update calendar events as time changes
        // TODO: If a lot of calendar events get added, this might become expensive.
        // Maybe don't do this every tick?
        let new_calendar = self
            .state
            .ecs()
            .read_resource::<Settings>()
            .calendar_mode
            .calendar_now();
        *self.state.ecs_mut().write_resource::<Calendar>() = new_calendar;

        // This tick function is the centre of the Veloren universe. Most server-side
        // things are managed from here, and as such it's important that it
        // stays organised. Please consult the core developers before making
        // significant changes to this code. Here is the approximate order of
        // things. Please update it as this code changes.
        //
        // 1) Collect input from the frontend, apply input effects to the
        //    state of the game
        // 2) Go through any events (timer-driven or otherwise) that need handling
        //    and apply them to the state of the game
        // 3) Go through all incoming client network communications, apply them to
        //    the game state
        // 4) Perform a single LocalState tick (i.e: update the world and entities
        //    in the world)
        // 5) Go through the terrain update queue and apply all changes to
        //    the terrain
        // 6) Send relevant state updates to all clients
        // 7) Check for persistence updates related to character data, and message the
        //    relevant entities
        // 8) Update Metrics with current data
        // 9) Finish the tick, passing control of the main thread back
        //    to the frontend

        // 1) Build up a list of events for this frame, to be passed to the frontend.
        let mut frontend_events = Vec::new();

        // 2)

        let before_new_connections = Instant::now();

        // 3) Handle inputs from clients
        self.handle_new_connections(&mut frontend_events);

        let before_state_tick = Instant::now();

        // 4) Tick the server's LocalState.
        // 5) Fetch any generated `TerrainChunk`s and insert them into the terrain.
        // in sys/terrain.rs
        self.state.tick(
            dt,
            |dispatcher_builder| {
                add_local_systems(dispatcher_builder);
                sys::msg::add_server_systems(dispatcher_builder);
                sys::add_server_systems(dispatcher_builder);
                #[cfg(feature = "worldgen")]
                rtsim::add_server_systems(dispatcher_builder);
            },
            false,
        );

        let before_handle_events = Instant::now();

        // Process any pending request to disconnect all clients, the disconnections
        // will be processed once handle_events() is called below
        let disconnect_type = self.disconnect_all_clients_if_requested();

        // Handle entity links (such as mounting)
        self.state.maintain_links();

        // Handle game events
        frontend_events.append(&mut self.handle_events());

        let before_update_terrain_and_regions = Instant::now();

        // Apply terrain changes and update the region map after processing server
        // events so that changes made by server events will be immediately
        // visible to client synchronization systems, minimizing the latency of
        // `ServerEvent` mediated effects
        self.state.update_region_map();
        // NOTE: apply_terrain_changes sends the *new* value since it is not being
        // synchronized during the tick.
        self.state.apply_terrain_changes();

        let before_sync = Instant::now();

        // 6) Synchronise clients with the new state of the world.
        sys::run_sync_systems(self.state.ecs_mut());

        let before_world_tick = Instant::now();

        // Tick the world
        self.world.tick(dt);

        let before_entity_cleanup = Instant::now();

        // In the event of a request to disconnect all players without persistence, we
        // must run the terrain system a second time after the messages to
        // perform client disconnections have been processed. This ensures that any
        // items on the ground are deleted.
        if let Some(DisconnectType::WithoutPersistence) = disconnect_type {
            run_now::<terrain::Sys>(self.state.ecs_mut());
        }

        // Prevent anchor entity chains which are not currently supported
        let anchors = self.state.ecs().read_storage::<comp::Anchor>();
        let anchored_anchor_entities: Vec<Entity> = (
            &self.state.ecs().entities(),
            &self.state.ecs().read_storage::<comp::Anchor>(),
        )
            .join()
            .filter_map(|(_, anchor)| match anchor {
                Anchor::Entity(anchor_entity) => Some(*anchor_entity),
                _ => None,
            })
            .filter(|anchor_entity| anchors.get(*anchor_entity).is_some())
            .collect();
        drop(anchors);

        for entity in anchored_anchor_entities {
            if cfg!(debug_assertions) {
                panic!("Entity anchor chain detected");
            }
            error!(
                "Detected an anchor entity that itself has an anchor entity - anchor chains are \
                 not currently supported. The entity's Anchor component has been deleted"
            );
            self.state.delete_component::<Anchor>(entity);
        }

        // Remove NPCs that are outside the view distances of all players
        // This is done by removing NPCs in unloaded chunks
        let to_delete = {
            let terrain = self.state.terrain();
            (
                &self.state.ecs().entities(),
                &self.state.ecs().read_storage::<comp::Pos>(),
                !&self.state.ecs().read_storage::<Presence>(),
                self.state.ecs().read_storage::<comp::Anchor>().maybe(),
            )
                .join()
                .filter(|(_, pos, _, anchor)| {
                    let chunk_key = terrain.pos_key(pos.0.map(|e| e.floor() as i32));
                    match anchor {
                        Some(Anchor::Chunk(hc)) => {
                            // Check if both this chunk and the NPCs `home_chunk` is unloaded. If
                            // so, we delete them. We check for
                            // `home_chunk` in order to avoid duplicating
                            // the entity under some circumstances.
                            terrain.get_key(chunk_key).is_none() && terrain.get_key(*hc).is_none()
                        },
                        Some(Anchor::Entity(entity)) => !self.state.ecs().is_alive(*entity),
                        None => terrain.get_key(chunk_key).is_none(),
                    }
                })
                .map(|(entity, _, _, _)| entity)
                .collect::<Vec<_>>()
        };

        for entity in to_delete {
            // Assimilate entities that are part of the real-time world simulation
            if let Some(rtsim_entity) = self
                .state
                .ecs()
                .read_storage::<RtSimEntity>()
                .get(entity)
                .copied()
            {
                self.state
                    .ecs()
                    .write_resource::<RtSim>()
                    .assimilate_entity(rtsim_entity.0);
            }

            if let Err(e) = self.state.delete_entity_recorded(entity) {
                error!(?e, "Failed to delete agent outside the terrain");
            }
        }

        if let Some(DisconnectType::WithoutPersistence) = disconnect_type {
            info!(
                "Disconnection of all players without persistence complete, signalling to \
                 persistence thread that character updates may continue to be processed"
            );
            self.state
                .ecs()
                .fetch_mut::<CharacterUpdater>()
                .disconnected_success();
        }

        // 7 Persistence updates
        let before_persistence_updates = Instant::now();

        let character_loader = self
            .state
            .ecs()
            .read_resource::<persistence::character_loader::CharacterLoader>();

        let character_updater = self
            .state
            .ecs()
            .read_resource::<persistence::character_updater::CharacterUpdater>();

        // Get character-related database responses and notify the requesting client
        character_loader
            .messages()
            .chain(character_updater.messages())
            .for_each(|query_result| match query_result.result {
                CharacterLoaderResponseKind::CharacterList(result) => match result {
                    Ok(character_list_data) => self.notify_client(
                        query_result.entity,
                        ServerGeneral::CharacterListUpdate(character_list_data),
                    ),
                    Err(error) => self.notify_client(
                        query_result.entity,
                        ServerGeneral::CharacterActionError(error.to_string()),
                    ),
                },
                CharacterLoaderResponseKind::CharacterCreation(result) => match result {
                    Ok((character_id, list)) => {
                        self.notify_client(
                            query_result.entity,
                            ServerGeneral::CharacterListUpdate(list),
                        );
                        self.notify_client(
                            query_result.entity,
                            ServerGeneral::CharacterCreated(character_id),
                        );
                    },
                    Err(error) => self.notify_client(
                        query_result.entity,
                        ServerGeneral::CharacterActionError(error.to_string()),
                    ),
                },
                CharacterLoaderResponseKind::CharacterEdit(result) => match result {
                    Ok((character_id, list)) => {
                        self.notify_client(
                            query_result.entity,
                            ServerGeneral::CharacterListUpdate(list),
                        );
                        self.notify_client(
                            query_result.entity,
                            ServerGeneral::CharacterEdited(character_id),
                        );
                    },
                    Err(error) => self.notify_client(
                        query_result.entity,
                        ServerGeneral::CharacterActionError(error.to_string()),
                    ),
                },
                CharacterLoaderResponseKind::CharacterData(result) => {
                    let message = match *result {
                        Ok(character_data) => {
                            let PersistedComponents {
                                body,
                                stats,
                                skill_set,
                                inventory,
                                waypoint,
                                pets,
                                active_abilities,
                                map_marker,
                            } = character_data;
                            let character_data = (
                                body,
                                stats,
                                skill_set,
                                inventory,
                                waypoint,
                                pets,
                                active_abilities,
                                map_marker,
                            );
                            ServerEvent::UpdateCharacterData {
                                entity: query_result.entity,
                                components: character_data,
                            }
                        },
                        Err(error) => {
                            // We failed to load data for the character from the DB. Notify the
                            // client to push the state back to character selection, with the error
                            // to display
                            self.notify_client(
                                query_result.entity,
                                ServerGeneral::CharacterDataLoadError(error.to_string()),
                            );

                            // Clean up the entity data on the server
                            ServerEvent::ExitIngame {
                                entity: query_result.entity,
                            }
                        },
                    };

                    self.state
                        .ecs()
                        .read_resource::<EventBus<ServerEvent>>()
                        .emit_now(message);
                },
            });

        drop(character_loader);
        drop(character_updater);

        {
            // Check for new chunks; cancel and regenerate all chunks if the asset has been
            // reloaded. Note that all of these assignments are no-ops, so the
            // only work we do here on the fast path is perform a relaxed read on an atomic.
            // boolean.
            let index = &mut self.index;
            let world = &mut self.world;
            let ecs = self.state.ecs_mut();
            let slow_jobs = ecs.write_resource::<SlowJobPool>();

            index.reload_if_changed(|index| {
                let mut chunk_generator = ecs.write_resource::<ChunkGenerator>();
                let client = ecs.read_storage::<Client>();
                let mut terrain = ecs.write_resource::<common::terrain::TerrainGrid>();

                // Cancel all pending chunks.
                chunk_generator.cancel_all();

                if client.is_empty() {
                    // No clients, so just clear all terrain.
                    terrain.clear();
                } else {
                    // There's at least one client, so regenerate all chunks.
                    terrain.iter().for_each(|(pos, _)| {
                        chunk_generator.generate_chunk(
                            None,
                            pos,
                            &slow_jobs,
                            Arc::clone(world),
                            index.clone(),
                            (
                                *ecs.read_resource::<TimeOfDay>(),
                                (*ecs.read_resource::<Calendar>()).clone(),
                            ),
                        );
                    });
                }
            });
        }

        let end_of_server_tick = Instant::now();

        // 8) Update Metrics
        run_now::<sys::metrics::Sys>(self.state.ecs());

        {
            // Report timing info
            let tick_metrics = self.state.ecs().read_resource::<metrics::TickMetrics>();

            let tt = &tick_metrics.tick_time;
            tt.with_label_values(&["new connections"])
                .set((before_state_tick - before_new_connections).as_nanos() as i64);
            tt.with_label_values(&["handle server events"])
                .set((before_update_terrain_and_regions - before_handle_events).as_nanos() as i64);
            tt.with_label_values(&["update terrain and region map"])
                .set((before_sync - before_update_terrain_and_regions).as_nanos() as i64);
            tt.with_label_values(&["state"])
                .set((before_handle_events - before_state_tick).as_nanos() as i64);
            tt.with_label_values(&["world tick"])
                .set((before_entity_cleanup - before_world_tick).as_nanos() as i64);
            tt.with_label_values(&["entity cleanup"])
                .set((before_persistence_updates - before_entity_cleanup).as_nanos() as i64);
            tt.with_label_values(&["persistence_updates"])
                .set((end_of_server_tick - before_persistence_updates).as_nanos() as i64);
            tick_metrics.tick_time_hist.observe(
                end_of_server_tick
                    .duration_since(before_state_tick)
                    .as_secs_f64(),
            );
        }

        // 9) Finish the tick, pass control back to the frontend.

        Ok(frontend_events)
    }

    /// Clean up the server after a tick.
    pub fn cleanup(&mut self) {
        // Cleanup the local state
        self.state.cleanup();

        // Maintain persisted terrain
        #[cfg(feature = "persistent_world")]
        self.state
            .ecs()
            .try_fetch_mut::<TerrainPersistence>()
            .map(|mut t| t.maintain());
    }

    fn initialize_client(
        &mut self,
        client: crate::connection_handler::IncomingClient,
    ) -> Result<Option<specs::Entity>, Error> {
        if self.settings().max_players <= self.state.ecs().read_storage::<Client>().join().count() {
            trace!(
                ?client.participant,
                "to many players, wont allow participant to connect"
            );
            client.send(ServerInit::TooManyPlayers)?;
            return Ok(None);
        }

        let entity = self
            .state
            .ecs_mut()
            .create_entity_synced()
            .with(client)
            .build();
        self.state
            .ecs()
            .read_resource::<metrics::PlayerMetrics>()
            .clients_connected
            .inc();
        // Send client all the tracked components currently attached to its entity as
        // well as synced resources (currently only `TimeOfDay`)
        debug!("Starting initial sync with client.");
        self.state
            .ecs()
            .read_storage::<Client>()
            .get(entity)
            .expect(
                "We just created this entity with a Client component using build(), and we have \
                 &mut access to the ecs so it can't have been deleted yet.",
            )
            .send(ServerInit::GameSync {
                // Send client their entity
                entity_package: TrackedStorages::fetch(self.state.ecs())
                    .create_entity_package(entity, None, None, None)
                    .expect(
                        "We just created this entity as marked() (using create_entity_synced) so \
                         it definitely has a uid",
                    ),
                time_of_day: *self.state.ecs().read_resource(),
                max_group_size: self.settings().max_player_group_size,
                client_timeout: self.settings().client_timeout,
                world_map: self.map.clone(),
                recipe_book: default_recipe_book().cloned(),
                material_stats: (&*self
                    .state
                    .ecs()
                    .read_resource::<comp::item::tool::MaterialStatManifest>())
                    .clone(),
                ability_map: (&*self
                    .state
                    .ecs()
                    .read_resource::<comp::item::tool::AbilityMap>())
                    .clone(),
            })?;
        Ok(Some(entity))
    }

    /// Disconnects all clients if requested by either an admin command or
    /// due to a persistence transaction failure and returns the processed
    /// DisconnectionType
    fn disconnect_all_clients_if_requested(&mut self) -> Option<DisconnectType> {
        let mut character_updater = self.state.ecs().fetch_mut::<CharacterUpdater>();

        let disconnect_type = self.get_disconnect_all_clients_requested(&mut character_updater);
        if let Some(disconnect_type) = disconnect_type {
            let with_persistence = disconnect_type == DisconnectType::WithPersistence;
            let clients = self.state.ecs().read_storage::<Client>();
            let entities = self.state.ecs().entities();

            info!(
                "Disconnecting all clients ({} persistence) as requested",
                if with_persistence { "with" } else { "without" }
            );
            for (_, entity) in (&clients, &entities).join() {
                info!("Emitting client disconnect event for entity: {:?}", entity);
                let event = if with_persistence {
                    ServerEvent::ClientDisconnect(entity, comp::DisconnectReason::Kicked)
                } else {
                    ServerEvent::ClientDisconnectWithoutPersistence(entity)
                };
                self.state
                    .ecs()
                    .read_resource::<EventBus<ServerEvent>>()
                    .emit_now(event);
            }

            self.disconnect_all_clients_requested = false;
        }

        disconnect_type
    }

    fn get_disconnect_all_clients_requested(
        &self,
        character_updater: &mut CharacterUpdater,
    ) -> Option<DisconnectType> {
        let without_persistence_requested = character_updater.disconnect_all_clients_requested();
        let with_persistence_requested = self.disconnect_all_clients_requested;

        if without_persistence_requested {
            return Some(DisconnectType::WithoutPersistence);
        };
        if with_persistence_requested {
            return Some(DisconnectType::WithPersistence);
        };
        None
    }

    /// Handle new client connections.
    fn handle_new_connections(&mut self, frontend_events: &mut Vec<Event>) {
        while let Ok(sender) = self.connection_handler.info_requester_receiver.try_recv() {
            // can fail, e.g. due to timeout or network prob.
            trace!("sending info to connection_handler");
            let _ = sender.send(crate::connection_handler::ServerInfoPacket {
                info: self.get_server_info(),
                time: self.state.get_time(),
            });
        }

        while let Ok(incoming) = self.connection_handler.client_receiver.try_recv() {
            match self.initialize_client(incoming) {
                Ok(None) => (),
                Ok(Some(entity)) => {
                    frontend_events.push(Event::ClientConnected { entity });
                    debug!("Done initial sync with client.");
                },
                Err(e) => {
                    debug!(?e, "failed initializing a new client");
                },
            }
        }
    }

    pub fn notify_client<S>(&self, entity: EcsEntity, msg: S)
    where
        S: Into<ServerMsg>,
    {
        self.state
            .ecs()
            .read_storage::<Client>()
            .get(entity)
            .map(|c| c.send(msg));
    }

    pub fn notify_players(&mut self, msg: ServerGeneral) { self.state.notify_players(msg); }

    pub fn generate_chunk(&mut self, entity: EcsEntity, key: Vec2<i32>) {
        let ecs = self.state.ecs();
        let slow_jobs = ecs.read_resource::<SlowJobPool>();
        ecs.write_resource::<ChunkGenerator>().generate_chunk(
            Some(entity),
            key,
            &slow_jobs,
            Arc::clone(&self.world),
            self.index.clone(),
            (
                *ecs.read_resource::<TimeOfDay>(),
                (*ecs.read_resource::<Calendar>()).clone(),
            ),
        );
    }

    fn process_command(&mut self, entity: EcsEntity, name: String, args: Vec<String>) {
        // Find the command object and run its handler.
        if let Ok(command) = name.parse::<ChatCommand>() {
            command.execute(self, entity, args);
        } else {
            #[cfg(feature = "plugins")]
            {
                let plugin_manager = self.state.ecs().read_resource::<PluginMgr>();
                let ecs_world = EcsWorld {
                    entities: &self.state.ecs().entities(),
                    health: self.state.ecs().read_component().into(),
                    uid: self.state.ecs().read_component().into(),
                    uid_allocator: &self.state.ecs().read_resource::<UidAllocator>().into(),
                    player: self.state.ecs().read_component().into(),
                };
                let uid = if let Some(uid) = ecs_world.uid.get(entity).copied() {
                    uid
                } else {
                    self.notify_client(
                        entity,
                        ServerGeneral::server_msg(
                            comp::ChatType::CommandError,
                            "Can't get player UUID (player may be disconnected?)",
                        ),
                    );
                    return;
                };
                let rs = plugin_manager.execute_event(
                    &ecs_world,
                    &plugin_api::event::ChatCommandEvent {
                        command: name.clone(),
                        command_args: args.clone(),
                        player: plugin_api::event::Player { id: uid },
                    },
                );
                match rs {
                    Ok(e) => {
                        if e.is_empty() {
                            self.notify_client(
                                entity,
                                ServerGeneral::server_msg(
                                    comp::ChatType::CommandError,
                                    format!(
                                        "Unknown command '/{}'.\nType '/help' for available \
                                         commands",
                                        name
                                    ),
                                ),
                            );
                        } else {
                            e.into_iter().for_each(|e| match e {
                                Ok(e) => {
                                    if !e.is_empty() {
                                        self.notify_client(
                                            entity,
                                            ServerGeneral::server_msg(
                                                comp::ChatType::CommandInfo,
                                                e.join("\n"),
                                            ),
                                        );
                                    }
                                },
                                Err(e) => {
                                    self.notify_client(
                                        entity,
                                        ServerGeneral::server_msg(
                                            comp::ChatType::CommandError,
                                            format!(
                                                "Error occurred while executing command '/{}'.\n{}",
                                                name, e
                                            ),
                                        ),
                                    );
                                },
                            });
                        }
                    },
                    Err(e) => {
                        error!(?e, "Can't execute command {} {:?}", name, args);
                        self.notify_client(
                            entity,
                            ServerGeneral::server_msg(
                                comp::ChatType::CommandError,
                                format!(
                                    "Internal error while executing '/{}'.\nContact the server \
                                     administrator",
                                    name
                                ),
                            ),
                        );
                    },
                }
            }
        }
    }

    fn entity_admin_role(&self, entity: EcsEntity) -> Option<comp::AdminRole> {
        self.state
            .read_component_copied::<comp::Admin>(entity)
            .map(|admin| admin.0)
    }

    pub fn number_of_players(&self) -> i64 {
        self.state.ecs().read_storage::<Client>().join().count() as i64
    }

    /// NOTE: Do *not* allow this to be called from any command that doesn't go
    /// through the CLI!
    pub fn add_admin(&mut self, username: &str, role: comp::AdminRole) {
        let mut editable_settings = self.editable_settings_mut();
        let login_provider = self.state.ecs().fetch::<LoginProvider>();
        let data_dir = self.data_dir();
        if let Some(entity) = add_admin(
            username,
            role,
            &login_provider,
            &mut editable_settings,
            &data_dir.path,
        )
        .and_then(|uuid| {
            let state = &self.state;
            (
                &state.ecs().entities(),
                &state.read_storage::<comp::Player>(),
            )
                .join()
                .find(|(_, player)| player.uuid() == uuid)
                .map(|(e, _)| e)
        }) {
            drop((data_dir, login_provider, editable_settings));
            // Add admin component if the player is ingame; if they are not, we can ignore
            // the write failure.
            self.state
                .write_component_ignore_entity_dead(entity, comp::Admin(role));
        };
    }

    /// NOTE: Do *not* allow this to be called from any command that doesn't go
    /// through the CLI!
    pub fn remove_admin(&self, username: &str) {
        let mut editable_settings = self.editable_settings_mut();
        let login_provider = self.state.ecs().fetch::<LoginProvider>();
        let data_dir = self.data_dir();
        if let Some(entity) = remove_admin(
            username,
            &login_provider,
            &mut editable_settings,
            &data_dir.path,
        )
        .and_then(|uuid| {
            let state = &self.state;
            (
                &state.ecs().entities(),
                &state.read_storage::<comp::Player>(),
            )
                .join()
                .find(|(_, player)| player.uuid() == uuid)
                .map(|(e, _)| e)
        }) {
            // Remove admin component if the player is ingame
            self.state
                .ecs()
                .write_storage::<comp::Admin>()
                .remove(entity);
        };
    }

    /// Useful for testing without a client
    /// view_distance: distance in chunks that are persisted, this acts like the
    /// player view distance so it is actually a bit farther due to a buffer
    /// zone
    #[cfg(feature = "worldgen")]
    pub fn create_centered_persister(&mut self, view_distance: u32) {
        let world_dims_chunks = self.world.sim().get_size();
        let world_dims_blocks = TerrainChunkSize::blocks(world_dims_chunks);
        // NOTE: origin is in the corner of the map
        // TODO: extend this function to have picking a random position or specifiying a
        // position as options
        //let mut rng = rand::thread_rng();
        // // Pick a random position but not to close to the edge
        // let rand_pos = world_dims_blocks.map(|e| e as i32).map(|e| e / 2 +
        // rng.gen_range(-e/2..e/2 + 1));
        let pos = comp::Pos(Vec3::from(world_dims_blocks.map(|e| e as f32 / 2.0)));
        self.state
            .create_persister(pos, view_distance, &self.world, &self.index)
            .build();
    }

    /// Sets the SQL log mode at runtime
    pub fn set_sql_log_mode(&mut self, sql_log_mode: SqlLogMode) {
        // Unwrap is safe here because we only perform a variable assignment with the
        // RwLock taken meaning that no panic can occur that would cause the
        // RwLock to become poisoned. This justification also means that calling
        // unwrap() on the associated read() calls for this RwLock is also safe
        // as long as no code that can panic is introduced here.
        let mut database_settings = self.database_settings.write().unwrap();
        database_settings.sql_log_mode = sql_log_mode;
        // Drop the RwLockWriteGuard to avoid performing unnecessary actions (logging)
        // with the lock taken.
        drop(database_settings);
        info!("SQL log mode changed to {:?}", sql_log_mode);
    }

    pub fn disconnect_all_clients(&mut self) {
        info!("Disconnecting all clients due to local console command");
        self.disconnect_all_clients_requested = true;
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.metrics_shutdown.notify_one();

        self.state
            .notify_players(ServerGeneral::Disconnect(DisconnectReason::Shutdown));

        #[cfg(feature = "persistent_world")]
        self.state
            .ecs()
            .try_fetch_mut::<TerrainPersistence>()
            .map(|mut terrain_persistence| {
                info!("Unloading terrain persistence...");
                terrain_persistence.unload_all()
            });
    }
}

#[must_use]
pub fn handle_edit<T, S: settings::EditableSetting>(
    data: T,
    result: Option<(String, Result<(), settings::SettingError<S>>)>,
) -> Option<T> {
    use crate::settings::SettingError;
    let (info, result) = result?;
    match result {
        Ok(()) => {
            info!("{}", info);
            Some(data)
        },
        Err(SettingError::Io(err)) => {
            warn!(
                ?err,
                "Failed to write settings file to disk, but succeeded in memory (success message: \
                 {})",
                info,
            );
            Some(data)
        },
        Err(SettingError::Integrity(err)) => {
            error!(?err, "Encountered an error while validating the request",);
            None
        },
    }
}

/// If successful returns the Some(uuid) of the added admin
///
/// NOTE: Do *not* allow this to be called from any command that doesn't go
/// through the CLI!
#[must_use]
pub fn add_admin(
    username: &str,
    role: comp::AdminRole,
    login_provider: &LoginProvider,
    editable_settings: &mut EditableSettings,
    data_dir: &std::path::Path,
) -> Option<common::uuid::Uuid> {
    use crate::settings::EditableSetting;
    let role_ = role.into();
    match login_provider.username_to_uuid(username) {
        Ok(uuid) => handle_edit(
            uuid,
            editable_settings.admins.edit(data_dir, |admins| {
                match admins.insert(uuid, settings::AdminRecord {
                    username_when_admined: Some(username.into()),
                    date: chrono::Utc::now(),
                    role: role_,
                }) {
                    None => Some(format!(
                        "Successfully added {} ({}) as {:?}!",
                        username, uuid, role
                    )),
                    Some(old_admin) if old_admin.role == role_ => {
                        info!("{} ({}) already has role: {:?}!", username, uuid, role);
                        None
                    },
                    Some(old_admin) => Some(format!(
                        "{} ({}) role changed from {:?} to {:?}!",
                        username, uuid, old_admin.role, role
                    )),
                }
            }),
        ),
        Err(err) => {
            error!(
                ?err,
                "Could not find uuid for this name; either the user does not exist or there was \
                 an error communicating with the auth server."
            );
            None
        },
    }
}

/// If successful returns the Some(uuid) of the removed admin
///
/// NOTE: Do *not* allow this to be called from any command that doesn't go
/// through the CLI!
#[must_use]
pub fn remove_admin(
    username: &str,
    login_provider: &LoginProvider,
    editable_settings: &mut EditableSettings,
    data_dir: &std::path::Path,
) -> Option<common::uuid::Uuid> {
    use crate::settings::EditableSetting;
    match login_provider.username_to_uuid(username) {
        Ok(uuid) => handle_edit(
            uuid,
            editable_settings.admins.edit(data_dir, |admins| {
                if let Some(admin) = admins.remove(&uuid) {
                    Some(format!(
                        "Successfully removed {} ({}) with role {:?} from the admins list",
                        username, uuid, admin.role,
                    ))
                } else {
                    info!("{} ({}) is not an admin!", username, uuid);
                    None
                }
            }),
        ),
        Err(err) => {
            error!(
                ?err,
                "Could not find uuid for this name; either the user does not exist or there was \
                 an error communicating with the auth server."
            );
            None
        },
    }
}
