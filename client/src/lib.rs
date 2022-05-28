#![deny(unsafe_code)]
#![deny(clippy::clone_on_ref_ptr)]
#![feature(label_break_value, option_zip)]

pub mod addr;
pub mod cmd;
pub mod error;

// Reexports
pub use crate::error::Error;
pub use authc::AuthClientError;
pub use common_net::msg::ServerInfo;
pub use specs::{
    join::Join,
    saveload::{Marker, MarkerAllocator},
    Builder, DispatcherBuilder, Entity as EcsEntity, ReadStorage, World, WorldExt,
};

use crate::addr::ConnectionArgs;
use byteorder::{ByteOrder, LittleEndian};
use common::{
    character::{CharacterId, CharacterItem},
    comp::{
        self,
        chat::{KillSource, KillType},
        controller::CraftEvent,
        group,
        inventory::item::{modular, tool, ItemKind},
        invite::{InviteKind, InviteResponse},
        skills::Skill,
        slot::{EquipSlot, InvSlotId, Slot},
        CharacterState, ChatMode, CommandGenerator, ControlAction, ControlEvent, Controller,
        ControllerInputs, GroupManip, InputKind, InventoryAction, InventoryEvent,
        InventoryUpdateEvent, MapMarkerChange, RemoteController, UtteranceKind, Vel,
    },
    event::{EventBus, LocalEvent},
    grid::Grid,
    link::Is,
    lod,
    mounting::Rider,
    outcome::Outcome,
    recipe::{ComponentRecipeBook, RecipeBook},
    resources::{MonotonicTime, PlayerEntity, Time, TimeOfDay},
    spiral::Spiral2d,
    terrain::{
        block::Block, map::MapConfig, neighbors, BiomeKind, SitesKind, SpriteKind, TerrainChunk,
        TerrainChunkSize,
    },
    trade::{PendingTrade, SitePrices, TradeAction, TradeId, TradeResult},
    uid::{Uid, UidAllocator},
    vol::RectVolSize,
};
#[cfg(feature = "tracy")] use common_base::plot;
use common_base::{prof_span, span};
use common_net::{
    msg::{
        self, validate_chat_msg,
        world_msg::{EconomyInfo, PoiInfo, SiteId, SiteInfo},
        ChatMsgValidationError, ClientGeneral, ClientMsg, ClientRegister, ClientType,
        DisconnectReason, InviteAnswer, Notification, PingMsg, PlayerInfo, PlayerListUpdate,
        PresenceKind, RegisterError, ServerGeneral, ServerInit, ServerRegisterAnswer,
        MAX_BYTES_CHAT_MSG,
    },
    sync::WorldSyncExt,
};
use common_state::State;
use common_systems::{add_local_systems, add_rewind_systems};
use comp::BuffKind;
use hashbrown::{HashMap, HashSet};
use image::DynamicImage;
use network::{ConnectAddr, Network, Participant, Pid, Stream};
use num::traits::FloatConst;
use rayon::prelude::*;
use specs::Component;
use std::{
    collections::VecDeque,
    mem,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::runtime::Runtime;
use tracing::{debug, error, trace, warn};
use vek::*;

const PING_ROLLING_AVERAGE_SECS: usize = 10;

#[derive(Debug)]
pub enum Event {
    Chat(comp::ChatMsg),
    InviteComplete {
        target: Uid,
        answer: InviteAnswer,
        kind: InviteKind,
    },
    TradeComplete {
        result: TradeResult,
        trade: PendingTrade,
    },
    Disconnect,
    DisconnectionNotification(u64),
    InventoryUpdated(InventoryUpdateEvent),
    Kicked(String),
    Notification(Notification),
    SetViewDistance(u32),
    Outcome(Outcome),
    CharacterCreated(CharacterId),
    CharacterEdited(CharacterId),
    CharacterError(String),
    MapMarker(comp::MapMarkerUpdate),
}

pub struct WorldData {
    /// Just the "base" layer for LOD; currently includes colors and nothing
    /// else. In the future we'll add more layers, like shadows, rivers, and
    /// probably foliage, cities, roads, and other structures.
    pub lod_base: Grid<u32>,
    /// The "height" layer for LOD; currently includes only land altitudes, but
    /// in the future should also water depth, and probably other
    /// information as well.
    pub lod_alt: Grid<u32>,
    /// The "shadow" layer for LOD.  Includes east and west horizon angles and
    /// an approximate max occluder height, which we use to try to
    /// approximate soft and volumetric shadows.
    pub lod_horizon: Grid<u32>,
    /// A fully rendered map image for use with the map and minimap; note that
    /// this can be constructed dynamically by combining the layers of world
    /// map data (e.g. with shadow map data or river data), but at present
    /// we opt not to do this.
    ///
    /// The first two elements of the tuple are the regular and topographic maps
    /// respectively. The third element of the tuple is the world size (as a 2D
    /// grid, in chunks), and the fourth element holds the minimum height for
    /// any land chunk (i.e. the sea level) in its x coordinate, and the maximum
    /// land height above this height (i.e. the max height) in its y coordinate.
    map: (Vec<Arc<DynamicImage>>, Vec2<u16>, Vec2<f32>),
}

impl WorldData {
    pub fn chunk_size(&self) -> Vec2<u16> { self.map.1 }

    pub fn map_layers(&self) -> &Vec<Arc<DynamicImage>> { &self.map.0 }

    pub fn map_image(&self) -> &Arc<DynamicImage> { &self.map.0[0] }

    pub fn min_chunk_alt(&self) -> f32 { self.map.2.x }

    pub fn max_chunk_alt(&self) -> f32 { self.map.2.y }
}

pub struct SiteInfoRich {
    pub site: SiteInfo,
    pub economy: Option<EconomyInfo>,
}

pub struct Client {
    registered: bool,
    presence: Option<PresenceKind>,
    runtime: Arc<Runtime>,
    server_info: ServerInfo,
    world_data: WorldData,
    player_list: HashMap<Uid, PlayerInfo>,
    character_list: CharacterList,
    sites: HashMap<SiteId, SiteInfoRich>,
    pois: Vec<PoiInfo>,
    pub chat_mode: ChatMode,
    recipe_book: RecipeBook,
    component_recipe_book: ComponentRecipeBook,
    available_recipes: HashMap<String, Option<SpriteKind>>,
    lod_zones: HashMap<Vec2<i32>, lod::Zone>,
    lod_last_requested: Option<Instant>,

    max_group_size: u32,
    // Client has received an invite (inviter uid, time out instant)
    invite: Option<(Uid, Instant, Duration, InviteKind)>,
    group_leader: Option<Uid>,
    // Note: potentially representable as a client only component
    group_members: HashMap<Uid, group::Role>,
    // Pending invites that this client has sent out
    pending_invites: HashSet<Uid>,
    // The pending trade the client is involved in, and it's id
    pending_trade: Option<(TradeId, PendingTrade, Option<SitePrices>)>,

    local_command_gen: CommandGenerator,
    next_control: Controller,
    inter_tick_rewind_time: Option<Duration>,
    _rewind_fluctuation_budget: f64,

    network: Option<Network>,
    participant: Option<Participant>,
    general_stream: Stream,
    ping_stream: Stream,
    register_stream: Stream,
    character_screen_stream: Stream,
    in_game_stream: Stream,
    terrain_stream: Stream,

    client_timeout: Duration,
    last_server_ping: f64,
    last_server_pong: f64,
    last_ping_delta: f64,
    ping_deltas: VecDeque<f64>,

    tick: u64,
    state: State,

    view_distance: Option<u32>,
    lod_distance: f32,
    // TODO: move into voxygen
    loaded_distance: f32,

    pending_chunks: HashMap<Vec2<i32>, Instant>,
    target_time_of_day: Option<TimeOfDay>,
}

/// Holds data related to the current players characters, as well as some
/// additional state to handle UI.
#[derive(Debug, Default)]
pub struct CharacterList {
    pub characters: Vec<CharacterItem>,
    pub loading: bool,
}

impl Client {
    pub async fn new(
        addr: ConnectionArgs,
        runtime: Arc<Runtime>,
        // TODO: refactor to avoid needing to use this out parameter
        mismatched_server_info: &mut Option<ServerInfo>,
    ) -> Result<Self, Error> {
        let network = Network::new(Pid::new(), &runtime);

        let participant = match addr {
            ConnectionArgs::Tcp {
                hostname,
                prefer_ipv6,
            } => addr::try_connect(&network, &hostname, prefer_ipv6, ConnectAddr::Tcp).await?,
            ConnectionArgs::Quic {
                hostname,
                prefer_ipv6,
            } => {
                warn!(
                    "QUIC is enabled. This is experimental and you won't be able to connect to \
                     TCP servers unless deactivated"
                );
                let config = quinn::ClientConfig::with_native_roots();
                addr::try_connect(&network, &hostname, prefer_ipv6, |a| {
                    ConnectAddr::Quic(a, config.clone(), hostname.clone())
                })
                .await?
            },
            ConnectionArgs::Mpsc(id) => network.connect(ConnectAddr::Mpsc(id)).await?,
        };

        let stream = participant.opened().await?;
        let mut ping_stream = participant.opened().await?;
        let mut register_stream = participant.opened().await?;
        let character_screen_stream = participant.opened().await?;
        let in_game_stream = participant.opened().await?;
        let terrain_stream = participant.opened().await?;

        register_stream.send(ClientType::Game)?;
        let server_info: ServerInfo = register_stream.recv().await?;
        if server_info.git_hash != *common::util::GIT_HASH {
            warn!(
                "Server is running {}[{}], you are running {}[{}], versions might be incompatible!",
                server_info.git_hash,
                server_info.git_date,
                common::util::GIT_HASH.to_string(),
                common::util::GIT_DATE.to_string(),
            );

            // Pass the server info back to the caller to ensure they can access it even
            // if this function errors.
            mem::swap(mismatched_server_info, &mut Some(server_info.clone()));
        }
        debug!("Auth Server: {:?}", server_info.auth_provider);

        ping_stream.send(PingMsg::Ping)?;

        // Wait for initial sync
        let mut ping_interval = tokio::time::interval(core::time::Duration::from_secs(1));
        let (
            state,
            lod_base,
            lod_alt,
            lod_horizon,
            world_map,
            sites,
            pois,
            recipe_book,
            component_recipe_book,
            max_group_size,
            client_timeout,
        ) = match loop {
            tokio::select! {
                res = register_stream.recv() => break res?,
                _ = ping_interval.tick() => ping_stream.send(PingMsg::Ping)?,
            }
        } {
            ServerInit::GameSync {
                entity_package,
                time_of_day,
                max_group_size,
                client_timeout,
                world_map,
                recipe_book,
                component_recipe_book,
                material_stats,
                ability_map,
            } => {
                // Initialize `State`
                let mut state = State::client();
                // Client-only components
                state.ecs_mut().register::<comp::Last<CharacterState>>();

                let entity = state.ecs_mut().apply_entity_package(entity_package);
                *state.ecs_mut().write_resource() = time_of_day;
                *state.ecs_mut().write_resource() = PlayerEntity(Some(entity));
                state.ecs_mut().insert(material_stats);
                state.ecs_mut().insert(ability_map);

                let map_size_lg = common::terrain::MapSizeLg::new(world_map.dimensions_lg)
                    .map_err(|_| {
                        Error::Other(format!(
                            "Server sent bad world map dimensions: {:?}",
                            world_map.dimensions_lg,
                        ))
                    })?;
                let map_size = map_size_lg.chunks();
                let max_height = world_map.max_height;
                let sea_level = world_map.sea_level;
                let rgba = world_map.rgba;
                let alt = world_map.alt;
                if rgba.size() != map_size.map(|e| e as i32) {
                    return Err(Error::Other("Server sent a bad world map image".into()));
                }
                if alt.size() != map_size.map(|e| e as i32) {
                    return Err(Error::Other("Server sent a bad altitude map.".into()));
                }
                let [west, east] = world_map.horizons;
                let scale_angle =
                    |a: u8| (a as f32 / 255.0 * <f32 as FloatConst>::FRAC_PI_2()).tan();
                let scale_height = |h: u8| h as f32 / 255.0 * max_height;
                let scale_height_big = |h: u32| (h >> 3) as f32 / 8191.0 * max_height;
                ping_stream.send(PingMsg::Ping)?;

                debug!("Preparing image...");
                let unzip_horizons = |(angles, heights): &(Vec<_>, Vec<_>)| {
                    (
                        angles.iter().copied().map(scale_angle).collect::<Vec<_>>(),
                        heights
                            .iter()
                            .copied()
                            .map(scale_height)
                            .collect::<Vec<_>>(),
                    )
                };
                let horizons = [unzip_horizons(&west), unzip_horizons(&east)];

                // Redraw map (with shadows this time).
                let mut world_map_rgba = vec![0u32; rgba.size().product() as usize];
                let mut world_map_topo = vec![0u32; rgba.size().product() as usize];
                let mut map_config = common::terrain::map::MapConfig::orthographic(
                    map_size_lg,
                    core::ops::RangeInclusive::new(0.0, max_height),
                );
                map_config.horizons = Some(&horizons);
                let rescale_height = |h: f32| h / max_height;
                let bounds_check = |pos: Vec2<i32>| {
                    pos.reduce_partial_min() >= 0
                        && pos.x < map_size.x as i32
                        && pos.y < map_size.y as i32
                };
                ping_stream.send(PingMsg::Ping)?;
                fn sample_pos(
                    map_config: &MapConfig,
                    pos: Vec2<i32>,
                    alt: &Grid<u32>,
                    rgba: &Grid<u32>,
                    map_size: &Vec2<u16>,
                    map_size_lg: &common::terrain::MapSizeLg,
                    max_height: f32,
                ) -> common::terrain::map::MapSample {
                    let rescale_height = |h: f32| h / max_height;
                    let scale_height_big = |h: u32| (h >> 3) as f32 / 8191.0 * max_height;
                    let bounds_check = |pos: Vec2<i32>| {
                        pos.reduce_partial_min() >= 0
                            && pos.x < map_size.x as i32
                            && pos.y < map_size.y as i32
                    };
                    let MapConfig {
                        gain,
                        is_contours,
                        is_height_map,
                        is_stylized_topo,
                        ..
                    } = *map_config;
                    let mut is_contour_line = false;
                    let mut is_border = false;
                    let (rgb, alt, downhill_wpos) = if bounds_check(pos) {
                        let posi = pos.y as usize * map_size.x as usize + pos.x as usize;
                        let [r, g, b, _a] = rgba[pos].to_le_bytes();
                        let is_water = r == 0 && b > 102 && g < 77;
                        let alti = alt[pos];
                        // Compute contours (chunks are assigned in the river code below)
                        let altj = rescale_height(scale_height_big(alti));
                        let contour_interval = 150.0;
                        let chunk_contour = (altj * gain / contour_interval) as u32;

                        // Compute downhill.
                        let downhill = {
                            let mut best = -1;
                            let mut besth = alti;
                            for nposi in neighbors(*map_size_lg, posi) {
                                let nbh = alt.raw()[nposi];
                                let nalt = rescale_height(scale_height_big(nbh));
                                let nchunk_contour = (nalt * gain / contour_interval) as u32;
                                if !is_contour_line && chunk_contour > nchunk_contour {
                                    is_contour_line = true;
                                }
                                let [nr, ng, nb, _na] = rgba.raw()[nposi].to_le_bytes();
                                let n_is_water = nr == 0 && nb > 102 && ng < 77;

                                if !is_border && is_water && !n_is_water {
                                    is_border = true;
                                }

                                if nbh < besth {
                                    besth = nbh;
                                    best = nposi as isize;
                                }
                            }
                            best
                        };
                        let downhill_wpos = if downhill < 0 {
                            None
                        } else {
                            Some(
                                Vec2::new(
                                    (downhill as usize % map_size.x as usize) as i32,
                                    (downhill as usize / map_size.x as usize) as i32,
                                ) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
                            )
                        };
                        (Rgb::new(r, g, b), alti, downhill_wpos)
                    } else {
                        (Rgb::zero(), 0, None)
                    };
                    let alt = f64::from(rescale_height(scale_height_big(alt)));
                    let wpos = pos * TerrainChunkSize::RECT_SIZE.map(|e| e as i32);
                    let downhill_wpos = downhill_wpos
                        .unwrap_or(wpos + TerrainChunkSize::RECT_SIZE.map(|e| e as i32));
                    let is_path = rgb.r == 0x37 && rgb.g == 0x29 && rgb.b == 0x23;
                    let rgb = rgb.map(|e: u8| e as f64 / 255.0);
                    let is_water = rgb.r == 0.0 && rgb.b > 0.4 && rgb.g < 0.3;

                    let rgb = if is_height_map {
                        if is_path {
                            // Path color is Rgb::new(0x37, 0x29, 0x23)
                            Rgb::new(0.9, 0.9, 0.63)
                        } else if is_water {
                            Rgb::new(0.23, 0.47, 0.53)
                        } else if is_contours && is_contour_line {
                            // Color contour lines
                            Rgb::new(0.15, 0.15, 0.15)
                        } else {
                            // Color hill shading
                            let lightness = (alt + 0.2).min(1.0) as f64;
                            Rgb::new(lightness, 0.9 * lightness, 0.5 * lightness)
                        }
                    } else if is_stylized_topo {
                        if is_path {
                            Rgb::new(0.9, 0.9, 0.63)
                        } else if is_water {
                            if is_border {
                                Rgb::new(0.10, 0.34, 0.50)
                            } else {
                                Rgb::new(0.23, 0.47, 0.63)
                            }
                        } else if is_contour_line {
                            Rgb::new(0.25, 0.25, 0.25)
                        } else {
                            // Stylized colors
                            Rgb::new(
                                (rgb.r + 0.25).min(1.0),
                                (rgb.g + 0.23).min(1.0),
                                (rgb.b + 0.10).min(1.0),
                            )
                        }
                    } else {
                        Rgb::new(rgb.r, rgb.g, rgb.b)
                    }
                    .map(|e| (e * 255.0) as u8);
                    common::terrain::map::MapSample {
                        rgb,
                        alt,
                        downhill_wpos,
                        connections: None,
                    }
                }
                // Generate standard shaded map
                map_config.is_shaded = true;
                map_config.generate(
                    |pos| {
                        sample_pos(
                            &map_config,
                            pos,
                            &alt,
                            &rgba,
                            &map_size,
                            &map_size_lg,
                            max_height,
                        )
                    },
                    |wpos| {
                        let pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, f| e / f as i32);
                        rescale_height(if bounds_check(pos) {
                            scale_height_big(alt[pos])
                        } else {
                            0.0
                        })
                    },
                    |pos, (r, g, b, a)| {
                        world_map_rgba[pos.y * map_size.x as usize + pos.x] =
                            u32::from_le_bytes([r, g, b, a]);
                    },
                );
                // Generate map with topographical lines and stylized colors
                map_config.is_contours = true;
                map_config.is_stylized_topo = true;
                map_config.generate(
                    |pos| {
                        sample_pos(
                            &map_config,
                            pos,
                            &alt,
                            &rgba,
                            &map_size,
                            &map_size_lg,
                            max_height,
                        )
                    },
                    |wpos| {
                        let pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, f| e / f as i32);
                        rescale_height(if bounds_check(pos) {
                            scale_height_big(alt[pos])
                        } else {
                            0.0
                        })
                    },
                    |pos, (r, g, b, a)| {
                        world_map_topo[pos.y * map_size.x as usize + pos.x] =
                            u32::from_le_bytes([r, g, b, a]);
                    },
                );
                ping_stream.send(PingMsg::Ping)?;
                let make_raw = |rgb| -> Result<_, Error> {
                    let mut raw = vec![0u8; 4 * world_map_rgba.len()];
                    LittleEndian::write_u32_into(rgb, &mut raw);
                    Ok(Arc::new(
                        image::DynamicImage::ImageRgba8({
                            // Should not fail if the dimensions are correct.
                            let map =
                                image::ImageBuffer::from_raw(u32::from(map_size.x), u32::from(map_size.y), raw);
                            map.ok_or_else(|| Error::Other("Server sent a bad world map image".into()))?
                        })
                        // Flip the image, since Voxygen uses an orientation where rotation from
                        // positive x axis to positive y axis is counterclockwise around the z axis.
                        .flipv(),
                    ))
                };
                ping_stream.send(PingMsg::Ping)?;
                let lod_base = rgba;
                let lod_alt = alt;
                let world_map_rgb_img = make_raw(&world_map_rgba)?;
                let world_map_topo_img = make_raw(&world_map_topo)?;
                let world_map_layers = vec![world_map_rgb_img, world_map_topo_img];
                let horizons = (west.0, west.1, east.0, east.1)
                    .into_par_iter()
                    .map(|(wa, wh, ea, eh)| u32::from_le_bytes([wa, wh, ea, eh]))
                    .collect::<Vec<_>>();
                let lod_horizon = horizons;
                let map_bounds = Vec2::new(sea_level, max_height);
                debug!("Done preparing image...");

                Ok((
                    state,
                    lod_base,
                    lod_alt,
                    Grid::from_raw(map_size.map(|e| e as i32), lod_horizon),
                    (world_map_layers, map_size, map_bounds),
                    world_map.sites,
                    world_map.pois,
                    recipe_book,
                    component_recipe_book,
                    max_group_size,
                    client_timeout,
                ))
            },
            ServerInit::TooManyPlayers => Err(Error::TooManyPlayers),
        }?;
        ping_stream.send(PingMsg::Ping)?;

        debug!("Initial sync done");

        Ok(Self {
            registered: false,
            presence: None,
            runtime,
            server_info,
            world_data: WorldData {
                lod_base,
                lod_alt,
                lod_horizon,
                map: world_map,
            },
            player_list: HashMap::new(),
            character_list: CharacterList::default(),
            sites: sites
                .iter()
                .map(|s| {
                    (s.id, SiteInfoRich {
                        site: s.clone(),
                        economy: None,
                    })
                })
                .collect(),
            pois,
            recipe_book,
            component_recipe_book,
            available_recipes: HashMap::default(),
            chat_mode: ChatMode::default(),

            lod_zones: HashMap::new(),
            lod_last_requested: None,

            max_group_size,
            invite: None,
            group_leader: None,
            group_members: HashMap::new(),
            pending_invites: HashSet::new(),
            pending_trade: None,

            local_command_gen: CommandGenerator::default(),
            next_control: Controller::default(),
            inter_tick_rewind_time: None,
            _rewind_fluctuation_budget: 0.0,

            network: Some(network),
            participant: Some(participant),
            general_stream: stream,
            ping_stream,
            register_stream,
            character_screen_stream,
            in_game_stream,
            terrain_stream,

            client_timeout,

            last_server_ping: 0.0,
            last_server_pong: 0.0,
            last_ping_delta: 0.0,
            ping_deltas: VecDeque::new(),

            tick: 0,
            state,
            view_distance: None,
            lod_distance: 4.0,
            loaded_distance: 0.0,

            pending_chunks: HashMap::new(),
            target_time_of_day: None,
        })
    }

    /// Request a state transition to `ClientState::Registered`.
    pub async fn register(
        &mut self,
        username: String,
        password: String,
        mut auth_trusted: impl FnMut(&str) -> bool,
    ) -> Result<(), Error> {
        // Authentication
        let token_or_username = match &self.server_info.auth_provider {
            Some(addr) => {
                // Query whether this is a trusted auth server
                if auth_trusted(addr) {
                    let (scheme, authority) = match addr.split_once("://") {
                        Some((s, a)) => (s, a),
                        None => return Err(Error::AuthServerUrlInvalid(addr.to_string())),
                    };

                    let scheme = match scheme.parse::<authc::Scheme>() {
                        Ok(s) => s,
                        Err(_) => return Err(Error::AuthServerUrlInvalid(addr.to_string())),
                    };

                    let authority = match authority.parse::<authc::Authority>() {
                        Ok(a) => a,
                        Err(_) => return Err(Error::AuthServerUrlInvalid(addr.to_string())),
                    };

                    Ok(authc::AuthClient::new(scheme, authority)?
                        .sign_in(&username, &password)
                        .await?
                        .serialize())
                } else {
                    Err(Error::AuthServerNotTrusted)
                }
            },
            None => Ok(username),
        }?;

        self.send_msg_err(ClientRegister { token_or_username })?;

        match self.register_stream.recv::<ServerRegisterAnswer>().await? {
            Err(RegisterError::AuthError(err)) => Err(Error::AuthErr(err)),
            Err(RegisterError::InvalidCharacter) => Err(Error::InvalidCharacter),
            Err(RegisterError::NotOnWhitelist) => Err(Error::NotOnWhitelist),
            Err(RegisterError::Kicked(err)) => Err(Error::Kicked(err)),
            Err(RegisterError::Banned(reason)) => Err(Error::Banned(reason)),
            Ok(()) => {
                self.registered = true;
                Ok(())
            },
        }
    }

    fn send_msg_err<S>(&mut self, msg: S) -> Result<(), network::StreamError>
    where
        S: Into<ClientMsg>,
    {
        prof_span!("send_msg_err");
        let msg: ClientMsg = msg.into();
        #[cfg(debug_assertions)]
        {
            const C_TYPE: ClientType = ClientType::Game;
            let verified = msg.verify(C_TYPE, self.registered, self.presence);

            // Due to the fact that character loading is performed asynchronously after
            // initial connect it is possible to receive messages after a character load
            // error while in the wrong state.
            if !verified {
                warn!(
                    "Received ClientType::Game message when not in game (Registered: {} Presence: \
                     {:?}), dropping message: {:?} ",
                    self.registered, self.presence, msg
                );
                return Ok(());
            }
        }
        match msg {
            ClientMsg::Type(msg) => self.register_stream.send(msg),
            ClientMsg::Register(msg) => self.register_stream.send(msg),
            ClientMsg::General(msg) => {
                #[cfg(feature = "tracy")]
                let (mut ingame, mut terrain) = (0.0, 0.0);
                let stream = match msg {
                    ClientGeneral::RequestCharacterList
                    | ClientGeneral::CreateCharacter { .. }
                    | ClientGeneral::EditCharacter { .. }
                    | ClientGeneral::DeleteCharacter(_)
                    | ClientGeneral::Character(_)
                    | ClientGeneral::Spectate => &mut self.character_screen_stream,
                    //Only in game
                    ClientGeneral::Control(_)
                    | ClientGeneral::SetViewDistance(_)
                    | ClientGeneral::BreakBlock(_)
                    | ClientGeneral::PlaceBlock(_, _)
                    | ClientGeneral::ExitInGame
                    | ClientGeneral::UnlockSkill(_)
                    | ClientGeneral::RequestSiteInfo(_)
                    | ClientGeneral::UnlockSkillGroup(_)
                    | ClientGeneral::RequestLossyTerrainCompression { .. }
                    | ClientGeneral::AcknowledgePersistenceLoadError
                    | ClientGeneral::UpdateMapMarker(_) => {
                        #[cfg(feature = "tracy")]
                        {
                            ingame = 1.0;
                        }
                        &mut self.in_game_stream
                    },
                    //Only in game, terrain
                    ClientGeneral::TerrainChunkRequest { .. }
                    | ClientGeneral::LodZoneRequest { .. } => {
                        #[cfg(feature = "tracy")]
                        {
                            terrain = 1.0;
                        }
                        &mut self.terrain_stream
                    },
                    //Always possible
                    ClientGeneral::ChatMsg(_)
                    | ClientGeneral::Command(_, _)
                    | ClientGeneral::Terminate => &mut self.general_stream,
                };
                #[cfg(feature = "tracy")]
                {
                    plot!("ingame_sends", ingame);
                    plot!("terrain_sends", terrain);
                }
                stream.send(msg)
            },
            ClientMsg::Ping(msg) => self.ping_stream.send(msg),
        }
    }

    pub fn request_lossy_terrain_compression(&mut self, lossy_terrain_compression: bool) {
        self.send_msg(ClientGeneral::RequestLossyTerrainCompression {
            lossy_terrain_compression,
        })
    }

    fn send_msg<S>(&mut self, msg: S)
    where
        S: Into<ClientMsg>,
    {
        let res = self.send_msg_err(msg);
        if let Err(e) = res {
            warn!(
                ?e,
                "connection to server no longer possible, couldn't send msg"
            );
        }
    }

    /// Request a state transition to `ClientState::Character`.
    pub fn request_character(&mut self, character_id: CharacterId) {
        self.send_msg(ClientGeneral::Character(character_id));

        //Assume we are in_game unless server tells us otherwise
        self.presence = Some(PresenceKind::Character(character_id));
    }

    /// Load the current players character list
    pub fn load_character_list(&mut self) {
        self.character_list.loading = true;
        self.send_msg(ClientGeneral::RequestCharacterList);
    }

    /// New character creation
    pub fn create_character(
        &mut self,
        alias: String,
        mainhand: Option<String>,
        offhand: Option<String>,
        body: comp::Body,
    ) {
        self.character_list.loading = true;
        self.send_msg(ClientGeneral::CreateCharacter {
            alias,
            mainhand,
            offhand,
            body,
        });
    }

    pub fn edit_character(&mut self, alias: String, id: CharacterId, body: comp::Body) {
        self.character_list.loading = true;
        self.send_msg(ClientGeneral::EditCharacter { alias, id, body });
    }

    /// Character deletion
    pub fn delete_character(&mut self, character_id: CharacterId) {
        self.character_list.loading = true;
        self.send_msg(ClientGeneral::DeleteCharacter(character_id));
    }

    /// Send disconnect message to the server
    pub fn logout(&mut self) {
        debug!("Sending logout from server");
        self.send_msg(ClientGeneral::Terminate);
        self.registered = false;
        self.presence = None;
    }

    /// Request a state transition to `ClientState::Registered` from an ingame
    /// state.
    pub fn request_remove_character(&mut self) {
        self.chat_mode = ChatMode::World;
        self.send_msg(ClientGeneral::ExitInGame);
    }

    pub fn set_view_distance(&mut self, view_distance: u32) {
        let view_distance = view_distance.max(1).min(65);
        self.view_distance = Some(view_distance);
        self.send_msg(ClientGeneral::SetViewDistance(view_distance));
    }

    pub fn set_lod_distance(&mut self, lod_distance: u32) {
        let lod_distance = lod_distance.max(0).min(1000) as f32 / lod::ZONE_SIZE as f32;
        self.lod_distance = lod_distance;
    }

    pub fn use_slot(&mut self, slot: Slot) {
        self.control_action(ControlAction::InventoryAction(InventoryAction::Use(slot)))
    }

    pub fn swap_slots(&mut self, a: Slot, b: Slot) {
        match (a, b) {
            (Slot::Equip(equip), slot) | (slot, Slot::Equip(equip)) => self.control_action(
                ControlAction::InventoryAction(InventoryAction::Swap(equip, slot)),
            ),
            (Slot::Inventory(inv1), Slot::Inventory(inv2)) => {
                self.next_control
                    .events
                    .push(ControlEvent::InventoryEvent(InventoryEvent::Swap(
                        inv1, inv2,
                    )));
            },
        }
    }

    pub fn drop_slot(&mut self, slot: Slot) {
        match slot {
            Slot::Equip(equip) => {
                self.control_action(ControlAction::InventoryAction(InventoryAction::Drop(equip)))
            },
            Slot::Inventory(inv) => self
                .next_control
                .events
                .push(ControlEvent::InventoryEvent(InventoryEvent::Drop(inv))),
        }
    }

    pub fn sort_inventory(&mut self) {
        self.control_action(ControlAction::InventoryAction(InventoryAction::Sort));
    }

    pub fn perform_trade_action(&mut self, action: TradeAction) {
        if let Some((id, _, _)) = self.pending_trade {
            if let TradeAction::Decline = action {
                self.pending_trade.take();
            }
            self.next_control
                .events
                .push(ControlEvent::PerformTradeAction(id, action));
        }
    }

    pub fn is_dead(&self) -> bool { self.current::<comp::Health>().map_or(false, |h| h.is_dead) }

    pub fn is_gliding(&self) -> bool {
        self.current::<CharacterState>()
            .map_or(false, |cs| matches!(cs, CharacterState::Glide(_)))
    }

    pub fn split_swap_slots(&mut self, a: comp::slot::Slot, b: comp::slot::Slot) {
        match (a, b) {
            (Slot::Equip(equip), slot) | (slot, Slot::Equip(equip)) => self.control_action(
                ControlAction::InventoryAction(InventoryAction::Swap(equip, slot)),
            ),
            (Slot::Inventory(inv1), Slot::Inventory(inv2)) => self.next_control.events.push(
                ControlEvent::InventoryEvent(InventoryEvent::SplitSwap(inv1, inv2)),
            ),
        }
    }

    pub fn split_drop_slot(&mut self, slot: comp::slot::Slot) {
        match slot {
            Slot::Equip(equip) => {
                self.control_action(ControlAction::InventoryAction(InventoryAction::Drop(equip)))
            },
            Slot::Inventory(inv) => self
                .next_control
                .events
                .push(ControlEvent::InventoryEvent(InventoryEvent::SplitDrop(inv))),
        }
    }

    pub fn pick_up(&mut self, entity: EcsEntity) {
        // Get the health component from the entity

        if let Some(uid) = self.state.read_component_copied(entity) {
            // If we're dead, exit before sending the message
            if self.is_dead() {
                return;
            }
            self.next_control
                .events
                .push(ControlEvent::InventoryEvent(InventoryEvent::Pickup(uid)));
        }
    }

    pub fn npc_interact(&mut self, npc_entity: EcsEntity) {
        // If we're dead, exit before sending message
        if self.is_dead() {
            return;
        }

        if let Some(uid) = self.state.read_component_copied(npc_entity) {
            self.next_control.events.push(ControlEvent::Interact(uid));
        }
    }

    pub fn player_list(&self) -> &HashMap<Uid, PlayerInfo> { &self.player_list }

    pub fn character_list(&self) -> &CharacterList { &self.character_list }

    pub fn server_info(&self) -> &ServerInfo { &self.server_info }

    pub fn world_data(&self) -> &WorldData { &self.world_data }

    pub fn recipe_book(&self) -> &RecipeBook { &self.recipe_book }

    pub fn component_recipe_book(&self) -> &ComponentRecipeBook { &self.component_recipe_book }

    pub fn available_recipes(&self) -> &HashMap<String, Option<SpriteKind>> {
        &self.available_recipes
    }

    pub fn lod_zones(&self) -> &HashMap<Vec2<i32>, lod::Zone> { &self.lod_zones }

    /// Returns whether the specified recipe can be crafted and the sprite, if
    /// any, that is required to do so.
    pub fn can_craft_recipe(&self, recipe: &str) -> (bool, Option<SpriteKind>) {
        self.recipe_book
            .get(recipe)
            .zip(self.inventories().get(self.entity()))
            .map(|(recipe, inv)| {
                (
                    recipe.inventory_contains_ingredients(inv).is_ok(),
                    recipe.craft_sprite,
                )
            })
            .unwrap_or((false, None))
    }

    pub fn craft_recipe(
        &mut self,
        recipe: &str,
        slots: Vec<(u32, InvSlotId)>,
        craft_sprite: Option<(Vec3<i32>, SpriteKind)>,
    ) -> bool {
        let (can_craft, required_sprite) = self.can_craft_recipe(recipe);
        let has_sprite = required_sprite.map_or(true, |s| Some(s) == craft_sprite.map(|(_, s)| s));
        if can_craft && has_sprite {
            self.next_control.events.push(ControlEvent::InventoryEvent(
                InventoryEvent::CraftRecipe {
                    craft_event: CraftEvent::Simple {
                        recipe: recipe.to_string(),
                        slots,
                    },
                    craft_sprite: craft_sprite.map(|(pos, _)| pos),
                },
            ));
            true
        } else {
            false
        }
    }

    /// Checks if the item in the given slot can be salvaged.
    pub fn can_salvage_item(&self, slot: InvSlotId) -> bool {
        self.inventories()
            .get(self.entity())
            .and_then(|inv| inv.get(slot))
            .map_or(false, |item| item.is_salvageable())
    }

    /// Salvage the item in the given inventory slot. `salvage_pos` should be
    /// the location of a relevant crafting station within range of the player.
    pub fn salvage_item(&mut self, slot: InvSlotId, salvage_pos: Vec3<i32>) -> bool {
        let is_salvageable = self.can_salvage_item(slot);
        if is_salvageable {
            self.next_control.events.push(ControlEvent::InventoryEvent(
                InventoryEvent::CraftRecipe {
                    craft_event: CraftEvent::Salvage(slot),
                    craft_sprite: Some(salvage_pos),
                },
            ));
        }
        is_salvageable
    }

    /// Crafts modular weapon from components in the provided slots.
    /// `sprite_pos` should be the location of the necessary crafting station in
    /// range of the player.
    /// Returns whether or not the networking event was sent (which is based on
    /// whether the player has two modular components in the provided slots)
    pub fn craft_modular_weapon(
        &mut self,
        primary_component: InvSlotId,
        secondary_component: InvSlotId,
        sprite_pos: Option<Vec3<i32>>,
    ) -> bool {
        let inventories = self.inventories();
        let inventory = inventories.get(self.entity());

        enum ModKind {
            Primary,
            Secondary,
        }

        // Closure to get inner modular component info from item in a given slot
        let mod_kind = |slot| match inventory
            .and_then(|inv| inv.get(slot).map(|item| item.kind()))
            .as_deref()
        {
            Some(ItemKind::ModularComponent(modular::ModularComponent::ToolPrimaryComponent {
                ..
            })) => Some(ModKind::Primary),
            Some(ItemKind::ModularComponent(
                modular::ModularComponent::ToolSecondaryComponent { .. },
            )) => Some(ModKind::Secondary),
            _ => None,
        };

        if let (Some(ModKind::Primary), Some(ModKind::Secondary)) =
            (mod_kind(primary_component), mod_kind(secondary_component))
        {
            drop(inventories);
            self.next_control.events.push(ControlEvent::InventoryEvent(
                InventoryEvent::CraftRecipe {
                    craft_event: CraftEvent::ModularWeapon {
                        primary_component,
                        secondary_component,
                    },
                    craft_sprite: sprite_pos,
                },
            ));
            true
        } else {
            false
        }
    }

    pub fn craft_modular_weapon_component(
        &mut self,
        toolkind: tool::ToolKind,
        material: InvSlotId,
        modifier: Option<InvSlotId>,
        slots: Vec<(u32, InvSlotId)>,
        sprite_pos: Option<Vec3<i32>>,
    ) {
        self.next_control
            .events
            .push(ControlEvent::InventoryEvent(InventoryEvent::CraftRecipe {
                craft_event: CraftEvent::ModularWeaponPrimaryComponent {
                    toolkind,
                    material,
                    modifier,
                    slots,
                },
                craft_sprite: sprite_pos,
            }));
    }

    fn update_available_recipes(&mut self) {
        self.available_recipes = self
            .recipe_book
            .iter()
            .map(|(name, _)| name.clone())
            .filter_map(|name| {
                let (can_craft, required_sprite) = self.can_craft_recipe(&name);
                if can_craft {
                    Some((name, required_sprite))
                } else {
                    None
                }
            })
            .collect();
    }

    /// Unstable, likely to be removed in a future release
    pub fn sites(&self) -> &HashMap<SiteId, SiteInfoRich> { &self.sites }

    /// Unstable, likely to be removed in a future release
    pub fn pois(&self) -> &Vec<PoiInfo> { &self.pois }

    pub fn sites_mut(&mut self) -> &mut HashMap<SiteId, SiteInfoRich> { &mut self.sites }

    pub fn enable_lantern(&mut self) { self.next_control.events.push(ControlEvent::EnableLantern); }

    pub fn disable_lantern(&mut self) {
        self.next_control.events.push(ControlEvent::DisableLantern);
    }

    pub fn remove_buff(&mut self, buff_id: BuffKind) {
        self.next_control
            .events
            .push(ControlEvent::RemoveBuff(buff_id));
    }

    pub fn unlock_skill(&mut self, skill: Skill) {
        self.send_msg(ClientGeneral::UnlockSkill(skill));
    }

    pub fn max_group_size(&self) -> u32 { self.max_group_size }

    pub fn invite(&self) -> Option<(Uid, std::time::Instant, std::time::Duration, InviteKind)> {
        self.invite
    }

    pub fn group_info(&self) -> Option<(String, Uid)> {
        self.group_leader.map(|l| ("Group".into(), l)) // TODO
    }

    pub fn group_members(&self) -> &HashMap<Uid, group::Role> { &self.group_members }

    pub fn pending_invites(&self) -> &HashSet<Uid> { &self.pending_invites }

    pub fn pending_trade(&self) -> &Option<(TradeId, PendingTrade, Option<SitePrices>)> {
        &self.pending_trade
    }

    pub fn is_trading(&self) -> bool { self.pending_trade.is_some() }

    pub fn send_invite(&mut self, invitee: Uid, kind: InviteKind) {
        self.next_control
            .events
            .push(ControlEvent::InitiateInvite(invitee, kind));
    }

    pub fn accept_invite(&mut self) {
        // Clear invite
        self.invite.take();
        self.next_control
            .events
            .push(ControlEvent::InviteResponse(InviteResponse::Accept));
    }

    pub fn decline_invite(&mut self) {
        // Clear invite
        self.invite.take();
        self.next_control
            .events
            .push(ControlEvent::InviteResponse(InviteResponse::Decline));
    }

    pub fn leave_group(&mut self) {
        self.next_control
            .events
            .push(ControlEvent::GroupManip(GroupManip::Leave));
    }

    pub fn kick_from_group(&mut self, uid: Uid) {
        self.next_control
            .events
            .push(ControlEvent::GroupManip(GroupManip::Kick(uid)));
    }

    pub fn assign_group_leader(&mut self, uid: Uid) {
        self.next_control
            .events
            .push(ControlEvent::GroupManip(GroupManip::AssignLeader(uid)));
    }

    pub fn is_riding(&self) -> bool {
        self.state
            .ecs()
            .read_storage::<Is<Rider>>()
            .get(self.entity())
            .is_some()
    }

    pub fn is_lantern_enabled(&self) -> bool {
        self.state
            .ecs()
            .read_storage::<comp::LightEmitter>()
            .get(self.entity())
            .is_some()
    }

    pub fn mount(&mut self, entity: EcsEntity) {
        if let Some(uid) = self.state.read_component_copied(entity) {
            self.next_control.events.push(ControlEvent::Mount(uid));
        }
    }

    pub fn unmount(&mut self) { self.next_control.events.push(ControlEvent::Unmount); }

    pub fn respawn(&mut self) {
        if self
            .state
            .ecs()
            .read_storage::<comp::Health>()
            .get(self.entity())
            .map_or(false, |h| h.is_dead)
        {
            self.next_control.events.push(ControlEvent::Respawn);
        }
    }

    pub fn map_marker_event(&mut self, event: MapMarkerChange) {
        self.send_msg(ClientGeneral::UpdateMapMarker(event));
    }

    /// Checks whether a player can swap their weapon+ability `Loadout` settings
    /// and sends the `ControlAction` event that signals to do the swap.
    pub fn swap_loadout(&mut self) { self.control_action(ControlAction::SwapEquippedWeapons) }

    /// Determine whether the player is wielding, if they're even capable of
    /// being in a wield state.
    pub fn is_wielding(&self) -> Option<bool> {
        self.state
            .ecs()
            .read_storage::<CharacterState>()
            .get(self.entity())
            .map(|cs| cs.is_wield())
    }

    pub fn toggle_wield(&mut self) {
        match self.is_wielding() {
            Some(true) => self.control_action(ControlAction::Unwield),
            Some(false) => self.control_action(ControlAction::Wield),
            None => warn!("Can't toggle wield, client entity doesn't have a `CharacterState`"),
        }
    }

    pub fn toggle_sit(&mut self) {
        let is_sitting = self
            .state
            .ecs()
            .read_storage::<CharacterState>()
            .get(self.entity())
            .map(|cs| matches!(cs, CharacterState::Sit));

        match is_sitting {
            Some(true) => self.control_action(ControlAction::Stand),
            Some(false) => self.control_action(ControlAction::Sit),
            None => warn!("Can't toggle sit, client entity doesn't have a `CharacterState`"),
        }
    }

    pub fn toggle_dance(&mut self) {
        let is_dancing = self
            .state
            .ecs()
            .read_storage::<CharacterState>()
            .get(self.entity())
            .map(|cs| matches!(cs, CharacterState::Dance));

        match is_dancing {
            Some(true) => self.control_action(ControlAction::Stand),
            Some(false) => self.control_action(ControlAction::Dance),
            None => warn!("Can't toggle dance, client entity doesn't have a `CharacterState`"),
        }
    }

    pub fn utter(&mut self, kind: UtteranceKind) {
        self.next_control.events.push(ControlEvent::Utterance(kind));
    }

    pub fn toggle_sneak(&mut self) {
        let is_sneaking = self
            .state
            .ecs()
            .read_storage::<CharacterState>()
            .get(self.entity())
            .map(CharacterState::is_stealthy);

        match is_sneaking {
            Some(true) => self.control_action(ControlAction::Stand),
            Some(false) => self.control_action(ControlAction::Sneak),
            None => warn!("Can't toggle sneak, client entity doesn't have a `CharacterState`"),
        }
    }

    pub fn toggle_glide(&mut self) {
        let using_glider = self
            .state
            .ecs()
            .read_storage::<CharacterState>()
            .get(self.entity())
            .map(|cs| matches!(cs, CharacterState::GlideWield(_) | CharacterState::Glide(_)));

        match using_glider {
            Some(true) => self.control_action(ControlAction::Unwield),
            Some(false) => self.control_action(ControlAction::GlideWield),
            None => warn!("Can't toggle glide, client entity doesn't have a `CharacterState`"),
        }
    }

    pub fn handle_input(
        &mut self,
        input: InputKind,
        pressed: bool,
        select_pos: Option<Vec3<f32>>,
        target_entity: Option<EcsEntity>,
    ) {
        if pressed {
            self.control_action(ControlAction::StartInput {
                input,
                target_entity: target_entity.and_then(|e| self.state.read_component_copied(e)),
                select_pos,
            });
        } else {
            self.control_action(ControlAction::CancelInput(input));
        }
    }

    fn control_action(&mut self, control_action: ControlAction) {
        self.next_control.actions.push(control_action);
    }

    pub fn view_distance(&self) -> Option<u32> { self.view_distance }

    pub fn loaded_distance(&self) -> f32 { self.loaded_distance }

    pub fn position(&self) -> Option<Vec3<f32>> {
        self.state
            .read_storage::<comp::Pos>()
            .get(self.entity())
            .map(|v| v.0)
    }

    pub fn current_chunk(&self) -> Option<Arc<TerrainChunk>> {
        let chunk_pos = Vec2::from(self.position()?)
            .map2(TerrainChunkSize::RECT_SIZE, |e: f32, sz| {
                (e as u32).div_euclid(sz) as i32
            });

        self.state.terrain().get_key_arc(chunk_pos).cloned()
    }

    pub fn current<C: Component>(&self) -> Option<C>
    where
        C: Clone,
    {
        self.state.read_storage::<C>().get(self.entity()).cloned()
    }

    pub fn current_biome(&self) -> BiomeKind {
        match self.current_chunk() {
            Some(chunk) => chunk.meta().biome(),
            _ => BiomeKind::Void,
        }
    }

    pub fn current_site(&self) -> SitesKind {
        let mut player_alt = 0.0;
        if let Some(position) = self.current::<comp::Pos>() {
            player_alt = position.0.z;
        }
        let mut contains_cave = false;
        let mut terrain_alt = 0.0;
        let mut contains_dungeon = false;
        let mut contains_settlement = false;
        if let Some(chunk) = self.current_chunk() {
            terrain_alt = chunk.meta().alt();
            contains_cave = chunk.meta().contains_cave();
            contains_dungeon = chunk.meta().contains_dungeon();
            contains_settlement = chunk.meta().contains_settlement();
        }
        if player_alt < (terrain_alt - 25.0) && contains_cave {
            SitesKind::Cave
        } else if player_alt < (terrain_alt - 25.0) && contains_dungeon {
            SitesKind::Dungeon
        } else if contains_settlement {
            SitesKind::Settlement
        } else {
            SitesKind::Void
        }
    }

    pub fn request_site_economy(&mut self, id: SiteId) {
        self.send_msg(ClientGeneral::RequestSiteInfo(id))
    }

    pub fn inventories(&self) -> ReadStorage<comp::Inventory> { self.state.read_storage() }

    /// Send a chat message to the server.
    pub fn send_chat(&mut self, message: String) {
        match validate_chat_msg(&message) {
            Ok(()) => self.send_msg(ClientGeneral::ChatMsg(message)),
            Err(ChatMsgValidationError::TooLong) => tracing::warn!(
                "Attempted to send a message that's too long (Over {} bytes)",
                MAX_BYTES_CHAT_MSG
            ),
        }
    }

    /// Send a command to the server.
    pub fn send_command(&mut self, name: String, args: Vec<String>) {
        self.send_msg(ClientGeneral::Command(name, args));
    }

    /// Remove all cached terrain
    pub fn clear_terrain(&mut self) {
        self.state.clear_terrain();
        self.pending_chunks.clear();
    }

    pub fn place_block(&mut self, pos: Vec3<i32>, block: Block) {
        self.send_msg(ClientGeneral::PlaceBlock(pos, block));
    }

    pub fn remove_block(&mut self, pos: Vec3<i32>) {
        self.send_msg(ClientGeneral::BreakBlock(pos));
    }

    pub fn collect_block(&mut self, pos: Vec3<i32>) {
        self.control_action(ControlAction::InventoryAction(InventoryAction::Collect(
            pos,
        )));
    }

    pub fn change_ability(&mut self, slot: usize, new_ability: comp::ability::AuxiliaryAbility) {
        let auxiliary_key = self
            .inventories()
            .get(self.entity())
            .map_or((None, None), |inv| {
                let tool_kind = |slot| {
                    inv.equipped(slot).and_then(|item| match &*item.kind() {
                        comp::item::ItemKind::Tool(tool) => Some(tool.kind),
                        _ => None,
                    })
                };

                (
                    tool_kind(EquipSlot::ActiveMainhand),
                    tool_kind(EquipSlot::ActiveOffhand),
                )
            });

        self.next_control.events.push(ControlEvent::ChangeAbility {
            slot,
            auxiliary_key,
            new_ability,
        });
    }

    pub fn acknolwedge_persistence_load_error(&mut self) {
        self.send_msg(ClientGeneral::AcknowledgePersistenceLoadError)
    }

    /// Execute a single client tick, handle input and update the game state by
    /// the given duration.
    pub fn tick(
        &mut self,
        inputs: ControllerInputs,
        dt: Duration,
        add_foreign_systems: impl Fn(&mut DispatcherBuilder),
    ) -> Result<Vec<Event>, Error> {
        span!(_guard, "tick", "Client::tick");
        // This tick function is the centre of the Veloren universe. Most client-side
        // things are managed from here, and as such it's important that it
        // stays organised. Please consult the core developers before making
        // significant changes to this code. Here is the approximate order of
        // things. Please update it as this code changes.
        //
        // 1) Handle messages from the server
        // 2) Collect input from the frontend, apply input effects to the state
        //    of the game
        // 3) Go through any events (timer-driven or otherwise) that need handling
        //    and apply them to the state of the game
        // 4) Perform a single LocalState tick (i.e: update the world and entities
        //    in the world)
        // 5) Go through the terrain update queue and apply all changes
        //    to the terrain
        // 6) Sync information to the server
        // 7) Finish the tick, passing actions of the main thread back
        //    to the frontend

        // 1) Build up a list of events for this frame, to be passed to the frontend.
        let mut frontend_events = Vec::new();
        self.inter_tick_rewind_time = None;

        // Prepare for new events
        {
            prof_span!("Last<CharacterState> comps update");
            let ecs = self.state.ecs();
            let mut last_character_states = ecs.write_storage::<comp::Last<CharacterState>>();
            for (entity, _, character_state) in (
                &ecs.entities(),
                &ecs.read_storage::<comp::Body>(),
                &ecs.read_storage::<CharacterState>(),
            )
                .join()
            {
                if let Some(l) = last_character_states
                    .entry(entity)
                    .ok()
                    .map(|l| l.or_insert_with(|| comp::Last(character_state.clone())))
                    // TODO: since this just updates when the variant changes we should
                    // just store the variant to avoid the clone overhead
                    .filter(|l| !character_state.same_variant(&l.0))
                {
                    *l = comp::Last(character_state.clone());
                }
            }
        }

        // Handle new messages from the server.
        frontend_events.append(&mut self.handle_new_messages()?);

        // Simulate Ahead
        common_base::plot!("recived_time_sync", 0.0);
        if let Some(rewind_time) = self.inter_tick_rewind_time {
            common_base::plot!("recived_time_sync", 1.0);
            let _time = self.state.ecs().read_resource::<Time>().0 as f64;
            let simulate_ahead = self
                .state
                .ecs()
                .read_storage::<RemoteController>()
                .get(self.entity())
                .map(|rc| rc.simulate_ahead())
                .unwrap_or_default();
            // We substract `dt` here, as otherwise we
            // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms,
            // rewind_tick=130ms end_tick=100+130+60=290
            // Tick2: server_time=130ms dt=30ms ping=30 simulate_ahead=130ms,
            // rewind_tick=130ms end_tick=130+130+30=290
            // Tick3: server_time=160ms dt=60ms ping=30 simulate_ahead=130ms,
            // rewind_tick=130ms end_tick=160+130+60=350 with dt substraction
            // Tick1: server_time=100ms dt=60ms ping=30 simulate_ahead=130ms,
            // rewind_tick=70ms end_tick=100+70+60=230 Tick2: server_time=130ms
            // dt=30ms ping=30 simulate_ahead=130ms, rewind_tick=100ms
            // end_tick=130+100+30=260 Tick3: server_time=160ms dt=60ms ping=30
            // simulate_ahead=130ms, rewind_tick=70ms end_tick=160+70+60=290
            let simulate_ahead = simulate_ahead.max(dt) - dt;
            // measurements lead to the effect that smooth_diff is == 0.0 when we add 2
            // server ticks here.
            let simulate_ahead = simulate_ahead + Duration::from_secs_f64(1.0 / 30.0);

            let _strict_end_tick_time =
                simulate_ahead.as_secs_f64() + /*simulated dt of this tick*/dt.as_secs_f64();

            // Simulate_ahead still fluctionates because Server Tick != Client Tick, and we
            // cant control the phase in which the sync happens.
            // In order to dampen it, we calculate the smooth_time and make sure to not
            // derive to much from it
            let smooth_diff = simulate_ahead.as_secs_f64() - rewind_time.as_secs_f64();

            //const WARP_PERCENT: f64 = 0.05; // make sure we end up not further than 5%
            // from the estimated tick let warp_budget
            let simulate_ahead = if smooth_diff / dt.as_secs_f64() > 0.05 {
                // use
                simulate_ahead
            } else {
                simulate_ahead
            };

            common_base::plot!("smooth_diff", smooth_diff);

            //let simulate_ahead = simulate_ahead.max(dt)/* - dt*/;
            //let simulate_ahead = rewind_time.min(simulate_ahead);
            tracing::warn!(?simulate_ahead, ?dt, "simulating ahead again");
            common_base::plot!("rewind_time", rewind_time.as_secs_f64());
            self.state.rewind_tick(
                simulate_ahead,
                |dispatch_builder| {
                    add_rewind_systems(dispatch_builder);
                },
                false,
            );
        }

        common_base::plot!("dt", dt.as_secs_f64());
        self.state.tick(
            dt,
            |dispatch_builder| {
                add_local_systems(dispatch_builder);
                add_foreign_systems(dispatch_builder);
            },
            true,
        );
        let time = self.state.ecs().read_resource::<Time>().0;
        common_base::plot!("tick_afterwards", time);
        let vel = self
            .state
            .ecs()
            .read_storage::<Vel>()
            .get(self.entity())
            .cloned()
            .unwrap_or(Vel(Vec3::zero()));

        common_base::plot!("vel_x_after", vel.0.x as f64);
        common_base::plot!("vel_y_after", vel.0.y as f64);
        let pos = self
            .state
            .ecs()
            .read_storage::<common::comp::Pos>()
            .get(self.entity())
            .cloned()
            .unwrap_or(common::comp::Pos(Vec3::zero()));

        common_base::plot!("pos_x_after", pos.0.x as f64);
        common_base::plot!("pos_y_after", pos.0.y as f64);
        common_base::plot!("pos_z_after", pos.0.z as f64);

        // 2) Handle input from frontend.
        // Pass character actions from frontend input to the player's entity.
        if self.presence.is_some() {
            prof_span!("handle and send inputs");
            self.next_control.inputs = inputs;
            let con = std::mem::take(&mut self.next_control);
            let time = Duration::from_secs_f64(self.state.ecs().read_resource::<Time>().0) + dt;
            let monotonic_time =
                Duration::from_secs_f64(self.state.ecs().read_resource::<MonotonicTime>().0);
            let rcon = self.local_command_gen.gen(time, con);
            let commands = self
                .state
                .ecs()
                .write_storage::<RemoteController>()
                .entry(self.entity())
                .map(|rc| {
                    let rc = rc.or_insert_with(RemoteController::default);
                    rc.push(rcon);
                    rc.prepare_commands(monotonic_time)
                });

            match commands {
                Ok(commands) => self.send_msg_err(ClientGeneral::Control(commands))?,
                Err(e) => {
                    error!(?e, "couldn't create RemoteController for own entity");
                },
            };
        }

        // 3) Update client local data
        // Check if the invite has timed out and remove if so
        if self
            .invite
            .map_or(false, |(_, timeout, dur, _)| timeout.elapsed() > dur)
        {
            self.invite = None;
        }

        // Lerp towards the target time of day - this ensures a smooth transition for
        // large jumps in TimeOfDay such as when using /time
        if let Some(target_tod) = self.target_time_of_day {
            let mut tod = self.state.ecs_mut().write_resource::<TimeOfDay>();
            tod.0 = Lerp::lerp(tod.0, target_tod.0, dt.as_secs_f64());
            if tod.0 >= target_tod.0 {
                self.target_time_of_day = None;
            }
        }

        // 4) Tick the client's LocalState
        self.state.tick(
            dt,
            |dispatch_builder| {
                add_local_systems(dispatch_builder);
                add_foreign_systems(dispatch_builder);
            },
            true,
        );
        // TODO: avoid emitting these in the first place
        self.state
            .ecs()
            .fetch::<EventBus<common::event::ServerEvent>>()
            .recv_all();

        // 5) Terrain
        self.tick_terrain()?;

        // Send a ping to the server once every second
        if self.state.get_time() - self.last_server_ping > 1. {
            self.send_msg_err(PingMsg::Ping)?;
            self.last_server_ping = self.state.get_time();
        }

        /*
        // Output debug metrics
        if log_enabled!(Level::Info) && self.tick % 600 == 0 {
            let metrics = self
                .state
                .terrain()
                .iter()
                .fold(ChonkMetrics::default(), |a, (_, c)| a + c.get_metrics());
            info!("{:?}", metrics);
        }
        */

        // 7) Finish the tick, pass control back to the frontend.
        self.tick += 1;
        Ok(frontend_events)
    }

    /// Clean up the client after a tick.
    pub fn cleanup(&mut self) {
        // Cleanup the local state
        self.state.cleanup();
    }

    /// Handles terrain addition and removal.
    ///
    /// Removes old terrain chunks outside the view distance.
    /// Sends requests for missing chunks within the view distance.
    fn tick_terrain(&mut self) -> Result<(), Error> {
        let pos = self
            .state
            .read_storage::<comp::Pos>()
            .get(self.entity())
            .cloned();
        if let (Some(pos), Some(view_distance)) = (pos, self.view_distance) {
            prof_span!("terrain");
            let chunk_pos = self.state.terrain().pos_key(pos.0.map(|e| e as i32));

            // Remove chunks that are too far from the player.
            let mut chunks_to_remove = Vec::new();
            self.state.terrain().iter().for_each(|(key, _)| {
                // Subtract 2 from the offset before computing squared magnitude
                // 1 for the chunks needed bordering other chunks for meshing
                // 1 as a buffer so that if the player moves back in that direction the chunks
                //   don't need to be reloaded
                if (chunk_pos - key)
                    .map(|e: i32| (e.unsigned_abs()).saturating_sub(2))
                    .magnitude_squared()
                    > view_distance.pow(2)
                {
                    chunks_to_remove.push(key);
                }
            });
            for key in chunks_to_remove {
                self.state.remove_chunk(key);
            }

            let mut current_tick_send_chunk_requests = 0;
            // Request chunks from the server.
            self.loaded_distance = ((view_distance * TerrainChunkSize::RECT_SIZE.x) as f32).powi(2);
            // +1 so we can find a chunk that's outside the vd for better fog
            for dist in 0..view_distance as i32 + 1 {
                // Only iterate through chunks that need to be loaded for circular vd
                // The (dist - 2) explained:
                // -0.5 because a chunk is visible if its corner is within the view distance
                // -0.5 for being able to move to the corner of the current chunk
                // -1 because chunks are not meshed if they don't have all their neighbors
                //     (notice also that view_distance is decreased by 1)
                //     (this subtraction on vd is omitted elsewhere in order to provide
                //     a buffer layer of loaded chunks)
                let top = if 2 * (dist - 2).max(0).pow(2) > (view_distance - 1).pow(2) as i32 {
                    ((view_distance - 1).pow(2) as f32 - (dist - 2).pow(2) as f32)
                        .sqrt()
                        .round() as i32
                        + 1
                } else {
                    dist
                };

                let mut skip_mode = false;
                for i in -top..top + 1 {
                    let keys = [
                        chunk_pos + Vec2::new(dist, i),
                        chunk_pos + Vec2::new(i, dist),
                        chunk_pos + Vec2::new(-dist, i),
                        chunk_pos + Vec2::new(i, -dist),
                    ];

                    for key in keys.iter() {
                        if self.state.terrain().get_key(*key).is_none() {
                            if !skip_mode && !self.pending_chunks.contains_key(key) {
                                const TOTAL_PENDING_CHUNKS_LIMIT: usize = 12;
                                const CURRENT_TICK_PENDING_CHUNKS_LIMIT: usize = 2;
                                if self.pending_chunks.len() < TOTAL_PENDING_CHUNKS_LIMIT
                                    && current_tick_send_chunk_requests
                                        < CURRENT_TICK_PENDING_CHUNKS_LIMIT
                                {
                                    self.send_msg_err(ClientGeneral::TerrainChunkRequest {
                                        key: *key,
                                    })?;
                                    current_tick_send_chunk_requests += 1;
                                    self.pending_chunks.insert(*key, Instant::now());
                                } else {
                                    skip_mode = true;
                                }
                            }

                            let dist_to_player =
                                (self.state.terrain().key_pos(*key).map(|x| x as f32)
                                    + TerrainChunkSize::RECT_SIZE.map(|x| x as f32) / 2.0)
                                    .distance_squared(pos.0.into());

                            if dist_to_player < self.loaded_distance {
                                self.loaded_distance = dist_to_player;
                            }
                        }
                    }
                }
            }
            self.loaded_distance = self.loaded_distance.sqrt()
                - ((TerrainChunkSize::RECT_SIZE.x as f32 / 2.0).powi(2)
                    + (TerrainChunkSize::RECT_SIZE.y as f32 / 2.0).powi(2))
                .sqrt();

            // If chunks are taking too long, assume they're no longer pending.
            let now = Instant::now();
            self.pending_chunks
                .retain(|_, created| now.duration_since(*created) < Duration::from_secs(3));

            // Manage LoD zones
            let lod_zone = pos.0.xy().map(|e| lod::from_wpos(e as i32));

            // Request LoD zones that are in range
            if self
                .lod_last_requested
                .map_or(true, |i| i.elapsed() > Duration::from_secs(5))
            {
                if let Some(rpos) = Spiral2d::new()
                    .take((1 + self.lod_distance.ceil() as i32 * 2).pow(2) as usize)
                    .filter(|rpos| !self.lod_zones.contains_key(&(lod_zone + *rpos)))
                    .min_by_key(|rpos| rpos.magnitude_squared())
                    .filter(|rpos| {
                        rpos.map(|e| e as f32).magnitude() < (self.lod_distance - 0.5).max(0.0)
                    })
                {
                    self.send_msg_err(ClientGeneral::LodZoneRequest {
                        key: lod_zone + rpos,
                    })?;
                    self.lod_last_requested = Some(Instant::now());
                }
            }

            // Cull LoD zones out of range
            self.lod_zones.retain(|p, _| {
                (*p - lod_zone).map(|e| e as f32).magnitude_squared() < self.lod_distance.powi(2)
            });
        }

        Ok(())
    }

    fn handle_server_msg(
        &mut self,
        frontend_events: &mut Vec<Event>,
        msg: ServerGeneral,
    ) -> Result<(), Error> {
        prof_span!("handle_server_msg");
        match msg {
            ServerGeneral::Disconnect(reason) => match reason {
                DisconnectReason::Shutdown => return Err(Error::ServerShutdown),
                DisconnectReason::Kicked(reason) => {
                    debug!("sending ClientMsg::Terminate because we got kicked");
                    frontend_events.push(Event::Kicked(reason));
                    self.send_msg_err(ClientGeneral::Terminate)?;
                },
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::Init(list)) => {
                self.player_list = list
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::Add(uid, player_info)) => {
                if let Some(old_player_info) = self.player_list.insert(uid, player_info.clone()) {
                    warn!(
                        "Received msg to insert {} with uid {} into the player list but there was \
                         already an entry for {} with the same uid that was overwritten!",
                        player_info.player_alias, uid, old_player_info.player_alias
                    );
                }
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::Moderator(uid, moderator)) => {
                if let Some(player_info) = self.player_list.get_mut(&uid) {
                    player_info.is_moderator = moderator;
                } else {
                    warn!(
                        "Received msg to update admin status of uid {}, but they were not in the \
                         list.",
                        uid
                    );
                }
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::SelectedCharacter(
                uid,
                char_info,
            )) => {
                if let Some(player_info) = self.player_list.get_mut(&uid) {
                    player_info.character = Some(char_info);
                } else {
                    warn!(
                        "Received msg to update character info for uid {}, but they were not in \
                         the list.",
                        uid
                    );
                }
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::LevelChange(uid, next_level)) => {
                if let Some(player_info) = self.player_list.get_mut(&uid) {
                    player_info.character = match &player_info.character {
                        Some(character) => Some(msg::CharacterInfo {
                            name: character.name.to_string(),
                        }),
                        None => {
                            warn!(
                                "Received msg to update character level info to {} for uid {}, \
                                 but this player's character is None.",
                                next_level, uid
                            );

                            None
                        },
                    };
                }
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::Remove(uid)) => {
                // Instead of removing players, mark them as offline because we need to
                // remember the names of disconnected players in chat.
                //
                // TODO: consider alternatives since this leads to an ever growing list as
                // players log out and in. Keep in mind we might only want to
                // keep only so many messages in chat the history. We could
                // potentially use an ID that's more persistent than the Uid.
                // One of the reasons we don't just store the string of the player name
                // into the message is to make alias changes reflected in older messages.

                if let Some(player_info) = self.player_list.get_mut(&uid) {
                    if player_info.is_online {
                        player_info.is_online = false;
                    } else {
                        warn!(
                            "Received msg to remove uid {} from the player list by they were \
                             already marked offline",
                            uid
                        );
                    }
                } else {
                    warn!(
                        "Received msg to remove uid {} from the player list by they weren't in \
                         the list!",
                        uid
                    );
                }
            },
            ServerGeneral::PlayerListUpdate(PlayerListUpdate::Alias(uid, new_name)) => {
                if let Some(player_info) = self.player_list.get_mut(&uid) {
                    player_info.player_alias = new_name;
                } else {
                    warn!(
                        "Received msg to alias player with uid {} to {} but this uid is not in \
                         the player list",
                        uid, new_name
                    );
                }
            },
            ServerGeneral::ChatMsg(m) => frontend_events.push(Event::Chat(m)),
            ServerGeneral::ChatMode(m) => {
                self.chat_mode = m;
            },
            ServerGeneral::SetPlayerEntity(uid) => {
                if let Some(entity) = self.state.ecs().entity_from_uid(uid.0) {
                    let old_player_entity = core::mem::replace(
                        &mut *self.state.ecs_mut().write_resource(),
                        PlayerEntity(Some(entity)),
                    );
                    if let Some(old_entity) = old_player_entity.0 {
                        // Transfer controller to the new entity.
                        let mut controllers = self.state.ecs().write_storage::<Controller>();
                        if let Some(controller) = controllers.remove(old_entity) {
                            if let Err(e) = controllers.insert(entity, controller) {
                                error!(
                                    ?e,
                                    "Failed to insert controller when setting new player entity!"
                                );
                            }
                        }
                    }
                    if let Some(presence) = self.presence {
                        self.presence = Some(match presence {
                            PresenceKind::Spectator => PresenceKind::Spectator,
                            PresenceKind::Character(_) => PresenceKind::Possessor,
                            PresenceKind::Possessor => PresenceKind::Possessor,
                        });
                    }
                } else {
                    return Err(Error::Other("Failed to find entity from uid.".into()));
                }
            },
            ServerGeneral::TimeOfDay(time_of_day, calendar) => {
                self.target_time_of_day = Some(time_of_day);
                *self.state.ecs_mut().write_resource() = calendar;
            },
            ServerGeneral::EntitySync(entity_sync_package) => {
                self.state
                    .ecs_mut()
                    .apply_entity_sync_package(entity_sync_package);
            },
            ServerGeneral::CompSync(comp_sync_package) => {
                self.state
                    .ecs_mut()
                    .apply_comp_sync_package(comp_sync_package);
            },
            ServerGeneral::CreateEntity(entity_package) => {
                self.state.ecs_mut().apply_entity_package(entity_package);
            },
            ServerGeneral::DeleteEntity(entity) => {
                if self.uid() != Some(entity) {
                    self.state
                        .ecs_mut()
                        .delete_entity_and_clear_from_uid_allocator(entity.0);
                }
            },
            ServerGeneral::Notification(n) => {
                frontend_events.push(Event::Notification(n));
            },
            _ => unreachable!("Not a general msg"),
        }
        Ok(())
    }

    fn handle_server_in_game_msg(
        &mut self,
        frontend_events: &mut Vec<Event>,
        msg: ServerGeneral,
    ) -> Result<(), Error> {
        prof_span!("handle_server_in_game_msg");
        match msg {
            ServerGeneral::TimeSync(time) => {
                // Even with a stable network, expect time to oscillate around the actual time
                // by SERVER_TICK (33.3ms)
                let old_time = self.state.ecs().read_resource::<Time>().0;
                let diff = old_time - time.0;
                self.state.ecs().write_resource::<Time>().0 = time.0;
                if diff > 0.0 {
                    tracing::warn!(?old_time, ?diff, "Time was reverted by server");
                    let rewind_time = self.inter_tick_rewind_time.unwrap_or_default()
                        + Duration::from_secs_f64(diff);
                    self.inter_tick_rewind_time = Some(rewind_time);
                } else {
                    tracing::warn!(?old_time, ?diff, "Time was advanced by server");
                }
            },

            ServerGeneral::AckControl {
                acked_ids,
                highest_ahead_command,
                predict_available,
            } => {
                if let Some(remote_controller) = self
                    .state
                    .ecs()
                    .write_storage::<RemoteController>()
                    .get_mut(self.entity())
                {
                    common_base::plot!("server_predict_available", predict_available as f64);
                    common_base::plot!("highest_ahead_command2", highest_ahead_command);

                    // for now ignore the time send by the server as its based on TIME and just use
                    // MonotonicTime
                    let monotonic_time = Duration::from_secs_f64(
                        self.state.ecs().read_resource::<MonotonicTime>().0,
                    );
                    //let time = Duration::from_secs_f64(time.0);
                    remote_controller.acked(acked_ids, monotonic_time, highest_ahead_command);
                    remote_controller.maintain(None);
                }
            },
            ServerGeneral::GroupUpdate(change_notification) => {
                use comp::group::ChangeNotification::*;
                // Note: we use a hashmap since this would not work with entities outside
                // the view distance
                match change_notification {
                    Added(uid, role) => {
                        // Check if this is a newly formed group by looking for absence of
                        // other non pet group members
                        if !matches!(role, group::Role::Pet)
                            && !self
                                .group_members
                                .values()
                                .any(|r| !matches!(r, group::Role::Pet))
                        {
                            frontend_events
                                .push(Event::Chat(comp::ChatType::Meta.chat_msg(
                                    "Type /g or /group to chat with your group members",
                                )));
                        }
                        if let Some(player_info) = self.player_list.get(&uid) {
                            frontend_events.push(Event::Chat(
                                comp::ChatType::GroupMeta("Group".into()).chat_msg(format!(
                                    "[{}] joined group",
                                    self.personalize_alias(uid, player_info.player_alias.clone())
                                )),
                            ));
                        }
                        if self.group_members.insert(uid, role) == Some(role) {
                            warn!(
                                "Received msg to add uid {} to the group members but they were \
                                 already there",
                                uid
                            );
                        }
                    },
                    Removed(uid) => {
                        if let Some(player_info) = self.player_list.get(&uid) {
                            frontend_events.push(Event::Chat(
                                comp::ChatType::GroupMeta("Group".into()).chat_msg(format!(
                                    "[{}] left group",
                                    self.personalize_alias(uid, player_info.player_alias.clone())
                                )),
                            ));
                            frontend_events.push(Event::MapMarker(
                                comp::MapMarkerUpdate::GroupMember(uid, MapMarkerChange::Remove),
                            ));
                        }
                        if self.group_members.remove(&uid).is_none() {
                            warn!(
                                "Received msg to remove uid {} from group members but by they \
                                 weren't in there!",
                                uid
                            );
                        }
                    },
                    NewLeader(leader) => {
                        self.group_leader = Some(leader);
                    },
                    NewGroup { leader, members } => {
                        self.group_leader = Some(leader);
                        self.group_members = members.into_iter().collect();
                        // Currently add/remove messages treat client as an implicit member
                        // of the group whereas this message explicitly includes them so to
                        // be consistent for now we will remove the client from the
                        // received hashset
                        if let Some(uid) = self.uid() {
                            self.group_members.remove(&uid);
                        }
                        frontend_events.push(Event::MapMarker(comp::MapMarkerUpdate::ClearGroup));
                    },
                    NoGroup => {
                        self.group_leader = None;
                        self.group_members = HashMap::new();
                        frontend_events.push(Event::MapMarker(comp::MapMarkerUpdate::ClearGroup));
                    },
                }
            },
            ServerGeneral::Invite {
                inviter,
                timeout,
                kind,
            } => {
                self.invite = Some((inviter, std::time::Instant::now(), timeout, kind));
            },
            ServerGeneral::InvitePending(uid) => {
                if !self.pending_invites.insert(uid) {
                    warn!("Received message about pending invite that was already pending");
                }
            },
            ServerGeneral::InviteComplete {
                target,
                answer,
                kind,
            } => {
                if !self.pending_invites.remove(&target) {
                    warn!(
                        "Received completed invite message for invite that was not in the list of \
                         pending invites"
                    )
                }
                frontend_events.push(Event::InviteComplete {
                    target,
                    answer,
                    kind,
                });
            },
            // Cleanup for when the client goes back to the `presence = None`
            ServerGeneral::ExitInGameSuccess => {
                self.presence = None;
                self.clean_state();
            },
            ServerGeneral::InventoryUpdate(inventory, event) => {
                match event {
                    InventoryUpdateEvent::BlockCollectFailed { .. } => {},
                    InventoryUpdateEvent::EntityCollectFailed { .. } => {},
                    _ => {
                        // Push the updated inventory component to the client
                        // FIXME: Figure out whether this error can happen under normal gameplay,
                        // if not find a better way to handle it, if so maybe consider kicking the
                        // client back to login?
                        let entity = self.entity();
                        if let Err(e) = self
                            .state
                            .ecs_mut()
                            .write_storage()
                            .insert(entity, inventory)
                        {
                            warn!(
                                ?e,
                                "Received an inventory update event for client entity, but this \
                                 entity was not found... this may be a bug."
                            );
                        }
                    },
                }

                self.update_available_recipes();

                frontend_events.push(Event::InventoryUpdated(event));
            },
            ServerGeneral::SetViewDistance(vd) => {
                self.view_distance = Some(vd);
                frontend_events.push(Event::SetViewDistance(vd));
            },
            ServerGeneral::Outcomes(outcomes) => {
                frontend_events.extend(outcomes.into_iter().map(Event::Outcome))
            },
            ServerGeneral::Knockback(impulse) => {
                self.state
                    .ecs()
                    .read_resource::<EventBus<LocalEvent>>()
                    .emit_now(LocalEvent::ApplyImpulse {
                        entity: self.entity(),
                        impulse,
                    });
            },
            ServerGeneral::UpdatePendingTrade(id, trade, pricing) => {
                tracing::trace!("UpdatePendingTrade {:?} {:?}", id, trade);
                self.pending_trade = Some((id, trade, pricing));
            },
            ServerGeneral::FinishedTrade(result) => {
                if let Some((_, trade, _)) = self.pending_trade.take() {
                    self.update_available_recipes();
                    frontend_events.push(Event::TradeComplete { result, trade })
                }
            },
            ServerGeneral::SiteEconomy(economy) => {
                if let Some(rich) = self.sites_mut().get_mut(&economy.id) {
                    rich.economy = Some(economy);
                }
            },
            ServerGeneral::MapMarker(event) => {
                frontend_events.push(Event::MapMarker(event));
            },
            _ => unreachable!("Not a in_game message"),
        }
        Ok(())
    }

    fn handle_server_terrain_msg(&mut self, msg: ServerGeneral) -> Result<(), Error> {
        prof_span!("handle_server_terrain_mgs");
        match msg {
            ServerGeneral::TerrainChunkUpdate { key, chunk } => {
                if let Some(chunk) = chunk.ok().and_then(|c| c.to_chunk()) {
                    self.state.insert_chunk(key, Arc::new(chunk));
                }
                self.pending_chunks.remove(&key);
            },
            ServerGeneral::LodZoneUpdate { key, zone } => {
                self.lod_zones.insert(key, zone);
                self.lod_last_requested = None;
            },
            ServerGeneral::TerrainBlockUpdates(blocks) => {
                if let Some(mut blocks) = blocks.decompress() {
                    blocks.drain().for_each(|(pos, block)| {
                        self.state.set_block(pos, block);
                    });
                }
            },
            _ => unreachable!("Not a terrain message"),
        }
        Ok(())
    }

    fn handle_server_character_screen_msg(
        &mut self,
        events: &mut Vec<Event>,
        msg: ServerGeneral,
    ) -> Result<(), Error> {
        prof_span!("handle_server_character_screen_msg");
        match msg {
            ServerGeneral::CharacterListUpdate(character_list) => {
                self.character_list.characters = character_list;
                self.character_list.loading = false;
            },
            ServerGeneral::CharacterActionError(error) => {
                warn!("CharacterActionError: {:?}.", error);
                events.push(Event::CharacterError(error));
            },
            ServerGeneral::CharacterDataLoadError(error) => {
                trace!("Handling join error by server");
                self.presence = None;
                self.clean_state();
                events.push(Event::CharacterError(error));
            },
            ServerGeneral::CharacterCreated(character_id) => {
                events.push(Event::CharacterCreated(character_id));
            },
            ServerGeneral::CharacterEdited(character_id) => {
                events.push(Event::CharacterEdited(character_id));
            },
            ServerGeneral::CharacterSuccess => {
                debug!("client is now in ingame state on server");
                if let Some(vd) = self.view_distance {
                    self.set_view_distance(vd);
                }
            },
            _ => unreachable!("Not a character_screen msg"),
        }
        Ok(())
    }

    fn handle_ping_msg(&mut self, msg: PingMsg) -> Result<(), Error> {
        prof_span!("handle_ping_msg");
        match msg {
            PingMsg::Ping => {
                self.send_msg_err(PingMsg::Pong)?;
            },
            PingMsg::Pong => {
                self.last_server_pong = self.state.get_time();
                self.last_ping_delta = self.state.get_time() - self.last_server_ping;

                // Maintain the correct number of deltas for calculating the rolling average
                // ping. The client sends a ping to the server every second so we should be
                // receiving a pong reply roughly every second.
                while self.ping_deltas.len() > PING_ROLLING_AVERAGE_SECS - 1 {
                    self.ping_deltas.pop_front();
                }
                self.ping_deltas.push_back(self.last_ping_delta);
            },
        }
        Ok(())
    }

    fn handle_messages(&mut self, frontend_events: &mut Vec<Event>) -> Result<u64, Error> {
        let mut cnt = 0;
        #[cfg(feature = "tracy")]
        let (mut terrain_cnt, mut ingame_cnt) = (0, 0);
        loop {
            let cnt_start = cnt;

            while let Some(msg) = self.general_stream.try_recv()? {
                cnt += 1;
                self.handle_server_msg(frontend_events, msg)?;
            }
            while let Some(msg) = self.ping_stream.try_recv()? {
                cnt += 1;
                self.handle_ping_msg(msg)?;
            }
            while let Some(msg) = self.character_screen_stream.try_recv()? {
                cnt += 1;
                self.handle_server_character_screen_msg(frontend_events, msg)?;
            }
            while let Some(msg) = self.in_game_stream.try_recv()? {
                cnt += 1;
                #[cfg(feature = "tracy")]
                {
                    ingame_cnt += 1;
                }
                self.handle_server_in_game_msg(frontend_events, msg)?;
            }
            while let Some(msg) = self.terrain_stream.try_recv()? {
                cnt += 1;
                #[cfg(feature = "tracy")]
                {
                    if let ServerGeneral::TerrainChunkUpdate { chunk, .. } = &msg {
                        terrain_cnt += chunk.as_ref().map(|x| x.approx_len()).unwrap_or(0);
                    }
                }
                self.handle_server_terrain_msg(msg)?;
            }

            if cnt_start == cnt {
                #[cfg(feature = "tracy")]
                {
                    plot!("terrain_recvs", terrain_cnt as f64);
                    plot!("ingame_recvs", ingame_cnt as f64);
                }
                return Ok(cnt);
            }
        }
    }

    /// Handle new server messages.
    fn handle_new_messages(&mut self) -> Result<Vec<Event>, Error> {
        prof_span!("handle_new_messages");
        let mut frontend_events = Vec::new();

        // Check that we have an valid connection.
        // Use the last ping time as a 1s rate limiter, we only notify the user once per
        // second
        if self.state.get_time() - self.last_server_ping > 1. {
            let duration_since_last_pong = self.state.get_time() - self.last_server_pong;

            // Dispatch a notification to the HUD warning they will be kicked in {n} seconds
            const KICK_WARNING_AFTER_REL_TO_TIMEOUT_FRACTION: f64 = 0.75;
            if duration_since_last_pong
                >= (self.client_timeout.as_secs() as f64
                    * KICK_WARNING_AFTER_REL_TO_TIMEOUT_FRACTION)
                && self.state.get_time() - duration_since_last_pong > 0.
            {
                frontend_events.push(Event::DisconnectionNotification(
                    (self.state.get_time() - duration_since_last_pong).round() as u64,
                ));
            }
        }

        let msg_count = self.handle_messages(&mut frontend_events)?;

        if msg_count == 0
            && self.state.get_time() - self.last_server_pong > self.client_timeout.as_secs() as f64
        {
            return Err(Error::ServerTimeout);
        }

        Ok(frontend_events)
    }

    pub fn entity(&self) -> EcsEntity {
        self.state
            .ecs()
            .read_resource::<PlayerEntity>()
            .0
            .expect("Client::entity should always have PlayerEntity be Some")
    }

    pub fn uid(&self) -> Option<Uid> { self.state.read_component_copied(self.entity()) }

    pub fn presence(&self) -> Option<PresenceKind> { self.presence }

    pub fn registered(&self) -> bool { self.registered }

    pub fn get_tick(&self) -> u64 { self.tick }

    pub fn get_ping_ms(&self) -> f64 { self.last_ping_delta * 1000.0 }

    pub fn get_ping_ms_rolling_avg(&self) -> f64 {
        let mut total_weight = 0.;
        let pings = self.ping_deltas.len() as f64;
        (self
            .ping_deltas
            .iter()
            .enumerate()
            .fold(0., |acc, (i, ping)| {
                let weight = i as f64 + 1. / pings;
                total_weight += weight;
                acc + (weight * ping)
            })
            / total_weight)
            * 1000.0
    }

    /// Get a reference to the client's runtime thread pool. This pool should be
    /// used for any computationally expensive operations that run outside
    /// of the main thread (i.e., threads that block on I/O operations are
    /// exempt).
    pub fn runtime(&self) -> &Arc<Runtime> { &self.runtime }

    /// Get a reference to the client's game state.
    pub fn state(&self) -> &State { &self.state }

    /// Get a mutable reference to the client's game state.
    pub fn state_mut(&mut self) -> &mut State { &mut self.state }

    /// Returns an iterator over the aliases of all the online players on the
    /// server
    pub fn players(&self) -> impl Iterator<Item = &str> {
        self.player_list()
            .values()
            .filter_map(|player_info| player_info.is_online.then(|| &*player_info.player_alias))
    }

    /// Return true if this client is a moderator on the server
    pub fn is_moderator(&self) -> bool {
        let client_uid = self
            .state
            .read_component_copied::<Uid>(self.entity())
            .expect("Client doesn't have a Uid!!!");

        self.player_list
            .get(&client_uid)
            .map_or(false, |info| info.is_moderator)
    }

    /// Clean client ECS state
    fn clean_state(&mut self) {
        let client_uid = self
            .uid()
            .map(|u| u.into())
            .expect("Client doesn't have a Uid!!!");

        // Clear ecs of all entities
        self.state.ecs_mut().delete_all();
        self.state.ecs_mut().maintain();
        self.state.ecs_mut().insert(UidAllocator::default());

        // Recreate client entity with Uid
        let entity_builder = self.state.ecs_mut().create_entity();
        let uid = entity_builder
            .world
            .write_resource::<UidAllocator>()
            .allocate(entity_builder.entity, Some(client_uid));

        let entity = entity_builder.with(uid).build();
        self.state.ecs().write_resource::<PlayerEntity>().0 = Some(entity);
    }

    /// Change player alias to "You" if client belongs to matching player
    fn personalize_alias(&self, uid: Uid, alias: String) -> String {
        let client_uid = self.uid().expect("Client doesn't have a Uid!!!");
        if client_uid == uid {
            "You".to_string() // TODO: Localize
        } else {
            alias
        }
    }

    /// Format a message for the client (voxygen chat box or chat-cli)
    pub fn format_message(&self, msg: &comp::ChatMsg, character_name: bool) -> String {
        let comp::ChatMsg {
            chat_type, message, ..
        } = &msg;
        let name_of_uid = |uid| {
            let ecs = self.state.ecs();
            (
                &ecs.read_storage::<comp::Stats>(),
                &ecs.read_storage::<Uid>(),
            )
                .join()
                .find(|(_, u)| u == &uid)
                .map(|(c, _)| c.name.clone())
        };
        let alias_of_uid = |uid| {
            self.player_list.get(uid).map_or(
                name_of_uid(uid).unwrap_or_else(|| "<?>".to_string()),
                |player_info| {
                    if player_info.is_moderator {
                        format!(
                            "MOD - {}",
                            self.personalize_alias(*uid, player_info.player_alias.clone())
                        )
                    } else {
                        self.personalize_alias(*uid, player_info.player_alias.clone())
                    }
                },
            )
        };
        let message_format = |uid, message, group| {
            let alias = alias_of_uid(uid);
            let name = if character_name {
                name_of_uid(uid)
            } else {
                None
            };
            match (group, name) {
                (Some(group), None) => format!("({}) [{}]: {}", group, alias, message),
                (None, None) => format!("[{}]: {}", alias, message),
                (Some(group), Some(name)) => {
                    format!("({}) [{}] {}: {}", group, alias, name, message)
                },
                (None, Some(name)) => format!("[{}] {}: {}", alias, name, message),
            }
        };
        match chat_type {
            // For ChatType::{Online, Offline, Kill} these message strings are localized
            // in voxygen/src/hud/chat.rs before being formatted here.
            // Kill messages are generated in server/src/events/entity_manipulation.rs
            // fn handle_destroy
            comp::ChatType::Online(uid) => {
                // Default message formats if no localized message string is set by hud
                // Needed for cli clients that don't set localization info
                if message.is_empty() {
                    format!("[{}] came online", alias_of_uid(uid))
                } else {
                    message.replace("{name}", &alias_of_uid(uid))
                }
            },
            comp::ChatType::Offline(uid) => {
                // Default message formats if no localized message string is set by hud
                // Needed for cli clients that don't set localization info
                if message.is_empty() {
                    format!("[{}] went offline", alias_of_uid(uid))
                } else {
                    message.replace("{name}", &alias_of_uid(uid))
                }
            },
            comp::ChatType::CommandError => message.to_string(),
            comp::ChatType::CommandInfo => message.to_string(),
            comp::ChatType::FactionMeta(_) => message.to_string(),
            comp::ChatType::GroupMeta(_) => message.to_string(),
            comp::ChatType::Kill(kill_source, victim) => {
                // Default message formats if no localized message string is set by hud
                // Needed for cli clients that don't set localization info
                if message.is_empty() {
                    match kill_source {
                        KillSource::Player(attacker_uid, KillType::Buff(buff_kind)) => format!(
                            "[{}] died of {} caused by [{}]",
                            alias_of_uid(victim),
                            format!("{:?}", buff_kind).to_lowercase().as_str(),
                            alias_of_uid(attacker_uid)
                        ),
                        KillSource::Player(attacker_uid, KillType::Melee) => format!(
                            "[{}] killed [{}]",
                            alias_of_uid(attacker_uid),
                            alias_of_uid(victim)
                        ),
                        KillSource::Player(attacker_uid, KillType::Projectile) => format!(
                            "[{}] shot [{}]",
                            alias_of_uid(attacker_uid),
                            alias_of_uid(victim)
                        ),
                        KillSource::Player(attacker_uid, KillType::Explosion) => format!(
                            "[{}] blew up [{}]",
                            alias_of_uid(attacker_uid),
                            alias_of_uid(victim)
                        ),
                        KillSource::Player(attacker_uid, KillType::Energy) => format!(
                            "[{}] used magic to kill [{}]",
                            alias_of_uid(attacker_uid),
                            alias_of_uid(victim)
                        ),
                        KillSource::Player(attacker_uid, KillType::Other) => format!(
                            "[{}] killed [{}]",
                            alias_of_uid(attacker_uid),
                            alias_of_uid(victim)
                        ),
                        KillSource::NonExistent(KillType::Buff(buff_kind)) => format!(
                            "[{}] died of {}",
                            alias_of_uid(victim),
                            format!("{:?}", buff_kind).to_lowercase().as_str()
                        ),
                        KillSource::NonPlayer(attacker_name, KillType::Buff(buff_kind)) => format!(
                            "[{}] died of {} caused by {}",
                            alias_of_uid(victim),
                            format!("{:?}", buff_kind).to_lowercase().as_str(),
                            attacker_name
                        ),
                        KillSource::NonPlayer(attacker_name, KillType::Melee) => {
                            format!("{} killed [{}]", attacker_name, alias_of_uid(victim))
                        },
                        KillSource::NonPlayer(attacker_name, KillType::Projectile) => {
                            format!("{} shot [{}]", attacker_name, alias_of_uid(victim))
                        },
                        KillSource::NonPlayer(attacker_name, KillType::Explosion) => {
                            format!("{} blew up [{}]", attacker_name, alias_of_uid(victim))
                        },
                        KillSource::NonPlayer(attacker_name, KillType::Energy) => format!(
                            "{} used magic to kill [{}]",
                            attacker_name,
                            alias_of_uid(victim)
                        ),
                        KillSource::NonPlayer(attacker_name, KillType::Other) => {
                            format!("{} killed [{}]", attacker_name, alias_of_uid(victim))
                        },
                        KillSource::Environment(environment) => {
                            format!("[{}] died in {}", alias_of_uid(victim), environment)
                        },
                        KillSource::FallDamage => {
                            format!("[{}] died from fall damage", alias_of_uid(victim))
                        },
                        KillSource::Suicide => {
                            format!("[{}] died from self-inflicted wounds", alias_of_uid(victim))
                        },
                        KillSource::NonExistent(_) => format!("[{}] died", alias_of_uid(victim)),
                        KillSource::Other => format!("[{}] died", alias_of_uid(victim)),
                    }
                } else {
                    match kill_source {
                        KillSource::Player(attacker_uid, _) => message
                            .replace("{attacker}", &alias_of_uid(attacker_uid))
                            .replace("{victim}", &alias_of_uid(victim)),
                        KillSource::NonExistent(KillType::Buff(_)) => {
                            message.replace("{victim}", &alias_of_uid(victim))
                        },
                        KillSource::NonPlayer(attacker_name, _) => message
                            .replace("{attacker}", attacker_name)
                            .replace("{victim}", &alias_of_uid(victim)),
                        KillSource::Environment(environment) => message
                            .replace("{name}", &alias_of_uid(victim))
                            .replace("{environment}", environment),
                        KillSource::FallDamage => message.replace("{name}", &alias_of_uid(victim)),
                        KillSource::Suicide => message.replace("{name}", &alias_of_uid(victim)),
                        KillSource::NonExistent(_) => {
                            message.replace("{name}", &alias_of_uid(victim))
                        },
                        KillSource::Other => message.replace("{name}", &alias_of_uid(victim)),
                    }
                }
            },
            comp::ChatType::Tell(from, to) => {
                let from_alias = alias_of_uid(from);
                let to_alias = alias_of_uid(to);
                if Some(*from) == self.uid() {
                    format!("To [{}]: {}", to_alias, message)
                } else {
                    format!("From [{}]: {}", from_alias, message)
                }
            },
            comp::ChatType::Say(uid) => message_format(uid, message, None),
            comp::ChatType::Group(uid, s) => message_format(uid, message, Some(s)),
            comp::ChatType::Faction(uid, s) => message_format(uid, message, Some(s)),
            comp::ChatType::Region(uid) => message_format(uid, message, None),
            comp::ChatType::World(uid) => message_format(uid, message, None),
            // NPCs can't talk. Should be filtered by hud/mod.rs for voxygen and should be filtered
            // by server (due to not having a Pos) for chat-cli
            comp::ChatType::Npc(_uid, _r) => "".to_string(),
            comp::ChatType::NpcSay(uid, _r) => message_format(uid, message, None),
            comp::ChatType::NpcTell(from, to, _r) => {
                let from_alias = alias_of_uid(from);
                let to_alias = alias_of_uid(to);
                if Some(*from) == self.uid() {
                    format!("To [{}]: {}", to_alias, message)
                } else {
                    format!("From [{}]: {}", from_alias, message)
                }
            },
            comp::ChatType::Meta => message.to_string(),
        }
    }

    /// Execute a single client tick:
    /// - handles messages from the server
    /// - sends physics update
    /// - requests chunks
    ///
    /// The game state is purposefully not simulated to reduce the overhead of
    /// running the client. This method is for use in testing a server with
    /// many clients connected.
    #[cfg(feature = "tick_network")]
    #[allow(clippy::needless_collect)] // False positive
    pub fn tick_network(&mut self, dt: Duration) -> Result<(), Error> {
        span!(_guard, "tick_network", "Client::tick_network");
        // Advance state time manually since we aren't calling `State::tick`
        self.state
            .ecs()
            .write_resource::<common::resources::Time>()
            .0 += dt.as_secs_f64();

        // Handle new messages from the server.
        self.handle_new_messages()?;

        // 5) Terrain
        self.tick_terrain()?;
        let empty = Arc::new(TerrainChunk::new(
            0,
            Block::empty(),
            Block::empty(),
            common::terrain::TerrainChunkMeta::void(),
        ));
        let mut terrain = self.state.terrain_mut();
        // Replace chunks with empty chunks to save memory
        let to_clear = terrain
            .iter()
            .filter_map(|(key, chunk)| (chunk.sub_chunks_len() != 0).then(|| key))
            .collect::<Vec<_>>();
        to_clear.into_iter().for_each(|key| {
            terrain.insert(key, Arc::clone(&empty));
        });
        drop(terrain);

        // Send a ping to the server once every second
        if self.state.get_time() - self.last_server_ping > 1. {
            self.send_msg_err(PingMsg::Ping)?;
            self.last_server_ping = self.state.get_time();
        }

        // 6) Update the server about the player's physics attributes.
        if self.presence.is_some() {
            if let (Some(pos), Some(vel), Some(ori)) = (
                self.state.read_storage().get(self.entity()).cloned(),
                self.state.read_storage().get(self.entity()).cloned(),
                self.state.read_storage().get(self.entity()).cloned(),
            ) {
                self.in_game_stream
                    .send(ClientGeneral::PlayerPhysics { pos, vel, ori })?;
            }
        }

        // 7) Finish the tick, pass control back to the frontend.
        self.tick += 1;

        Ok(())
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        trace!("Dropping client");
        if self.registered {
            if let Err(e) = self.send_msg_err(ClientGeneral::Terminate) {
                warn!(
                    ?e,
                    "Error during drop of client, couldn't send disconnect package, is the \
                     connection already closed?",
                );
            }
        } else {
            trace!("no disconnect msg necessary as client wasn't registered")
        }

        tokio::task::block_in_place(|| {
            if let Err(e) = self
                .runtime
                .block_on(self.participant.take().unwrap().disconnect())
            {
                warn!(?e, "error when disconnecting, couldn't send all data");
            }
        });
        //explicitly drop the network here while the runtime is still existing
        drop(self.network.take());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// THIS TEST VERIFIES THE CONSTANT API.
    /// CHANGING IT WILL BREAK 3rd PARTY APPLICATIONS (please extend) which
    /// needs to be informed (or fixed)
    ///  - torvus: https://gitlab.com/veloren/torvus
    /// CONTACT @Core Developer BEFORE MERGING CHANGES TO THIS TEST
    fn constant_api_test() {
        use common::clock::Clock;

        const SPT: f64 = 1.0 / 60.0;

        let runtime = Arc::new(Runtime::new().unwrap());
        let runtime2 = Arc::clone(&runtime);
        let veloren_client: Result<Client, Error> = runtime.block_on(Client::new(
            ConnectionArgs::Tcp {
                hostname: "127.0.0.1:9000".to_owned(),
                prefer_ipv6: false,
            },
            runtime2,
            &mut None,
        ));

        let _ = veloren_client.map(|mut client| {
            //register
            let username: String = "Foo".to_string();
            let password: String = "Bar".to_string();
            let auth_server: String = "auth.veloren.net".to_string();
            let _result: Result<(), Error> =
                runtime.block_on(client.register(username, password, |suggestion: &str| {
                    suggestion == auth_server
                }));

            //clock
            let mut clock = Clock::new(Duration::from_secs_f64(SPT));

            //tick
            let events_result: Result<Vec<Event>, Error> =
                client.tick(comp::ControllerInputs::default(), clock.dt(), |_| {});

            //chat functionality
            client.send_chat("foobar".to_string());

            let _ = events_result.map(|mut events| {
                // event handling
                if let Some(event) = events.pop() {
                    match event {
                        Event::Chat(msg) => {
                            let msg: comp::ChatMsg = msg;
                            let _s: String = client.format_message(&msg, true);
                        },
                        Event::Disconnect => {},
                        Event::DisconnectionNotification(_) => {
                            tracing::debug!("Will be disconnected soon! :/")
                        },
                        Event::Notification(notification) => {
                            let notification: Notification = notification;
                            tracing::debug!("Notification: {:?}", notification);
                        },
                        _ => {},
                    }
                };
            });

            client.cleanup();
            clock.tick();
        });
    }
}
