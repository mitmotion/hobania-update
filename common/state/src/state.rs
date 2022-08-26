#[cfg(feature = "plugins")]
use crate::plugin::memory_manager::EcsWorld;
#[cfg(feature = "plugins")]
use crate::plugin::PluginMgr;
#[cfg(feature = "plugins")]
use common::uid::UidAllocator;
use common::{
    calendar::Calendar,
    comp,
    event::{EventBus, LocalEvent, ServerEvent},
    link::Is,
    mounting::{Mount, Rider},
    outcome::Outcome,
    region::RegionMap,
    resources::{
        DeltaTime, EntitiesDiedLastTick, GameMode, PlayerEntity, PlayerPhysicsSettings, Time,
        TimeOfDay,
    },
    slowjob::{self, SlowJobPool},
    terrain::{Block, TerrainChunk, TerrainGrid},
    time::DayPeriod,
    trade::Trades,
    vol::{ReadVol, WriteVol},
    weather::{Weather, WeatherGrid},
};
use common_base::span;
use common_ecs::{PhysicsMetrics, SysMetrics};
use common_net::sync::{interpolation as sync_interp, WorldSyncExt};
use core::{convert::identity, time::Duration};
use hashbrown::{HashMap, HashSet};
use rayon::{ThreadPool, ThreadPoolBuilder};
use specs::{
    prelude::Resource,
    shred::{Fetch, FetchMut},
    storage::{MaskedStorage as EcsMaskedStorage, Storage as EcsStorage},
    Component, DispatcherBuilder, Entity as EcsEntity, WorldExt,
};
use std::sync::Arc;
use vek::*;

/// How much faster should an in-game day be compared to a real day?
// TODO: Don't hard-code this.
const DAY_CYCLE_FACTOR: f64 = 24.0 * 2.0;

/// At what point should we stop speeding up physics to compensate for lag? If
/// we speed physics up too fast, we'd skip important physics events like
/// collisions. This constant determines the upper limit. If delta time exceeds
/// this value, the game's physics will begin to produce time lag. Ideally, we'd
/// avoid such a situation.
const MAX_DELTA_TIME: f32 = 1.0;

#[derive(Default)]
pub struct BlockChange {
    blocks: HashMap<Vec3<i32>, Block>,
}

impl BlockChange {
    pub fn set(&mut self, pos: Vec3<i32>, block: Block) { self.blocks.insert(pos, block); }

    pub fn try_set(&mut self, pos: Vec3<i32>, block: Block) -> Option<()> {
        if !self.blocks.contains_key(&pos) {
            self.blocks.insert(pos, block);
            Some(())
        } else {
            None
        }
    }

    pub fn clear(&mut self) { self.blocks.clear(); }
}

#[derive(Default)]
pub struct TerrainChanges {
    pub new_chunks: HashSet<Vec2<i32>>,
    pub modified_chunks: HashSet<Vec2<i32>>,
    pub removed_chunks: HashSet<Vec2<i32>>,
    pub modified_blocks: HashMap<Vec3<i32>, Block>,
}

impl TerrainChanges {
    pub fn clear(&mut self) {
        self.new_chunks.clear();
        self.modified_chunks.clear();
        self.removed_chunks.clear();
    }
}

/// A type used to represent game state stored on both the client and the
/// server. This includes things like entity components, terrain data, and
/// global states like weather, time of day, etc.
pub struct State {
    ecs: specs::World,
    // Avoid lifetime annotation by storing a thread pool instead of the whole dispatcher
    thread_pool: Arc<ThreadPool>,
}

pub type Pools = (usize, GameMode/*u64*/, Arc<ThreadPool>/*, slowjob::SlowJobPool*/);


impl State {
    pub fn pools(game_mode: GameMode) -> Pools {
        let num_cpu = num_cpus::get()/* - 1*/;

        let thread_name_infix = match game_mode {
            GameMode::Server => "s",
            GameMode::Client => "c",
            GameMode::Singleplayer => "sp",
        };
        let rayon_threads = match game_mode {
            GameMode::Server | GameMode::Client => num_cpu / 2,
            GameMode::Singleplayer => num_cpu / 4,
        }/*num_cpu*/.max(common::consts::MIN_RECOMMENDED_RAYON_THREADS);
        let rayon_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(rayon_threads)
                // .thread_name(move |i| format!("rayon-{}", i))
                .thread_name(move |i| format!("rayon-{}-{}", thread_name_infix, i))
                .build()
                .unwrap(),
        );

        // let num_cpu = num_cpu as u64;
        /* let slow_limit = /* match game_mode {
            GameMode::Server | GameMode::Client => num_cpu / 2,
            GameMode::Singleplayer => num_cpu / 4
        }.max(1); */(2 * num_cpu).max(1); */
        /* let slow_limit = 2 * (num_cpu - 1).max(1);
        let cores = core_affinity::get_core_ids().unwrap_or(vec![]).into_iter().take(slow_limit).collect::<Vec<_>>();
        let floating = slow_limit.saturating_sub(cores.len());
        tracing::trace!(?slow_limit, "Slow Thread limit");
        let slow_pool = slowjob::ThreadPool::with_affinity(&cores, floating, slowjob::large());
        // let slow_pool = slowjob::large_pool(slow_limit.min(64)/* as usize */);
        let slowjob = SlowJobPool::new(
            slow_limit as u64,
            10_000,
            /*Arc::clone(*/slow_pool/*)*/,
        ); */

        (num_cpu - 1/*slow_limit*//* as u64*/, game_mode, rayon_pool/*, slowjob*/)
    }

    /// Create a new `State` in client mode.
    pub fn client(pools: Pools) -> Self { Self::new(GameMode::Client, pools) }

    /// Create a new `State` in server mode.
    pub fn server(pools: Pools) -> Self { Self::new(GameMode::Server, pools) }

    pub fn new(ecs_role: GameMode, pools: Pools) -> Self {
        /* let thread_name_infix = match game_mode {
            GameMode::Server => "s",
            GameMode::Client => "c",
            GameMode::Singleplayer => "sp",
        }; */

        let num_cpu = /*num_cpus::get()*/pools.0/* / 2 + pools.0 / 4*/;
        let game_mode = pools.1;
        /* let rayon_threads = match game_mode {
            GameMode::Server | GameMode::Client => num_cpu/* / 2*/,
            GameMode::Singleplayer => num_cpu/* / 4*// 2,
        }/*num_cpu*/;
        let rayon_threads = rayon_threads.max(common::consts::MIN_RECOMMENDED_RAYON_THREADS);

        let thread_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(rayon_threads)
                .thread_name(move |i| format!("rayon-{}-{}", thread_name_infix, i))
                .build()
                .unwrap(),
        ); */

        // let num_cpu = num_cpu as u64;
        let (start, step, total) = match (game_mode, ecs_role) {
            (_, GameMode::Singleplayer) => todo!("Singleplayer is not a valid ECS role (yet)"),
            (GameMode::Server | GameMode::Client, _) => (0, 1, num_cpu),
            (GameMode::Singleplayer, GameMode::Server) => (0, 2, num_cpu / 2/* + num_cpu / 4 */),
            (GameMode::Singleplayer, GameMode::Client) => (1, 2, num_cpu - num_cpu / 2/* + num_cpu / 4 */),
        };
        let total = total.max(common::consts::MIN_RECOMMENDED_RAYON_THREADS);
        let cores = core_affinity::get_core_ids().unwrap_or(vec![]).into_iter().skip(start).step_by(step).take(total).collect::<Vec<_>>();
        let floating = total.saturating_sub(cores.len());
        // TODO: NUMA utils
        let slow_pool = slowjob::ThreadPool::with_affinity(&cores, floating, slowjob::large());
        /* let slow_pool = if cores.is_empty() {
            // We need *some* workers, so just start on all cores.
            slowjob::large_pool(1)/*slow_limit.min(64)/* as usize */)*/
        } else {
            let slow_pool = with_affinity(
                cores: &[CoreId],
                floating: usize,
                parker: P
            ) -> ThreadPool<P>
        } */
        /* let slow_limit = match game_mode {
            GameMode::Server | GameMode::Client => num_cpu / 2,
            GameMode::Singleplayer => num_cpu / 4
        }.max(1);/*(/*2 * */num_cpu / 2 + num_cpu / 4).max(1);*/ */
        tracing::trace!(?game_mode, ?ecs_role, ?num_cpu, ?start, ?step, ?total, "Slow Thread limit");
        // dbg!(game_mode, ecs_role, num_cpu, start, step, total, cores, floating, "Slow Thread limit");
        let slowjob = SlowJobPool::new(
            /* slow_limit as u64, */
            total as u64,
            10_000,
            /*Arc::clone(*/slow_pool/*)*/,
        );

        Self {
            ecs: Self::setup_ecs_world(ecs_role, /*num_cpu as u64*//*, &thread_pool, *//*pools.1*/slowjob/*pools.3*/),
            thread_pool: pools.2,
        }
    }

    /// Creates ecs world and registers all the common components and resources
    // TODO: Split up registering into server and client (e.g. move
    // EventBus<ServerEvent> to the server)
    fn setup_ecs_world(ecs_role: GameMode, /*num_cpu: u64*//*, thread_pool: &Arc<ThreadPool>, */slowjob: SlowJobPool) -> specs::World {
        let mut ecs = specs::World::new();
        // Uids for sync
        ecs.register_sync_marker();
        // Register server -> all clients synced components.
        ecs.register::<comp::Body>();
        ecs.register::<comp::Player>();
        ecs.register::<comp::Stats>();
        ecs.register::<comp::SkillSet>();
        ecs.register::<comp::ActiveAbilities>();
        ecs.register::<comp::Buffs>();
        ecs.register::<comp::Auras>();
        ecs.register::<comp::Energy>();
        ecs.register::<comp::Combo>();
        ecs.register::<comp::Health>();
        ecs.register::<comp::Poise>();
        ecs.register::<comp::CanBuild>();
        ecs.register::<comp::LightEmitter>();
        ecs.register::<comp::Item>();
        ecs.register::<comp::Scale>();
        ecs.register::<Is<Mount>>();
        ecs.register::<Is<Rider>>();
        ecs.register::<comp::Mass>();
        ecs.register::<comp::Density>();
        ecs.register::<comp::Collider>();
        ecs.register::<comp::Sticky>();
        ecs.register::<comp::Immovable>();
        ecs.register::<comp::CharacterState>();
        ecs.register::<comp::Object>();
        ecs.register::<comp::Group>();
        ecs.register::<comp::Shockwave>();
        ecs.register::<comp::ShockwaveHitEntities>();
        ecs.register::<comp::BeamSegment>();
        ecs.register::<comp::Alignment>();
        ecs.register::<comp::LootOwner>();

        // Register components send from clients -> server
        ecs.register::<comp::Controller>();

        // Register components send directly from server -> all but one client
        ecs.register::<comp::PhysicsState>();

        // Register components synced from client -> server -> all other clients
        ecs.register::<comp::Pos>();
        ecs.register::<comp::Vel>();
        ecs.register::<comp::Ori>();
        ecs.register::<comp::Inventory>();

        // Register common unsynced components
        ecs.register::<comp::PreviousPhysCache>();
        ecs.register::<comp::PosVelOriDefer>();

        // Register client-local components
        // TODO: only register on the client
        ecs.register::<comp::LightAnimation>();
        ecs.register::<sync_interp::InterpBuffer<comp::Pos>>();
        ecs.register::<sync_interp::InterpBuffer<comp::Vel>>();
        ecs.register::<sync_interp::InterpBuffer<comp::Ori>>();

        // Register server-local components
        // TODO: only register on the server
        ecs.register::<comp::Last<comp::Pos>>();
        ecs.register::<comp::Last<comp::Vel>>();
        ecs.register::<comp::Last<comp::Ori>>();
        ecs.register::<comp::Agent>();
        ecs.register::<comp::WaypointArea>();
        ecs.register::<comp::ForceUpdate>();
        ecs.register::<comp::InventoryUpdate>();
        ecs.register::<comp::Admin>();
        ecs.register::<comp::Waypoint>();
        ecs.register::<comp::MapMarker>();
        ecs.register::<comp::Projectile>();
        ecs.register::<comp::Melee>();
        ecs.register::<comp::ItemDrop>();
        ecs.register::<comp::ChatMode>();
        ecs.register::<comp::Faction>();
        ecs.register::<comp::invite::Invite>();
        ecs.register::<comp::invite::PendingInvites>();
        ecs.register::<comp::Beam>();

        // Register synced resources used by the ECS.
        ecs.insert(TimeOfDay(0.0));
        ecs.insert(Calendar::default());
        ecs.insert(WeatherGrid::new(Vec2::zero()));

        // Register unsynced resources used by the ECS.
        ecs.insert(Time(0.0));
        ecs.insert(DeltaTime(0.0));
        ecs.insert(PlayerEntity(None));
        ecs.insert(TerrainGrid::new().unwrap());
        ecs.insert(BlockChange::default());
        ecs.insert(crate::build_areas::BuildAreas::default());
        ecs.insert(TerrainChanges::default());
        ecs.insert(EventBus::<LocalEvent>::default());
        ecs.insert(ecs_role);
        ecs.insert(EventBus::<Outcome>::default());
        ecs.insert(common::CachedSpatialGrid::default());
        ecs.insert(EntitiesDiedLastTick::default());

        /* let slow_limit = match game_mode {
            GameMode::Server | GameMode::Client => num_cpu / 2,
            GameMode::Singleplayer => num_cpu / 4
        }.max(1); */
        // let slow_limit = (num_cpu / 2 + num_cpu / 4).max(1);
        /* tracing::trace!(?slow_limit, "Slow Thread limit");
        let thread_pool = slowjob::large_pool(slow_limit.min(64) as usize);
        ecs.insert(SlowJobPool::new(
            slow_limit,
            10_000,
            /*Arc::clone(*/thread_pool/*)*/,
        )); */
        ecs.insert(slowjob);

        // TODO: only register on the server
        ecs.insert(EventBus::<ServerEvent>::default());
        ecs.insert(comp::group::GroupManager::default());
        ecs.insert(RegionMap::new());
        ecs.insert(SysMetrics::default());
        ecs.insert(PhysicsMetrics::default());
        ecs.insert(Trades::default());
        ecs.insert(PlayerPhysicsSettings::default());

        // Load plugins from asset directory
        #[cfg(feature = "plugins")]
        ecs.insert(match PluginMgr::from_assets() {
            Ok(plugin_mgr) => {
                let ecs_world = EcsWorld {
                    entities: &ecs.entities(),
                    health: ecs.read_component().into(),
                    uid: ecs.read_component().into(),
                    uid_allocator: &ecs.read_resource::<UidAllocator>().into(),
                    player: ecs.read_component().into(),
                };
                if let Err(e) = plugin_mgr
                    .execute_event(&ecs_world, &plugin_api::event::PluginLoadEvent {
                        game_mode: ecs_role,
                    })
                {
                    tracing::debug!(?e, "Failed to run plugin init");
                    tracing::info!("Plugins disabled, enable debug logging for more information.");
                    PluginMgr::default()
                } else {
                    plugin_mgr
                }
            },
            Err(e) => {
                tracing::debug!(?e, "Failed to read plugins from assets");
                tracing::info!("Plugins disabled, enable debug logging for more information.");
                PluginMgr::default()
            },
        });

        ecs
    }

    /// Register a component with the state's ECS.
    #[must_use]
    pub fn with_component<T: Component>(mut self) -> Self
    where
        <T as Component>::Storage: Default,
    {
        self.ecs.register::<T>();
        self
    }

    /// Write a component attributed to a particular entity, ignoring errors.
    ///
    /// This should be used *only* when we can guarantee that the rest of the
    /// code does not rely on the insert having succeeded (meaning the
    /// entity is no longer alive!).
    ///
    /// Returns None if the entity was dead or there was no previous entry for
    /// this component; otherwise, returns Some(old_component).
    pub fn write_component_ignore_entity_dead<C: Component>(
        &mut self,
        entity: EcsEntity,
        comp: C,
    ) -> Option<C> {
        self.ecs
            .write_storage()
            .insert(entity, comp)
            .ok()
            .and_then(identity)
    }

    /// Delete a component attributed to a particular entity.
    pub fn delete_component<C: Component>(&mut self, entity: EcsEntity) -> Option<C> {
        self.ecs.write_storage().remove(entity)
    }

    /// Read a component attributed to a particular entity.
    pub fn read_component_cloned<C: Component + Clone>(&self, entity: EcsEntity) -> Option<C> {
        self.ecs.read_storage().get(entity).cloned()
    }

    /// Read a component attributed to a particular entity.
    pub fn read_component_copied<C: Component + Copy>(&self, entity: EcsEntity) -> Option<C> {
        self.ecs.read_storage().get(entity).copied()
    }

    /// Given mutable access to the resource R, assuming the resource
    /// component exists (this is already the behavior of functions like `fetch`
    /// and `write_component_ignore_entity_dead`).  Since all of our resources
    /// are generated up front, any failure here is definitely a code bug.
    pub fn mut_resource<R: Resource>(&mut self) -> &mut R {
        self.ecs.get_mut::<R>().expect(
            "Tried to fetch an invalid resource even though all our resources should be known at \
             compile time.",
        )
    }

    /// Get a read-only reference to the storage of a particular component type.
    pub fn read_storage<C: Component>(&self) -> EcsStorage<C, Fetch<EcsMaskedStorage<C>>> {
        self.ecs.read_storage::<C>()
    }

    /// Get a reference to the internal ECS world.
    pub fn ecs(&self) -> &specs::World { &self.ecs }

    /// Get a mutable reference to the internal ECS world.
    pub fn ecs_mut(&mut self) -> &mut specs::World { &mut self.ecs }

    pub fn thread_pool(&self) -> &Arc<ThreadPool> { &self.thread_pool }

    /// Get a reference to the `TerrainChanges` structure of the state. This
    /// contains information about terrain state that has changed since the
    /// last game tick.
    pub fn terrain_changes(&self) -> Fetch<TerrainChanges> { self.ecs.read_resource() }

    /// Get a reference the current in-game weather grid.
    pub fn weather_grid(&self) -> Fetch<WeatherGrid> { self.ecs.read_resource() }

    /// Get a mutable reference the current in-game weather grid.
    pub fn weather_grid_mut(&mut self) -> FetchMut<WeatherGrid> { self.ecs.write_resource() }

    /// Get the current weather at a position in worldspace.
    pub fn weather_at(&self, pos: Vec2<f32>) -> Weather {
        self.weather_grid().get_interpolated(pos)
    }

    /// Get the max weather near a position in worldspace.
    pub fn max_weather_near(&self, pos: Vec2<f32>) -> Weather {
        self.weather_grid().get_max_near(pos)
    }

    /// Get the current in-game time of day.
    ///
    /// Note that this should not be used for physics, animations or other such
    /// localised timings.
    pub fn get_time_of_day(&self) -> f64 { self.ecs.read_resource::<TimeOfDay>().0 }

    /// Get the current in-game day period (period of the day/night cycle)
    pub fn get_day_period(&self) -> DayPeriod { self.get_time_of_day().into() }

    /// Get the current in-game time.
    ///
    /// Note that this does not correspond to the time of day.
    pub fn get_time(&self) -> f64 { self.ecs.read_resource::<Time>().0 }

    /// Get the current delta time.
    pub fn get_delta_time(&self) -> f32 { self.ecs.read_resource::<DeltaTime>().0 }

    /// Get a reference to this state's terrain.
    pub fn terrain(&self) -> Fetch<TerrainGrid> { self.ecs.read_resource() }

    /// Get a reference to this state's terrain.
    pub fn slow_job_pool(&self) -> Fetch<SlowJobPool> { self.ecs.read_resource() }

    /// Get a writable reference to this state's terrain.
    pub fn terrain_mut(&self) -> FetchMut<TerrainGrid> { self.ecs.write_resource() }

    /// Get a block in this state's terrain.
    pub fn get_block(&self, pos: Vec3<i32>) -> Option<Block> {
        self.terrain().get(pos).ok().copied()
    }

    /// Set a block in this state's terrain.
    pub fn set_block(&self, pos: Vec3<i32>, block: Block) {
        self.ecs.write_resource::<BlockChange>().set(pos, block);
    }

    /// Check if the block at given position `pos` has already been modified
    /// this tick.
    pub fn can_set_block(&self, pos: Vec3<i32>) -> bool {
        !self
            .ecs
            .read_resource::<BlockChange>()
            .blocks
            .contains_key(&pos)
    }

    /// Removes every chunk of the terrain.
    pub fn clear_terrain(&mut self) {
        let removed_chunks = &mut self.ecs.write_resource::<TerrainChanges>().removed_chunks;

        self.terrain_mut().drain().for_each(|(key, _)| {
            removed_chunks.insert(key);
        });
    }

    /// Insert the provided chunk into this state's terrain.
    pub fn insert_chunk(&mut self, key: Vec2<i32>, chunk: Arc<TerrainChunk>) {
        if self
            .ecs
            .write_resource::<TerrainGrid>()
            .insert(key, chunk)
            .is_some()
        {
            self.ecs
                .write_resource::<TerrainChanges>()
                .modified_chunks
                .insert(key);
        } else {
            self.ecs
                .write_resource::<TerrainChanges>()
                .new_chunks
                .insert(key);
        }
    }

    /// Remove the chunk with the given key from this state's terrain, if it
    /// exists.
    pub fn remove_chunk(&self, key: Vec2<i32>) -> Option<Arc<TerrainChunk>> {
        self
            .ecs
            .write_resource::<TerrainGrid>()
            .remove(key)
            .map(|chunk| {
                self.ecs
                    .write_resource::<TerrainChanges>()
                    .removed_chunks
                    .insert(key);
                chunk
            })
    }

    // Run RegionMap tick to update entity region occupancy
    pub fn update_region_map(&self) {
        span!(_guard, "update_region_map", "State::update_region_map");
        self.ecs.write_resource::<RegionMap>().tick(
            self.ecs.read_storage::<comp::Pos>(),
            self.ecs.read_storage::<comp::Vel>(),
            self.ecs.entities(),
        );
    }

    // Apply terrain changes
    pub fn apply_terrain_changes(&self) { self.apply_terrain_changes_internal(false); }

    /// `during_tick` is true if and only if this is called from within
    /// [State::tick].
    ///
    /// This only happens if [State::tick] is asked to update terrain itself
    /// (using `update_terrain_and_regions: true`).  [State::tick] is called
    /// from within both the client and the server ticks, right after
    /// handling terrain messages; currently, client sets it to true and
    /// server to false.
    fn apply_terrain_changes_internal(&self, during_tick: bool) {
        span!(
            _guard,
            "apply_terrain_changes",
            "State::apply_terrain_changes"
        );
        let mut terrain = self.ecs.write_resource::<TerrainGrid>();
        let mut modified_blocks =
            std::mem::take(&mut self.ecs.write_resource::<BlockChange>().blocks);
        // Apply block modifications
        // Only include in `TerrainChanges` if successful
        modified_blocks.retain(|pos, block| {
            let res = terrain.set(*pos, *block);
            if let (&Ok(old_block), true) = (&res, during_tick) {
                // NOTE: If the changes are applied during the tick, we push the *old* value as
                // the modified block (since it otherwise can't be recovered after the tick).
                // Otherwise, the changes will be applied after the tick, so we push the *new*
                // value.
                *block = old_block;
            }
            res.is_ok()
        });
        self.ecs.write_resource::<TerrainChanges>().modified_blocks = modified_blocks;
    }

    /// Execute a single tick, simulating the game state by the given duration.
    pub fn tick(
        &mut self,
        dt: Duration,
        add_systems: impl Fn(&mut DispatcherBuilder),
        update_terrain_and_regions: bool,
    ) {
        span!(_guard, "tick", "State::tick");
        // Change the time accordingly.
        self.ecs.write_resource::<TimeOfDay>().0 += dt.as_secs_f64() * DAY_CYCLE_FACTOR;
        self.ecs.write_resource::<Time>().0 += dt.as_secs_f64();

        // Update delta time.
        // Beyond a delta time of MAX_DELTA_TIME, start lagging to avoid skipping
        // important physics events.
        self.ecs.write_resource::<DeltaTime>().0 = dt.as_secs_f32().min(MAX_DELTA_TIME);

        if update_terrain_and_regions {
            self.update_region_map();
        }

        span!(guard, "create dispatcher");
        // Run systems to update the world.
        // Create and run a dispatcher for ecs systems.
        let mut dispatch_builder =
            DispatcherBuilder::new().with_pool(Arc::clone(&self.thread_pool));
        // TODO: Consider alternative ways to do this
        add_systems(&mut dispatch_builder);
        // This dispatches all the systems in parallel.
        let mut dispatcher = dispatch_builder.build();
        drop(guard);

        span!(guard, "run systems");
        dispatcher.dispatch(&self.ecs);
        drop(guard);

        span!(guard, "maintain ecs");
        self.ecs.maintain();
        drop(guard);

        if update_terrain_and_regions {
            self.apply_terrain_changes_internal(true);
        }

        // Process local events
        span!(guard, "process local events");

        let outcomes = self.ecs.read_resource::<EventBus<Outcome>>();
        let mut outcomes_emitter = outcomes.emitter();

        let events = self.ecs.read_resource::<EventBus<LocalEvent>>().recv_all();
        for event in events {
            let mut velocities = self.ecs.write_storage::<comp::Vel>();
            let physics = self.ecs.read_storage::<comp::PhysicsState>();
            match event {
                LocalEvent::Jump(entity, impulse) => {
                    if let Some(vel) = velocities.get_mut(entity) {
                        vel.0.z = impulse + physics.get(entity).map_or(0.0, |ps| ps.ground_vel.z);
                    }
                },
                LocalEvent::ApplyImpulse { entity, impulse } => {
                    if let Some(vel) = velocities.get_mut(entity) {
                        vel.0 = impulse;
                    }
                },
                LocalEvent::Boost {
                    entity,
                    vel: extra_vel,
                } => {
                    if let Some(vel) = velocities.get_mut(entity) {
                        vel.0 += extra_vel;
                    }
                },
                LocalEvent::CreateOutcome(outcome) => {
                    outcomes_emitter.emit(outcome);
                },
            }
        }
        drop(guard);
    }

    /// Clean up the state after a tick.
    pub fn cleanup(&mut self) {
        span!(_guard, "cleanup", "State::cleanup");
        // Clean up data structures from the last tick.
        self.ecs.write_resource::<TerrainChanges>().clear();
    }
}
