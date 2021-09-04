#![allow(dead_code)] // TODO: Remove this when rtsim is fleshed out

mod chunks;
mod entity;
mod load_chunks;
mod tick;
mod unload_chunks;

use self::chunks::Chunks;
use common::{
    comp,
    rtsim::{Memory, RtSimController, RtSimEntity, RtSimId},
    terrain::TerrainChunk,
    vol::RectRasterableVol,
};
use common_ecs::{dispatch, System};
use common_state::State;
use rand::prelude::*;
use slab::Slab;
use specs::{DispatcherBuilder, WorldExt};
use vek::*;

pub use self::entity::{Brain, Entity, RtSimEntityKind};

pub struct RtSim {
    tick: u64,
    chunks: Chunks,
    entities: Slab<Entity>,
}

impl RtSim {
    pub fn new(world_chunk_size: Vec2<u32>) -> Self {
        Self {
            tick: 0,
            chunks: Chunks::new(world_chunk_size),
            entities: Slab::new(),
        }
    }

    pub fn hook_load_chunk(&mut self, key: Vec2<i32>) {
        if let Some(chunk) = self.chunks.chunk_mut(key) {
            if !chunk.is_loaded {
                chunk.is_loaded = true;
                self.chunks.chunks_to_load.push(key);
            }
        }
    }

    pub fn hook_unload_chunk(&mut self, key: Vec2<i32>) {
        if let Some(chunk) = self.chunks.chunk_mut(key) {
            if chunk.is_loaded {
                chunk.is_loaded = false;
                self.chunks.chunks_to_unload.push(key);
            }
        }
    }

    pub fn assimilate_entity(&mut self, entity: RtSimId) {
        // tracing::info!("Assimilated rtsim entity {}", entity);
        self.entities.get_mut(entity).map(|e| e.is_loaded = false);
    }

    pub fn reify_entity(&mut self, entity: RtSimId) {
        // tracing::info!("Reified rtsim entity {}", entity);
        self.entities.get_mut(entity).map(|e| e.is_loaded = true);
    }

    pub fn update_entity(&mut self, entity: RtSimId, pos: Vec3<f32>) {
        self.entities.get_mut(entity).map(|e| e.pos = pos);
    }

    pub fn destroy_entity(&mut self, entity: RtSimId) {
        // tracing::info!("Destroyed rtsim entity {}", entity);
        self.entities.remove(entity);
    }

    pub fn get_entity(&self, entity: RtSimId) -> Option<&Entity> { self.entities.get(entity) }

    pub fn insert_entity_memory(&mut self, entity: RtSimId, memory: Memory) {
        self.entities
            .get_mut(entity)
            .map(|entity| entity.brain.add_memory(memory));
    }

    pub fn forget_entity_enemy(&mut self, entity: RtSimId, name: &str) {
        if let Some(entity) = self.entities.get_mut(entity) {
            entity.brain.forget_enemy(name);
        }
    }

    pub fn set_entity_mood(&mut self, entity: RtSimId, memory: Memory) {
        self.entities
            .get_mut(entity)
            .map(|entity| entity.brain.set_mood(memory));
    }
}

pub fn add_server_systems(dispatch_builder: &mut DispatcherBuilder) {
    dispatch::<unload_chunks::Sys>(dispatch_builder, &[]);
    dispatch::<load_chunks::Sys>(dispatch_builder, &[&unload_chunks::Sys::sys_name()]);
    dispatch::<tick::Sys>(dispatch_builder, &[
        &load_chunks::Sys::sys_name(),
        &unload_chunks::Sys::sys_name(),
    ]);
}

pub fn init(
    state: &mut State,
    #[cfg(feature = "worldgen")] world: &world::World,
    #[cfg(feature = "worldgen")] index: world::IndexRef,
) {
    #[cfg(feature = "worldgen")]
    let mut rtsim = RtSim::new(world.sim().get_size());
    #[cfg(not(feature = "worldgen"))]
    let mut rtsim = RtSim::new(Vec2::new(40, 40));

    // TODO: Determine number of rtsim entities based on things like initial site
    // populations rather than world size
    #[cfg(feature = "worldgen")]
    {
        for _ in 0..world.sim().get_size().product() / 400 {
            let pos = rtsim
                .chunks
                .size()
                .map2(TerrainChunk::RECT_SIZE, |sz, chunk_sz| {
                    thread_rng().gen_range(0..sz * chunk_sz) as i32
                });

            rtsim.entities.insert(Entity {
                is_loaded: false,
                pos: Vec3::from(pos.map(|e| e as f32)),
                seed: thread_rng().gen(),
                controller: RtSimController::default(),
                last_time_ticked: 0.0,
                kind: RtSimEntityKind::Random,
                brain: Default::default(),
            });
        }
        for (site_id, site) in world
            .civs()
            .sites
            .iter()
            .filter_map(|(site_id, site)| site.site_tmp.map(|id| (site_id, &index.sites[id])))
        {
            use world::site::SiteKind;
            match &site.kind {
                SiteKind::Dungeon(dungeon) => match dungeon.dungeon_difficulty() {
                    Some(5) => {
                        let pos = site.get_origin();
                        if let Some(nearest_village) = world
                            .civs()
                            .sites
                            .iter()
                            .filter(|s| s.1.is_settlement())
                            .min_by_key(|(_, site)| {
                                let wpos = site.center * TerrainChunk::RECT_SIZE.map(|e| e as i32);
                                wpos.map(|e| e as f32)
                                    .distance_squared(pos.map(|x| x as f32))
                                    as u32
                            })
                            .map(|(id, _)| id)
                        {
                            for _ in 0..25 {
                                rtsim.entities.insert(Entity {
                                    is_loaded: false,
                                    pos: Vec3::from(pos.map(|e| e as f32)),
                                    seed: thread_rng().gen(),
                                    controller: RtSimController::default(),
                                    last_time_ticked: 0.0,
                                    kind: RtSimEntityKind::Cultist,
                                    brain: Brain::raid(site_id, nearest_village),
                                });
                            }
                        }
                    },
                    _ => {},
                },
                _ => {},
            }
        }
    }

    state.ecs_mut().insert(rtsim);
    state.ecs_mut().register::<RtSimEntity>();
    tracing::info!("Initiated real-time world simulation");
}
