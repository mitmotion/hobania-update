use common_net::msg::PresenceKind;
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use specs::{Component, DenseVecStorage, DerefFlaggedStorage, NullStorage, VecStorage};
use vek::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Presence {
    pub view_distance: u32,
    pub kind: PresenceKind,
    pub lossy_terrain_compression: bool,
}

impl Presence {
    pub fn new(view_distance: u32, kind: PresenceKind) -> Self {
        Self {
            view_distance,
            kind,
            lossy_terrain_compression: false,
        }
    }
}

impl Component for Presence {
    // Presence seems <= 64 bits, so it isn't worth using DenseVecStorage.
    type Storage = DerefFlaggedStorage<Self, VecStorage<Self>>;
}

// Distance from fuzzy_chunk before snapping to current chunk
pub const CHUNK_FUZZ: u32 = 2;
// Distance out of the range of a region before removing it from subscriptions
pub const REGION_FUZZ: u32 = 16;

#[derive(Clone, Debug)]
pub struct RegionSubscription {
    pub fuzzy_chunk: Vec2<i32>,
    pub regions: HashSet<Vec2<i32>>,
}

impl Component for RegionSubscription {
    type Storage = DerefFlaggedStorage<Self, DenseVecStorage<Self>>;
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct RepositionOnChunkLoad;

impl Component for RepositionOnChunkLoad {
    type Storage = NullStorage<Self>;
}
