use common::lod;
use hashbrown::HashMap;
use vek::*;
use world::World;

static EMPTY_ZONE: lod::Zone = lod::Zone {
    objects: Vec::new(),
};

pub struct Lod {
    pub zones: HashMap<Vec2<i32>, lod::Zone>,
}

impl Lod {
    pub fn from_world(world: &World, index: world::IndexRef) -> Self {
        let mut zones = HashMap::new();

        let zone_sz = (world.sim().get_size() + lod::ZONE_SIZE - 1) / lod::ZONE_SIZE;

        for i in 0..zone_sz.x {
            for j in 0..zone_sz.y {
                let zone_pos = Vec2::new(i, j).map(|e| e as i32);
                zones.insert(zone_pos, world.get_lod_zone(zone_pos, index));
            }
        }

        Self { zones }
    }

    pub fn zone(&self, zone_pos: Vec2<i32>) -> &lod::Zone {
        self.zones.get(&zone_pos).unwrap_or(&EMPTY_ZONE)
    }
}
