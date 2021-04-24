use crate::{
    sim::WorldSim,
    util::Grid,
};
use common::{
    terrain::TerrainChunkSize,
    vol::RectVolSize,
    lod::{LodZone, LodTree},
};
use vek::*;

pub struct LodInfo {
    zones: Grid<LodZone>,
}

impl LodInfo {
    pub fn new(sim: &WorldSim) -> Self {
        Self {
            zones: Grid::populate_from(
                sim.get_size().map(|e| (e.next_power_of_two() / LodZone::SIZE).max(1) as i32),
                |pos| {
                    let to_wpos = |pos| pos * (TerrainChunkSize::RECT_SIZE * LodZone::SIZE).map(|e| e as i32);

                    let zone_wpos = to_wpos(pos);
                    let trees = sim
                        .get_trees_area(zone_wpos, to_wpos(pos + 1))
                        .map(|attr| LodTree {
                            pos: (attr.pos - zone_wpos).map(|e| e as u16),
                            alt: attr.alt_approx as u16,
                        })
                        // TODO: filter those outside zone bounds
                        .collect();

                    LodZone { trees }
                },
            ),
        }
    }

    pub fn get_zone(&self, pos: Vec2<i32>) -> Option<&LodZone> {
        self.zones.get(pos)
    }
}
