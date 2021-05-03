use super::*;
use crate::Land;
use common::terrain::{Block, BlockKind, SpriteKind};
use rand::prelude::*;
use vek::*;

pub struct House {
    _door_tile: Vec2<i32>,
    tile_aabr: Aabr<i32>,
    bounds: Aabr<i32>,
    alt: i32,
    levels: u32,
    roof_color: Rgb<u8>,
    roof_inset: Vec2<bool>,
}

impl House {
    pub fn generate(
        land: &Land,
        rng: &mut impl Rng,
        site: &Site,
        door_tile: Vec2<i32>,
        tile_aabr: Aabr<i32>,
    ) -> Self {
        Self {
            _door_tile: door_tile,
            tile_aabr,
            bounds: Aabr {
                min: site.tile_wpos(tile_aabr.min),
                max: site.tile_wpos(tile_aabr.max),
            },
            alt: land.get_alt_approx(site.tile_center_wpos(door_tile)) as i32 + 2,
            levels: rng.gen_range(1..2 + (tile_aabr.max - tile_aabr.min).product() / 6) as u32,
            roof_color: {
                let colors = [
                    Rgb::new(21, 43, 48),
                    Rgb::new(11, 23, 38),
                    Rgb::new(45, 28, 21),
                    Rgb::new(10, 55, 40),
                    Rgb::new(5, 35, 15),
                    Rgb::new(40, 5, 11),
                    Rgb::new(55, 45, 11),
                ];
                *colors.choose(rng).unwrap()
            },
            roof_inset: match rng.gen_range(0..3) {
                0 => Vec2::new(true, false),
                1 => Vec2::new(false, true),
                _ => Vec2::new(true, true),
            },
        }
    }
}

impl Render for House {
    fn render<F: FnMut(Primitive) -> Id<Primitive>, G: FnMut(Id<Primitive>, Fill)>(
        &self,
        site: &Site,
        mut prim: F,
        mut fill: G,
    ) {
        let storey = 5;
        let roof = storey * self.levels as i32;
        let foundations = 12;
        let plot = prim(Primitive::Plot);
        let void = prim(Primitive::Void);
        let can_spill = prim(Primitive::Or(plot, void));

        //let wall_block = Fill::Brick(BlockKind::Rock, Rgb::new(80, 75, 85), 24);
        let wall_block = Fill::Brick(BlockKind::Rock, Rgb::new(158, 150, 121), 24);
        let structural_wood = Fill::Block(Block::new(BlockKind::Wood, Rgb::new(55, 25, 8)));

        // Walls
        let inner = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min + 1).with_z(self.alt),
            max: self.bounds.max.with_z(self.alt + roof),
        }));
        let outer = prim(Primitive::Aabb(Aabb {
            min: self.bounds.min.with_z(self.alt - foundations),
            max: (self.bounds.max + 1).with_z(self.alt + roof),
        }));
        fill(outer, wall_block);
        fill(inner, Fill::Block(Block::empty()));
        let walls = prim(Primitive::Xor(outer, inner));

        // wall pillars
        let mut pillars_y = prim(Primitive::Empty);
        for x in self.tile_aabr.min.x..self.tile_aabr.max.x + 2 {
            let pillar = prim(Primitive::Aabb(Aabb {
                min: site
                    .tile_wpos(Vec2::new(x, self.tile_aabr.min.y))
                    .with_z(self.alt),
                max: (site.tile_wpos(Vec2::new(x, self.tile_aabr.max.y + 1)) + Vec2::unit_x())
                    .with_z(self.alt + roof),
            }));
            pillars_y = prim(Primitive::Or(pillars_y, pillar));
        }
        let mut pillars_x = prim(Primitive::Empty);
        for y in self.tile_aabr.min.y..self.tile_aabr.max.y + 2 {
            let pillar = prim(Primitive::Aabb(Aabb {
                min: site
                    .tile_wpos(Vec2::new(self.tile_aabr.min.x, y))
                    .with_z(self.alt),
                max: (site.tile_wpos(Vec2::new(self.tile_aabr.max.x + 1, y)) + Vec2::unit_y())
                    .with_z(self.alt + roof),
            }));
            pillars_x = prim(Primitive::Or(pillars_x, pillar));
        }
        let pillars = prim(Primitive::And(pillars_x, pillars_y));
        fill(pillars, structural_wood);

        // For each storey...
        for i in 0..self.levels + 1 {
            let height = storey * i as i32;
            let window_height = storey - 3;

            // Windows x axis
            {
                let mut windows = prim(Primitive::Empty);
                for y in self.tile_aabr.min.y..self.tile_aabr.max.y {
                    let window = prim(Primitive::Aabb(Aabb {
                        min: (site.tile_wpos(Vec2::new(self.tile_aabr.min.x, y))
                            + Vec2::unit_y() * 2)
                            .with_z(self.alt + height + 2),
                        max: (site.tile_wpos(Vec2::new(self.tile_aabr.max.x, y + 1))
                            + Vec2::new(1, -1))
                        .with_z(self.alt + height + 2 + window_height),
                    }));
                    windows = prim(Primitive::Or(windows, window));
                }
                fill(
                    prim(Primitive::And(walls, windows)),
                    Fill::Block(Block::air(SpriteKind::Window1).with_ori(2).unwrap()),
                );
            }
            // Windows y axis
            {
                let mut windows = prim(Primitive::Empty);
                for x in self.tile_aabr.min.x..self.tile_aabr.max.x {
                    let window = prim(Primitive::Aabb(Aabb {
                        min: (site.tile_wpos(Vec2::new(x, self.tile_aabr.min.y))
                            + Vec2::unit_x() * 2)
                            .with_z(self.alt + height + 2),
                        max: (site.tile_wpos(Vec2::new(x + 1, self.tile_aabr.max.y))
                            + Vec2::new(-1, 1))
                        .with_z(self.alt + height + 2 + window_height),
                    }));
                    windows = prim(Primitive::Or(windows, window));
                }
                fill(
                    prim(Primitive::And(walls, windows)),
                    Fill::Block(Block::air(SpriteKind::Window1).with_ori(0).unwrap()),
                );
            }

            // Floor
            let floor = prim(Primitive::Aabb(Aabb {
                min: (self.bounds.min + 1).with_z(self.alt + height),
                max: self.bounds.max.with_z(self.alt + height + 1),
            }));
            fill(
                floor,
                Fill::Block(Block::new(BlockKind::Rock, Rgb::new(89, 44, 14))),
            );

            let slice = prim(Primitive::Aabb(Aabb {
                min: self.bounds.min.with_z(self.alt + height),
                max: (self.bounds.max + 1).with_z(self.alt + height + 1),
            }));
            fill(prim(Primitive::AndNot(slice, floor)), structural_wood);
        }

        let roof_lip = 2;
        let roof_height = (self.bounds.min - self.bounds.max)
            .map(|e| e.abs())
            .reduce_min()
            / 2
            + roof_lip
            + 1;

        // Roof
        let roof_vol = prim(Primitive::Pyramid {
            aabb: Aabb {
                min: (self.bounds.min - roof_lip).with_z(self.alt + roof),
                max: (self.bounds.max + 1 + roof_lip).with_z(self.alt + roof + roof_height),
            },
            inset: self.roof_inset.map(|e| if e { roof_height } else { 0 }),
        });
        let eaves = prim(Primitive::Offset(roof_vol, -Vec3::unit_z()));
        let tiles = prim(Primitive::AndNot(roof_vol, eaves));
        fill(
            prim(Primitive::And(tiles, can_spill)),
            Fill::Block(Block::new(BlockKind::Wood, self.roof_color)),
        );
        let roof_inner = prim(Primitive::Aabb(Aabb {
            min: self.bounds.min.with_z(self.alt + roof),
            max: (self.bounds.max + 1).with_z(self.alt + roof + roof_height),
        }));
        fill(prim(Primitive::And(eaves, roof_inner)), wall_block);

        // Foundations
        let foundations = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min - 1).with_z(self.alt - foundations),
            max: (self.bounds.max + 2).with_z(self.alt + 1),
        }));
        fill(
            prim(Primitive::And(foundations, can_spill)),
            Fill::Block(Block::new(BlockKind::Rock, Rgb::new(31, 33, 32))),
        );
    }
}
