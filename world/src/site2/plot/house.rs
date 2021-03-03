use super::*;
use crate::{util::SQUARE_4, Land};
use common::terrain::{Block, BlockKind, SpriteKind};
use rand::prelude::*;
use vek::*;

pub struct House {
    door_tile: Vec2<i32>,
    tile_aabr: Aabr<i32>,
    bounds: Aabr<i32>,
    alt: i32,
    levels: u32,
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
            door_tile,
            tile_aabr,
            bounds: Aabr {
                min: site.tile_wpos(tile_aabr.min),
                max: site.tile_wpos(tile_aabr.max),
            },
            alt: land.get_alt_approx(site.tile_center_wpos(door_tile)) as i32 + 2,
            levels: rng.gen_range(1..3),
        }
    }
}

// upstream?
trait IntoVec3<T> {
    fn into(self) -> Vec3<T>;
}

impl<T: Clone> IntoVec3<T> for Vec3<T> {
    fn into(self) -> Vec3<T> { self }
}

impl<T: Clone> IntoVec3<T> for (T, T, T) {
    fn into(self) -> Vec3<T> { Vec3::new(self.0, self.1, self.2) }
}

impl<T: Clone> IntoVec3<T> for (Vec2<T>, T) {
    fn into(self) -> Vec3<T> { Vec3::new(self.0.x, self.0.y, self.1) }
}

trait IntoBlock {
    fn into(self) -> Block;
}

impl IntoBlock for Block {
    fn into(self) -> Block { self }
}

impl<C: Into<Rgb<u8>>> IntoBlock for (BlockKind, C) {
    fn into(self) -> Block { Block::new(self.0, self.1.into()) }
}

trait PrimHelper {
    fn aabb<T: IntoVec3<i32>>(&mut self, min: T, max: T) -> Id<Primitive>;
    fn pyramid<T: IntoVec3<i32>>(&mut self, min: T, max: T, inset: i32) -> Id<Primitive>;
    fn empty(&mut self) -> Id<Primitive>;
    fn or(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive>;
    fn xor(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive>;
    fn and(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive>;
}

trait FillHelper {
    fn fill<B: IntoBlock>(&mut self, prim: Id<Primitive>, block: B);
}

impl<F: FnMut(Primitive) -> Id<Primitive>> PrimHelper for F {
    fn aabb<T: IntoVec3<i32>>(&mut self, min: T, max: T) -> Id<Primitive> {
        self(Primitive::Aabb(Aabb {
            min: min.into(),
            max: max.into(),
        }))
    }

    fn pyramid<T: IntoVec3<i32>>(&mut self, min: T, max: T, inset: i32) -> Id<Primitive> {
        self(Primitive::Pyramid {
            aabb: Aabb {
                min: min.into(),
                max: max.into(),
            },
            inset,
        })
    }

    fn empty(&mut self) -> Id<Primitive> { self(Primitive::Empty) }

    fn or(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive> {
        self(Primitive::Or(a, b))
    }

    fn xor(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive> {
        self(Primitive::Xor(a, b))
    }

    fn and(&mut self, a: Id<Primitive>, b: Id<Primitive>) -> Id<Primitive> {
        self(Primitive::And(a, b))
    }
}

impl<G: FnMut(Fill)> FillHelper for G {
    fn fill<B: IntoBlock>(&mut self, prim: Id<Primitive>, block: B) {
        self(Fill {
            prim,
            block: block.into(),
        })
    }
}

impl Structure for House {
    fn render<F: FnMut(Primitive) -> Id<Primitive>, G: FnMut(Fill)>(
        &self,
        site: &Site,
        mut prim: F,
        mut fill: G,
    ) {
        let storey = 6;
        let roof = storey * self.levels as i32;
        let foundations = 12;

        // Walls
        let outer = prim.aabb(
            (self.bounds.min, self.alt - foundations),
            (self.bounds.max, self.alt + roof),
        );
        let inner = prim.aabb(
            (self.bounds.min + 1, self.alt + 0),
            (self.bounds.max - 1, self.alt + roof),
        );
        let walls = prim.xor(outer, inner);
        fill.fill(walls, (BlockKind::Rock, (181, 170, 148)));

        // wall pillars
        let mut pillars = prim.empty();
        for x in self.tile_aabr.min.x + 1..self.tile_aabr.max.x {
            let pillar = prim.aabb(
                (site.tile_wpos((x, self.tile_aabr.min.y)), self.alt),
                (
                    site.tile_wpos((x, self.tile_aabr.max.y)) + Vec2::unit_x(),
                    self.alt + roof,
                ),
            );
            pillars = prim.or(pillars, pillar);
        }
        for y in self.tile_aabr.min.y + 1..self.tile_aabr.max.y {
            let pillar = prim.aabb(
                (site.tile_wpos((self.tile_aabr.min.x, y)), self.alt),
                (
                    site.tile_wpos((self.tile_aabr.max.x, y)) + Vec2::unit_y(),
                    self.alt + roof,
                ),
            );
            pillars = prim.or(pillars, pillar);
        }
        fill.fill(prim.and(walls, pillars), (BlockKind::Wood, (89, 44, 14)));

        // For each storey...
        for i in 0..self.levels + 1 {
            let height = storey * i as i32;

            // Windows x axis
            {
                let mut windows = prim.empty();
                for y in self.tile_aabr.min.y..self.tile_aabr.max.y {
                    let window = prim.aabb(
                        (
                            site.tile_wpos((self.tile_aabr.min.x, y)) + Vec2::unit_y() * 2,
                            self.alt + height + 2,
                        ),
                        (
                            site.tile_wpos((self.tile_aabr.max.x, y + 1)) - Vec2::unit_y() * 1,
                            self.alt + height + 5,
                        ),
                    );
                    windows = prim.or(windows, window);
                }
                fill.fill(
                    prim.and(walls, windows),
                    Block::air(SpriteKind::Window1).with_ori(2).unwrap(),
                );
            }
            // Windows y axis
            {
                let mut windows = prim.empty();
                for x in self.tile_aabr.min.x..self.tile_aabr.max.x {
                    let window = prim.aabb(
                        (
                            site.tile_wpos((x, self.tile_aabr.min.y)) + Vec2::unit_x() * 2,
                            self.alt + height + 2,
                        ),
                        (
                            site.tile_wpos((x + 1, self.tile_aabr.max.y)) - Vec2::unit_x() * 1,
                            self.alt + height + 5,
                        ),
                    );
                    windows = prim.or(windows, window);
                }
                fill.fill(
                    prim.and(walls, windows),
                    Block::air(SpriteKind::Window1).with_ori(0).unwrap(),
                );
            }

            // Floor
            fill.fill(
                prim.aabb(
                    (self.bounds.min, self.alt + height + 0),
                    (self.bounds.max, self.alt + height + 1),
                ),
                (BlockKind::Rock, (89, 44, 14)),
            );
        }

        // Corner pillars
        for &rpos in SQUARE_4.iter() {
            let pos = self.bounds.min + (self.bounds.max - self.bounds.min) * rpos;
            fill(Fill {
                prim: prim.aabb(
                    (pos - 1, self.alt - foundations),
                    (pos + 1, self.alt + roof),
                ),
                block: Block::new(BlockKind::Wood, Rgb::new(89, 44, 14)),
            });
        }

        let roof_lip = 3;
        let roof_height = (self.bounds.min - self.bounds.max)
            .map(|e| e.abs())
            .reduce_min()
            / 2
            + roof_lip;

        // Roof
        fill.fill(
            prim.pyramid(
                (self.bounds.min - roof_lip, self.alt + roof),
                (self.bounds.max + roof_lip, self.alt + roof + roof_height),
                roof_height,
            ),
            (BlockKind::Wood, (21, 43, 48)),
        );

        // Foundations
        fill.fill(
            prim.aabb(
                (self.bounds.min - 1, self.alt - foundations),
                (self.bounds.max + 1, self.alt + 1),
            ),
            (BlockKind::Rock, (31, 33, 32)),
        );
    }
}
