use super::*;
use vek::*;

pub struct Cemetary {
    root: Vec2<i32>,
    tile_aabr: Aabr<i32>,
    bounds: Aabr<i32>,
    alt: i32,
    height: i32,
    door_dir: Vec2<i32>,
}

impl Structure for Cemetary {
    type Config = ();

    fn choose_location<R: Rng>(
        cfg: Self::Config,
        land: &Land,
        site: &Site,
        rng: &mut R,
    ) -> Option<Self> {
        let (tile_aabr, root) = site.tiles.find_near(Vec2::zero(), |tile, _| {
            if rng.gen_range(0..16) == 0 {
                site.tiles.grow_aabr(tile, 15..20, (2, 2), 2).ok()
            } else {
                None
            }
        })?;
        let center = (tile_aabr.min + (tile_aabr.max - 1)) / 2;

        Some(Self {
            root,
            tile_aabr,
            bounds: Aabr {
                min: site.tile_wpos(tile_aabr.min),
                max: site.tile_wpos(tile_aabr.max),
            },
            alt: land.get_alt_approx(site.tile_center_wpos(center)) as i32,
            height: 4,
            door_dir: match rng.gen_range(0..4) {
                0 => Vec2::unit_x(),
                1 => -Vec2::unit_x(),
                2 => Vec2::unit_y(),
                3 => -Vec2::unit_y(),
                _ => unreachable!(),
            },
        })
    }

    fn generate<R: Rng>(self, land: &Land, site: &mut Site, rng: &mut R) {
        let aabr = self.tile_aabr;

        let plot = site.create_plot(Plot {
            root_tile: self.root,
            tiles: aabr_tiles(aabr).collect(),
            seed: rng.gen(),
            kind: PlotKind::Cemetary(self),
        });

        site.blit_aabr(aabr, Tile {
            kind: TileKind::Building,
            plot: Some(plot),
        });
    }
}

impl Render for Cemetary {
    fn render<F: FnMut(Primitive) -> Id<Primitive>, G: FnMut(Id<Primitive>, Fill)>(
        &self,
        site: &Site,
        mut prim: F,
        mut fill: G,
    ) {
        let mut rng = thread_rng();
        let mid = self.bounds.min + (self.bounds.max - self.bounds.min) / 2;
        let roof = 5;
        let wall_block = Fill::Block(Block::air(SpriteKind::IronBar));
        let lamp = Fill::Block(Block::air(SpriteKind::WallLampSmall));
        let floor_sprite = Fill::Block(match rng.gen_bool(0.1) {
            true => Block::air(match rng.gen_range(0..5) {
                0 => SpriteKind::ShortGrass,
                1 => SpriteKind::MediumGrass,
                _ => SpriteKind::LargeGrass,
            }),
            false => Block::empty(),
        });
        let structural_wood = Fill::Block(Block::new(BlockKind::Rock, Rgb::new(80, 80, 80)));
        let path = Fill::Block(Block::new(
            BlockKind::Rock,
            Rgb::new(54, 34, 29) + rng.gen_range(0..10),
        ));

        // Walls
        let inner = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min + 1).with_z(self.alt),
            max: self.bounds.max.with_z(self.alt + roof),
        }));
        let outer = prim(Primitive::Aabb(Aabb {
            min: self.bounds.min.with_z(self.alt),
            max: (self.bounds.max + 1).with_z(self.alt + roof),
        }));
        fill(outer, wall_block);

        let walls = prim(Primitive::Xor(outer, inner));

        // lol

        // Windows x axis
        {
            let mut windows = prim(Primitive::Empty);
            for y in self.tile_aabr.min.y..self.tile_aabr.max.y + 1 {
                let window = prim(Primitive::Aabb(Aabb {
                    min: (site.tile_wpos(Vec2::new(self.tile_aabr.min.x, y)) + Vec2::unit_x() * 1)
                        .with_z(self.alt + 3 + 1),
                    max: (site.tile_wpos(Vec2::new(self.tile_aabr.max.x + 1, y + 1))
                        - Vec2::unit_x() * 1)
                        .with_z(self.alt + 3 + 1 + 1),
                }));
                windows = prim(Primitive::Or(windows, window));
            }
            fill(
                prim(Primitive::And(walls, windows)),
                Fill::Block(Block::air(SpriteKind::IronBarCross).with_ori(0).unwrap()),
            );
        }
        // Windows y axis
        {
            let mut windows = prim(Primitive::Empty);
            for x in self.tile_aabr.min.x..self.tile_aabr.max.x + 1 {
                let window = prim(Primitive::Aabb(Aabb {
                    min: (site.tile_wpos(Vec2::new(x, self.tile_aabr.min.y)) + Vec2::unit_y() * 1)
                        .with_z(self.alt + 1 + 3),
                    max: (site.tile_wpos(Vec2::new(x + 1, self.tile_aabr.max.y)))
                        .with_z(self.alt + 1 + 3 + 1),
                }));
                windows = prim(Primitive::Or(windows, window));
            }
            fill(
                prim(Primitive::And(walls, windows)),
                Fill::Block(Block::air(SpriteKind::IronBarCross).with_ori(2).unwrap()),
            );
        }

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
        fill(inner, Fill::Block(Block::empty()));

        // Floor
        let floor = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min).with_z(self.alt),
            max: (self.bounds.max + 1).with_z(self.alt + 1),
        }));
        fill(
            floor,
            Fill::Block(Block::new(
                BlockKind::Grass,
                Rgb::new(48, 79, 30) + rng.gen_range(0..5),
            )),
        );

        let slice = prim(Primitive::Aabb(Aabb {
            min: self.bounds.min.with_z(self.alt),
            max: (self.bounds.max + 1).with_z(self.alt + 2),
        }));
        fill(prim(Primitive::AndNot(slice, floor)), structural_wood);

        // Grass
        let grass = Aabb {
            min: (self.bounds.min + 1).with_z(self.alt + 1),
            max: self.bounds.max.with_z(self.alt + 2),
        };

        for x in grass.min.x..grass.max.x {
            for y in grass.min.y..grass.max.y {
                let grass_sprite = Fill::Block(match rng.gen_bool(0.1) {
                    true => Block::air(match rng.gen_range(0..5) {
                        0 => SpriteKind::LargeGrass,
                        1 => SpriteKind::MediumGrass,
                        _ => SpriteKind::ShortGrass,
                    }),
                    false => Block::empty(),
                });
                let grass_block = prim(Primitive::Aabb(Aabb {
                    min: Vec2::new(x, y).with_z(self.alt + 1),
                    max: Vec2::new(x + 1, y + 1).with_z(self.alt + 2),
                }));
                fill(grass_block, grass_sprite);
            }
        }

        // Tombstones

        let dalle_block = Fill::Block(Block::new(BlockKind::Rock, Rgb::new(40, 40, 40)));

        for x in grass.min.x..grass.max.x {
            for y in grass.min.y..grass.max.y {
                let tomb = prim(Primitive::Aabb(Aabb {
                    min: Vec2::new(x, y).with_z(self.alt + 1),
                    max: Vec2::new(x + 1, y + 1).with_z(self.alt + 2),
                }));
                let dalle = prim(Primitive::Aabb(Aabb {
                    min: Vec2::new(x, y).with_z(self.alt + 0),
                    max: Vec2::new(x + 1, y + 1).with_z(self.alt + 1),
                }));
                if x % 4 == 0 && y % 5 == 1 {
                    let tomb_sprite = Fill::Block(Block::air(SpriteKind::Tombstones));
                    fill(tomb, tomb_sprite);
                }
                if x % 4 == 0 && (y % 5 == 0 || y % 5 == 1) {
                    fill(dalle, dalle_block);
                }
            }
        }

        // Paths
        for x in grass.min.x..grass.max.x {
            for y in grass.min.y..grass.max.y {
                let block = prim(Primitive::Aabb(Aabb {
                    min: Vec2::new(x, y).with_z(self.alt),
                    max: Vec2::new(x + 1, y + 1).with_z(self.alt + 1),
                }));
                if (y % 5 == 3 || y % 5 == 4) && rng.gen_bool(0.95) {
                    fill(block, path);
                }
            }
        }

        /*let path_x = prim(Primitive::Aabb(Aabb {
            min: Vec3::new(mid.x - 2, grass.min.y, self.alt),
            max: Vec3::new(mid.x + 3, grass.max.y, self.alt + 1),
        }));*/
        for x in grass.min.x..grass.max.x {
            for y in grass.min.y..grass.max.y {
                let block = prim(Primitive::Aabb(Aabb {
                    min: Vec2::new(x, y).with_z(self.alt),
                    max: Vec2::new(x + 1, y + 1).with_z(self.alt + 1),
                }));
                if x > mid.x - 2 && x < mid.x + 2 && rng.gen_bool(0.95) {
                    fill(block, path);
                }
            }
        }
        //fill(path_x, path);

        /*let test1 = prim(Primitive::Cylinder(Aabb {
            min: Vec3::new(mid.x - inline_tweak::tweak!(5), self.bounds.min.y - inline_tweak::tweak!(5), self.alt),
            max: Vec3::new(mid.x + inline_tweak::tweak!(5), self.bounds.min.y + inline_tweak::tweak!(5), self.alt + 3),
        }));
        let test2 = prim(Primitive::Cylinder(Aabb {
            min: Vec3::new(mid.x - 3, mid.y - 4, self.alt),
            max: Vec3::new(mid.x + 3, mid.y + 4, self.alt),
        }));
        let xor = prim(Primitive::Xor(test1, test2));

        let rotation = Mat3::new(
            1, 0, 0,
            0, 0, -1,
            0, 1, 0
        );
        let rotated = prim(Primitive::Rotate(test1, rotation));
        let translated =  prim(Primitive::Offset(rotated, Vec3::new(0, 5, 0)));
        fill(translated, structural_wood);*/

        let high = prim(Primitive::Pyramid {
            aabb: Aabb {
                min: Vec2::new(mid.x - 3, self.bounds.min.y).with_z(self.alt + 6),
                max: Vec2::new(mid.x + 3, self.bounds.min.y + 1).with_z(self.alt + 9),
            },
            inset: Vec2::broadcast(5),
        });
        let door = prim(Primitive::Aabb(Aabb {
            min: Vec2::new(mid.x - 3, self.bounds.min.y).with_z(self.alt + 1),
            max: Vec2::new(mid.x + 3, self.bounds.min.y + 1).with_z(self.alt + 6),
        }));
        let door_empty = prim(Primitive::Aabb(Aabb {
            min: Vec2::new(mid.x - 2, self.bounds.min.y).with_z(self.alt + 1),
            max: Vec2::new(mid.x + 2, self.bounds.min.y + 1).with_z(self.alt + 6),
        }));
        let lamp_1 = prim(Primitive::Aabb(Aabb {
            min: Vec2::new(mid.x - 3, self.bounds.min.y - 1).with_z(self.alt + 3),
            max: Vec2::new(mid.x - 2, self.bounds.min.y).with_z(self.alt + 4),
        }));
        let lamp_2 = prim(Primitive::Aabb(Aabb {
            min: Vec2::new(mid.x + 2, self.bounds.min.y - 1).with_z(self.alt + 3),
            max: Vec2::new(mid.x + 3, self.bounds.min.y).with_z(self.alt + 4),
        }));
        fill(door, structural_wood);
        fill(door_empty, Fill::Block(Block::empty()));
        fill(high, structural_wood);
        fill(lamp_1, lamp);
        fill(lamp_2, lamp);
    }
}
