use super::*;
use vek::*;

pub struct Hut {
    root: Vec2<i32>,
    tile_aabr: Aabr<i32>,
    bounds: Aabr<i32>,
    alt: i32,
    height: i32,
    door_dir: Vec2<i32>,
}

impl Structure for Hut {
    type Config = ();

    fn choose_location<R: Rng>(cfg: Self::Config, land: &Land, site: &Site, rng: &mut R) -> Option<Self> {
        let (tile_aabr, root) = site.tiles.find_near(
            Vec2::zero(),
            |tile, _| if rng.gen_range(0..16) == 0 {
                site.tiles.grow_aabr(tile, 4..9, (2, 2), 2).ok()
            } else {
                None
            },
        )?;
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
            kind: PlotKind::Hut(self),
        });

        site.blit_aabr(aabr, Tile {
            kind: TileKind::Building,
            plot: Some(plot),
        });
    }
}

impl Render for Hut {
    fn render<F: FnMut(Primitive) -> Id<Primitive>, G: FnMut(Id<Primitive>, Fill)>(
        &self,
        site: &Site,
        mut prim: F,
        mut fill: G,
    ) {
        let daub = Fill::Block(Block::new(BlockKind::Wood, Rgb::new(110, 50, 16)));

        let outer = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min + 1).with_z(self.alt),
            max: (self.bounds.max - 1).with_z(self.alt + self.height),
        }));
        let inner = prim(Primitive::Aabb(Aabb {
            min: (self.bounds.min + 2).with_z(self.alt),
            max: (self.bounds.max - 2).with_z(self.alt + self.height),
        }));

        let door_pos = site.tile_center_wpos(self.root);
        let door = prim(Primitive::Aabb(Aabb {
            min: door_pos.with_z(self.alt),
            max: (door_pos + self.door_dir * 32).with_z(self.alt + 2),
        }.made_valid()));

        let space = prim(Primitive::And(inner, door));

        let walls = prim(Primitive::AndNot(outer, space));

        fill(walls, daub);

        let roof_lip = 2;
        let roof_height = (self.bounds.min - self.bounds.max)
            .map(|e| e.abs())
            .reduce_min()
            .saturating_sub(1)
            / 2
            + roof_lip
            + 1;
        let roof = prim(Primitive::Pyramid{
            aabb: Aabb {
                min: (self.bounds.min + 1 - roof_lip).with_z(self.alt + self.height),
                max: (self.bounds.max - 1 + roof_lip).with_z(self.alt + self.height + roof_height),
            },
            inset: Vec2::broadcast(roof_height),
        });
        fill(roof, daub);
    }
}
