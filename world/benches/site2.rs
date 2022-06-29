use common::{
    generation::EntityInfo,
    store::{Id, Store},
    terrain::{Block, BlockKind, SpriteKind, TerrainChunk, TerrainChunkMeta, TerrainChunkSize},
    vol::RectVolSize,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use vek::{Aabr, Rgb, Vec2, Vec3};
use veloren_world::{
    config::CONFIG,
    sim::{FileOpts, WorldOpts, DEFAULT_WORLD_MAP},
    site2::{plot::PlotKind, Fill, Filler, Plot, Primitive, Site, Structure},
    CanvasInfo, Land, World,
};

/* #[allow(dead_code)]
fn count_prim_kinds(prims: &Store<Primitive>) -> HashMap<String, usize> {
    let mut ret = HashMap::new();
    for prim in prims.values() {
        match &prim {
            Primitive::Empty => {
                *ret.entry("Empty".to_string()).or_default() += 1;
            },
            Primitive::Aabb(_) => {
                *ret.entry("Aabb".to_string()).or_default() += 1;
            },
            Primitive::Pyramid { .. } => {
                *ret.entry("Pyramid".to_string()).or_default() += 1;
            },
            Primitive::Ramp { .. } => {
                *ret.entry("Ramp".to_string()).or_default() += 1;
            },
            Primitive::Gable { .. } => {
                *ret.entry("Gable".to_string()).or_default() += 1;
            },
            Primitive::Cylinder(_) => {
                *ret.entry("Cylinder".to_string()).or_default() += 1;
            },
            Primitive::Cone(_) => {
                *ret.entry("Cone".to_string()).or_default() += 1;
            },
            Primitive::Sphere(_) => {
                *ret.entry("Sphere".to_string()).or_default() += 1;
            },
            Primitive::Superquadric { .. } => {
                *ret.entry("Superquadratic".to_string()).or_default() += 1;
            },
            Primitive::Plane(_, _, _) => {
                *ret.entry("Plane".to_string()).or_default() += 1;
            },
            Primitive::Segment { .. } => {
                *ret.entry("Segment".to_string()).or_default() += 1;
            },
            Primitive::SegmentPrism { .. } => {
                *ret.entry("SegmentPrism".to_string()).or_default() += 1;
            },
            Primitive::Sampling(_, _) => {
                *ret.entry("Sampling".to_string()).or_default() += 1;
            },
            Primitive::Prefab(_) => {
                *ret.entry("Prefab".to_string()).or_default() += 1;
            },
            Primitive::Intersect(_, _) => {
                *ret.entry("Intersect".to_string()).or_default() += 1;
            },
            Primitive::Union(_, _) => {
                *ret.entry("Union".to_string()).or_default() += 1;
            },
            Primitive::Without(_, _) => {
                *ret.entry("Without".to_string()).or_default() += 1;
            },
            Primitive::RotateAbout(_, _, _) => {
                *ret.entry("RotateAbout".to_string()).or_default() += 1;
            },
            Primitive::Translate(_, _) => {
                *ret.entry("Translate".to_string()).or_default() += 1;
            },
            Primitive::Scale(_, _) => {
                *ret.entry("Scale".to_string()).or_default() += 1;
            },
            Primitive::Repeat(_, _, _) => {
                *ret.entry("Repeat".to_string()).or_default() += 1;
            },
        }
    }
    ret
} */

fn render_plots<'a>(/*canvas: &mut Canvas<'a>,*/
    arena: &'a mut bumpalo::Bump,
    info: CanvasInfo<'a>,
    site: &'a Site,
    render_area: Aabr<i32>,
) {
    for plot in site.plots() {
        let structure: &dyn Structure<_> = match plot.kind() {
            PlotKind::House(house) => house,
            PlotKind::Workshop(workshop) => workshop,
            PlotKind::Castle(castle) => castle,
            PlotKind::Dungeon(dungeon) => dungeon,
            PlotKind::Gnarling(gnarling) => gnarling,
            PlotKind::GiantTree(giant_tree) => giant_tree,
            PlotKind::CliffTower(cliff_tower) => cliff_tower,
            PlotKind::Citadel(citadel) => citadel,
            _ => continue,
        };

        let mut null_canvas = NullCanvas;
        let result = structure.render_collect(site, /*canvas*/arena, info, render_area, /*|_canvas| */&mut null_canvas);
        // NOTE: Clearing out the primitives between renders costs us nothing, because any
        // chunks that get deallocated were going to be eventually deallocated anyway, while
        // the current chunk remains for reuse.  So this just ends up saving memory.
        arena.reset();

        /* //println!("{:?}", count_prim_kinds(&result.0));
        iter_fills(canvas, site, plot, result); */
    }
}

/// Null canvas for avoiding writing to the real cnavas.
struct NullCanvas;
impl Filler for NullCanvas {
    #[inline]
    fn map<F: Fill>(&mut self, pos: Vec3<i32>, f: F) {
        // black_box(pos);
        // black_box(f);
        // black_box(f.sample_at(pos, Block::empty()));
        black_box((pos, if F::NEEDS_BLOCK {
            f.sample_at(pos, black_box(Block::empty()))
        } else /* if F::NEEDS_POS {
            f.sample_at(pos, Block::empty())
        } else */{
            f.sample_at(pos, Block::empty())
        }));
    }

    #[inline]
    fn spawn(&mut self, entity: EntityInfo) {
        black_box(entity);
    }
}

/* fn iter_fills(
    canvas: &CanvasInfo<'_>,
    _site: &Site,
    _plot: &Plot,
    (prim_tree, fills, _): (
        Store<Primitive>,
        Vec<(Id<Primitive>, Fill)>,
        Vec<EntityInfo>,
    ),
) {
    let arena = bumpalo::Bump::new();
    let mut cache = HashMap::default();

    /* let wpos2d = canvas.wpos();
    let chunk_aabr = Aabr {
        min: wpos2d,
        max: wpos2d + TerrainChunkSize::RECT_SIZE.as_::<i32>(),
    }; */

    /* let info = canvas.info(); */

    for (prim, fill) in fills {
        let aabb = Fill::get_bounds(&mut cache, &prim_tree, prim)/*Aabr::new_empty(Vec2::zero())*/;
        // let mut aabb = Fill::get_bounds(&prim_tree, prim);
        /* fill.sample_at(
            &arena,
            &mut cache,
            &prim_tree,
            prim,
            aabb.into(),
            |pos, f| canvas.map(pos, f)
        ); */
        /*Fill::get_bounds_disjoint*/fill.sample_at(&arena, &mut cache, &prim_tree, prim, aabb.into(), canvas, /* |pos| {
            /* canvas.map(pos, |block| {
                let current_block =
                    fill.sample_at(
                        /* &mut bounds_cache,
                        &prim_tree,
                        prim, */
                        pos,
                        &info,
                        block,
                    );
                /* if let (Some(last_block), None) = (last_block, current_block) {
                    spawn(pos, last_block);
                }
                last_block = current_block; */
                current_block.unwrap_or(block)
            }); */
            black_box(/*fill.sample_at(pos, canvas, Block::empty())*/pos);
        }*/&mut NullCanvas);

        /* for /*mut */aabb in Fill::get_bounds_disjoint(&mut cache, &prim_tree, prim) {
            /* aabb.min = Vec2::max(aabb.min.xy(), chunk_aabr.min).with_z(aabb.min.z);
            aabb.max = Vec2::min(aabb.max.xy(), chunk_aabr.max).with_z(aabb.max.z); */

            for x in aabb.min.x..aabb.max.x {
                for y in aabb.min.y..aabb.max.y {
                    for z in aabb.min.z..aabb.max.z {
                        /* let col_tile = site.wpos_tile(Vec2::new(x, y));
                        if
                        /* col_tile.is_building() && */
                        col_tile
                            .plot()
                            .and_then(|p| site.plots().nth(p.id() as usize).unwrap().z_range())
                            .zip(plot.z_range())
                            .map_or(false, |(a, b)| a.end > b.end)
                        {
                            continue;
                        } */
                        let pos = Vec3::new(x, y, z);
                        black_box(fill.sample_at(&mut cache, &prim_tree, prim, pos, canvas, Block::empty()));
                    }
                }
            }
        } */
    }
} */

fn dungeon(c: &mut Criterion) {
    let pool = ThreadPoolBuilder::new().build().unwrap();
    let (world, index) = World::generate(
        /*230*/59686,
        WorldOpts {
            seed_elements: true,
            world_file: FileOpts::LoadAsset(DEFAULT_WORLD_MAP.into()),
            ..WorldOpts::default()
        },
        &pool,
    );
    let wpos = Vec2::zero();
    let seed = [1; 32];

    let air = Block::air(SpriteKind::Empty);
    /* let stone = Block::new(
        BlockKind::Rock,
        zcache_grid
            .get(grid_border + TerrainChunkSize::RECT_SIZE.map(|e| e as i32) / 2)
            .and_then(|zcache| zcache.as_ref())
            .map(|zcache| zcache.sample.stone_col)
            .unwrap_or_else(|| index.colors.deep_stone_color.into()),
    ); */
    let water = Block::new(BlockKind::Water, Rgb::zero());
    let mut chunk = TerrainChunk::new(CONFIG.sea_level as i32, water, air, TerrainChunkMeta::void());

    let find_bounds = |site: &Site| {
        let aabr = site.plots()
            .fold(
                Aabr::new_empty(Vec2::zero()),
                |aabr, plot| {
                    // NOTE: Approx
                    aabr.union(plot.find_bounds())
                },
            );
        let chunks = Vec2::from(aabr.size())
            .map2(TerrainChunkSize::RECT_SIZE, |e: i32, sz| (e as f32 / sz as f32).ceil() as u32);
        // .as_::<i32>();
        (chunks.x as u64 * chunks.y as u64, aabr)
    };

    let mut bench_group = |gen_name, render_name, f: fn(&Land, &mut _, _) -> _| {
        c.bench_function(gen_name, |b| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            b.iter(|| {
                f(&Land::from_sim(world.sim()), &mut rng, wpos);
            });
        });
        CanvasInfo::with_mock_canvas_info(index.as_index_ref(), world.sim(), |&info| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            let site = f(&info.land(), &mut rng, wpos);
            let mut render_dungeon_group = c.benchmark_group(render_name);
            let (throughput, render_area) = find_bounds(&site);
            render_dungeon_group.throughput(criterion::Throughput::Elements(throughput));
            render_dungeon_group.bench_function("identity_allocator", |b| {
                b.iter(|| {
                    let mut arena = bumpalo::Bump::new();
                    /* let mut canvas = Canvas {
                        info,
                        chunk: &mut chunk,
                        arena: &mut arena,
                        entities: Vec::new(),
                    }; */
                    render_plots(/*&mut canvas, */&mut arena, info, &site, render_area);
                });
            });
        });
    };

    bench_group("generate_city", "render_city", Site::generate_city);
    bench_group("generate_cliff_town", "render_cliff_town", Site::generate_cliff_town);
    bench_group("generate_dungeon", "render_dungeon", Site::generate_dungeon);
    bench_group("generate_giant_tree", "render_giant_tree", Site::generate_giant_tree);
    bench_group("generate_gnarling", "render_gnarling", Site::generate_gnarling);
    bench_group("generate_citadel", "render_citadel", Site::generate_citadel);

    c.bench_function("generate_chunk", |b| {
        // let chunk_pos = (world.sim().map_size_lg().chunks() >> 1).as_();
        // let chunk_pos = Vec2::new(9500 / 32, 29042 / 32);
        // let chunk_pos = Vec2::new(26944 / 32, 26848 / 32);
        let chunk_pos = Vec2::new(842, 839);
        // let chunk_pos = Vec2::new(24507/32, 20682/32);
        // let chunk_pos = Vec2::new(19638/32, 19621/32);
        b.iter(|| {
            black_box(world.generate_chunk(index.as_index_ref(), chunk_pos, || false, None));
        });
    });

    c.bench_function("deserialize_chunk", |b| {
        // let chunk_pos = (world.sim().map_size_lg().chunks() >> 1).as_();
        // let chunk_pos = Vec2::new(9500 / 32, 29042 / 32);
        // let chunk_pos = Vec2::new(26944 / 32, 26848 / 32);
        let chunk_pos = Vec2::new(842, 839);
        let chunk = world.generate_chunk(index.as_index_ref(), chunk_pos, || false, None).unwrap().0;
        let serialized = bincode::serialize(&chunk).unwrap();
        // let chunk_pos = Vec2::new(24507/32, 20682/32);
        // let chunk_pos = Vec2::new(19638/32, 19621/32);
        b.iter(|| {
            black_box(bincode::deserialize::<TerrainChunk>(&serialized).unwrap());
        });
    });

    /* c.bench_function("generate_dungeon", |b| {
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        b.iter(|| {
            Site::generate_dungeon(&Land::empty(), &mut rng, wpos);
        });
    });
    {
        CanvasInfo::with_mock_canvas_info(index.as_index_ref(), world.sim(), |&info| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            let site = Site::generate_dungeon(&info.land(), &mut rng, wpos);
            let mut render_dungeon_group = c.benchmark_group("render_dungeon");
            render_dungeon_group.throughput(criterion::Throughput::Elements(find_bounds(&site)));
            render_dungeon_group.bench_function("identity_allocator", |b| {
                b.iter(|| {
                    let mut arena = bumpalo::Bump::new();
                    let mut canvas = Canvas {
                        info,
                        chunk: &mut chunk,
                        arena: &mut arena,
                        entities: Vec::new(),
                    };
                    render_plots(&mut canvas, &site);
                });
            });
        });
    }
    c.bench_function("generate_gnarling", |b| {
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        b.iter(|| {
            Site::generate_gnarling(&Land::empty(), &mut rng, wpos);
        });
    });
    {
        CanvasInfo::with_mock_canvas_info(index.as_index_ref(), world.sim(), |&info| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            let site = Site::generate_gnarling(&info.land(), &mut rng, wpos);
            let mut render_gnarling_group = c.benchmark_group("render_gnarling");
            render_gnarling_group.throughput(criterion::Throughput::Elements(find_bounds(&site)));
            render_gnarling_group.bench_function("identity_allocator", |b| {
                b.iter(|| {
                    let mut arena = bumpalo::Bump::new();
                    let mut canvas = Canvas {
                        info,
                        chunk: &mut chunk,
                        arena: &mut arena,
                        entities: Vec::new(),
                    };
                    render_plots(&mut canvas, &site);
                });
            });
        });
    } */
}

criterion_group!(benches, dungeon);
criterion_main!(benches);
