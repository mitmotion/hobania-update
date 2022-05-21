use common::{
    generation::EntityInfo,
    store::{Id, Store},
    terrain::Block,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashbrown::HashMap;
use rand::prelude::*;
use rayon::ThreadPoolBuilder;
use vek::{Vec2, Vec3};
use veloren_world::{
    sim::{FileOpts, WorldOpts, DEFAULT_WORLD_MAP},
    site2::{
        plot::PlotKind, Fill, 
        Primitive, Site, Structure,
    },
    CanvasInfo, Land, World,
};

#[allow(dead_code)]
fn count_prim_kinds(prims: &Store<Primitive>) -> HashMap<String, usize> {
    let mut ret = HashMap::new();
    for prim in prims.values() {
        match &prim {
            Primitive::Empty => { *ret.entry("Empty".to_string()).or_default() += 1; }
            Primitive::Aabb(_) => { *ret.entry("Aabb".to_string()).or_default() += 1; }
            Primitive::Pyramid { .. } => { *ret.entry("Pyramid".to_string()).or_default() += 1; }
            Primitive::Ramp { .. } => { *ret.entry("Ramp".to_string()).or_default() += 1; },
            Primitive::Gable { .. } => { *ret.entry("Gable".to_string()).or_default() += 1; },
            Primitive::Cylinder(_) => { *ret.entry("Cylinder".to_string()).or_default() += 1; },
            Primitive::Cone(_) => { *ret.entry("Cone".to_string()).or_default() += 1; },
            Primitive::Sphere(_) => { *ret.entry("Sphere".to_string()).or_default() += 1; },
            Primitive::Superquadric { .. } => { *ret.entry("Superquadratic".to_string()).or_default() += 1; },
            Primitive::Plane(_, _, _) => { *ret.entry("Plane".to_string()).or_default() += 1; },
            Primitive::Segment { .. } => { *ret.entry("Segment".to_string()).or_default() += 1; },
            Primitive::SegmentPrism { .. } => { *ret.entry("SegmentPrism".to_string()).or_default() += 1; },
            Primitive::Sampling(_, _) => { *ret.entry("Sampling".to_string()).or_default() += 1; },
            Primitive::Prefab(_) => { *ret.entry("Prefab".to_string()).or_default() += 1; },
            Primitive::Intersect(_, _) => { *ret.entry("Intersect".to_string()).or_default() += 1; },
            Primitive::Union(_, _) => { *ret.entry("Union".to_string()).or_default() += 1; },
            Primitive::Without(_, _) => { *ret.entry("Without".to_string()).or_default() += 1; },
            Primitive::RotateAbout(_, _, _) => { *ret.entry("RotateAbout".to_string()).or_default() += 1; },
            Primitive::Translate(_, _) => { *ret.entry("Translate".to_string()).or_default() += 1; },
            Primitive::Scale(_, _) => { *ret.entry("Scale".to_string()).or_default() += 1; },
            Primitive::Repeat(_, _, _) => { *ret.entry("Repeat".to_string()).or_default() += 1; },
        }
    }
    ret
}

fn render_plots(canvas: &CanvasInfo<'_>, site: &Site) {
    for plot in site.plots() {
        let result = match &plot.kind() {
            PlotKind::House(house) => house.render_collect(site, canvas),
            PlotKind::Workshop(workshop) => workshop.render_collect(site, canvas),
            PlotKind::Castle(castle) => castle.render_collect(site, canvas),
            PlotKind::Dungeon(dungeon) => dungeon.render_collect(site, canvas),
            PlotKind::Gnarling(gnarling) => gnarling.render_collect(site, canvas),
            PlotKind::GiantTree(giant_tree) => giant_tree.render_collect(site, canvas),
            _ => continue,
        };
        //println!("{:?}", count_prim_kinds(&result.0));
        iter_fills(canvas, result);
    }
}

fn iter_fills(
    canvas: &CanvasInfo<'_>,
    (prim_tree, fills, _): (Store<Primitive>, Vec<(Id<Primitive>, Fill)>, Vec<EntityInfo>),
) {
    for (prim, fill) in fills {
        let aabb = Fill::get_bounds(&prim_tree, prim);

        for x in aabb.min.x..aabb.max.x {
            for y in aabb.min.y..aabb.max.y {
                for z in aabb.min.z..aabb.max.z {
                    let pos = Vec3::new(x, y, z);
                    black_box(fill.sample_at(
                        &prim_tree,
                        prim,
                        pos,
                        canvas,
                        Block::empty(),
                    ));
                }
            }
        }
    }
}

fn dungeon(c: &mut Criterion) {
    let pool = ThreadPoolBuilder::new().build().unwrap();
    let (world, index) = World::generate(
        230,
        WorldOpts {
            seed_elements: true,
            world_file: FileOpts::LoadAsset(DEFAULT_WORLD_MAP.into()),
            ..WorldOpts::default()
        },
        &pool,
    );
    let wpos = Vec2::zero();
    let seed = [1; 32];
    c.bench_function("generate_dungeon", |b| {
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        b.iter(|| {
            Site::generate_dungeon(&Land::empty(), &mut rng, wpos);
        });
    });
    {
        let mut render_dungeon_group = c.benchmark_group("render_dungeon");
        render_dungeon_group.bench_function("identity_allocator", |b| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            CanvasInfo::with_mock_canvas_info(index.as_index_ref(), world.sim(), |canvas| {
                let site = Site::generate_dungeon(&canvas.land(), &mut rng, wpos);
                b.iter(|| {
                    render_plots(canvas, &site);
                });
            })
        });
    }
    c.bench_function("generate_gnarling", |b| {
        let mut rng = rand::rngs::StdRng::from_seed(seed);
        b.iter(|| {
            Site::generate_gnarling(&Land::empty(), &mut rng, wpos);
        });
    });
    {
        let mut render_gnarling_group = c.benchmark_group("render_gnarling");
        render_gnarling_group.bench_function("identity_allocator", |b| {
            let mut rng = rand::rngs::StdRng::from_seed(seed);
            CanvasInfo::with_mock_canvas_info(index.as_index_ref(), world.sim(), |canvas| {
                let site = Site::generate_gnarling(&canvas.land(), &mut rng, wpos);
                b.iter(|| {
                    render_plots(canvas, &site);
                });
            })
        });
    }
}

criterion_group!(benches, dungeon);
criterion_main!(benches);
