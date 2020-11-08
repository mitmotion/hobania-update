use criterion::{black_box, criterion_group, criterion_main, Benchmark, Criterion};
use vek::*;
use veloren_world::{sim, World};

// 136 chunks total
const MIN_CHUNK: Vec2<i32> = Vec2 { x: 382, y: 406 };
const MAX_CHUNK: Vec2<i32> = Vec2 { x: 398, y: 413 };

pub fn criterion_benchmark(c: &mut Criterion) {
    // Setup world
    let (world, index) = World::generate(42, sim::WorldOpts {
        // NOTE: If this gets too expensive, we can turn it off.
        // TODO: Consider an option to turn off all erosion as well, or even provide altitude
        // directly with a closure.
        seed_elements: true,
        world_file: sim::FileOpts::LoadAsset(sim::DEFAULT_WORLD_MAP.into()),
        ..Default::default()
    });
    //let index = Box::leak(Box::new(index)).as_index_ref();
    let to_gen = (MIN_CHUNK.x..MAX_CHUNK.x + 1)
        .flat_map(|x| (MIN_CHUNK.y..MAX_CHUNK.y + 1).map(move |y| Vec2::new(x, y)))
        .collect::<Vec<_>>();

    c.bench(
        "chunk_generation",
        Benchmark::new("generating area of forest chunks", move |b| {
            let index = index.as_index_ref();
            b.iter(|| {
                black_box(&to_gen)
                    .iter()
                    .map(|pos| (pos, world.generate_chunk(index, *pos, || false).unwrap()))
                    .collect::<Vec<_>>()
            })
        })
        // Lower sample size to save time
        .sample_size(15),
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
