use std::ops::{Add, Mul, Sub};
use vek::*;
use veloren_world::{sim::WorldOpts, util::Sampler, World};

const W: usize = 640;
const H: usize = 480;

fn main() {
    let threadpool = rayon::ThreadPoolBuilder::new().build().unwrap();
    let (world, index) = World::generate(
        0,
        WorldOpts {
            seed_elements: true,
            ..WorldOpts::default()
        },
        &threadpool,
    );

    let index = index.as_index_ref();

    let mut win =
        minifb::Window::new("World Viewer", W, H, minifb::WindowOptions::default()).unwrap();

    let mut focus = Vec2::zero();
    let mut gain = 1.0;
    let mut scale = 4;

    while win.is_open() {
        let mut buf = vec![0; W * H];

        for i in 0..W {
            for j in 0..H {
                let pos = focus + Vec2::new(i as i32, j as i32) * scale;

                let sampler = world.sample_blocks(pos >> common::terrain::TERRAIN_CHUNK_BLOCKS_LG as i32, index);
                let (alt, place) = sampler
                    .map(|sampler| {
                        let sample = sampler.column_gen.get(pos);
                        (
                            sample.alt.sub(64.0).add(gain).mul(0.7).max(0.0).min(255.0) as u8,
                            sampler.column_gen.sim_chunk.place,
                        )
                    })
                    .unwrap_or((0, None));

                let place_color = place
                    .map(|p| ((p.id() % 256) as u8 * 17, (p.id() % 256) as u8 * 13))
                    .unwrap_or((0, 0));

                buf[j * W + i] = u32::from_le_bytes([place_color.0, place_color.1, alt, alt]);
            }
        }

        let spd = 32;
        if win.is_key_down(minifb::Key::W) {
            focus.y -= spd * scale;
        }
        if win.is_key_down(minifb::Key::A) {
            focus.x -= spd * scale;
        }
        if win.is_key_down(minifb::Key::S) {
            focus.y += spd * scale;
        }
        if win.is_key_down(minifb::Key::D) {
            focus.x += spd * scale;
        }
        if win.is_key_down(minifb::Key::Q) {
            gain += 10.0;
        }
        if win.is_key_down(minifb::Key::E) {
            gain -= 10.0;
        }
        if win.is_key_down(minifb::Key::R) {
            scale += 6;
        }
        if win.is_key_down(minifb::Key::F) {
            scale -= 6;
        }

        win.update_with_buffer(&buf, W, H).unwrap();
    }
}
