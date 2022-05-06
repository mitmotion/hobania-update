#![allow(clippy::branches_sharing_code)] // TODO: evaluate
use common::{
    terrain::{
        map::{MapConfig, MapDebug, MapSample},
        uniform_idx_as_vec2, vec2_as_uniform_idx, TerrainChunkSize,
    },
    vol::RectVolSize,
};
use rayon::prelude::*;
use std::{f64, io::Write, path::PathBuf, time::SystemTime};
use tracing::{warn, Level};
use tracing_subscriber::{
    filter::{EnvFilter, LevelFilter},
    FmtSubscriber,
};
use vek::*;
use veloren_world::{
    sim::{self, get_horizon_map, sample_pos, sample_wpos, WorldOpts},
    util::Sampler,
    ColumnSample, World, CONFIG,
};

const W: usize = 1024;
const H: usize = 1024;

fn main() {
    FmtSubscriber::builder()
        .with_max_level(Level::ERROR)
        .with_env_filter(EnvFilter::from_default_env().add_directive(LevelFilter::INFO.into()))
        .init();
    let threadpool = rayon::ThreadPoolBuilder::new().build().unwrap();

    // To load a map file of your choice, replace map_file with the name of your map
    // (stored locally in the map directory of your Veloren root), and swap the
    // sim::FileOpts::Save line below for the sim::FileOpts::Load one.
    let map_file =
        // "map_1575990726223.bin";
        // "map_1575987666972.bin";
        // "map_1576046079066.bin";
        // "maps/071090_2x.bin";
        "map_1579539133272.bin";
    let mut _map_file = PathBuf::from("./maps");
    _map_file.push(map_file);

    let (world, index) = World::generate(
        5284,
        WorldOpts {
            seed_elements: false,
            world_file: sim::FileOpts::LoadAsset(veloren_world::sim::DEFAULT_WORLD_MAP.into()),
            // world_file: sim::FileOpts::Load(_map_file),
            // world_file: sim::FileOpts::Save(sim::SizeOpts::default()),
            // world_file: sim::FileOpts::Save(sim::SizeOpts::new(12, 12, 4.0)),
            calendar: None,
        },
        &threadpool,
    );
    let index = index.as_index_ref();
    tracing::info!("Sampling data...");
    let sampler = world.sim();
    let map_size_lg = sampler.map_size_lg();

    let samples_data = {
        let column_sample = world.sample_columns();
        (0..map_size_lg.chunks_len())
            .into_par_iter()
            .map(|posi| {
                column_sample.get((
                    uniform_idx_as_vec2(map_size_lg, posi)
                        * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
                    index,
                    None,
                ))
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    };
    let refresh_map_samples = |config: &MapConfig, samples: Option<&[Option<ColumnSample>]>| {
        (0..map_size_lg.chunks_len())
            .into_par_iter()
            .map(|posi| {
                sample_pos(
                    config,
                    sampler,
                    samples,
                    uniform_idx_as_vec2(map_size_lg, posi),
                )
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    };
    let get_map_sample = |map_samples: &[MapSample], pos: Vec2<i32>| {
        if pos.reduce_partial_min() >= 0
            && pos.x < map_size_lg.chunks().x as i32
            && pos.y < map_size_lg.chunks().y as i32
        {
            map_samples[vec2_as_uniform_idx(map_size_lg, pos)].clone()
        } else {
            MapSample {
                alt: 0.0,
                rgb: Rgb::new(0, 0, 0),
                connections: None,
                downhill_wpos: (pos + 1) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
            }
        }
    };

    let refresh_horizons = |is_basement, is_water| {
        get_horizon_map(
            map_size_lg,
            Aabr {
                min: Vec2::zero(),
                max: map_size_lg.chunks().map(|e| e as i32),
            },
            CONFIG.sea_level,
            CONFIG.sea_level + sampler.max_height,
            |posi| {
                let sample = sampler.get(uniform_idx_as_vec2(map_size_lg, posi)).unwrap();
                if is_basement {
                    sample.alt
                } else {
                    sample.basement
                }
                .max(if is_water {
                    sample.water_alt
                } else {
                    -f32::INFINITY
                })
            },
            |a| a,
            |h| h,
            /* |[al, ar]| [al, ar],
             * |[hl, hr]| [hl, hr], */
        )
        .ok()
    };

    let mut win =
        minifb::Window::new("World Viewer", W, H, minifb::WindowOptions::default()).unwrap();

    let mut focus = Vec3::new(0.0, 0.0, CONFIG.sea_level as f64);
    // Altitude is divided by gain and clamped to [0, 1]; thus, decreasing gain
    // makes smaller differences in altitude appear larger.
    let mut gain = /*CONFIG.mountain_scale*/sampler.max_height;
    // The Z component during normal calculations is multiplied by gain; thus,
    let mut fov = 1.0;
    let mut scale =
        (map_size_lg.chunks().x as f64 / W as f64).max(map_size_lg.chunks().y as f64 / H as f64);

    // Right-handed coordinate system: light is going left, down, and "backwards"
    // (i.e. on the map, where we translate the y coordinate on the world map to
    // z in the coordinate system, the light comes from -y on the map and points
    // towards +y on the map).  In a right handed coordinate system, the
    // "camera" points towards -z, so positive z is backwards "into" the camera.
    //
    // "In world space the x-axis will be pointing east, the y-axis up and the
    // z-axis will be pointing south"
    let mut light_direction = Vec3::new(-/*0.8*/1.3, -1.0, 0.3);

    let mut is_basement = false;
    let mut is_water = true;
    let mut is_shaded = true;
    let mut is_temperature = true;
    let mut is_humidity = true;

    let mut horizons = refresh_horizons(is_basement, is_water);
    let mut samples = None;

    let mut samples_changed = true;
    let mut map_samples: Box<[_]> = Box::new([]);
    while win.is_open() {
        let config = MapConfig {
            map_size_lg,
            dimensions: Vec2::new(W, H),
            focus,
            gain,
            fov,
            scale,
            light_direction,
            horizons: horizons.as_ref(), /* .map(|(a, b)| (&**a, &**b)) */

            is_basement,
            is_water,
            is_ice: true,
            is_shaded,
            is_temperature,
            is_humidity,
            is_debug: true,
            is_contours: false,
            is_height_map: false,
            is_stylized_topo: false,
        };

        if samples_changed {
            map_samples = refresh_map_samples(&config, samples);
        };

        let mut buf = vec![0; W * H];
        let MapDebug {
            rivers,
            lakes,
            oceans,
            quads,
        } = config.generate(
            |pos| get_map_sample(&map_samples, pos),
            |pos| sample_wpos(&config, sampler, pos),
            |pos, (r, g, b, a)| {
                let i = pos.x;
                let j = pos.y;
                buf[j * W + i] = u32::from_le_bytes([b, g, r, a]);
            },
        );

        if win.is_key_down(minifb::Key::F4) {
            // Feedback is important since on large maps it can be hard to tell if the
            // keypress registered or not.
            println!("Taking screenshot...");
            if let Some(len) = (W * H)
                .checked_mul(scale as usize)
                .and_then(|acc| acc.checked_mul(scale as usize))
            {
                let x = (W as f64 * scale) as usize;
                let y = (H as f64 * scale) as usize;
                let config = MapConfig {
                    dimensions: Vec2::new(x, y),
                    scale: 1.0,
                    ..config
                };
                let mut buf = vec![0u8; 4 * len];
                config.generate(
                    |pos| get_map_sample(&map_samples, pos),
                    |pos| sample_wpos(&config, sampler, pos),
                    |pos, (r, g, b, a)| {
                        let i = pos.x;
                        let j = pos.y;
                        (&mut buf[(j * x + i) * 4..])
                            .write_all(&[r, g, b, a])
                            .unwrap();
                    },
                );
                // TODO: Justify fits in u32.
                let world_map = image::RgbaImage::from_raw(x as u32, y as u32, buf)
                    .expect("Image dimensions must be valid");
                let mut path = PathBuf::from("./screenshots");
                if !path.exists() {
                    if let Err(e) = std::fs::create_dir(&path) {
                        warn!(?e, ?path, "Couldn't create folder for screenshot");
                    }
                }
                path.push(format!(
                    "worldmap_{}.png",
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                ));
                if let Err(e) = world_map.save(&path) {
                    warn!(?e, ?path, "Couldn't save screenshot");
                }
            }
        }

        let spd = 32.0;
        let lspd = 0.1;
        if win.is_key_down(minifb::Key::P) {
            println!(
                "\
                 Gain / Shade gain: {:?} / {:?}\nScale / Focus: {:?} / {:?}\nLight: {:?}
                 Land(adjacent): (X = temp, Y = humidity): {:?}\nRivers: {:?}\nLakes: \
                 {:?}\nOceans: {:?}\nTotal water: {:?}\nTotal land(adjacent): {:?}",
                gain,
                fov,
                scale,
                focus,
                light_direction,
                quads,
                rivers,
                lakes,
                oceans,
                rivers + lakes + oceans,
                quads.iter().map(|x| x.iter().sum::<u32>()).sum::<u32>()
            );
        }
        if win.get_mouse_down(minifb::MouseButton::Left) {
            if let Some((mx, my)) = win.get_mouse_pos(minifb::MouseMode::Clamp) {
                let chunk_pos = (Vec2::<f64>::from(focus)
                    + (Vec2::new(mx as f64, my as f64) * scale))
                    .map(|e| e as i32);
                let block_pos = chunk_pos.map2(TerrainChunkSize::RECT_SIZE, |e, f| e * f as i32);
                println!(
                    "Block: ({}, {}), Chunk: ({}, {})",
                    block_pos.x, block_pos.y, chunk_pos.x, chunk_pos.y
                );
                if let Some(chunk) = sampler.get(chunk_pos) {
                    //println!("Chunk info: {:#?}", chunk);
                    if let Some(id) = &chunk.place {
                        let place = world.civs().place(*id);
                        println!("Place {} info: {:#?}", id.id(), place);

                        if let Some(site) = world.civs().sites().find(|site| site.place == *id) {
                            println!("Site: {}", site);
                        }
                    }
                }
            }
        }
        let is_camera = win.is_key_down(minifb::Key::C);
        if win.is_key_down(minifb::Key::B) {
            is_basement ^= true;
            samples_changed = true;
            horizons = horizons.and_then(|_| refresh_horizons(is_basement, is_water));
        }
        if win.is_key_down(minifb::Key::H) {
            is_humidity ^= true;
            samples_changed = true;
        }
        if win.is_key_down(minifb::Key::T) {
            is_temperature ^= true;
            samples_changed = true;
        }
        if win.is_key_down(minifb::Key::O) {
            is_water ^= true;
            samples_changed = true;
            horizons = horizons.and_then(|_| refresh_horizons(is_basement, is_water));
        }
        if win.is_key_down(minifb::Key::L) {
            if is_camera {
                // TODO: implement removing horizon mapping.
                horizons = if horizons.is_some() {
                    None
                } else {
                    refresh_horizons(is_basement, is_water)
                };
                samples_changed = true;
            } else {
                is_shaded ^= true;
                samples_changed = true;
            }
        }
        if win.is_key_down(minifb::Key::M) {
            samples = samples.xor(Some(&*samples_data));
            samples_changed = true;
        }
        if win.is_key_down(minifb::Key::W) {
            if is_camera {
                light_direction.z -= lspd;
            } else {
                focus.y -= spd * scale;
            }
        }
        if win.is_key_down(minifb::Key::A) {
            if is_camera {
                light_direction.x -= lspd;
            } else {
                focus.x -= spd * scale;
            }
        }
        if win.is_key_down(minifb::Key::S) {
            if is_camera {
                light_direction.z += lspd;
            } else {
                focus.y += spd * scale;
            }
        }
        if win.is_key_down(minifb::Key::D) {
            if is_camera {
                light_direction.x += lspd;
            } else {
                focus.x += spd * scale;
            }
        }
        if win.is_key_down(minifb::Key::Q) {
            if is_camera {
                if (fov * 2.0).is_normal() {
                    fov *= 2.0;
                }
            } else {
                gain += 64.0;
            }
        }
        if win.is_key_down(minifb::Key::E) {
            if is_camera {
                if (fov / 2.0).is_normal() {
                    fov /= 2.0;
                }
            } else {
                gain = (gain - 64.0).max(64.0);
            }
        }
        if win.is_key_down(minifb::Key::R) {
            if is_camera {
                focus.z += spd * scale;
                samples_changed = true;
            } else if (scale * 2.0).is_normal() {
                scale *= 2.0;
            }
        }
        if win.is_key_down(minifb::Key::F) {
            if is_camera {
                focus.z -= spd * scale;
                samples_changed = true;
            } else if (scale / 2.0).is_normal() {
                scale /= 2.0;
            }
        }

        win.update_with_buffer(&buf, W, H).unwrap();
    }
}
