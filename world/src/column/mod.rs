use crate::{
    all::ForestKind,
    block::StructureMeta,
    generator::{Generator, SpawnRules, TownGen},
    sim::{
        local_cells, neighbors, quadratic_nearest_point, river_spline_coeffs, uniform_idx_as_vec2,
        vec2_as_uniform_idx, LocationInfo, RiverKind, SimChunk, WorldSim,
    },
    util::{RandomPerm, Sampler, UnitChooser},
    CONFIG,
};
use common::{
    assets,
    terrain::{BlockKind, Structure, TerrainChunkSize},
    util::f32_to_f7,
    vol::RectVolSize,
};
use lazy_static::lazy_static;
use noise::NoiseFn;
use std::{
    cmp::Reverse,
    f32, f64, iter,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};
use vek::*;

pub struct ColumnGen<'a> {
    pub sim: &'a WorldSim,
}

static UNIT_CHOOSER: UnitChooser = UnitChooser::new(0x700F4EC7);
static DUNGEON_RAND: RandomPerm = RandomPerm::new(0x42782335);

lazy_static! {
    pub static ref DUNGEONS: Vec<Arc<Structure>> = vec![
        assets::load_map("world.structure.dungeon.ruins", |s: Structure| s
            .with_center(Vec3::new(57, 58, 61))
            .with_default_kind(BlockKind::Dense))
        .unwrap(),
        assets::load_map("world.structure.dungeon.ruins_2", |s: Structure| s
            .with_center(Vec3::new(53, 57, 60))
            .with_default_kind(BlockKind::Dense))
        .unwrap(),
        assets::load_map("world.structure.dungeon.ruins_3", |s: Structure| s
            .with_center(Vec3::new(58, 45, 72))
            .with_default_kind(BlockKind::Dense))
        .unwrap(),
        assets::load_map(
            "world.structure.dungeon.meso_sewer_temple",
            |s: Structure| s
                .with_center(Vec3::new(63, 62, 60))
                .with_default_kind(BlockKind::Dense)
        )
        .unwrap(),
        assets::load_map("world.structure.dungeon.ruins_maze", |s: Structure| s
            .with_center(Vec3::new(60, 60, 116))
            .with_default_kind(BlockKind::Dense))
        .unwrap(),
    ];
}

impl<'a> ColumnGen<'a> {
    pub fn new(sim: &'a WorldSim) -> Self { Self { sim } }

    fn get_local_structure(&self, wpos: Vec2<i32>) -> Option<StructureData> {
        let (pos, seed) = self
            .sim
            .gen_ctx
            .region_gen
            .get(wpos)
            .iter()
            .copied()
            .min_by_key(|(pos, _)| pos.distance_squared(wpos))
            .unwrap();

        let chunk_pos = pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);
        let chunk = self.sim.get(chunk_pos)?;

        if seed % 5 == 2
            && chunk.temp > CONFIG.desert_temp
            && chunk.alt > chunk.water_alt + 5.0
            && chunk.chaos <= 0.35
        {
            /*Some(StructureData {
                pos,
                seed,
                meta: Some(StructureMeta::Pyramid { height: 140 }),
            })*/
            None
        } else if seed % 17 == 2 && chunk.chaos < 0.2 {
            Some(StructureData {
                pos,
                seed,
                meta: Some(StructureMeta::Volume {
                    units: UNIT_CHOOSER.get(seed),
                    volume: &DUNGEONS[DUNGEON_RAND.get(seed) as usize % DUNGEONS.len()],
                }),
            })
        } else {
            None
        }
    }

    fn gen_close_structures(&self, wpos: Vec2<i32>) -> [Option<StructureData>; 9] {
        let mut metas = [None; 9];
        self.sim
            .gen_ctx
            .structure_gen
            .get(wpos)
            .iter()
            .copied()
            .enumerate()
            .for_each(|(i, (pos, seed))| {
                metas[i] = self.get_local_structure(pos).or(Some(StructureData {
                    pos,
                    seed,
                    meta: None,
                }));
            });
        metas
    }
}

/* /// Find the arc length along a quadratic curve from t = 0 to t = 1.
fn quadratic_arc_length(spline: &Vec3<Vec2<f64>>) -> f64 {
    // let a = 4 * (A_x^2 + A_y^2)
    // let b = 4 * (A_x * B_x + A_y * B_y)
    // let c = (B_x^2 + B_y^2)
    //
    // Then
    //   L_B = integral{0 to t}(sqrt(at^2 + bt + c)dt)
    //
    // This has a closed-form solution...
    //
    // Suppose a = 0...
    //
    // Then A_x = 0 and A_y = 0, which means that b = 0 as well.  So you have
    //
    //   L_B = integral{0 to 1}(sqrt(B_x^2 + B_y^2)dt)
    //
    // which means
    //
    //   L_B = sqrt(B_x^2 + B_y^2) t + constant
    //       = sqrt(B_x^2 + B_y^2)
    //
    // Otherwise, a ≠ 0 (thus b ≠ 0 either).
    // Then
    //   let u = b^2 / (4 * a)
    //   let k = (4 * a * c - b^2) / (4 * a)
    //   let x = a^(1/2) * t + u^(1/2)
    //
    //   Evaluating x at t = 0 and t = 1...
    //   x at t=0 = u^(1/2) = b^2 / (4 * a)
    //   x at t=1 = a^(1/2) + u^(1/2) = a^(1/2) + b^2 / (4 * a)
    //
    //   L_B = 1/2 * (x * (x^2 + k)^(1/2) + k * ln(abs(x + (x^2 + k)^(1/2)))) + const
    let a = spline.x.magnitude_squared() * 4.0;
    if a == 0.0 {
        // No squared component, so we are linear and can just use the straight-line distance.
        return spline.y.magnitude();
    }
    let b = (spline.x.x * spline.y.x + spline.x.y * spline.y.y) * 4.0;
    let c = spline.z.magnitude_squared();
    let u = b * b / (a * 4.0);
    let k = (a * c * 4.0 - b * b) / (a * 4.0);
    let x = |t: f64| a.sqrt() * t + u.sqrt();
    let l_b = |x: f64| (x * (x * x + k).sqrt() + k * ((x + (x * x + k).sqrt()).ln())) * 0.5;
    l_b(x(1.0)) - l_b(x(0.0))
} */

impl<'a> Sampler<'a> for ColumnGen<'a> {
    type Index = Vec2<i32>;
    type Sample = Option<ColumnSample<'a>>;

    fn get(&self, wpos: Vec2<i32>) -> Option<ColumnSample<'a>> {
        let wposf = wpos.map(|e| e as f64);
        let chunk_pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);

        let sim = &self.sim;

        let turb = Vec2::new(
            sim.gen_ctx.turb_x_nz.get((wposf.div(48.0)).into_array()) as f32,
            sim.gen_ctx.turb_y_nz.get((wposf.div(48.0)).into_array()) as f32,
        ) * 12.0;
        let wposf_turb = wposf + turb.map(|e| e as f64);

        let chaos = sim.get_interpolated(wpos, |chunk| chunk.chaos)?;
        let temp = sim.get_interpolated(wpos, |chunk| chunk.temp)?;
        let humidity = sim.get_interpolated(wpos, |chunk| chunk.humidity)?;
        let rockiness = sim.get_interpolated(wpos, |chunk| chunk.rockiness)?;
        let tree_density = sim.get_interpolated(wpos, |chunk| chunk.tree_density)?;
        let spawn_rate = sim.get_interpolated(wpos, |chunk| chunk.spawn_rate)?;
        let alt = sim.get_interpolated_monotone(wpos, |chunk| chunk.alt)?;
        let sim_chunk = sim.get(chunk_pos)?;
        let neighbor_coef = TerrainChunkSize::RECT_SIZE.map(|e| e as f64);
        let my_chunk_idx = vec2_as_uniform_idx(chunk_pos);
        let neighbor_river_data = local_cells(my_chunk_idx).filter_map(|neighbor_idx: usize| {
            let neighbor_pos = uniform_idx_as_vec2(neighbor_idx);
            let neighbor_chunk = sim.get(neighbor_pos)?;
            Some((neighbor_pos, neighbor_chunk, &neighbor_chunk.river))
        });
        let lake_width = (TerrainChunkSize::RECT_SIZE.x as f64) * 1.0 + 1.0; // * (2.0f64.sqrt()) + 0.0/* + 12.0*/;
        let neighbor_river_data = neighbor_river_data.map(|(posj, chunkj, river)| {
            let kind = match river.river_kind {
                Some(kind) => kind,
                None => {
                    return (posj, chunkj, river, None);
                },
            };
            let downhill_pos = if let Some(pos) = chunkj.downhill {
                pos
            } else {
                match kind {
                    RiverKind::River { .. } => {
                        log::error!("What? River: {:?}, Pos: {:?}", river, posj);
                        panic!("How can a river have no downhill?");
                    },
                    RiverKind::Lake { .. } => {
                        return (posj, chunkj, river, None);
                    },
                    RiverKind::Ocean => posj,
                }
            };
            let downhill_wpos = downhill_pos.map(|e| e as f64);
            let downhill_pos =
                downhill_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);
            let neighbor_pos = posj.map(|e| e as f64) * neighbor_coef;
            let direction = neighbor_pos - downhill_wpos;
            let river_width_min = if let RiverKind::River { cross_section } = kind {
                cross_section.x as f64
            } else {
                lake_width
            };
            let downhill_chunk = sim.get(downhill_pos).expect("How can this not work?");
            let coeffs =
                river_spline_coeffs(neighbor_pos, chunkj.river.spline_derivative, downhill_wpos);
            let (direction, coeffs, downhill_chunk, river_t, river_pos, river_dist) = match kind {
                RiverKind::River { .. } => {
                    if let Some((t, pt, dist)) = quadratic_nearest_point(&coeffs, wposf) {
                        (true, coeffs, downhill_chunk, t, pt, dist.sqrt())
                    } else {
                        let ndist = wposf.distance_squared(neighbor_pos);
                        let ddist = wposf.distance_squared(downhill_wpos);
                        let (closest_pos, closest_dist, closest_t) = if ndist <= ddist {
                            (neighbor_pos, ndist, 0.0)
                        } else {
                            (downhill_wpos, ddist, 1.0)
                        };
                        (
                            false,
                            coeffs,
                            downhill_chunk,
                            closest_t,
                            closest_pos,
                            closest_dist.sqrt(),
                        )
                    }
                },
                RiverKind::Lake { neighbor_pass_pos } => {
                    let pass_dist = neighbor_pass_pos
                        .map2(
                            neighbor_pos
                                .map2(TerrainChunkSize::RECT_SIZE, |f, g| (f as i32, g as i32)),
                            |e, (f, g)| ((e - f) / g).abs(),
                        )
                        .reduce_partial_max();
                    let spline_derivative = river.spline_derivative;
                    let neighbor_pass_pos = if pass_dist <= 1 {
                        neighbor_pass_pos
                    } else {
                        downhill_wpos.map(|e| e as i32)
                    };
                    let pass_dist = neighbor_pass_pos
                        .map2(
                            neighbor_pos
                                .map2(TerrainChunkSize::RECT_SIZE, |f, g| (f as i32, g as i32)),
                            |e, (f, g)| ((e - f) / g).abs(),
                        )
                        .reduce_partial_max();
                    if pass_dist > 1 {
                        return (posj, chunkj, river, None);
                    }
                    let neighbor_pass_wpos = neighbor_pass_pos.map(|e| e as f64);
                    let neighbor_pass_pos = neighbor_pass_pos
                        .map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);
                    let coeffs =
                        river_spline_coeffs(neighbor_pos, spline_derivative, neighbor_pass_wpos);
                    let direction = neighbor_pos - neighbor_pass_wpos;
                    if let Some((t, pt, dist)) = quadratic_nearest_point(&coeffs, wposf) {
                        (
                            true,
                            coeffs,
                            sim.get(neighbor_pass_pos).expect("Must already work"),
                            t,
                            pt,
                            dist.sqrt(),
                        )
                    } else {
                        let ndist = wposf.distance_squared(neighbor_pos);
                        /* let ddist = wposf.distance_squared(neighbor_pass_wpos); */
                        let (closest_pos, closest_dist, closest_t) = /*if ndist <= ddist */ {
                                (neighbor_pos, ndist, 0.0)
                            } /* else {
                                (neighbor_pass_wpos, ddist, 1.0)
                            } */;
                        (
                            false,
                            coeffs,
                            sim.get(neighbor_pass_pos).expect("Must already work"),
                            closest_t,
                            closest_pos,
                            closest_dist.sqrt(),
                        )
                    }
                },
                RiverKind::Ocean => {
                    let ndist = wposf.distance_squared(neighbor_pos);
                    let (closest_pos, closest_dist, closest_t) = (neighbor_pos, ndist, 0.0);
                    (
                        false,
                        coeffs,
                        sim.get(closest_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
                            e as i32 / sz as i32
                        }))
                        .expect("Must already work"),
                        closest_t,
                        closest_pos,
                        closest_dist.sqrt(),
                    )
                },
            };
            let river_width_max =
                if let Some(RiverKind::River { cross_section }) = downhill_chunk.river.river_kind {
                    cross_section.x as f64
                } else {
                    lake_width
                };
            let river_width_noise = (sim.gen_ctx.small_nz.get((river_pos.div(16.0)).into_array()))
                .max(-1.0)
                .min(1.0)
                .mul(0.5)
                .sub(0.5) as f64;
            let river_width = Lerp::lerp(
                river_width_min,
                river_width_max,
                river_t.max(0.0).min(1.0).powf(0.5),
            );
            /* let river_width = if (river_t < 0.0 || river_t > 1.0) && river_pos.dist() {
                0.0
            }; */

            let river_width = river_width * (1.0 + river_width_noise * 0.3);
            // To find the distance, we just evaluate the quadratic equation at river_t and
            // see if it's within width (but we should be able to use it for a
            // lot more, and this probably isn't the very best approach anyway
            // since it will bleed out). let river_pos = coeffs.x * river_t *
            // river_t + coeffs.y * river_t + coeffs.z;
            let res = Vec2::new(0.0, (river_dist - (river_width * 0.5).max(1.0)).max(0.0));
            (
                posj,
                chunkj,
                river,
                Some((
                    direction,
                    res,
                    river_width,
                    (river_t, (river_pos, coeffs), downhill_chunk),
                )),
            )
        });

        // Find the average distance to each neighboring body of water.
        let mut real_river_count = 0.0f64;
        // let mut alt_for_lake = f64::NEG_INFINITY;
        let mut lake_dist = 1.0f64;
        let mut river_count = 0.0f64;
        let mut overlap_count = 0.0f64;
        let mut river_distance_product = 1.0f64;
        let mut river_overlap_distance_product = 0.0f64;
        let mut max_river = None;
        let mut max_key = None;
        let mut avg_river_velocity = Vec3::zero();
        let river_gouge = 0.5f32/*1.0*/;
        // IDEA:
        // For every "nearby" chunk, check whether it is a river.  If so, find the
        // closest point on the river segment to wposf (if two point are
        // equidistant, choose the earlier one), calling this point river_pos
        // and the length (from 0 to 1) along the river segment for the nearby
        // chunk river_t.  Let river_dist be the distance from river_pos to wposf.
        //
        // Let river_alt be the interpolated river height at this point
        // (from the alt/water altitude at the river, to the alt/water_altitude of the
        // downhill river, increasing with river_t).
        //
        // Now, if river_dist is <= river_width * 0.5, then we don't care what altitude
        // we use, and mark that we are on a river (we decide what river to use
        // using a heuristic, and set the solely according to the computed
        // river_alt for that point).
        //
        // Otherwise, we let dist = river_dist - river_width * 0.5.
        //
        // If dist >= TerrainChunkSize::RECT_SIZE.x, we don't include this river in the
        // calculation of the correct altitude for this point.
        //
        // Otherwise (i.e. dist < TerrainChunkSize::RECT_SIZE.x), we want to bias the
        // altitude of this point towards the altitude of the river.
        // Specifically, as the dist goes from TerrainChunkSize::RECT_SIZE.x to
        // 0, the weighted altitude of this point should go from
        // alt to river_alt.
        neighbor_river_data.for_each(|(river_chunk_idx, river_chunk, river, dist)| {
            match river.river_kind {
                Some(kind) => {
                    if kind.is_river() && !dist.is_some() {
                        // Ostensibly near a river segment, but not "usefully" so (there is no
                        // closest point between t = 0.0 and t = 1.0).
                        return;
                    } else {
                        let river_dist = dist.map(|(has_flow, dist, river_width, (river_t, (river_pos, coeffs), downhill_river))| {
                            // let downhill
                            let (dist_y, downhill_height) = if let RiverKind::River { cross_section } = kind {
                                if dist.y == 0.0 && /*(river_t >= 0.0 && river_t <= 1.0 || river_wpos.distance_squared(river_pos) < 0.5 || downhill_river_wpos.distance_squared(river_pos) < 0.5)*/has_flow {
                                    let river_segment_length = 1.0;
                                    let downhill_river_velocity = if downhill_river.river.is_river() {
                                        downhill_river.river.velocity
                                    } else {
                                        river.velocity
                                    };

                                    let velocity_magnitude = Lerp::lerp(river.velocity, downhill_river_velocity, river_t as f32) as f64;
                                    // let river_wpos = river_chunk_idx.map(|e| e as f64) * neighbor_coef;
                                    let downhill_river_wpos = river_chunk.downhill.expect("River must have a downhill").map(|e| e as f64);
                                    // let downhill_river_wpos = downhill_river_pos.map(|e| e as f64) * neighbor_coef;
                                    /* let downhill2_river_wpos = if let Some(downhill2_pos) = downhill_river.downhill {
                                        downhill2_pos.map(|e| e as f64)
                                    } else {
                                        downhill_river_wpos
                                    };
                                    let downhill_coeffs =
                                        river_spline_coeffs(downhill_river_wpos, downhill_river.river.spline_derivative, downhill2_river_wpos); */

                                    let river_slope = /*if river_t >= 1.0 || downhill_river_wpos == wposf {
                                        downhill_river.river.spline_derivative.map(|e| e as f64)
                                    } else {
                                        */coeffs.x * 2.0 * (river_t.max(0.0).min(1.0)) + coeffs.y
                                    /*}*/;
                                    let downhill_slope = downhill_river.river.spline_derivative.map(|e| e as f64);
                                    /* let slope = Lerp::lerp(river_slope, downhill_slope, river_t); */
                                    // let slope = coeffs.x * 2.0 * river_t.max(0.0).min(1.0) + Lerp::lerp(coeffs.y, downhill_slope, river_t);
                                    let slope = /*if river_pos.distance_squared(downhill_river_wpos) < 0.5 {
                                        downhill_slope
                                    } else {
                                        river_slope
                                    }/*river_slope*/*/river_slope;
                                    let dz = (downhill_river.alt.max(downhill_river.water_alt) - river_chunk.alt.max(river_chunk.water_alt)) as f64 / river_segment_length;
                                    let mut velocity = Vec3::new(slope.x, slope.y, dz);
                                    if velocity != Vec3::zero() {
                                        velocity.normalize();
                                    }
                                    velocity *= velocity_magnitude;
                                    /* let river_dist = wposf.distance(river_pos);
                                    let river_height_factor = river_dist / (river_width * 0.5);
                                    let height_factor = Lerp::lerp(cross_section.y as f64, 0.0, river_height_factor * river_height_factor); */
                                    // let height_factor = cross_section.y as f64;
                                    let height_factor = river_chunk.flux as f64;
                                    avg_river_velocity += velocity * height_factor;
                                    real_river_count += height_factor;
                                }
                                (dist.y, Lerp::lerp(
                                    river_chunk.alt.max(river_chunk.water_alt),
                                    downhill_river.alt.max(downhill_river.water_alt),
                                    river_t as f32,
                                ) as f64 - river_gouge as f64)
                            } else {
                                let neighbor_pos =
                                    river_chunk_idx.map(|e| e as f64) * neighbor_coef;
                                let dist_ = (wposf - neighbor_pos).map(|e| e.abs()).reduce_partial_max();
                                let dist_upon =
                                    (dist_ - TerrainChunkSize::RECT_SIZE.x as f64 * 0.5).max(0.0);
                                (dist.y.min(dist_upon), if /*dist.y == 0.0*/dist_upon == 0.0 {
                                    /* -(wposf - neighbor_pos).magnitude() *//*-dist*/
                                    /*f64::INFINITY*/river_chunk.water_alt as f64
                                } else {
                                    /* -(wposf - neighbor_pos).magnitude() *//*-dist*/river_chunk.water_alt as f64
                                })
                                // river_chunk.water_alt
                            };
                            (Reverse((dist.x, dist_y/*, river_t*/)), has_flow, /*Reverse(river_t), */downhill_height)
                        });
                        let river_dist = river_dist.or_else(|| {
                            if !kind.is_river() {
                                let neighbor_pos =
                                    river_chunk_idx.map(|e| e as f64) * neighbor_coef;
                                let dist =
                                    (wposf - neighbor_pos).map(|e| e.abs()).reduce_partial_max();
                                // let dist = (wposf - neighbor_pos).magnitude();
                                let dist_upon =
                                    (dist - TerrainChunkSize::RECT_SIZE.x as f64 * 0.5).max(0.0);
                                let dist_ = river_chunk.water_alt as f64;
                                // let dist_ = if dist_upon == 0.0 { f64::INFINITY } else { -dist };/*river_chunk.water_alt as f64;*/
                                Some((
                                    Reverse((0.0, dist_upon /*, -f64::INFINITY*/)),
                                    true,
                                    /*Reverse(-f64::INFINITY), */ dist_,
                                ))
                            } else {
                                None
                            }
                        });
                        let river_key = (river_dist, Reverse(kind));
                        if max_key < Some(river_key) {
                            max_river = Some((river_chunk_idx, river_chunk, river, dist));
                            max_key = Some(river_key);
                        }
                    }

                    // NOTE: we scale by the distance to the river divided by the difference
                    // between the edge of the river that we intersect, and the remaining distance
                    // until the nearest point in "this" chunk (i.e. the one whose top-left corner
                    // is chunk_pos) that is at least 2 chunks away from the river source.
                    if let Some((_, dist, _, (river_t, _, downhill_river_chunk))) = dist {
                        let max_distance = if !river.is_river() {
                            /*(*/
                            TerrainChunkSize::RECT_SIZE.x as f64 /* * (1.0 - (2.0f64.sqrt() / 2.0))) + 4.0*/ - lake_width * 0.5
                        } else {
                            TerrainChunkSize::RECT_SIZE.x as f64
                        };
                        let river_dist = if !river.is_river() {
                            let neighbor_pos = river_chunk_idx.map(|e| e as f64) * neighbor_coef;
                            let dist_ =
                                (wposf - neighbor_pos).map(|e| e.abs()).reduce_partial_max();
                            // let dist = (wposf - neighbor_pos).magnitude();
                            let dist_upon =
                                (dist_ - TerrainChunkSize::RECT_SIZE.x as f64 * 1.0).max(0.0);
                            if
                            /*downhill_river_chunk.river.is_river()*/
                            false {
                                dist.y
                            } else {
                                dist_upon.min(dist.y)
                            }
                        } else {
                            dist.y
                        };
                        let scale_factor = max_distance;

                        if !(dist.x == 0.0 && river_dist < scale_factor) {
                            return;
                        }
                        // We basically want to project outwards from river_pos, along the current
                        // tangent line, to chunks <= river_width * 1.0 away from this
                        // point.  We *don't* want to deal with closer chunks because they

                        // NOTE: river_width <= 2 * max terrain chunk size width, so this should not
                        // lead to division by zero.
                        // NOTE: If distance = 0.0 this goes to zero, which is desired since it
                        // means points that actually intersect with rivers will not be interpolated
                        // with the "normal" height of this point.
                        // NOTE: We keep the maximum at 1.0 so we don't undo work from another river
                        // just by being far away.
                        let river_scale = river_dist / scale_factor;
                        let river_alt = if
                        /*river.is_river()*/
                        true {
                            Lerp::lerp(
                                river_chunk.alt.max(river_chunk.water_alt),
                                downhill_river_chunk.alt.max(downhill_river_chunk.water_alt),
                                river_t as f32,
                            )
                        } else {
                            Lerp::lerp(river_chunk.alt, downhill_river_chunk.alt, river_t as f32)
                        };
                        let river_alt = Lerp::lerp(river_alt, alt, river_scale as f32);
                        let river_alt_diff = river_alt - alt;
                        let river_alt_inv = river_alt_diff as f64;
                        if river.is_river()
                        /* || downhill_river_chunk.river.is_river() */
                        {
                            river_overlap_distance_product += (1.0 - river_scale) * river_alt_inv;
                            overlap_count += 1.0 - river_scale;
                            river_count += 1.0;
                            river_distance_product *= river_scale;
                        } else {
                            // if !river.is_river() && /*river_t != 0.0*/!downhill_river_chunk.river.is_river() {
                            // let lake_alt = /*Lerp::lerp(river_chunk.alt, downhill_river_chunk.alt, river_t as f32) as f64*/alt as f64;
                            // alt_for_lake = alt_for_lake.max(lake_alt);
                            lake_dist = /*(river_dist / (TerrainChunkSize::RECT_SIZE.x as f64 * 0.5))*/river_scale.min(lake_dist);
                        }
                    }
                }
                None => {}
            }
        });

        let river_scale_factor = lake_dist
            * if river_count == 0.0 {
                1.0
            } else {
                let river_scale_factor = river_distance_product;
                assert!(river_scale_factor >= 0.0);
                if river_scale_factor == 0.0 {
                    0.0
                } else {
                    river_scale_factor.powf(if river_count == 0.0 {
                        1.0
                    } else {
                        1.0 / river_count
                    })
                }
            };

        let alt_for_river = alt
            + (if overlap_count == 0.0 {
                0.0
            } else {
                river_overlap_distance_product / overlap_count
            } * lake_dist) as f32;
        let avg_river_velocity = if real_river_count == 0.0 {
            Vec3::zero()
        } else {
            avg_river_velocity / real_river_count
        };
        let cliff_hill = (sim
            .gen_ctx
            .small_nz
            .get((wposf_turb.div(128.0)).into_array()) as f32)
            .mul(4.0);

        let riverless_alt_delta = (sim.gen_ctx.small_nz.get(
            (wposf_turb.div(200.0 * (32.0 / TerrainChunkSize::RECT_SIZE.x as f64))).into_array(),
        ) as f32)
            .min(1.0)
            .max(-1.0)
            .abs()
            .mul(3.0)
            + (sim.gen_ctx.small_nz.get(
                (wposf_turb.div(400.0 * (32.0 / TerrainChunkSize::RECT_SIZE.x as f64)))
                    .into_array(),
            ) as f32)
                .min(1.0)
                .max(-1.0)
                .abs()
                .mul(3.0);

        let (/*downhill_pos, */downhill_chunk) =
            if wpos == chunk_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e * sz as i32) {
                Some(sim_chunk)
            } else {
                local_cells(my_chunk_idx).filter_map(|neighbor_idx| {
                    let neighbor_pos = uniform_idx_as_vec2(neighbor_idx);
                    let dist = neighbor_pos - chunk_pos;
                    let in_bounds = dist.x >= -1
                        && dist.y >= -1
                        && dist.x <= 1
                        && dist.y <= 1;
                    let in_bounds =
                        in_bounds && (dist.x >= 0 && dist.y >= 0);
                    if in_bounds {
                        Some((/*neighbor_pos, */sim.get(neighbor_pos)?))
                    } else {
                        None
                    }
                })
                .min_by(|&(/*_, */a), &(/*_, */b)| (a.water_alt).partial_cmp(&(b.water_alt)).unwrap() )
            };
        /* let downhill_pos =
        min_chunk
        .and_then(|chunk| chunk.downhill)
        .map(|pos| pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32)); */

        /* .unwrap_or_else(||
            sim_chunk.downhill.map()
        );*/
        let downhill = sim_chunk.downhill;
        /* let downhill_pos = */

        /* // Minimum neighbor by altitude who is *adjacent* to this one.
            neighbors(my_chunk_idx).filter_map(|neighbor_idx| {
                let neighbor_pos = uniform_idx_as_vec2(neighbor_idx);
                let neighbor_chunk = sim.get(neighbor_pos)?
                Some((neighbor_pos, neighbor_chunk)
            })
            .min_by()

                .map(|downhill_chunk|
            downhill_pos
            .map(|downhill_chunk| {
                downhill_chunk.water_alt
                /*.min(sim_chunk.water_alt)
                .max(sim_chunk.alt.min(sim_chunk.water_alt))*/
            })
            .unwrap_or(CONFIG.sea_level)
        }
        .unwrap_or_else(|| downhill
        .map(|pos| pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32))
        .and_then(|downhill_pos| sim.get(downhill_pos)));*/

        /* let downhill_water_alt = sim.get_interpolated_monotone(wpos, |chunk|
         * chunk.water_alt)?;
         */
        /* let downhill_pos = downhill
        .map(|pos| pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32))
        .and_then(|downhill_pos| sim.get(downhill_pos)); */
        let downhill_water_alt =
            if wpos == chunk_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e * sz as i32) {
                sim_chunk.water_alt
            } else {
                /* downhill_pos
                .map(|downhill_chunk| {
                    downhill_chunk.water_alt
                    /*.min(sim_chunk.water_alt)
                    .max(sim_chunk.alt.min(sim_chunk.water_alt))*/
                })
                .unwrap_or(CONFIG.sea_level) */
                downhill_chunk
                    .map(|downhill_chunk| downhill_chunk.water_alt)
                    .unwrap_or(CONFIG.sea_level)
                /* // Minimum neighbor.
                neighbors(my_chunk_idx).filter_map(|neighbor_idx| {
                    let neighbor_pos = uniform_idx_as_vec2(neighbor_idx);
                    let neighbor_chunk = sim.get(neighbor_pos)?
                    Some((neighbor_pos, neighbor_chunk)
                })
                .min_by()

                    .map(|downhill_chunk|
                downhill_pos
                .map(|downhill_chunk| {
                    downhill_chunk.water_alt
                    /*.min(sim_chunk.water_alt)
                    .max(sim_chunk.alt.min(sim_chunk.water_alt))*/
                })
                .unwrap_or(CONFIG.sea_level)
                .unwrap_or_else(|| downhill
                .map(|pos| pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32))
                .and_then(|downhill_pos| sim.get(downhill_pos))); */
            };
        debug_assert!(sim_chunk.water_alt >= CONFIG.sea_level);

        let is_cliffs = sim_chunk.is_cliffs;
        let near_cliffs = sim_chunk.near_cliffs;

        let (in_water, alt_, water_level, warp_factor) = if let Some((
            max_border_river_pos,
            river_chunk,
            max_border_river,
            max_border_river_dist,
        )) = max_river
        {
            // This is flowing into a lake, or a lake, or is at least a non-ocean tile.
            //
            // If we are <= water_alt, we are in the lake; otherwise, we are flowing into
            // it.
            let (in_water, new_alt, new_water_alt, warp_factor) = max_border_river
                .river_kind
                .and_then(|river_kind| {
                    if let RiverKind::River { cross_section } = river_kind {
                        if max_border_river_dist.map(|(in_river, dist, _, _)| (in_river, dist.y <= 0.0))
                            != Some((true, true))
                        {
                            return None;
                        }
                        let (_, _, river_width, (river_t, (river_pos, _), downhill_river_chunk)) =
                            max_border_river_dist.unwrap();
                        let river_alt = Lerp::lerp(
                            river_chunk.alt.max(river_chunk.water_alt),
                            downhill_river_chunk.alt.max(downhill_river_chunk.water_alt),
                            river_t as f32,
                        );
                        let new_alt = (river_alt - river_gouge).max(downhill_water_alt);
                        let river_dist = wposf.distance(river_pos);
                        let river_height_factor = river_dist / (river_width * 0.5);

                        Some((
                            true,
                            Lerp::lerp(
                                new_alt - cross_section.y, /*.max(1.0)*/
                                new_alt/*river_alt.max(downhill_water_alt)*/,                   /* - cross_section.y.min(1.0)*/
                                (river_height_factor * river_height_factor) as f32,
                            ),
                            new_alt,
                            0.0,
                        ))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| {
                    max_border_river
                        .river_kind
                        .and_then(|river_kind| {
                            match river_kind {
                                RiverKind::Ocean => {
                                    let (
                                        _,
                                        dist,
                                        /*river_width*/ _,
                                        (river_t, (river_pos, _), downhill_river_chunk),
                                    ) = if let Some(dist) = max_border_river_dist {
                                        dist
                                    } else {
                                        log::error!(
                                            "Ocean: {:?} Here: {:?}, Ocean: {:?}",
                                            max_border_river,
                                            chunk_pos,
                                            max_border_river_pos
                                        );
                                        panic!(
                                            "Oceans should definitely have a downhill! ...Right?"
                                        );
                                    };
                                    let lake_water_alt = Lerp::lerp(
                                        river_chunk.alt.max(river_chunk.water_alt),
                                        downhill_river_chunk
                                            .alt
                                            .max(downhill_river_chunk.water_alt),
                                        river_t as f32,
                                    );
                                    /*let downhill_water_alt = Lerp::lerp(
                                        river_chunk.water_alt,
                                        downhill_river_chunk.water_alt,
                                        river_t
                                    );*/

                                    if dist == Vec2::zero() {
                                        // let river_dist = wposf.distance(river_pos);
                                        // let _river_height_factor = river_dist / (river_width * 0.5);
                                        return Some((
                                            true,
                                            alt_for_river.min(lake_water_alt - 1.0/* - river_gouge*/),
                                            downhill_water_alt.max(lake_water_alt/* - river_gouge*/),
                                            0.0,
                                        ));
                                    }

                                    Some((
                                        river_scale_factor <= 1.0,
                                        alt_for_river,
                                        downhill_water_alt,
                                        river_scale_factor as f32,
                                    ))
                                },
                                RiverKind::Lake { .. } => {
                                    let lake_pos = (max_border_river_pos.map(|e| e as f64) * neighbor_coef);
                                    let lake_dist = lake_pos.distance(wposf);
                                    let downhill_river_chunk = max_border_river_pos;
                                    let lake_id_dist = downhill_river_chunk - chunk_pos;
                                    let in_bounds = lake_id_dist.x >= -1
                                        && lake_id_dist.y >= -1
                                        && lake_id_dist.x <= 1
                                        && lake_id_dist.y <= 1;
                                    let in_bounds =
                                        in_bounds && (lake_id_dist.x >= 0 && lake_id_dist.y >= 0);
                                    let (_, dist, _, (river_t, _, downhill_river_chunk)) =
                                        if let Some(dist) = max_border_river_dist {
                                            dist
                                        } else {
                                           /* return Some((
                                               false,
                                               alt,
                                               downhill_water_alt,
                                               river_scale_factor as f32,
                                           )); */
                                           if lake_dist
                                                <= TerrainChunkSize::RECT_SIZE.x as f64 * 0.5/*lake_width*/
                                                || in_bounds
                                            {
                                                let gouge_factor = 0.0;
                                                return Some((
                                                    in_bounds
                                                        || downhill_water_alt
                                                            .max(river_chunk.water_alt)
                                                            > /*alt_for_river*/alt,
                                                    alt_for_river/*alt*/,
                                                    (downhill_water_alt
                                                        .max(river_chunk.water_alt/* - river_gouge*/)),
                                                    /*river_scale_factor as f32
                                                        * (1.0 - gouge_factor)*/
                                                    river_scale_factor as f32,
                                                ));
                                            } else {
                                                return Some((
                                                    false,
                                                    alt_for_river,
                                                    downhill_water_alt,
                                                    river_scale_factor as f32,
                                                ));
                                            }
                                        };

                                    let lake_dist_real = lake_dist;
                                    let lake_height_factor = lake_dist_real / (lake_width * 0.5);
                                    let lake_dist = dist.y;
                                    let lake_water_alt = Lerp::lerp(
                                        river_chunk.alt.max(river_chunk.water_alt),
                                        downhill_river_chunk
                                            .alt
                                            .max(downhill_river_chunk.water_alt),
                                        river_t as f32,
                                    );
                                    /* let downhill_water_alt = Lerp::lerp(
                                        river_chunk.water_alt,
                                        downhill_river_chunk.alt.max(downhill_river_chunk.water_alt),
                                        river_t as f32,
                                    ); */
                                    if dist.y /*== Vec2::zero()*/<= 0.0 {
                                        return Some((
                                            true,
                                            /* Lerp::lerp(
                                                alt_for_river.min(lake_water_alt - 1.0/* - river_gouge*/),
                                                alt_for_river,
                                                lake_height_factor as f32,
                                            ), */
                                            /*alt.min(lake_water_alt - 1.0),*/
                                            /*alt_for_river.min(lake_water_alt/* - 1.0*//* - river_gouge*/)*/
                                            alt_for_river.min(lake_water_alt - 1.0),
                                            downhill_water_alt.max(lake_water_alt/* - river_gouge*/),
                                            0.0,
                                        ));
                                    }
                                    let in_bounds_dist = (wposf - lake_pos).map(|e| e.abs()).reduce_partial_max();
                                    let in_bounds_dist_ = (in_bounds_dist - TerrainChunkSize::RECT_SIZE.x as f64 * 1.0).max(0.0);
                                    let in_bounds = in_bounds_dist_ <= 0.0;
                                    let in_bounds_ = in_bounds_dist_.min(dist.y);

                                    let out_of_bounds_ = false; // downhill_river_chunk.river.is_river(); // /* !in_bounds && */in_bounds && lake_dist >= TerrainChunkSize::RECT_SIZE.x as f64 * 1.0 - lake_width * 0.5 - 1.0;
                                    if /*lake_dist <= TerrainChunkSize::RECT_SIZE.x as f64 * 1.0 - lake_width * 0.5*/in_bounds {
                                        return Some((
                                            river_scale_factor <= 1.0,
                                            if out_of_bounds_ {
                                                Lerp::lerp(alt_for_river.min(lake_water_alt), alt_for_river, (dist.y / TerrainChunkSize::RECT_SIZE.x as f64) as f32)
                                            } else {
                                                alt_for_river/*alt.max(lake_water_alt.max(downhill_water_alt))*/
                                            },
                                            downhill_water_alt.max(lake_water_alt/* - river_gouge*/),
                                            if out_of_bounds_ { 0.0 } else { /*river_scale_factor as f32*/0.0 },
                                        ));
                                    } /*else if lake_dist <= TerrainChunkSize::RECT_SIZE.x as f64 * 0.5 {
                                        return Some((
                                            alt_for_river < lake_water_alt || in_bounds,
                                            alt_for_river.min(lake_water_alt - 1.0/* - river_gouge*/),
                                            // alt.min(lake_water_alt - 1.0/* - river_gouge*/),
                                            downhill_water_alt
                                                .max(lake_water_alt/* - river_gouge*/),
                                            0.0,
                                        ));

                                    }
                                    } */else if in_bounds_dist_ <= TerrainChunkSize::RECT_SIZE.x as f64 * 0.5 || dist.y <= TerrainChunkSize::RECT_SIZE.x as f64 * 1.0 {
                                        // let lake_scale_factor = in_bounds_ / (TerrainChunkSize::RECT_SIZE.x as f64 * 0.5) as f64;
                                        return Some((
                                            river_scale_factor /* * lake_scale_factor*/ <= 1.0,
                                            /*Lerp::lerp(alt, alt_for_river, lake_scale_factor as f32)*/
                                            alt_for_river,
                                            downhill_water_alt,
                                            /*(river_scale_factor * lake_scale_factor) as
                                            * f32*/river_scale_factor as f32,
                                        ));
                                    }
                                    /* if lake_dist <= TerrainChunkSize::RECT_SIZE.x as f64 * 1.0
                                        || in_bounds
                                    {
                                        let gouge_factor = if in_bounds && lake_dist <= /*1.0*//*0.5*/0.0/*.5*/ {
                                            1.0
                                        } else {
                                            0.0
                                        };
                                        let in_bounds_ =
                                            (lake_dist <= TerrainChunkSize::RECT_SIZE.x as f64 * 1.0 - lake_width * 0.5)
                                            /*lake_dist <= TerrainChunkSize::RECT_SIZE.x as f64 * 0.5 || *//*|| in_bounds*//* + 1.0 || in_bounds*/;
                                        if gouge_factor == 1.0 {
                                            return Some((
                                                true,
                                                alt_for_river.min(lake_water_alt - 1.0/* - river_gouge*/),
                                                // alt.min(lake_water_alt - 1.0/* - river_gouge*/),
                                                downhill_water_alt
                                                    .max(lake_water_alt/* - river_gouge*/),
                                                0.0,
                                            ));
                                        } else {
                                            return Some((
                                                true,
                                                alt_for_river.max(downhill_water_alt),
                                                if in_bounds_ {
                                                    downhill_water_alt
                                                        .max(lake_water_alt/* - river_gouge*/)
                                                } else {
                                                    downhill_water_alt
                                                        .max(lake_water_alt/* - river_gouge*/)
                                                },
                                                river_scale_factor as f32 * (1.0 - gouge_factor),
                                            ));
                                        }
                                    } */
                                    Some((
                                        river_scale_factor <= 1.0,
                                        alt_for_river,
                                        downhill_water_alt,
                                        river_scale_factor as f32,
                                    ))
                                },
                                RiverKind::River { .. } => {
                                    // FIXME: Make water altitude accurate.
                                    let (
                                        _,
                                        dist,
                                        /*river_width*/ _,
                                        (river_t, (_river_pos, _), downhill_river_chunk),
                                    ) = max_border_river_dist.expect("Rivers must have a dist.");
                                    let downhill_water_alt = Lerp::lerp(
                                        river_chunk.water_alt,
                                        downhill_river_chunk.water_alt,
                                        river_t as f32
                                    );
                                    let river_alt = Lerp::lerp(
                                        river_chunk.alt.max(river_chunk.water_alt),
                                        downhill_river_chunk.alt.max(downhill_river_chunk.water_alt),
                                        river_t as f32,
                                    );
                                    Some((
                                        river_scale_factor <= 1.0,
                                        if dist.y <= 1.0 {
                                            alt_for_river.max(river_alt.max(downhill_water_alt))
                                        } else {
                                            alt_for_river
                                        },
                                        downhill_water_alt,
                                        if dist.y <= 1.0 {
                                            0.0
                                        } else {
                                            river_scale_factor as f32
                                        },
                                    ))
                                },
                            }
                        })
                        .unwrap_or((
                            false,
                            alt_for_river,
                            downhill_water_alt,
                            river_scale_factor as f32,
                        ))
                });
            (in_water, new_alt, new_water_alt, warp_factor)
        } else {
            (false, alt_for_river, downhill_water_alt, 1.0)
        };
        // NOTE: To disable warp, uncomment this line.
        // let warp_factor = 0.0;

        // println!("Pos: {:?}, downhill_water_alt: {:?} alt_for_river: {:?}, water_level: {:?}", wposf, downhill_water_alt, alt_for_river, water_level);

        /*let alt = if water_level < alt_ {
            let riverless_alt_delta = Lerp::lerp(0.0, riverless_alt_delta, warp_factor);
            alt_ + riverless_alt_delta
        } else {
            alt_
        };*/
        let riverless_alt_delta = Lerp::lerp(0.0, riverless_alt_delta, warp_factor);
        let alt = alt_ + riverless_alt_delta;
        let basement =
            alt + sim.get_interpolated_monotone(wpos, |chunk| chunk.basement.sub(chunk.alt))?;

        let rock = (sim.gen_ctx.small_nz.get(
            Vec3::new(wposf.x, wposf.y, alt as f64)
                .div(100.0)
                .into_array(),
        ) as f32)
            .mul(rockiness)
            .sub(0.4)
            .max(0.0)
            .mul(8.0);

        let wposf3d = Vec3::new(wposf.x, wposf.y, alt as f64);

        let marble_small = (sim.gen_ctx.hill_nz.get((wposf3d.div(3.0)).into_array()) as f32)
            .powf(3.0)
            .add(1.0)
            .mul(0.5);
        let marble = (sim.gen_ctx.hill_nz.get((wposf3d.div(48.0)).into_array()) as f32)
            .mul(0.75)
            .add(1.0)
            .mul(0.5)
            .add(marble_small.sub(0.5).mul(0.25));

        // Colours
        let cold_grass = Rgb::new(0.0, 0.5, 0.25);
        let warm_grass = Rgb::new(0.4, 0.8, 0.0);
        let dark_grass = Rgb::new(0.15, 0.4, 0.1);
        let wet_grass = Rgb::new(0.1, 0.8, 0.2);
        let cold_stone = Rgb::new(0.57, 0.67, 0.8);
        let hot_stone = Rgb::new(0.07, 0.07, 0.06);
        let warm_stone = Rgb::new(0.77, 0.77, 0.64);
        let beach_sand = Rgb::new(0.9, 0.82, 0.6);
        let desert_sand = Rgb::new(0.95, 0.75, 0.5);
        let snow = Rgb::new(0.8, 0.85, 1.0);

        let stone_col = Rgb::new(195, 187, 201);
        let dirt = Lerp::lerp(
            Rgb::new(0.075, 0.07, 0.3),
            Rgb::new(0.75, 0.55, 0.1),
            marble,
        );
        let tundra = Lerp::lerp(snow, Rgb::new(0.01, 0.3, 0.0), 0.4 + marble * 0.6);
        let dead_tundra = Lerp::lerp(warm_stone, Rgb::new(0.3, 0.12, 0.2), marble);
        let cliff = Rgb::lerp(cold_stone, hot_stone, marble);

        let grass = Rgb::lerp(
            cold_grass,
            warm_grass,
            marble.sub(0.5).add(1.0.sub(humidity).mul(0.5)).powf(1.5),
        );
        let snow_moss = Rgb::lerp(snow, cold_grass, 0.4 + marble.powf(1.5) * 0.6);
        let moss = Rgb::lerp(dark_grass, cold_grass, marble.powf(1.5));
        let rainforest = Rgb::lerp(wet_grass, warm_grass, marble.powf(1.5));
        let sand = Rgb::lerp(beach_sand, desert_sand, marble);

        let tropical = Rgb::lerp(
            Rgb::lerp(
                grass,
                Rgb::new(0.15, 0.2, 0.15),
                marble_small
                    .sub(0.5)
                    .mul(0.2)
                    .add(0.75.mul(1.0.sub(humidity)))
                    .powf(0.667),
            ),
            Rgb::new(0.87, 0.62, 0.56),
            marble.powf(1.5).sub(0.5).mul(4.0),
        );

        // For below desert humidity, we are always sand or rock, depending on altitude
        // and temperature.
        let ground = Lerp::lerp(
            Lerp::lerp(
                dead_tundra,
                sand,
                temp.sub(CONFIG.snow_temp)
                    .div(CONFIG.desert_temp.sub(CONFIG.snow_temp))
                    .mul(0.5),
            ),
            dirt,
            humidity
                .sub(CONFIG.desert_hum)
                .div(CONFIG.forest_hum.sub(CONFIG.desert_hum))
                .mul(1.0),
        );

        let sub_surface_color = Lerp::lerp(cliff, ground, alt.sub(basement).mul(0.25));

        // From desert to forest humidity, we go from tundra to dirt to grass to moss to
        // sand, depending on temperature.
        let ground = Rgb::lerp(
            ground,
            Rgb::lerp(
                Rgb::lerp(
                    Rgb::lerp(
                        Rgb::lerp(
                            tundra,
                            // snow_temp to temperate_temp
                            dirt,
                            temp.sub(CONFIG.snow_temp)
                                .div(CONFIG.temperate_temp.sub(CONFIG.snow_temp))
                                /*.sub((marble - 0.5) * 0.05)
                                .mul(256.0)*/
                                .mul(1.0),
                        ),
                        // temperate_temp to tropical_temp
                        grass,
                        temp.sub(CONFIG.temperate_temp)
                            .div(CONFIG.tropical_temp.sub(CONFIG.temperate_temp))
                            .mul(4.0),
                    ),
                    // tropical_temp to desert_temp
                    moss,
                    temp.sub(CONFIG.tropical_temp)
                        .div(CONFIG.desert_temp.sub(CONFIG.tropical_temp))
                        .mul(1.0),
                ),
                // above desert_temp
                sand,
                temp.sub(CONFIG.desert_temp)
                    .div(1.0 - CONFIG.desert_temp)
                    .mul(4.0),
            ),
            humidity
                .sub(CONFIG.desert_hum)
                .div(CONFIG.forest_hum.sub(CONFIG.desert_hum))
                .mul(1.0),
        );
        // From forest to jungle humidity, we go from snow to dark grass to grass to
        // tropics to sand depending on temperature.
        let ground = Rgb::lerp(
            ground,
            Rgb::lerp(
                Rgb::lerp(
                    Rgb::lerp(
                        snow_moss,
                        // temperate_temp to tropical_temp
                        grass,
                        temp.sub(CONFIG.temperate_temp)
                            .div(CONFIG.tropical_temp.sub(CONFIG.temperate_temp))
                            .mul(4.0),
                    ),
                    // tropical_temp to desert_temp
                    tropical,
                    temp.sub(CONFIG.tropical_temp)
                        .div(CONFIG.desert_temp.sub(CONFIG.tropical_temp))
                        .mul(1.0),
                ),
                // above desert_temp
                sand,
                temp.sub(CONFIG.desert_temp)
                    .div(1.0 - CONFIG.desert_temp)
                    .mul(4.0),
            ),
            humidity
                .sub(CONFIG.forest_hum)
                .div(CONFIG.jungle_hum.sub(CONFIG.forest_hum))
                .mul(1.0),
        );
        // From jungle humidity upwards, we go from snow to grass to rainforest to
        // tropics to sand.
        let ground = Rgb::lerp(
            ground,
            Rgb::lerp(
                Rgb::lerp(
                    Rgb::lerp(
                        snow_moss,
                        // temperate_temp to tropical_temp
                        rainforest,
                        temp.sub(CONFIG.temperate_temp)
                            .div(CONFIG.tropical_temp.sub(CONFIG.temperate_temp))
                            .mul(4.0),
                    ),
                    // tropical_temp to desert_temp
                    tropical,
                    temp.sub(CONFIG.tropical_temp)
                        .div(CONFIG.desert_temp.sub(CONFIG.tropical_temp))
                        .mul(4.0),
                ),
                // above desert_temp
                sand,
                temp.sub(CONFIG.desert_temp)
                    .div(1.0 - CONFIG.desert_temp)
                    .mul(4.0),
            ),
            humidity.sub(CONFIG.jungle_hum).mul(1.0),
        );

        // Snow covering
        let snow_cover = temp
            .sub(CONFIG.snow_temp)
            .max(-humidity.sub(CONFIG.desert_hum))
            .mul(16.0)
            .add((marble_small - 0.5) * 0.5);
        let (alt, ground, sub_surface_color) = if snow_cover <= 0.5 && alt > water_level {
            // Allow snow cover.
            (
                alt + 1.0 - snow_cover.max(0.0),
                Rgb::lerp(snow, ground, snow_cover),
                Lerp::lerp(sub_surface_color, ground, alt.sub(basement).mul(0.15)),
            )
        } else {
            (alt, ground, sub_surface_color)
        };

        // Caves
        let cave_at = |wposf: Vec2<f64>| {
            (sim.gen_ctx.cave_0_nz.get(
                Vec3::new(wposf.x, wposf.y, alt as f64 * 8.0)
                    .div(800.0)
                    .into_array(),
            ) as f32)
                .powf(2.0)
                .neg()
                .add(1.0)
                .mul((1.32 - chaos).min(1.0))
        };
        let cave_xy = cave_at(wposf);
        let cave_alt = alt - 24.0
            + (sim
                .gen_ctx
                .cave_1_nz
                .get(Vec2::new(wposf.x, wposf.y).div(48.0).into_array()) as f32)
                * 8.0
            + (sim
                .gen_ctx
                .cave_1_nz
                .get(Vec2::new(wposf.x, wposf.y).div(500.0).into_array()) as f32)
                .add(1.0)
                .mul(0.5)
                .powf(15.0)
                .mul(150.0);

        let near_ocean = max_river.and_then(|(_, _, river_data, _)| {
            if (river_data.is_lake() || river_data.river_kind == Some(RiverKind::Ocean))
                && ((alt <= water_level.max(CONFIG.sea_level + 5.0) && !is_cliffs) || !near_cliffs)
            {
                Some(water_level)
            } else {
                None
            }
        });

        let (water_packed, water_velocity) = /*if let Some((
            _,
            river_chunk,
            river_data,
            Some((
                _,
                dist,
                /*river_width*/ _,
                (river_t, (/*river_pos*/ _, coeffs), downhill_river_chunk),
            )),
        )) = max_river */
        {
            if /*river_data.is_river() && dist == Vec2::zero()*/
            avg_river_velocity != Vec3::zero()
            {
                /* let river_alt = Lerp::lerp(
                    river_chunk.alt.max(river_chunk.water_alt),
                    downhill_river_chunk.alt.max(downhill_river_chunk.water_alt),
                    river_t as f32,
                ); */
                /* let river_segment_length = quadratic_arc_length(coeffs);
                assert!(river_segment_length > 0.0); */

                /* let river_segment_length = 1.0;
                // Idea: since x and y are in terms of "distance per t", t goes from 0.0 to 1.0,
                // and z linearly goes from river_chunk.alt to downhill_river_chunk.alt from t=0.0
                // to 1.0, the z part is already in the same units as x and y, meaning we don't
                // have to do anything too expensive like compute the "actual" arc length.
                // TODO: Shouldn't the z component be a quadratic term due to gravity or something?
                // Check this out.
                let dz = (downhill_river_chunk.alt.max(downhill_river_chunk.water_alt)
                    - river_chunk.alt.max(river_chunk.water_alt)) as f64
                    / river_segment_length;
                let slope = coeffs.x * 2.0 * river_t.max(0.0).min(1.0) + coeffs.y;
                let downhill_river_velocity = if downhill_river_chunk.river.is_river() {
                    downhill_river_chunk.river.velocity
                } else {
                    river_data.velocity
                };
                let velocity_magnitude =
                    Lerp::lerp(river_data.velocity, downhill_river_velocity, river_t as f32) as f64;
                // let velocity_magnitude = river_data.velocity as f64;
                let mut velocity = Vec3::new(slope.x, slope.y, dz);
                if velocity != Vec3::zero() {
                    velocity.normalize();
                } */
                let velocity_magnitude = avg_river_velocity.magnitude();
                let mut velocity = avg_river_velocity;
                velocity.normalize();
                // Compute Euler angles.
                let angle_h = velocity.y.atan2(velocity.x);
                // Make sure to prevent rounding errors from causing asin() to return NaN.
                let angle_p = velocity.z.min(1.0).asin();
                // Encode Euler angles in a packed bit representation, such that we use just 4 bits
                // for angle_h and 4 bits for angle_p.
                // We exploit the fact that angle_p is always within [-pi/2, 0] to save some bits;
                // turns into between 0 and 15.
                if !(angle_p <= 0.0) {
                    log::error!(
                        "angle_p <= 0: avg_river_velocity={:?}, velocity={:?}, angle_p={:?}",
                        avg_river_velocity,
                        velocity,
                        angle_p
                    );
                    panic!("assertion failed: angle_p <= 0.0");
                }
                let encoded_angle_p = angle_p
                    .mul(-f64::consts::FRAC_2_PI)
                    .mul(15.0)
                    .round()
                    .max(0.0)
                    .min(15.0) as u32;
                // angle_h is between -pi and pi, turns into between 0 and 15.
                let encoded_angle_h = angle_h
                    .mul(f64::consts::FRAC_1_PI)
                    .mul(31.5)
                    .add(31.5)
                    .round()
                    .max(0.0)
                    .min(63.0) as u32;
                // Velocity magnitude is squeezed into 6 bits.  For now, we interpret the 64 values
                // as numbers from 0 to 16 increasing in increments of 0.25, but we will probably
                // change this at some point.
                let encoded_velocity = f32_to_f7(velocity_magnitude as f32)
                    .expect("velocity_magnitude must be positive") as u32;
                /* .min(7.5)
                    .mul(16.0)
                    .round()
                    .max(0.0)
                    .min(63.0) as u32;
                let velocity = (velocity * velocity_magnitude).map(|e| e as f32); */
                let velocity = avg_river_velocity.map(|e| e as f32);
                // Now, we encode the velocity into a packed 14-bit RGB sequence.  Subvoxel heights
                // will come later.
                let water_packed =
                    encoded_angle_h | (encoded_angle_p << 6) | (encoded_velocity << 10);
                (water_packed, velocity)
            /* let new_alt = river_alt - river_gouge;
            let river_dist = wposf.distance(river_pos);
            let river_height_factor = river_dist / (river_width * 0.5); */
            } else {
                (0, Vec3::zero())
            }
        }/* else {
            (0, Vec3::zero())
        }*/;

        let ocean_level = if let Some(_sea_level) = near_ocean {
            alt - CONFIG.sea_level
        } else {
            5.0
        };

        Some(ColumnSample {
            alt,
            basement,
            chaos,
            water_level,
            water_packed,
            water_velocity,
            warp_factor,
            surface_color: Rgb::lerp(
                Rgb::lerp(cliff, sand, alt.sub(basement).mul(0.25)),
                // Land
                ground,
                // Beach
                ((ocean_level - 1.0) / 2.0).max(0.0),
            ),
            sub_surface_color,
            // No growing directly on bedrock.
            tree_density: Lerp::lerp(0.0, tree_density, alt.sub(2.0).sub(basement).mul(0.5)),
            forest_kind: sim_chunk.forest_kind,
            close_structures: self.gen_close_structures(wpos),
            cave_xy,
            cave_alt,
            marble,
            marble_small,
            rock,
            is_cliffs,
            near_cliffs,
            cliff_hill,
            close_cliffs: sim.gen_ctx.cliff_gen.get(wpos),
            temp,
            humidity,
            spawn_rate,
            location: sim_chunk.location.as_ref(),
            stone_col,

            chunk: sim_chunk,
            spawn_rules: sim_chunk
                .structures
                .town
                .as_ref()
                .map(|town| TownGen.spawn_rules(town, wpos))
                .unwrap_or(SpawnRules::default())
                .and(SpawnRules {
                    cliffs: !in_water,
                    trees: true,
                }),
        })
    }
}

#[derive(Clone)]
pub struct ColumnSample<'a> {
    pub alt: f32,
    pub basement: f32,
    pub chaos: f32,
    pub water_level: f32,
    pub water_packed: u32,
    pub water_velocity: Vec3<f32>,
    pub warp_factor: f32,
    pub surface_color: Rgb<f32>,
    pub sub_surface_color: Rgb<f32>,
    pub tree_density: f32,
    pub forest_kind: ForestKind,
    pub close_structures: [Option<StructureData>; 9],
    pub cave_xy: f32,
    pub cave_alt: f32,
    pub marble: f32,
    pub marble_small: f32,
    pub rock: f32,
    pub is_cliffs: bool,
    pub near_cliffs: bool,
    pub cliff_hill: f32,
    pub close_cliffs: [(Vec2<i32>, u32); 9],
    pub temp: f32,
    pub humidity: f32,
    pub spawn_rate: f32,
    pub location: Option<&'a LocationInfo>,
    pub stone_col: Rgb<u8>,

    pub chunk: &'a SimChunk,
    pub spawn_rules: SpawnRules,
}

#[derive(Copy, Clone)]
pub struct StructureData {
    pub pos: Vec2<i32>,
    pub seed: u32,
    pub meta: Option<StructureMeta>,
}
