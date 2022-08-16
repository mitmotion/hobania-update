#![allow(clippy::clone_on_copy)] // TODO: fix after wgpu branch

use crate::{
    mesh::{
        greedy::{self, GreedyConfig, GreedyMesh},
        MeshGen,
    },
    render::{ColLightInfo, FluidVertex, Mesh, TerrainVertex},
    scene::terrain::BlocksOfInterest,
};
use common::{
    terrain::{Block, TerrainChunk},
    util::either_with,
    vol::{ReadVol, RectRasterableVol},
    volumes::vol_grid_2d::{CachedVolGrid2d, VolGrid2d},
};
use common_base::span;
use std::{collections::VecDeque, fmt::Debug, sync::Arc};
use tracing::error;
use vek::*;

#[derive(Clone, Copy, PartialEq)]
enum FaceKind {
    /// Opaque face that is facing something non-opaque; either
    /// water (Opaque(true)) or something else (Opaque(false)).
    Opaque(bool),
    /// Fluid face that is facing something non-opaque, non-tangible,
    /// and non-fluid (most likely air).
    Fluid,
}

pub const SUNLIGHT: u8 = 24;
pub const SUNLIGHT_INV: f32 = 1.0 / SUNLIGHT as f32;
pub const MAX_LIGHT_DIST: i32 = SUNLIGHT as i32;

/// Working around lack of existential types.
///
/// See [https://github.com/rust-lang/rust/issues/42940]
type CalcLightFn<V, I> = impl Fn(Vec3<i32>) -> f32 + 'static + Send + Sync;

#[inline(always)]
/* #[allow(unsafe_code)] */
fn flat_get<'a>(flat: &'a Vec<Block>, w: i32, h: i32, d: i32) -> impl Fn(Vec3<i32>) -> Block + 'a {
    let wh = w * h;
    let flat = &flat[0..(d * wh) as usize];
    #[inline(always)] move |Vec3 { x, y, z }| {
        // z can range from -1..range.size().d + 1
        let z = z + 1;
        flat[((z * wh + y * w + x) as usize)]
        /* unsafe { *flat.get_unchecked((z * wh + y * w + x) as usize) } */
        /* match flat.get((x * hd + y * d + z) as usize).copied() {
            Some(b) => b,
            None => panic!("x {} y {} z {} d {} h {}", x, y, z, d, h),
        } */
    }

    /* let hd = h * d;
    let flat = &flat[0..(w * hd) as usize];
    #[inline(always)] move |Vec3 { x, y, z }| {
        // z can range from -1..range.size().d + 1
        let z = z + 1;
        /* flat[((x * hd + y * d + z) as usize)] */
        unsafe { *flat.get_unchecked((x * hd + y * d + z) as usize) }
        /* match flat.get((x * hd + y * d + z) as usize).copied() {
            Some(b) => b,
            None => panic!("x {} y {} z {} d {} h {}", x, y, z, d, h),
        } */
    } */
}

fn calc_light<'a,
   V: RectRasterableVol<Vox = Block> + ReadVol + Debug + 'static,
   I: Iterator<Item=(Vec3<i32>, u8)>,
   /* F: /*for<'x> */for<'a> fn(&'a Vec<Block>) -> G, */
   /* G: /*[&'x &'a (); 0], */Fn(Vec3<i32>) -> Block, */
>(
    is_sunlight: bool,
    // When above bounds
    default_light: u8,
    bounds: Aabb<i32>,
    range: Aabb<i32>,
    vol: &'a VolGrid2d<V>,
    lit_blocks: I,
    flat: &'a Vec<Block>,
    (w, h, d): (i32, i32, i32)
) -> CalcLightFn<V, I> {
    span!(_guard, "calc_light");
    const UNKNOWN: u8 = 255;
    const OPAQUE: u8 = 254;

    let outer = Aabb {
        min: bounds.min/* - Vec3::new(SUNLIGHT as i32, SUNLIGHT as i32, 1) */ - Vec3::new(0, 0, 1),
        max: bounds.max/* + Vec3::new(SUNLIGHT as i32, SUNLIGHT as i32, 1) */ + Vec3::new(0, 0, 1),
    };

    let range_delta = outer.min - range.min;

    /* let mut vol_cached = vol.cached(); */

    let mut light_map_ = vec![UNKNOWN; outer.size().product() as usize];
    let (w_, h_, d_) = outer.clone().size().into_tuple();
    let wh_ = w_ * h_;
    let light_map = &mut light_map_[0..(wh_ * d_) as usize];
    let lm_idx = {
        #[inline(always)] move |x, y, z| {
            (wh_ * z + h_ * x + y) as usize
        }
    };
    /* // Light propagation queue
    let mut prop_que = lit_blocks
        .map(|(pos, light)| {
            let rpos = pos - outer.min;
            light_map[lm_idx(rpos.x, rpos.y, rpos.z)] = light.min(SUNLIGHT); // Brightest light
            (rpos.x as u8, rpos.y as u8, rpos.z as u16)
        })
        .collect::<VecDeque<_>>(); */
    let mut prop_que = vec![Vec::new(); usize::from(SUNLIGHT) + 1];
    let mut prop_queue = &mut prop_que[..usize::from(SUNLIGHT) + 1];
    lit_blocks.for_each(|(pos, light)| {
        /* println!("Lighting {:?}: {:?}", pos, light); */
        let rpos = pos - outer.min;
        let glow = light.min(SUNLIGHT);
        light_map[lm_idx(rpos.x, rpos.y, rpos.z)] = glow; // Brightest light
        prop_que[usize::from(glow)].push((rpos.x as u8, rpos.y as u8, rpos.z as u16));
    });

    /* // Start sun rays
    if is_sunlight {
        for x in 0..outer.size().w {
            for y in 0..outer.size().h {
                let mut light = SUNLIGHT as f32;
                for z in (0..outer.size().d).rev() {
                    let (min_light, attenuation) = vol_cached
                        .get(outer.min + Vec3::new(x, y, z))
                        .map_or((0, 0.0), |b| b.get_max_sunlight());

                    if light > min_light as f32 {
                        light = (light - attenuation).max(min_light as f32);
                    }

                    light_map[lm_idx(x, y, z)] = light.floor() as u8;

                    if light <= 0.0 {
                        break;
                    } else {
                        prop_que.push_back((x as u8, y as u8, z as u16));
                    }
                }
            }
        }
    } */

    // Determines light propagation
    let flat_get = flat_get(flat, w, h, d);
    let propagate = #[inline(always)] |src: u8,
                     dest: &mut u8,
                     pos: Vec3<i32>,
                     prop_que: &mut /*VecDeque*/Vec<_>,
                     /* vol: &mut CachedVolGrid2d<V> */| {
        let dst = *dest;
        /* if dst != OPAQUE */{
            if dst < src || dst == UNKNOWN /* {
                if */&& /* vol
                    .get(outer.min + pos)
                    .ok()
                    .map_or(false, |b| b.is_fluid()) */
                    flat_get(/*[], */pos + range_delta).is_fluid()
                {
                    /* *dest = src.saturating_sub(1); */
                    *dest = src;
                    // Can't propagate further
                    if /* *dest */src > 1 {
                        prop_que./*push_back*/push((pos.x as u8, pos.y as u8, pos.z as u16));
                    }
                }/* else {
                    *dest = OPAQUE;
                } */
            /*} else if *dest < src/* .saturating_sub(1) */ {
                *dest = src/* - 1 */;
                // Can't propagate further
                if /* *dest */src > 1 {
                    prop_que./*push_back*/push((pos.x as u8, pos.y as u8, pos.z as u16));
                }
            } */
        }
    };

    // Propagate light
    //
    // NOTE: We start at 2 because starting at 1 would propagate light of brightness 0 to
    // neighbors.
    (2..usize::from(SUNLIGHT) + 1).rev().for_each(|light| {
        let (front, back) = prop_que.split_at_mut(light);
        let prop_que = front.last_mut().expect("Split at least at index 2, so front must have at least 1 element.");
        let front = back.first_mut().expect("Split at most at SUNLIGHT, and array has length SUNLIGHT+1, so back must have at least 1 element.");
        let light = light as u8;
        // NOTE: Always in bounds and ≥ 1, since light ≥ 2.
        let new_light = light - 1;
        /* println!("Light: {:?}", light); */
        /* while let Some(pos) = prop_que.pop_front() */front.iter().for_each(|pos| {
            let pos = Vec3::new(pos.0 as i32, pos.1 as i32, pos.2 as i32);
            let light_ = light_map[lm_idx(pos.x, pos.y, pos.z)];
            if light != light_ {
                // This block got modified before it could emit anything.
                return;
            }
            /* println!("Pos: {:?}", pos); */

            // Up
            // Bounds checking
            // NOTE: Array accesses are all safe even if they are technically out of bounds,
            // because we have margin on all sides and the light sources only come from within the
            // proper confines of the volume.  This allows us to fetch them before the if
            // statements.
            {
            /* let light_map = light_map.as_mut_ptr();
            let z_up = &mut *light_map.offset(lm_idx(pos.x, pos.y, pos.z + 1) as isize);
            let z_down = &mut *light_map.offset(lm_idx(pos.x, pos.y, pos.z - 1) as isize);
            let y_up = &mut *light_map.offset(lm_idx(pos.x, pos.y + 1, pos.z) as isize);
            let y_down = &mut *light_map.offset(lm_idx(pos.x, pos.y - 1, pos.z) as isize);
            let x_up = &mut *light_map.offset(lm_idx(pos.x + 1, pos.y, pos.z) as isize);
            let x_down = &mut *light_map.offset(lm_idx(pos.x - 1, pos.y, pos.z) as isize); */
            if pos.z + 1 < outer.size().d {
                let z_up = &mut light_map[lm_idx(pos.x, pos.y, pos.z + 1)];
                propagate(
                    new_light,
                    z_up,
                    Vec3::new(pos.x, pos.y, pos.z + 1),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            // Down
            if pos.z > 0 {
                let z_down = &mut light_map[lm_idx(pos.x, pos.y, pos.z - 1)];
                propagate(
                    new_light,
                    z_down,
                    Vec3::new(pos.x, pos.y, pos.z - 1),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            // The XY directions
            if pos.y + 1 < outer.size().h {
                let y_up = &mut light_map[lm_idx(pos.x, pos.y + 1, pos.z)];
                propagate(
                    new_light,
                    y_up,
                    Vec3::new(pos.x, pos.y + 1, pos.z),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            if pos.y > 0 {
                let y_down = &mut light_map[lm_idx(pos.x, pos.y - 1, pos.z)];
                propagate(
                    new_light,
                    y_down,
                    Vec3::new(pos.x, pos.y - 1, pos.z),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            if pos.x + 1 < outer.size().w {
                let x_up = &mut light_map[lm_idx(pos.x + 1, pos.y, pos.z)];
                propagate(
                    new_light,
                    x_up,
                    Vec3::new(pos.x + 1, pos.y, pos.z),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            if pos.x > 0 {
                let x_down = &mut light_map[lm_idx(pos.x - 1, pos.y, pos.z)];
                propagate(
                    new_light,
                    x_down,
                    Vec3::new(pos.x - 1, pos.y, pos.z),
                    /*&mut */prop_que,
                    /* &mut vol_cached, */
                )
            }
            }
        });
    });

    let min_bounds = Aabb {
        min: bounds.min - Vec3::unit_z(),
        max: bounds.max + Vec3::unit_z(),
    };

    /* // Minimise light map to reduce duplication. We can now discard light info
    // for blocks outside of the chunk borders.
    let mut light_map2 = vec![UNKNOWN; min_bounds.size().product() as usize];
    let lm_idx2 = {
        let (w, h, _) = min_bounds.clone().size().into_tuple();
        move |x, y, z| (w * h * z + h * x + y) as usize
    };
    for x in 0..min_bounds.size().w {
        for y in 0..min_bounds.size().h {
            for z in 0..min_bounds.size().d {
                let off = min_bounds.min - outer.min;
                light_map2[lm_idx2(x, y, z)] = light_map[lm_idx(x + off.x, y + off.y, z + off.z)];
            }
        }
    }

    drop(light_map_); */
    let light_map2 = light_map_;

    #[inline(always)] move |wpos| {
        if is_sunlight { return 1.0 }/* else { 0.0 } */
        let pos = wpos - min_bounds.min;
        let l = light_map2
            .get(/*lm_idx2*/lm_idx(pos.x, pos.y, pos.z))
            .copied()
            .unwrap_or(default_light);

        if /* l != OPAQUE && */l != UNKNOWN {
            l as f32 * SUNLIGHT_INV
        } else {
            0.0
        }
    }
}

type V = TerrainChunk;

#[allow(clippy::type_complexity)]
#[inline(always)]
pub fn generate_mesh<'a/*, V: RectRasterableVol<Vox = Block> + ReadVol + Debug + 'static*/>(
    vol: &'a VolGrid2d<V>,
    (range, max_texture_size, boi): (Aabb<i32>, Vec2<u16>, &'a BlocksOfInterest),
) -> MeshGen<
    TerrainVertex,
    FluidVertex,
    TerrainVertex,
    (
        Aabb<f32>,
        ColLightInfo,
        Arc<dyn Fn(Vec3<i32>) -> f32 + Send + Sync>,
        Arc<dyn Fn(Vec3<i32>) -> f32 + Send + Sync>,
    ),
> {
    span!(
        _guard,
        "generate_mesh",
        "<&VolGrid2d as Meshable<_, _>>::generate_mesh"
    );

    let mut opaque_limits = None::<Limits>;
    let mut fluid_limits = None::<Limits>;
    let mut air_limits = None::<Limits>;
    let mut flat;
    let (w, h, d) = range.size().into_tuple();
    // z can range from -1..range.size().d + 1
    let d = d + 2;

    /// Representative block for air.
    const AIR: Block = Block::air(common::terrain::sprite::SpriteKind::Empty);
    /// Representative block for liquid.
    ///
    /// FIXME: Can you really skip meshing for general liquids?  Probably not...
    const LIQUID: Block = Block::water(common::terrain::sprite::SpriteKind::Empty);
    /// Representtive block for solids.
    ///
    /// FIXME: Really hacky!
    const OPAQUE: Block = Block::lava(common::terrain::sprite::SpriteKind::Empty);

    const ALL_OPAQUE: u8 = 0b1;
    const ALL_LIQUID: u8 = 0b10;
    const ALL_AIR: u8 = 0b100;
    // For each horizontal slice of the chunk, we keep track of what kinds of blocks are in it.
    // This allows us to compute limits after the fact, much more precisely than keeping track of a
    // single intersection would; it also lets us skip homogeoneous slices entirely.
    let mut row_kinds = vec![0; d as usize];
    /* {
        span!(_guard, "copy to flat array");
        let hd = h * d;
        /*let flat = */{
            let mut arena = bumpalo::Bump::new();

            /* let mut volume = vol.cached(); */
            // TODO: Once we can manage it sensibly, consider using something like
            // Option<Block> instead of just assuming air.
            /*let mut */flat = vec![AIR; (w * /*h * d*/hd) as usize]
                /* Vec::with_capacity((w * /*h * d*/hd) as usize) */
                ;
            let row_kinds = &mut row_kinds[0..d as usize];
            let flat = &mut flat/*.spare_capacity_mut()*/[0..(w * hd) as usize];
            /* /*volume*/vol.iter().for_each(|(chunk_key, chunk)| {
                let corner = chunk.key_pos(chunk_key);
            }); */
            let flat_range = Aabb {
                min: range.min - Vec3::new(0, 0, 1),
                max: range.max - Vec3::new(1, 1, 0),
            };
            let min_chunk_key = vol.pos_key(flat_range.min);
            let max_chunk_key = vol.pos_key(flat_range.max);
            (min_chunk_key.x..max_chunk_key.x + 1).for_each(|key_x| {
            (min_chunk_key.y..max_chunk_key.y + 1).for_each(|key_y| {
            let key = Vec2::new(key_x, key_y);
            let chonk = vol.get_key(key).expect("All keys in range must have chonks.");
            /* vol.iter().for_each(|(key, chonk)| { */
                let chonk = &*chonk;
                let pos = vol.key_pos(key);
                /* // Avoid diagonals.
                if pos.x != range.min.x + 1 && pos.y != range.min.y + 1 { return; } */
                // Calculate intersection of Aabb and this chunk
                // TODO: should we do this more implicitly as part of the loop
                // TODO: this probably has to be computed in the chunk.for_each_in() as well
                // maybe remove here?
                let intersection_ = flat_range.intersection(Aabb {
                    min: pos.with_z(i32::MIN),
                    // -1 here since the Aabb is inclusive and chunk_offs below will wrap it if
                    // it's outside the range of the chunk
                    max: (pos + VolGrid2d::<V>::chunk_size().map(|e| e as i32) - 1).with_z(i32::MAX),
                });

                // Map intersection into chunk coordinates
                let x_diff = intersection_.min.x - flat_range.min.x;
                let y_diff = intersection_.min.y - flat_range.min.y;
                let z_diff = -range.min.z;
                let y_rem = flat_range.max.y - intersection_.max.y;
                let x_off = ((y_diff + y_rem) * d) as usize;

                let intersection = Aabb {
                    min: VolGrid2d::<V>::chunk_offs(intersection_.min) + Vec3::new(0, 0, z_diff),
                    max: VolGrid2d::<V>::chunk_offs(intersection_.max) + Vec3::new(1, 1, z_diff + 1),
                };
                let z_diff = z_diff + chonk.get_min_z();
                let z_max = chonk.get_max_z() - chonk.get_min_z();
                let below = *chonk.below();

                /* [[0  ..1]; [0  ..1];   [0..d]]
                [[0  ..1]; [1  ..h-1]; [0..d]]
                [[0  ..1]; [h-1..h];   [0..d]]
                // How to handle middle?
                // Answer:
                [[w-1..w]; [0  ..1];   [0..d]]
                [[w-1..w]; [1  ..h-1]; [0..d]]
                [[w-1..w]; [h-1..h];   [0..d]]

                [1,1; d] */

                let flat_chunk = chonk.make_flat(/*&stone_slice, &air_slice, */&arena);

                let mut i = (x_diff * hd + y_diff * d) as usize;
                let hd_ = (intersection.size().h * d) as usize;

                let min_z_ = z_diff - intersection.min.z;
                let max_z_ = z_max + z_diff - intersection.min.z;
                let row_fill = if below.is_opaque() {
                    /* opaque_limits = opaque_limits
                        .map(|l| l.including(z_diff))
                        .or_else(|| Some(Limits::from_value(z_diff))); */
                    ALL_OPAQUE
                } else if below.is_liquid() {
                    /* fluid_limits = fluid_limits
                        .map(|l| l.including(z_diff))
                        .or_else(|| Some(Limits::from_value(z_diff))); */
                    ALL_LIQUID
                } else {
                    /* // Assume air
                    air_limits = air_limits
                        .map(|l| l.including(z_diff))
                        .or_else(|| Some(Limits::from_value(z_diff))); */
                    ALL_AIR
                };

                let skip_count = min_z_.max(0);
                let take_count = (max_z_.min(d) - skip_count).max(0);
                let skip_count = skip_count as usize;
                let take_count = take_count as usize;

                // Fill the bottom rows with their below type.
                row_kinds.iter_mut().take(skip_count).for_each(|row| {
                    *row |= row_fill;
                });
                // Fill the top rows with air (we just assume that's the above type, since it
                // always is in practice).
                row_kinds.iter_mut().skip(skip_count + take_count).for_each(|row| {
                    *row |= ALL_AIR;
                });

                // dbg!(pos, intersection_, intersection, range, flat_range, x_diff, y_diff, z_diff, y_rem, x_off, i);
                (intersection.min.x..intersection.max.x).for_each(|x| {
                let flat = &mut flat[i..i + /*intersection.size().y * intersection.size().z*/hd_];
                flat.chunks_exact_mut(d as usize).enumerate().for_each(|(y, flat)| {
                let y = y as i32 + intersection.min.y;
                /* (intersection.min.y..intersection.max.y).for_each(|y| { */
                /* let mut i = (x * hd + y * d) as usize; */
                /* chonk.for_each_in(intersection, |pos_offset, block| {
                    pos_offset.z += z_diff;
                }); */

                // intersection.min.z = range.min.z - 1 - range.min.z = -1
                // z_diff = chonk.get_min_z() - range.min.z
                // min_z_ = chonk.get_min_z() - (range.min.z - 1)
                //
                // max_z_ = (chonk.get_max_z() - (range.min.z - 1)).min(d - skip_count)
                flat[0..skip_count].fill(below);
                flat.into_iter().zip(row_kinds.into_iter()).enumerate().skip(skip_count).take(take_count).for_each(|(z, (flat, row))| {
                let z = z as i32 + intersection.min.z;
                /* (intersection.min.z..intersection.max.z).for_each(|z| { */
                /* let mut i = ((x_diff + (x - intersection.min.x)) * hd + (y_diff + (y - intersection.min.y)) * d + (z - intersection.min.z)) as usize; */
            /* vol.iter(flat_range, |wpos, block| {
                let z = wpos.z - range.min.z; */
            /* let mut i = 0;
            for x in 0..range.size().w {
                for y in 0..range.size().h {
                    for z in -1..range.size().d + 1 {
                        let wpos = range.min + Vec3::new(x, y, z);
                        let block = volume
                            .get(wpos)
                            .map(|b| *b)
                            // TODO: Replace with None or some other more reasonable value,
                            // since it's not clear this will work properly with liquid.
                            .unwrap_or(AIR); */
                    /* if let Ok(&block) = chonk.get(Vec3::new(x, y, z - z_diff)) */
                    let block_pos = Vec3::new(x, y, z - z_diff);
                    let block = /* if block_pos.z < 0 {
                        *chonk.below()
                    } else if block_pos.z >= z_max {
                        *chonk.above()
                    } else */{
                        let grp_id = common::terrain::TerrainSubChunk::grp_idx(block_pos) as usize;
                        let rel_id = common::terrain::TerrainSubChunk::rel_idx(block_pos) as usize;
                        flat_chunk[grp_id][rel_id]
                    };
                    /* let block = chonk.get(block_pos).copied().unwrap_or(AIR); */
                    {
                        *row |= if block.is_opaque() {
                            /* opaque_limits = opaque_limits
                                .map(|l| l.including(z))
                                .or_else(|| Some(Limits::from_value(z))); */
                            ALL_OPAQUE
                        } else if block.is_liquid() {
                            /* fluid_limits = fluid_limits
                                .map(|l| l.including(z))
                                .or_else(|| Some(Limits::from_value(z))); */
                            ALL_LIQUID
                        } else {
                            // Assume air
                            /* air_limits = air_limits
                                .map(|l| l.including(z))
                                .or_else(|| Some(Limits::from_value(z))); */
                            ALL_AIR
                        };
                        /*flat[i] = block*//*unsafe { flat.get_unchecked_mut(i) }*//*flat[i].write(block);*/
                        /* flat.write(block); */
                        *flat = block;
                    }
                    /* i += 1; */
                    /* }
                }
            } */
            /* flat */
            /* }); */
                /* }); */
                });
                // i += d;
                /* }); */
                });
                // i += x_off;
                i += hd as usize;
                });

                arena.reset();
            /* }); */
            });
            });

            // Compute limits (TODO: see if we can skip this, or make it more precise?).
            row_kinds.iter().enumerate().for_each(|(z, row)| {
                let z = z as i32 /* + intersection.min.z */- 1;
                if row & ALL_OPAQUE != 0 {
                    opaque_limits = opaque_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
                if row & ALL_LIQUID != 0 {
                    fluid_limits = fluid_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
                if row & ALL_AIR != 0 {
                    air_limits = air_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
            });
        }
        /* unsafe { flat.set_len((w * hd) as usize); } */
    } */
    {
        span!(_guard, "copy to flat array");
        let wh = w * h;
        {
            let mut arena = bumpalo::Bump::new();

            flat = vec![AIR; (d * wh) as usize];
            let row_kinds = &mut row_kinds[0..d as usize];
            let flat = &mut flat[0..(d * wh) as usize];
            let flat_range = Aabb {
                min: range.min - Vec3::new(0, 0, 1),
                max: range.max - Vec3::new(1, 1, 0),
            };
            let min_chunk_key = vol.pos_key(flat_range.min);
            let max_chunk_key = vol.pos_key(flat_range.max);
            (min_chunk_key.x..max_chunk_key.x + 1).for_each(|key_x| {
            (min_chunk_key.y..max_chunk_key.y + 1).for_each(|key_y| {
            let key = Vec2::new(key_x, key_y);
            let chonk = vol.get_key(key).expect("All keys in range must have chonks.");
                let chonk = &*chonk;
                let pos = vol.key_pos(key);
                let intersection_ = flat_range.intersection(Aabb {
                    min: pos.with_z(i32::MIN),
                    max: (pos + VolGrid2d::<V>::chunk_size().map(|e| e as i32) - 1).with_z(i32::MAX),
                });

                // Map intersection into chunk coordinates
                let x_diff = intersection_.min.x - flat_range.min.x;
                let y_diff = intersection_.min.y - flat_range.min.y;
                let z_diff = -range.min.z;
                /* let y_rem = flat_range.max.y - intersection_.max.y;
                let x_off = ((y_diff + y_rem) * d) as usize; */

                let intersection = Aabb {
                    min: VolGrid2d::<V>::chunk_offs(intersection_.min) + Vec3::new(0, 0, z_diff),
                    max: VolGrid2d::<V>::chunk_offs(intersection_.max) + Vec3::new(1, 1, z_diff + 1),
                };
                let z_diff = z_diff + chonk.get_min_z();
                let z_max = chonk.get_max_z() - chonk.get_min_z();
                let below = *chonk.below();

                let flat_chunk = chonk.make_flat(&arena);

                let min_z_ = z_diff - intersection.min.z;
                let max_z_ = z_max + z_diff - intersection.min.z;

                let row_fill = if below.is_opaque() {
                    ALL_OPAQUE
                } else if below.is_liquid() {
                    ALL_LIQUID
                } else {
                    ALL_AIR
                };

                let skip_count = min_z_.max(0);
                let take_count = (max_z_.min(d) - skip_count).max(0);
                let skip_count = skip_count as usize;
                let take_count = take_count as usize;

                row_kinds.iter_mut().take(skip_count).for_each(|row| {
                    *row |= row_fill;
                });
                row_kinds.iter_mut().skip(skip_count + take_count).for_each(|row| {
                    *row |= ALL_AIR;
                });

                // dbg!(pos, intersection_, intersection, range, flat_range, x_diff, y_diff, z_diff, y_rem, x_off, i);
                flat.chunks_exact_mut(wh as usize).take(skip_count).for_each(|flat| {
                flat.chunks_exact_mut(w as usize).skip(y_diff as usize).take((intersection.max.y - intersection.min.y) as usize).for_each(|flat| {
                flat.into_iter().skip(x_diff as usize).take((intersection.max.x - intersection.min.x) as usize).for_each(|flat| {
                    *flat = below;
                });
                });
                });

                flat.chunks_exact_mut(wh as usize).zip(row_kinds.into_iter()).enumerate().skip(skip_count).take(take_count).for_each(|(z, (flat, row_))| {
                let mut row = *row_;
                let z = z as i32 + intersection.min.z - z_diff;
                flat.chunks_exact_mut(w as usize).skip(y_diff as usize).enumerate().take((intersection.max.y - intersection.min.y) as usize).for_each(|(y, flat)| {
                let y = y as i32 + intersection.min.y;
                flat.into_iter().skip(x_diff as usize).enumerate().take((intersection.max.x - intersection.min.x) as usize).for_each(|(x, flat)| {
                    let x = x as i32 + intersection.min.x;
                    let block_pos = Vec3::new(x, y, z);
                    let block = {
                        let grp_id = common::terrain::TerrainSubChunk::grp_idx(block_pos) as usize;
                        let rel_id = common::terrain::TerrainSubChunk::rel_idx(block_pos) as usize;
                        flat_chunk[grp_id][rel_id]
                    };
                    {
                        row |= if block.is_opaque() {
                            ALL_OPAQUE
                        } else if block.is_liquid() {
                            ALL_LIQUID
                        } else {
                            ALL_AIR
                        };
                        *flat = block;
                    }
                });
                });
                *row_ = row;
                });
                /* (intersection.min.z..intersection.max.z).for_each(|z| {
                let flat = &mut flat[i..i + /*intersection.size().y * intersection.size().z*/hd_];
                flat.chunks_exact_mut(d as usize).enumerate().for_each(|(y, flat)| {
                let y = y as i32 + intersection.min.y;
                flat[0..skip_count].fill(below);
                flat.into_iter().zip(row_kinds.into_iter()).enumerate().skip(skip_count).take(take_count).for_each(|(z, (flat, row))| {
                let z = z as i32 + intersection.min.z;
                /* let mut i = ((x_diff + (x - intersection.min.x)) * hd + (y_diff + (y - intersection.min.y)) * d + (z - intersection.min.z)) as usize; */
                    let block_pos = Vec3::new(x, y, z - z_diff);
                    let block = {
                        let grp_id = common::terrain::TerrainSubChunk::grp_idx(block_pos) as usize;
                        let rel_id = common::terrain::TerrainSubChunk::rel_idx(block_pos) as usize;
                        flat_chunk[grp_id][rel_id]
                    };
                    {
                        *row |= if block.is_opaque() {
                            ALL_OPAQUE
                        } else if block.is_liquid() {
                            ALL_LIQUID
                        } else {
                            ALL_AIR
                        };
                        *flat = block;
                    }
                });
                });
                i += hd as usize;
                }); */

                arena.reset();
            });
            });

            // Compute limits (TODO: see if we can skip this, or make it more precise?).
            row_kinds.iter().enumerate().for_each(|(z, row)| {
                let z = z as i32 /* + intersection.min.z */- 1;
                if row & ALL_OPAQUE != 0 {
                    opaque_limits = opaque_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
                if row & ALL_LIQUID != 0 {
                    fluid_limits = fluid_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
                if row & ALL_AIR != 0 {
                    air_limits = air_limits
                        .map(|l| l.including(z))
                        .or_else(|| Some(Limits::from_value(z)));
                }
            });
        }
        /* unsafe { flat.set_len((w * hd) as usize); } */
    }

    // Constrain iterated area
    let mut opaque_mesh = Mesh::new();
    let mut fluid_mesh = Mesh::new();
    let (z_start, z_end) = match (air_limits, fluid_limits, opaque_limits) {
        (Some(air), Some(fluid), Some(opaque)) => {
            let air_fluid = air.intersection(fluid);
            /* if let Some(intersection) = air_fluid.filter(|limits| limits.min + 1 == limits.max) {
                // If there is a planar air-fluid boundary, just draw it directly and avoid
                // redundantly meshing the whole fluid volume, then interect the ground-fluid
                // and ground-air meshes to make sure we don't miss anything.
                either_with(air.intersection(opaque), fluid.intersection(opaque), Limits::union)
            } else */{
                // Otherwise, do a normal three-way intersection.
                air.three_way_intersection(fluid, opaque)
            }
        },
        (Some(air), Some(fluid), None) => air.intersection(fluid),
        (Some(air), None, Some(opaque)) => air.intersection(opaque),
        (None, Some(fluid), Some(opaque)) => fluid.intersection(opaque),
        // No interfaces (Note: if there are multiple fluid types this could change)
        (Some(_), None, None) | (None, Some(_), None) | (None, None, Some(_)) => None,
        (None, None, None) => {
            error!("Impossible unless given an input AABB that has a height of zero");
            None
        },
    }
    .map_or((0, 0), |limits| {
        let (start, end) = limits.into_tuple();
        let start = start.max(0);
        let end = end.min(range.size().d - 1).max(start);
        (start, end)
    });

    // Calculate chunk lighting (sunlight defaults to 1.0, glow to 0.0)
    let mut glow_range = range;
    if let Some(opaque) = opaque_limits {
        glow_range.min.z = z_start.max(opaque.min) + range.min.z;
        glow_range.max.z = (z_end.min(opaque.max) + range.min.z).max(glow_range.min.z);
    }
    // Find blocks that should glow
    // TODO: Search neighbouring chunks too!
    let mut glow_block_min = glow_range.max.z;
    let mut glow_block_max = glow_range.min.z;
    let mut glow_blocks = boi.lights
        .iter()
        .map(|(pos, glow)| {
            let pos_z = pos.z.clamped(glow_range.min.z, glow_range.max.z);
            let glow = (i32::from(*glow) - (pos_z - pos.z).abs()).max(0);
            glow_block_min = glow_block_min.min(pos_z - glow);
            glow_block_max = glow_block_max.max(pos_z + glow);
            ((*pos + glow_range.min.xy()).with_z(pos_z), glow as u8)
        })
        // FIXME: Why is Rust forcing me to collect to Vec here?
        .collect::<Vec<_>>();
    glow_range.min.z = glow_block_min.clamped(glow_range.min.z, glow_range.max.z);
    glow_range.max.z = glow_block_max.clamped(glow_range.min.z, glow_range.max.z);
    /* if glow_range.min.z != glow_range.max.z {
        println!("{:?}", glow_range);
    } */

    let mut light_range = glow_range;
    light_range.min.z = light_range.max.z;

    /* // Sort glowing blocks in decreasing order by glow strength.  This makes it somewhat less
    // likely that a smaller glow will have to be drawn.
    glow_blocks.sort_unstable_by_key(|(_, glow)| core::cmp::Reverse(*glow)); */

    /*  DefaultVolIterator::new(vol, range.min - MAX_LIGHT_DIST, range.max + MAX_LIGHT_DIST)
    .filter_map(|(pos, block)| block.get_glow().map(|glow| (pos, glow))); */

    /* let mut glow_blocks = Vec::new(); */

    /* // TODO: This expensive, use BlocksOfInterest instead
    let mut volume = vol.cached();
    for x in -MAX_LIGHT_DIST..range.size().w + MAX_LIGHT_DIST {
        for y in -MAX_LIGHT_DIST..range.size().h + MAX_LIGHT_DIST {
            for z in -1..range.size().d + 1 {
                let wpos = range.min + Vec3::new(x, y, z);
                volume
                    .get(wpos)
                    .ok()
                    .and_then(|b| b.get_glow())
                    .map(|glow| glow_blocks.push((wpos, glow)));
            }
        }
    } */
    let light = calc_light(true, SUNLIGHT, light_range, range, vol, core::iter::empty(), &flat, (w, h, d));
    let glow = calc_light(false, 0, glow_range, range, vol, glow_blocks.into_iter(), &flat, (w, h, d));

    let max_size = max_texture_size;
    assert!(z_end >= z_start);

    let flat_get = flat_get(&flat, w, h, d);
    let get_color =
        #[inline(always)] |_: &mut (), pos: Vec3<i32>| flat_get(pos).get_color().unwrap_or_else(Rgb::zero);
    let get_light = #[inline(always)] |_: &mut (), pos: Vec3<i32>| {
        if flat_get(pos).is_opaque() {
            0.0
        } else {
            light(pos + range.min)
        }
    };
    let get_ao = #[inline(always)] |_: &mut (), pos: Vec3<i32>| {
        if flat_get(pos).is_opaque() { 0.0 } else { 1.0 }
    };
    let get_glow = #[inline(always)] |_: &mut (), pos: Vec3<i32>| glow(pos + range.min);
    let get_opacity = #[inline(always)] |_: &mut (), pos: Vec3<i32>| !flat_get(pos).is_opaque();
    let should_draw = #[inline(always)] |_: &mut (), /*pos*/_: Vec3<i32>, from: Block, to: Block,/* delta: Vec3<i32>,*/ _uv: Vec2<Vec3<i32>>| {
        should_draw_greedy(/*pos, */from, to/*, delta, #[inline(always)] |pos| flat_get(pos) */)
    };

    let mut greedy =
        GreedyMesh::<guillotiere::SimpleAtlasAllocator>::new(max_size, greedy::terrain_config());
    let greedy_size = Vec3::new(range.size().w - 2, range.size().h - 2, z_end - z_start + 1);
    let mesh_delta = Vec3::new(0.0, 0.0, (z_start + range.min.z) as f32);
    let max_bounds: Vec3<f32> = greedy_size.as_::<f32>();
    let mut do_draw_greedy = #[inline(always)] |z_start: i32, z_end: i32| {
    // dbg!(range.min, z_start, z_end);
    let greedy_size = Vec3::new(range.size().w - 2, range.size().h - 2, z_end - z_start + 1);
    // NOTE: Terrain sizes are limited to 32 x 32 x 16384 (to fit in 24 bits: 5 + 5
    // + 14). FIXME: Make this function fallible, since the terrain
    // information might be dynamically generated which would make this hard
    // to enforce.
    assert!(greedy_size.x <= 32 && greedy_size.y <= 32 && greedy_size.z <= 16384);
    // NOTE: Cast is safe by prior assertion on greedy_size; it fits into a u16,
    // which always fits into a f32.
    // NOTE: Cast is safe by prior assertion on greedy_size; it fits into a u16,
    // which always fits into a usize.
    let greedy_size = greedy_size.as_::<usize>();
    let greedy_size_cross = Vec3::new(greedy_size.x - 1, greedy_size.y - 1, greedy_size.z);
    let draw_delta = Vec3::new(1, 1, z_start);

    // NOTE: Conversion to f32 is fine since this i32 is actually in bounds for u16.
    let mesh_delta = Vec3::new(0.0, 0.0, (z_start + range.min.z) as f32);
    let create_opaque =
        #[inline(always)] |atlas_pos, pos: Vec3<f32>, norm, meta| TerrainVertex::new(atlas_pos, pos + mesh_delta, norm, meta);
    let create_transparent = #[inline(always)] |_atlas_pos: Vec2<u16>, pos: Vec3<f32>, norm: Vec3<f32>| FluidVertex::new(pos + mesh_delta, norm);

    greedy.push(GreedyConfig {
        data: (),
        draw_delta,
        greedy_size,
        greedy_size_cross,
        get_vox: #[inline(always)] |_: &mut (), pos| flat_get(pos),
        get_ao,
        get_light,
        get_glow,
        get_opacity,
        should_draw,
        push_quad: #[inline(always)] |atlas_origin, dim, origin, draw_dim, norm, meta: &FaceKind| match meta {
            FaceKind::Opaque(meta) => {
                opaque_mesh.push_quad(greedy::create_quad(
                    atlas_origin,
                    dim,
                    origin,
                    draw_dim,
                    norm,
                    meta,
                    |atlas_pos, pos, norm, &meta| create_opaque(atlas_pos, pos, norm, meta),
                ));
            },
            FaceKind::Fluid => {
                fluid_mesh.push_quad(greedy::create_quad(
                    atlas_origin,
                    dim,
                    origin,
                    draw_dim,
                    norm,
                    &(),
                    |atlas_pos, pos, norm, &_meta| create_transparent(atlas_pos, pos, norm),
                ));
            },
        },
        make_face_texel: #[inline(always)] |data: &mut (), pos, light, glow, ao| {
            TerrainVertex::make_col_light(light, glow, get_color(data, pos), ao)
        },
    });
    };

    let mut z_start = z_start;
    let mut row_iter = row_kinds.iter().enumerate();
    let mut needs_draw = false;
    row_kinds.array_windows().enumerate().skip(z_start as usize).take((z_end - z_start + 1) as usize).for_each(|(z, &[from_row, to_row])| {
        let z = z as i32;
        // Evaluate a "canonicalized" greedy mesh algorithm on this pair of row kinds, to see if we're
        // about to switch (requiring us to draw a surface).
        let from = match from_row {
            ALL_AIR => Some(AIR),
            ALL_LIQUID => Some(LIQUID),
            ALL_OPAQUE => Some(OPAQUE),
            _ => None,
        };
        let to = match to_row {
            ALL_AIR => Some(AIR),
            ALL_LIQUID => Some(LIQUID),
            ALL_OPAQUE => Some(OPAQUE),
            _ => None,
        };
        // There are two distinct cases:
        let (from, to) = match from.zip(to) {
            None => {
                // At least one of the two rows is not homogeneous.
                if !needs_draw {
                    // The from is homogeneous (since !needs_draw), but the to is not.  We should
                    // start a new draw without drawing the old volume.
                    z_start = z;
                    needs_draw = true;
                }
                // Otherwise, we were in the middle of drawing the previous row, so we just extend
                // the current draw.
                return;
            },
            Some(pair) => pair,
        };
        let old_needs_draw = needs_draw;
        // The from *and* to are both homogeneous, so we can compute whether we should draw
        // a surface between them.
        needs_draw = should_draw_greedy(from, to).is_some();
        if needs_draw == old_needs_draw {
            // We continue the current draw (or nondraw).
            return;
        }
        if old_needs_draw {
            // old_needs_draw is true, so we need to start a fresh draw and end an earlier draw,
            // drawing the existing volume (needs_draw is false).
            do_draw_greedy(z_start, z - 1);
        }
        // We always must start a fresh draw.
        z_start = z;
    });
    // Finally, draw any remaining terrain, if necessary.
    if needs_draw {
        /* if z_start != z_end {
            dbg!(range.min, z_start, z_end);
        } */
        do_draw_greedy(z_start, z_end);
    }

    let min_bounds = mesh_delta;
    let bounds = Aabb {
        min: min_bounds,
        max: max_bounds + min_bounds,
    };
    // WGPU requires this alignment.
    let (col_lights, col_lights_size) = greedy.finalize(
        Vec2::new((wgpu::COPY_BYTES_PER_ROW_ALIGNMENT / 4) as u16, 1),
    );

    (
        opaque_mesh,
        fluid_mesh,
        Mesh::new(),
        (
            bounds,
            (col_lights, col_lights_size),
            Arc::new(light),
            Arc::new(glow),
        ),
    )
}

/// NOTE: Make sure to reflect any changes to how meshing is performanced in
/// [scene::terrain::Terrain::skip_remesh].
#[inline(always)]
fn should_draw_greedy(
    /* pos: Vec3<i32>, */
    from: Block,
    to: Block,
    /* delta: Vec3<i32>,
    flat_get: impl Fn(Vec3<i32>) -> Block, */
) -> Option<(bool, FaceKind)> {
    /* let from = flat_get(pos - delta);
    let to = flat_get(pos); */
    // Don't use `is_opaque`, because it actually refers to light transmission
    /* let from = from.kind() as u8 & 0xF;
    let to = to.kind() as u8 & 0xF;
    (from ^ to) | ((from.overflowing_sub(1) > to.overflowing_sub(1)) as u8 << 2) */
    use common::terrain::BlockKind;
    match (from.kind(), to.kind()) {
        (BlockKind::Air, BlockKind::Water) => Some((false, FaceKind::Fluid)),
        (BlockKind::Water, BlockKind::Air) => Some((true, FaceKind::Fluid)),
        (BlockKind::Air, BlockKind::Air) | (BlockKind::Water, BlockKind::Water) => None,
        (BlockKind::Air, _) => Some((false, FaceKind::Opaque(false))),
        (_, BlockKind::Air) => Some((true, FaceKind::Opaque(false))),
        (BlockKind::Water, _) => Some((false, FaceKind::Opaque(true))),
        (_, BlockKind::Water) => Some((true, FaceKind::Opaque(true))),
        _ => None,
    }
    /* let from_filled = from.is_filled();
    if from_filled == to.is_filled() {
        // Check the interface of liquid and non-tangible non-liquid (e.g. air).
        if from_filled {
            None
        } else {
            let from_liquid = /*from.is_liquid()*/!from.is_air();
            if from_liquid == /*to.is_liquid()*/!to.is_air()/*from.is_filled() || to.is_filled()*//* from_filled */ {
                None
            } else {
                // While liquid is not culled, we still try to keep a consistent orientation as
                // we do for land; if going from liquid to non-liquid,
                // forwards-facing; otherwise, backwards-facing.
                Some((from_liquid, FaceKind::Fluid))
            }
        }
    } else {
        // If going from unfilled to filled, backward facing; otherwise, forward
        // facing.  Also, if either from or to is fluid, set the meta accordingly.
        Some((
            from_filled,
            FaceKind::Opaque(if from_filled {
                /* to.is_liquid() */!to.is_air()
            } else {
                /* from.is_liquid() */!from.is_air()
            }),
        ))
    } */
}

/// 1D Aabr
#[derive(Copy, Clone, Debug)]
struct Limits {
    min: i32,
    max: i32,
}

impl Limits {
    fn from_value(v: i32) -> Self { Self { min: v, max: v } }

    fn including(mut self, v: i32) -> Self {
        if v < self.min {
            self.min = v
        } else if v > self.max {
            self.max = v
        }
        self
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    // Find limits that include the overlap of the two
    fn intersection(self, other: Self) -> Option<Self> {
        // Expands intersection by 1 since that fits our use-case
        // (we need to get blocks on either side of the interface)
        let min = self.min.max(other.min) - 1;
        let max = self.max.min(other.max) + 1;

        (min < max).then_some(Self { min, max })
    }

    // Find limits that include any areas of overlap between two of the three
    fn three_way_intersection(self, two: Self, three: Self) -> Option<Self> {
        let intersection = self.intersection(two);
        let intersection = either_with(self.intersection(three), intersection, Limits::union);
        either_with(two.intersection(three), intersection, Limits::union)
    }

    fn into_tuple(self) -> (i32, i32) { (self.min, self.max) }
}
