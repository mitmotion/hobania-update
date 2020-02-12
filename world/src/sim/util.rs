use super::WORLD_SIZE;
use bitvec::prelude::{bitbox, bitvec, BitBox};
use common::{terrain::TerrainChunkSize, vol::RectVolSize};
use noise::{MultiFractal, NoiseFn, Perlin, Point2, Point3, Point4, Seedable};
use num::Float;
use rayon::prelude::*;
use roots::{find_roots_cubic, find_roots_quadratic, find_roots_quartic};
use std::{f32, f64, ops::Mul, u32};
use vek::*;

/// Calculates the smallest distance along an axis (x, y) from an edge of
/// the world.  This value is maximal at WORLD_SIZE / 2 and minimized at the
/// extremes (0 or WORLD_SIZE on one or more axes).  It then divides the
/// quantity by cell_size, so the final result is 1 when we are not in a cell
/// along the edge of the world, and ranges between 0 and 1 otherwise (lower
/// when the chunk is closer to the edge).
pub fn map_edge_factor(posi: usize) -> f32 {
    uniform_idx_as_vec2(posi)
        .map2(WORLD_SIZE.map(|e| e as i32), |e, sz| {
            (sz / 2 - (e - sz / 2).abs()) as f32 / (16.0 / 1024.0 * sz as f32)
        })
        .reduce_partial_min()
        .max(0.0)
        .min(1.0)
}

/// Computes the cumulative distribution function of the weighted sum of k
/// independent, uniformly distributed random variables between 0 and 1.  For
/// each variable i, we use weights[i] as the weight to give samples[i] (the
/// weights should all be positive).
///
/// If the precondition is met, the distribution of the result of calling this
/// function will be uniformly distributed while preserving the same information
/// that was in the original average.
///
/// For N > 33 the function will no longer return correct results since we will
/// overflow u32.
///
/// NOTE:
///
/// Per [1], the problem of determing the CDF of
/// the sum of uniformly distributed random variables over *different* ranges is
/// considerably more complicated than it is for the same-range case.
/// Fortunately, it also provides a reference to [2], which contains a complete
/// derivation of an exact rule for the density function for this case.  The CDF
/// is just the integral of the cumulative distribution function [3],
/// which we use to convert this into a CDF formula.
///
/// This allows us to sum weighted, uniform, independent random variables.
///
/// At some point, we should probably contribute this back to stats-rs.
///
/// 1. https://www.r-bloggers.com/sums-of-random-variables/,
/// 2. Sadooghi-Alvandi, S., A. Nematollahi, & R. Habibi, 2009.
///    On the Distribution of the Sum of Independent Uniform Random Variables.
///    Statistical Papers, 50, 171-175.
/// 3. hhttps://en.wikipedia.org/wiki/Cumulative_distribution_function
pub fn cdf_irwin_hall<const N: usize>(weights: &[f32; N], samples: [f32; N]) -> f32 {
    // Let J_k = {(j_1, ... , j_k) : 1 ≤ j_1 < j_2 < ··· < j_k ≤ N }.
    //
    // Let A_N = Π{k = 1 to n}a_k.
    //
    // The density function for N ≥ 2 is:
    //
    //   1/(A_N * (N - 1)!) * (x^(N-1) + Σ{k = 1 to N}((-1)^k *
    //   Σ{(j_1, ..., j_k) ∈ J_k}(max(0, x - Σ{l = 1 to k}(a_(j_l)))^(N - 1))))
    //
    // So the cumulative distribution function is its integral, i.e. (I think)
    //
    // 1/(product{k in A}(k) * N!) * (x^N + sum(k in 1 to N)((-1)^k *
    // sum{j in Subsets[A, {k}]}(max(0, x - sum{l in j}(l))^N)))
    //
    // which is also equivalent to
    //
    //   (letting B_k = { a in Subsets[A, {k}] : sum {l in a} l }, B_(0,1) = 0 and
    //            H_k = { i : 1 ≤ 1 ≤ N! / (k! * (N - k)!) })
    //
    //   1/(product{k in A}(k) * N!) * sum(k in 0 to N)((-1)^k *
    //   sum{l in H_k}(max(0, x - B_(k,l))^N))
    //
    // We should be able to iterate through the whole power set
    // instead, and figure out K by calling count_ones(), so we can compute the
    // result in O(2^N) iterations.
    let x: f64 = weights
        .iter()
        .zip(samples.iter())
        .map(|(&weight, &sample)| weight as f64 * sample as f64)
        .sum();

    let mut y = 0.0f64;
    for subset in 0u32..(1 << N) {
        // Number of set elements
        let k = subset.count_ones();
        // Add together exactly the set elements to get B_subset
        let z = weights
            .iter()
            .enumerate()
            .filter(|(i, _)| subset & (1 << i) as u32 != 0)
            .map(|(_, &k)| k as f64)
            .sum::<f64>();
        // Compute max(0, x - B_subset)^N
        let z = (x - z).max(0.0).powi(N as i32);
        // The parity of k determines whether the sum is negated.
        y += if k & 1 == 0 { z } else { -z };
    }

    // Divide by the product of the weights.
    y /= weights.iter().map(|&k| k as f64).product::<f64>();

    // Remember to multiply by 1 / N! at the end.
    (y / (1..(N as i32) + 1).product::<i32>() as f64) as f32
}

/// First component of each element of the vector is the computed CDF of the
/// noise function at this index (i.e. its position in a sorted list of value
/// returned by the noise function applied to every chunk in the game).  Second
/// component is the cached value of the noise function that generated the
/// index.
///
/// NOTE: Length should always be WORLD_SIZE.x * WORLD_SIZE.y.
pub type InverseCdf<F = f32> = Box<[(f32, F)]>;

/// Computes the position Vec2 of a SimChunk from an index, where the index was
/// generated by uniform_noise.
pub fn uniform_idx_as_vec2(idx: usize) -> Vec2<i32> {
    Vec2::new((idx % WORLD_SIZE.x) as i32, (idx / WORLD_SIZE.x) as i32)
}

/// Computes the index of a Vec2 of a SimChunk from a position, where the index
/// is generated by uniform_noise.  NOTE: Both components of idx should be
/// in-bounds!
pub fn vec2_as_uniform_idx(idx: Vec2<i32>) -> usize {
    (idx.y as usize * WORLD_SIZE.x + idx.x as usize) as usize
}

/// Compute inverse cumulative distribution function for arbitrary function f,
/// the hard way.  We pre-generate noise values prior to worldgen, then sort
/// them in order to determine the correct position in the sorted order.  That
/// lets us use `(index + 1) / (WORLDSIZE.y * WORLDSIZE.x)` as a uniformly
/// distributed (from almost-0 to 1) regularization of the chunks.  That is, if
/// we apply the computed "function" F⁻¹(x, y) to (x, y) and get out p, it means
/// that approximately (100 * p)% of chunks have a lower value for F⁻¹ than p.
/// The main purpose of doing this is to make sure we are using the entire range
/// we want, and to allow us to apply the numerous results about distributions
/// on uniform functions to the procedural noise we generate, which lets us much
/// more reliably control the *number* of features in the world while still
/// letting us play with the *shape* of those features, without having arbitrary
/// cutoff points / discontinuities (which tend to produce ugly-looking /
/// unnatural terrain).
///
/// As a concrete example, before doing this it was very hard to tweak humidity
/// so that either most of the world wasn't dry, or most of it wasn't wet, by
/// combining the billow noise function and the computed altitude.  This is
/// because the billow noise function has a very unusual distribution that is
/// heavily skewed towards 0.  By correcting for this tendency, we can start
/// with uniformly distributed billow noise and altitudes and combine them to
/// get uniformly distributed humidity, while still preserving the existing
/// shapes that the billow noise and altitude functions produce.
///
/// f takes an index, which represents the index corresponding to this chunk in
/// any any SimChunk vector returned by uniform_noise, and (for convenience) the
/// float-translated version of those coordinates.
/// f should return a value with no NaNs.  If there is a NaN, it will panic.
/// There are no other conditions on f.  If f returns None, the value will be
/// set to NaN, and will be ignored for the purposes of computing the uniform
/// range.
///
/// Returns a vec of (f32, f32) pairs consisting of the percentage of chunks
/// with a value lower than this one, and the actual noise value (we don't need
/// to cache it, but it makes ensuring that subsequent code that needs the noise
/// value actually uses the same one we were using here easier).  Also returns
/// the "inverted index" pointing from a position to a noise.
pub fn uniform_noise<F: Float + Send>(
    f: impl Fn(usize, Vec2<f64>) -> Option<F> + Sync,
) -> (InverseCdf<F>, Box<[(usize, F)]>) {
    let mut noise = (0..WORLD_SIZE.x * WORLD_SIZE.y)
        .into_par_iter()
        .filter_map(|i| {
            f(
                i,
                (uniform_idx_as_vec2(i) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32))
                    .map(|e| e as f64),
            )
            .map(|res| (i, res))
        })
        .collect::<Vec<_>>();

    // sort_unstable_by is equivalent to sort_by here since we include a unique
    // index in the comparison.  We could leave out the index, but this might
    // make the order not reproduce the same way between different versions of
    // Rust (for example).
    noise.par_sort_unstable_by(|f, g| (f.1, f.0).partial_cmp(&(g.1, g.0)).unwrap());

    // Construct a vector that associates each chunk position with the 1-indexed
    // position of the noise in the sorted vector (divided by the vector length).
    // This guarantees a uniform distribution among the samples (excluding those
    // that returned None, which will remain at zero).
    let mut uniform_noise = vec![(0.0, F::nan()); WORLD_SIZE.x * WORLD_SIZE.y].into_boxed_slice();
    // NOTE: Consider using try_into here and elsewhere in this function, since
    // i32::MAX technically doesn't fit in an f32 (even if we should never reach
    // that limit).
    let total = noise.len() as f32;
    for (noise_idx, &(chunk_idx, noise_val)) in noise.iter().enumerate() {
        uniform_noise[chunk_idx] = ((1 + noise_idx) as f32 / total, noise_val);
    }
    (uniform_noise, noise.into_boxed_slice())
}

/// Iterate through all cells adjacent and including four chunks whose top-left
/// point is posi. This isn't just the immediate neighbors of a chunk plus the
/// center, because it is designed to cover neighbors of a point in the chunk's
/// "interior."
///
/// This is what's used during cubic interpolation, for example, as it
/// guarantees that for any point between the given chunk (on the top left) and
/// its top-right/down-right/down neighbors, the twelve chunks surrounding this
/// box (its "perimeter") are also inspected.
pub fn local_cells(posi: usize) -> impl Clone + Iterator<Item = usize> {
    let pos = uniform_idx_as_vec2(posi);
    // NOTE: want to keep this such that the chunk index is in ascending order!
    let grid_size = 3i32;
    let grid_bounds = 2 * grid_size + 1;
    (0..grid_bounds * grid_bounds)
        .into_iter()
        .map(move |index| {
            Vec2::new(
                pos.x + (index % grid_bounds) - grid_size,
                pos.y + (index / grid_bounds) - grid_size,
            )
        })
        .filter(|pos| {
            pos.x >= 0 && pos.y >= 0 && pos.x < WORLD_SIZE.x as i32 && pos.y < WORLD_SIZE.y as i32
        })
        .map(vec2_as_uniform_idx)
}

// NOTE: want to keep this such that the chunk index is in ascending order!
pub const NEIGHBOR_DELTA: [(i32, i32); 8] = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Iterate through all cells adjacent to a chunk.
pub fn neighbors(posi: usize) -> impl Clone + Iterator<Item = usize> {
    let pos = uniform_idx_as_vec2(posi);
    NEIGHBOR_DELTA
        .iter()
        .map(move |&(x, y)| Vec2::new(pos.x + x, pos.y + y))
        .filter(|pos| {
            pos.x >= 0 && pos.y >= 0 && pos.x < WORLD_SIZE.x as i32 && pos.y < WORLD_SIZE.y as i32
        })
        .map(vec2_as_uniform_idx)
}

// Note that we should already have okay cache locality since we have a grid.
pub fn uphill<'a>(dh: &'a [isize], posi: usize) -> impl Clone + Iterator<Item = usize> + 'a {
    neighbors(posi).filter(move |&posj| dh[posj] == posi as isize)
}

/// Compute the neighbor "most downhill" from all chunks.
///
/// TODO: See if allocating in advance is worthwhile.
pub fn downhill<F: Float>(
    h: impl Fn(usize) -> F + Sync,
    is_ocean: impl Fn(usize) -> bool + Sync,
) -> Box<[isize]> {
    // Constructs not only the list of downhill nodes, but also computes an ordering
    // (visiting nodes in order from roots to leaves).
    (0..WORLD_SIZE.x * WORLD_SIZE.y)
        .into_par_iter()
        .map(|posi| {
            let nh = h(posi);
            if is_ocean(posi) {
                -2
            } else {
                let mut best = -1;
                let mut besth = nh;
                for nposi in neighbors(posi) {
                    let nbh = h(nposi);
                    if nbh < besth {
                        besth = nbh;
                        best = nposi as isize;
                    }
                }
                best
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

/// Find all ocean tiles from a height map, using an inductive definition of
/// ocean as one of:
/// - posi is at the side of the world (map_edge_factor(posi) == 0.0)
/// - posi has a neighboring ocean tile, and has a height below sea level
///   (oldh(posi) <= 0.0).
pub fn get_oceans<F: Float>(oldh: impl Fn(usize) -> F + Sync) -> BitBox {
    // We can mark tiles as ocean candidates by scanning row by row, since the top
    // edge is ocean, the sides are connected to it, and any subsequent ocean
    // tiles must be connected to it.
    let mut is_ocean = bitbox![0; WORLD_SIZE.x * WORLD_SIZE.y];
    let mut stack = Vec::new();
    let mut do_push = |pos| {
        let posi = vec2_as_uniform_idx(pos);
        if oldh(posi) <= F::zero() {
            stack.push(posi);
        }
    };
    for x in 0..WORLD_SIZE.x as i32 {
        do_push(Vec2::new(x, 0));
        do_push(Vec2::new(x, WORLD_SIZE.y as i32 - 1));
    }
    for y in 1..WORLD_SIZE.y as i32 - 1 {
        do_push(Vec2::new(0, y));
        do_push(Vec2::new(WORLD_SIZE.x as i32 - 1, y));
    }
    while let Some(chunk_idx) = stack.pop() {
        // println!("Ocean chunk {:?}: {:?}", uniform_idx_as_vec2(chunk_idx),
        // oldh(chunk_idx));
        if *is_ocean.at(chunk_idx) {
            continue;
        }
        *is_ocean.at(chunk_idx) = true;
        stack.extend(neighbors(chunk_idx).filter(|&neighbor_idx| {
            // println!("Ocean neighbor: {:?}: {:?}", uniform_idx_as_vec2(neighbor_idx),
            // oldh(neighbor_idx));
            oldh(neighbor_idx) <= F::zero()
        }));
    }
    is_ocean
}

pub fn river_spline_coeffs(
    // _sim: &WorldSim,
    chunk_pos: Vec2<f64>,
    spline_derivative: Vec2<f32>,
    downhill_pos: Vec2<f64>,
) -> Vec3<Vec2<f64>> {
    let dxy = downhill_pos - chunk_pos;
    // Since all splines have been precomputed, we don't have to do that much work
    // to evaluate the spline.  The spline is just ax^2 + bx + c = 0, where
    //
    // a = dxy - chunk.river.spline_derivative
    // b = chunk.river.spline_derivative
    // c = chunk_pos
    let spline_derivative = spline_derivative.map(|e| e as f64);
    Vec3::new(dxy - spline_derivative, spline_derivative, chunk_pos)
}

/// Find the nearest point from a quadratic spline to this point (in terms of t,
/// the "distance along the curve" by which our spline is parameterized).  Note
/// that if t < 0.0 or t >= 1.0, we probably shouldn't be considered "on the
/// curve"... hopefully this works out okay and gives us what we want (a
/// river that extends outwards tangent to a quadratic curve, with width
/// configured by distance along the line).
pub fn quadratic_nearest_point(
    spline: &Vec3<Vec2<f64>>,
    point: Vec2<f64>,
) -> Option<(f64, Vec2<f64>, f64)> {
    let a = spline.z.x;
    let b = spline.y.x;
    let c = spline.x.x;
    let d = point.x;
    let e = spline.z.y;
    let f = spline.y.y;
    let g = spline.x.y;
    let h = point.y;
    // This is equivalent to solving the following cubic equation (derivation is a
    // bit annoying):
    //
    // A = 2(c^2 + g^2)
    // B = 3(b * c + g * f)
    // C = ((a - d) * 2 * c + b^2 + (e - h) * 2 * g + f^2)
    // D = ((a - d) * b + (e - h) * f)
    //
    // Ax³ + Bx² + Cx + D = 0
    //
    // Once solved, this yield up to three possible values for t (reflecting minimal
    // and maximal values).  We should choose the minimal such real value with t
    // between 0.0 and 1.0.  If we fall outside those bounds, then we are
    // outside the spline and return None.
    let a_ = (c * c + g * g) * 2.0;
    let b_ = (b * c + g * f) * 3.0;
    let a_d = a - d;
    let e_h = e - h;
    let c_ = a_d * c * 2.0 + b * b + e_h * g * 2.0 + f * f;
    let d_ = a_d * b + e_h * f;
    let roots = find_roots_cubic(a_, b_, c_, d_);
    let roots = roots.as_ref();

    let min_root = roots
        .into_iter()
        .copied()
        .filter_map(|root| {
            let river_point = spline.x * root * root + spline.y * root + spline.z;
            let river_zero = spline.z;
            let river_one = spline.x + spline.y + spline.z;
            if root > 0.0 && root < 1.0 {
                Some((root, river_point))
            } else if
            /*root <= 0.0 && */
            river_point.distance_squared(river_zero) < /*0.5*/1e-2 {
                Some((/*root*/ 0.0, /*river_point*/ river_zero))
            } else if
            /*root >= 1.0 && */
            river_point.distance_squared(river_one) < /*0.5*/1e-2 {
                Some((/*root*/ 1.0, /*river_point*/ river_one))
            } else {
                None
            }
        })
        .map(|(root, river_point)| {
            let river_distance = river_point.distance_squared(point);
            (root, river_point, river_distance)
        })
        // In the (unlikely?) case that distances are equal, prefer the earliest point along the
        // river.
        .min_by(|&(ap, _, a), &(bp, _, b)| {
            (a, /*ap < 0.0 || ap > 1.0, */ ap)
                .partial_cmp(&(b, /*bp < 0.0 || bp > 1.0, */ bp))
                .unwrap()
        });
    min_root
}

/// Transform a line in parametric form (f(t) = b t + c)) to implicit form
/// (f(x, y) = A x + B y + C)
fn implicitize_line(coeffs: &Vec2<Vec2<f64>>) -> [f64; 3] {
    // b_y * x - b_x * y + (b_x * c_y - b_y * c_x) = 0
    let b_x = coeffs.x.x;
    let c_x = coeffs.y.x;
    let b_y = coeffs.x.y;
    let c_y = coeffs.y.y;
    [b_y, -b_x, b_x * c_y - b_y * c_x]
}

/// Transform a quadratic curve in parametric form (f(t) = a t^2 + b t + c) to
/// implicit form (f(x, y) = A x ^2 + B * x * y + C * y^2 + D * x + E * y + F =
/// 0)
pub fn implicitize_quadratic(coeffs: &Vec3<Vec2<f64>>) -> [f64; 6] {
    // (a_y^2) * x^2 +
    // (-2 * a_x * a_y) * x * y +
    // (a_x^2) * y^2 +
    // (a_y * b_x * b_y + 2 * a_x * a_y * c_y - (2 * a_y^2 * c_x + a_x * b_y^2)) * x
    // + (a_x * b_x * b_y + 2 * a_x * a_y * c_x - (2 * a_x^2 * c_y + a_y *
    // b_x^2)) * y + (a_x * b_y^2 * c_x - a_y * b_x * b_y * c_x + a_y^2 * c_x^2
    // +  a_y * b_x^2 * c_y - a_x * b_x * b_y * c_y + a_x^2 * c_y^2 -
    //  2 * a_x * a_y * c_x * c_y) =
    // f(x, y)
    let a_x = coeffs.x.x;
    let b_x = coeffs.y.x;
    let c_x = coeffs.z.x;
    let a_y = coeffs.x.y;
    let b_y = coeffs.y.y;
    let c_y = coeffs.z.y;
    [
        a_y * a_y,
        -2.0 * a_x * a_y,
        a_x * a_x,
        a_y * b_x * b_y + 2.0 * a_x * a_y * c_y - (2.0 * a_y * a_y * c_x + a_x * b_y * b_y),
        a_x * b_x * b_y + 2.0 * a_x * a_y * c_x - (2.0 * a_x * a_x * c_y + a_y * b_x * b_x),
        a_x * b_y * b_y * c_x - a_y * b_x * b_y * c_x
            + a_y * a_y * c_x * c_x
            + a_y * b_x * b_x * c_y
            - a_x * b_x * b_y * c_y
            + a_x * a_x * c_y * c_y
            - 2.0 * a_x * a_y * c_x * c_y,
    ]
}

fn intersect_point_point(point1: Vec2<f64>, point2: Vec2<f64>) -> Result<Vec<Vec2<f64>>, ()> {
    // TODO: Do we really care whether they are the same line if one is a point?
    // Currently we don't differentiate this case.
    if (point1 - point2).map(|e| e.abs()).reduce_partial_max() < 1e-2 {
        Ok(vec![point1])
    } else {
        Ok(vec![])
    }
}

fn intersect_line_point(line: &Vec2<Vec2<f64>>, point: Vec2<f64>) -> Result<Vec<Vec2<f64>>, ()> {
    // First, check for degenerate cases.
    // TODO: change these to point < e.
    // TODO: Do we really care whether they are the same line if one is a point?
    // Currently we don't differentiate this case.
    if line.x == Vec2::zero() {
        // Line is a point.
        return intersect_point_point(line.y, point);
    }
    // Otherwise, implicitize line and check whether the value of the equation at
    // this point is within e.
    let a = implicitize_line(line);
    let e = 100.0 * f64::EPSILON.sqrt();
    if a[0] * point.x + a[1] * point.y + a[2] < e {
        Ok(vec![point])
    } else {
        Ok(vec![])
    }
}

fn intersect_line_line(
    line1: &Vec2<Vec2<f64>>,
    line2: &Vec2<Vec2<f64>>,
) -> Result<Vec<Vec2<f64>>, ()> {
    // First, check for degenerate cases.
    // TODO: change these to point < e.
    if line1.x == Vec2::zero() {
        // Line is a point.
        return intersect_line_point(line2, line1.y);
    }
    if line2.x == Vec2::zero() {
        // Line is a point.
        return intersect_line_point(line1, line2.y);
    }
    // Otherwise, we can compute the intersection point directly from the
    // parameterized version.
    let x1 = line1.y.x;
    let y1 = line1.y.y;
    let dx2 = -line1.x.x;
    let dy2 = -line1.x.y;
    let x3 = line2.y.x;
    let y3 = line2.y.y;
    let dx4 = -line2.x.x;
    let dy4 = -line2.x.y;
    // We now perform a case distinction on the kind of equation we are dealing
    // with...
    let e = 100.0 * f64::EPSILON.sqrt();
    let d_ = dx2 * dy4 - dy2 * dx4;
    if d_.abs() < e {
        // The lines are the same slope (i.e. parallel), but may or may not coincide.
        let a = implicitize_line(line1);
        let b = implicitize_line(line2);
        if (a[2] - b[2]).abs() < e {
            // The lines coincide (approximately) at all points.
            return Err(());
        }
        // Otherwise, they are offset in a way that will cause them never to
        // intersect.
        return Ok(Vec::new());
    }
    // The lines are not parallel, so their intersection can be defined using
    // determinants.
    let t = ((x1 - x3) * dy4 - (y1 - y3) * dx4) / d_;
    let x = x1 - dx2 * t;
    let y = y1 - dy2 * t;
    Ok(vec![Vec2::new(x, y)])
    /* // Otherwise, implicitize lines and check where (and whether) they intersect.
    let a = implicitize_line(line1);
    let b = implicitize_line(line2);

    // We now perform a case distinction on the kind of equation we are dealing with...
    let d_ = a[0] * b[1] - b[0] * a[1];
    if d_.abs() < e {
        // The lines are the same shape, but offset in a way that will cause them never to
        // intersect (i.e. parallel).
        return Ok(Vec::new());
    }

    // The lines are not parallel, so their intersection can be defined using determinants.
    let x0 =
    let x = () / d_;
    Ok(vec![])

    // Otherwise, both lines are not parallel.
    if a[1] >= e || a[2] {
        // Line 1 is not vertical, so we can get its slope.
        let y =
    }
    if (a[0] - b[0]).abs() > e {
        // We have a nonzero linear Y component, so we can solve for X.
        //
        // If a and b are both vertical or both horizontal,
        // We can solve for x and y by solving a simple linear equation.
        //
        // ax + by + c = 0
        // ax + c = -by
        // (ax + c) / (-b) = y
        //
        // Now setting both y's equal (which they must be at an intersection point), we have:
        //
        // (a[0]x + a[2]) / (-a[1]) = (b[0]x + b[2]) / (-b[1])
        // (a[0]x + a[2]) / a[1] = (b[0]x + b[2]) / b[1]
        // b[1] * (a[0]x + a[2]) = a[1] * (b[0]x + b[2])
        // a[0] * b[1] * x + a[2] * b[1] = a[1] * b[0] * x + a[1] * b[2]
        // (a[0] * b[1] - a[1] * b[0]) * x = (a[1] * b[2] - a[2] * a[1])
        let x = (b[2] * a[1] - b[1] * a[2]) / d_;
        /* let a_ = a[0] / a[1];
        let b_ = a[1] / a[0];
        let x = (b[1] - a[1]) / (a[0] - b[0]); */
        // Now, solve for y by working backwards from x.
        let y = (a[0] * x + a[2]) / a[1];
        Ok(vec![Vec2::new(x, y)])
    } else {
        // The X component is zero, so we can't solve for X.
        if (a[1] - b[1]).abs() > e {
            // We have a nonzero linear Y coefficient, so we can still solve for Y.
            let y = (b[0] - a[0]) / (a[1] - b[1]);
            // Now, solve for x by working backwards from y.
            let x = a[0] * x + a[1];
            Ok(vec![Vec2::new(x, y)])
        } else {
            // x and y coefficients are zero, meaning the lines are either identical or don't
            // intersect.
            if (a[2] - b[2]).abs() < e {
                // The lines are approximately identical.
                return Err(());
            } else {
                // The lines are the same shape, but offset in a way that will cause them never to
                // intersect.
                return Ok(Vec::new());
            }
        }
    } */
}

fn intersect_quadratic_point(
    curve: &Vec3<Vec2<f64>>,
    point: Vec2<f64>,
) -> Result<Vec<Vec2<f64>>, ()> {
    // TODO: Do we really care whether they are the same curve if one is a point?
    // Currently we consider these "Ok".
    // TODO: If needed, fix so that this doesn't care whether the points are in
    // bounds or not (since intersect_quadratics currently doesn't try to filter
    // roots in this way).
    return Ok(quadratic_nearest_point(curve, point)
        .into_iter()
        .filter(|&(_, _, dist_squared)| dist_squared < 1e-2)
        .map(|(_, pt, _)| pt)
        .collect());
}

fn intersect_quadratic_line(
    curve: &Vec3<Vec2<f64>>,
    line: &Vec2<Vec2<f64>>,
) -> Result<Vec<Vec2<f64>>, ()> {
    // First, check for degenerate cases.
    // TODO: change these to < e.
    if line.x == Vec2::zero() {
        // Line is a point.
        return intersect_quadratic_point(curve, line.y);
    }
    if curve.x == Vec2::zero() {
        // Curve is a line.
        return intersect_line_line(line, &Vec2::new(curve.y, curve.z));
    }
    // TODO: Implement.
    println!("Problem case.");
    return Err(());
}

/// Solve intersection of two parametric quadratic curves.
///
/// Based on an algorithm for computing the intersection of bivariate quadratic
/// equations (as part of the intersection of conic sections), from
///
/// http://www.piprime.fr/files/asymptote/geometry/modules/geometry.asy.html#intersectionpoints%28bqe,bqe%29
/// (it is LGPL).
///
/// Returns Err if the cones are (approximately) identical, meaning the answer
/// to the question is "every point".  Otherwise, returns a list of intersection
/// points.
pub fn intersect_quadratics(
    curve1: &Vec3<Vec2<f64>>,
    curve2: &Vec3<Vec2<f64>>,
) -> Result<Vec<Vec2<f64>>, ()> {
    // First, check for degenerate cases.
    // TODO: change these to < e.
    if curve1.x == Vec2::zero() {
        // Curve is a line.
        return intersect_quadratic_line(curve2, &Vec2::new(curve1.y, curve1.z));
    }
    if curve2.x == Vec2::zero() {
        // Curve is a line.
        return intersect_quadratic_line(curve1, &Vec2::new(curve2.y, curve2.z));
    }

    // Transform the curves from parametric form to implicit form.
    let a = implicitize_quadratic(curve1);
    let b = implicitize_quadratic(curve2);
    // We now perform a case distinction on the kind of equation we are dealing
    // with...
    let e = 100.0 * f64::EPSILON.sqrt();
    let x = if (a[0] - b[0]).abs() > e || (a[1] - b[1]).abs() > e || (a[2] - b[2]).abs() > e {
        // The complex case: we have one or more of squared x, squared y, or interaction
        // terms between x and y.  We start by solving for the x that minimizes
        // the distance between the two curves, and then later will figure out
        // which are the associated y's.
        let a_ = -2.0 * a[0] * a[2] * b[0] * b[2] + a[0] * a[2] * b[1] * b[1]
            - a[0] * a[1] * b[2] * b[1]
            + a[1] * a[1] * b[0] * b[2]
            - a[2] * a[1] * b[0] * b[1]
            + a[0] * a[0] * b[2] * b[2]
            + a[2] * a[2] * b[0] * b[0];
        let b_ = -a[2] * a[1] * b[0] * b[4] - a[2] * a[4] * b[0] * b[1] - a[1] * a[3] * b[2] * b[1]
            + 2.0 * a[0] * a[2] * b[1] * b[4]
            - a[0] * a[1] * b[2] * b[4]
            + a[1] * a[1] * b[2] * b[3]
            - 2.0 * a[2] * a[3] * b[0] * b[2]
            - 2.0 * a[0] * a[2] * b[2] * b[3]
            + a[2] * a[3] * b[1] * b[1]
            - a[2] * a[1] * b[1] * b[3]
            + 2.0 * a[1] * a[4] * b[0] * b[2]
            + 2.0 * a[2] * a[2] * b[0] * b[3]
            - a[0] * a[4] * b[2] * b[1]
            + 2.0 * a[0] * a[3] * b[2] * b[2];
        let c_ = -a[3] * a[4] * b[2] * b[1] + a[2] * a[5] * b[1] * b[1]
            - a[1] * a[5] * b[2] * b[1]
            - a[1] * a[3] * b[2] * b[4]
            + a[1] * a[1] * b[2] * b[5]
            - 2.0 * a[2] * a[3] * b[2] * b[3]
            + 2.0 * a[2] * a[2] * b[0] * b[5]
            + 2.0 * a[0] * a[5] * b[2] * b[2]
            + a[3] * a[3] * b[2] * b[2]
            - 2.0 * a[2] * a[5] * b[0] * b[2]
            + 2.0 * a[1] * a[4] * b[2] * b[3]
            - a[2] * a[4] * b[1] * b[3]
            - 2.0 * a[0] * a[2] * b[2] * b[5]
            + a[2] * a[2] * b[3] * b[3]
            + 2.0 * a[2] * a[3] * b[1] * b[4]
            - a[2] * a[4] * b[0] * b[4]
            + a[4] * a[4] * b[0] * b[2]
            - a[2] * a[1] * b[3] * b[4]
            - a[2] * a[1] * b[1] * b[5]
            - a[0] * a[4] * b[2] * b[4]
            + a[0] * a[2] * b[4] * b[4];
        let d_ = -a[4] * a[5] * b[2] * b[1]
            + a[2] * a[3] * b[4] * b[4]
            + 2.0 * a[3] * a[5] * b[2] * b[2]
            - a[2] * a[1] * b[4] * b[5]
            - a[2] * a[4] * b[3] * b[4]
            + 2.0 * a[2] * a[2] * b[3] * b[5]
            - 2.0 * a[2] * a[3] * b[2] * b[5]
            - a[3] * a[4] * b[2] * b[4]
            - 2.0 * a[2] * a[5] * b[2] * b[3]
            - a[2] * a[4] * b[1] * b[5]
            + 2.0 * a[1] * a[4] * b[2] * b[5]
            - a[1] * a[5] * b[2] * b[4]
            + a[4] * a[4] * b[2] * b[3]
            + 2.0 * a[2] * a[5] * b[1] * b[4];
        let e_ = -2.0 * a[2] * a[5] * b[2] * b[5]
            + a[4] * a[4] * b[2] * b[5]
            + a[5] * a[5] * b[2] * b[2]
            - a[4] * a[5] * b[2] * b[4]
            + a[2] * a[5] * b[4] * b[4]
            + a[2] * a[2] * b[5] * b[5]
            - a[2] * a[4] * b[4] * b[5];
        find_roots_quartic(a_, b_, c_, d_, e_)
    } else {
        // There are no square terms or interacting terms (x * y), so we have a simpler
        // equation (which may require different techniques in order to solve
        // it).
        if (a[4] - b[4]).abs() > e {
            // We have a nonzero linear Y component, so we can solve for X.
            let d = (b[4] - a[4]) * (b[4] - a[4]);
            let a_ = (a[0] * b[4] * b[4]
                + (-a[1] * b[3] - 2.0 * a[0] * a[4] + a[1] * a[3]) * b[4]
                + a[2] * b[3] * b[3]
                + (a[1] * a[4] - 2.0 * a[2] * a[3]) * b[3]
                + a[0] * a[4] * a[4]
                - a[1] * a[3] * a[4]
                + a[2] * a[3] * a[3])
                / d;
            let b_ = -((a[1] * b[4] - 2.0 * a[2] * b[3] - a[1] * a[4] + 2.0 * a[2] * a[3]) * b[5]
                - a[3] * b[4] * b[4]
                + (a[4] * b[3] - a[1] * a[5] + a[3] * a[4]) * b[4]
                + (2.0 * a[2] * a[5] - a[4] * a[4]) * b[3]
                + (a[1] * a[4] - 2.0 * a[2] * a[3]) * a[5])
                / d;
            let c_ = a[2] * (a[5] - b[5]) * (a[5] - b[5]) / d
                + a[4] * (a[5] - b[5]) / (b[4] - a[4])
                + a[5];
            find_roots_quadratic(a_, b_, c_)
        } else {
            // The Y component is zero, so we can't solve for X.
            if (a[3] - b[3]).abs() > e {
                // The X component is nonzero, so we can still solve for Y.
                let d = b[3] - a[3];
                let a_ = a[2];
                let b_ = (-a[1] * b[5] + a[4] * b[3] + a[1] * a[5] - a[3] * a[4]) / d;
                let c_ = a[0] * (a[5] - b[5]) * (a[5] - b[5]) / (d * d)
                    + a[3] * (a[5] - b[5]) / d
                    + a[5];
                let y = find_roots_quadratic(a_, b_, c_);
                // Now, solve for x by working backwards from y.
                let mut points = Vec::new();

                for &y in y.as_ref() {
                    let a_ = a[0];
                    let b_ = a[1] * y + a[3];
                    let c_ = a[2] * y * y + a[4] * y + a[5];
                    let x = find_roots_quadratic(a_, b_, c_);
                    points.extend(
                        x.as_ref()
                            .into_iter()
                            .filter(|&x| {
                                (b[0] * x * x
                                    + b[1] * x * y
                                    + b[2] * y * y
                                    + b[3] * x
                                    + b[4] * y
                                    + b[5])
                                    .abs()
                                    < 1e-5
                            })
                            .map(|&x| Vec2::new(x, y)),
                    );
                }
                return Ok(points);
            } else {
                // Both the X and Y components are 0.
                if (a[5] - b[5]).abs() < e {
                    // The cones are approximately identical.
                    return Err(());
                } else {
                    // The cones are the same shape, but offset in a way that will cause them never
                    // to intersect.
                    return Ok(Vec::new());
                }
            }
        }
    };
    // We hav solved for x; now, solve for y by working backwards from x.
    let mut points = Vec::new();
    for &x in x.as_ref() {
        let a_ = a[2];
        let b_ = a[1] * x + a[4];
        let c_ = a[0] * x * x + a[3] * x + a[5];
        let y = find_roots_quadratic(a_, b_, c_);
        points.extend(
            y.as_ref()
                .into_iter()
                .filter(|&y| {
                    (b[0] * x * x + b[1] * x * y + b[2] * y * y + b[3] * x + b[4] * y + b[5]).abs()
                        < 1e-5
                })
                .map(|&y| Vec2::new(x, y)),
        );
    }
    return Ok(points);
}

/// A 2-dimensional vector, for internal use.
type Vector2<T> = [T; 2];
/// A 3-dimensional vector, for internal use.
type Vector3<T> = [T; 3];
/// A 4-dimensional vector, for internal use.
type Vector4<T> = [T; 4];

#[inline]
fn zip_with2<T, U, V, F>(a: Vector2<T>, b: Vector2<U>, f: F) -> Vector2<V>
where
    T: Copy,
    U: Copy,
    F: Fn(T, U) -> V,
{
    let (ax, ay) = (a[0], a[1]);
    let (bx, by) = (b[0], b[1]);
    [f(ax, bx), f(ay, by)]
}

#[inline]
fn zip_with3<T, U, V, F>(a: Vector3<T>, b: Vector3<U>, f: F) -> Vector3<V>
where
    T: Copy,
    U: Copy,
    F: Fn(T, U) -> V,
{
    let (ax, ay, az) = (a[0], a[1], a[2]);
    let (bx, by, bz) = (b[0], b[1], b[2]);
    [f(ax, bx), f(ay, by), f(az, bz)]
}

#[inline]
fn zip_with4<T, U, V, F>(a: Vector4<T>, b: Vector4<U>, f: F) -> Vector4<V>
where
    T: Copy,
    U: Copy,
    F: Fn(T, U) -> V,
{
    let (ax, ay, az, aw) = (a[0], a[1], a[2], a[3]);
    let (bx, by, bz, bw) = (b[0], b[1], b[2], b[3]);
    [f(ax, bx), f(ay, by), f(az, bz), f(aw, bw)]
}

#[inline]
fn mul2<T>(a: Vector2<T>, b: T) -> Vector2<T>
where
    T: Copy + Mul<T, Output = T>,
{
    zip_with2(a, const2(b), Mul::mul)
}

#[inline]
fn mul3<T>(a: Vector3<T>, b: T) -> Vector3<T>
where
    T: Copy + Mul<T, Output = T>,
{
    zip_with3(a, const3(b), Mul::mul)
}

#[inline]
fn mul4<T>(a: Vector4<T>, b: T) -> Vector4<T>
where
    T: Copy + Mul<T, Output = T>,
{
    zip_with4(a, const4(b), Mul::mul)
}

#[inline]
fn const2<T: Copy>(x: T) -> Vector2<T> { [x, x] }

#[inline]
fn const3<T: Copy>(x: T) -> Vector3<T> { [x, x, x] }

#[inline]
fn const4<T: Copy>(x: T) -> Vector4<T> { [x, x, x, x] }

fn build_sources(seed: u32, octaves: usize) -> Vec<Perlin> {
    let mut sources = Vec::with_capacity(octaves);
    for x in 0..octaves {
        sources.push(Perlin::new().set_seed(seed + x as u32));
    }
    sources
}

/// Noise function that outputs hybrid Multifractal noise.
///
/// The result of this multifractal noise is that valleys in the noise should
/// have smooth bottoms at all altitudes.
#[derive(Clone, Debug)]
pub struct HybridMulti {
    /// Total number of frequency octaves to generate the noise with.
    ///
    /// The number of octaves control the _amount of detail_ in the noise
    /// function. Adding more octaves increases the detail, with the drawback
    /// of increasing the calculation time.
    pub octaves: usize,

    /// The number of cycles per unit length that the noise function outputs.
    pub frequency: f64,

    /// A multiplier that determines how quickly the frequency increases for
    /// each successive octave in the noise function.
    ///
    /// The frequency of each successive octave is equal to the product of the
    /// previous octave's frequency and the lacunarity value.
    ///
    /// A lacunarity of 2.0 results in the frequency doubling every octave. For
    /// almost all cases, 2.0 is a good value to use.
    pub lacunarity: f64,

    /// A multiplier that determines how quickly the amplitudes diminish for
    /// each successive octave in the noise function.
    ///
    /// The amplitude of each successive octave is equal to the product of the
    /// previous octave's amplitude and the persistence value. Increasing the
    /// persistence produces "rougher" noise.
    ///
    /// H = 1.0 - fractal increment = -ln(persistence) / ln(lacunarity).  For
    /// a fractal increment between 0 (inclusive) and 1 (exclusive), keep
    /// persistence between 1 / lacunarity (inclusive, for low fractal
    /// dimension) and 1 (exclusive, for high fractal dimension).
    pub persistence: f64,

    /// An offset that is added to the output of each sample of the underlying
    /// Perlin noise function.  Because each successive octave is weighted in
    /// part by the previous signal's output, increasing the offset will weight
    /// the output more heavily towards 1.0.
    pub offset: f64,

    seed: u32,
    sources: Vec<Perlin>,
}

impl HybridMulti {
    pub const DEFAULT_FREQUENCY: f64 = 2.0;
    pub const DEFAULT_LACUNARITY: f64 = /* std::f64::consts::PI * 2.0 / 3.0 */2.0;
    pub const DEFAULT_OCTAVES: usize = 6;
    pub const DEFAULT_OFFSET: f64 = /* 0.25 *//* 0.5*/ 0.7;
    // -ln(2^(-0.25))/ln(2) = 0.25
    // 2^(-0.25) ~ 13/16
    pub const DEFAULT_PERSISTENCE: f64 = /* 0.25 *//* 0.5*/ 13.0 / 16.0;
    pub const DEFAULT_SEED: u32 = 0;
    pub const MAX_OCTAVES: usize = 32;

    pub fn new() -> Self {
        Self {
            seed: Self::DEFAULT_SEED,
            octaves: Self::DEFAULT_OCTAVES,
            frequency: Self::DEFAULT_FREQUENCY,
            lacunarity: Self::DEFAULT_LACUNARITY,
            persistence: Self::DEFAULT_PERSISTENCE,
            offset: Self::DEFAULT_OFFSET,
            sources: build_sources(Self::DEFAULT_SEED, Self::DEFAULT_OCTAVES),
        }
    }

    pub fn set_offset(self, offset: f64) -> Self { Self { offset, ..self } }
}

impl Default for HybridMulti {
    fn default() -> Self { Self::new() }
}

impl MultiFractal for HybridMulti {
    fn set_octaves(self, mut octaves: usize) -> Self {
        if self.octaves == octaves {
            return self;
        }

        octaves = octaves.max(1).min(Self::MAX_OCTAVES);
        Self {
            octaves,
            sources: build_sources(self.seed, octaves),
            ..self
        }
    }

    fn set_frequency(self, frequency: f64) -> Self { Self { frequency, ..self } }

    fn set_lacunarity(self, lacunarity: f64) -> Self { Self { lacunarity, ..self } }

    fn set_persistence(self, persistence: f64) -> Self {
        Self {
            persistence,
            ..self
        }
    }
}

impl Seedable for HybridMulti {
    fn set_seed(self, seed: u32) -> Self {
        if self.seed == seed {
            return self;
        }

        Self {
            seed,
            sources: build_sources(seed, self.octaves),
            ..self
        }
    }

    fn seed(&self) -> u32 { self.seed }
}

/// 2-dimensional `HybridMulti` noise
impl NoiseFn<Point2<f64>> for HybridMulti {
    fn get(&self, mut point: Point2<f64>) -> f64 {
        // First unscaled octave of function; later octaves are scaled.
        point = mul2(point, self.frequency);
        // Offset and bias to scale into [offset - 1.0, 1.0 + offset] range.
        let bias = 1.0;
        let mut result = (self.sources[0].get(point) + self.offset) * bias * self.persistence;
        let mut exp_scale = 1.0;
        let mut scale = self.persistence;
        let mut weight = result;

        // Spectral construction inner loop, where the fractal is built.
        for x in 1..self.octaves {
            // Prevent divergence.
            weight = weight.min(1.0);

            // Raise the spatial frequency.
            point = mul2(point, self.lacunarity);

            // Get noise value, and scale it to the [offset - 1.0, 1.0 + offset] range.
            let mut signal = (self.sources[x].get(point) + self.offset) * bias;

            // Scale the amplitude appropriately for this frequency.
            exp_scale *= self.persistence;
            signal *= exp_scale;

            // Add it in, weighted by previous octave's noise value.
            result += weight * signal;

            // Update the weighting value.
            weight *= signal;
            scale += exp_scale;
        }

        // Scale the result to the [-1,1] range
        (result / scale) / bias - self.offset
    }
}

/// 3-dimensional `HybridMulti` noise
impl NoiseFn<Point3<f64>> for HybridMulti {
    fn get(&self, mut point: Point3<f64>) -> f64 {
        // First unscaled octave of function; later octaves are scaled.
        point = mul3(point, self.frequency);
        // Offset and bias to scale into [offset - 1.0, 1.0 + offset] range.
        let bias = 1.0;
        let mut result = (self.sources[0].get(point) + self.offset) * bias * self.persistence;
        let mut exp_scale = 1.0;
        let mut scale = self.persistence;
        let mut weight = result;

        // Spectral construction inner loop, where the fractal is built.
        for x in 1..self.octaves {
            // Prevent divergence.
            weight = weight.min(1.0);

            // Raise the spatial frequency.
            point = mul3(point, self.lacunarity);

            // Get noise value, and scale it to the [0, 1.0] range.
            let mut signal = (self.sources[x].get(point) + self.offset) * bias;

            // Scale the amplitude appropriately for this frequency.
            exp_scale *= self.persistence;
            signal *= exp_scale;

            // Add it in, weighted by previous octave's noise value.
            result += weight * signal;

            // Update the weighting value.
            weight *= signal;
            scale += exp_scale;
        }

        // Scale the result to the [-1,1] range
        (result / scale) / bias - self.offset
    }
}

/// 4-dimensional `HybridMulti` noise
impl NoiseFn<Point4<f64>> for HybridMulti {
    fn get(&self, mut point: Point4<f64>) -> f64 {
        // First unscaled octave of function; later octaves are scaled.
        point = mul4(point, self.frequency);
        // Offset and bias to scale into [offset - 1.0, 1.0 + offset] range.
        let bias = 1.0;
        let mut result = (self.sources[0].get(point) + self.offset) * bias * self.persistence;
        let mut exp_scale = 1.0;
        let mut scale = self.persistence;
        let mut weight = result;

        // Spectral construction inner loop, where the fractal is built.
        for x in 1..self.octaves {
            // Prevent divergence.
            weight = weight.min(1.0);

            // Raise the spatial frequency.
            point = mul4(point, self.lacunarity);

            // Get noise value, and scale it to the [0, 1.0] range.
            let mut signal = (self.sources[x].get(point) + self.offset) * bias;

            // Scale the amplitude appropriately for this frequency.
            exp_scale *= self.persistence;
            signal *= exp_scale;

            // Add it in, weighted by previous octave's noise value.
            result += weight * signal;

            // Update the weighting value.
            weight *= signal;
            scale += exp_scale;
        }

        // Scale the result to the [-1,1] range
        (result / scale) / bias - self.offset
    }
}

/// Noise function that applies a scaling factor and a bias to the output value
/// from the source function.
///
/// The function retrieves the output value from the source function, multiplies
/// it with the scaling factor, adds the bias to it, then outputs the value.
pub struct ScaleBias<'a, F: 'a> {
    /// Outputs a value.
    pub source: &'a F,

    /// Scaling factor to apply to the output value from the source function.
    /// The default value is 1.0.
    pub scale: f64,

    /// Bias to apply to the scaled output value from the source function.
    /// The default value is 0.0.
    pub bias: f64,
}

impl<'a, F> ScaleBias<'a, F> {
    pub fn new(source: &'a F) -> Self {
        ScaleBias {
            source,
            scale: 1.0,
            bias: 0.0,
        }
    }

    pub fn set_scale(self, scale: f64) -> Self { ScaleBias { scale, ..self } }

    pub fn set_bias(self, bias: f64) -> Self { ScaleBias { bias, ..self } }
}

impl<'a, F: NoiseFn<T> + 'a, T> NoiseFn<T> for ScaleBias<'a, F> {
    #[cfg(not(target_os = "emscripten"))]
    fn get(&self, point: T) -> f64 { (self.source.get(point)).mul_add(self.scale, self.bias) }

    #[cfg(target_os = "emscripten")]
    fn get(&self, point: T) -> f64 { (self.source.get(point) * self.scale) + self.bias }
}
