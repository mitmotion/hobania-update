use crate::{
    all::ForestKind,
    sim::{local_cells, Cave, Path, RiverKind, SimChunk, WorldSim},
    site::SpawnRules,
    util::{RandomField, Sampler},
    IndexRef, CONFIG,
};
use common::{
    calendar::{Calendar, CalendarEvent},
    terrain::{
        quadratic_nearest_point, river_spline_coeffs, uniform_idx_as_vec2, vec2_as_uniform_idx,
        TerrainChunkSize,
    },
    vol::RectVolSize,
};
use core::simd::{Simd, SimdElement};
use noise::NoiseFn;
use serde::Deserialize;
use std::{
    f32,
    ops::{Add, Div, Mul, Sub},
    simd::StdFloat,
};
use tracing::error;
use vek::*;

pub struct ColumnGen1D<'a, 'b> {
    pub(crate) parent: &'b ColumnGen<'a>,
    wpos_row: i32,
    wposy: f32,
    cubic_y: Cubic<f32>,

    /// Interpolation splines.
    chaos_spline: CatmullRom<f32>,
    temp_spline: CatmullRom<f32>,
    humidity_spline: CatmullRom<f32>,
    rockiness_spline: CatmullRom<f32>,
    tree_density_spline: CatmullRom<f32>,
    spawn_rate_spline: CatmullRom<f32>,
    near_water_spline: CatmullRom<f32>,
    // water_vel_spline: CatmullRom<Vec3<f32>>,
    water_vel_spline_x: CatmullRom<f32>,
    water_vel_spline_y: CatmullRom<f32>,
    water_vel_spline_z: CatmullRom<f32>,
    alt_spline: CatmullRom<f32>,
    alt_spline_mono: CatmullRom<f32>,
    basement_sub_alt_spline: CatmullRom<f32>,
    surface_veg_spline: CatmullRom<f32>,
    cliff_height_spline: CatmullRom<f32>,
}

/// A structure specialized with data for column generation to dramatically speed it up.
pub struct ColumnGen<'a> {
    pub sim: &'a WorldSim,
    chunk_pos: Vec2<i32>,
    index: IndexRef<'a>,
    /* calendar: Option<&'a Calendar>, */

    /// Interpolation splines.
    pub(crate) chaos_spline: CatmullRom2D<f32>,
    pub(crate) temp_spline: CatmullRom2D<f32>,
    pub(crate) humidity_spline: CatmullRom2D<f32>,
    pub(crate) rockiness_spline: CatmullRom2D<f32>,
    pub(crate) tree_density_spline: CatmullRom2D<f32>,
    pub(crate) spawn_rate_spline: CatmullRom2D<f32>,
    pub(crate) near_water_spline: CatmullRom2D<f32>,
    // pub(crate) water_vel_spline: CatmullRom2D<Vec3<f32>>,
    pub(crate) water_vel_spline_x: CatmullRom2D<f32>,
    pub(crate) water_vel_spline_y: CatmullRom2D<f32>,
    pub(crate) water_vel_spline_z: CatmullRom2D<f32>,
    pub(crate) alt_spline: CatmullRom2D<f32>,
    pub(crate) alt_spline_mono: Steffen2D<f32>,
    pub(crate) basement_sub_alt_spline: Steffen2D<f32>,
    pub(crate) surface_veg_spline: Steffen2D<f32>,
    pub(crate) cliff_height_spline: CatmullRom2D<f32>,

    /// Chunk data
    pub(crate) sim_chunk: &'a SimChunk,
    // pub(crate) spawn_rules: SpawnRules,
}

#[derive(Deserialize)]
pub struct Colors {
    pub cold_grass: (f32, f32, f32),
    pub warm_grass: (f32, f32, f32),
    pub dark_grass: (f32, f32, f32),
    pub wet_grass: (f32, f32, f32),
    pub cold_stone: (f32, f32, f32),
    pub hot_stone: (f32, f32, f32),
    pub warm_stone: (f32, f32, f32),
    pub beach_sand: (f32, f32, f32),
    pub desert_sand: (f32, f32, f32),
    pub snow: (f32, f32, f32),
    pub snow_moss: (f32, f32, f32),

    pub stone_col: (u8, u8, u8),

    pub dirt_low: (f32, f32, f32),
    pub dirt_high: (f32, f32, f32),

    pub snow_high: (f32, f32, f32),
    pub warm_stone_high: (f32, f32, f32),

    pub grass_high: (f32, f32, f32),
    pub tropical_high: (f32, f32, f32),
}

/// Generalised power function, pushes values in the range 0-1 to extremes.
fn power(x: f64, t: f64) -> f64 {
    if x < 0.5 {
        (2.0 * x).powf(t) / 2.0
    } else {
        1.0 - (-2.0 * x + 2.0).powf(t) / 2.0
    }
}

/// Initial tension factor.
const TENSION: f32 = 0.5;

/// Vector representing precomputed cubic points.
pub type Cubic<T> = /*vek::vec::repr_simd::Vec4<T>*/T;

/// Catmul-Rom spline sampler.
#[derive(Debug)]
pub struct CatmullRom<T> {
    /// Precomputed spline vector.
    vec: vek::vec::repr_simd::Vec4<T>,
}

/// Catmul-Rom 2D spline sampler.
///
/// With naive multiplication, it seems like one has to choose a dimension (x or y) with which to
/// cache the Catmul-Rom matrix, since the other dimension appears to depend on values computed
/// from the first.  However, by applying some matrix identities, we can construct a single matrix
/// that can be multiplied one way to get the x vector, and another way to get the y vector.  This
/// allows speedups in the point-query case, while being just as efficient for computing by-row or
/// by-column.
#[derive(Debug)]
pub struct CatmullRom2D<T> {
    /// Precomputed spline matrix.
    mat: vek::mat::repr_simd::column_major::Mat4<T>,
}

/// M Steffen spline sampler, which tweaks the cubic interpolation parameters to preserve
/// monotonicity between points.
///
/// Unlike Catmull-Rom spline generation, here it's hard (or at least nonobvious) to see how to do
/// much better than computing the parameters on demand by row / column.
pub struct Steffen2D<T: SimdElement> {
    /// Each row computes the spline over y³ at that row.
    vec: Vec4<Simd<T, 4>>,
}

/// 2D spline generator.
pub struct SplineGen2D<Sample> {
    /// Grid from which to sample when generating the matrix.
    samples: vek::mat::repr_simd::column_major::Mat4<Sample>,
}

/// Generate a cubic vector at a point in [0,1] (for use with CatmullRom2D::compute generated
/// from a structure).
#[inline]
pub fn cubic(x: f32) -> Cubic<f32> {
    /* let x2 = x * x;
    Vec4::new(1.0, x, x2, x2 * x) */
    x
}

impl<Sampler> SplineGen2D<Sampler> {
    /// Construct a new Catmul-Rom generator (centered at a point) that caches a sampled structure
    /// at each point (get_sampler returns the Sampler).  This can be used to simplify generation
    /// of future spline samplers, since they don't have to repeatedly re-look-up the base
    /// structures (which may be expensive or fallible).
    ///
    /// Note that the matrix is transposed from what you'd normally expect; this is because the
    /// generate function is slightly more efficient to compute row-wise when starting from the
    /// transposed matrix.
    ///
    /// Also note that it *might* actually be faster to, rather than caching the samples, cache the
    /// index (so the CPU can learn the stride when indexing into the matrix); we defer this
    /// investigation to another day.
    #[inline]
    pub fn new(x: i32, y: i32, mut s: impl FnMut(i32, i32) -> Option<Sampler>) -> Option<Self> {
        // NOTE: Expanded out to avoid potentially not being able to allocate in place due to all
        // the ? early termination.
        /* Mat4::new(
            s(x - 1, y - 1)?, s(x, y - 1)?, s(x + 1, y - 1)?, s(x + 2, y - 1)?,
            s(x - 1, y)?,     s(x, y)?,     s(x + 1, y)?,     s(x + 2, y)?,
            s(x - 1, y + 1)?, s(x, y + 1)?, s(x + 1, y + 1)?, s(x + 2, y + 1)?,
            s(x - 1, y + 2)?, s(x, y + 2)?, s(x + 1, y + 2)?, s(x + 2, y + 2)?,
        ) */
        let samples = vek::mat::repr_simd::column_major::Mat4::new(
            s(x - 1, y - 1)?, s(x - 1, y)?, s(x - 1, y + 1)?, s(x - 1, y + 2)?,
            s(x,     y - 1)?, s(x,     y)?, s(x,     y + 1)?, s(x,     y + 2)?,
            s(x + 1, y - 1)?, s(x + 1, y)?, s(x + 1, y + 1)?, s(x + 1, y + 2)?,
            s(x + 2, y - 1)?, s(x + 2, y)?, s(x + 2, y + 1)?, s(x + 2, y + 2)?,
        );
        Some(Self { samples })
    }

    /// Precompute a matrix from the spline data that can efficiently determine the value of the
    /// spline at a point or row within the range of values from the generator's x,y to x+1, y+1.
    #[inline]
    pub fn generate<T, F: FnMut(Sampler) -> T>(&self, f: F) -> CatmullRom2D<T>
        where
            Sampler: Copy,
            T: MulAdd<T, T, Output = T> + Mul<Output = T> + Copy + From<f32>,
    {
        // Justification:
        //
        // Let T be the tension matrix.  We have
        // (letting p_i,j = the value at row i, col j, u = y, and v = x):
        //
        // q₀ = [1 u u² u³] T [p₀₀ p₀₁ p₀₂ p₀₃]^⊤
        // q₁ = [1 u u² u³] T [p₁₀ p₁₁ p₁₂ p₁₃]^⊤
        // q₂ = [1 u u² u³] T [p₂₀ p₂₁ p₂₂ p₂₃]^⊤
        // q₃ = [1 u u² u³] T [p₃₀ p₃₁ p₃₂ p₃₃]^⊤
        //
        // Then our final value is:
        //
        // r = [1 v v² v³] T [q₀ q₁ q₂ q₃]^⊤
        //   = [1 v v² v³] T [[1 u u² u³] T [p₀₀ p₀₁ p₀₂ p₀₃]^⊤
        //                    [1 u u² u³] T [p₁₀ p₁₁ p₁₂ p₁₃]^⊤
        //                    [1 u u² u³] T [p₂₀ p₂₁ p₂₂ p₂₃]^⊤
        //                    [1 u u² u³] T [p₃₀ p₃₁ p₃₂ p₃₃]^⊤]^⊤
        //   = [1 v v² v³] T ([1 u u² u³] (T [p₀₀ p₀₁ p₀₂ p₀₃
        //                                    p₁₀ p₁₁ p₁₂ p₁₃
        //                                    p₂₀ p₂₁ p₂₂ p₂₃
        //                                    p₃₀ p₃₁ p₃₂ p₃₃]^⊤))^⊤
        //   = [1 v v² v³] T (T [p₀₀ p₀₁ p₀₂ p₀₃
        //                       p₁₀ p₁₁ p₁₂ p₁₃
        //                       p₂₀ p₂₁ p₂₂ p₂₃
        //                       p₃₀ p₃₁ p₃₂ p₃₃]^⊤)^⊤ [1 u u² u³]^⊤
        //   = [1 v v² v³] T [p₀₀ p₀₁ p₀₂ p₀₃
        //                    p₁₀ p₁₁ p₁₂ p₁₃
        //                    p₂₀ p₂₁ p₂₂ p₂₃
        //                    p₃₀ p₃₁ p₃₂ p₃₃] T^⊤ [1 u u² u³]^⊤
        //
        // Note that since this computes with y first, it is easier to vectorize going row first,
        // then multiplying by column, which we usually do in practice when we care about the order
        // at all.  If this is not desired, just flip the positions of row and column when
        // generating the sampler matrix; the final result will be the same.
        /// Pre-computed tension matrix.
        let tension_mat: vek::mat::repr_simd::column_major::Mat4<T> = vek::mat::repr_simd::column_major::Mat4::<f32>::new(
            0.0, 1.0, 0.0, 0.0,
            -TENSION, 0.0, TENSION, 0.0,
            2.0 * TENSION, TENSION - 3.0, 3.0 - 2.0 * TENSION, -TENSION,
            -TENSION, 2.0 - TENSION, TENSION - 2.0, TENSION,
        ).map(T::from);
        /// Same matrix transposed (needed in order to compute the 2D Catmul-Rom matrix).
        let tension_mat_transpose: vek::mat::repr_simd::column_major::Mat4<T> = vek::mat::repr_simd::column_major::Mat4::<f32>::new(
            0.0, -TENSION, 2.0 * TENSION, -TENSION,
            1.0, 0.0, TENSION - 3.0, 2.0 - TENSION,
            0.0, TENSION, 3.0 - 2.0 * TENSION, TENSION - 2.0,
            0.0, 0.0, -TENSION, TENSION,
        ).map(T::from);

        CatmullRom2D {
            mat: tension_mat * self.samples.map(f) * tension_mat_transpose
        }
    }

    /// Precompute a matrix from the spline data that can efficiently determine the value of the
    /// Steffen spline at a row within the range of values from the generator's x,y to x+1,y+1.
    ///
    /// NOTE: For simplicity, since we need to call signum and this would need to be handled
    /// vectorwise when accepting non-f32 samples, we just hardcode to f32 for now.
    #[inline]
    pub fn generate_monotone<F: FnMut(Sampler) -> f32>(&self, f: F) -> Steffen2D<f32>
        where Sampler: Copy,
    {
        // For arbitrary G (the best we can do--though we might be able to exploit monotonicity by
        // dividing into more patches):
        //
        // q₀ = [1 u u² u³] G([p₀₀ p₀₁ p₀₂ p₀₃]^⊤)
        // q₁ = [1 u u² u³] G([p₁₀ p₁₁ p₁₂ p₁₃]^⊤)
        // q₂ = [1 u u² u³] G([p₂₀ p₂₁ p₂₂ p₂₃]^⊤)
        // q₃ = [1 u u² u³] G([p₃₀ p₃₁ p₃₂ p₃₃]^⊤)
        //
        // r = [1 v v² v³] G([q₀ q₁ q₂ q₃]^⊤)
        //   = [1 v v² v³] G([[1 u u² u³] G([p₀₀ p₀₁ p₀₂ p₀₃]^⊤)
        //                    [1 u u² u³] G([p₁₀ p₁₁ p₁₂ p₁₃]^⊤)
        //                    [1 u u² u³] G([p₂₀ p₂₁ p₂₂ p₂₃]^⊤)
        //                    [1 u u² u³] G([p₃₀ p₃₁ p₃₂ p₃₃]^⊤)]^⊤)
        //   = [1 v v² v³] G([[1 u u² u³] G([p₀₀ p₀₁ p₀₂ p₀₃]^⊤)
        //                    [1 u u² u³] G([p₁₀ p₁₁ p₁₂ p₁₃]^⊤)
        //                    [1 u u² u³] G([p₂₀ p₂₁ p₂₂ p₂₃]^⊤)
        //                    [1 u u² u³] G([p₃₀ p₃₁ p₃₂ p₃₃]^⊤)]^⊤)
        //   = [1 v v² v³]  G([1 u u² u³] G([p₀₀ p₀₁ p₀₂ p₀₃
        //                                   p₁₀ p₁₁ p₁₂ p₁₃
        //                                   p₂₀ p₂₁ p₂₂ p₂₃
        //                                   p₃₀ p₃₁ p₃₂ p₃₃]^⊤))^⊤
        //   = [1 v v² v³] G([1 u u² u³] G')^⊤
        //   = [1 v v² v³] G(G'(P)^⊤ [1 u u² u³]^⊤)^⊤^⊤
        //   = [1 v v² v³] G(G'(P)^⊤ [1 u u² u³]^⊤)
        //
        // r = [1 v v² v³] G([q₀ q₁ q₂ q₃]^⊤)
        let mat = self.samples.map(f);
        let a = Simd::from(mat.cols.x.into_array());
        let b = Simd::from(mat.cols.y.into_array());
        let c = Simd::from(mat.cols.z.into_array());
        let d = Simd::from(mat.cols.w.into_array());

        let half = Simd::splat(0.5);
        let secant = |b, c| c - b;
        let parabola = |a, c| (c - a) * half;
        let slope = |s_a: Simd<f32, 4>, s_b: Simd<f32, 4>, p_b: Simd<f32, 4>| {
            (s_a.signum() + s_b.signum()) * (s_a.abs().min(s_b.abs()).min(p_b.abs() * half))
        };

        // Compute secants.
        let s_a = secant(a, b);
        let s_b = secant(b, c);
        let s_c = secant(c, d);
        // Computing slopes from parabolas.
        let p_b = parabola(a, c);
        let p_c = parabola(b, d);
        // Get slopes (setting distance between neighbors to 1.0).
        let slope_b = slope(s_a, s_b, p_b);
        let slope_c = slope(s_b, s_c, p_c);

        // Interpolating splines.
        let neg_two = Simd::splat(-2.0);
        let three = Simd::splat(3.0);
        let co0 = slope_b + slope_c + s_b * neg_two;
        // = a * -0.5 + c * 0.5 + b * -0.5 + d * 0.5 - 2 * (c - b)
        // = a * -0.5 + b * 1.5 - c * 1.5 + d * 0.5;
        let co1 = s_b * three + slope_b * neg_two - slope_c;
        // = (3.0 * (c - b) - 2.0 * (a * -0.5 + c * 0.5) - (b * -0.5 + d * 0.5))
        // = a + b * -2.5 + c * 2.0 + d * -0.5;
        let co2 = slope_b;
        // = a * -0.5 + c * 0.5;
        let co3 = b;

        // 1, y, y², y³ -- maybe the wrong order, oh well.
        Steffen2D {
            vec: Vec4::new(co3, co2, co1, co0),
        }
    }
}

impl<T: MulAdd<T, T, Output = T> + Mul<Output = T> + Copy + From<f32>> CatmullRom<T>
{
    /// Evaluate the spline at a point, given a cubic vector computed from x.  x should be between
    /// 0 and 1, representing a fractional point between the 0 and 1 points of the position
    /// given to the CatmullRom2D::generate function (specifically, the x position).
    ///
    /// NOTE: Point is assumed to be ≥ 0 and < 1; if it is not, this may not produce correct
    /// results.
    ///
    /// NOTE 2: Point can actually probably be computed a bit outside of this range, but that's the
    /// range I'm confident in.
    #[inline]
    pub fn eval_at(&self, x: Cubic<f32>) -> T {
        // NOTE: Technically, x ⋅ vec, but the order (hopefully?) shouldn't matter.
        //
        // FIXME: mul_add?
        /* let prod = T::from(x.y).mul_add(self.vec.y, self.vec.x);
        let prod = T::from(x.z).mul_add(self.vec.z, prod);
        T::from(x.w).mul_add(self.vec.w, prod) */
        let x = T::from(x);
        let z = x.mul_add(self.vec.w, self.vec.z);
        let z = x.mul_add(z, self.vec.y);
        x.mul_add(z, self.vec.x)
    }
}

impl<T: MulAdd<T, T, Output = T> + Mul<Output = T> + Copy + From<f32>> CatmullRom2D<T> {
    /// Evaluate the spline at a row, given a cubic vector computed from y.
    ///
    /// NOTE: See the description of [CatmullRom::eval_at] for restrictions on y, except that where
    /// "x" is used, replace it with "y".
    #[inline]
    pub fn eval_at_row(&self, y: Cubic<f32>) -> CatmullRom<T>
    {
        let y = vek::vec::repr_simd::Vec4::<T>::broadcast(T::from(y));
        let vec = y.mul_add(self.mat.cols.w, self.mat.cols.z);
        let vec = y.mul_add(vec, self.mat.cols.y);
        CatmullRom {
            vec: y.mul_add(vec, self.mat.cols.x)/*self.mat * y.map(T::from)*/,
        }
    }

    /// Evaluate the spline at a column, given a cubic vector computed from x.
    ///
    /// NOTE: See the description of [CatmullRom::eval_at] for restrictions on x.
    #[inline]
    pub fn eval_at_col(&self, x: Cubic<f32>) -> CatmullRom<T>
    {
        Self {
            mat: self.mat.transposed()
        }.eval_at_row(x)
        /* CatmullRom {
            vec: x.map(T::from) * self.mat
        } */
    }

    /// Convenient shorthand for `spline.eval_at_row(y).eval_at(x)`
    #[inline]
    pub fn eval_at(&self, x: Cubic<f32>, y: Cubic<f32>) -> T
    {
        self.eval_at_row(y).eval_at(x)
    }
}

impl Steffen2D<f32> {
    /// Evaluate the spline at a row, given a cubic vector computed from y.
    ///
    /// NOTE: See the description of [CatmullRom::eval_at] for restrictions on y, except that where
    /// "x" is used, replace it with "y".
    ///
    /// NOTE: Ideally, we would compute this for 4 y vectors at once, allowing us to continue to
    /// benefit from SIMD acceleration all the way through.
    #[inline]
    pub fn eval_at_row(&self, y: Cubic<f32>) -> CatmullRom<f32>
    {
        // First, we compute the interpolated value for f at row y,
        // for each x endpoint (we can do this vectorized).
        let y = Simd::splat(y);
        let vec = y.mul_add(self.vec.w, self.vec.z);
        let vec = y.mul_add(vec, self.vec.y);
        let [a, b, c, d] = y.mul_add(vec, self.vec.x).to_array();

        // Now, we perform ordinary scalar interpolation along the vector.
        let half = 0.5;
        let secant = |b, c| c - b;
        let parabola = |a, c| (c - a) * half;
        let slope = |s_a: f32, s_b: f32, p_b: f32| {
            (s_a.signum() + s_b.signum()) * (s_a.abs().min(s_b.abs()).min(p_b.abs() * half))
        };

        // Compute secants.
        let s_a = secant(a, b);
        let s_b = secant(b, c);
        let s_c = secant(c, d);
        // Computing slopes from parabolas.
        let p_b = parabola(a, c);
        let p_c = parabola(b, d);
        // Get slopes (setting distance between neighbors to 1.0).
        let slope_b = slope(s_a, s_b, p_b);
        let slope_c = slope(s_b, s_c, p_c);

        // Interpolating splines.
        let neg_two = -2.0;
        let three = 3.0;
        let co0 = slope_b + slope_c + s_b * neg_two;
        // = a * -0.5 + c * 0.5 + b * -0.5 + d * 0.5 - 2 * (c - b)
        // = a * -0.5 + b * 1.5 - c * 1.5 + d * 0.5;
        let co1 = s_b * three + slope_b * neg_two - slope_c;
        // = (3.0 * (c - b) - 2.0 * (a * -0.5 + c * 0.5) - (b * -0.5 + d * 0.5))
        // = a + b * -2.5 + c * 2.0 + d * -0.5;
        let co2 = slope_b;
        // = a * -0.5 + c * 0.5;
        let co3 = b;

        // 1, x, x², x³ -- maybe the wrong order, oh well.
        CatmullRom {
            vec: vek::vec::repr_simd::Vec4::new(co3, co2, co1, co0),
        }
    }

    /// Convenient shorthand for `spline.eval_at_row(y).eval_at(x)`
    #[inline]
    pub fn eval_at(&self, x: Cubic<f32>, y: Cubic<f32>) -> f32
    {
        self.eval_at_row(y).eval_at(x)
    }
}

impl<'a> ColumnGen<'a> {
    /// Precompute metadata needed for generating columns in a chunk.  Doing this should
    /// significantly accelerate column generation.  If it can't do so for whatever reason, returns
    /// None.
    pub fn new(sim: &'a WorldSim, chunk_pos: Vec2<i32>, index: IndexRef<'a>/*, calendar: Option<&'a Calendar>*/) -> Option<Self> {
        // let chunk_pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);

        // NOTE: Interpolation will go wonky at the edges since we take max() rather than go
        // negative, but that's fine.
        let catmull_rom_gen = SplineGen2D::new(
            chunk_pos.x.max(0),
            chunk_pos.y.max(0),
            |x, y| sim.get(Vec2::new(x, y)),
        )?;
        let chaos_spline = catmull_rom_gen.generate(|chunk| chunk.chaos);
        let temp_spline = catmull_rom_gen.generate(|chunk| chunk.temp);
        let humidity_spline = catmull_rom_gen.generate(|chunk| chunk.humidity);
        let rockiness_spline = catmull_rom_gen.generate(|chunk| chunk.rockiness);
        let tree_density_spline = catmull_rom_gen.generate(|chunk| chunk.tree_density);
        let spawn_rate_spline = catmull_rom_gen.generate(|chunk| chunk.spawn_rate);
        let near_water_spline = catmull_rom_gen.generate(|chunk| if chunk.river.near_water() { 1.0 } else { 0.0 });
        /* let water_vel_spline = catmull_rom_gen.generate(|chunk| {
            if chunk.river.river_kind.is_some() {
                chunk.river.velocity.x
            } else {
                Vec3::zero()
            }
        }); */
        let water_vel_spline_x = catmull_rom_gen.generate(|chunk| {
            if chunk.river.river_kind.is_some() {
                chunk.river.velocity.x
            } else {
                0.0
            }
        });
        let water_vel_spline_y = catmull_rom_gen.generate(|chunk| {
            if chunk.river.river_kind.is_some() {
                chunk.river.velocity.y
            } else {
                0.0
            }
        });
        let water_vel_spline_z = catmull_rom_gen.generate(|chunk| {
            if chunk.river.river_kind.is_some() {
                chunk.river.velocity.z
            } else {
                0.0
            }
        });
        let alt_spline = catmull_rom_gen.generate(|chunk| chunk.alt);
        let alt_spline_mono = catmull_rom_gen.generate_monotone(|chunk| chunk.alt);
        let basement_sub_alt_spline = catmull_rom_gen.generate_monotone(|chunk| chunk.basement.sub(chunk.alt));
        let surface_veg_spline = catmull_rom_gen.generate_monotone(|chunk| chunk.surface_veg);
        let cliff_height_spline = catmull_rom_gen.generate(|chunk| chunk.cliff_height);

        // NOTE: Center of matrix.
        let sim_chunk = /*sim.get(chunk_pos)?*/catmull_rom_gen.samples.cols.y.y;
        /* let spawn_rules = sim_chunk
            .sites
            .iter()
            .map(|site| index.sites[*site].spawn_rules(wpos))
            .fold(SpawnRules::default(), |a, b| a.combine(b)); */

        Some(Self {
            sim,
            chunk_pos,
            index,
            /* calendar, */

            chaos_spline,
            temp_spline,
            humidity_spline,
            rockiness_spline,
            tree_density_spline,
            spawn_rate_spline,
            near_water_spline,
            // water_vel_spline,
            water_vel_spline_x,
            water_vel_spline_y,
            water_vel_spline_z,
            alt_spline,
            alt_spline_mono,
            basement_sub_alt_spline,
            surface_veg_spline,
            cliff_height_spline,

            sim_chunk,
            // spawn_rules,
        })
    }

    pub fn eval_at_row<'b>(&'b self, wpos_row: i32) -> ColumnGen1D<'a, 'b> {
        let rel_chunk_pos = wpos_row - self.chunk_pos.y * TerrainChunkSize::RECT_SIZE.y as i32;
        // let rel_chunk_pos = wpos_row - chunk_pos.y * TerrainChunkSize::RECT_SIZE.as_::<i32>();
        let wposy = rel_chunk_pos/*.y*/ as f32 / TerrainChunkSize::RECT_SIZE.y as f32;
        let cubic_y = cubic(wposy);

        ColumnGen1D {
            parent: self,
            wpos_row,
            wposy,
            cubic_y,

            chaos_spline: self.chaos_spline.eval_at_row(cubic_y),
            temp_spline: self.temp_spline.eval_at_row(cubic_y),
            humidity_spline: self.humidity_spline.eval_at_row(cubic_y),
            rockiness_spline: self.rockiness_spline.eval_at_row(cubic_y),
            tree_density_spline: self.tree_density_spline.eval_at_row(cubic_y),
            spawn_rate_spline: self.spawn_rate_spline.eval_at_row(cubic_y),
            near_water_spline: self.near_water_spline.eval_at_row(cubic_y),
            // water_vel_spline: self.water_vel_spline.eval_at_row(cubic_y),
            water_vel_spline_x: self.water_vel_spline_x.eval_at_row(cubic_y),
            water_vel_spline_y: self.water_vel_spline_y.eval_at_row(cubic_y),
            water_vel_spline_z: self.water_vel_spline_z.eval_at_row(cubic_y),
            alt_spline: self.alt_spline.eval_at_row(cubic_y),
            alt_spline_mono: self.alt_spline_mono.eval_at_row(cubic_y),
            basement_sub_alt_spline: self.basement_sub_alt_spline.eval_at_row(cubic_y),
            surface_veg_spline: self.surface_veg_spline.eval_at_row(cubic_y),
            cliff_height_spline: self.cliff_height_spline.eval_at_row(cubic_y),
            /* spawn_rules, */
        }
    }
}

impl<'a, 'b> Sampler<'a, 'b> for ColumnGen1D<'a, 'b> {
    type Index = /*(Vec2<i32>, IndexRef<'a>, Option<&'a Calendar>)*//*Vec2<i32>*/i32;
    type Sample = /*Option<*/ColumnSample/*<'a>*//*>*/;

    /// Precondition: wpos should be inside the chunk associated with self.chunk_pos.
    fn get(&self, /*(wpos, index, calendar)*/wpos_col: Self::Index) -> ColumnSample/*<'a>*/ {
        let &Self {
            parent,
            wpos_row,
            wposy,
            cubic_y,
            ..
        } = self;

        let &ColumnGen {
            sim,
            chunk_pos,
            index,
            /* calendar, */
            sim_chunk,
           ..
        } = parent;

        let wpos = Vec2::new(wpos_col, wpos_row);
        let wposf = wpos.map(|e| e as f64);
        // let chunk_pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);

        let rel_chunk_pos = wpos_col - chunk_pos.x * TerrainChunkSize::RECT_SIZE.x as i32;
        // Should be between 0 and 1 if precondition is satisfied.
        let wposx = rel_chunk_pos/*.x*/ as f32 / TerrainChunkSize::RECT_SIZE.x as f32;
        // let wposy = rel_chunk_pos.y as f32 / TerrainChunkSize::RECT_SIZE.y as f32;
        let cubic_x = cubic(wposx);
        // let cubic_y = cubic(wposy);

        // let turb = Vec2::new(
        //     sim.gen_ctx.turb_x_nz.get((wposf.div(48.0)).into_array()) as f32,
        //     sim.gen_ctx.turb_y_nz.get((wposf.div(48.0)).into_array()) as f32,
        // ) * 12.0;
        let wposf_turb = wposf; // + turb.map(|e| e as f64);

        let chaos = self.chaos_spline.eval_at(cubic_x/*, cubic_y*/);
        /* if chunk_pos.x == 600 && chunk_pos.y == 472 {
            dbg!(&);
        } */
        let temp = self.temp_spline.eval_at(cubic_x/*, cubic_y*/);
        let humidity = self.humidity_spline.eval_at(cubic_x/*, cubic_y*/);
        let rockiness = self.rockiness_spline.eval_at(cubic_x/*, cubic_y*/);
        let tree_density = self.tree_density_spline.eval_at(cubic_x/*, cubic_y*/);
        let spawn_rate = self.spawn_rate_spline.eval_at(cubic_x/*, cubic_y*/);
        let near_water = self.near_water_spline.eval_at(cubic_x/*, cubic_y*/);
        // let water_vel = self.water_vel_spline.eval_at(cubic_x/*, cubic_y*/);
        let water_vel = Vec3::new(
            self.water_vel_spline_x.eval_at(cubic_x/*, cubic_y*/),
            self.water_vel_spline_y.eval_at(cubic_x/*, cubic_y*/),
            self.water_vel_spline_z.eval_at(cubic_x/*, cubic_y*/),
        );
        let cliff_height = self.cliff_height_spline.eval_at(cubic_x/*, cubic_y*/);

        // let alt = sim.get_interpolated_monotone/*get_interpolated*/(wpos, |chunk| chunk.alt)?;
        let alt = self.alt_spline_mono.eval_at(cubic_x/*, cubic_y*/);
        // let basement_sub_alt = sim.get_interpolated_monotone/*get_interpolated*/(wpos, |chunk| chunk.basement.sub(chunk.alt))?;
        let basement_sub_alt = self.basement_sub_alt_spline.eval_at(cubic_x/*, cubic_y*/);
        // let surface_veg = sim.get_interpolated_monotone/*get_interpolated*/(wpos, |chunk| chunk.surface_veg)?;
        let surface_veg = self.surface_veg_spline.eval_at(cubic_x/*, cubic_y*/);

        let calendar = sim.calendar.as_ref();

        let neighbor_coef = TerrainChunkSize::RECT_SIZE.map(|e| e as f64);
        let my_chunk_idx = vec2_as_uniform_idx(sim.map_size_lg(), chunk_pos);
        /* let neighbor_river_data =
            local_cells(self.sim.map_size_lg(), my_chunk_idx).filter_map(|neighbor_idx: usize| {
                let neighbor_pos = uniform_idx_as_vec2(self.sim.map_size_lg(), neighbor_idx);
                let neighbor_chunk = sim.get(neighbor_pos)?;
                Some((neighbor_pos, neighbor_chunk, &neighbor_chunk.river))
            }); */

        const SAMP_RES: i32 = 8;
        const SAMP_RES_X: f32 = SAMP_RES as f32 / TerrainChunkSize::RECT_SIZE.x as f32;
        const SAMP_RES_Y: f32 = SAMP_RES as f32 / TerrainChunkSize::RECT_SIZE.y as f32;
        let cubic_delta = |x, delta: f32| cubic(x + SAMP_RES_X);
        let alt_y = &self.alt_spline/*.eval_at_row(cubic_y)*/;
        let altx0 = alt_y.eval_at(cubic_delta(wposx, -SAMP_RES_X));
        let altx1 = alt_y.eval_at(cubic_delta(wposx, SAMP_RES_X));
        let alt_x = parent.alt_spline.eval_at_col(cubic_x);
        let alty0 = alt_x.eval_at(cubic_delta(wposy, -SAMP_RES_Y));
        let alty1 = alt_x.eval_at(cubic_delta(wposy, SAMP_RES_Y));
        let gradient =
            Vec2::new(altx1 - altx0, alty1 - alty0)
                        .map(f32::abs)
                        .magnitude()
                        / SAMP_RES as f32;

        let wposf3d = /*Vec3::new(wposf.x, wposf.y, alt as f64)*/wposf;

        let hill_nz = |wposf: Vec2<f64>| {
            // 0.0f32
            /*(*/sim.gen_ctx.fast_hill_nz/*_fast_turb_x_nz*/.get(wposf.into_array()) as f32/* * 0.5 + 0.5)*//*.sqrt()*/
        };

        let small_nz = |wposf: Vec2<f64>| {
            sim.gen_ctx.small_nz.get(wposf.into_array()) as f32
            // 0.0f32
        };

        let marble_small = hill_nz(wposf3d.div(3.0))
            .powi(3)
            .add(1.0)
            .mul(0.5);
        let marble_mid = hill_nz(wposf3d.div(12.0))
            .mul(0.75)
            .add(1.0)
            .mul(0.5);
        //.add(marble_small.sub(0.5).mul(0.25));
        let marble = hill_nz(wposf3d.div(48.0))
            .mul(0.75)
            .add(1.0)
            .mul(0.5);
        let marble_mixed = marble
            .add(marble_mid.sub(0.5).mul(0.5))
            .add(marble_small.sub(0.5).mul(0.25));

        /* let lake_width = (TerrainChunkSize::RECT_SIZE.x as f64 * 2.0f64.sqrt()) + 6.0;
        let neighbor_river_data = neighbor_river_data
            .map(|(posj, chunkj, river)| {
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
                            error!(?river, ?posj, "What?");
                            panic!("How can a river have no downhill?");
                        },
                        RiverKind::Lake { .. } => {
                            return (posj, chunkj, river, None);
                        },
                        RiverKind::Ocean => posj,
                    }
                };
                let downhill_wpos = downhill_pos.map(|e| e as f64);
                let downhill_pos = downhill_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
                    e.div_euclid(sz as i32)
                });
                let neighbor_wpos = posj.map(|e| e as f64) * neighbor_coef + neighbor_coef * 0.5;
                let direction = neighbor_wpos - downhill_wpos;
                let river_width_min = if let RiverKind::River { cross_section } = kind {
                    cross_section.x as f64
                } else {
                    lake_width
                };
                let downhill_chunk = sim.get(downhill_pos).expect("How can this not work?");
                let coeffs = river_spline_coeffs(
                    neighbor_wpos,
                    chunkj.river.spline_derivative,
                    downhill_wpos,
                );
                let (direction, coeffs, downhill_chunk, river_t, river_pos, river_dist) = match kind
                {
                    RiverKind::River { .. } => {
                        if let Some((t, pt, dist)) = quadratic_nearest_point(
                            &coeffs,
                            wposf,
                            Vec2::new(neighbor_wpos, downhill_wpos),
                        ) {
                            let (t, pt, dist) = if dist > wposf.distance_squared(neighbor_wpos) {
                                (0.0, neighbor_wpos, wposf.distance_squared(neighbor_wpos))
                            } else if dist > wposf.distance_squared(downhill_wpos) {
                                (1.0, downhill_wpos, wposf.distance_squared(downhill_wpos))
                            } else {
                                (t, pt, dist)
                            };
                            (direction, coeffs, downhill_chunk, t, pt, dist.sqrt())
                        } else {
                            let ndist = wposf.distance_squared(neighbor_wpos);
                            let ddist = wposf.distance_squared(downhill_wpos);
                            let (closest_pos, closest_dist, closest_t) = if ndist <= ddist {
                                (neighbor_wpos, ndist, 0.0)
                            } else {
                                (downhill_wpos, ddist, 1.0)
                            };
                            (
                                direction,
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
                                neighbor_wpos
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
                                neighbor_wpos
                                    .map2(TerrainChunkSize::RECT_SIZE, |f, g| (f as i32, g as i32)),
                                |e, (f, g)| ((e - f) / g).abs(),
                            )
                            .reduce_partial_max();
                        if pass_dist > 1 {
                            return (posj, chunkj, river, None);
                        }
                        let neighbor_pass_wpos =
                            neighbor_pass_pos.map(|e| e as f64) + neighbor_coef * 0.5;
                        let neighbor_pass_pos = neighbor_pass_pos
                            .map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / sz as i32);
                        let coeffs = river_spline_coeffs(
                            neighbor_wpos,
                            spline_derivative,
                            neighbor_pass_wpos,
                        );
                        let direction = neighbor_wpos - neighbor_pass_wpos;

                        // Lakes get a special distance function to avoid cookie-cutter edges
                        if matches!(
                            downhill_chunk.river.river_kind,
                            Some(RiverKind::Lake { .. } | RiverKind::Ocean)
                        ) {
                            let water_chunk = posj.map(|e| e as f64);
                            let lake_width_noise = sim
                                .gen_ctx
                                .small_nz
                                .get((wposf.map(|e| e as f64).div(32.0)).into_array());
                            let water_aabr = Aabr {
                                min: water_chunk * neighbor_coef + 4.0 - lake_width_noise * 8.0,
                                max: (water_chunk + 1.0) * neighbor_coef - 4.0
                                    + lake_width_noise * 8.0,
                            };
                            let pos = water_aabr.projected_point(wposf);
                            (
                                direction,
                                coeffs,
                                sim.get(neighbor_pass_pos).expect("Must already work"),
                                0.5,
                                pos,
                                pos.distance(wposf),
                            )
                        } else if let Some((t, pt, dist)) = quadratic_nearest_point(
                            &coeffs,
                            wposf,
                            Vec2::new(neighbor_wpos, neighbor_pass_wpos),
                        ) {
                            (
                                direction,
                                coeffs,
                                sim.get(neighbor_pass_pos).expect("Must already work"),
                                t,
                                pt,
                                dist.sqrt(),
                            )
                        } else {
                            let ndist = wposf.distance_squared(neighbor_wpos);
                            /* let ddist = wposf.distance_squared(neighbor_pass_wpos); */
                            let (closest_pos, closest_dist, closest_t) = /*if ndist <= ddist */ {
                                (neighbor_wpos, ndist, 0.0)
                            } /* else {
                                (neighbor_pass_wpos, ddist, 1.0)
                            } */;
                            (
                                direction,
                                coeffs,
                                sim.get(neighbor_pass_pos).expect("Must already work"),
                                closest_t,
                                closest_pos,
                                closest_dist.sqrt(),
                            )
                        }
                    },
                    RiverKind::Ocean => {
                        let water_chunk = posj.map(|e| e as f64);
                        let lake_width_noise = sim
                            .gen_ctx
                            .small_nz
                            .get((wposf.map(|e| e as f64).div(32.0)).into_array());
                        let water_aabr = Aabr {
                            min: water_chunk * neighbor_coef + 4.0 - lake_width_noise * 8.0,
                            max: (water_chunk + 1.0) * neighbor_coef - 4.0 + lake_width_noise * 8.0,
                        };
                        let pos = water_aabr.projected_point(wposf);
                        (
                            direction,
                            coeffs,
                            sim.get(posj).expect("Must already work"),
                            0.5,
                            pos,
                            pos.distance(wposf),
                        )
                    },
                };
                let river_width_max = if let Some(RiverKind::River { cross_section }) =
                    downhill_chunk.river.river_kind
                {
                    // Harmless hack that prevents a river growing wildly outside its bounds to
                    // create water walls
                    (cross_section.x as f64).min(river_width_min * 1.75)
                } else if let Some(RiverKind::River { cross_section }) = chunkj.river.river_kind {
                    Lerp::lerp(cross_section.x as f64, lake_width, 0.5)
                } else {
                    // 0.5 prevents rivers pooling into lakes having extremely wide bounds, creating
                    // water walls
                    lake_width * 0.5
                };
                let river_width_noise =
                    (sim.gen_ctx.small_nz.get((river_pos.div(16.0)).into_array()))
                        .max(-1.0)
                        .min(1.0)
                        .mul(0.5)
                        .sub(0.5) as f64;
                let river_width = Lerp::lerp(
                    river_width_min,
                    river_width_max,
                    river_t.clamped(0.0, 1.0).powf(3.0),
                );

                let river_width = river_width.max(2.0) * (1.0 + river_width_noise * 0.3);
                // To find the distance, we just evaluate the quadratic equation at river_t and
                // see if it's within width (but we should be able to use it for a
                // lot more, and this probably isn't the very best approach anyway
                // since it will bleed out). let river_pos = coeffs.x * river_t *
                // river_t + coeffs.y * river_t + coeffs.z;
                // let river_width = 32.0f64;
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
            })
            .collect::<Vec<_>>();

        debug_assert!(sim_chunk.water_alt >= CONFIG.sea_level); */

        /// A type that makes managing surface altitude weighting much simpler.
        #[derive(Default)]
        struct WeightedSum<T> {
            sum: T,
            weight: T,
            min: Option<T>,
            max: Option<T>,
        }
        impl WeightedSum<f32> {
            /// Add a weight to the sum.
            fn with(self, value: f32, weight: f32) -> Self {
                Self {
                    sum: self.sum + value * weight,
                    weight: self.weight + weight,
                    ..self
                }
            }

            /// Add an upper bound to the result.
            fn with_min(self, min: f32) -> Self {
                Self {
                    min: Some(self.min.unwrap_or(min).min(min)),
                    ..self
                }
            }

            /// Add a lower bound to the result.
            fn with_max(self, max: f32) -> Self {
                Self {
                    max: Some(self.max.unwrap_or(max).max(max)),
                    ..self
                }
            }

            /// Evaluate the weighted sum, if weightings were applied.
            fn eval(&self) -> Option<f32> {
                if self.weight > 0.0 {
                    let res = self.sum / self.weight;
                    let res = self.min.map_or(res, |m| m.min(res));
                    let res = self.max.map_or(res, |m| m.max(res));
                    Some(res)
                } else {
                    None
                }
            }

            /// Evaluate the weighted sum, or use a default value if no
            /// weightings were applied.
            fn eval_or(&self, default: f32) -> f32 {
                let res = if self.weight > 0.0 {
                    self.sum / self.weight
                } else {
                    default
                };
                let res = self.min.map_or(res, |m| m.min(res));
                self.max.map_or(res, |m| m.max(res))
            }
        }

        /// Determine whether a river should become a waterfall
        fn is_waterfall(
            chunk_pos: Vec2<i32>,
            river_chunk: &SimChunk,
            downhill_chunk: &SimChunk,
        ) -> bool {
            // Waterfalls are rare, so use some hacky RNG seeded with the position to
            // reflect that. Additionally, the river must experience a rapid
            // change in elevation. Pooling into a lake produces a rapid.
            // TODO: Find a better way to produce rapids along the course of a river?
            (RandomField::new(3119).chance(chunk_pos.with_z(0), 0.1)
                || matches!(
                    downhill_chunk.river.river_kind,
                    Some(RiverKind::Lake { .. })
                ))
                && (river_chunk.water_alt > downhill_chunk.water_alt + 0.0)
        }

        /// Determine the altitude of a river based on the altitude of the
        /// spline ends and a tweening factor.
        fn river_water_alt(a: f32, b: f32, t: f32, is_waterfall: bool) -> f32 {
            let t = if is_waterfall {
                // Waterfalls bias the water altitude toward extremes
                power(t as f64, 3.0 + (a - b).clamped(0.0, 16.0) as f64) as f32
            } else {
                t
            };
            Lerp::lerp(a, b, t)
        }

        // Use this to temporarily alter the sea level
        let base_sea_level = CONFIG.sea_level - 1.0 + 0.01;

        /* // What's going on here?
        //
        // We're iterating over nearby bodies of water and calculating a weighted sum
        // for the river water level, the lake water level, and the 'unbounded
        // water level' (the maximum water body altitude, which we use later to
        // prevent water walls). In doing so, we also apply various clamping strategies
        // to catch lots of nasty edge cases, as well as calculating the
        // distance to the nearest body of water.
        //
        // The clamping strategies employed prevent very specific, annoying artifacts
        // such as 'water walls' (vertical columns of water that are physically
        // implausible) and 'backflows' (regions where a body of water appears to
        // flow upstream due to irregular humps along its course).
        //
        // It is incredibly difficult to explain exactly what every part of this code is
        // doing without visual examples. Needless to say, any changes to this
        // code *at all* should be very ruggedly tested to ensure that
        // they do not result in artifacts, even in edge cases. The exact configuration
        // of this code is the product of hundreds of hours of testing and
        // refinement and I ask that you do not take that effort lightly.
        let (
            river_water_level,
            in_river,
            lake_water_level,
            lake_dist,
            water_dist,
            unbounded_water_level,
        ) = neighbor_river_data.iter().copied().fold(
            (
                WeightedSum::default().with_max(base_sea_level),
                false,
                WeightedSum::default().with_max(base_sea_level),
                10000.0f32,
                None,
                WeightedSum::default().with_max(base_sea_level),
            ),
            |(
                mut river_water_level,
                mut in_river,
                lake_water_level,
                mut lake_dist,
                water_dist,
                mut unbounded_water_level,
            ),
             (river_chunk_idx, river_chunk, river, dist_info)| match (
                river.river_kind,
                dist_info,
            ) {
                (
                    Some(kind),
                    Some((_, _, river_width, (river_t, (river_pos, _), downhill_chunk))),
                ) => {
                    // Distance from river center
                    let river_dist = river_pos.distance(wposf);
                    // Distance from edge of river
                    let river_edge_dist = (river_dist - river_width * 0.5).max(0.0) as f32;
                    // 0.0 = not near river, 1.0 = in middle of river
                    let near_center = ((river_dist / (river_width * 0.5)) as f32)
                        .min(1.0)
                        .mul(f32::consts::PI)
                        .cos()
                        .add(1.0)
                        .mul(0.5);

                    match kind {
                        RiverKind::River { .. } => {
                            // Alt of river water *is* the alt of land (ignoring gorge, which gets
                            // applied later)
                            let river_water_alt = river_water_alt(
                                river_chunk.alt.max(river_chunk.water_alt),
                                downhill_chunk.alt.max(downhill_chunk.water_alt),
                                river_t as f32,
                                is_waterfall(river_chunk_idx, river_chunk, downhill_chunk),
                            );

                            river_water_level =
                                river_water_level.with(river_water_alt, near_center);

                            if river_edge_dist <= 0.0 {
                                in_river = true;
                            }
                        },
                        // Slightly wider threshold is chosen in case the lake bounds are a bit
                        // wrong
                        RiverKind::Lake { .. } | RiverKind::Ocean => {
                            let lake_water_alt = if matches!(kind, RiverKind::Ocean) {
                                base_sea_level
                            } else {
                                river_water_alt(
                                    river_chunk.alt.max(river_chunk.water_alt),
                                    downhill_chunk.alt.max(downhill_chunk.water_alt),
                                    river_t as f32,
                                    is_waterfall(river_chunk_idx, river_chunk, downhill_chunk),
                                )
                            };

                            if river_edge_dist > 0.0 && river_width > lake_width * 0.99 {
                                let unbounded_water_alt = lake_water_alt
                                    - ((river_edge_dist - 8.0).max(0.0) / 5.0).powf(2.0);
                                unbounded_water_level = unbounded_water_level
                                    .with(unbounded_water_alt, 1.0 / (1.0 + river_edge_dist * 5.0));
                                //.with_max(unbounded_water_alt);
                            }

                            river_water_level = river_water_level.with(lake_water_alt, near_center);

                            lake_dist = lake_dist.min(river_edge_dist);

                            // Lake border prevents a lake failing to propagate its altitude to
                            // nearby rivers
                            let off = 0.0;
                            let len = 3.0;
                            if river_edge_dist <= off {
                                // lake_water_level = lake_water_level
                                //     // Make sure the closest lake is prioritised
                                //     .with(lake_water_alt, near_center + 0.1 / (1.0 +
                                // river_edge_dist));     // .with_min(lake_water_alt);
                                //
                                river_water_level = river_water_level.with_min(
                                    lake_water_alt
                                        + ((((river_dist - river_width * 0.5) as f32 + len - off)
                                            .max(0.0))
                                            / len)
                                            .powf(1.5)
                                            * 32.0,
                                );
                            }
                        },
                    };

                    let river_edge_dist_unclamped = (river_dist - river_width * 0.5) as f32;
                    let water_dist = Some(
                        water_dist
                            .unwrap_or(river_edge_dist_unclamped)
                            .min(river_edge_dist_unclamped),
                    );

                    (
                        river_water_level,
                        in_river,
                        lake_water_level,
                        lake_dist,
                        water_dist,
                        unbounded_water_level,
                    )
                },
                (_, _) => (
                    river_water_level,
                    in_river,
                    lake_water_level,
                    lake_dist,
                    water_dist,
                    unbounded_water_level,
                ),
            },
        );
        let unbounded_water_level = unbounded_water_level.eval_or(base_sea_level);
        // Calculate a final, canonical altitude for the water in this column by
        // combining and clamping the attributes we found while iterating over
        // nearby bodies of water.
        let water_level = match (
            river_water_level.eval(),
            lake_water_level
                .eval()
                .filter(|_| lake_dist <= 0.0 || in_river),
        ) {
            (Some(r), Some(l)) => r.max(l),
            (r, l) => r.or(l).unwrap_or(base_sea_level).max(unbounded_water_level),
        }
        .max(base_sea_level); */
        let water_level = sim_chunk.water_alt;
        let water_dist = None::<f32>;

        let riverless_alt = alt;

        // What's going on here?
        //
        // Now that we've figured out the altitude of the water in this column, we can
        // determine the altitude of the river banks. This initially appears
        // somewhat backward (surely the river basin determines the water level?)
        // but it is necessary to prevent backflows. Here, the surface of the water is
        // king because we require global information to determine it without
        // backflows. The river banks simply reflect the will of the water. We care
        // much less about a river bank that's slightly rugged and irregular than we do
        // about the surface of the water itself being rugged and irregular (and
        // hence physically implausible). From that perspective, it makes sense
        // that we determine river banks after the water level because it is the one
        // that we are most at liberty to screw up.
        //
        // Similar to the iteration above, we perform a fold over nearby bodies of water
        // and use the distance to the water to come up wight a weighted sum for
        // the altitude. The way we determine this altitude differs somewhat
        // between rivers, lakes, and the ocean and also whether we are *inside* said
        // bodies of water or simply near their edge.
        //
        // As with the previous iteration, a lot of this code is extremely delicate and
        // has been carefully designed to handle innumeral edge cases. Please
        // test any changes to this code extremely well to avoid regressions: some
        // edge cases are very rare indeed!
        /* let alt = neighbor_river_data.into_iter().fold(
            WeightedSum::default().with(riverless_alt, 1.0),
            |alt, (river_chunk_idx, river_chunk, river, dist_info)| match (
                river.river_kind,
                dist_info,
            ) {
                (
                    Some(kind),
                    Some((_, _, river_width, (river_t, (river_pos, _), downhill_chunk))),
                ) => {
                    // Distance from river center
                    let river_dist = river_pos.distance(wposf);
                    // Distance from edge of river
                    let river_edge_dist = (river_dist - river_width * 0.5).max(0.0) as f32;

                    let water_alt = match kind {
                        RiverKind::River { cross_section } => {
                            // Alt of river water *is* the alt of land
                            let river_water_alt = river_water_alt(
                                river_chunk.alt.max(river_chunk.water_alt),
                                downhill_chunk.alt.max(downhill_chunk.water_alt),
                                river_t as f32,
                                is_waterfall(river_chunk_idx, river_chunk, downhill_chunk),
                            );
                            Some((river_water_alt, cross_section.y as f32, None))
                        },
                        RiverKind::Lake { .. } | RiverKind::Ocean => {
                            let lake_water_alt = if matches!(kind, RiverKind::Ocean) {
                                base_sea_level
                            } else {
                                river_water_alt(
                                    river_chunk.alt.max(river_chunk.water_alt),
                                    downhill_chunk.alt.max(downhill_chunk.water_alt),
                                    river_t as f32,
                                    is_waterfall(river_chunk_idx, river_chunk, downhill_chunk),
                                )
                            };

                            let depth = water_level
                                - Lerp::lerp(
                                    riverless_alt.min(water_level),
                                    water_level - 4.0,
                                    0.5,
                                );

                            let min_alt = Lerp::lerp(
                                riverless_alt,
                                lake_water_alt,
                                ((river_dist / (river_width * 0.5) - 0.5) * 2.0).clamped(0.0, 1.0)
                                    as f32,
                            );

                            Some((
                                lake_water_alt,
                                // TODO: The depth given to us by the erosion code is technically
                                // correct, but it also
                                // looks terrible. Come up with a good solution to this.
                                /* river_width as f32 * 0.15 */
                                depth,
                                Some(min_alt),
                            ))
                        },
                    };

                    const BANK_STRENGTH: f32 = 100.0;
                    if let Some((water_alt, water_depth, min_alt)) = water_alt {
                        if river_edge_dist <= 0.0 {
                            const MIN_DEPTH: f32 = 1.0;
                            let near_center = ((river_dist / (river_width * 0.5)) as f32)
                                .min(1.0)
                                .mul(f32::consts::PI)
                                .cos()
                                .add(1.0)
                                .mul(0.5);
                            // Waterfalls 'boost' the depth of the river to prevent artifacts. This
                            // is also necessary when rivers become very
                            // steep without explicitly being waterfalls.
                            // TODO: Come up with a more principled way of doing this without
                            // guessing magic numbers
                            let waterfall_boost =
                                if is_waterfall(river_chunk_idx, river_chunk, downhill_chunk) {
                                    (river_chunk.alt - downhill_chunk.alt).max(0.0).powf(2.0)
                                        * (1.0 - (river_t as f32 - 0.5).abs() * 2.0).powf(3.5)
                                        / 20.0
                                } else {
                                    // Handle very steep rivers gracefully
                                    (river_chunk.alt - downhill_chunk.alt).max(0.0) * 2.0
                                        / TerrainChunkSize::RECT_SIZE.x as f32
                                };
                            let riverbed_depth =
                                near_center * water_depth + MIN_DEPTH + waterfall_boost;
                            // Handle rivers debouching into the ocean nicely by 'flattening' their
                            // bottom
                            let riverbed_alt = (water_alt - riverbed_depth)
                                .max(riverless_alt.min(base_sea_level - MIN_DEPTH));
                            alt.with(
                                min_alt.unwrap_or(riverbed_alt).min(riverbed_alt),
                                near_center * BANK_STRENGTH,
                            )
                            .with_min(min_alt.unwrap_or(riverbed_alt).min(riverbed_alt))
                        } else {
                            const GORGE: f32 = 0.25;
                            const BANK_SCALE: f32 = 24.0;
                            // Weighting of this riverbank on nearby terrain (higher when closer to
                            // the river). This 'pulls' the riverbank
                            // toward the river's altitude to make sure that we get a smooth
                            // transition from normal terrain to the water.
                            let weight = Lerp::lerp(
                                BANK_STRENGTH
                                    / (1.0
                                        + (river_edge_dist as f32 - 3.0).max(0.0) * BANK_STRENGTH
                                            / BANK_SCALE),
                                0.0,
                                power((river_edge_dist / BANK_SCALE).clamped(0.0, 1.0) as f64, 2.0)
                                    as f32,
                            );
                            let alt = alt.with(water_alt + GORGE, weight);

                            let alt = if matches!(kind, RiverKind::Ocean) {
                                alt
                            } else if (0.0..1.5).contains(&river_edge_dist)
                                && water_dist.map_or(false, |d| d >= 0.0)
                            {
                                alt.with_max(water_alt + GORGE)
                            } else {
                                alt
                            };

                            if matches!(kind, RiverKind::Ocean) {
                                alt
                            } else if lake_dist > 0.0 && water_level < unbounded_water_level {
                                alt.with_max(unbounded_water_level)
                            } else {
                                alt
                            }
                        }
                    } else {
                        alt
                    }
                },
                (_, _) => alt,
            },
        );
        let alt = alt
            .eval_or(riverless_alt)
            .max(if water_dist.map_or(true, |d| d > 0.0) {
                // Terrain below sea level breaks things, so force it to never happen
                base_sea_level + 0.5
            } else {
                f32::MIN
            }); */

        let riverless_alt_delta = small_nz(wposf_turb.div(200.0 * (32.0 / TerrainChunkSize::RECT_SIZE.x as f64)))
            .min(1.0)
            .max(-1.0)
            .abs()
            .mul(3.0)
            + small_nz(wposf_turb.div(400.0 * (32.0 / TerrainChunkSize::RECT_SIZE.x as f64)))
                .min(1.0)
                .max(-1.0)
                .abs()
                .mul(3.0);

        // Cliffs
        let cliff_factor = (alt
            + hill_nz(wposf.div(64.0)) * 8.0
            + hill_nz(wposf.div(350.0)) * 128.0)
            .rem_euclid(200.0)
            / 64.0
            - 1.0;
        let cliff_scale =
            (hill_nz(wposf.div(128.0)) * 1.5 + 0.75 +
             hill_nz(wposf.div(48.0)) * 0.1)
                .clamped(0.0, 1.0)
                .powf(2.0);
        let cliff_height = cliff_height * cliff_scale;
        let cliff = if cliff_factor < 0.0 {
            cliff_factor.abs().powf(1.5)
        } else {
            0.0
        } * (1.0 - near_water * 3.0).max(0.0).powi(2);
        let cliff_offset = cliff * cliff_height;
        let riverless_alt_delta = riverless_alt_delta + (cliff - 0.5) * cliff_height;

        let warp_factor = water_dist.map_or(1.0, |d| ((d - 0.0) / 64.0).clamped(0.0, 1.0));

        // NOTE: To disable warp, uncomment this line.
        // let warp_factor = 0.0;

        let surface_rigidity = 1.0 - temp.max(0.0) * (1.0 - tree_density);
        let surface_rigidity =
            surface_rigidity.max(((basement_sub_alt + 3.0) / 1.5).clamped(0.0, 2.0));

        // Adjust this to make rock placement better
        let rock_density = rockiness
            + water_dist
                .filter(|wd| *wd > 2.0)
                .map(|wd| (1.0 - wd / 32.0).clamped(0.0, 1.0).powf(0.5) * 10.0)
                .unwrap_or(0.0);

        // Columns near water have a more stable temperature and so get pushed towards
        // the average (0)
        let temp = Lerp::lerp(
            Lerp::lerp(temp, 0.0, 0.1),
            temp,
            water_dist
                .map(|water_dist| water_dist / 20.0)
                .unwrap_or(1.0)
                .clamped(0.0, 1.0),
        );
        // Columns near water get a humidity boost
        let humidity = Lerp::lerp(
            Lerp::lerp(humidity, 1.0, 0.25),
            humidity,
            water_dist
                .map(|water_dist| water_dist / 20.0)
                .unwrap_or(1.0)
                .clamped(0.0, 1.0),
        );

        // Colours
        let Colors {
            cold_grass,
            warm_grass,
            dark_grass,
            wet_grass,
            cold_stone,
            hot_stone,
            warm_stone,
            beach_sand,
            desert_sand,
            snow,
            snow_moss,
            stone_col,
            dirt_low,
            dirt_high,
            snow_high,
            warm_stone_high,
            grass_high,
            tropical_high,
        } = index.colors.column;

        let cold_grass = cold_grass.into();
        let warm_grass = warm_grass.into();
        let dark_grass = dark_grass.into();
        let wet_grass = wet_grass.into();
        let cold_stone = cold_stone.into();
        let hot_stone = hot_stone.into();
        let warm_stone: Rgb<f32> = warm_stone.into();
        let beach_sand = beach_sand.into();
        let desert_sand = desert_sand.into();
        let snow = snow.into();
        let stone_col = stone_col.into();
        let dirt_low: Rgb<f32> = dirt_low.into();
        let dirt_high = dirt_high.into();
        let snow_high = snow_high.into();
        let warm_stone_high = warm_stone_high.into();
        let grass_high = grass_high.into();
        let tropical_high = tropical_high.into();

        let dirt = Lerp::lerp(dirt_low, dirt_high, marble_mixed);
        let tundra = Lerp::lerp(snow, snow_high, 0.4 + marble_mixed * 0.6);
        let dead_tundra = Lerp::lerp(warm_stone, warm_stone_high, marble_mixed);
        let cliff = Rgb::lerp(cold_stone, hot_stone, marble_mixed);

        let grass = Rgb::lerp(
            cold_grass,
            warm_grass,
            marble_mixed
                .sub(0.5)
                .add(1.0.sub(humidity).mul(0.5))
                .powf(1.5),
        );
        let snow_moss = Rgb::lerp(
            snow_moss.into(),
            cold_grass,
            0.4 + marble_mixed.powf(1.5) * 0.6,
        );
        let moss = Rgb::lerp(dark_grass, cold_grass, marble_mixed.powf(1.5));
        let rainforest = Rgb::lerp(wet_grass, warm_grass, marble_mixed.powf(1.5));
        let sand = Rgb::lerp(beach_sand, desert_sand, marble_mixed);

        let tropical = Rgb::lerp(
            Rgb::lerp(
                grass,
                grass_high,
                marble_small
                    .sub(0.5)
                    .mul(0.2)
                    .add(0.75.mul(1.0.sub(humidity)))
                    .powf(0.667),
            ),
            tropical_high,
            marble_mixed.powf(1.5).sub(0.5).mul(4.0),
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

        let sub_surface_color = Lerp::lerp(cliff, ground, basement_sub_alt.mul(-0.25));

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
                .mul(1.25),
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
        let thematic_snow = calendar.map_or(false, |c| c.is_event(CalendarEvent::Christmas));
        let snow_factor = temp
            .sub(if thematic_snow {
                CONFIG.tropical_temp
            } else {
                CONFIG.snow_temp
            })
            .max(-humidity.sub(CONFIG.desert_hum))
            .mul(4.0)
            .max(-0.25)
            // 'Simulate' avalanches moving snow from areas with high gradients to areas with high flux
            .add((gradient/*.unwrap_or(0.0)*/ - 0.5).max(0.0) * 0.1)
            // .add(-flux * 0.003 * gradient.unwrap_or(0.0))
            .add(((marble - 0.5) / 0.5) * 0.25)
            .add(((marble_mid - 0.5) / 0.5) * 0.125)
            .add(((marble_small - 0.5) / 0.5) * 0.0625);
        let snow_cover = snow_factor <= 0.0;

        let spawn_rules = /*sim_chunk
            .sites
            .iter()
            .map(|site| index.sites[*site].spawn_rules(wpos))
            .fold(SpawnRules::default(), |a, b| a.combine(b))*/SpawnRules::default();
        let warp_factor = warp_factor * spawn_rules.max_warp.sqrt();
        let warp = ((marble_mid * 0.2 + marble * 0.8) * 2.0 - 1.0)
            * 15.0
            * gradient/*.unwrap_or(0.0)*/.min(1.0)
            * surface_rigidity
            * warp_factor;
        let riverless_alt_delta = Lerp::lerp(0.0, riverless_alt_delta, warp_factor);
        let warp = riverless_alt_delta + warp;
        let alt = alt + warp;

        let (snow_delta, ground, sub_surface_color) = if snow_cover && alt > water_level {
            // Allow snow cover.
            (
                1.0 - snow_factor.max(0.0),
                Rgb::lerp(snow, ground, snow_factor),
                Lerp::lerp(sub_surface_color, ground, basement_sub_alt.mul(-0.15)),
            )
        } else {
            (0.0, ground, sub_surface_color)
        };
        let alt = alt + snow_delta;
        let basement_sub_alt = basement_sub_alt - snow_delta;

        // Make river banks not have grass
        let ground = water_dist
            .map(|wd| Lerp::lerp(sub_surface_color, ground, (wd / 3.0).clamped(0.0, 1.0)))
            .unwrap_or(ground);

        // Ground under thick trees should be receive less sunlight and so often become
        // dirt
        let ground = Lerp::lerp(ground, sub_surface_color, marble_mid * tree_density);

        let ice_depth = if snow_factor < -0.25
            && water_vel.magnitude_squared() < (0.1f32 + marble_mid * 0.2).powi(2)
        {
            let cliff = hill_nz(wposf3d.div(180.0))
                .add((marble_mid - 0.5) * 0.2)
                .abs()
                .powi(3)
                .mul(32.0);
            let cliff_ctrl = hill_nz(wposf3d.div(128.0))
                .sub(0.4)
                .add((marble_mid - 0.5) * 0.2)
                .mul(32.0)
                .clamped(0.0, 1.0);

            (((1.0 - Lerp::lerp(marble, Lerp::lerp(marble_mid, marble_small, 0.25), 0.5)) * 5.0
                - 1.5)
                .max(0.0)
                + cliff * cliff_ctrl)
                .min((water_level - alt).max(0.0))
        } else {
            0.0
        };

        ColumnSample {
            alt,
            riverless_alt,
            basement: alt + basement_sub_alt,
            chaos,
            water_level,
            // warp_factor,
            surface_color: Rgb::lerp(
                sub_surface_color,
                Rgb::lerp(
                    // Beach
                    Rgb::lerp(cliff, sand, basement_sub_alt.mul(-0.25)),
                    // Land
                    ground,
                    ((alt - base_sea_level) / 12.0).clamped(0.0, 1.0),
                ),
                surface_veg,
            ),
            sub_surface_color,
            // No growing directly on bedrock.
            // And, no growing on sites that don't want them TODO: More precise than this when we
            // apply trees as a post-processing layer
            tree_density: if spawn_rules.trees {
                Lerp::lerp(0.0, tree_density, -1.0 - basement_sub_alt * 0.5)
            } else {
                0.0
            },
            // forest_kind: sim_chunk.forest_kind,
            // marble,
            // marble_mid,
            // marble_small,
            rock_density,
            temp,
            humidity,
            spawn_rate,
            stone_col,
            water_dist,
            gradient,
            path: /* if spawn_rules.paths {
                sim.get_nearest_path(wpos)
            } else {*/
                None
            /*}*/,
            cave: /*sim.get_nearest_cave(wpos)*/None,
            snow_cover,
            cliff_offset,
            cliff_height,
            // water_vel,
            ice_depth,

            // chunk: core::marker::PhantomData,
            /* chunk: sim_chunk, */

            // ..ColumnSample::default()
        }
    }
}

impl<'a, 'b> Sampler<'a, 'b> for ColumnGen<'a> {
    type Index = /*(Vec2<i32>, IndexRef<'a>, Option<&'a Calendar>)*/Vec2<i32>;
    type Sample = ColumnSample/*<'a>*/;

    /// Precondition: wpos should be inside the chunk associated with self.chunk_pos.
    fn get(&self, /*(wpos, index, calendar)*/wpos: Self::Index) -> ColumnSample/*<'a>*/ {
        let col = self.eval_at_row(wpos.y);
        col.get(wpos.x)
    }
}

#[derive(Clone, Default)]
pub struct ColumnSample/*<'a>*/ {
    pub alt: f32,
    pub riverless_alt: f32,
    pub basement: f32,
    pub chaos: f32,
    pub water_level: f32,
    // pub warp_factor: f32,
    pub surface_color: Rgb<f32>,
    pub sub_surface_color: Rgb<f32>,
    pub tree_density: f32,
    // pub forest_kind: ForestKind,
    // pub marble: f32,
    // pub marble_mid: f32,
    // pub marble_small: f32,
    pub rock_density: f32,
    pub temp: f32,
    pub humidity: f32,
    pub spawn_rate: f32,
    pub stone_col: Rgb<u8>,
    pub water_dist: Option<f32>,
    pub gradient: /*Option<*/f32/*>*/,
    pub path: Option<(f32, Vec2<f32>, Path, Vec2<f32>)>,
    pub cave: Option<(f32, Vec2<f32>, Cave, Vec2<f32>)>,
    pub snow_cover: bool,
    pub cliff_offset: f32,
    pub cliff_height: f32,
    // pub water_vel: Vec3<f32>,
    pub ice_depth: f32,

    /* pub chunk: &'a SimChunk, */
    // pub chunk: core::marker::PhantomData<&'a SimChunk>,
}
