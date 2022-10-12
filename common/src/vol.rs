use crate::{ray::Ray, volumes::scaled::Scaled};
use std::fmt::Debug;
use vek::*;

/// Used to specify a volume's compile-time size. This exists as a substitute
/// until const generics are implemented.
pub trait VolSize: Clone {
    const SIZE: Vec3<u32>;
}

pub trait RectVolSize: Clone {
    const RECT_SIZE: Vec2<u32>;
}

/// A voxel.
pub trait Vox: Sized + Clone + PartialEq {
    fn empty() -> Self;
    fn is_empty(&self) -> bool;

    #[must_use]
    fn or(self, other: Self) -> Self { if self.is_empty() { other } else { self } }
}

/// A volume that contains voxel data.
pub trait BaseVol {
    type Vox;
    type Error: Debug;

    fn scaled_by(self, scale: Vec3<f32>) -> Scaled<Self>
    where
        Self: Sized,
    {
        Scaled { inner: self, scale }
    }
}

/// Implementing `BaseVol` for any `&'a BaseVol` makes it possible to implement
/// `IntoVolIterator` for references.
impl<'a, T: BaseVol> BaseVol for &'a T {
    type Error = T::Error;
    type Vox = T::Vox;
}

// Utility types

/// A volume that is a cuboid.
pub trait SizedVol: BaseVol {
    /// Returns the (inclusive) lower bound of the volume.
    fn lower_bound(&self) -> Vec3<i32>;

    /// Returns the (exclusive) upper bound of the volume.
    fn upper_bound(&self) -> Vec3<i32>;

    /// Returns the size of the volume.
    fn size(&self) -> Vec3<u32> { (self.upper_bound() - self.lower_bound()).map(|e| e as u32) }
}

/// A volume that is compile-time sized and has its lower bound at `(0, 0, 0)`.
/// The name `RasterableVol` was chosen because such a volume can be used with
/// `VolGrid3d`.
pub trait RasterableVol: BaseVol {
    const SIZE: Vec3<u32>;
}

impl<V: RasterableVol> SizedVol for V {
    fn lower_bound(&self) -> Vec3<i32> { Vec3::zero() }

    fn upper_bound(&self) -> Vec3<i32> { V::SIZE.map(|e| e as i32) }
}

/// A volume whose cross section with the XY-plane is a rectangle.
pub trait RectSizedVol: BaseVol {
    fn lower_bound_xy(&self) -> Vec2<i32>;

    fn upper_bound_xy(&self) -> Vec2<i32>;

    fn size_xy(&self) -> Vec2<u32> {
        (self.upper_bound_xy() - self.lower_bound_xy()).map(|e| e as u32)
    }
}

/// A volume that is compile-time sized in x and y direction and has its lower
/// bound at `(0, 0, z)`. In z direction there's no restriction on the lower
/// or upper bound. The name `RectRasterableVol` was chosen because such a
/// volume can be used with `VolGrid2d`.
pub trait RectRasterableVol: BaseVol {
    const RECT_SIZE: Vec2<u32>;
}

impl<V: RectRasterableVol> RectSizedVol for V {
    fn lower_bound_xy(&self) -> Vec2<i32> { Vec2::zero() }

    fn upper_bound_xy(&self) -> Vec2<i32> { V::RECT_SIZE.map(|e| e as i32) }
}

/// A volume that provides read access to its voxel data.
pub trait ReadVol: BaseVol {
    /// Get a reference to the voxel at the provided position in the volume.
    fn get(&self, pos: Vec3<i32>) -> Result<&Self::Vox, Self::Error>;

    /// Get a reference to the voxel at the provided position in the volume. Many volumes provide a fast path,
    /// provided the position is always in-bounds. Note that this function is still safe.
    fn get_unchecked(&self, pos: Vec3<i32>) -> &Self::Vox { self.get(pos).unwrap() }

    /// NOTE: By default, this ray will simply run from `from` to `to` without
    /// stopping.  To make something interesting happen, call `until` or
    /// `for_each`.
    fn ray(
        &self,
        from: Vec3<f32>,
        to: Vec3<f32>,
    ) -> Ray<Self, fn(&Self::Vox) -> bool, fn(&Self::Vox, Vec3<i32>)>
    where
        Self: Sized,
    {
        Ray::new(self, from, to, |_| false)
    }

    /// Call provided closure with each block in the supplied Aabb
    /// Portions of the Aabb outside the volume are ignored
    fn for_each_in(&self, aabb: Aabb<i32>, mut f: impl FnMut(Vec3<i32>, Self::Vox))
    where
        Self::Vox: Copy,
    {
        for z in aabb.min.z..aabb.max.z + 1 {
            for y in aabb.min.y..aabb.max.y + 1 {
                for x in aabb.min.x..aabb.max.x + 1 {
                    if let Ok(block) = self.get(Vec3::new(x, y, z)) {
                        f(Vec3::new(x, y, z), *block);
                    }
                }
            }
        }
    }
}

/// A volume that provides the ability to sample (i.e., clone a section of) its
/// voxel data.
///
/// TODO (haslersn): Do we still need this now that we have `IntoVolIterator`?
pub trait SampleVol<I>: BaseVol {
    type Sample: BaseVol + ReadVol;
    /// Take a sample of the volume by cloning voxels within the provided range.
    ///
    /// Note that value and accessibility of voxels outside the bounds of the
    /// sample is implementation-defined and should not be used.
    ///
    /// Note that the resultant volume has a coordinate space relative to the
    /// sample, not the original volume.
    fn sample(&self, range: I) -> Result<Self::Sample, Self::Error>;
}

/// A volume that provides write access to its voxel data.
pub trait WriteVol: BaseVol {
    /// Set the voxel at the provided position in the volume to the provided
    /// value.
    fn set(&mut self, pos: Vec3<i32>, vox: Self::Vox) -> Result<Self::Vox, Self::Error>;

    /// Map a voxel to another using the provided function.
    // TODO: Is `map` the right name? Implies a change in type.
    fn map<F: FnOnce(Self::Vox) -> Self::Vox>(
        &mut self,
        pos: Vec3<i32>,
        f: F,
    ) -> Result<Self::Vox, Self::Error>
    where
        Self: ReadVol,
        Self::Vox: Clone,
    {
        // This is *deliberately* not using a get_mut since this might trigger a
        // repr change of the underlying volume
        self.set(pos, f(self.get(pos)?.clone()))
    }
}

/// A volume (usually rather a reference to a volume) that is convertible into
/// an iterator to a cuboid subsection of the volume.
pub trait IntoVolIterator<'a>: BaseVol
where
    Self::Vox: 'a,
{
    type IntoIter: Iterator<Item = (Vec3<i32>, &'a Self::Vox)>;

    fn vol_iter(self, lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self::IntoIter;
}

pub trait IntoPosIterator: BaseVol {
    type IntoIter: Iterator<Item = Vec3<i32>>;

    fn pos_iter(self, lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self::IntoIter;
}

// Helpers

/// A volume (usually rather a reference to a volume) that is convertible into
/// an iterator.
pub trait IntoFullVolIterator<'a>: BaseVol
where
    Self::Vox: 'a,
{
    type IntoIter: Iterator<Item = (Vec3<i32>, &'a Self::Vox)>;

    fn full_vol_iter(self) -> Self::IntoIter;
}

/// For any `&'a SizedVol: IntoVolIterator` we implement `IntoFullVolIterator`.
/// Unfortunately we can't just implement `IntoIterator` in this generic way
/// because it's defined in another crate. That's actually the only reason why
/// the trait `IntoFullVolIterator` exists.
// TODO: See whether relaxed orphan rules permit this to be replaced now
impl<'a, T: 'a + SizedVol> IntoFullVolIterator<'a> for &'a T
where
    Self: IntoVolIterator<'a>,
{
    type IntoIter = <Self as IntoVolIterator<'a>>::IntoIter;

    fn full_vol_iter(self) -> Self::IntoIter {
        self.vol_iter(self.lower_bound(), self.upper_bound())
    }
}

pub trait IntoFullPosIterator: BaseVol {
    type IntoIter: Iterator<Item = Vec3<i32>>;

    fn full_pos_iter(self) -> Self::IntoIter;
}

impl<'a, T: 'a + SizedVol> IntoFullPosIterator for &'a T
where
    Self: IntoPosIterator,
{
    type IntoIter = <Self as IntoPosIterator>::IntoIter;

    fn full_pos_iter(self) -> Self::IntoIter {
        self.pos_iter(self.lower_bound(), self.upper_bound())
    }
}

// Defaults

/// Convenience iterator type that can be used to quickly implement
/// `IntoPosIterator`.
pub struct DefaultPosIterator {
    current: Vec3<i32>,
    begin: Vec2<i32>,
    end: Vec3<i32>,
}

impl DefaultPosIterator {
    pub fn new(lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self {
        debug_assert!(lower_bound.map2(upper_bound, |l, u| l <= u).reduce_and());
        let end = if lower_bound.map2(upper_bound, |l, u| l < u).reduce_and() {
            upper_bound
        } else {
            // Special case because our implementation doesn't handle empty ranges for x or
            // y:
            lower_bound
        };
        Self {
            current: lower_bound,
            begin: From::from(lower_bound),
            end,
        }
    }
}

impl Iterator for DefaultPosIterator {
    type Item = Vec3<i32>;

    fn next(&mut self) -> Option<Vec3<i32>> {
        if self.current.z == self.end.z {
            return None;
        }
        let ret = self.current;
        self.current.x += 1;
        if self.current.x == self.end.x {
            self.current.x = self.begin.x;
            self.current.y += 1;
            if self.current.y == self.end.y {
                self.current.y = self.begin.y;
                self.current.z += 1;
            }
        }
        Some(ret)
    }
}

/// Convenience iterator type that can be used to quickly implement
/// `IntoVolIterator`.
pub struct DefaultVolIterator<'a, T: ReadVol> {
    vol: &'a T,
    pos_iter: DefaultPosIterator,
}

impl<'a, T: ReadVol> DefaultVolIterator<'a, T> {
    pub fn new(vol: &'a T, lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self {
        Self {
            vol,
            pos_iter: DefaultPosIterator::new(lower_bound, upper_bound),
        }
    }
}

impl<'a, T: ReadVol> Iterator for DefaultVolIterator<'a, T> {
    type Item = (Vec3<i32>, &'a T::Vox);

    fn next(&mut self) -> Option<(Vec3<i32>, &'a T::Vox)> {
        for pos in &mut self.pos_iter {
            if let Ok(vox) = self.vol.get(pos) {
                return Some((pos, vox));
            }
        }
        None
    }
}

impl<'b, T: ReadVol> ReadVol for &'b T {
    #[inline(always)]
    fn get(&self, pos: Vec3<i32>) -> Result<&'_ Self::Vox, Self::Error> { (*self).get(pos) }
}
