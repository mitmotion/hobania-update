use crate::{
    vol::{
        BaseVol, IntoPosIterator, IntoVolIterator, ReadVol, RectRasterableVol, RectVolSize,
        VolSize, WriteVol,
    },
    volumes::chunk::{Chunk, ChunkError, ChunkPosIter, ChunkVolIter},
};
use core::{hash::Hash, marker::PhantomData};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use vek::*;

#[derive(Debug)]
pub enum ChonkError {
    SubChunkError(ChunkError),
    OutOfBounds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubChunkSize<V, Storage, ChonkSize: RectVolSize> {
    storage: Storage,
    phantom: PhantomData<(V, ChonkSize)>,
}

impl<V, Storage: core::ops::Deref<Target=Vec<V>>, ChonkSize: RectVolSize> core::ops::Deref for SubChunkSize<V, Storage, ChonkSize> {
    type Target = Vec<V>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.storage
    }
}

impl<V, Storage: core::ops::DerefMut<Target=Vec<V>>, ChonkSize: RectVolSize> core::ops::DerefMut for SubChunkSize<V, Storage, ChonkSize> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.storage
    }
}

impl<V, Storage: From<Vec<V>>, ChonkSize: RectVolSize> From<Vec<V>> for SubChunkSize<V, Storage, ChonkSize> {
    #[inline]
    fn from(storage: Vec<V>) -> Self {
        Self {
            storage: storage.into(),
            phantom: PhantomData,
        }
    }
}

// TODO (haslersn): Assert ChonkSize::RECT_SIZE.x == ChonkSize::RECT_SIZE.y
impl<V, Storage, ChonkSize: RectVolSize> VolSize<V> for SubChunkSize<V, Storage, ChonkSize>
    /* where Storage: Clone + core::ops::Deref<Target=Vec<V>> + core::ops::DerefMut + From<Vec<V>>,
     * */
{
    const SIZE: Vec3<u32> = Vec3 {
        x: ChonkSize::RECT_SIZE.x,
        y: ChonkSize::RECT_SIZE.x,
        // NOTE: Currently, use 32 instead of 2 for RECT_SIZE.x = 128.
        z: ChonkSize::RECT_SIZE.x / 2,
    };
}

pub type SubChunk<V, Storage, S, M> = Chunk<V, SubChunkSize<V, Storage, S>, PhantomData<M>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chonk<V, Storage, S: RectVolSize, M: Clone> {
    z_offset: i32,
    sub_chunks: Vec<SubChunk<V, Storage, S, M>>,
    below: V,
    above: V,
    meta: M,
    phantom: PhantomData<S>,
}

impl<V, Storage: core::ops::DerefMut<Target=Vec<V>>, S: RectVolSize, M: Clone> Chonk<V, Storage, S, M> {
    pub fn new(z_offset: i32, below: V, above: V, meta: M) -> Self {
        Self {
            z_offset,
            sub_chunks: Vec::new(),
            below,
            above,
            meta,
            phantom: PhantomData,
        }
    }

    pub fn below(&self) -> &V { &self.below }

    pub fn above(&self) -> &V { &self.above }

    pub fn meta(&self) -> &M { &self.meta }

    #[inline]
    pub fn get_min_z(&self) -> i32 { self.z_offset }

    #[inline]
    pub fn get_max_z(&self) -> i32 {
        self.z_offset + (self.sub_chunks.len() as u32 * SubChunkSize::<V, Storage, S>::SIZE.z) as i32
    }

    /// Flattened version of this chonk.
    ///
    /// It's not acutally flat, it just skips the indirection through the index.  The idea is to
    /// use a constant stride for row access so the prefetcher can process it more easily.
    pub fn make_flat<'a>(&'a self, arena: &'a bumpalo::Bump) -> Vec<&'a [V]>
        where
            V: Copy + Hash + Eq,
            [(); SubChunk::<V, Storage, S, M>::GROUP_VOLUME as usize]:,
    {
        // Cache of slices per block type to maximize cacheline reuse.
        let mut default_slices = HashMap::new();
        let mut flat = Vec::with_capacity(self.sub_chunks.len() * /*SubChunkSize::<V, Storage, S>::SIZE.z as usize **/
                                     /* SubChunk::<V, Storage, S, M>::VOLUME as usize */
                                          SubChunk::<V, Storage, S, M>::GROUP_COUNT_TOTAL as usize);
        self.sub_chunks.iter().enumerate().for_each(|(idx, sub_chunk)| {
            let default = *sub_chunk.default();
            let slice = *default_slices.entry(default)
                .or_insert_with(move || {
                    &*arena.alloc_slice_fill_copy(SubChunk::<V, Storage, S, M>::GROUP_VOLUME as usize, default)
                });
            sub_chunk.push_flat(&mut flat, slice);
        });
        flat
    }

    #[inline]
    /// Approximate max z.
    ///
    /// NOTE: Column must be in range; results are undefined otherwise.
    // #[allow(unsafe_code)]
    pub fn get_max_z_col(&self, col: Vec2<i32>) -> i32
        where V: Eq,
    {
        self.get_max_z()
        /* let group_size = SubChunk::<V, Storage, S, M>::GROUP_SIZE;
        let group_count = SubChunk::<V, Storage, S, M>::GROUP_COUNT;
        let col = (col.as_::<u32>() % Self::RECT_SIZE);
        // FIXME: Make abstract.
        let grp_pos = col.map2(group_size.xy(), |e, s| e / s);
        let grp_idx_2d = grp_pos.x * (group_count.y * group_count.z)
            + (grp_pos.y * group_count.z);
        /* dbg!(col, group_size, group_count, grp_pos, grp_idx_2d); */

        /* let grp_idx: [u8; SubChunk<V, Storage, S, M>::GROUP_SIZE.z] =
            [grp_idx_2d, grp_idx_2d + 1, grp_idx_2d + 2, grp_idx_2d + 3]; */
        // let grp_idx = Chunk::grp_idx(col.with_z(0));
        let grp_idx_2d = grp_idx_2d as u8;
        let grp_idx0 = grp_idx_2d as usize;
        let grp_idx1 = (grp_idx_2d + 1) as usize;
        let grp_idx2 = (grp_idx_2d + 2) as usize;
        let grp_idx3 = (grp_idx_2d + 3) as usize;
        // Find first subchunk with either a different default from our above, or whose group at
        // the relevant index is not the default.
        let group_offset_z = self.sub_chunks.iter().enumerate().rev().find_map(|(sub_chunk_idx, sub_chunk)| {
            if sub_chunk.default() != &self.above {
                return Some((sub_chunk_idx + 1) * 4);
            }
            let num_groups = sub_chunk.num_groups() as u8;
            let indices = /*&*/sub_chunk.indices()/*[0..256]*/;
            unsafe {
                let idx0 = *indices.get_unchecked(grp_idx0);
                let idx1 = *indices.get_unchecked(grp_idx2);
                let idx2 = *indices.get_unchecked(grp_idx1);
                let idx3 = *indices.get_unchecked(grp_idx3);
                if idx3 >= num_groups {
                    return Some(sub_chunk_idx * 4 + grp_idx3);
                }
                if idx2 >= num_groups {
                    return Some(sub_chunk_idx * 4 + grp_idx2);
                }
                if idx1 >= num_groups {
                    return Some(sub_chunk_idx * 4 + grp_idx1);
                }
                if idx0 >= num_groups {
                    return Some(sub_chunk_idx * 4 + grp_idx0);
                }
            }
            return None;
        }).unwrap_or(0);
        let offset: u32 = group_offset_z as u32 * SubChunk::<V, Storage, S, M>::GROUP_SIZE.z;
        self.get_min_z() + offset as i32 */
    }

    pub fn sub_chunks_len(&self) -> usize { self.sub_chunks.len() }

    pub fn sub_chunk_groups(&self) -> usize {
        self.sub_chunks.iter().map(SubChunk::num_groups).sum()
    }

    pub fn sub_chunks<'a>(&'a self) -> impl Iterator<Item = &'a SubChunk<V, Storage, S, M>> {
        self.sub_chunks.iter()
    }

    /// Iterate through the voxels in this chunk, attempting to avoid those that
    /// are unchanged (i.e: match the `below` and `above` voxels). This is
    /// generally useful for performance reasons.
    pub fn iter_changed(&self) -> impl Iterator<Item = (Vec3<i32>, &V)> + '_ {
        self.sub_chunks
            .iter()
            .enumerate()
            .filter(|(_, sc)| sc.num_groups() > 0)
            .flat_map(move |(i, sc)| {
                let z_offset = self.z_offset + i as i32 * SubChunkSize::<V, Storage, S>::SIZE.z as i32;
                sc.vol_iter(Vec3::zero(), SubChunkSize::<V, Storage, S>::SIZE.map(|e| e as i32))
                    .map(move |(pos, vox)| (pos + Vec3::unit_z() * z_offset, vox))
            })
    }

    // Returns the index (in self.sub_chunks) of the SubChunk that contains
    // layer z; note that this index changes when more SubChunks are prepended
    #[inline]
    fn sub_chunk_idx(&self, z: i32) -> i32 {
        let diff = z - self.z_offset;
        diff >> (SubChunkSize::<V, Storage, S>::SIZE.z - 1).count_ones()
    }

    // Converts a z coordinate into a local z coordinate within a sub chunk
    fn sub_chunk_z(&self, z: i32) -> i32 {
        let diff = z - self.z_offset;
        diff & (SubChunkSize::<V, Storage, S>::SIZE.z - 1) as i32
    }

    // Returns the z offset of the sub_chunk that contains layer z
    fn sub_chunk_min_z(&self, z: i32) -> i32 { z - self.sub_chunk_z(z) }

    /// Compress chunk by using more intelligent defaults.
    pub fn defragment(&mut self)
    where
        Storage: From<Vec<V>>,
        V: zerocopy::AsBytes + Clone + Eq + Hash,
        [(); { core::mem::size_of::<V>() }]:,
    {
        // First, defragment all subchunks.
        self.sub_chunks.iter_mut().for_each(SubChunk::defragment);
        // For each homogeneous subchunk (i.e. those where all blocks are the same),
        // find those which match `below` at the bottom of the chunk, or `above`
        // at the top, since these subchunks are redundant and can be removed.
        // Note that we find (and drain) the above chunks first, so that when we
        // remove the below chunks we have fewer remaining chunks to backshift.
        // Note that we use `take_while` instead of `rposition` here because `rposition`
        // goes one past the end, which we only want in the forward direction.
        let above_count = self
            .sub_chunks
            .iter()
            .rev()
            .take_while(|subchunk| subchunk.homogeneous() == Some(&self.above))
            .count();
        // Unfortunately, `TakeWhile` doesn't implement `ExactSizeIterator` or
        // `DoubleEndedIterator`, so we have to recreate the same state by calling
        // `nth_back` (note that passing 0 to nth_back goes back 1 time, not 0
        // times!).
        let mut subchunks = self.sub_chunks.iter();
        if above_count > 0 {
            subchunks.nth_back(above_count - 1);
        }
        // `above_index` is now the number of remaining elements, since all the elements
        // we drained were at the end.
        let above_index = subchunks.len();
        // `below_len` now needs to be applied to the state after the `above` chunks are
        // drained, to make sure we don't accidentally have overlap (this is
        // possible if self.above == self.below).
        let below_len = subchunks.position(|subchunk| subchunk.homogeneous() != Some(&self.below));
        let below_len = below_len
            // NOTE: If `below_index` is `None`, then every *remaining* chunk after we drained
            // `above` was full and matched `below`.
            .unwrap_or(above_index);
        // Now, actually remove the redundant chunks.
        self.sub_chunks.truncate(above_index);
        self.sub_chunks.drain(..below_len);
        // Finally, bump the z_offset to account for the removed subchunks at the
        // bottom. TODO: Add invariants to justify why `below_len` must fit in
        // i32.
        self.z_offset += below_len as i32 * SubChunkSize::<V, Storage, S>::SIZE.z as i32;
    }
}

impl<V, Storage, S: RectVolSize, M: Clone> BaseVol for Chonk<V, Storage, S, M> {
    type Error = ChonkError;
    type Vox = V;
}

impl<V, Storage, S: RectVolSize, M: Clone> RectRasterableVol for Chonk<V, Storage, S, M> {
    const RECT_SIZE: Vec2<u32> = S::RECT_SIZE;
}

impl<V, Storage: core::ops::DerefMut<Target=Vec<V>>, S: RectVolSize, M: Clone> ReadVol for Chonk<V, Storage, S, M> {
    #[inline(always)]
    fn get(&self, pos: Vec3<i32>) -> Result<&V, Self::Error> {
        if pos.z < self.get_min_z() {
            // Below the terrain
            Ok(&self.below)
        } else if pos.z >= self.get_max_z() {
            // Above the terrain
            Ok(&self.above)
        } else {
            // Within the terrain
            let sub_chunk_idx = self.sub_chunk_idx(pos.z);
            let rpos = pos
                - Vec3::unit_z()
                    * (self.z_offset + sub_chunk_idx * SubChunkSize::<V, Storage, S>::SIZE.z as i32);
            self.sub_chunks[sub_chunk_idx as usize]
                .get(rpos)
                .map_err(Self::Error::SubChunkError)
        }
    }
}

impl<V: Clone + PartialEq, Storage: Clone + core::ops::DerefMut<Target=Vec<V>> + From<Vec<V>>, S: Clone + RectVolSize, M: Clone> WriteVol for Chonk<V, Storage, S, M> {
    #[inline(always)]
    fn set(&mut self, pos: Vec3<i32>, block: Self::Vox) -> Result<V, Self::Error> {
        let mut sub_chunk_idx = self.sub_chunk_idx(pos.z);

        if pos.z < self.get_min_z() {
            // Make sure we're not adding a redundant chunk.
            if block == self.below {
                return Ok(self.below.clone());
            }
            // Prepend exactly sufficiently many SubChunks via Vec::splice
            let c = SubChunk::<V, Storage, S, M>::filled(self.below.clone(), /*self.meta.clone()*/PhantomData);
            let n = (-sub_chunk_idx) as usize;
            self.sub_chunks.splice(0..0, std::iter::repeat(c).take(n));
            self.z_offset += sub_chunk_idx * SubChunkSize::<V, Storage, S>::SIZE.z as i32;
            sub_chunk_idx = 0;
        } else if pos.z >= self.get_max_z() {
            // Make sure we're not adding a redundant chunk.
            if block == self.above {
                return Ok(self.above.clone());
            }
            // Append exactly sufficiently many SubChunks via Vec::extend
            let c = SubChunk::<V, Storage, S, M>::filled(self.above.clone(), /*self.meta.clone()*/PhantomData);
            let n = 1 + sub_chunk_idx as usize - self.sub_chunks.len();
            self.sub_chunks.extend(std::iter::repeat(c).take(n));
        }

        let rpos = pos
            - Vec3::unit_z() * (self.z_offset + sub_chunk_idx * SubChunkSize::<V, Storage, S>::SIZE.z as i32);
        self.sub_chunks[sub_chunk_idx as usize] // TODO (haslersn): self.sub_chunks.get(...).and_then(...)
            .set(rpos, block)
            .map_err(Self::Error::SubChunkError)
    }
}

struct ChonkIterHelper<V, Storage, S: RectVolSize, M: Clone> {
    sub_chunk_min_z: i32,
    lower_bound: Vec3<i32>,
    upper_bound: Vec3<i32>,
    phantom: PhantomData<Chonk<V, Storage, S, M>>,
}

impl<V, Storage, S: RectVolSize, M: Clone> Iterator for ChonkIterHelper<V, Storage, S, M> {
    type Item = (i32, Vec3<i32>, Vec3<i32>);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.lower_bound.z >= self.upper_bound.z {
            return None;
        }
        let mut lb = self.lower_bound;
        let mut ub = self.upper_bound;
        let current_min_z = self.sub_chunk_min_z;
        lb.z -= current_min_z;
        ub.z -= current_min_z;
        ub.z = std::cmp::min(ub.z, SubChunkSize::<V, Storage, S>::SIZE.z as i32);
        self.sub_chunk_min_z += SubChunkSize::<V, Storage, S>::SIZE.z as i32;
        self.lower_bound.z = self.sub_chunk_min_z;
        Some((current_min_z, lb, ub))
    }
}

pub struct ChonkPosIter<V, Storage, S: RectVolSize, M: Clone> {
    outer: ChonkIterHelper<V, Storage, S, M>,
    opt_inner: Option<(i32, ChunkPosIter<V, SubChunkSize<V, Storage, S>, PhantomData<M>>)>,
}

impl<V, Storage, S: RectVolSize, M: Clone> Iterator for ChonkPosIter<V, Storage, S, M> {
    type Item = Vec3<i32>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((sub_chunk_min_z, ref mut inner)) = self.opt_inner {
                if let Some(mut pos) = inner.next() {
                    pos.z += sub_chunk_min_z;
                    return Some(pos);
                }
            }
            match self.outer.next() {
                None => return None,
                Some((sub_chunk_min_z, lb, ub)) => {
                    self.opt_inner = Some((sub_chunk_min_z, SubChunk::<V, Storage, S, M>::pos_iter(lb, ub)))
                },
            }
        }
    }
}

enum InnerChonkVolIter<'a, V, Storage, S: RectVolSize, M: Clone> {
    Vol(ChunkVolIter<'a, V, SubChunkSize<V, Storage, S>, PhantomData<M>>),
    Pos(ChunkPosIter<V, SubChunkSize<V, Storage, S>, PhantomData<M>>),
}

pub struct ChonkVolIter<'a, V, Storage, S: RectVolSize, M: Clone> {
    chonk: &'a Chonk<V, Storage, S, M>,
    outer: ChonkIterHelper<V, Storage, S, M>,
    opt_inner: Option<(i32, InnerChonkVolIter<'a, V, Storage, S, M>)>,
}

impl<'a, V, Storage: core::ops::DerefMut<Target=Vec<V>>, S: RectVolSize, M: Clone> Iterator for ChonkVolIter<'a, V, Storage, S, M> {
    type Item = (Vec3<i32>, &'a V);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((sub_chunk_min_z, ref mut inner)) = self.opt_inner {
                let got = match inner {
                    InnerChonkVolIter::<'a, V, Storage, S, M>::Vol(iter) => iter.next(),
                    InnerChonkVolIter::<'a, V, Storage, S, M>::Pos(iter) => iter.next().map(|pos| {
                        if sub_chunk_min_z < self.chonk.get_min_z() {
                            (pos, &self.chonk.below)
                        } else {
                            (pos, &self.chonk.above)
                        }
                    }),
                };
                if let Some((mut pos, vox)) = got {
                    pos.z += sub_chunk_min_z;
                    return Some((pos, vox));
                }
            }
            match self.outer.next() {
                None => return None,
                Some((sub_chunk_min_z, lb, ub)) => {
                    let inner = if sub_chunk_min_z < self.chonk.get_min_z()
                        || sub_chunk_min_z >= self.chonk.get_max_z()
                    {
                        InnerChonkVolIter::<'a, V, Storage, S, M>::Pos(SubChunk::<V, Storage, S, M>::pos_iter(lb, ub))
                    } else {
                        InnerChonkVolIter::<'a, V, Storage, S, M>::Vol(
                            self.chonk.sub_chunks
                                [self.chonk.sub_chunk_idx(sub_chunk_min_z) as usize]
                                .vol_iter(lb, ub),
                        )
                    };
                    self.opt_inner = Some((sub_chunk_min_z, inner));
                },
            }
        }
    }
}

impl<'a, V, Storage: core::ops::DerefMut<Target=Vec<V>>, S: RectVolSize, M: Clone> IntoPosIterator for &'a Chonk<V, Storage, S, M> {
    type IntoIter = ChonkPosIter<V, Storage, S, M>;

    fn pos_iter(self, lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self::IntoIter {
        Self::IntoIter {
            outer: ChonkIterHelper::<V, Storage, S, M> {
                sub_chunk_min_z: self.sub_chunk_min_z(lower_bound.z),
                lower_bound,
                upper_bound,
                phantom: PhantomData,
            },
            opt_inner: None,
        }
    }
}

impl<'a, V, Storage: core::ops::DerefMut<Target=Vec<V>>, S: RectVolSize, M: Clone> IntoVolIterator<'a> for &'a Chonk<V, Storage, S, M> {
    type IntoIter = ChonkVolIter<'a, V, Storage, S, M>;

    fn vol_iter(self, lower_bound: Vec3<i32>, upper_bound: Vec3<i32>) -> Self::IntoIter {
        Self::IntoIter {
            chonk: self,
            outer: ChonkIterHelper::<V, Storage, S, M> {
                sub_chunk_min_z: self.sub_chunk_min_z(lower_bound.z),
                lower_bound,
                upper_bound,
                phantom: PhantomData,
            },
            opt_inner: None,
        }
    }
}
