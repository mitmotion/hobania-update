use super::*;
use crate::{
    block::block_from_structure,
    site2::util::Dir,
    util::{RandomField, Sampler},
    CanvasInfo,
    ColumnSample,
};
use common::{
    generation::EntityInfo,
    store::{Id, Store},
    terrain::{
        structure::{Structure as PrefabStructure, StructureBlock},
        Block, BlockKind,
    },
    vol::ReadVol,
};
use fxhash::FxHasher64;
use hashbrown::{/*hash_map::Entry, */HashMap};
use noisy_float::types::n32;
use num::cast::AsPrimitive;
use std::{
    cell::RefCell,
    hash::BuildHasherDefault,
    simd::StdFloat,
};
use vek::*;

const PRINT_MESSAGES: bool = false;

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Node<'a> {
    Empty, // Placeholder

    // Shapes
    Aabb(Aabb<i32>),
    Pyramid {
        aabb: Aabb<i32>,
        inset: i32,
    },
    Ramp {
        aabb: Aabb<i32>,
        inset: i32,
        dir: Dir,
    },
    Gable {
        aabb: Aabb<i32>,
        inset: i32,
        // X axis parallel or Y axis parallel
        dir: Dir,
    },
    Cylinder(Aabb<i32>),
    Cone(Aabb<i32>),
    Sphere(Aabb<i32>),
    /// An Aabb with rounded corners. The degree relates to how rounded the
    /// corners are. A value less than 1.0 results in concave faces. A value
    /// of 2.0 results in an ellipsoid. Values greater than 2.0 result in a
    /// rounded aabb. Values less than 0.0 are clamped to 0.0 as negative values
    /// would theoretically yield shapes extending to infinity.
    Superquadric {
        aabb: Aabb<i32>,
        degree: f32,
    },
    Plane(Aabr<i32>, Vec3<i32>, Vec2<f32>),
    /// A line segment from start to finish point with a given radius for both
    /// points, together with an optional z_scale controlling scaling of the structure along the z
    /// axis.
    Segment {
        segment: LineSegment3<f32>,
        /* radius: f32, */
        r0: f32,
        r1: f32,
        z_scale: f32,
    },
    /// A prism created by projecting a line segment with a given radius along
    /// the z axis up to a provided height
    SegmentPrism {
        segment: LineSegment3<f32>,
        radius: f32,
        height: f32,
    },
    /* /// A sampling function is always a subset of another primitive to avoid
    /// needing infinite bounds
    Sampling(Id<Primitive<'a>>, &'a dyn Fn(Vec3<i32>)-> bool), */
    Prefab(&'static PrefabStructure),

    // Combinators
    Intersect([Id<Primitive<'a>>; 2]),
    IntersectAll(&'a [Id<Primitive<'a>>]),
    Union(Id<Primitive<'a>>, Id<Primitive<'a>>),
    UnionAll(&'a [Id<Primitive<'a>>]),
    // Not commutative; third argument is true when we want to evaluate the first argument before
    // the second.
    Without(Id<Primitive<'a>>, Id<Primitive<'a>>, bool),
    // Operators
    Translate(Id<Primitive<'a>>, Vec3<i32>),
    /* Scale(Id<Primitive<'a>>, Vec3<f32>), */
    RotateAbout(Id<Primitive<'a>>, Mat3<i32>, Vec3<f32>),
    /* /// Repeat a primitive a number of times in a given direction, overlapping
    /// between repeats are unspecified.
    Repeat(Id<Primitive<'a>>, Vec3<i32>, /*u32*/u16), */
}

#[derive(Debug)]
pub struct Primitive<'a>(Node<'a>);

pub type PrimMap<V> = HashMap<u64, V, BuildHasherDefault<FxHasher64>>;

pub type BoundsMap = PrimMap<Vec<Aabb<i32>>>;

impl<'a> std::fmt::Debug for Node<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Node::Empty => f.debug_tuple("Empty").finish(),
            Node::Aabb(aabb) => f.debug_tuple("Aabb").field(&aabb).finish(),
            Node::Pyramid { aabb, inset } => {
                f.debug_tuple("Pyramid").field(&aabb).field(&inset).finish()
            },
            Node::Ramp { aabb, inset, dir } => f
                .debug_tuple("Ramp")
                .field(&aabb)
                .field(&inset)
                .field(&dir)
                .finish(),
            Node::Gable { aabb, inset, dir } => f
                .debug_tuple("Gable")
                .field(&aabb)
                .field(&inset)
                .field(&dir)
                .finish(),
            Node::Cylinder(aabb) => f.debug_tuple("Cylinder").field(&aabb).finish(),
            Node::Cone(aabb) => f.debug_tuple("Cone").field(&aabb).finish(),
            Node::Sphere(aabb) => f.debug_tuple("Sphere").field(&aabb).finish(),
            Node::Superquadric { aabb, degree } => f
                .debug_tuple("Superquadric")
                .field(&aabb)
                .field(&degree)
                .finish(),
            Node::Plane(aabr, origin, gradient) => f
                .debug_tuple("Plane")
                .field(&aabr)
                .field(&origin)
                .field(&gradient)
                .finish(),
            Node::Segment { segment, r0, r1, z_scale } => f
                .debug_tuple("Segment")
                .field(&segment)
                .field(&r0)
                .field(&r1)
                .field(&z_scale)
                .finish(),
            Node::SegmentPrism {
                segment,
                radius,
                height,
            } => f
                .debug_tuple("SegmentPrism")
                .field(&segment)
                .field(&radius)
                .field(&height)
                .finish(),
            /* Node::Sampling(prim, _) => f.debug_tuple("Sampling").field(&prim).finish(), */
            Node::Prefab(prefab) => f.debug_tuple("Prefab").field(&prefab.get_bounds()).finish(),
            Node::Intersect([a, b]) => f.debug_tuple("Intersect").field(&a).field(&b).finish(),
            Node::IntersectAll(prims) => f.debug_tuple("IntersectAll").field(prims).finish(),
            Node::Union(a, b) => f.debug_tuple("Union").field(&a).field(&b).finish(),
            Node::UnionAll(prims) => f.debug_tuple("UnionAll").field(prims).finish(),
            Node::Without(a, b, _) => f.debug_tuple("Without").field(&a).field(&b).finish(),
            Node::Translate(a, vec) => {
                f.debug_tuple("Translate").field(&a).field(&vec).finish()
            },
            /* Node::Scale(a, vec) => f.debug_tuple("Scale").field(&a).field(&vec).finish(),
             * */
            Node::RotateAbout(a, mat, vec) => f
                .debug_tuple("RotateAbout")
                .field(&a)
                .field(&mat)
                .field(&vec)
                .finish(),
            /* Node::Repeat(a, offset, n) => f
                .debug_tuple("Repeat")
                .field(&a)
                .field(&offset)
                .field(&n)
                .finish(), */
        }
    }
}

/*
impl PartialEq<Primitive> for Node {
    fn eq(&self, other: &Primitive) -> bool {
        match (self, other) {
            (Node::Empty, Node::Empty) => true,
            (Node::Aabb(aabb_l), Node::Aabb(aabb_r)) => aabb_l == aabb_r,
            (
                Node::Pyramid {
                    aabb: aabb_l,
                    inset: inset_l,
                },
                Node::Pyramid {
                    aabb: aabb_r,
                    inset: inset_r,
                },
            ) => aabb_l == aabb_r && inset_l == inset_r,
            (
                Node::Ramp {
                    aabb: aabb_l,
                    inset: inset_l,
                    dir: dir_l,
                },
                Node::Ramp {
                    aabb: aabb_r,
                    inset: inset_r,
                    dir: dir_r,
                },
            ) => aabb_l == aabb_r && inset_l == inset_r && dir_l == dir_r,
            (
                Node::Gable {
                    aabb: aabb_l,
                    inset: inset_l,
                    dir: dir_l,
                },
                Node::Gable {
                    aabb: aabb_r,
                    inset: inset_r,
                    dir: dir_r,
                },
            ) => aabb_l == aabb_r && inset_l == inset_r && dir_l == dir_r,
            (Node::Cylinder(aabb_l), Node::Cylinder(aabb_r)) => aabb_l == aabb_r,
            (Node::Cone(aabb_l), Node::Cone(aabb_r)) => aabb_l == aabb_r,
            (Node::Sphere(aabb_l), Node::Sphere(aabb_r)) => aabb_l == aabb_r,
            (
                Node::Superquadric {
                    aabb: aabb_l,
                    degree: degree_l,
                },
                Node::Superquadric {
                    aabb: aabb_r,
                    degree: degree_r,
                },
            ) => aabb_l == aabb_r && degree_l == degree_r,
            (
                Node::Plane(aabr_l, origin_l, gradient_l),
                Node::Plane(aabr_r, origin_r, gradient_r),
            ) => aabr_l == aabr_r && origin_l == origin_r && gradient_l == gradient_r,
            (
                Node::Segment {
                    segment: segment_l,
                    radius: radius_l,
                },
                Node::Segment {
                    segment: segment_r,
                    radius: radius_r,
                },
            ) => segment_l == segment_r && radius_l == radius_r,
            (
                Node::SegmentPrism {
                    segment: segment_l,
                    radius: radius_l,
                    height: height_l,
                },
                Node::SegmentPrism {
                    segment: segment_r,
                    radius: radius_r,
                    height: height_r,
                },
            ) => segment_l == segment_r && radius_l == radius_r && height_l == height_r,
            (Node::Sampling(prim_l, f_l), Node::Sampling(prim_r, f_r)) => {
                // Since the equality is only being used here to check if two primitives are
                // equal as an optimization (combining them if they are equal), false negatives
                // are acceptable here (they don't impact correctness, only performance).
                #[allow(clippy::vtable_address_comparisons)]
                {
                    prim_l == prim_r
                        && std::ptr::eq(f_l.as_ref() as *const _, f_r.as_ref() as *const _)
                }
            },
            (Node::Prefab(prefab_l), Node::Prefab(prefab_r)) => prefab_l == prefab_r,
            (Node::Intersect(a_l, b_l), Node::Intersect(a_r, b_r)) => {
                a_l == a_r && b_l == b_r
            },
            (Node::Union(a_l, b_l), Node::Union(a_r, b_r)) => a_l == a_r && b_l == b_r,
            (Node::Without(a_l, b_l), Node::Without(a_r, b_r)) => {
                a_l == a_r && b_l == b_r
            },
            (Node::Rotate(a_l, mat_l), Node::Rotate(a_r, mat_r)) => {
                a_l == a_r && mat_l == mat_r
            },
            (Node::Translate(a_l, vec_l), Node::Translate(a_r, vec_r)) => {
                a_l == a_r && vec_l == vec_r
            },
            (Node::Scale(a_l, vec_l), Node::Scale(a_r, vec_r)) => {
                a_l == a_r && vec_l == vec_r
            },
            (Node::Repeat(a_l, offset_l, n_l), Node::Repeat(a_r, offset_r, n_r)) => {
                a_l == a_r && offset_l == offset_r && n_l == n_r
            },
            _ => false,
        }
    }
}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash_vec2 =
            |v: &Vec2<f32>, state: &mut H| v.iter().for_each(|f| f.to_le_bytes().hash(state));
        let hash_vec =
            |v: &Vec3<f32>, state: &mut H| v.iter().for_each(|f| f.to_le_bytes().hash(state));
        match self {
            Node::Empty => {
                "Empty".hash(state);
            },
            Node::Aabb(aabb) => {
                "Aabb".hash(state);
                aabb.hash(state);
            },
            Node::Pyramid { aabb, inset } => {
                "Pyramid".hash(state);
                aabb.hash(state);
                inset.hash(state);
            },
            Node::Ramp { aabb, inset, dir } => {
                "Ramp".hash(state);
                aabb.hash(state);
                inset.hash(state);
                dir.hash(state);
            },
            Node::Gable { aabb, inset, dir } => {
                "Gable".hash(state);
                aabb.hash(state);
                inset.hash(state);
                dir.hash(state);
            },
            Node::Cylinder(aabb) => {
                "Cylinder".hash(state);
                aabb.hash(state);
            },
            Node::Cone(aabb) => {
                "Cone".hash(state);
                aabb.hash(state);
            },
            Node::Sphere(aabb) => {
                "Sphere".hash(state);
                aabb.hash(state);
            },
            Node::Superquadric { aabb, degree } => {
                "Superquadric".hash(state);
                aabb.hash(state);
                degree.to_le_bytes().hash(state);
            },
            Node::Plane(aabr, origin, gradient) => {
                "Plane".hash(state);
                aabr.hash(state);
                origin.hash(state);
                hash_vec2(gradient, state);
            },
            Node::Segment { segment, radius } => {
                "Segment".hash(state);
                hash_vec(&segment.start, state);
                hash_vec(&segment.end, state);
                radius.to_le_bytes().hash(state);
            },
            Node::SegmentPrism {
                segment,
                radius,
                height,
            } => {
                "SegmentPrism".hash(state);
                hash_vec(&segment.start, state);
                hash_vec(&segment.end, state);
                radius.to_le_bytes().hash(state);
                height.to_le_bytes().hash(state);
            },
            Node::Sampling(prim, _) => {
                "Sampling".hash(state);
                prim.hash(state);
            },
            Node::Prefab(prefab) => {
                "Prefab".hash(state);
                prefab.hash(state);
            },
            Node::Intersect(a, b) => {
                "Intersect".hash(state);
                a.hash(state);
                b.hash(state);
            },
            Node::Union(a, b) => {
                "Union".hash(state);
                a.hash(state);
                b.hash(state);
            },
            Node::Without(a, b) => {
                "Without".hash(state);
                a.hash(state);
                b.hash(state);
            },
            Node::Rotate(a, mat) => {
                "Rotate".hash(state);
                a.hash(state);
                mat.hash(state);
            },
            Node::Translate(a, vec) => {
                "Translate".hash(state);
                a.hash(state);
                vec.hash(state);
            },
            Node::Scale(a, vec) => {
                "Scale".hash(state);
                a.hash(state);
                hash_vec(vec, state);
            },
            Node::Repeat(a, offset, n) => {
                "Repeat".hash(state);
                a.hash(state);
                offset.hash(state);
                n.hash(state);
            },
        }
    }
} */

impl<'a> Node<'a> {
    /* fn intersect(a: impl Into<Id<Primitive<'a>>>, b: impl Into<Id<Primitive<'a>>>) -> Self {
        Self::Intersect(a.into(), b.into())
    }

    fn union(a: impl Into<Id<Primitive<'a>>>, b: impl Into<Id<Primitive<'a>>>) -> Self {
        Self::Union(a.into(), b.into())
    } */

    /// NOTE: Negation of cbpv function type:
    ///
    /// -(A ⊸ B) ≡ -(¬A ⅋ B) ≡ A ⊗ -B
    ///
    /// where A is positive and B is negative.
    fn without<'painter, Arg: Positive, Ret: Negative>(painter: &'painter Painter<'a>, a_: /*impl Into<Id<Primitive<'a>>>*/PrimitiveRef<'painter, 'a, Arg>, b_: /*impl Into<Id<Primitive<'a>>>*/PrimitiveRef<'painter, 'a, Ret>) -> PrimitiveRef<'painter, 'a, Arg> {
        let a = a_.id;
        let b = b_.id;
        let (a_bounds, b_bounds) = {
            let prims = painter.prims.borrow();
            let mut cache = painter.bounds_cache.borrow_mut();
            (Painter::get_bounds(&mut cache, &prims, a),
             Painter::get_bounds(&mut cache, &prims, b),)
        };
        let a_volume = a_bounds.size().as_::<f32>().product();
        if a_volume < 1.0 {
            return /*Self::Empty*/painter.empty().as_kind();
        }
        let ab_volume = a_bounds.intersection(b_bounds).size().as_::<f32>().product();
        if ab_volume < 1.0 {
            return /*Self::Union(a, painter.empty())*/a_;
        }
        let b_scale = ab_volume / a_volume;
        // cost(B | bound(A)) = cost(B) * vol(bound(A) ∩⁺ bound(B)) / vol(bound(A))
        // Since if B evaluates to true, we won't draw anything, and if A evaluates to false, we won't
        // draw anything, we test B first only if 1 - cost(B | bound(A)) < cost(A).  Otherwise, we
        // test A first.
        /* let b_volume = b_bounds.size().as_::<f32>().product();
        // (We multiply the estimates by volume to get a better sense of how many elements we can
        // discard early,as well). */
        // We do *NOT* multiply by volume estimates on both sides, because we always run through
        // the voxels in a_bound anyway.
        let cost_a = painter.depth(a)/* * a_volume*/;
        let cost_b = /*(*/1.0 - painter.depth(b) * b_scale/*) * b_volume*/;
        painter.prim(Self::Without(a, b, cost_a <= cost_b))
    }

    fn translate(a: impl Into<Id<Primitive<'a>>>, trans: Vec3<i32>) -> Self {
        Self::Translate(a.into(), trans)
    }

    /* pub fn scale(a: impl Into<Id<Primitive<'a>>>, scale: Vec3<f32>) -> Self {
        Self::Scale(a.into(), scale)
    } */

    fn rotate_about(
        a: impl Into<Id<Primitive<'a>>>,
        rot: Mat3<i32>,
        point: Vec3<impl AsPrimitive<f32>>,
    ) -> Self {
        Self::RotateAbout(a.into(), rot, point.as_())
    }

    /* pub fn repeat(a: impl Into<Id<Primitive<'a>>>, offset: Vec3<i32>, count: /*u32*/u16) -> Self {
        if count == 0 {
            return Self::Empty
        } else {
            // TODO: Check bounds for a.
            Self::Repeat(a.into(), offset, count)
        }
    } */
}

/// For the moment, there's no real advantage to making Fill be any different from a function,
/// so we just use that for now.  In the long term that may change though, so we use a type
/// alias and ask people to go through the functions below rather than constructing fills
/// explicitly.
pub trait Fill {
    const NEEDS_BLOCK: bool;
    const NEEDS_POS: bool;

    fn sample_at(&self, pos: Vec3<i32>, old_block: Block) -> Option<Block>;
}

impl<'a, F: Fill> Fill for &'a F {
    const NEEDS_BLOCK: bool = F::NEEDS_BLOCK;
    const NEEDS_POS: bool = F::NEEDS_POS;

    #[inline]
    fn sample_at(&self, pos: Vec3<i32>, old_block: Block) -> Option<Block> {
        (*self).sample_at(pos, old_block)
    }
}

#[derive(Clone,Copy)]
pub struct FillConst(Block);

impl Fill for FillConst {
    const NEEDS_BLOCK: bool = false;
    const NEEDS_POS: bool = false;

    #[inline]
    fn sample_at(&self, _: Vec3<i32>, _: Block) -> Option<Block> {
        Some(self.0)
    }
}

#[derive(Clone,Copy)]
pub struct FillMerge<F>(F);

impl<F: Fn(Block) -> Block> Fill for FillMerge<F> {
    const NEEDS_BLOCK: bool = true;
    const NEEDS_POS: bool = false;

    #[inline]
    fn sample_at(&self, _: Vec3<i32>, old_block: Block) -> Option<Block> {
        Some((&self.0)(old_block))
    }
}

#[derive(Clone,Copy)]
pub struct FillVar<F>(F);

impl<F: Fn(Vec3<i32>) -> Option<Block>> Fill for FillVar<F> {
    const NEEDS_BLOCK: bool = false;
    const NEEDS_POS: bool = true;

    #[inline]
    fn sample_at(&self, pos: Vec3<i32>, _: Block) -> Option<Block> {
        (&self.0)(pos)
    }
}

/// Data used by the rasterizer, responsible for executing fills (which consume primitives).
pub struct FillFn<'a, 'b, F> {
    // marker: core::marker::PhantomData<&'b ()>,
    /// Concrete implementation of rasterization and entity spawning.
    filler: &'a mut F,
    /// NOTE: It's not completely clear whether render_area belongs here!  But it doesn't obviously
    /// belong anywhere else, so this is fine for now.
    render_area: Aabr<i32>,
    /// TODO: I think this only exists to aid in the prefab hack; verify whether we really need it.
    canvas_info: CanvasInfo<'b>,
}

impl<'a, 'b, F: Filler> FillFn<'a, 'b, F> {
    fn fill(&mut self, painter: &Painter<'a>, prim: Id<Primitive<'a>>, fill: impl Fill + Copy) {
        let arena = painter.arena;
        let prim_tree = painter.prims.borrow();
        let mut bounds_cache = /*&mut self.bounds_cache*/painter.bounds_cache.borrow_mut();
        Painter::get_bounds_disjoint(
            arena,
            &mut *bounds_cache,
            &*prim_tree,
            prim,
            self.render_area,
            move |pos| self.filler.map(pos, fill)
        );

        /* Fill::get_bounds_disjoint(
            arena,
            cache,
            tree,
            prim,
            mask,
            |pos| filler.map(pos, fill)
        ); */

        /* fill.sample_at(
            self.arena,
            &mut self.bounds_cache,
            prim_tree,
            prim,
            self.render_area,
            self.filler,
        ); */
    }

    /// Spawns an entity if it is in the render_aabr, otherwise does nothing.
    #[inline]
    pub fn spawn(&mut self, entity: EntityInfo) {
        if self.render_area.contains_point(entity.pos.xy().as_()) {
            self.filler.spawn(entity);
        }
    }

    #[inline]
    pub fn block(&self, block: Block) -> impl Fill + Copy {
        FillConst(block)
    }

    #[inline]
    pub fn sprite(&self, sprite: SpriteKind) -> impl Fill + Copy {
        FillMerge(move |old_block: Block| {
            if old_block.is_filled() {
                Block::air(sprite)
            } else {
                old_block.with_sprite(sprite)
            }
        })
    }

    #[inline]
    pub fn rotated_sprite(&self, sprite: SpriteKind, ori: u8) -> impl Fill + Copy {
        FillMerge(move |old_block: Block| {
            if old_block.is_filled() {
                Block::air(sprite)
                    .with_ori(ori)
                    .unwrap_or_else(|| Block::air(sprite))
            } else {
                old_block
                    .with_sprite(sprite)
                    .with_ori(ori)
                    .unwrap_or_else(|| old_block.with_sprite(sprite))
            }
        })
    }

    #[inline]
    pub fn brick(&self, bk: BlockKind, col: Rgb<u8>, range: u8) -> impl Fill + Copy {
        // FIXME: Statically cache somewhere, or maybe just remove in favor of
        // Sampling.
        FillVar(move |pos: Vec3<i32>| {
            Some(Block::new(
               bk,
               col + (RandomField::new(13)
                   .get((pos + Vec3::new(pos.z, pos.z, 0)) / Vec3::new(2, 2, 1))
                   % range as u32) as u8,
            ))
        })
    }

    #[inline]
    pub fn gradient<'c>(&self, gradient: &'c util::gradient::Gradient, bk: BlockKind) -> impl Fill + Copy + 'c {
        FillVar(move |pos: Vec3<i32>| {
            Some(Block::new(bk, gradient.sample(pos.as_())))
        })
    }

    pub fn block_from_structure(&self, sb: StructureBlock, structure_pos: Vec2<i32>, seed: u32, col_sample: &'b ColumnSample) -> impl Fill + Copy + 'b
    {
        /* let col_sample = /*if let Some(col_sample) = */self.canvas_info.col(self.canvas_info.wpos)/* {
            col_sample
        } else {
            // Don't draw--technically we should probably not assume this much about
            // the filler implementation though.
            //
            // FIXME: Fix this for alternate fillers if it turns out to matter.
            return
        }*/; */
        let index = self.canvas_info.index;
        let calendar = self.canvas_info.calendar();

        FillVar(move |pos| {
            block_from_structure(
                index,
                sb,
                pos,
                structure_pos,
                seed,
                col_sample,
                Block::air,
                calendar,
            )
        })
    }

    #[inline]
    // TODO: the offset tr for Prefab is a hack that breaks the compositionality of Translate,
    // we probably need an evaluator for the primitive tree that gets which point is queried at
    // leaf nodes given an input point to make Translate/Rotate work generally
    pub fn prefab(&self, p: &'static PrefabStructure, tr: Vec3<i32>, seed: u32) -> impl Fill + Copy + 'b {
        let col_sample = /*if let Some(col_sample) = */self.canvas_info.col(self.canvas_info.wpos)/* {
            col_sample
        } else {
            // Don't draw--technically we should probably not assume this much about
            // the filler implementation though.
            //
            // FIXME: Fix this for alternate fillers if it turns out to matter.
            return
        }*/;
        let index = self.canvas_info.index;
        let p_bounds = p.get_bounds().center().xy();
        let calendar = self.canvas_info.calendar();

        FillVar(move |pos| {
            p.get(pos - tr).ok().and_then(|&sb| {
                block_from_structure(
                    index,
                    sb,
                    pos - tr,
                    p_bounds,
                    seed,
                    col_sample?,
                    Block::air,
                    calendar,
                )
            })
        })
    }

    #[inline]
    pub fn sampling<S: Fn(Vec3<i32>) -> Option<Block>/* + ?Sized*/>(&self, f: S) -> FillVar<S> {
        FillVar(f)
    }

    /// The area that the canvas is currently rendering.
    #[inline]
    pub fn render_aabr(&self) -> Aabr<i32> { self.render_area }
}
/* pub enum Fill<'a> {
    Sprite(SpriteKind),
    RotatedSprite(SpriteKind, u8),
    Block(Block),
    Brick(BlockKind, Rgb<u8>, u8),
    Gradient(util::gradient::Gradient, BlockKind),
    // TODO: the offset field for Prefab is a hack that breaks the compositionality of Translate,
    // we probably need an evaluator for the primitive tree that gets which point is queried at
    // leaf nodes given an input point to make Translate/Rotate work generally
    Prefab(&'static PrefabStructure, Vec3<i32>, u32),
    Sampling(&'a dyn Fn(Vec3<i32>) -> Option<Block>),
} */

/// Hack to avoid adding a const generic parameter, since the compiler seems to give up when
/// recursion is involved and just add it as a regular parameter, which adds measurable overhead to
/// each call, plus possibly also a runtime branch even though it's theoretically not necesary.
trait CheckAabr {
    const CHECK_AABR: bool;
}

struct TopAabr;
impl CheckAabr for TopAabr {
    const CHECK_AABR: bool = false;
}

struct SubAabr;
impl CheckAabr for SubAabr {
    const CHECK_AABR: bool = true;
}

enum StackMember<'a> {
    /* /// Focused additive disjunction:
    ///
    /// c₁ : (Γ, x: A ⊢ Δ)
    /// c₂ : (Γ, y: B ⊢ Δ)
    /// -------------------------------------- (⊕L)
    /// Γ | μ̃[ι₁(x).c₁ | ι₂(y).c₂] : A ⊕ B ⊢ Δ
    ///
    /// 〈ι₁(υ₁) ┃ μ̃[ι₁(x).c₁ | ι₂(y).c₂]〉→β 〈υ₁‖μ̃ x. c₁〉
    /// 〈ι₂(υ₂) ┃ μ̃[ι₁(x).c₁ | ι₂(y).c₂]〉→β 〈υ₂‖μ̃ y. c₂〉
    PlusDisj {
        mat: Mat4<i32>,
        mask: Aabr<i32>,
        rhs: &'a [Id<Primitive<'a>>],
    }, */
    /// Focused multiplicative conjunction:
    ///
    /// c : (Γ, x: A, y: B ⊢ Δ)
    /// -------------------------- (⊗L)
    /// Γ | μ̃[(x,y).c] : A ⊗ B ⊢ Δ
    ///
    /// 〈(υ₁,υ₂) ┃ μ̃[(x,y).c]〉→β 〈υ₁ υ₂‖μ̃ x y. c〉
    MulConj {
        mat: Mat4<i32>,
        /* mask: Aabr<i32>, */
        rhs: &'a [Id<Primitive<'a>>],
    },
    /* /// Focused additive conjunction:
    ///
    /// c₁ : (Γ ⊢ α: A, Δ)
    /// c₂ : (Γ ⊢ β: B, Δ)
    /// -------------------------------------- (&R)
    /// Γ ⊢ μ(π₁[α].c₁ | π₂[β].c₂) : A & B | Δ
    ///
    /// 〈μ(π₁[α].c₁ | π₂[β].c₂) ┃ π₁[e₁]〉→β 〈μ α. c₁‖e₁〉
    /// 〈μ(π₁[α].c₁ | π₂[β].c₂) ┃ π₂[e₂]〉→β 〈μ β. c₂‖e₂〉
    PlusConj, */
    /// Focused multiplicative disjunction:
    ///
    /// c : (Γ ⊢ α: A, β: B, Δ)
    /// -------------------------- (⅋R)
    /// Γ ⊢ μ([α,β].c) : A ⅋ B | Δ
    ///
    /// 〈μ([x,y].c) ┃ [e₁,e₂]〉→β〈μ x y. c‖e₁ e₂〉
    MulDisj,
}

/// A trait implemented by something that allows fills, e.g. a canvas.
///
/// Currently only abstracted for testing purposes (plus lack of rank-2 type polymorphism
/// without dynamic dispatch).
pub trait Filler {
    /// Maps the old block at position pos to a new block by calling f on it.
    fn map<F: Fill>(&mut self, pos: Vec3<i32>, f: F);

    /// Spawns an entity.
    fn spawn(&mut self, entity: EntityInfo);
}

impl Painter<'_> {
    fn contains_at<'a, /*const _CHECK_AABR: bool*/Check: CheckAabr>(
        /*cache: &mut BoundsMap,*/
        tree: &Store<Primitive<'a>>,
        prim: &Primitive<'a>,
        pos: /*vek::vec::repr_simd::Vec4*/Vec3<i32>,
    ) -> bool {
        /* println!("Prim {:?}: {:?}", prim.id(), tree[prim]); */
        /* const CHECK_AABR: bool = Check::CHECK_AABR; */
        // Custom closure because vek's impl of `contains_point` is inclusive :(
        let aabb_contains = |aabb: Aabb<i32>, pos: /*vek::vec::repr_simd::Vec4*/Vec3<i32>| {
            /* const CHECK_ABBR: bool = Check::CHECK_AABR; */
            !Check::CHECK_AABR ||
            /* let res = */{
                /* #[cfg(feature = "simd")]
                {
                    // TODO: Enforce aabr.max.x > aabr.min.x and aabr.max.y > aabr.min.y at
                    // compile time.
                    /* let min = vek::vec::repr_simd::Vec8::new(
                        aabb.min.x, aabb.min.y, aabb.min.z, 0,
                        pos.x, pos.y, pos.z, 0,
                    );
                    let max = vek::vec::repr_simd::Vec8::new(
                        pos.x, pos.y, pos.z, 0,
                        aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1, 0,
                    );
                    let cmp = min.cmple_simd(max); */
                    let min = vek::vec::repr_simd::Vec4::new(aabb.min.x, aabb.min.y, aabb.min.z, 0);
                    let max = vek::vec::repr_simd::Vec4::new(aabb.max.x, aabb.max.y, aabb.max.z, 1);
                    // let max = vek::vec::repr_simd::Vec4::new(aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1, 0);
                    let pos = vek::vec::repr_simd::Vec4::new(pos.x, pos.y, pos.z, 0);
                    let is_le = min.cmple_simd(pos);
                    let is_gt = max.cmpgt_simd/*cmpge_simd*/(pos);
                    let cmp = is_le & is_gt;
                    /* let res = */cmp.reduce_and()/*;
                    if !Check::CHECK_AABR && !res {
                        dbg!(prim.id(), tree[prim], min, max, pos, is_le, is_gt, cmp, res);
                    }
                    res */
                }
                #[cfg(not(feature = "simd"))] */
                {
                    (aabb.min.x..aabb.max.x).contains(&pos.x)
                        && (aabb.min.y..aabb.max.y).contains(&pos.y)
                        && (aabb.min.z..aabb.max.z).contains(&pos.z)
                }
            }/*;
            if !Check::CHECK_AABR {
                assert!(res);
            }
            res */
        };
    
        /*loop */{
            match &/*tree[prim]*/prim.0 {
                Node::Empty => false,
                Node::Aabb(aabb) =>
                    /* !Check::CHECK_AABR || */aabb_contains(*aabb, pos),
                Node::Ramp { aabb, inset, dir } => {
                    let inset = (*inset).max(aabb.size().reduce_min());
                    let inner = match dir {
                        Dir::X => Aabr {
                            min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                            max: Vec2::new(aabb.max.x, aabb.max.y),
                        },
                        Dir::NegX => Aabr {
                            min: Vec2::new(aabb.min.x, aabb.min.y),
                            max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                        },
                        Dir::Y => Aabr {
                            min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                            max: Vec2::new(aabb.max.x, aabb.max.y),
                        },
                        Dir::NegY => Aabr {
                            min: Vec2::new(aabb.min.x, aabb.min.y),
                            max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                        },
                    }.made_valid();
                    (/* !Check::CHECK_AABR || */aabb_contains(*aabb, pos))
                        && (inner.projected_point(pos.xy()) - pos.xy())
                            .map(|e| e.abs())
                            .reduce_max() as f32
                            / (inset as f32)
                            < 1.0
                                - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
                },
                Node::Pyramid { aabb, inset } => {
                    let inset = (*inset).max(aabb.size().reduce_min());
                    let inner = Aabr {
                        min: aabb.min.xy() - 1 + inset,
                        max: aabb.max.xy() - inset,
                    }.made_valid();
                    (/* !Check::CHECK_AABR || */aabb_contains(*aabb, pos))
                        && (inner.projected_point(pos.xy()) - pos.xy())
                            .map(|e| e.abs())
                            .reduce_max() as f32
                            / (inset as f32)
                            < 1.0
                                - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
                },
                Node::Gable { aabb, inset, dir } => {
                    let inset = (*inset).max(aabb.size().reduce_min());
                    let inner = if dir.is_y() {
                        Aabr {
                            min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                            max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                        }
                    } else {
                        Aabr {
                            min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                            max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                        }
                    }.made_valid();
                    (/* !Check::CHECK_AABR || */aabb_contains(*aabb, pos))
                        && (inner.projected_point(pos.xy()) - pos.xy())
                            .map(|e| e.abs())
                            .reduce_max() as f32
                            / (inset as f32)
                            < 1.0
                                - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
                },
                Node::Cylinder(aabb) => {
                    (!Check::CHECK_AABR || (aabb.min.z..aabb.max.z).contains(&pos.z))
                        && (pos
                            .xy()
                            .as_()
                            .distance_squared(aabb.as_().center().xy() - 0.5)
                            as f32)
                            < (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2)
                },
                Node::Cone(aabb) => {
                    (!Check::CHECK_AABR || (aabb.min.z..aabb.max.z).contains(&pos.z))
                        && pos
                            .xy()
                            .as_()
                            .distance_squared(aabb.as_().center().xy() - 0.5)
                            < (((aabb.max.z - pos.z) as f32 / aabb.size().d as f32)
                                * (aabb.size().w.min(aabb.size().h) as f32 / 2.0))
                                .powi(2)
                },
                Node::Sphere(aabb) => {
                    (/* !Check::CHECK_AABR || */aabb_contains(*aabb, pos))
                        && pos.as_().distance_squared(aabb.as_().center() - 0.5)
                            < (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2)
                },
                Node::Superquadric { aabb, degree } => {
                    let degree = degree.max(0.0);
                    let center = aabb.center().map(|e| e as f32);
                    let a: f32 = aabb.max.x as f32 - center.x - 0.5;
                    let b: f32 = aabb.max.y as f32 - center.y - 0.5;
                    let c: f32 = aabb.max.z as f32 - center.z - 0.5;
                    let rpos = pos.as_::<f32>() - center;
                    (/* !Check::CHECK_AABR || */aabb_contains(*aabb, pos))
                        && (rpos.x / a).abs().powf(degree)
                            + (rpos.y / b).abs().powf(degree)
                            + (rpos.z / c).abs().powf(degree)
                            < 1.0
                },
                Node::Plane(aabr, origin, gradient) => {
                    // Maybe <= instead of ==
                    (!Check::CHECK_AABR || {
                        // TODO: Enforce aabr.max.x > aabr.min.x and aabr.max.y > aabr.min.y at
                        // compile time.
                        /* #[cfg(feature = "simd")]
                        {
                            let min_pos =
                                vek::vec::repr_simd::Vec4::new(aabr.min.x, aabr.min.y, pos.x, pos.y);
                            let pos_max =
                                vek::vec::repr_simd::Vec4::new(pos.x, pos.y, aabr.max.x - 1, aabr.max.y - 1);
                            let cmp = min_pos.cmple_simd(pos_max);
                            cmp.reduce_and()
                        }
                        #[cfg(not(feature = "simd"))] */
                        {
                            (aabr.min.x..aabr.max.x).contains(&pos.x)
                            && (aabr.min.y..aabr.max.y).contains(&pos.y)
                        }
                    })
                        && pos.z
                            == origin.z
                                + ((pos.xy() - origin.xy())
                                    .map(|x| x.abs())
                                    .as_()
                                    .dot(*gradient) as i32)
                },
                // TODO: Aabb calculation could be improved here by only considering the relevant radius
                Node::Segment { segment, /*radius*/r0, r1, z_scale } => {
                    /* fn length_factor(line: LineSegment3<f32>, p: Vec3<f32>) -> f32 {
                        let len_sq = line.start.distance_squared(line.end);
                        if len_sq < 0.001 {
                            0.0
                        } else {
                            (p - line.start).dot(line.end - line.start) / len_sq
                        }
                    } */
                    let pos = pos.map(|e| e as f32) + 0.5;
                    let distance = segment.end - segment.start;
                    let distance2 = distance.magnitude_squared();
                    let length = pos - segment.start.as_();
                    let t =
                        (length.as_().dot(distance) / distance2).clamped(0.0, 1.0);
                    /* let t = length_factor(*segment, pos); */
                    let projected_point = /*segment.projected_point(pos)*/
                        segment.start + distance * t;

                    let mut diff = projected_point - pos;
                    diff.z *= z_scale;
                    let radius = Lerp::lerp(r0, r1, t)/* - 0.25 */;
                    diff.magnitude_squared() < radius * radius/* - 0.25/*0.01*/*/
                    /* segment.distance_to_point(pos.map(|e| e as f32)) < radius - 0.25 */
                },
                Node::SegmentPrism {
                    segment,
                    radius,
                    height,
                } => {
                    let segment_2d = LineSegment2 {
                        start: segment.start.xy(),
                        end: segment.end.xy(),
                    };
                    let projected_point_2d: Vec2<f32> =
                        segment_2d.as_().projected_point(pos.xy().as_());
                    let xy_check = projected_point_2d.distance(pos.xy().as_()) < radius - 0.25;
                    let projected_z = {
                        let len_sq: f32 = segment_2d
                            .start
                            .as_()
                            .distance_squared(segment_2d.end.as_());
                        if len_sq < 0.1 {
                            segment.start.z as f32
                        } else {
                            let frac = ((projected_point_2d - segment_2d.start.as_())
                                .dot(segment_2d.end.as_() - segment_2d.start.as_())
                                / len_sq)
                                .clamp(0.0, 1.0);
                            (segment.end.z as f32 - segment.start.z as f32) * frac
                                + segment.start.z as f32
                        }
                    };
                    let z_check = (projected_z..=(projected_z + height)).contains(&(pos.z as f32));
                    xy_check && z_check
                },
                /* Node::Sampling(a, f) => {
                    let prim = &tree[*a];
                    if PRINT_MESSAGES {
                        println!("Prim {:?}: {:?}", a.id(), prim);
                    }
                    f(pos) && Self::contains_at::<Check>(cache, tree, prim, pos)
                }, */
                Node::Prefab(p) => !matches!(p.get(pos), Err(_) | Ok(StructureBlock::None)),
                // Intersections mean we can no longer assume we're always in the bounding
                // volume.
                Node::Intersect([a, b]) => {
                    let a_ = &tree[*a];
                    let b_ = &tree[*b];
                    ({
                        if PRINT_MESSAGES {
                            println!("Prim {:?}: {:?}", a.id(), a_);
                        }
                        Self::contains_at::<SubAabr>(/*cache, */tree, a_, pos)
                    }) && {
                        if PRINT_MESSAGES {
                            println!("Prim {:?}: {:?}", b.id(), b_);
                        }
                        Self::contains_at::<SubAabr>(/*cache, */tree, b_, pos)
                    }
                },
                Node::IntersectAll(prims) => {
                    prims.into_iter().all(|prim| {
                        let prim_ = &tree[*prim];
                        if PRINT_MESSAGES {
                            println!("Prim {:?}: {:?}", prim.id(), prim_);
                        }
                        Self::contains_at::<SubAabr>(/*cache, */tree, prim_, pos)
                    })
                },
                // Unions mean we can no longer assume we're always in the bounding box (we need to
                // handle this at the fill level, at least with our current approach...).
                Node::Union(a, b) => {
                    let a_ = &tree[*a];
                    if PRINT_MESSAGES {
                        println!("Prim {:?}: {:?}", a.id(), a_);
                    }
                    let b_ = &tree[*b];
                    if PRINT_MESSAGES {
                        println!("Prim {:?}: {:?}", b.id(), b_);
                    }
                    Self::contains_at::<SubAabr>(/*cache, */tree, a_, pos) ||
                    Self::contains_at::<SubAabr>(/*cache, */tree, b_, pos)
                },
                Node::UnionAll(prims) => {
                    prims.into_iter().any(|prim| {
                        let prim_ = &tree[*prim];
                        if PRINT_MESSAGES {
                            println!("Prim {:?}: {:?}", prim.id(), prim_);
                        }
                        Self::contains_at::<SubAabr>(/*cache, */tree, prim_, pos)
                    })
                },
                // A difference is a weird kind of intersection, so we can't preserve being in the
                // bounding volume for b, but can for a.
                Node::Without(a, b, a_first) => {
                    let a_ = &tree[*a];
                    let b_ = &tree[*b];
                    if *a_first {
                        ({
                            if PRINT_MESSAGES {
                                println!("Prim {:?}: {:?}", a.id(), a_);
                            }
                            Self::contains_at::<Check>(/*cache, */tree, a_, pos)
                        }) && !{
                            if PRINT_MESSAGES {
                                println!("Prim {:?}: {:?}", b.id(), b_);
                            }
                            Self::contains_at::<SubAabr>(/*cache, */tree, b_, pos)
                        }
                    } else {
                        !({
                            if PRINT_MESSAGES {
                                println!("Prim {:?}: {:?}", b.id(), b_);
                            }
                            Self::contains_at::<SubAabr>(/*cache, */tree, b_, pos)
                        }) && {
                            if PRINT_MESSAGES {
                                println!("Prim {:?}: {:?}", a.id(), a_);
                            }
                            Self::contains_at::<Check>(/*cache, */tree, a_, pos)
                        }
                    }
                },
                /* Node::Rotate(prim, mat) => {
                    let aabb = Self::get_bounds(cache, tree, *prim);
                    let diff = pos - (aabb.min + mat.cols.map(|x| x.reduce_min()));
                    Self::contains_at(cache, tree, *prim, aabb.min + mat.transposed() * diff)
                }, */
                Node::Translate(prim, vec) => {
                    let prim_ = &tree[*prim];
                    if PRINT_MESSAGES {
                        println!("Prim {:?}: {:?}", prim.id(), prim_);
                    }
                    Self::contains_at::<Check>(/*cache, */tree, prim_, pos.map2(*vec, i32::saturating_sub))
                },
                /* Node::Scale(prim, vec) => {
                    let center = Self::get_bounds(cache, tree, *prim).center().as_::<f32>()
                        - Vec3::broadcast(0.5);
                    let fpos = pos.as_::<f32>();
                    let spos = (center + ((center - fpos) / vec))
                        .map(|x| x.round())
                        .as_::<i32>();
                    Self::contains_at::<Check>(cache, tree, *prim, spos)
                }, */
                Node::RotateAbout(prim, mat, vec) => {
                    let mat = mat.as_::<f32>().transposed();
                    let vec = vec - 0.5;
                    let prim_ = &tree[*prim];
                    if PRINT_MESSAGES {
                        println!("Prim {:?}: {:?}", prim.id(), prim_);
                    }
                    Self::contains_at::<Check>(/*cache, */tree, prim_, (vec + mat * (pos.as_::<f32>() - vec)).as_())
                },
                /* // Since Repeat is a union, we also can't assume we're in the current bounding box
                // here.
                Node::Repeat(prim, offset, count) => {
                    if count == &0 {
                        false
                    } else {
                        let count = count - 1;
                        let aabb = Self::get_bounds(cache, tree, *prim);
                        let aabb_corner = {
                            let min_red = aabb.min.map2(*offset, |a, b| if b < 0 { 0 } else { a });
                            let max_red = aabb.max.map2(*offset, |a, b| if b < 0 { a } else { 0 });
                            min_red + max_red
                        };
                        let diff = pos - aabb_corner;
                        let min = diff
                            .map2(*offset, |a, b| if b == 0 { i32::MAX } else { a / b })
                            .reduce_min()
                            .clamp(0, count as i32);
                        let pos = pos - offset * min;
                        Self::contains_at::<SubAabr>(cache, tree, *prim, pos)
                    }
                }, */
            }
        }
    }
    /* pub fn sample_at(
        &self,
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
        pos: Vec3<i32>,
        canvas_info: &CanvasInfo,
        old_block: Block,
    ) -> Option<Block> {
        // Top level, so we don't need to check aabr.
        /* println!("\nRoot: {:?}", pos); */
        if Self::contains_at::<TopAabr>(cache, tree, prim, pos) {
            match self {
                Fill::Block(block) => Some(*block),
                Fill::Sprite(sprite) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                } else {
                    old_block.with_sprite(*sprite)
                }),
                Fill::RotatedSprite(sprite, ori) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| Block::air(*sprite))
                } else {
                    old_block
                        .with_sprite(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| old_block.with_sprite(*sprite))
                }),
                Fill::Brick(bk, col, range) => Some(Block::new(
                    *bk,
                    *col + (RandomField::new(13)
                        .get((pos + Vec3::new(pos.z, pos.z, 0)) / Vec3::new(2, 2, 1))
                        % *range as u32) as u8,
                )),
                Fill::Gradient(gradient, bk) => Some(Block::new(*bk, gradient.sample(pos.as_()))),
                Fill::Prefab(p, tr, seed) => p.get(pos - tr).ok().and_then(|sb| {
                    let col_sample = canvas_info.col(canvas_info.wpos)?;
                    block_from_structure(
                        canvas_info.index,
                        *sb,
                        pos - tr,
                        p.get_bounds().center().xy(),
                        *seed,
                        col_sample,
                        Block::air,
                        canvas_info.calendar(),
                    )
                }),
                Fill::Sampling(f) => f(pos),
            }
        } else {
            None
        }
    } */

/* pub fn sample_at<'a, F: Filler>(
    fill: impl Fill,
    arena: &'a bumpalo::Bump,
    cache: &mut BoundsMap,
    tree: &Store<Primitive<'a>>,
    prim: Id<Primitive<'a>>,
    mask: Aabr<i32>,
    filler: &mut F,
    /* mut hit: impl FnMut(Vec3<i32>), */

    /* /* cache: &mut BoundsMap,
    tree: &Store<Primitive<'a>>,
    prim: Id<Primitive<'a>>, */
    pos: Vec3<i32>,
    canvas_info: &CanvasInfo,
    old_block: Block, */
) {
    Fill::get_bounds_disjoint(
        arena,
        cache,
        tree,
        prim,
        mask,
        |pos| filler.map(pos, fill)
    );
    /* // Top level, so we don't need to check aabr.
    /* println!("\nRoot: {:?}", pos); */
    /* if Self::contains_at::<TopAabr>(cache, tree, prim, pos) */{

        // Macro to make sure that calls to get_bounds_disjoint can be statically dispatched;
        // this may or may not be necessary / help.
        //
        // TODO: Evaluate to make sure dynamic dispatch is a bottleneck...
        macro_rules! dispatch_sample_at {
            ($pos:ident, $f: expr) => {
                Fill::get_bounds_disjoint(
                    arena,
                    cache,
                    tree,
                    prim,
                    mask,
                    |$pos| filler.map($pos, $f)
                );
            }
        }

        match self {
            &Fill::Block(block) => {
                dispatch_sample_at!(pos, |_| Some(block));
            },
            &Fill::Sprite(sprite) => {
                dispatch_sample_at!(pos, |old_block| {
                    Some(if old_block.is_filled() {
                        Block::air(sprite)
                    } else {
                        old_block.with_sprite(sprite)
                    })
                });
            },
            &Fill::RotatedSprite(sprite, ori) => {
                dispatch_sample_at!(pos, |old_block| {
                    Some(if old_block.is_filled() {
                        Block::air(sprite)
                            .with_ori(ori)
                            .unwrap_or_else(|| Block::air(sprite))
                    } else {
                        old_block
                            .with_sprite(sprite)
                            .with_ori(ori)
                            .unwrap_or_else(|| old_block.with_sprite(sprite))
                    })
                });
            },
            &Fill::Brick(bk, col, range) => {
                // FIXME: Statically cache somewhere, or maybe just remove in favor of
                // Sampling.
                dispatch_sample_at!(pos, |_| {
                    Some(Block::new(
                        bk,
                        col + (RandomField::new(13)
                            .get((pos + Vec3::new(pos.z, pos.z, 0)) / Vec3::new(2, 2, 1))
                            % range as u32) as u8,
                    ))
                });
            },
            &Fill::Gradient(ref gradient, bk) => {
                dispatch_sample_at!(pos, |_| {
                    Some(Block::new(bk, gradient.sample(pos.as_())))
                });
            },
            &Fill::Prefab(p, tr, seed) => {
                let col_sample = if let Some(col_sample) = canvas_info.col(canvas_info.wpos) {
                    col_sample
                } else {
                    // Don't draw--technically we should probably not assume this much about
                    // the filler implementation though.
                    //
                    // FIXME: Fix this for alternate fillers if it turns out to matter.
                    return
                };
                let index = canvas_info.index;
                let p_bounds = p.get_bounds().center().xy();
                let calendar = canvas_info.calendar();

                dispatch_sample_at!(pos, |_| {
                    p.get(pos - tr).ok().and_then(|&sb| {
                        block_from_structure(
                            index,
                            sb,
                            pos - tr,
                            p_bounds,
                            seed,
                            col_sample,
                            Block::air,
                            calendar,
                        )
                    })
                });
            },
            Fill::Sampling(f) => {
                dispatch_sample_at!(pos, |_| f(pos));
            },
        }
    }/* else {
        None
    }*/ */
} */

    /* pub fn sample_at(
        &self,
        /* cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>, */
        pos: Vec3<i32>,
        canvas_info: &CanvasInfo,
        old_block: Block,
    ) -> Option<Block> {
        // Top level, so we don't need to check aabr.
        /* println!("\nRoot: {:?}", pos); */
        /* if Self::contains_at::<TopAabr>(cache, tree, prim, pos) */{
            match self {
                Fill::Block(block) => Some(*block),
                Fill::Sprite(sprite) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                } else {
                    old_block.with_sprite(*sprite)
                }),
                Fill::RotatedSprite(sprite, ori) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| Block::air(*sprite))
                } else {
                    old_block
                        .with_sprite(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| old_block.with_sprite(*sprite))
                }),
                Fill::Brick(bk, col, range) => Some(Block::new(
                    *bk,
                    *col + (RandomField::new(13)
                        .get((pos + Vec3::new(pos.z, pos.z, 0)) / Vec3::new(2, 2, 1))
                        % *range as u32) as u8,
                )),
                Fill::Gradient(gradient, bk) => Some(Block::new(*bk, gradient.sample(pos.as_()))),
                Fill::Prefab(p, tr, seed) => p.get(pos - tr).ok().and_then(|sb| {
                    let col_sample = canvas_info.col(canvas_info.wpos)?;
                    block_from_structure(
                        canvas_info.index,
                        *sb,
                        pos - tr,
                        p.get_bounds().center().xy(),
                        *seed,
                        col_sample,
                        Block::air,
                        canvas_info.calendar(),
                    )
                }),
                Fill::Sampling(f) => f(pos),
            }
        }/* else {
            None
        }*/
    } */

    fn get_bounds_inner_prim<'a>(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'_>>,
        prim: &Primitive<'_>,
    ) -> Vec<Aabb<i32>> {
        fn or_zip_with<T, F: FnOnce(T, T) -> T>(a: Option<T>, b: Option<T>, f: F) -> Option<T> {
            match (a, b) {
                (Some(a), Some(b)) => Some(f(a, b)),
                (Some(a), _) => Some(a),
                (_, b) => b,
            }
        }
        fn handle_union<'a>(
            cache: &mut BoundsMap,
            tree: &Store<Primitive<'a>>,
            xs: &[Id<Primitive<'a>>],
        ) -> Vec<Aabb<i32>> {
            fn jaccard(x: Aabb<i32>, y: Aabb<i32>) -> f32 {
                let s_intersection = x.intersection(y).size().as_::<f32>().magnitude();
                let s_union = x.union(y).size().as_::<f32>().magnitude();
                s_intersection / s_union
            }
            let mut inputs = Vec::new();
            inputs.extend(xs.into_iter().flat_map(|x| Painter::get_bounds_inner(cache, tree, *x)));
            inputs
            /* let mut results = Vec::new();
            if let Some(aabb) = inputs.pop() {
                results.push(aabb);
                for a in &inputs {
                    let best = results
                        .iter()
                        .enumerate()
                        .max_by_key(|(_, b)| (jaccard(*a, **b) * 1000.0) as usize);
                    match best {
                        Some((i, b)) if jaccard(*a, *b) > 0.3 => {
                            let mut aabb = results.swap_remove(i);
                            aabb = aabb.union(*a);
                            results.push(aabb);
                        },
                        _ => results.push(*a),
                    }
                }
                results
            } else {
                results
            } */
        }

        match &prim.0 {
            Node::Empty => vec![],
            Node::Aabb(aabb) => vec![*aabb],
            Node::Pyramid { aabb, .. } => vec![*aabb],
            Node::Gable { aabb, .. } => vec![*aabb],
            Node::Ramp { aabb, .. } => vec![*aabb],
            Node::Cylinder(aabb) => vec![*aabb],
            Node::Cone(aabb) => vec![*aabb],
            Node::Sphere(aabb) => vec![*aabb],
            Node::Superquadric { aabb, .. } => vec![*aabb],
            Node::Plane(aabr, origin, gradient) => {
                let half_size = aabr.half_size().reduce_max();
                let longest_dist = ((aabr.center() - origin.xy()).map(|x| x.abs())
                    + half_size
                    + aabr.size().reduce_max() % 2)
                    .map(|x| x as f32);
                let z = if gradient.x.signum() == gradient.y.signum() {
                    Vec2::new(0, longest_dist.dot(*gradient) as i32)
                } else {
                    (longest_dist * gradient).as_()
                };
                let aabb = Aabb {
                    min: aabr.min.with_z(origin.z + z.reduce_min().min(0)),
                    max: aabr.max.with_z(origin.z + z.reduce_max().max(0)),
                };
                vec![aabb.made_valid()]
            },
            Node::Segment { segment, /* radius */r0, r1, z_scale } => {
                /* let aabb = Aabb {
                    min: segment.start,
                    max: segment.end,
                }
                .made_valid();
                vec![Aabb {
                    min: (aabb.min - *radius).floor().as_(),
                    max: (aabb.max + *radius).ceil().as_(),
                }] */
                // TODO: Incorporate z scale.
                let aabb = Aabb {
                    min: segment.start,
                    max: segment.end/* - 1.0*/,
                }
                .made_valid();
                // NOTE: Since sampled points get multiplied by z_scale to determine bounds, the
                // actual bounds are smaller by the same factor.
                let mut rad_diff = Vec3::broadcast(r0.max(*r1));
                rad_diff.z /= z_scale;
                vec![Aabb {
                    min: (aabb.min - rad_diff).floor().as_(),
                    max: (aabb.max + rad_diff).ceil().as_(),
                    /* min: (aabb.min - rad_diff).floor().as_(),
                    max: (aabb.max + rad_diff).ceil().as_() + 1, */
                }]
            },
            Node::SegmentPrism {
                segment,
                radius,
                height,
            } => {
                /* // NOTE: This will probably cause some issues with the cost model... ah well.
                let delta = segment.end - segment.start;
                // NOTE: Shearing preserves volume.
                vec![Aabb {
                    min: Vec3::zero(),
                    max: Vec3::new(2.0 * *radius, delta.xy().magnitude(), *height).as_(),
                }] */
                let aabb = Aabb {
                    min: segment.start,
                    max: segment.end,
                }
                .made_valid();
                let min = {
                    let xy = (aabb.min.xy() - *radius).floor();
                    xy.with_z(aabb.min.z).as_()
                };
                let max = {
                    let xy = (aabb.max.xy() + *radius).ceil();
                    xy.with_z((aabb.max.z + *height).ceil()).as_()
                };
                vec![Aabb { min, max }]
            },
            /* Node::Sampling(a, _) => Self::get_bounds_inner(cache, tree, *a), */
            Node::Prefab(p) => vec![p.get_bounds()],
            Node::Intersect([a, b]) => or_zip_with(
                Self::get_bounds_opt(cache, tree, *a),
                Self::get_bounds_opt(cache, tree, *b),
                |a, b| a.intersection(b),
            )
            .into_iter()
            .collect(),
            Node::IntersectAll(prims) => prims.iter()
                .filter_map(|prim| Self::get_bounds_opt(cache, tree, *prim))
                .reduce(Aabb::intersection)
                .into_iter()
                .collect(),
            Node::Union(a, b) => handle_union(cache, tree, &[*a, *b]),
            Node::UnionAll(prims) => handle_union(cache, tree, prims),
            Node::Without(a, _, _) => Self::get_bounds_inner(cache, tree, *a),
            /* Node::Rotate(prim, mat) => Self::get_bounds_inner(cache, tree, *prim)
                .into_iter()
                .map(|aabb| {
                    let extent = *mat * Vec3::from(aabb.size());
                    let new_aabb: Aabb<i32> = Aabb {
                        min: aabb.min,
                        max: aabb.min + extent,
                    };
                    new_aabb.made_valid()
                })
                .collect(), */
            Node::Translate(prim, vec) => Self::get_bounds_inner(cache, tree, *prim)
                .into_iter()
                .map(|aabb| Aabb {
                    min: aabb.min.map2(*vec, i32::saturating_add),
                    max: aabb.max.map2(*vec, i32::saturating_add),
                })
                .collect(),
            /* Node::Scale(prim, vec) => Self::get_bounds_inner(cache, tree, *prim)
                .into_iter()
                .map(|aabb| {
                    let center = aabb.center();
                    Aabb {
                        min: center + ((aabb.min - center).as_::<f32>() * vec).as_::<i32>(),
                        max: center + ((aabb.max - center).as_::<f32>() * vec).as_::<i32>(),
                    }
                })
                .collect(), */
            Node::RotateAbout(prim, mat, vec) => Self::get_bounds_inner(cache, tree, *prim)
                .into_iter()
                .map(|aabb| {
                    let mat = mat.as_::<f32>();
                    // - 0.5 because we want the point to be at the minimum of the voxel
                    let vec = vec - 0.5;
                    let new_aabb = Aabb::<f32> {
                        min: vec + mat * (aabb.min.as_() - vec),
                        // - 1 becuase we want the AABB to be inclusive when we rotate it, we then
                        //   add 1 back to make it exclusive again
                        max: vec + mat * ((aabb.max - 1).as_() - vec),
                    }
                    .made_valid();
                    Aabb::<i32> {
                        min: new_aabb.min.as_(),
                        max: new_aabb.max.as_() + 1,
                    }
                })
                .collect(),
            /* Node::Repeat(prim, offset, count) => {
                if count == &0 {
                    vec![]
                } else {
                    let count = count - 1;
                    Self::get_bounds_inner(cache, tree, *prim)
                        .into_iter()
                        .map(|aabb| Aabb {
                            min: aabb
                                .min
                                .map2(aabb.min + offset * count as i32, |a, b| a.min(b)),
                            max: aabb
                                .max
                                .map2(aabb.max + offset * count as i32, |a, b| a.max(b)),
                        })
                        .collect()
                }
            }, */
        }
    }

    fn get_bounds_inner<'a: 'b, 'b>(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'b>>,
        prim: Id<Primitive<'b>>,
    ) -> Vec<Aabb<i32>> {
        let id = prim.id();
        if let Some(bounds) = cache.get(&id) {
            bounds.clone()
        } else {
            let bounds = Self::get_bounds_inner_prim(cache, tree, &tree[prim]);
            cache.insert(id, bounds.clone());
            bounds
        }
    }

    /* pub fn get_bounds_disjoint(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
    ) -> Vec<Aabb<i32>> {
        Self::get_bounds_inner(cache, tree, prim)
    } */

    fn get_bounds_disjoint_inner<'a, 'b>(
        arena: &'a bumpalo::Bump,
        cache: &mut BoundsMap,
        tree: &'b Store<Primitive<'b>>,
        mut prim: Id<Primitive<'b>>,
        mut stack_bot: usize,
        stack: &mut Vec<StackMember<'b>>,
        mut mat: Mat4<i32>,
        mut mask: Aabb<i32>,
        hit: &mut impl FnMut(Vec3<i32>),
    ) {
        // The idea here is that we want to evaluate as many terms as we can *without* having to
        // sample point by point.  Therefore, we divide iteration into two parts...
        //
        // * Block-by-block iteration that iterates over an AABB and evaluates the chosen primitive
        //   at each block.  This is what calls the hit function.
        //
        // * Top-level iteration that proceeds by block and produces disjoint AABBs.  This
        //   continues until we hit a negative type (i.e. an intersection, negation, etc.)
        //   or a primitive, at which point we switch to block-by-block iteration.  We also always
        //   switch to block-by-block iteration when the volume size gets low enough (currently
        //   4×4×4 blocks) under the theory that at that point the overhead of tracking the AABBs
        //   will exceed the performance benefits.

        // Top level, so we don't need to check the aabr for validity; instead, we iterate over the
        // provided AABB!
        #[inline]
        fn aabb_iter(
            mut aabb: Aabb<i32>,
            mat: Mat4<i32>,
            mask: Aabb<i32>,
            mut test: impl FnMut(Vec3<i32>) -> bool,
            hit: &mut impl FnMut(Vec3<i32>)
        ) {
            // We use the matrix to determine which offset/ori to use for each iteration.
            // Specifically, mat.x will represent the real x axis, mat.y the real y axis,
            // and mat.z the real z axis.
            //
            // Because of this, to find the axes we actually want to use for our own purposes, we
            // need to first read the offset, then transpose, then read the three axes.
            let offset = Vec3::<i32>::from(mat.cols[3]);
            // TODO: Optimize
            let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
            let dz = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_z())*/mat.cols[2]);
            let dy = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_y())*/mat.cols[1]);
            let dx = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_x())*/mat.cols[0]);
            // TODO: Optimize.
            let mut mask: Aabb<i32> = Aabb {
                min: Vec3::from(inv_mat * Vec4::from_point(mask.min))/*.with_z(aabb.min.z)*/,
                max: Vec3::from(inv_mat * Vec4::from_point(mask.max - 1))/*.with_z(aabb.max.z)*/,
            };
            mask.make_valid();
            mask.max += 1;
            aabb.intersect(mask);
            if PRINT_MESSAGES {
                println!("\n\nVol: {:?}", aabb);
            }
            let mut hit_count = 0;
            // TODO: Optimize.
            // It *might* actually make sense to make a const generic so we specialize the six
            // possible orientations differently, and avoid vector arithmetic, but for now we
            // don't do that.
            //
            // TODO: Orient by *output* z first, then *output* y, then *output* x, instead of
            // *input* (which is what we're currently doing because it's easy).
            /* let mat_ = mat.transpose();
            let dz = mat * Vec4::unit_z();
            let dy = mat * Vec4 */
            //
            // Apply matrix transformations to find the new bounding box.
            //
            // NOTE: Matrix only supports floats for FP scaling, which is questionable.
            /* aabb.iter_mut().for_each(|v| mat * v);
            // It may have a bad orientation, so make sure it's valid (TODO: consider fixing this
            // at rotation time instead?).
            aabb.make_valid(); */
            // TODO: Optimize a lot!

             // Whether to optimize the layering for input or output should likely depend on
             // the primitive type, but for the most part we assume that sampling output in
             // layered order is more important, since most of our sampling primitives are
             // not memory-bound (we currently do zyx iteration, but maybe other orders are
             // better, who knows?).
             //
             // Another potential optimization when we are *actually* operating over primitives and
             // convex solids: assuming the primitives are convex, we can start by sampling the
             // corner in the next plane that is within the bounding box and touching the corner of
             // the last AABB.  If it's empty, we can skip anything further away, thanks to
             // convexity.  Moreover, as we move towards the center of the plane, we can skip
             // testing any remaining components on the two lines in the corner (or else we'd fail
             // to be convex).
             //
             // We can even (somewhat!) extend this to difference of unions, since this is
             // equivalent to intersection by negations.  Specifically, we focus on the case of
             // disjoint union (in which case there are no intersecting components).  Then we know
             // that *within* the bounding box of each union'd component, we'll eventually reach
             // something convex, and we can find the intersection between the positive convex part
             // and the negative convex part.  We can use these as bounds to carve out of the
             // positive part, before reporting all the interior points.  The final set is still
             // not convex, but we don't have to do a huge amount of work here.  As long as the
             // union is disjoint, we should end up in a "sufficiently small" convex set without
             // (?) exponential blowup.
            (aabb.min.z..aabb.max.z).for_each(|z| {
                let offset = offset + dz * z;
                (aabb.min.y..aabb.max.y).for_each(|y| {
                    let offset = offset + dy * y;
                    (aabb.min.x..aabb.max.x)
                        .inspect(|&x| if PRINT_MESSAGES { println!("\nPos: {:?}", Vec3::new(x, y, z)) })
                        .filter(|&x| test(Vec3::new(x, y, z)))
                        .inspect(|_| if PRINT_MESSAGES { hit_count += 1 })
                        .map(|x| offset + dx * x)
                        .for_each(&mut *hit)
                })
            });
            if PRINT_MESSAGES {
                println!("Hit rate: {:?} / {:?}", hit_count, aabb.size().product());
            }
        }
            /* const CHECK_ABBR: bool = Check::CHECK_AABR; */
            /* !Check::CHECK_AABR ||
            /* let res = */{
                /* #[cfg(feature = "simd")]
                {
                    // TODO: Enforce aabr.max.x > aabr.min.x and aabr.max.y > aabr.min.y at
                    // compile time.
                    /* let min = vek::vec::repr_simd::Vec8::new(
                        aabb.min.x, aabb.min.y, aabb.min.z, 0,
                        pos.x, pos.y, pos.z, 0,
                    );
                    let max = vek::vec::repr_simd::Vec8::new(
                        pos.x, pos.y, pos.z, 0,
                        aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1, 0,
                    );
                    let cmp = min.cmple_simd(max); */
                    let min = vek::vec::repr_simd::Vec4::new(aabb.min.x, aabb.min.y, aabb.min.z, 0);
                    let max = vek::vec::repr_simd::Vec4::new(aabb.max.x, aabb.max.y, aabb.max.z, 1);
                    // let max = vek::vec::repr_simd::Vec4::new(aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1, 0);
                    let pos = vek::vec::repr_simd::Vec4::new(pos.x, pos.y, pos.z, 0);
                    let is_le = min.cmple_simd(pos);
                    let is_gt = max.cmpgt_simd/*cmpge_simd*/(pos);
                    let cmp = is_le & is_gt;
                    /* let res = */cmp.reduce_and()/*;
                    if !Check::CHECK_AABR && !res {
                        dbg!(prim.id(), tree[prim], min, max, pos, is_le, is_gt, cmp, res);
                    }
                    res */
                }
                #[cfg(not(feature = "simd"))] */
                {
                    (aabb.min.x..aabb.max.x).contains(&pos.x)
                        && (aabb.min.y..aabb.max.y).contains(&pos.y)
                        && (aabb.min.z..aabb.max.z).contains(&pos.z)
                }
            }/*;
            if !Check::CHECK_AABR {
                assert!(res);
            }
            res */ */

        // Note that because we are on the top level, we never have to worry about AABB
        // containment--we automatically inherit the AABB of the value.  In theory, we should
        // need a mask for this to bound to chunks, but it is likely that in the future we want
        // to generate sites all at once anyway, eliminating the need for an additional mask.
        //
        // Therefore, this part of the code resembles normalization by evaluation, specifically
        // finding the spine and converting to weak head normal form.

        // The function to call when we finally hit is performance-sensitive, so we don't want to
        // make the function that gets called in the AABB inner loop require dynamic dispatch.
        // Our observation is that there is not much of a tradeoff at the top level--any sampler
        // function that gets pushed down was already dynamic, and will be evaluated
        // disjunctively, so we just have to avoid dynamic dispatch in the case that there was
        // no sample (*most* sample figures are also pretty simple, although one in dungeons is
        // fairly complicated, but I've been told not to worry about dungeons or samplers too
        // much).
        //
        // Another thing that is likely to get pushed down is a difference operation.  However,
        // it should be clear that which evaluation order to use is dependent on the volume it
        // ultimately takes a bite out of... and also on whether more differences are discovered
        // down the line (since they can just be evaluated disjunctively with the original!).
        // Again this is more like fast and simple term rewriting than optimization.  In
        // particular, since our cost drops as we go deeper into the tree, we should expect that
        // eventually it will make more sense to evaluate ourselves first; if that doesn't happen,
        // it will be because we saw more differences, leading to higher estimates for them (due to
        // it being a union estimate); since we delay cost evaluation of the difference until we
        // actually see a difference, and the relative cost of difference first vs us first
        // monotonically increases until we see another difference, we can reasonably assume we're
        // not missing anything by skipping these evaluations.
        //
        // Another pushdown candidate is Repeat-related.  In theory, we should just be able to
        // evaluate the body once, and then (once it's been normalized) copy-paste it to the other
        // areas!  The way Coq would want us to do it (and maybe the right way) would be to
        // evaluate shared, fully reduced terms to bitmasks (happily I think we know ahead of time
        // whether the term is actually shared?), maybe integrating this into the cost model.  The
        // real question is whether a more general *partial* evaluation strategy is worth it for
        // terms that aren't fully reduced.  One nice thing about the way we handle transformations
        // is that any fully reduced model can be easily translated in that way, and right now
        // transformation matrix is one of our only real contexts (besides difference and
        // sampling, so the decision of whether to push them down could be informed by whether they
        // were shared, the cost model, etc.).  In any case, something like Repeat is limited
        // enough that these things would barely matter (outside of intersect, difference, and
        // *maybe* sampling, which can't be trusted; none of these are used at the moment).
        //
        // Finally, the most obvious pushdown candidate (already discussed): transformation matrix
        // will be fully pushed down, and not evaluated until the first negative term.  Since
        // outside of Repeat we don't have much interesting context-dependent sharing (we assume
        // unions are mostly disjoint), we can do this without many problems, even pushing through
        // difference and sampling when possible.
        //
        // So to start off (all only at the top level):
        //
        // * Everything can push through unions.
        // * Repeat will be pushed down only until the first negative term, but will not
        //   be pushed down through difference or sampling (because these are not invariant with
        //   respect to changes to the model matrix).  Currently Repeat is only used with
        //   primitives I think, so this shouldn't matter.
        // * Difference will be pushed down and evaluated as unions of each other.  We'll
        //   reevaluate the cost each time we hit a new difference.  They can push down through
        //   other Differences, through Samplers (through the positive side, specifically), but not
        //   through Repeats.
        // * Sampling can be pushed through other Samplers (creates a new closure capturing the old
        //   one), through Differences, but not through Repeats.
        // * Transformation matrices can be pushed through Difference and Sampling, but not Repeat
        //   (at least, not naively; they could work if we could do a suspended environment context
        //   for Repeat, which holds the transform as of when they were created so it can be
        //   applied after translating the transformed version from afterwards).
        let handle_intersect = |stack: &mut Vec<StackMember<'b>>, mat, mask, x, rhs| {
             // We mask with the translated intersection bounds
             // (forcing).  and then *push down* the branches
             // of higher cost as a suspended term to be evaluated later.
             //
             // TODO: Actually lazily evaluate, maybe?
             //
             // TODO: Don't recurse.
             //
             // Push an intersection stack member.
             stack.push(StackMember::MulConj {
                 mat,
                 /* mask, */
                 rhs,
             });
        };

        // NOTE: We have to handle the annoying cases where both min and max
        // are on one side of the center (after masking).  For now, we just
        // explicitly check their signs and handle this case separately, but
        // there's probably a more elegant way.
        //
        // NOTE: In all code below, we can assume aabb.min.z ≤ aabb.max.z from
        // the call to is_valid.
        //
        // FIXME: To prove casts of positive i32 to usize correct,
        // statically assert that u32 fits in usize.
        let make_bounds = |min: i32, max: i32| {
            let (bounds_min, bounds_max) =
                if min >= 0 {
                    // NOTE: Since max ≥ min > 0, we know max
                    // is positive.
                    (
                        min as usize..min as usize,
                        min as usize..max as usize,
                    )
                } else if max < 0 {
                    // NOTE: -max is positive because max is
                    // negative.  Therefore, 1 - max is also positive.
                    // 1 - min ≥ 1 - max because
                    // max ≥ min, so we can assume this forms a valid
                    // range.
                    (
                        (1 - max) as usize..(1 - min) as usize,
                        (1 - max) as usize..(1 - max) as usize,
                    )
                } else {
                    // NOTE: 1 < 1 - min since min < 0.
                    // max is already verified to be non-negative,
                    // so it's ≥ 0.  So we can assume all ranges are valid.
                    (
                        1..(1 - min) as usize,
                        0..max as usize,
                    )
                };

            // NOTE: Since min is non-increasing and max is non-decreasing,
            // and both sides form a valid range from start..end, max-ing
            // the starts and min-ing the ends must still form a valid range.
            let bounds =
                bounds_min.start.min(bounds_max.start)..
                bounds_min.end.max(bounds_max.end);
            (bounds_min, bounds_max, bounds)
        };

        loop {
            let prim_ = &tree[prim];
            if PRINT_MESSAGES {
                println!("Prim {:?}: {:?}", prim.id(), prim_);
            }
            match &prim_.0 {
                // InterSectAll([]) should not happen, but we treat it the same as Empty currently.
                Node::Empty | Node::IntersectAll([]) => return,
                // Arguably we should have a super fast special case for this, but for now
                // we just defer to the usual primitive handler.
                Node::Aabb(aabb) => {
                    // If we have an intersection on the stack, push it down as a union.
                    if let &[StackMember::MulConj { mat: mat_, rhs: &[prim__] }, ..] = &stack[stack_bot..] {
                        /* let offset = Vec3::<i32>::from(mat.cols[3]);
                        // TODO: Optimize
                        let inv_mat = mat./*inverted_affine_transform_no_scale*/transposed();
                        let dz = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_z())*/mat.cols[2]);
                        let dy = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_y())*/mat.cols[1]);
                        let dx = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_x())*/mat.cols[0]); */
                        // TODO: Optimize.
                        let mut aabb = Aabb {
                            min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                        };
                        aabb.make_valid();
                        aabb.max += 1;
                        mask.intersect(aabb);
                        if !mask.is_valid() {
                            return;
                        }
                        /* let prim2_ = &tree[prim2];
                        if let Node::UnionAll(prims2) = prim2_ {
                            let mask = aabb;
                            prims.into_iter().for_each(move |prim2| {
                                Self::get_bounds_disjoint_inner(arena, cache, tree, *prim, stack, mat, mask, &mut *hit)
                            });
                        } */
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        /* return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            prim,
                            &mut Vec::new(),
                            mat,
                            mask,
                            hit,
                        ); */
                        // *stack = Vec::new();
                    } else {
                    /* !Check::CHECK_AABR || */return aabb_iter(*aabb, mat, mask, |_| true, &mut *hit)
                    }
                },
                Node::Ramp { aabb/* , inset, dir */, .. }
                | Node::Pyramid { aabb, .. }
                | Node::Gable { aabb, .. }
                | Node::Cone(aabb, ..)
                // | Node::Cylinder(aabb)
                // | Node::Sphere(aabb)
                // | Node::Superquadric { aabb, .. }
                if stack.len() > stack_bot => {
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let mut aabb = Aabb {
                            min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                        };
                        aabb.make_valid();
                        aabb.max += 1;
                        mask.intersect(aabb);
                        if !mask.is_valid() {
                            return;
                        }
                        let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            /*prim__*/prim,
                            /* &mut Vec::new(), */
                            stack_bot,
                            stack,
                            /*mat_*/mat,
                            mask,
                            &mut ((&mut |pos| {
                                let pos_ = Vec3::from(inv_mat * Vec4::from_point(pos));
                                if Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos_) {
                                    hit(pos);
                                }
                            }) as &mut dyn FnMut(Vec3<i32>)),
                        );
                    }
                },
                Node::Cylinder(aabb) if stack.len() > stack_bot => {
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let mut aabb = Aabb {
                            min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                        };
                        aabb.make_valid();
                        aabb.max += 1;
                        mask.intersect(aabb);
                        if !mask.is_valid() {
                            return;
                        }
                        let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        let center = aabb.as_().center().xy() - 0.5;
                        let radius_2 = (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2);
                        return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            /*prim__*/prim,
                            /* &mut Vec::new(), */
                            stack_bot,
                            stack,
                            /*mat_*/mat,
                            mask,
                            &mut ((&mut |pos| {
                                let pos_ = Vec3::<i32>::from(inv_mat * Vec4::from_point(pos));
                                if pos_.xy().as_().distance_squared(center) < radius_2 {
                                    hit(pos);
                                }
                            }) as &mut dyn FnMut(Vec3<i32>)),
                        );
                    }
                },
                Node::Sphere(aabb) if stack.len() > stack_bot => {
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let mut aabb = Aabb {
                            min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                        };
                        aabb.make_valid();
                        aabb.max += 1;
                        mask.intersect(aabb);
                        if !mask.is_valid() {
                            return;
                        }
                        let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        let center = aabb.as_().center() - 0.5;
                        let radius_2 = (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2);
                        return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            /*prim__*/prim,
                            /* &mut Vec::new(), */
                            stack_bot,
                            stack,
                            /*mat_*/mat,
                            mask,
                            &mut ((&mut |pos| {
                                let pos_ = Vec3::<i32>::from(inv_mat * Vec4::from_point(pos));
                                if pos_.as_().distance_squared(center) < radius_2 {
                                    hit(pos);
                                }
                            }) as &mut dyn FnMut(Vec3<i32>)),
                        );
                    }
                },
                Node::Superquadric { aabb, degree } if stack.len() > stack_bot => {
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let degree = degree.max(1.0);

                        if degree == 2.0 {
                            let center = aabb.center()/*.map(|e| e as f32)*/;
                            mat = mat * Mat4::<i32>::translation_3d(center);
                            let mut aabb = Aabb {
                                // TODO: Try to avoid generating 3/4 of the circle; the interaction with
                                // masking will require care.
                                min: aabb.min - center,
                                max: aabb.max - center,
                            };
                            let mut a_ = aabb.max.x;
                            let mut b_ = aabb.max.y;
                            let mut c_ = aabb.max.z;
                            let a: f32 = a_ as f32/* - center.x as f32 */- 0.5;
                            let b: f32 = b_ as f32/* - center.y as f32 */- 0.5;
                            let c: f32 = c_ as f32/* - center.z as f32 */- 0.5;
                            // NOTE: Guaranteed positive since a,b,c are all positive (due to AABB being
                            // enforced valid at construction time) and degree must be ≥ 1.0.
                            let a_inv_pow = a.recip().powf(degree);
                            let b_inv_pow = b.recip().powf(degree);
                            let c_inv_pow = c.recip().powf(degree);
                            let abc_inv_pow = Vec3::new(a_inv_pow, b_inv_pow, c_inv_pow);

                            let mut aabb = Aabb {
                                min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                                max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                            };
                            aabb.make_valid();
                            aabb.max += 1;
                            mask.intersect(aabb);
                            if !mask.is_valid() {
                                return;
                            }
                            let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                            mat = mat_;
                            // println!("Intersected {}: {}?", prim.id(), prim_.id());
                            prim = prim__;
                            stack_bot += 1;
                            return Self::get_bounds_disjoint_inner(
                                arena,
                                cache,
                                tree,
                                /*prim__*/prim,
                                /* &mut Vec::new(), */
                                stack_bot,
                                stack,
                                /*mat_*/mat,
                                mask,
                                &mut ((&mut |pos| {
                                    let pos_ = Vec3::<i32>::from(inv_mat * Vec4::from_point(pos));
                                    let rpos = pos_.as_::<f32>()/* - center*/;
                                    if (rpos * rpos).dot(abc_inv_pow) < 1.0 {
                                        hit(pos);
                                    }
                                }) as &mut dyn FnMut(Vec3<i32>)),
                            );
                        } else {
                            let mut aabb = Aabb {
                                min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                                max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                            };
                            aabb.make_valid();
                            aabb.max += 1;
                            mask.intersect(aabb);
                            if !mask.is_valid() {
                                return;
                            }
                            let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                            mat = mat_;
                            // println!("Intersected {}: {}?", prim.id(), prim_.id());
                            prim = prim__;
                            stack_bot += 1;
                            return Self::get_bounds_disjoint_inner(
                                arena,
                                cache,
                                tree,
                                /*prim__*/prim,
                                /* &mut Vec::new(), */
                                stack_bot,
                                stack,
                                /*mat_*/mat,
                                mask,
                                &mut ((&mut |pos| {
                                    let pos_ = Vec3::<i32>::from(inv_mat * Vec4::from_point(pos));
                                    if Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos_) {
                                        hit(pos);
                                    }
                                }) as &mut dyn FnMut(Vec3<i32>)),
                            );
                        }
                    }
                },
                Node::Segment { segment, /*radius*/r0, r1, z_scale } if stack.len() > stack_bot => {
                    // TODO: Optimize further?
                    /* let aabb = Self::get_bounds(cache, tree, prim);
                    if !(aabb.size().w > 8 || aabb.size().h > 8 || aabb.size().d > 16) {
                        /* return aabb_iter(
                            aabb.as_(),
                            mat,
                            mask,
                            |_| true,
                            // |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                            hit,
                        ); */
                        return
                    } */
                    // NOTE: vol(frustum) = 1/3 h(A(r₀) + A(r₁) + √(A(r₀)A(r₁)))
                    // = 1/3 h (r₀² + r₁² + √(r₀²r₁²))
                    // = 1/3 h (r₀² + r₁² + r₀r₁)
                    let y_size = (segment.end - segment.start).magnitude();
                    let r0 = r0 - 0.5;
                    let r1 = r1 - 0.5;
                    let radius_squared = (r0 * r0 + r1 * r1 + r0 * r1) / 3.0;
                    let radius = radius_squared.sqrt();
                    let mat__ = mat.as_::<f32>();
                    // Orient the matrix to point towards the segment.
                    //
                    // NOTE: We try to make sure the up vector is aligned to Z, unless we are
                    // orthogonal to X in which case we align with X.  Since we are drawing a
                    // prism, there's no real difference between X, Y, or Z here, as long as it's
                    // an orthogonal direction.  Note that we already know that not all three
                    // segments differ.
                    //
                    // TODO: Avoid this branch by returning a different primitive if we're a
                    // straight line.
                    let up = if segment.start.x == segment.end.x {
                        Vec3::unit_x()
                    } else {
                        Vec3::unit_z()
                    };
                    let rotate_mat = Mat4::<f32>::model_look_at_lh(segment.start, segment.end, up);
                    // NOTE: Pre-inverted.
                    let translate_mat = Mat4::<f32>::new(
                        1.0, 0.0, 0.0, -radius,
                        0.0, 1.0, 0.0, -radius,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    );
                    // Inverted, so multiply backwards
                    let rotate_translate_mat = rotate_mat * translate_mat;
                    let mat__ = mat__ * rotate_translate_mat;
                    // Fake AABB to draw.
                    let mut aabb = Aabb {
                        min: Vec3::zero(),
                        max: Vec3::new(2.0 * radius/* + 1.0*/, 2.0 * radius/* + 1.0*/, y_size/* + 1.0*/),
                    };
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let mut aabb_ = Aabb::<f32> {
                            min: Vec3::from(mat__ * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat__ * Vec4::from_point(aabb.max - 1.0))/*.with_z(aabb.max.z)*/,
                        };
                        aabb_.make_valid();
                        /* aabb_.max += 1.0;
                        aabb_.min = aabb_.min.map(f32::floor);
                        aabb_.max = aabb_.max.map(f32::ceil); */
                        let mut aabb_ = aabb_.as_::<i32>();
                        aabb_.max += 1;
                        mask.intersect(aabb_/*.as_()*/);
                        if !mask.is_valid() {
                            return;
                        }
                        let inv_mat = mat__/*mat_*//*.as_::<f32>()*/.inverted_affine_transform_no_scale/*transposed*/()/*.as_::<i32>()*/;
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        // let prim__ = &tree[prim__];
                        return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            prim,
                            stack_bot,
                            /* &mut Vec::new(), */
                            stack,
                            mat,
                            mask,
                            &mut ((&mut |pos: Vec3<i32>| {
                                let pos_ = Vec3::from(inv_mat * Vec4::from_point(pos.as_()));
                                if aabb.contains_point(pos_) {
                                    hit(pos);
                                }
                                /* if Self::contains_at::<TopAabr>(/*cache, */tree, prim_/*prim__*/, pos_) {
                                    hit(pos);
                                } */
                            }) as &mut dyn FnMut(Vec3<i32>)),
                            // hit,
                        );
                    }
                }
                Node::Plane(..)
                | Node::SegmentPrism { .. }
                | Node::Prefab(..) if stack.len() > stack_bot => {
                    // TODO: Improve?
                    let aabb = Self::get_bounds(cache, tree, prim);
                    if let StackMember::MulConj { mat: mat_, rhs: &[prim__] } = stack[stack_bot] {
                        let mut aabb = Aabb {
                            min: Vec3::from(mat * Vec4::from_point(aabb.min))/*.with_z(aabb.min.z)*/,
                            max: Vec3::from(mat * Vec4::from_point(aabb.max - 1))/*.with_z(aabb.max.z)*/,
                        };
                        aabb.make_valid();
                        aabb.max += 1;
                        mask.intersect(aabb);
                        if !mask.is_valid() {
                            return;
                        }
                        let inv_mat = mat/*mat_*/.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                        mat = mat_;
                        // println!("Intersected {}: {}?", prim.id(), prim_.id());
                        prim = prim__;
                        stack_bot += 1;
                        // let prim__ = &tree[prim__];
                        return Self::get_bounds_disjoint_inner(
                            arena,
                            cache,
                            tree,
                            prim,
                            stack_bot,
                            /* &mut Vec::new(), */
                            stack,
                            mat,
                            mask,
                            &mut ((&mut |pos| {
                                let pos_ = Vec3::from(inv_mat * Vec4::from_point(pos));
                                if Self::contains_at::<TopAabr>(/*cache, */tree, prim_/*prim__*/, pos_) {
                                    hit(pos);
                                }
                            }) as &mut dyn FnMut(Vec3<i32>)),
                            // hit,
                        );
                    }
                },
                Node::Ramp { aabb/* , inset, dir */, .. }
                | Node::Pyramid { aabb, .. }
                | Node::Gable { aabb, .. }
                | Node::Cone(aabb, ..)
                    => {
                   return aabb_iter(
                       *aabb,
                       mat,
                       mask,
                       |pos| Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos),
                       hit,
                   )
                },
                Node::Cylinder(aabb) => {
                    let center = aabb.as_().center().xy() - 0.5;
                    let radius_2 = (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2);
                    return aabb_iter(
                        *aabb,
                        mat,
                        mask,
                        |pos| pos.xy().as_().distance_squared(center) < radius_2,
                        hit,
                    )
                },
                Node::Sphere(aabb) => {
                    let center = aabb.as_().center() - 0.5;
                    let radius_2 = (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2);
                    return aabb_iter(
                        *aabb,
                        mat,
                        mask,
                        |pos| pos.as_().distance_squared(center) < radius_2,
                        hit,
                    )
                },
                Node::Superquadric { aabb, degree } => {
                    // NOTE: We should probably switch to superellipsoids as they are a bit more
                    // constrained / easier to solve (and special-case the Z axis).
                    //
                    // A superquadric is very symmetric (ignoring masking):
                    //
                    // * symmetric along the x, y, and z axes.
                    // * radially symmetric (up to scaling).
                    //
                    // For now, we don't make much effort to benefit from radial symmetry.
                    // However, do make an effort to benefit from symmetry along the axes, as
                    // follows:
                    //
                    // * generate one quadrant using SIMD, masked only up to the largest used
                    //   axis in each direction.

                    let degree = degree.max(1.0);

                    /* let half_size = aabb.half_size();
                    let aabb = Aabb {
                        min: -Vec3::from(half_size),
                        max: Vec3::from(half_size),
                    }; */
                    /* let a: f32 = aabb.max.x as f32 /*- center.x as f32 */- 0.5;
                    let b: f32 = aabb.max.y as f32 /*- center.y as f32 */- 0.5;
                    let c: f32 = aabb.max.z as f32 /*- center.z as f32 */- 0.5; */
                    /* let a: f32 = aabb.max.x.min(-aabb.min.x) as f32 /*- center.x as f32 */- 0.5;
                    let b: f32 = aabb.max.y.min(-aabb.min.y) as f32 /*- center.y as f32 */- 0.5;
                    let c: f32 = aabb.max.z.min(-aabb.min.z) as f32 /*- center.z as f32 */- 0.5; */
                    /* let a: f32 = aabb.max.x.max(-aabb.min.x) as f32 /*- center.x as f32 */- 0.5;
                    let b: f32 = aabb.max.y.max(-aabb.min.y) as f32 /*- center.y as f32 */- 0.5;
                    let c: f32 = aabb.max.z.max(-aabb.min.z) as f32 /*- center.z as f32 */- 0.5; */

                    // NOTE: Separate closure to help convince LLVM to outline this.
                    let mut do_superquadric = || {
                        // Reverse translation from center.
                        let center = aabb.center()/*.map(|e| e as f32)*/;
                        mat = mat * Mat4::<i32>::translation_3d(center);
                        let mut aabb = Aabb {
                            // TODO: Try to avoid generating 3/4 of the circle; the interaction with
                            // masking will require care.
                            min: aabb.min - center,
                            max: aabb.max - center,
                        };
                        let mut a_ = aabb.max.x;
                        let mut b_ = aabb.max.y;
                        let mut c_ = aabb.max.z;
                        // We rotate around the center so that the xy axes are shortest.
                        let sizes = aabb.size();
                        if sizes.h > sizes.d {
                            // Rotate about x axis so that y becomes z and z becomes y.
                            /* mat = mat * Mat4::new(
                                1, 0, 0, 0,
                                0, 0, -1, -1,
                                0, 1, 0, 0,
                                0, 0, 0, 1); */
                            mat = mat * Mat4::new(
                                1, 0, 0, 0,
                                0, 0, 1, 0,
                                0, -1, 0, /*-1*/0,
                                0, 0, 0, 1);
                            // x₁ = x₀
                            // y₁ = z₀
                            // z₁ = -y₀ - 1 ⇒ y₀ = -z₁ - 1

                            aabb.min.z = /*-aabb.min.z + 1*/-aabb.min.z + 1;
                            aabb.max.z = /*-(aabb.max.z - 1)*//*-aabb.max.z + 1*/-aabb.max.z + 1;
                            core::mem::swap(&mut aabb.min.z, &mut aabb.max.z);

                            /* aabb.min.y = /*-aabb.min.y + 1*/-aabb.min.y;
                            aabb.max.y = /*-(aabb.max.y - 1)*/-aabb.max.y;
                            core::mem::swap(&mut aabb.min.y, &mut aabb.max.y); */

                            core::mem::swap(&mut aabb.min.y, &mut aabb.min.z);
                            core::mem::swap(&mut aabb.max.y, &mut aabb.max.z);
                            core::mem::swap(&mut b_, &mut c_);
                        } else if sizes.w > sizes.d {
                            // Rotate about y axis so that x becomes z and z becomes x.
                            /* mat = mat * Mat4::new(
                                0, 0, 1, 0,
                                0, 1, 0, 0,
                                -1, 0, 0, -1,
                                0, 0, 0, 1); */
                            mat = mat * Mat4::new(
                                0, 0, -1, /*-1*/0,
                                0, 1, 0, 0,
                                1, 0, 0, 0,
                                0, 0, 0, 1);
                            // x₁ = -z₀ - 1 ⇒ z₀ = -x₁ - 1
                            // y₁ = y₀
                            // z₁ = x₀

                            aabb.min.x = /*-aabb.min.x + 1*/-aabb.min.x + 1;
                            aabb.max.x = /*-(aabb.max.x - 1)*/-aabb.max.x + 1;
                            core::mem::swap(&mut aabb.min.x, &mut aabb.max.x);

                            /* aabb.min.z = /*-aabb.min.z + 1*/-aabb.min.z;
                            aabb.max.z = /*-(aabb.max.z - 1)*/-aabb.max.z;
                            core::mem::swap(&mut aabb.min.z, &mut aabb.max.z); */

                            core::mem::swap(&mut aabb.min.x, &mut aabb.min.z);
                            core::mem::swap(&mut aabb.max.x, &mut aabb.max.z);
                            core::mem::swap(&mut a_, &mut c_);
                        }

                        let a: f32 = a_ as f32/* - center.x as f32 */- 0.5;
                        let b: f32 = b_ as f32/* - center.y as f32 */- 0.5;
                        let c: f32 = c_ as f32/* - center.z as f32 */- 0.5;
                        // NOTE: Guaranteed positive since a,b,c are all positive (due to AABB being
                        // enforced valid at construction time) and degree must be ≥ 1.0.
                        let a_inv_pow = a.recip().powf(degree);
                        let b_inv_pow = b.recip().powf(degree);
                        let c_inv_pow = c.recip().powf(degree);
                        let abc_inv_pow = Vec3::new(a_inv_pow, b_inv_pow, c_inv_pow);

                        let abc_pow_z = /*1.0*/c.powf(degree);
                        let abc_pow_x_ = /*1.0*/a.powf(degree);
                        let abc_pow_x = /*1.0*/abc_pow_x_ / abc_pow_z;
                        let abc_slope_z = /*1.0*/b / c;
                        let abc_inv_pow_x = -abc_inv_pow.x * abc_pow_z;
                        let abc_inv_pow_y = -abc_inv_pow.y * abc_pow_z;
                        // let abc_inv_pow_z = abc_inv_pow.z;
                        let degree_recip = degree.recip();
                        /* let mut aabb = Aabb {
                            min: aabb.min.with_z(0),
                            max: aabb.max.with_z(1),
                        }; */
                        /*return {*/
                            let mut aabb = aabb;
                            let offset = Vec3::<i32>::from(mat.cols[3]);
                            let inv_mat = mat.as_::<f32>().inverted_affine_transform_no_scale/*transposed*/().as_::<i32>();
                            let dz = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_z())*/mat.cols[2]);
                            let dy = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_y())*/mat.cols[1]);
                            let dx = Vec3::from(/*mat * Vec4::from_direction(Vec3::<i32>::unit_x())*/mat.cols[0]);
                            let mut mask: Aabb<i32> = Aabb {
                                min: Vec3::from(inv_mat * Vec4::from_point(mask.min))/*.with_z(aabb.min.z)*/,
                                max: Vec3::from(inv_mat * Vec4::from_point(mask.max - 1))/*.with_z(aabb.max.z)*/,
                            };
                            mask.make_valid();
                            mask.max += 1;
                            aabb.intersect(mask);
                            /* aabb.max.x -= 1;
                            aabb.max.y -= 1; */
                            if !aabb.is_valid() {
                                return;
                            }
                            if PRINT_MESSAGES {
                                println!("\n\nVol: {:?}", aabb);
                            }

                            let (x_bounds_min, x_bounds_max, x_bounds) =
                                make_bounds(aabb.min.x, aabb.max.x);

                            // NOTE: By construction, the valid range for indexing into this is
                            // (0..bounds.to - bounds.from).
                            let xs = &*arena.alloc_slice_fill_iter(/*(0..x_bounds)*/x_bounds.clone().map(|x| {
                                (x as f32).powf(degree).mul_add(abc_inv_pow_x, abc_pow_z)
                            }));
                            /* let xs = arena.alloc_slice_fill_iter((0..x_bounds).map(|x| {
                                (x as f32).powf(degree) * a_inv_pow
                            }));
                            // NOTE: Positive because aabb.max.x is positive, and we take the max
                            // of it and aabb.min.x.
                            //
                            // FIXME: Statically assert u32 fits in usize.
                            let xs = &xs[..x_bounds as usize];

                            // NOTE: We have to handle the annoying cases where both min and max
                            // are on one side of the center (after masking).  For now, we just
                            // explicitly check their signs and handle this case separately, but
                            // there's probably a more elegant way.
                            //
                            // NOTE: In all code below, we can assume aabb.min.z ≤ aabb.max.z from
                            // the call to is_valid.
                            //
                            // FIXME: To prove casts of positive i32 to usize correct,
                            // statically assert that u32 fits in usize.
                            let (z_bounds_min, z_bounds_max) =
                                if aabb.min.z >= 0 {
                                    // NOTE: Since aabb.max.z ≥ aabb.min.z > 0, we know aabb.max.z
                                    // is positive.
                                    (
                                        aabb.min.z as usize..aabb.min.z as usize,
                                        aabb.min.z as usize..aabb.max.z as usize,
                                    )
                                } else if aabb.max.z < 0 {
                                    // NOTE: -aabb.max.z is positive because aabb.max.z is
                                    // negative.  Therefore, 1 - aabb.max.z is also positive.
                                    // 1 - aabb.min.z ≥ 1 - aabb.max.z because
                                    // aabb.max.z ≥ aabb.min.z, so we can assume this forms a valid
                                    // range.
                                    (
                                        (1 - aabb.max.z) as usize..(1 - aabb.min.z) as usize,
                                        (1 - aabb.max.z) as usize..(1 - aabb.max.z) as usize,
                                    )
                                } else {
                                    // NOTE: 1 < 1 - aabb.min.z since aabb.min.z < 0.
                                    // aabb.max.z is already verified to be non-negative,
                                    // so it's ≥ 0.  So we can assume all ranges are valid.
                                    (
                                        1..(1 - aabb.min.z) as usize,
                                        0..aabb.max.z as usize,
                                    )
                                };

                            // NOTE: Since min is non-increasing and max is non-decreasing,
                            // and both sides form a valid range from start..end, max-ing
                            // the starts and min-ing the ends must still form a valid range.
                            let z_bounds =
                                z_bounds_min.start.min(z_bounds_max.start)..
                                z_bounds_min.end.max(z_bounds_max.end)
                            ;
                            // NOTE: By construction, the valid range for indexing into zs is
                            // (0..z_bounds.to - z_bounds.from).
                            let zs = arena.alloc_slice_fill_iter(z_bounds.clone().map(|z| {
                                (z as f32).powf(degree) * c_inv_pow
                            }));
                            // NOTE:
                            //
                            // By our earlier argument and validity of the ranges, for each
                            // z_component (z_bounds_min or z_bounds_max) we have:
                            //
                            // z_bounds.start ≤ z_component.start ≤ z_component.end ≤ z_bounds.end
                            //
                            // so subtracting 0 from each side, we have:
                            //
                            // 0 ≤ z_component.start - z_bounds.start
                            //   ≤ z_component.end - z_bounds.start
                            //   ≤ z_bounds.end - z_bounds.start
                            //
                            // so, since the inner inequality forms a valid range, and the range is
                            // fully contained within the valid indexing range for zs, doing this
                            // for each component must also form a valid indexing range for zs.
                            let zs_min = &zs[z_bounds_min.start - z_bounds.start..z_bounds_min.end - z_bounds.start];
                            let zs_max = &zs[z_bounds_max.start - z_bounds.start..z_bounds_max.end - z_bounds.start];

                            let offset = offset + aabb.min.z * dz;
                            /*return */{
                                let mut ydiv = 0.0f32;
                                let mut yoff = Vec3::zero();
                                let dy_ = b.recip();

                                let y_bounds = (1 - aabb.min.y).max(aabb.max.y)/* + 1*/;
                                (0..y_bounds).for_each(|y| {
                                    let divergence = 1.0 - ydiv.powf(degree);
                                    /* let x_bounds_ = ((divergence.powf(degree_recip) * a) as i32 + 1);
                                    let x_bounds = x_bounds_.min(x_bounds); */
                                    let mut do_y = |offset: Vec3<i32>| {
                                        let mut xoff = Vec3::zero();
                                        (0..x_bounds).zip(xs).for_each(|(x, &xdiv)| {
                                            let divergence = divergence - xdiv;

                                            let mut do_x = |offset: Vec3<i32>| {
                                                let mut pos = offset;
                                                let mut do_z = |&zdiv| {
                                                    let divergence = divergence - zdiv;
                                                    if divergence > 0.0 {
                                                        hit(pos);
                                                    }
                                                    pos += dz;
                                                };
                                                zs_min.iter().rev().for_each(&mut do_z);
                                                zs_max.iter().for_each(&mut do_z);
                                            };

                                            if divergence > 0.0 {
                                                if aabb.min.x <= -x {
                                                    do_x(offset - xoff);
                                                }
                                                if x < aabb.max.x {
                                                    do_x(offset + xoff);
                                                }
                                            }

                                            xoff += dx;
                                        });
                                    };

                                    if aabb.min.y <= -y {
                                        do_y(offset - yoff);
                                    }
                                    if y < aabb.max.y {
                                        do_y(offset + yoff);
                                    }

                                    ydiv += /*b_inv_pow*/dy_;
                                    yoff += dy;
                                });
                            };
                            return; */

                            /* let y_bounds = (1 - aabb.min.y).max(aabb.max.y)/* + 1*/; */
                            let (y_bounds_min, y_bounds_max, y_bounds) =
                                make_bounds(aabb.min.y, aabb.max.y);

                            // NOTE: Casts are correct because bounds were originally cast from i32.
                            let x_bounds_start = x_bounds.start as i32;
                            let x_bounds_end = x_bounds.end as i32;

                            /*return */{
                                /*(0../*b_*/y_bounds)*/y_bounds.for_each(|y| {
                                    // NOTE: Cast is correct because y was originally cast from
                                    // i32.
                                    let offset0 = offset - dy * y as i32;
                                    let offset1 = offset + dy * y as i32;
                                    // let divergence = (y as f32).powf(degree).mul_add(abc_inv_pow_y, /*1.0*/abc_pow_z)/*.max(0.0)*/;
                                    let divergence = (y as f32).powf(degree) * abc_inv_pow_y;
                                    /* if 0.0 < divergence */{
                                        let x_bounds_ = ((divergence.mul_add(abc_pow_x, abc_pow_x_)).powf(degree_recip) as i32 + 1);
                                        // let x_bounds_ = ((divergence * abc_pow_x).powf(degree_recip) as i32 + 1);
                                        /* let x_bounds_ =
                                            (divergence.powf(degree_recip).mul_add(abc_slope_z, 1.0)/* * abc_slope_z*/) as i32 + 1; */
                                        let x_bounds = x_bounds_start..x_bounds_end.min(x_bounds_);/*x_bounds_.min(x_bounds);*/

                                        /*(0../*a_ + 1*/x_bounds)*/x_bounds.zip(xs).for_each(|(x, &xdiv)| {
                                            let divergence = xdiv + divergence;
                                            // let divergence = (x as f32).powf(degree).mul_add(abc_inv_pow_x, divergence);
                                            /* if 0.0 < divergence */{
                                                let z = divergence.powf(degree_recip);/* as i32*/;

                                                let z0 = /*(-z + 1)*/((-z/* + 1.0*/) as i32).max(aabb.min.z);
                                                let z1 = ((z/* - 1.0*/ + 1.0) as i32).min(aabb.max.z/* - 1*/);
                                                let mut do_x = |offset: Vec3<i32>| {
                                                    (z0..z1).for_each(|z| {
                                                        let pos = offset + dz * z;
                                                        /* hit(Vec3::new(pos.x, pos.y, z)); */
                                                        hit(pos);
                                                    });
                                                };
                                                let mut do_y = |offset: Vec3<i32>| {
                                                    /* if aabb.min.x <= -x {
                                                        do_x(offset - dx * x);
                                                    }
                                                    /* do_x(offset + dx * (-x).max(aabb.min.x)); */
                                                    if x < aabb.max.x {
                                                        do_x(offset + dx * x);
                                                    } */
                                                    // NOTE: Casts are correct because x is
                                                    // positive and u32 should fit in a usize
                                                    // (FIXME: assert this statically).
                                                    if /*aabb.min.x <= -x*/x_bounds_min.contains(&(x as usize)) {
                                                        do_x(offset - dx * x);
                                                    }
                                                    if /*x < aabb.max.x*/x_bounds_max.contains(&(x as usize)) {
                                                        do_x(offset + dx * x);
                                                    }
                                                    /* do_x(offset + dx * (x/* + 1*/).min(aabb.max.x)); */
                                                };

                                                if /*aabb.min.y <= -y*/y_bounds_min.contains(&y) {
                                                    do_y(offset0);
                                                }
                                                if /*y < aabb.max.y*/y_bounds_max.contains(&y) {
                                                    do_y(offset1);
                                                }
                                            }
                                        });
                                    }
                                });
                            };
                            return;
                        /*}*/
                    };
                    if degree == 2.0 {
                        // Reverse translation from center.
                        let center = aabb.center()/*.map(|e| e as f32)*/;
                        mat = mat * Mat4::<i32>::translation_3d(center);
                        let mut aabb = Aabb {
                            // TODO: Try to avoid generating 3/4 of the circle; the interaction with
                            // masking will require care.
                            min: aabb.min - center,
                            max: aabb.max - center,
                        };
                        let mut a_ = aabb.max.x;
                        let mut b_ = aabb.max.y;
                        let mut c_ = aabb.max.z;
                        let a: f32 = a_ as f32/* - center.x as f32 */- 0.5;
                        let b: f32 = b_ as f32/* - center.y as f32 */- 0.5;
                        let c: f32 = c_ as f32/* - center.z as f32 */- 0.5;
                        // NOTE: Guaranteed positive since a,b,c are all positive (due to AABB being
                        // enforced valid at construction time) and degree must be ≥ 1.0.
                        let a_inv_pow = a.recip().powf(degree);
                        let b_inv_pow = b.recip().powf(degree);
                        let c_inv_pow = c.recip().powf(degree);
                        let abc_inv_pow = Vec3::new(a_inv_pow, b_inv_pow, c_inv_pow);
                        return aabb_iter(
                            aabb,
                            mat,
                            mask,
                            |pos| {
                                let rpos = pos.as_::<f32>()/* - center*/;
                                (rpos * rpos).dot(abc_inv_pow) < 1.0
                            },
                            hit,
                            /* &mut |pos| {
                                // Project out the hit eight ways (due to symmetry).
                                // NOTE: Technically, does not respect masking properly, but
                                // hopefully this doesn't impact anything.
                                hit(Vec3::new(-pos.x, -pos.y, -pos.z));
                                hit(Vec3::new(pos.x, -pos.y, -pos.z));
                                hit(Vec3::new(-pos.x, pos.y, -pos.z));
                                hit(Vec3::new(pos.x, pos.y, -pos.z));
                                hit(Vec3::new(-pos.x, -pos.y, pos.z));
                                hit(Vec3::new(pos.x, -pos.y, pos.z));
                                hit(Vec3::new(-pos.x, pos.y, pos.z));
                                hit(Vec3::new(pos.x, pos.y, pos.z));
                            }, */
                        )
                    } else {
                        return do_superquadric();
                    /* return aabb_iter(
                        *aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                        hit,
                    )*/
                    }
                },
                Node::Plane(..)
                /*| Node::Segment { .. } */=> {
                    // TODO: Optimize further?
                    let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos),
                        hit,
                    )
                },
                | &Node::Segment { segment, /*radius*/r0, r1 ,z_scale } => {
                    /* let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(aabb, mat, mask, |_| true, &mut *hit); */
                    // TODO: Optimize further?
                    let aabb = Self::get_bounds(cache, tree, prim);
                    /*if !(aabb.size().w > 8 || aabb.size().h > 8 || aabb.size().d > 16) */{
                        return aabb_iter(
                            aabb.as_(),
                            mat,
                            mask,
                            // |_| true,
                            |pos| Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos),
                            hit,
                        );
                        return
                    }
                    // NOTE: vol(frustum) = 1/3 h(A(r₀) + A(r₁) + √(A(r₀)A(r₁)))
                    // = 1/3 h (r₀² + r₁² + √(r₀²r₁²))
                    // = 1/3 h (r₀² + r₁² + r₀r₁)
                    let y_size = (segment.end - segment.start).magnitude();
                    let r0 = r0 - 0.5;
                    let r1 = r1 - 0.5;
                    let radius_squared = (r0 * r0 + r1 * r1 + r0 * r1) / 3.0;
                    let radius = radius_squared.sqrt();
                    let mat = mat.as_::<f32>();
                    // Orient the matrix to point towards the segment.
                    //
                    // NOTE: We try to make sure the up vector is aligned to Z, unless we are
                    // orthogonal to X in which case we align with X.  Since we are drawing a
                    // prism, there's no real difference between X, Y, or Z here, as long as it's
                    // an orthogonal direction.  Note that we already know that not all three
                    // segments differ.
                    //
                    // TODO: Avoid this branch by returning a different primitive if we're a
                    // straight line.
                    let up = if segment.start.x == segment.end.x {
                        Vec3::unit_x()
                    } else {
                        Vec3::unit_z()
                    };
                    let rotate_mat = Mat4::<f32>::model_look_at_lh(segment.start, segment.end, up);
                    // NOTE: Pre-inverted.
                    let translate_mat = Mat4::<f32>::new(
                        1.0, 0.0, 0.0, -radius,
                        0.0, 1.0, 0.0, -radius,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0,
                    );
                    // Inverted, so multiply backwards
                    let rotate_translate_mat = rotate_mat * translate_mat;
                    let mat = mat * rotate_translate_mat;
                    // Fake AABB to draw.
                    let mut aabb = Aabb {
                        min: Vec3::zero(),
                        max: Vec3::new(2.0 * radius/* + 1.0*/, 2.0 * radius/* + 1.0*/, y_size/* + 1.0*/),
                    };
                    // TODO: Optimize
                    // Need to assume a full affine transform to get proper inversion here.
                    //
                    // (Will this restriction even *work* with shearing?  I have no idea).
                    let inv_mat = mat.inverted_affine_transform_no_scale/*transposed*/();
                    // TODO: Optimize.
                    let mut mask: Aabb<f32> = Aabb {
                        min: Vec3::from(inv_mat * Vec4::from_point(mask.min.as_()))/*.with_z(aabb.min.z)*/,
                        max: Vec3::from(inv_mat * Vec4::from_point((mask.max - 1).as_()))/*.with_z(aabb.max.z)*/,
                    };
                    mask.make_valid();
                    mask.max += 1.0;
                    mask.min = mask.min.map(f32::floor);
                    mask.max = mask.max.map(f32::ceil);
                    aabb.intersect(mask);
                    /* if !aabb.is_valid() {
                        return;
                    } */

                    let dz = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_z())*/mat.cols[2]);
                    let dy = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_y())*/mat.cols[1]);
                    let dx = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_x())*/mat.cols[0]);
                    // We use the matrix to determine which offset/ori to use for each iteration.
                    // Specifically, mat.x will represent the real x axis, mat.y the real y axis,
                    // and mat.z the real z axis.
                    //
                    // Because of this, to find the axes we actually want to use for our own purposes, we
                    // need to first read the offset, then transpose, then read the three axes.
                    let offset = Vec3::<f32>::from(/*inv_mat * Vec4::<f32>::unit_w()*/mat.cols[3]) +
                        dx * aabb.min.x + dy * aabb.min.y + dz * aabb.min.z;

                    /* (segment.start.x as i32 & !0b11111..(segment.end.x as i32 - 1) & !0b11111 + 1)
                        .step_by(32)
                        .for_each(|chunk_x_| {
                            (segment.start.y as i32 & !0b11111..(segment.end.y as i32 - 1) & !0b11111 + 1)
                                .step_by(32)
                                .for_each(|chunk_y_| {
                                    (segment.start.z as i32 & !0b1111..(segment.end.z as i32 - 1) & !0b1111 + 1)
                                        .step_by(16)
                                        .for_each(|chunk_z_| {
                                            let chunk_x = chunk_x_ as f32;
                                            let chunk_y = chunk_y_ as f32;
                                            let chunk_z = chunk_z_ as f32;
                                            let dx_ = 0b11111;
                                            let dy_ = 0b11111;
                                            let dz_ = 0b1111;
                                            let dx = dx_ as f32;
                                            let dy = dy_ as f32;
                                            let dz = dz_ as f32;
                                            let a000 = Vec3::from(inv_mat * Vec4::new(chunk_x, chunk_y, chunk_z, 1.0));
                                            let a001 = Vec3::from(inv_mat * Vec4::new(chunk_x + dx, chunk_y, chunk_z, 1.0));
                                            let a010 = Vec3::from(inv_mat * Vec4::new(chunk_x, chunk_y + dy, chunk_z, 1.0));
                                            let a011 = Vec3::from(inv_mat * Vec4::new(chunk_x + dx, chunk_y + dy, chunk_z, 1.0));
                                            let a100 = Vec3::from(inv_mat * Vec4::new(chunk_x, chunk_y, chunk_z + dz, 1.0));
                                            let a101 = Vec3::from(inv_mat * Vec4::new(chunk_x + dx, chunk_y, chunk_z + dz, 1.0));
                                            let a110 = Vec3::from(inv_mat * Vec4::new(chunk_x, chunk_y + dy, chunk_z + dz, 1.0));
                                            let a111 = Vec3::from(inv_mat * Vec4::new(chunk_x + dx, chunk_y + dy, chunk_z + dz, 1.0));

                                            // Very rough bounding box.
                                            let aabb_intersect = Aabb {
                                                min: a000,
                                                max: a000,
                                            }
                                            .expanded_to_contain_point(a001)
                                            .expanded_to_contain_point(a010)
                                            .expanded_to_contain_point(a011)
                                            .expanded_to_contain_point(a100)
                                            .expanded_to_contain_point(a101)
                                            .expanded_to_contain_point(a110)
                                            .expanded_to_contain_point(a111);

                                            /* // Clip the points to the bounding box.
                                            let a000 = a000.clamped(aabb);
                                            let a001 = a001.clamped(aabb);
                                            let a010 = a010.clamped(aabb);
                                            let a011 = a011.clamped(aabb);
                                            let a100 = a100.clamped(aabb);
                                            let a101 = a101.clamped(aabb);
                                            let a110 = a110.clamped(aabb);
                                            let a111 = a111.clamped(aabb);

                                            // If there's really an intersection, at least one
                                            // diagonal should be contained.
                                            let line0 = Aabb {
                                                min: a000,
                                                max: a001,
                                            }.made_valid();
                                            let line1 = Aabb {
                                                min: a000,
                                                max: a010,
                                            }.made_valid();
                                            let line2 = Aabb {
                                                min: a000,
                                                max: a001,
                                            }.made_valid();
                                            let max = Vec4::from_point(aabb.max); */
                                            if /* line0.collides_with_aabb(aabb) ||
                                                line1.collides_with_aabb(aabb) ||
                                                line2.collides_with_aabb(aabb) */
                                                aabb_intersect.collides_with_aabb(aabb)
                                                /* a000.are_all_positive() && a000.partial_cmple_simd(max).reduce_and() ||
                                                a001.are_all_positive() && a001.partial_cmple_simd(max).reduce_and() ||
                                                a010.are_all_positive() && a010.partial_cmple_simd(max).reduce_and() ||
                                                /* a011.are_all_positive() && a011.partial_cmple_simd(max).reduce_and() || */
                                                a100.are_all_positive() && a100.partial_cmple_simd(max).reduce_and()/* ||
                                                a101.are_all_positive() && a101.partial_cmple_simd(max).reduce_and() ||
                                                a110.are_all_positive() && a110.partial_cmple_simd(max).reduce_and() ||
                                                a111.are_all_positive() && a111.partial_cmple_simd(max).reduce_and() */*/{
                                                let mut offset = Vec3::new(chunk_x_, chunk_y_, chunk_z_);
                                                (0..dz_).for_each(|z| {
                                                    offset.z += 1;
                                                    let mut offset = offset;
                                                    (0..dy_).for_each(|y| {
                                                        offset.y += 1;
                                                        (0..dx_)
                                                            /* .inspect(|&x| if PRINT_MESSAGES { println!("\nPos: {:?}", Vec3::new(x, y, z)) })
                                                            .filter(|&x| test(Vec3::new(x, y, z)))
                                                            .inspect(|_| if PRINT_MESSAGES { hit_count += 1 }) */
                                                            .map(|x| offset.with_x(x))
                                                            .for_each(&mut *hit)
                                                    })
                                                });
                                            }
                                        });
                                });
                        });
                    return; */

                    if PRINT_MESSAGES {
                        println!("Mask: {:?}\nAabb: {:?}\nTranslate: {:?}\nLookat: {:?}\nMat: {:?}\nInv: {:?}\n", mask, aabb, translate_mat, rotate_mat, mat, inv_mat);
                    }
                    /* aabb.intersect(mask);
                    println!("\n\nVol: {:?}", aabb); */
                    let mut hit_count = 0.0;
                    // TODO: Optimize.
                    // It *might* actually make sense to make a const generic so we specialize the six
                    // possible orientations differently, and avoid vector arithmetic, but for now we
                    // don't do that.
                    /* aabb.make_valid(); */
                    // TODO: Optimize a lot!

                    // Whether to optimize the layering for input or output should likely depend on
                    // the primitive type, but for the most part we assume that sampling output in
                    // layered order is more important, since most of our sampling primitives are
                    // not memory-bound (we currently do zyx iteration, but maybe other orders are
                    // better, who knows?).
                    //
                    // Another potential optimization when we are *actually* operating over primitives and
                    // convex solids: assuming the primitives are convex, we can start by sampling the
                    // corner in the next plane that is within the bounding box and touching the corner of
                    // the last AABB.  If it's empty, we can skip anything further away, thanks to
                    // convexity.  Moreover, as we move towards the center of the plane, we can skip
                    // testing any remaining components on the two lines in the corner (or else we'd fail
                    // to be convex).
                    //
                    // We can even (somewhat!) extend this to difference of unions, since this is
                    // equivalent to intersection by negations.  Specifically, we focus on the case of
                    // disjoint union (in which case there are no intersecting components).  Then we know
                    // that *within* the bounding box of each union'd component, we'll eventually reach
                    // something convex, and we can find the intersection between the positive convex part
                    // and the negative convex part.  We can use these as bounds to carve out of the
                    // positive part, before reporting all the interior points.  The final set is still
                    // not convex, but we don't have to do a huge amount of work here.  As long as the
                    // union is disjoint, we should end up in a "sufficiently small" convex set without
                    // (?) exponential blowup.
                    //
                    // NOTE: These subtractions *can* go negative, but thanks to the fact that f32
                    // to u32 saturates, it will end up making the range empty as desired.
                    (0..(aabb.max.z - aabb.min.z) as u32).for_each(|z| {
                        let offset = offset + dz * z as f32;
                        (0..(aabb.max.y - aabb.min.y) as u32).for_each(|y| {
                           let offset = offset + dy * y as f32;
                           (0..(aabb.max.x - aabb.min.x) as u32)
                               .inspect(|&x| if PRINT_MESSAGES { println!("\nPos: {:?}", Vec3::new(x, y, z)) })
                               /* .filter(|&x| test(Vec3::new(x, y, z))) */
                               .inspect(|_| if PRINT_MESSAGES { hit_count += 1.0 })
                               .map(|x| (offset + dx * x as f32).as_())
                               .inspect(|&new| if PRINT_MESSAGES { println!("\nNew: {:?}", new) })
                               .for_each(&mut *hit)
                       })
                    });
                    if PRINT_MESSAGES {
                        println!("Hit rate: {:?} / {:?}", hit_count, aabb.size().product());
                    }
                    return;
                },
                | &Node::SegmentPrism { segment, radius, height } => {
                    /* let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                        hit,
                    ); */
                    // The key to optimizing drawing AABBs is to switch how we sample: rather than
                    // sampling the input space first, then transforming it efficiently to the
                    // output space, we sample things directly within the output space, then
                    // transform it to input coordinates once we know there's a hit.
                    let mat = mat.as_::<f32>();
                    // Orient the matrix to point towards the xy part of the segment.
                    let rotate_mat = Mat4::<f32>::model_look_at_lh(segment.start, segment.end.with_z(segment.start.z), Vec3::unit_z());
                    // Shear the matrix along the y axis, keeping the x and y coordinates fixed.
                    // We know the y length is nonzero because we checked on construction that
                    // the start and end of the xy component of the line segment were not the same.
                    //
                    // We also translate the matrix to the left at the same time, to make the
                    // bounding box nicer.  Since this is only along the x axis it doesn't affect
                    // the shear.
                    let y_size = (segment.end.xy() - segment.start.xy()).magnitude();
                    let slope = (segment.end.z - segment.start.z) / y_size;
                    let shear_mat = Mat4::<f32>::new(
                        1.0, 0.0, 0.0, radius - 0.5,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, slope, 1.0, /*radius - 0.5*/0.0,
                        0.0, 0.0, 0.0, 1.0,
                    ).inverted_affine_transform_no_scale();
                    // Inverted, so multiply backwards
                    let rotate_shear_mat = rotate_mat * shear_mat;
                    let mat = mat * rotate_shear_mat;
                    // Fake AABB to draw.
                    let mut aabb = Aabb {
                        min: Vec3::zero(),
                        max: Vec3::new(2.0 * radius, height + 1.0, y_size/* + 2.0 * radius*/ + 1.0),
                    };

                    // FIXME: See if this works!
                    aabb.max -= 1.0;
                    mask.max -= 1;

                    // 1. Compute the xy triangle coordinates in "screen space"
                    //    (assuming w=0).
                    // TODO: Switch to chunk-relative coordinates (?).
                    type Coord = i32;
                    let v0: Vec3<Coord> = (mat * Vec4::from_point(aabb.min)).xyz().as_();
                    let v1: Vec3<Coord> = (mat * Vec4::from_point(aabb.min.with_z(aabb.max.z))).xyz().as_();
                    let v2: Vec3<Coord> = (mat * Vec4::from_point(aabb.min.with_x(aabb.max.x))).xyz().as_();
                    let v3: Vec3<Coord> = (mat * Vec4::from_point(aabb.max).with_y(aabb.min.y)).xyz().as_();

                    // NOTE: 5+5+14+14+14 = 52 bits, which can safely fit in f64.  So technically
                    // speaking an f64 can precisely hold any multiple of these bits, but it'd be
                    // better to just stay within chunk bounds and fit in 24 bits.
                    //
                    // TODO: We actually only need 19+19 = 38 bits (really 40 due to sign), but we
                    // can shrink this as long as we know no site can take up more than 11 × 11
                    // chunks, or 10×10 signed (a fairly reasonable restriction); or if we want to
                    // include z, then no more than 4×4 chunks for a single piece of geometry
                    // (maybe 3×3 signed?).  We can then snap it to the nearest chunk.  However,
                    // to do this, we need to make sure we actually handle these cases properly
                    // It would be best to avoid an extra offset at the end by starting out by
                    // prefetching only the chunks we need into a grid.
                    let orient_2d = |a: Vec2<Coord>, b: Vec2<Coord>, c: Vec2<Coord>| {
                        (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
                    };

                    const LG_LANES: usize = 2/*4*/;
                    const LANES: usize = /*4*//*16*/1 << LG_LANES;
                    type FVec = core::simd::Simd::<f32, LANES>;
                    type IVec = core::simd::Simd::<i32, LANES>;

                    let col_offset = IVec::from([0, 1, 0, 1]);//0
                    let row_offset = IVec::from([0, 0, 1, 1]);//0
                    // let col_offset = IVec::from([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
                    // let row_offset = IVec::from([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);

                    let mut rasterize = |v0: Vec3<Coord>, v1: Vec3<Coord>, v2: Vec3<Coord>| {
                        // 2. Compute triangle bounding box.
                        // TODO: optimize
                        /* let min_x = v0.x.min(v1.x).min(v2.x);
                        let min_y = v0.y.min(v1.y).min(v2.y);
                        let max_x = v0.x.max(v1.x).max(v2.x);
                        let max_y = v0.y.max(v1.y).max(v2.y); */

                        // 3. Clip against screen bounds.
                        // TODO: Do this (against the mask).

                        // 4. Rasterize.
                        // TODO: Optimize (simd).

                        /* // 4a. Triangle setup
                        // FIXME: Chunk-relative coordinates to avoid overflow.
                        let a01 = v0.y - v1.y;
                        let b01 = v1.x - v0.x;
                        let a12 = v1.y - v2.y;
                        let b12 = v2.x - v1.x;
                        let a20 = v2.y - v0.y;
                        let b20 = v0.x - v2.x; */

                        let a0 = v1.y - v2.y;
                        let a1 = v2.y - v0.y;
                        let a2 = v0.y - v1.y;

                        let b0 = v2.x - v1.x;
                        let b1 = v0.x - v2.x;
                        let b2 = v1.x - v0.x;

                        let c0 = v1.x * v2.y - v2.x * v1.y;
                        let c1 = v2.x * v0.y - v0.x * v2.y;
                        let c2 = v0.x * v1.y - v1.x * v0.y;

                        let mut tri_area = b2 * a1;
                        tri_area -= b1 * a2;
                        let one_over_tri_area = (tri_area as f32).recip();

                        let z0 = v0.z as f32;
                        let z1 = (v1.z - v0.z) as f32 * one_over_tri_area;
                        let z2 = (v2.z - v0.z) as f32 * one_over_tri_area;
                        /* let a0 = v1.y - v2.y;
                        let b0 = v2.x - v1.x;
                        let c0 = v1.x * v2.y - v2.x * v1.y;

                        let mut tri_area = a0 * v0.x;
                        tri_area += b0 * v0.y;
                        tri_area += c0;

                        let one_over_tri_area = 1.0 / tri_area as f32; */

                        // 4b. Barycentric coordinates at minX/minY corner.
                        /* let p: Vec2<Coord> = Vec2::new(min_x, min_y); */
                        let start_x = v0.x.min(v1.x).min(v2.x).max(mask.min.x) & -2i32;
                        let end_x = (v0.x.max(v1.x).max(v2.x) + 1).min(mask.max.x);
                        let start_y = v0.y.min(v1.y).min(v2.y).max(mask.min.y) & -2i32;
                        let end_y = (v0.y.max(v1.y).max(v2.y) + 1).max(mask.max.y);

                        let zz0 = FVec::splat(z0);
                        let zz1 = FVec::splat(z1);
                        let zz2 = FVec::splat(z2);

                        let start_xx = start_x;
                        let end_xx = end_x;
                        let start_yy = start_y;
                        let end_yy = end_y;

                        let aa0 = IVec::splat(a0);
                        let aa1 = IVec::splat(a1);
                        let aa2 = IVec::splat(a2);

                        let bb0 = IVec::splat(b0);
                        let bb1 = IVec::splat(b1);
                        let bb2 = IVec::splat(b2);

                        let aa0_inc = aa0 << IVec::splat(1);
                        let aa1_inc = aa1 << IVec::splat(1);
                        let aa2_inc = aa2 << IVec::splat(1);

                        // TODO: direct index access with row_id

                        // TODO: fma when supported.
                        let col = col_offset + IVec::splat(start_xx);
                        let aa0_col = aa0 * col;
                        let aa1_col = aa1 * col;
                        let aa2_col = aa2 * col;

                        let row = row_offset + IVec::splat(start_yy);
                        let bb0_row = bb0 * row + IVec::splat(c0);
                        let bb1_row = bb1 * row + IVec::splat(c1);
                        let bb2_row = bb2 * row + IVec::splat(c2);

                        let mut sum0_row = aa0_col + bb0_row;
                        let mut sum1_row = aa1_col + bb1_row;
                        let mut sum2_row = aa2_col + bb2_row;

                        let bb0_inc = bb0 << IVec::splat(1);
                        let bb1_inc = bb1 << IVec::splat(1);
                        let bb2_inc = bb2 << IVec::splat(1);

                        let mut zx = aa1_inc.cast::<f32>() * zz1;
                        zx = aa2_inc.cast::<f32>().mul_add(zz2, zx);

                        (start_yy..end_yy).step_by(LG_LANES).for_each(|r| {
                            // let index = row_idx;
                            let mut alpha = sum0_row;
                            let mut beta = sum1_row;
                            let mut gamma = sum2_row;
                            let mut depth = zz0;
                            depth = beta.cast::<f32>().mul_add(zz1, depth);
                            depth = gamma.cast::<f32>().mul_add(zz2, depth);
                            (start_xx..end_xx).step_by(LG_LANES).for_each(|c| {
                                let mask = alpha | beta | gamma;

                                if PRINT_MESSAGES { println!("\nWpos: {:?}", Vec3::new(alpha, beta, gamma)) }
                                if PRINT_MESSAGES {
                                    depth.cast::<i32>().to_array().into_iter().for_each(|l| {
                                        println!("Pos: {:?}", Vec3::new(c, r, l));
                                    });
                                }

                                // TODO: Optimize
                                // let i = 0;
                                (0..LANES).for_each(|i| {
                                    if mask.is_positive().test(i) /*alpha >= 0 && beta >= 0 && gamma >= 0*/ {
                                        let z = depth.cast::<i32>()[i];
                                        /*z.to_array().into_iter().for_each(|z| {*/
                                            (z..z + height.round() as i32 + 1).for_each(|l| {
                                                // TODO: Scatter, etc.
                                                /* for l in (z + l).to_array() { */
                                                    hit(Vec3::new(c + col_offset[i], r + row_offset[i], l));
                                                /* } */
                                            });
                                        /*}*/
                                    }
                                });

                                // index += 4;
                                alpha += aa0_inc;
                                beta += aa1_inc;
                                gamma += aa2_inc;
                                depth = depth + zx;
                            });

                            // row_idx += 2 * SCREENW;
                            sum0_row += bb0_inc;
                            sum1_row += bb1_inc;
                            sum2_row += bb2_inc;
                        });

                        /* let mut w0_row = orient_2d(v1.xy(), v2.xy(), p);
                        let mut w1_row = orient_2d(v2.xy(), v0.xy(), p);
                        let mut w2_row = orient_2d(v0.xy(), v1.xy(), p);

                        let z_inv = ((a01 * v2.x + b01 * v2.y - (v0.x * v1.y - v0.y * v1.x)) as f32).recip();

                        let a01_z = (a01 * (v1.z - v0.z)) as f32 * z_inv;
                        let a20_z = (a20  * (v2.z - v0.z)) as f32 * z_inv;
                        let b20_z = (b20  * (v1.z - v0.z)) as f32 * z_inv;
                        let b01_z = (b01  * (v2.z - v0.z)) as f32 * z_inv;
                        let mut z_row = v0.z as f32 + w1_row as f32 * a01_z + w2_row as f32 * a20_z;

                        // 4c. Rasterize.
                        let mut hit_count = 0;
                        (min_y..max_y + 1).for_each(|y| {
                            let mut w0 = w0_row;
                            let mut w1 = w1_row;
                            let mut w2 = w2_row;
                            let mut z = z_row;
                            (min_x..max_x + 1).for_each(|x| {
                                // If p is on or inside all edges, render pixel
                                if PRINT_MESSAGES { println!("\nWpos: {:?}", Vec3::new(w0, w1, w2)) }
                                /* if PRINT_MESSAGES { println!("\nPos: {:?}", Vec2::new(x, y)) }
                                 * */
                                if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                                    // TODO: Interpolate z
                                    if PRINT_MESSAGES {
                                        hit_count += 1;
                                        println!("\nNew: {:?}", Vec3::new(x, y/*0*/, z as i32));
                                    }
                                    // TODO: Iterate over all z.
                                    hit(Vec3::new(x, y, z as i32).as_());
                                }

                                // One step to the right
                                w0 += a12;
                                w1 += a20;
                                w2 += a01;
                                z += a20_z + a01_z;
                            });

                            // One row step
                            w0_row += b12;
                            w1_row += b20;
                            w2_row += b01;
                            z_row += b20_z + b01_z;
                        });
                        if PRINT_MESSAGES {
                            println!("Hit rate: {:?} / {:?}", hit_count, aabb.size().product());
                        } */
                    };

                    // rasterize(v2, v3, v1);
                    /* if v0.x > v1.x */{
                        rasterize(v0, v1, v2);
                        rasterize(v3, v2, v1);
                    }/* else {
                        rasterize(v2, v1, v0);
                        rasterize(v1, v2, v3);
                    }*/
                    // rasterize(v2, v1, v3);

                    return;

                    /* // TODO: Optimize
                    // Need to assume a full affine transform to get proper inversion here.
                    //
                    // (Will this restriction even *work* with shearing?  I have no idea).
                    let inv_mat = mat.inverted_affine_transform_no_scale();
                    let mut dz = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_z())*/mat.cols[2]);
                    let dy = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_y())*/mat.cols[1]);
                    let mut dx = Vec3::<f32>::from(/*inv_mat * Vec4::from_direction(Vec3::<f32>::unit_x())*/mat.cols[0]);

                    /* if dz.x != 0.0 && dz.y != 0.0 {
                        let dz_min = dz.x / dz.y;
                        let dz_max = dz.y / dz.x;
                        let dz_ratio = dz_max.max(dz_min);
                        /* if dz_ratio >= 2.0 */{
                            dx /= dz_ratio;
                            dz /= dz_ratio;
                            aabb.max.x *= dz_ratio;
                            aabb.max.z *= dz_ratio;
                        }
                    } */

                    // We use the matrix to determine which offset/ori to use for each iteration.
                    // Specifically, mat.x will represent the real x axis, mat.y the real y axis,
                    // and mat.z the real z axis.
                    //
                    // Because of this, to find the axes we actually want to use for our own purposes, we
                    // need to first read the offset, then transpose, then read the three axes.
                    let offset = Vec3::<f32>::from(/*inv_mat * Vec4::<f32>::unit_w()*/mat.cols[3]);

                    // TODO: Optimize.
                    let mut mask: Aabb<f32> = Aabb {
                        min: Vec3::from(inv_mat * Vec4::from_point(mask.min.as_())).with_z(aabb.min.z),
                        max: Vec3::from(inv_mat * Vec4::from_point(mask.max.as_())).with_z(aabb.max.z),
                    };
                    mask.make_valid();
                    if PRINT_MESSAGES {
                        println!("Mask: {:?}\nAabb: {:?}\nShear: {:?}\nLookat: {:?}\nMat: {:?}\nInv: {:?}\n", mask, aabb, shear_mat, rotate_mat, mat, inv_mat);
                    }
                    /* aabb.intersect(mask);
                    println!("\n\nVol: {:?}", aabb); */
                    let mut hit_count = 0.0;
                    // TODO: Optimize.
                    // It *might* actually make sense to make a const generic so we specialize the six
                    // possible orientations differently, and avoid vector arithmetic, but for now we
                    // don't do that.
                    //
                    // TODO: Orient by *output* z first, then *output* y, then *output* x, instead of
                    // *input* (which is what we're currently doing because it's easy).
                    /* let mat_ = mat.transpose();
                    let dz = mat * Vec4::unit_z();
                    let dy = mat * Vec4 */
                    //
                    // Apply matrix transformations to find the new bounding box.
                    //
                    // NOTE: Matrix only supports floats for FP scaling, which is questionable.
                    /* aabb.iter_mut().for_each(|v| mat * v);
                    // It may have a bad orientation, so make sure it's valid (TODO: consider fixing this
                    // at rotation time instead?).
                    aabb.make_valid(); */
                    // TODO: Optimize a lot!

                    // Whether to optimize the layering for input or output should likely depend on
                    // the primitive type, but for the most part we assume that sampling output in
                    // layered order is more important, since most of our sampling primitives are
                    // not memory-bound (we currently do zyx iteration, but maybe other orders are
                    // better, who knows?).
                    //
                    // Another potential optimization when we are *actually* operating over primitives and
                    // convex solids: assuming the primitives are convex, we can start by sampling the
                    // corner in the next plane that is within the bounding box and touching the corner of
                    // the last AABB.  If it's empty, we can skip anything further away, thanks to
                    // convexity.  Moreover, as we move towards the center of the plane, we can skip
                    // testing any remaining components on the two lines in the corner (or else we'd fail
                    // to be convex).
                    //
                    // We can even (somewhat!) extend this to difference of unions, since this is
                    // equivalent to intersection by negations.  Specifically, we focus on the case of
                    // disjoint union (in which case there are no intersecting components).  Then we know
                    // that *within* the bounding box of each union'd component, we'll eventually reach
                    // something convex, and we can find the intersection between the positive convex part
                    // and the negative convex part.  We can use these as bounds to carve out of the
                    // positive part, before reporting all the interior points.  The final set is still
                    // not convex, but we don't have to do a huge amount of work here.  As long as the
                    // union is disjoint, we should end up in a "sufficiently small" convex set without
                    // (?) exponential blowup.
                    (0..aabb.max.z as u32).for_each(|z| {
                        let mut offset = offset + dz * z as f32;
                        // offset = offset + dz/* * z as f32*/;
                        // let mut offset = offset;
                        (0..aabb.max.y as u32).for_each(|y| {
                            let offset = offset + dy * y as f32;
                            /* offset = offset + dy;
                            let mut offset = offset; */
                            (0..aabb.max.x as u32)
                            // Some(0.0).into_iter().chain(Some(aabb.max.x))
                                .inspect(|&x| if PRINT_MESSAGES { println!("\nPos: {:?}", Vec3::new(x as u32, y/*0*/, z)) })
                                /* .filter(|&x| test(Vec3::new(x, y, z))) */
                                .inspect(|_| if PRINT_MESSAGES { hit_count += 1.0 })
                                .map(|x| {
                                    /* offset = (offset + dx/* * x*/).floor()/* as f32*/; offset.as_() */
                                    (offset + dx * x as f32).as_()
                                })
                                .inspect(|&new| if PRINT_MESSAGES { println!("\nNew: {:?}", new) })
                                .for_each(&mut *hit)
                        })
                    });
                    // Because shearing can (and often does) lead to dz being greater than 1, it
                    // is often the case that we will have missed some of it while iterating.  To
                    // compensate, we find the remainder and iterate more slowly over that.
                    let remainder_z = aabb.max.z.fract();
                    if remainder_z > 0.0 {
                        let z = aabb.max.z as u32;
                        let mut offset = offset + dz * (aabb.max.z - 1.0);/*offset + dz * aabb.max.z.truncate();*/
                        // offset = offset + dz * (aabb.max.z - 1.0)/* * z as f32*/;
                        (0..aabb.max.y as u32).for_each(|y| {
                            let offset = offset + dy * y as f32;
                            // offset = offset + dy/* * y as f32*/;
                            let mut offset = offset;
                            (0..aabb.max.x as u32)
                            // Some(0.0).into_iter().chain(Some(aabb.max.x))
                                .inspect(|&x| if PRINT_MESSAGES { println!("\nPos: {:?}", Vec3::new(x as u32, y/*0*/, z)) })
                                /* .filter(|&x| test(Vec3::new(x, y, z))) */
                                .inspect(|_| if PRINT_MESSAGES { hit_count += /*1*/remainder_z })
                                .map(|x| {
                                    (offset + dx * x as f32).as_()
                                    /*offset = (offset + dx/* * x*//* as f32*/).floor(); offset.as_()*/
                                })
                                .inspect(|&new| if PRINT_MESSAGES { println!("\nNew: {:?}", new) })
                                .for_each(&mut *hit)
                        })
                    }
                    /* let dz_mag = dz.magnitude();
                    if dz_mag > 1.0 {
                        // Because shearing preserves the volume of the unsheared AABB, it is
                        // possible (and likely) for us to miss voxels if we only iterate the same
                        // number of voxels on the Z axis as the unsheared volume.  We compensate
                        // for such cases by iterating more times.
                        dz = dz / dz_mag;
                        aabb.max.z = dz_mag;
                    } */
                    if PRINT_MESSAGES {
                        println!("Hit rate: {:?} / {:?}", hit_count, aabb.size().product());
                    }
                    return; */
                },
                /* Node::Sampling(/*a, f*/..) => {
                    return;
                    // TODO: Optimize further--we should be able to push evaluation of the
                    // function down the tree, which could get us tighter bounds.
                    let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                        hit,
                    )
                }, */
                // TODO: Maybe combine sampling and filtering to benefit from optimizations
                // here?  On the other hand, Prefab is supposedly going away...
                Node::Prefab(p) => {
                    // return;
                    // TODO: Optimize further--we should be able to push evaluation of the
                    // function down the tree, which could get us tighter bounds.
                    let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| !matches!(p.get(pos), Err(_) | Ok(StructureBlock::None)),
                        hit,
                    )
                },
                // Intersections mean we can no longer assume we're always in the bounding
                // volume.
                //
                // TODO: Optimize (though it's less worth it outside the top level).
                Node::Intersect([xs @ .., x]) => {
                    handle_intersect(stack, mat, mask, x, xs);
                    prim = *x;
                    /* // Self::get_bounds_disjoint_inner(arena, cache, tree, a, stack, mat, mask, &mut *hit);
                    let aabb = Self::get_bounds(cache, tree, prim);
                    /* let aabb = get_bounds(cache, tree, prim); */
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                        hit,
                    ); */
                },
                Node::IntersectAll([xs @ .., x]) => {
                    handle_intersect(stack, mat, mask, x, xs);
                    prim = *x;
                    /* let aabb = Self::get_bounds(cache, tree, prim);
                    /* let aabb = get_bounds(cache, tree, prim); */
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(cache, tree, prim_, pos),
                        hit,
                    ); */
                },

                // Unions can assume we're still in the same bounding volume.
                //
                // TODO: Don't recurse, use a local stack.
                Node::Union(a, b) => {
                    let stack_top = stack.len();
                    Self::get_bounds_disjoint_inner(arena, cache, tree, *a, stack_bot, stack, mat, mask, &mut *hit);
                    // NOTE: We know the stack will not be smaller than this because we never
                    // truncate it further than the top of the stack would get.
                    stack.truncate(stack_top);
                    prim = *b;
                },
                Node::UnionAll(prims) => {
                    let stack_top = stack.len();
                    return prims.into_iter().for_each(move |prim| {
                        Self::get_bounds_disjoint_inner(arena, cache, tree, *prim, stack_bot, stack, mat, mask, &mut *hit);
                        // NOTE: We know the stack will not be smaller than this because we never
                        // truncate it further than the top of the stack would get.
                        stack.truncate(stack_top);
                    });
                },
                /* // Since Repeat is a union, we also treat it the way we treat unions.
                //
                // TODO: Avoid recursing, instead keep track of the current count (maybe
                // in a local stack).
                Node::Repeat(x, offset, count) => {
                    if count == 0 {
                        return;
                    } else {
                        let count = count - 1;
                        let aabb = Self::get_bounds(cache, tree, x);
                        let aabb_corner = {
                            let min_red = aabb.min.map2(offset, |a, b| if b < 0 { 0 } else { a });
                            let max_red = aabb.max.map2(offset, |a, b| if b < 0 { a } else { 0 });
                            min_red + max_red
                        };
                        let min_func = |vec: Vec3<i32>| {
                            let diff = vec - aabb_corner;
                            let min = diff
                                .map2(offset, |a, b| if b == 0 { i32::MAX } else { a / b })
                                .reduce_min()
                                .clamp(0, count as i32);
                            vec - offset * min
                        };
                        aabb = Aabb {
                            min: min_func(aabb.min),
                            max: min_func(aabb.max),
                        };
                        return Self::get_bounds_disjoint(cache, tree, x, hit);
                        // Self::contains_at::<SubAabr>(cache, tree, x, pos)
                    }
                }, */

                // A difference is a weird kind of intersection, so we can't preserve being in the
                // bounding volume for b, but can for a.
                //
                // TODO: Don't recurse, push down the difference instead.
                Node::Without(/*a, b, a_first*/..) => {
                    /* if *a_first {
                        return Self::get_bounds_disjoint(
                            cache,
                            tree,
                            a,
                            |pos| !Self::contains_at::<SubAabr>(cache, tree, b, pos),
                        );
                    } else {
                        return Self::contains_at(
                            cache,
                            tree,
                            b,
                            |pos| Self::get_bounds_disjoint
                        );
                        !Self::contains_at::<SubAabr>(cache, tree, *b, pos) &&
                        Self::contains_at::<Check>(cache, tree, *a, pos)
                    } */
                    return;
                    let aabb = Self::get_bounds(cache, tree, prim);
                    return aabb_iter(
                        aabb,
                        mat,
                        mask,
                        |pos| Self::contains_at::<TopAabr>(/*cache, */tree, prim_, pos),
                        hit,
                    )
                },
                // Transformation operations get pushed down, in the following sense:
                //
                // * Existing translation operations are just inverted, so we end up
                //   with the same AABB that we would have had with no translation.
                // * The vertices that we report to the user must be forwards transformed,
                //   however!  This is because it would be really hard otherwise to figure
                //   out the original point that triggered the draw (from the user's
                //   perspective), which is needed to actually sample.
                // * Rotations, translations, etc. do not actually affect anything on the top
                //   level, because density / cost estimates (for disjoint union) don't matter
                //   at the top level; everything commutes.  Therefore, there's no reason to
                //   apply them until we are forced (by an eliminator on negative types, e.g.
                //   case on negative or (⅋) or project on negative and (&)...
                // * Below the top level, it is still completely feasible to avoid handling
                //   them if the polarity changes back to positive, but we won't try to deal
                //   with that here.
                /* Node::Rotate(prim, mat) => {
                    let aabb = Self::get_bounds(cache, tree, *prim);
                    let diff = pos - (aabb.min + mat.cols.map(|x| x.reduce_min()));
                    Self::contains_at(cache, tree, *prim, aabb.min + mat.transposed() * diff)
                }, */
                &Node::Translate(a, vec) => {
                    // /*Reverse*/Perform a translation.
                    mat = mat * Mat4::<i32>::translation_3d(/*-*/vec);
                    prim = a;
                },
                /* // This transformation unfortunately rotates around the center of the structure
                // rather than the origin, so it's effectively:
                //
                // translate to center (-center) * inverse scaling matrix (by vec) * translate back (+center)
                //
                // NOTE: We actually avoid using Scale for now, but in the future it would be nice
                // to restrict to integers so we can optimize output of scaled vertices.  In fact
                // for any axis-aligned, convex shape, there may be some smallest shape we can
                // sample to determine the edges before upsampling... etc.
                Node::Scale(a, vec) => {
                    /* let center = Self::get_bounds(cache, tree, *prim).center().as_::<f32>()
                        - Vec3::broadcast(0.5);
                    let fpos = pos.as_::<f32>();
                    let spos = (center + ((center - fpos) / vec))
                        .map(|x| x.round())
                        .as_::<i32>();
                    Self::contains_at::<Check>(cache, tree, *prim, spos) */
                    Self::get_bounds(cache, tree, a)
                    .into_iter()
                    .map(|aabb| {
                        let center = aabb.center();
                        Aabb {
                            min: center + ((aabb.min - center).as_::<f32>() * vec).as_::<i32>(),
                            max: center + ((aabb.max - center).as_::<f32>() * vec).as_::<i32>(),
                        }
                    })
                }, */
                Node::RotateAbout(a, rotation_mat, vec) => {
                    let vec = vec - 0.5;
                    let rotation_mat = rotation_mat.as_::<f32>();
                    let rotation_mat =
                        Mat4::<f32>::translation_3d(vec) *
                        Mat4::from(rotation_mat) *
                        Mat4::<f32>::translation_3d(-vec);
                    mat = mat * rotation_mat.as_();
                    prim = *a;
                    /* let mat_ = mat.as_::<f32>().transposed();
                    // - 0.5 because we want the point to be at the minimum of the voxel; the
                    // answer will always turn out to be an integer.  We could also scale by 2,
                    // then divide by 2, to get the same effect without casting to float.
                    let vec = vec - 0.5;
                    // We perform a *forwards* rotation, so that we can iterate through the
                    // two bounding boxes simultaneously without needing to redo matrix
                    // multiplication each time (this should *probably* be faster).
                    // Translate origin to vec.
                    mat_ = Mat4::<f32>::translation_3d(-vec) * mat_;
                    // Rotate around origin.
                    mat_ = Mat4::from(rotation_mat) * mat_;
                    // Return origin to normal.
                    mat_ = Mat4::<f32>::translation_3d(vec) * mat_;
                    mat = mat_.as_(); */
                    /* Self::contains_at::<Check>(cache, tree, *prim, (vec + mat * (pos.as_::<f32>() - vec)).as_())

                    // Reverse the rotation.
                    let rotation_mat = rotation_mat.as_::<f32>().transposed();
                    // For some reason, we assume people are lying about the origin?  This feels
                    // very fraught with footguns...
                    let vec = vec - 0.5;
                    // Undo a rotation about the origin.
                    /* Self::contains_at::<Check>(cache, tree, *prim, (vec + mat * (pos.as_::<f32>() - vec)).as_())
                    Self::get_bounds(cache, tree, a)
                    .into_iter()
                    .map(|aabb| {
                        let mat = mat.as_::<f32>();
                        // - 0.5 because we want the point to be at the minimum of the voxel
                        let vec = vec - 0.5;
                        let new_aabb = Aabb::<f32> {
                            min: vec + mat * (aabb.min.as_() - vec),
                            // - 1 becuase we want the AABB to be inclusive when we rotate it, we then
                            //   add 1 back to make it exclusive again
                            max: vec + mat * ((aabb.max - 1).as_() - vec),
                        }
                        .made_valid();
                        Aabb::<i32> {
                            min: new_aabb.min.as_(),
                            max: new_aabb.max.as_() + 1,
                        }
                    }) */ */
                },
                /* Node::Translate(x, vec) => {
                    prim = x;
                    // TODO: Saturating sub?  Or figure out why we should even be doing that?
                    aabb.min -= vec;
                    aabb.max -= vec;
                    // Self::contains_at::<Check>(cache, tree, *prim, pos.map2(*vec, i32::saturating_sub))
                },
                Node::Scale(x, vec) => {
                    prim = x;
                    let center = Self::get_bounds(cache, tree, x).center().as_::<f32>()
                        - Vec3::broadcast(0.5);
                    let faabb = aabb.as_::<f32>();
                    let scale = |fv: Vec3<f32>| (center + ((center - fv) / vec))
                        .map(|x| x.round())
                        .as_::<i32>();
                    aabb.min = scale(faabb.min);
                    aabb.max = scale(faabb.max);
                },
                Node::RotateAbout(x, mat, vec) => {
                    prim = x;
                    let mat = mat.as_::<f32>().transposed();
                    let vec = vec - 0.5;
                    aabb += mat * (aabb.as_::<f32>() - vec).as_();
                }, */
            }
        }
    }

    pub fn get_bounds_disjoint<'a>(
        arena: &'a bumpalo::Bump,
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
        mask: Aabr<i32>,
        mut hit: impl FnMut(Vec3<i32>),
    ) {
        if PRINT_MESSAGES {
            println!("\n\nSite: {}", prim.id());
        }
        let mut stack = Vec::new();
        let bounds = Self::get_bounds(cache, tree, prim);
        let mask = Aabb {
            min: Vec3::from(mask.min).with_z(bounds.min.z),
            max: Vec3::from(mask.max).with_z(bounds.max.z),
        };
        Self::get_bounds_disjoint_inner(arena, cache, tree, prim, 0, &mut stack, Mat4::identity(), mask, &mut hit)
    }

    pub fn get_bounds_opt<'a>(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
    ) -> Option<Aabb<i32>> {
        Self::get_bounds_inner(cache, tree, prim)
            .into_iter()
            .reduce(|a, b| a.union(b))
    }

    pub fn get_bounds_prim<'a>(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: &Primitive<'a>,
    ) -> Aabb<i32> {
        Self::get_bounds_inner_prim(cache, tree, prim)
            .into_iter()
            .reduce(|a, b| a.union(b))
            .unwrap_or_else(|| Aabb::new_empty(Vec3::zero()))
    }

    pub fn get_bounds<'a>(
        cache: &mut BoundsMap,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
    ) -> Aabb<i32> {
        Self::get_bounds_opt(cache, tree, prim).unwrap_or_else(|| Aabb::new_empty(Vec3::zero()))
    }
}

/* impl<'a> Fill<'a> {
    fn Self::contains_at(tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>, pos: Vec3<i32>) -> bool {
        // Custom closure because vek's impl of `contains_point` is inclusive :(
        let aabb_contains = |aabb: Aabb<i32>, pos: Vec3<i32>| {
            (aabb.min.x..aabb.max.x).contains(&pos.x)
                && (aabb.min.y..aabb.max.y).contains(&pos.y)
                && (aabb.min.z..aabb.max.z).contains(&pos.z)
        };

        match &tree[prim] {
            Node::Empty => false,

            Node::Aabb(aabb) => aabb_contains(*aabb, pos),
            Node::Ramp { aabb, inset, dir } => {
                let inset = (*inset).max(aabb.size().reduce_min());
                let inner = match dir {
                    Dir::X => Aabr {
                        min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegX => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                    },
                    Dir::Y => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegY => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                    },
                };
                aabb_contains(*aabb, pos)
                    && (inner.projected_point(pos.xy()) - pos.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0
                            - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
            },
            Node::Pyramid { aabb, inset } => {
                let inset = (*inset).max(aabb.size().reduce_min());
                let inner = Aabr {
                    min: aabb.min.xy() - 1 + inset,
                    max: aabb.max.xy() - inset,
                };
                aabb_contains(*aabb, pos)
                    && (inner.projected_point(pos.xy()) - pos.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0
                            - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
            },
            Node::Gable { aabb, inset, dir } => {
                let inset = (*inset).max(aabb.size().reduce_min());
                let inner = if dir.is_y() {
                    Aabr {
                        min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                        max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                    }
                } else {
                    Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                        max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                    }
                };
                aabb_contains(*aabb, pos)
                    && (inner.projected_point(pos.xy()) - pos.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0
                            - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
            },
            Node::Cylinder(aabb) => {
                (aabb.min.z..aabb.max.z).contains(&pos.z)
                    && (pos
                        .xy()
                        .as_()
                        .distance_squared(aabb.as_().center().xy() - 0.5)
                        as f32)
                        < (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2)
            },
            Node::Cone(aabb) => {
                (aabb.min.z..aabb.max.z).contains(&pos.z)
                    && pos
                        .xy()
                        .as_()
                        .distance_squared(aabb.as_().center().xy() - 0.5)
                        < (((aabb.max.z - pos.z) as f32 / aabb.size().d as f32)
                            * (aabb.size().w.min(aabb.size().h) as f32 / 2.0))
                            .powi(2)
            },
            Node::Sphere(aabb) => {
                aabb_contains(*aabb, pos)
                    && pos.as_().distance_squared(aabb.as_().center() - 0.5)
                        < (aabb.size().w.min(aabb.size().h) as f32 / 2.0).powi(2)
            },
            Node::Superquadric { aabb, degree } => {
                let degree = degree.max(0.0);
                let center = aabb.center().map(|e| e as f32);
                let a: f32 = aabb.max.x as f32 - center.x - 0.5;
                let b: f32 = aabb.max.y as f32 - center.y - 0.5;
                let c: f32 = aabb.max.z as f32 - center.z - 0.5;
                let rpos = pos.as_::<f32>() - center;
                aabb_contains(*aabb, pos)
                    && (rpos.x / a).abs().powf(degree)
                        + (rpos.y / b).abs().powf(degree)
                        + (rpos.z / c).abs().powf(degree)
                        < 1.0
            },
            Node::Plane(aabr, origin, gradient) => {
                // Maybe <= instead of ==
                (aabr.min.x..aabr.max.x).contains(&pos.x)
                    && (aabr.min.y..aabr.max.y).contains(&pos.y)
                    && pos.z
                        == origin.z
                            + ((pos.xy() - origin.xy())
                                .map(|x| x.abs())
                                .as_()
                                .dot(*gradient) as i32)
            },
            // TODO: Aabb calculation could be improved here by only considering the relevant radius
            Node::Segment { segment, r0, r1 } => {
                let distance = segment.end - segment.start;
                let length = pos - segment.start.as_();
                let t =
                    (length.as_().dot(distance) / distance.magnitude_squared()).clamped(0.0, 1.0);
                segment.distance_to_point(pos.map(|e| e as f32)) < Lerp::lerp(r0, r1, t) - 0.25
            },
            Node::SegmentPrism {
                segment,
                radius,
                height,
            } => {
                let segment_2d = LineSegment2 {
                    start: segment.start.xy(),
                    end: segment.end.xy(),
                };
                let projected_point_2d: Vec2<f32> =
                    segment_2d.as_().projected_point(pos.xy().as_());
                let xy_check = projected_point_2d.distance(pos.xy().as_()) < radius - 0.25;
                let projected_z = {
                    let len_sq: f32 = segment_2d
                        .start
                        .as_()
                        .distance_squared(segment_2d.end.as_());
                    if len_sq < 0.1 {
                        segment.start.z as f32
                    } else {
                        let frac = ((projected_point_2d - segment_2d.start.as_())
                            .dot(segment_2d.end.as_() - segment_2d.start.as_())
                            / len_sq)
                            .clamp(0.0, 1.0);
                        (segment.end.z as f32 - segment.start.z as f32) * frac
                            + segment.start.z as f32
                    }
                };
                let z_check = (projected_z..=(projected_z + height)).contains(&(pos.z as f32));
                xy_check && z_check
            },
            Node::Sampling(a, f) => Self::contains_at(tree, *a, pos) && f(pos),
            Node::Prefab(p) => !matches!(p.get(pos), Err(_) | Ok(StructureBlock::None)),
            Node::Intersect(a, b) => {
                Self::contains_at(tree, *a, pos) && Self::contains_at(tree, *b, pos)
            },
            Node::IntersectAll(prims) => {
                prims.into_iter().all(|prim| Self::contains_at(tree, *prim, pos))
            },
            Node::Union(a, b) => {
                Self::contains_at(tree, *a, pos) || Self::contains_at(tree, *b, pos)
            },
            Node::UnionAll(prims) => {
                prims.into_iter().any(|prim| Self::contains_at(tree, *prim, pos))
            },
            Node::Without(a, b) => {
                Self::contains_at(tree, *a, pos) && !Self::contains_at(tree, *b, pos)
            },
            Node::Translate(prim, vec) => {
                Self::contains_at(tree, *prim, pos.map2(*vec, i32::saturating_sub))
            },
            Node::Scale(prim, vec) => {
                let center =
                    Self::get_bounds(tree, *prim).center().as_::<f32>() - Vec3::broadcast(0.5);
                let fpos = pos.as_::<f32>();
                let spos = (center + ((center - fpos) / vec))
                    .map(|x| x.round())
                    .as_::<i32>();
                Self::contains_at(tree, *prim, spos)
            },
            Node::RotateAbout(prim, mat, vec) => {
                let mat = mat.as_::<f32>().transposed();
                let vec = vec - 0.5;
                Self::contains_at(tree, *prim, (vec + mat * (pos.as_::<f32>() - vec)).as_())
            },
            Node::Repeat(prim, offset, count) => {
                if count == &0 {
                    false
                } else {
                    let count = count - 1;
                    let aabb = Self::get_bounds(tree, *prim);
                    let aabb_corner = {
                        let min_red = aabb.min.map2(*offset, |a, b| if b < 0 { 0 } else { a });
                        let max_red = aabb.max.map2(*offset, |a, b| if b < 0 { a } else { 0 });
                        min_red + max_red
                    };
                    let diff = pos - aabb_corner;
                    let min = diff
                        .map2(*offset, |a, b| if b == 0 { i32::MAX } else { a / b })
                        .reduce_min()
                        .clamp(0, count as i32);
                    let pos = pos - offset * min;
                    Self::contains_at(tree, *prim, pos)
                }
            },
        }
    }

    pub fn sample_at(
        &self,
        tree: &Store<Primitive<'a>>,
        prim: Id<Primitive<'a>>,
        pos: Vec3<i32>,
        canvas_info: &CanvasInfo,
        old_block: Block,
    ) -> Option<Block> {
        if Self::contains_at(tree, prim, pos) {
            match self {
                Fill::Block(block) => Some(*block),
                Fill::Sprite(sprite) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                } else {
                    old_block.with_sprite(*sprite)
                }),
                Fill::RotatedSprite(sprite, ori) => Some(if old_block.is_filled() {
                    Block::air(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| Block::air(*sprite))
                } else {
                    old_block
                        .with_sprite(*sprite)
                        .with_ori(*ori)
                        .unwrap_or_else(|| old_block.with_sprite(*sprite))
                }),
                Fill::Brick(bk, col, range) => Some(Block::new(
                    *bk,
                    *col + (RandomField::new(13)
                        .get((pos + Vec3::new(pos.z, pos.z, 0)) / Vec3::new(2, 2, 1))
                        % *range as u32) as u8,
                )),
                Fill::Gradient(gradient, bk) => Some(Block::new(*bk, gradient.sample(pos.as_()))),
                Fill::Prefab(p, tr, seed) => p.get(pos - tr).ok().and_then(|sb| {
                    let col_sample = canvas_info.col(canvas_info.wpos)?;
                    block_from_structure(
                        canvas_info.index,
                        *sb,
                        pos - tr,
                        p.get_bounds().center().xy(),
                        *seed,
                        col_sample,
                        Block::air,
                        canvas_info.calendar(),
                    )
                }),
                Fill::Sampling(f) => f(pos),
            }
        } else {
            None
        }
    }

    fn get_bounds_inner(tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>) -> Vec<Aabb<i32>> {
        fn or_zip_with<T, F: FnOnce(T, T) -> T>(a: Option<T>, b: Option<T>, f: F) -> Option<T> {
            match (a, b) {
                (Some(a), Some(b)) => Some(f(a, b)),
                (Some(a), _) => Some(a),
                (_, b) => b,
            }
        }

        match &tree[prim] {
            Node::Empty => vec![],
            Node::Aabb(aabb) => vec![*aabb],
            Node::Pyramid { aabb, .. } => vec![*aabb],
            Node::Gable { aabb, .. } => vec![*aabb],
            Node::Ramp { aabb, .. } => vec![*aabb],
            Node::Cylinder(aabb) => vec![*aabb],
            Node::Cone(aabb) => vec![*aabb],
            Node::Sphere(aabb) => vec![*aabb],
            Node::Superquadric { aabb, .. } => vec![*aabb],
            Node::Plane(aabr, origin, gradient) => {
                let half_size = aabr.half_size().reduce_max();
                let longest_dist = ((aabr.center() - origin.xy()).map(|x| x.abs())
                    + half_size
                    + aabr.size().reduce_max() % 2)
                    .map(|x| x as f32);
                let z = if gradient.x.signum() == gradient.y.signum() {
                    Vec2::new(0, longest_dist.dot(*gradient) as i32)
                } else {
                    (longest_dist * gradient).as_()
                };
                let aabb = Aabb {
                    min: aabr.min.with_z(origin.z + z.reduce_min().min(0)),
                    max: aabr.max.with_z(origin.z + z.reduce_max().max(0)),
                };
                vec![aabb.made_valid()]
            },
            Node::Segment { segment, r0, r1 } => {
                let aabb = Aabb {
                    min: segment.start,
                    max: segment.end,
                }
                .made_valid();
                vec![Aabb {
                    min: (aabb.min - r0.max(*r1)).floor().as_(),
                    max: (aabb.max + r0.max(*r1)).ceil().as_(),
                }]
            },
            Node::SegmentPrism {
                segment,
                radius,
                height,
            } => {
                let aabb = Aabb {
                    min: segment.start,
                    max: segment.end,
                }
                .made_valid();
                let min = {
                    let xy = (aabb.min.xy() - *radius).floor();
                    xy.with_z(aabb.min.z).as_()
                };
                let max = {
                    let xy = (aabb.max.xy() + *radius).ceil();
                    xy.with_z((aabb.max.z + *height).ceil()).as_()
                };
                vec![Aabb { min, max }]
            },
            Node::Sampling(a, _) => Self::get_bounds_inner(tree, *a),
            Node::Prefab(p) => vec![p.get_bounds()],
            Node::Intersect([a, b]) => or_zip_with(
                Self::get_bounds_opt(tree, *a),
                Self::get_bounds_opt(tree, *b),
                |a, b| a.intersection(b),
            )
            .into_iter()
            .collect(),

            Node::Union(a, b) => {
                fn jaccard(x: Aabb<i32>, y: Aabb<i32>) -> f32 {
                    let s_intersection = x.intersection(y).size().as_::<f32>().magnitude();
                    let s_union = x.union(y).size().as_::<f32>().magnitude();
                    s_intersection / s_union
                }
                let mut inputs = Vec::new();
                inputs.extend(Self::get_bounds_inner(tree, *a));
                inputs.extend(Self::get_bounds_inner(tree, *b));
                let mut results = Vec::new();
                if let Some(aabb) = inputs.pop() {
                    results.push(aabb);
                    for a in &inputs {
                        let best = results
                            .iter()
                            .enumerate()
                            .max_by_key(|(_, b)| (jaccard(*a, **b) * 1000.0) as usize);
                        match best {
                            Some((i, b)) if jaccard(*a, *b) > 0.3 => {
                                let mut aabb = results.swap_remove(i);
                                aabb = aabb.union(*a);
                                results.push(aabb);
                            },
                            _ => results.push(*a),
                        }
                    }
                    results
                } else {
                    results
                }
            },
            Node::Without(a, _) => Self::get_bounds_inner(tree, *a),
            Node::Translate(prim, vec) => Self::get_bounds_inner(tree, *prim)
                .into_iter()
                .map(|aabb| Aabb {
                    min: aabb.min.map2(*vec, i32::saturating_add),
                    max: aabb.max.map2(*vec, i32::saturating_add),
                })
                .collect(),
            /* Node::Scale(prim, vec) => Self::get_bounds_inner(tree, *prim)
                .into_iter()
                .map(|aabb| {
                    let center = aabb.center();
                    Aabb {
                        min: center + ((aabb.min - center).as_::<f32>() * vec).as_::<i32>(),
                        max: center + ((aabb.max - center).as_::<f32>() * vec).as_::<i32>(),
                    }
                })
                .collect(), */
            Node::RotateAbout(prim, mat, vec) => Self::get_bounds_inner(tree, *prim)
                .into_iter()
                .map(|aabb| {
                    let mat = mat.as_::<f32>();
                    // - 0.5 because we want the point to be at the minimum of the voxel
                    let vec = vec - 0.5;
                    let new_aabb = Aabb::<f32> {
                        min: vec + mat * (aabb.min.as_() - vec),
                        // - 1 becuase we want the AABB to be inclusive when we rotate it, we then
                        //   add 1 back to make it exclusive again
                        max: vec + mat * ((aabb.max - 1).as_() - vec),
                    }
                    .made_valid();
                    Aabb::<i32> {
                        min: new_aabb.min.as_(),
                        max: new_aabb.max.as_() + 1,
                    }
                })
                .collect(),
            /* Node::Repeat(prim, offset, count) => {
                if count == &0 {
                    vec![]
                } else {
                    let count = count - 1;
                    Self::get_bounds_inner(tree, *prim)
                        .into_iter()
                        .map(|aabb| Aabb {
                            min: aabb
                                .min
                                .map2(aabb.min + offset * count as i32, |a, b| a.min(b)),
                            max: aabb
                                .max
                                .map2(aabb.max + offset * count as i32, |a, b| a.max(b)),
                        })
                        .collect()
                }
            }, */
        }
    }

    pub fn get_bounds_disjoint(tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>) -> Vec<Aabb<i32>> {
        Self::get_bounds_inner(tree, prim)
    }

    pub fn get_bounds_opt(tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>) -> Option<Aabb<i32>> {
        Self::get_bounds_inner(tree, prim)
            .into_iter()
            .reduce(|a, b| a.union(b))
    }

    pub fn get_bounds(tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>) -> Aabb<i32> {
        Self::get_bounds_opt(tree, prim).unwrap_or_else(|| Aabb::new_empty(Vec3::zero()))
    }
} */

pub struct Painter<'a/*, N: NodeAllocator = DefaultNodeAllocator*/> {
    pub(super) arena: &'a bumpalo::Bump,
    prims: RefCell<Store<Primitive<'a>>>,
    // fills: RefCell<Vec<(Id<Primitive<'a>>, Fill<'a>)>>,
    cached_depth: RefCell<HashMap<Id<Primitive<'a>>, f32, BuildHasherDefault<FxHasher64>>>,
    bounds_cache: RefCell<BoundsMap>,
    // node_allocator: N,
    // entities: RefCell<Vec<EntityInfo>>,
    // render_area: Aabr<i32>,
}

impl<'a/*, N: NodeAllocator + Default*/> Painter<'a/*, N*/> {
    fn new(arena: &'a bumpalo::Bump/*, render_area: Aabr<i32>*/) -> Painter<'a/*, N*/> {
        Painter {
            arena,
            prims: RefCell::new(Store::default()),
            // fills: RefCell::new(Vec::new()),
            cached_depth: RefCell::new(HashMap::default()),
            bounds_cache: RefCell::new(HashMap::default()),
            // node_allocator: N::default(),
            // entities: RefCell::new(Vec::new()),
            // render_area,
        }
    }
}

/* pub fn calculate_depths(prims: &Store<Primitive>) -> Vec<usize> {
    let mut ret: Vec<usize> = Vec::with_capacity(prims.len());
    for (_id, prim) in prims.iter() {
        let depth = match prim {
            Node::Empty
            | Node::Aabb(_)
            | Node::Pyramid { .. }
            | Node::Ramp { .. }
            | Node::Gable { .. }
            | Node::Cylinder(_)
            | Node::Cone(_)
            | Node::Sphere(_)
            | Node::Superquadric { .. }
            | Node::Plane(_, _, _)
            | Node::Segment { .. }
            | Node::SegmentPrism { .. }
            | Node::Prefab(_) => 0,
            Node::Sampling(a, _)
            /* | Node::Rotate(a, _) */
            | Node::Translate(a, _)
            | Node::Scale(a, _)
            | Node::RotateAbout(a, _, _)
            | Node::Repeat(a, _, _) => 1 + ret[a.id() as usize],

            Node::Intersect(a, b)
            | Node::Union(a, b)
            | Node::Without(a, b) => (1 + ret[a.id() as usize]).max(1 + ret[b.id() as usize]),
            Node::IntersectAll(xs)
            | Node::UnionAll(xs) =>
                xs.len() +
                xs.into_iter().copied().map(|a| ret[a.id() as usize]).max().unwrap_or(0),
        };
        ret.push(depth);
    }
    ret
} */

/// Cost model:
///
/// Costs represent the proportion of blocks in the enclosing AABB consumed by the primitive.
/// Every context is always associated with a bounding AABB for that reason.
///
/// ---
///
/// Primitives (except sampled primitives, which assume they are using the whole AABB) have
/// reasonably accurate cost estimates as a percentage of their bounding box.  Trivial cases like
/// Empty or Plane can compute exact values, and many of the other primitives (for example, sphere)
/// have volume equations with closed-form solutions.
///
/// ---
///
/// Unions are assumed to (almost always) be disjoint, so they are estimated as:
///
/// bound(A ∪ B) = bound(A) ∪⁺ bound(B)
/// cost(A u B) = (cost(A) * vol(bound(A)) + cost(B) * vol(bound(B))) / vol(bound(A) u⁺ bound(B))
///
/// where ∪⁺ is a convex union (in our case, we currently use AABB union, which is suboptimal for
/// this purpose but still acceptable).  Ideally, all unions would be top level, so we would not
/// lose precision in that way.
///
/// Because unions are assumed to be disjoint, for any run through the *whole* AABB order of
/// evaluation doesn't matter.  However, in other contexts where not all AABBs are run
/// through, it may be desirable to terminate as early as possible.  Using the conservative
/// assumption that within the parent intersection, blocks are chosen independently on both
/// sides, we can make the rough estimate that it's better to sort the union in decreasing order by
/// cost.
///
/// ---
///
/// Intersections are assumed to join blocks independently within the shared part of the AABB, and
/// be zero outside of it.  The former part is not usually true, but it's at least an assumption we
/// can work with.  We also make the simplifying (false) assumption that the probability
/// distribution of blocks being within an intersection's AABB is uniform throughout it.  Given
/// these assumptions, we can estimate them as:
///
/// bound(A ∩ B) = bound(A) ∩⁺ bound(B)
/// cost(A ∩ B) = cost(A) * cost(B)
///
/// Note that ∩ of convex volumes is always convex, so we lose very little precision this way.
/// However, we still use ∩⁺ currently to get a less precise (AABB) bounding box.  As long as both
/// volumes are already bounding boxes, this doesn't really change our calculations much.
///
/// To determine the best intersection order, observe that we won't draw anything if either A or B
/// evaluate to false.  Therefore, we sort the intersection in increasing order by cost.
///
/// ---
///
/// Differences are estimated by assuming that the intersecting part is independent within A ∩ B,
/// and 0 elsewhere, so it is estimated as:
///
/// bound(A \ B) = bound(A)
/// cost(B | bound(A)) = cost(B) * vol(bound(A) ∩⁺ bound(B)) / vol(bound(A))
/// cost(A \ B) = cost(A) * (1 - cost(B | bound(A)))
///
/// In theory, we can sometimes do better if chopping off bound(B) still results in a convex
/// bounding volume (or AABB in this case), but we don't try to perform this optimization for now.
///
/// Since if B evaluates to true, we won't draw anything, and if A evaluates to false, we won't
/// draw anything, we test B first only if 1 - cost(B | bound(A)) < cost(A).  Otherwise, we
/// test A first.
///
/// ----
///
/// For all other nodes, we either desugar them to one of the formulae above (e.g. repeats are
/// equivalent to unions), or change the bounding box without changing the cost (e.g. scale
/// increases the size in all directions, translate moves it, and rotate_about also moves it [we
/// assume that rotate_about only performs 90 degree turns, which preserve bounding volumes]).
pub fn depth_with_cache<'a>(
    prims: &Store<Primitive<'a>>,
    cached_depth: &mut HashMap<Id<Primitive<'a>>, f32, BuildHasherDefault<FxHasher64>>,
    cached_bounds: &mut BoundsMap,
    prim: Id<Primitive<'a>>,
) -> f32 {
    fn aux<'a>(
        prims: &Store<Primitive<'a>>,
        cached_depth: &mut HashMap<Id<Primitive<'a>>, f32, BuildHasherDefault<FxHasher64>>,
        cached_bounds: &mut BoundsMap,
        prim: Id<Primitive<'a>>,
        /* prev_depth: f32, */
    ) -> f32 {
        /* let vacant_entry = match cached_depth.entry(prim) {
            Entry::Occupied(o) => return *o.into_mut() + prev_depth,
            Entry::Vacant(v) => v,
        }; */
        if let Some(depth) = cached_depth.get(&prim) {
            /* return depth; */
            return *depth/* + prev_depth*/;
        }
        let depth = match prims[prim].0 {
            // All these primitives are sums of 0 | 1, so they are positive
            // of depth 1.
            // ⊤/1 is both positive and negative, so it has depth 0.
            Node::Empty => 0.0,
            // Aabbs are always full (TODO: eliminate empty Aabbs at construction time).
            Node::Aabb(_) => 1.0,
            // TODO: Estimate properly (what is inset?)
            Node::Pyramid { .. } => {
                0.5
            },
            // Returns a `PrimitiveRef` of an Aabb with a slope cut into it. The
            // `inset` governs the slope. The `dir` determines which direction the
            // ramp points.
            // FIXME: Estimate properly (don't understand the description).
            Node::Ramp { /*aabb, inset, dir*/.. } => {
                /* let inset = (*inset).max(aabb.size().reduce_min());
                let inner = match dir {
                    Dir::X => Aabr {
                        min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegX => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                    },
                    Dir::Y => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                        max: Vec2::new(aabb.max.x, aabb.max.y),
                    },
                    Dir::NegY => Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y),
                        max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                    },
                }.made_valid();
                aabb_contains(*aabb, pos)
                    && (inner.projected_point(pos.xy()) - pos.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0
                            - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32 */
                0.5
            },
            Node::Gable { /*aabb, inset, dir*/.. } => {
                /* let inset = inset.max(aabb.size().reduce_min());
                // inner is a one-voxel-wide rectangle; if dir is y, the rectangle's long side is
                // on the y axis, if it's x then the long side is on the x axis.  The location on
                // the other axis of the rectangle is determined by the inset.
                let inner = if dir.is_y() {
                    Aabr {
                        min: Vec2::new(aabb.min.x - 1 + inset, aabb.min.y),
                        max: Vec2::new(aabb.max.x - inset, aabb.max.y),
                    }
                } else {
                    Aabr {
                        min: Vec2::new(aabb.min.x, aabb.min.y - 1 + inset),
                        max: Vec2::new(aabb.max.x, aabb.max.y - inset),
                    }
                }.made_valid();
                // Find the distance to the inner rectangle (specifically, the max distance along
                // all axes to the closest point on the rectangle).  Divide by the inset, which
                // governs the slope of the gable.
                //
                // Compare this to 1.0 - (z - z₀) / (z₁ - z₀).
                //
                // (x - x₀) / inset < 1 - (z - z₀) / (z₁ - z₀)
                //
                // (z₁ - z₀) / inset < ((z₁ - z₀) - (z - z₀)) / (x - x₀)
                // (z₁ - z₀) / inset < (z₁ - z) / (x - x₀)
                //
                // (z₁ - z) inset = (z₁ - z₀)(x - x₀)
                // z = 1/inset(z₁ + (z₀ - z₁)(x - x₀))
                //   = (z₀ - z₁)/inset * (x - x₀) + z₁/inset
                //
                // (x - x₀) / (z - z₀) < inset * (1 / (z - z₀) - 1 / (z₁ - z₀))
                //
                // (x - x₀) / (z - z₀) < inset * ((z₁ - z₀) / (z - z₀) - 1 / (z₁ - z₀))
                //
                // (x - x₀) / inset < 1 - (z - z₀) / (z₁ - z₀)
                //
                aabb_contains(aabb, pos)
                    && (inner.projected_point(pos.xy()) - pos.xy())
                        .map(|e| e.abs())
                        .reduce_max() as f32
                        / (inset as f32)
                        < 1.0
                            - ((pos.z - aabb.min.z) as f32 + 0.5) / (aabb.max.z - aabb.min.z) as f32
                */
                0.5
            },
            // Cylinder volume is π r² h; height is same as bounding box height so it always
            // cancels out (unless height is 0, which we check for to make sure we're well-formed).
            // As long as the radius exists, it will also cancel out at least one of the diameters,
            // so we know we're really just multiplying π/4 by the remaining bounding box axis.
            //
            // TODO: Prevent 0 height (and radius?) at construction time.
            Node::Cylinder(aabb) => {
                let aabb = Vec3::<i32>::from(aabb.size());
                let aabr = aabb.xy();
                let diameter = aabr.reduce_min();
                if diameter == 0 || aabb.z == 0 {
                    0.0
                } else {
                    let aabr = aabr.as_::<f32>();
                    let diameter = diameter as f32;
                    core::f32::consts::FRAC_PI_4 * (diameter / aabr.x) * (diameter / aabr.y)
                }
            },
            // TODO: Estimate properly
            Node::Cone(_) => {
                0.5
            },
            // Sphere volume is 4/3 π r³
            //
            // Radius is known by construction to be half the diameter across all axes, so this
            // becomes:
            //
            // 4/3 π (d/2)³ = π/6 d³
            //
            // Since the aabb is of size d³, we conclude that the ratio here is exactly π/6.
            Node::Sphere(_) => core::f32::consts::FRAC_PI_6,
            // TODO: Estimate properly
            Node::Superquadric { .. } => {
                0.5/*1.0*/
            },
            // TODO: Estimate properly
            Node::Plane(_, _, _) => {
                0.5
            },
            // We approximate line volume as a truncated cone:
            //
            // V = 1/3 π h (r₀² + r₀ r₁ + r₁²)
            //
            // We make no special effort to reduce this with knowledge about the bounding box.
            //
            // Also, to account for z_scale, we just divide the cylinder volume by z_scale;
            // since (I believe) z_scale is just a shearing transformation, and shearing
            // preserves volume, this should work.
            Node::Segment { segment, /* radius */r0, r1, z_scale } => {
                let aabb = Aabb {
                    min: segment.start,
                    max: segment.end,
                }
                .made_valid();
                let mut rad_diff = Vec3::broadcast(r0.max(r1));
                rad_diff.z /= z_scale;
                /* let aabb = Aabb {
                    min: (aabb.min - rad_diff).floor(),
                    max: (aabb.max + rad_diff).ceil() + 1.0,
                }; */
                let aabb = Aabb {
                    min: (aabb.min - rad_diff).floor(),
                    max: (aabb.max + rad_diff).ceil(),
                };
                let aabb = aabb.size().as_::<f32>();
                let distance = segment.end.distance(segment.start);

                if aabb.product() < 1.0 {
                    0.0
                } else {
                    core::f32::consts::FRAC_PI_3 *
                    (distance / (aabb.d * z_scale)) *
                    ((r0 * r0 + r0 * r1 + r1 * r1) / (aabb.w * aabb.h))
                }
            },
            // TODO: Estimate properly
            Node::SegmentPrism { .. } => {
                0.5
            },
            // Prefabs can perform a precise density calculation using the number of voxels in the
            // model.
            Node::Prefab(prefab) => {
                // Precision-losing behavior is fine on f32 overflow here (realistically, it should
                // never happen, but it's fine if it does).
                //
                // NOTE: We know get_bounds returns a nonzero value because we check on
                // construction of the Prefab that all bounds are nonzero.
                prefab.len() as f32 / prefab.get_bounds().size().as_::<f32>().product()
            },
            /* // Sampling is evil.  It is essentially an intersection with the other primitive, a,
            // using a's bounding box, and the *intention* is that it is completely covered by a
            // (so it should always be cheaper to hit the sampler first from a strict element count
            // perspective, and they are also not very independent!).  But we don't actually know
            // how much of a the sampler is using, so we have no good way to guess the overall
            // cost.  For now, we conservatively just guess that the density in a is 50%.  Worse,
            // samplers might be (usually will be) quite expensive, and we have no notion of the
            // underlying compute cost there (actually, we don't have such a notion for anything,
            // but that's neither here nor there).
            //
            // TODO: Consider taking an explicit density and/or compute cost estimate?
            Node::Sampling(a, _) => 0.5 * aux(prims, cached_depth, cached_bounds, a), */
            // These all just inherit the cost from a.
            /* | Node::Rotate(a, _) */
            Node::Translate(a, _)
            /* | Node::Scale(a, _) */
            | Node::RotateAbout(a, _, _) => aux(prims, cached_depth, cached_bounds, a),
            /* // Repeat is treated like a union of n instances of a; since it's only well-defined if
            // the union is disjoint, we just assume that it is, and compute the cost as the ratio
            // of repeating the shape n times to the actual selected bounding box.
            Node::Repeat(a, _, n) => {
                let a_bounds = Painter::get_bounds(cached_bounds, prims, a);
                let an_bounds = Painter::get_bounds(cached_bounds, prims, prim);
                let an_volume = an_bounds.size().as_::<f32>().product();
                if an_volume < 1.0 {
                    0.0
                } else {
                    let a_volume = a_bounds.size().as_::<f32>().product();
                    // In case the union isn't actually disjoint, we don't promise well-defined
                    // behavior, but we still clamp to 1 to keep the probabilities sane.
                    let an_scale = (a_volume / an_volume * f32::from(n)).min(1.0);
                    aux(prims, cached_depth, cached_bounds, a) * an_scale
                }
            }, */
            // Intersections are assumed independent...
            Node::Intersect([a, b]) => {
                /* // But only within the actual bounds used.  If the intersected AABB is very wide
                // compared to the estimated space used by the product of constituent volumes, we
                // drop our estimate to compensate.
                //
                // Note that this is *not* the Cartesian product solution (which blows up
                // insanely quickly in the general case), we're still using the independence
                // assumption, but we're re-estimating the actual volume used in case the AABBs fit
                // poorly.
                let a_bounds = Painter::get_bounds(cached_bounds, prims, a);
                let b_bounds = Painter::get_bounds(cached_bounds, prims, b); */
                /* // This could be quite expensive, but for now we just do the Cartesian product
                // because it can be extremely slow otherwise (one can imagine many heuristics
                // here). */
                let a_cost = aux(prims, cached_depth, cached_bounds, a);
                let b_cost = aux(prims, cached_depth, cached_bounds, b);
                a_cost * b_cost
            },
            // Same as IntersectAll, but iterated.
            Node::IntersectAll(xs) =>
                xs.into_iter().copied().map(|x| aux(prims, cached_depth, cached_bounds, x)).product(),
            Node::Union(a, b) => {
                let a_bounds = Painter::get_bounds(cached_bounds, prims, a);
                let b_bounds = Painter::get_bounds(cached_bounds, prims, b);
                let ab_volume = a_bounds.union(b_bounds).size().as_::<f32>().product();
                if ab_volume < 1.0 {
                    0.0
                } else {
                    let a_volume = a_bounds.size().as_::<f32>().product();
                    let b_volume = a_bounds.size().as_::<f32>().product();
                    // Unions are scaled according to expected volume:
                    //
                    // vol(A) / vol(A ∪ B)
                    //
                    // We multiply by this smaller value for improved precision for +.
                    let a_scale = a_volume / ab_volume;
                    let b_scale = b_volume / ab_volume;
                    // Since we estimate unions to be disjoint, but they sometimes aren't, we might
                    // end up exceeding 1, so clamp to 1.
                    //
                    // TODO: Estimate using Jaccard distance or related?
                    (aux(prims, cached_depth, cached_bounds, a) * a_scale + aux(prims, cached_depth, cached_bounds, b) * b_scale)
                        .min(1.0)
                }
            },
            // Same as Union, but iterated, and we look up the cached volume.
            Node::UnionAll(xs) => {
                let xs_bounds = Painter::get_bounds(cached_bounds, &prims, prim);
                let xs_volume = xs_bounds.size().as_::<f32>().product();
                if xs_volume < 1.0 {
                    0.0
                } else {
                    xs.into_iter().copied().map(|x| {
                        let x_bounds = Painter::get_bounds(cached_bounds, prims, x);
                        let x_volume = x_bounds.size().as_::<f32>().product();
                        let x_scale = x_volume / xs_volume;
                        aux(prims, cached_depth, cached_bounds, x) * x_scale
                    })
                    .sum::<f32>()
                    // Since we estimate unions to be disjoint, but they sometimes aren't, we might
                    // end up exceeding 1, so clamp to 1.
                    //
                    // TODO: Estimate using Jaccard distance or related?
                    .min(1.0)
                }
            },
            Node::Without(a, b, _) => {
                let a_bounds = Painter::get_bounds(cached_bounds, &prims, a);
                let b_bounds = Painter::get_bounds(cached_bounds, &prims, b);
                let a_volume = a_bounds.size().as_::<f32>().product();
                let ab_volume = a_bounds.intersection(b_bounds).size().as_::<f32>().product();
                // We know a_volume is positive since this is checked when the with is formed
                // (we also happen to know that b_scale is a positive probability).
                let b_scale = ab_volume / a_volume;
                aux(prims, cached_depth, cached_bounds, a) *
                (1.0 - aux(prims, cached_depth, cached_bounds, b) * b_scale)
            },
        };

        /* let depth = match &prims[prim] {
            // All these primitives are sums of 0 | 1, so they are positive
            // of depth 1.
            // ⊤/1 is both positive and negative, so it has depth 0.
            Node::Empty
            | Node::Aabb(_)
            | Node::Pyramid { .. }
            | Node::Ramp { .. }
            | Node::Gable { .. }
            | Node::Cylinder(_)
            | Node::Cone(_)
            | Node::Sphere(_)
            | Node::Superquadric { .. }
            | Node::Plane(_, _, _)
            | Node::Segment { .. }
            | Node::SegmentPrism { .. }
            | Node::Prefab(_) => prev_depth,
            Node::Sampling(a, _)
            /* | Node::Rotate(a, _) */
            | Node::Translate(a, _)
            | Node::Scale(a, _)
            | Node::RotateAbout(a, _, _)
            | Node::Repeat(a, _, _) => aux(prims, cached_depth, *a, 1.0 + prev_depth),

            Node::Intersect(a, b)
            | Node::Union(a, b)
            | Node::Without(a, b, _) => aux(prims, cached_depth, *a, 1.0 + prev_depth)
                .max(aux(prims, cached_depth, *b, 1.0 + prev_depth)),
            Node::IntersectAll(xs) | Node::UnionAll(xs) =>
                /* xs.len() */1.0 +
                xs.into_iter().copied().map(|prim| n32(aux(prims, cached_depth, prim, prev_depth)))
                .max().map(f32::from).unwrap_or(0.0),
        }; */
        /*vacant_entry.insert(depth - prev_depth);*/
        if PRINT_MESSAGES {
            println!("Cost {}: {}", prim.id(), depth);
        }
        cached_depth.insert(prim, depth/* - prev_depth*/);
        depth
    }
    aux(prims, cached_depth, cached_bounds, prim/*, 0.0*/)
}

impl<'a> Painter<'a> {
    /// Computes the depth of the tree rooted at `prim`
    /* pub fn depth(&self, prim: Id<Primitive<'a>>) -> usize {
        fn aux<'a>(prims: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>, prev_depth: usize) -> usize {
            match prims[prim] {
                Node::Empty
                | Node::Aabb(_)
                | Node::Pyramid { .. }
                | Node::Ramp { .. }
                | Node::Gable { .. }
                | Node::Cylinder(_)
                | Node::Cone(_)
                | Node::Sphere(_)
                | Node::Superquadric { .. }
                | Node::Plane(_, _, _)
                | Node::Segment { .. }
                | Node::SegmentPrism { .. }
                | Node::Prefab(_) => prev_depth,
                Node::Sampling(a, _)
                | Node::Translate(a, _)
                | Node::Scale(a, _)
                | Node::RotateAbout(a, _, _)
                | Node::Repeat(a, _, _) => aux(prims, a, 1 + prev_depth),
                Node::Intersect(a, b) | Node::Union(a, b) | Node::Without(a, b) => {
                    aux(prims, a, 1 + prev_depth).max(aux(prims, b, 1 + prev_depth))
                },
                Node::IntersectAll(xs) | Node::UnionAll(xs) => {
                    xs.len() + xs.into_iter().copied().map(|prim| aux(prims, prim, prev_depth)).max().unwrap_or(0)
                },
            }
        }
        let prims = self.prims.borrow();
        aux(&prims, prim, 0)
    } */
    fn depth(&self, prim: Id<Primitive<'a>>) -> f32 {
        let prims = self.prims.borrow();
        let mut cached_depth = self.cached_depth.borrow_mut();
        let mut cached_bounds = self.bounds_cache.borrow_mut();
        depth_with_cache(&prims, &mut cached_depth, &mut cached_bounds, prim)
    }

    /// Orders two primitives by depth, since (A && (B && C)) is cheaper to
    /// evaluate than ((A && B) && C) due to short-circuiting.
    fn order_by_depth(
        &self,
        a: Id<Primitive<'a>>,
        b: Id<Primitive<'a>>,
    ) -> (Id<Primitive<'a>>, Id<Primitive<'a>>) {
        /* let (vol_a, vol_b) = {
            let prims = self.prims.borrow();
            let mut cached_bounds = self.bounds_cache.borrow_mut();
            (Fill::get_bounds(&mut cached_bounds, &prims, a).size().as_::<f32>().product(),
             Fill::get_bounds(&mut cached_bounds, &prims, b).size().as_::<f32>().product())
        }; */
        if self.depth(a)/* * vol_a */ < self.depth(b)/* * vol_b */ {
            (a, b)
        } else {
            (b, a)
        }
    }

    /// Returns a `PrimitiveRef` of an axis aligned bounding box. The geometric
    /// name of this shape is a "right rectangular prism."
    ///
    /// TODO: Cbn or Cbv?
    pub fn aabb/*<Kind>*/(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, /*Cbv*//*Kind*/Red/*Cbn*/> {
        self.prim(Node::Aabb(aabb.made_valid()))
    }

    pub fn union_all<'painter, Kind>(&'painter self, mut iter: impl ExactSizeIterator<Item=/*Id<Primitive<'a>>*/PrimitiveRef<'painter, 'a, Kind>>) -> PrimitiveRef<'painter, 'a, Kind> {
        match iter.len() {
            0 => self.prim(Node::Empty),
            1 => /*PrimitiveRef {
                id: iter.next().unwrap(),
                painter: self,
            }*/iter.next().unwrap(),
            2 => /*PrimitiveRef {
                id: iter.next().unwrap(),
                painter: self,
            }*/iter.next().unwrap().union(iter.next().unwrap()),
            _ => {
                let xs = self.arena.alloc_slice_fill_iter(iter.map(Into::into));
                // NOTE: Would be nice to somehow generate the cache at the same time as the
                // iterator, so we didn't need to allocate external memory...
                let prims = self.prims.borrow();
                let mut cached_depth = self.cached_depth.borrow_mut();
                let mut cached_bounds = self.bounds_cache.borrow_mut();
                xs.sort_by_cached_key(|&x| core::cmp::Reverse(n32(depth_with_cache(&prims, &mut cached_depth, &mut cached_bounds, x)/* * Fill::get_bounds(&mut cached_bounds, &prims, x).size().as_::<f32>().product()*/)));
                drop(prims);
                self.prim(Node::UnionAll(xs))
            },
        }
    }

    pub fn intersect_all<'painter, Kind>(&'painter self, mut iter: impl ExactSizeIterator<Item=/*Id<Primitive<'a>>*/PrimitiveRef<'painter, 'a, Kind>>) -> PrimitiveRef<'painter, 'a, Kind> {
        match iter.len() {
            0 => self.empty().as_kind(),
            1 => /*PrimitiveRef {
                id: iter.next().unwrap(),
                painter: self,
            }*/iter.next().unwrap(),
            2 => /*PrimitiveRef {
                id: iter.next().unwrap(),
                painter: self,
            }*/iter.next().unwrap().intersect(iter.next().unwrap()),
            _ => {
                let xs = self.arena.alloc_slice_fill_iter(iter.map(Into::into));
                // NOTE: Would be nice to somehow generate the cache at the same time as the
                // iterator, so we didn't need to allocate external memory...
                let prims = self.prims.borrow();
                let mut cached_depth = self.cached_depth.borrow_mut();
                let mut cached_bounds = self.bounds_cache.borrow_mut();
                xs.sort_by_cached_key(|&x| n32(depth_with_cache(&prims, &mut cached_depth, &mut cached_bounds, x)/* * Fill::get_bounds(&mut cached_bounds, &prims, x).size().as_::<f32>().product()*/));
                drop(prims);
                self.prim(Node::IntersectAll(xs))
            },
        }
    }

    /// Returns a `PrimitiveRef` of a sphere using a radius check.
    pub fn sphere(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        debug_assert!({
            let size = aabb.size();
            size.w == size.h && size.h == size.d
        });
        self.prim(Node::Sphere(aabb.made_valid()))
    }

    /// Returns a `PrimitiveRef` of a sphere using a radius check where a radius
    /// and origin are parameters instead of a bounding box.
    pub fn sphere_with_radius(&self, origin: Vec3<i32>, radius: f32) -> PrimitiveRef<'_, 'a, Cbn> {
        let min = origin - Vec3::broadcast(radius.round() as i32);
        let max = origin + Vec3::broadcast(radius.round() as i32);
        self.prim(Node::Sphere(Aabb { min, max }))
    }

    /// Returns a `PrimitiveRef` of a sphere by returning an ellipsoid with
    /// congruent legs. The voxel artifacts are slightly different from the
    /// radius check `sphere()` method.
    pub fn sphere2(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        let aabb = aabb.made_valid();
        let radius = aabb.size().w.min(aabb.size().h) / 2;
        let aabb = Aabb {
            min: aabb.center() - radius,
            max: aabb.center() + radius,
        };
        let degree = 2.0;
        self.prim(Node::Superquadric { aabb, degree })
    }

    /// Returns a `PrimitiveRef` of an ellipsoid by constructing a superquadric
    /// with a degree value of 2.0.
    pub fn ellipsoid(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        let aabb = aabb.made_valid();
        let degree = 2.0;
        self.prim(Node::Superquadric { aabb, degree })
    }

    /// Returns a `PrimitiveRef` of a superquadric. A superquadric can be
    /// thought of as a rounded Aabb where the degree determines how rounded
    /// the corners are. Values from 0.0 to 1.0 produce concave faces or
    /// "inverse rounded corners." A value of 1.0 produces a stretched
    /// octahedron (or a non-stretched octahedron if the provided Aabb is a
    /// cube). Values from 1.0 to 2.0 produce an octahedron with convex
    /// faces. A degree of 2.0 produces an ellipsoid. Values larger than 2.0
    /// produce a rounded Aabb. The degree cannot be less than 0.0 without
    /// the shape extending to infinity.
    pub fn superquadric(&self, aabb: Aabb<i32>, degree: f32) -> PrimitiveRef<'_, 'a, Cbn> {
        let aabb = aabb.made_valid();
        // self.prim(Node::Aabb(aabb))
        self.prim(Node::Superquadric { aabb, degree })
    }

    /// Returns a `PrimitiveRef` of a rounded Aabb by producing a superquadric
    /// with a degree value of 3.0.
    pub fn rounded_aabb(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        let aabb = aabb.made_valid();
        self.prim(Node::Superquadric { aabb, degree: 3.0 })
    }

    /// Returns a `PrimitiveRef` of the largest cylinder that fits in the
    /// provided Aabb.
    pub fn cylinder(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        self.prim(Node::Cylinder(aabb.made_valid()))
    }

    /// Returns a `PrimitiveRef` of a cylinder using a radius check where a
    /// radius and origin are parameters instead of a bounding box.
    pub fn cylinder_with_radius(
        &self,
        origin: Vec3<i32>,
        radius: f32,
        height: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let min = origin - Vec2::broadcast(radius.round() as i32);
        let max = origin + Vec2::broadcast(radius.round() as i32).with_z(height.round() as i32);
        self.prim(Node::Cylinder(Aabb { min, max }))
    }

    /// Returns a `PrimitiveRef` of the largest cone that fits in the
    /// provided Aabb.
    pub fn cone(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        self.prim(Node::Cone(aabb.made_valid()))
    }

    /// Returns a `PrimitiveRef` of a cone using a radius check where a radius
    /// and origin are parameters instead of a bounding box.
    pub fn cone_with_radius(&self, origin: Vec3<i32>, radius: f32, height: f32) -> PrimitiveRef<'_, 'a, Cbn> {
        let min = origin - Vec2::broadcast(radius.round() as i32);
        let max = origin + Vec2::broadcast(radius.round() as i32).with_z(height.round() as i32);
        self.prim(Node::Cone(Aabb { min, max }))
    }

    /// Returns a `PrimitiveRef` of a 3-dimensional line segment with a provided
    /// radius.
    pub fn line(
        &self,
        a: Vec3<impl AsPrimitive<f32>>,
        b: Vec3<impl AsPrimitive<f32>>,
        radius: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        self.line_two_radius(a, b, radius, radius, 1.0)
    }

    /* /// Returns a `PrimitiveRef` that conforms to the provided sampling
    /// function.
    ///
    /// TODO: Cbn or Cbv?
    #[must_use]
    pub fn sampling<Kind>(&self, a: /*impl Into<Id<Primitive<'a>>>*/PrimitiveRef<'_, 'a, Kind>, f: &'a dyn Fn(Vec3<i32>) -> bool) -> PrimitiveRef<'_, 'a, /*Cbn*/Kind> {
        self.prim(Node::Sampling(a.into(), f))
    } */

    /// Returns a `PrimitiveRef` of a 3-dimensional line segment with two
    /// radius.
    pub fn line_two_radius(
        &self,
        a: Vec3<impl AsPrimitive<f32>>,
        b: Vec3<impl AsPrimitive<f32>>,
        r0: f32,
        r1: f32,
        z_scale: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let segment = LineSegment3 {
            start: a.as_(),
            end: b.as_(),
        };
        if segment.start.x == segment.end.x && segment.start.y == segment.end.y {
            if segment.start.z == segment.end.z {
                return self.prim(Node::Empty);
            } else {
                // TODO: Vertical plane (not really thanks to the radii, but should be).
            }
        }
        // TODO: Enforce radii and z_scale > 0.
        self.prim(Node::Segment {
            r0,
            r1,
            z_scale,
            segment,
        })
    }

    /// Returns a `PrimitiveRef` of a 3-dimensional line segment where the
    /// provided radius only affects the width of the shape. The height of
    /// the shape is determined by the `height` parameter. The height of the
    /// shape is extended upwards along the z axis from the line. The top and
    /// bottom of the shape are planar and parallel to each other and the line.
    pub fn segment_prism(
        &self,
        a: Vec3<impl AsPrimitive<f32>>,
        b: Vec3<impl AsPrimitive<f32>>,
        radius: f32,
        height: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let segment = LineSegment3 {
            start: a.as_().map(f32::round),
            end: b.as_().map(f32::round),
        };
        if segment.start.x == segment.end.x && segment.start.y == segment.end.y {
            // Shear is not defined with a vertical line.
            return self.prim(Node::Empty);
        }
        self.prim(Node::SegmentPrism {
            segment,
            radius,
            height,
        })
    }

    /// Returns a `PrimitiveRef` of a 3-dimensional cubic bezier curve by
    /// dividing the curve into line segments with one segment approximately
    /// every length of 5 blocks.
    pub fn cubic_bezier(
        &self,
        start: Vec3<impl AsPrimitive<f32>>,
        ctrl0: Vec3<impl AsPrimitive<f32>>,
        ctrl1: Vec3<impl AsPrimitive<f32>>,
        end: Vec3<impl AsPrimitive<f32>>,
        radius: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let bezier = CubicBezier3 {
            start: start.as_(),
            ctrl0: ctrl0.as_(),
            ctrl1: ctrl1.as_(),
            end: end.as_(),
        };
        let length = bezier.length_by_discretization(10);
        let num_segments = (0.2 * length).ceil() as u16;
        self.cubic_bezier_with_num_segments(bezier, radius, num_segments)
    }

    /// Returns a `PrimitiveRef` of a 3-dimensional cubic bezier curve by
    /// dividing the curve into `num_segments` line segments.
    pub fn cubic_bezier_with_num_segments(
        &self,
        bezier: CubicBezier3<f32>,
        radius: f32,
        num_segments: u16,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let range: Vec<_> = (0..=num_segments).collect();
        let bezier_prim = self.union_all(range.windows(2).map(|w| {
            let segment_start = bezier.evaluate(w[0] as f32 / num_segments as f32);
            let segment_end = bezier.evaluate(w[1] as f32 / num_segments as f32);
            self.line(segment_start, segment_end, radius).into()
        }));
        bezier_prim
    }

    /* /// Returns a `PrimitiveRef` of a 3-dimensional cubic bezier curve where the
    /// radius only governs the width of the curve. The height is governed
    /// by the `height` parameter where the shape extends upwards from the
    /// bezier curve by the value of `height`. The shape is constructed by
    /// dividing the curve into line segment prisms with one segment prism
    /// approximately every length of 5 blocks.
    pub fn cubic_bezier_prism(
        &self,
        start: Vec3<impl AsPrimitive<f32>>,
        ctrl0: Vec3<impl AsPrimitive<f32>>,
        ctrl1: Vec3<impl AsPrimitive<f32>>,
        end: Vec3<impl AsPrimitive<f32>>,
        radius: f32,
        height: f32,
    ) -> PrimitiveRef<'_, 'a> {
        let bezier = CubicBezier3 {
            start: start.as_(),
            ctrl0: ctrl0.as_(),
            ctrl1: ctrl1.as_(),
            end: end.as_(),
        };
        let length = bezier.length_by_discretization(10);
        let num_segments = (0.2 * length).ceil() as u16;
        self.cubic_bezier_prism_with_num_segments(bezier, radius, height, num_segments)
    }

    /// Returns a `PrimitiveRef` of a 3-dimensional cubic bezier curve where the
    /// radius only governs the width of the curve. The height is governed
    /// by the `height` parameter where the shape extends upwards from the
    /// bezier curve by the value of `height`. The shape is constructed by
    /// dividing the curve into `num_segments` line segment prisms.
    pub fn cubic_bezier_prism_with_num_segments(
        &self,
        bezier: CubicBezier3<f32>,
        radius: f32,
        height: f32,
        num_segments: u16,
    ) -> PrimitiveRef<'_, 'a> {
        let range: Vec<_> = (0..=num_segments).collect();
        let bezier_prim = Node::union_all(self, range.windows(2).map(|w| {
            let segment_start = bezier.evaluate(w[0] as f32 / num_segments as f32);
            let segment_end = bezier.evaluate(w[1] as f32 / num_segments as f32);
            self.segment_prism(segment_start, segment_end, radius, height).into()
        }));
        self.prim(bezier_prim)
    } */

    /// Returns a `PrimitiveRef` of a plane. The Aabr provides the bounds for
    /// the plane in the xy plane and the gradient determines its slope through
    /// the dot product. A gradient of <1.0, 0.0> creates a plane with a
    /// slope of 1.0 in the xz plane.
    pub fn plane(&self, aabr: Aabr<i32>, origin: Vec3<i32>, gradient: Vec2<f32>) -> PrimitiveRef<'_, 'a, Cbn> {
        /* // NOTE: A plane is now identical to a variant of a SegmentPrism.
        // NOTE: Slope is z/x.
        let extent = aabr.as_().size();
        let start = Vec3::new(aabr.min.x as f32, 0, 0);
        let end = Vec3::new(aabr.max.x as f32, 0, x_dist * gradient);
        self.segment_prism(
            origin + start,
            origin + end,
            // TODO: Make sure we're handling fractions properly?
            (aabr.max.y - aabr.min.y) as f32 / 2,
            1.0,
        );
    pub fn segment_prism(
        &self,
        a: Vec3<impl AsPrimitive<f32>>,
        b: Vec3<impl AsPrimitive<f32>>,
        radius: f32,
        height: f32,
    ) -> PrimitiveRef<'_, 'a, Cbn> {
        let segment = LineSegment3 {
            start: origin + Vec3::from(aabr.min).as_().map(f32::round),
            end: origin + aabr.max.with_z(aabr.max.x - aabr.min.x).as_().map(f32::round),
        };
        if segment.start.x == segment.end.x && segment.start.y == segment.end.y {
            // Shear is not defined with a vertical line.
            return self.prim(Node::Empty);
        }
        self.prim(Node::SegmentPrism {
            segment,
            radius,
            height,
        })
        let aabr = aabr.made_valid();
    } */
        self.prim(Node::Plane(aabr, origin, gradient))
    }

    /// Returns a `PrimitiveRef` of an Aabb with a slope cut into it. The
    /// `inset` governs the slope. The `dir` determines which direction the
    /// ramp points.
    pub fn ramp(&self, aabb: Aabb<i32>, inset: i32, dir: Dir) -> PrimitiveRef<'_, 'a, Cbn> {
        let aabb = aabb.made_valid();
        self.prim(Node::Ramp { aabb, inset, dir })
    }

    /// Returns a `PrimitiveRef` of a triangular prism with the base being
    /// vertical. A gable is a tent shape. The `inset` governs the slope of
    /// the gable. The `dir` determines which way the gable points.
    pub fn gable(&self, aabb: Aabb<i32>, inset: i32, dir: Dir) -> PrimitiveRef<'_, 'a, Red> {
        let aabb = aabb.made_valid();
        self.prim(Node::Gable { aabb, inset, dir })
    }

    /// Places a sprite at the provided location with the default rotation.
    pub fn sprite<F: Filler>(&self, pos: Vec3<i32>, sprite: SpriteKind, filler: &mut FillFn<'a, '_, F>) {
        self.aabb/*::<Cbn>*/(Aabb {
            min: pos,
            max: pos + 1,
        })
        .fill(filler.sprite(sprite), filler)
    }

    /// Places a sprite at the provided location with the provided orientation.
    pub fn rotated_sprite<F: Filler>(&self, pos: Vec3<i32>, sprite: SpriteKind, ori: u8, filler: &mut FillFn<'a, '_, F>) {
        self.aabb/*::<Cbn>*/(Aabb {
            min: pos,
            max: pos + 1,
        })
        .fill(filler.rotated_sprite(sprite, ori), filler)
    }

    /// Returns a `PrimitiveRef` of the largest pyramid with a slope of 1 that
    /// fits in the provided Aabb.
    pub fn pyramid(&self, aabb: Aabb<i32>) -> PrimitiveRef<'_, 'a, Cbn> {
        let inset = 0;
        let aabb = aabb.made_valid();
        self.prim(Node::Ramp {
            aabb,
            inset,
            dir: Dir::X,
        })
        .intersect(self.prim(Node::Ramp {
            aabb,
            inset,
            dir: Dir::NegX,
        }))
        .intersect(self.prim(Node::Ramp {
            aabb,
            inset,
            dir: Dir::Y,
        }))
        .intersect(self.prim(Node::Ramp {
            aabb,
            inset,
            dir: Dir::NegY,
        }))
    }

    #[must_use]
    /// Constructs a prefab structure at the current origin (I think).
    ///
    /// TODO: Cbn or Cbv?
    pub fn prefab(&self, prefab: &'static PrefabStructure) -> PrimitiveRef<'_, 'a, Cbn> {
        if prefab.get_bounds().size().reduce_min() == 0 {
            self.empty().as_kind()
        } else {
            self.prim(Node::Prefab(prefab.into()))
        }
    }

    /// Used to create a new `PrimitiveRef`. Requires the desired `Primitive<'a>` to
    /// be supplied.
    fn prim<'painter, Kind>(&'painter self, prim: Node<'a>) -> PrimitiveRef<'painter, 'a, Kind> {
        PrimitiveRef {
            id: self.prims.borrow_mut().insert(Primitive(prim)),
            painter: self,
            kind: core::marker::PhantomData,
        }
    }

    /// Returns a `PrimitiveRef` of an empty primitive. Useful when additional
    /// primitives are unioned within a loop.
    ///
    /// TODO: Cbn or Cbv?
    pub fn empty/*<Kind>*/(&self) -> PrimitiveRef<'_, 'a, /*Cbn*//*Kind*/Red> { self.prim(Node::Empty) }

    /// Fills the supplied primitive with the provided `Fill`.
    pub fn fill<F: Filler>(&self, prim: impl Into<Id<Primitive<'a>>>, fill: impl Fill + Copy, filler: &mut FillFn<'a, '_, F>) {
        // NOTE: Historically it was beneficial to split out top-level unions here in order to make
        // the call tree shorter, but this is not really necessary when drawing immediately.  In
        // the future, to get a better version of this effect (once we switch to linear-by-default
        // PrimitiveRefs), we may change UnionAll to use a Vec or SmallVec representation, and/or
        // some sort of Vec-in-arena approach.
        return filler.fill(self, prim.into(), fill);
        /* let prim = prim.into();
        let prims = self.prims.borrow();
        fn fill_inner<'a, F: Filler>(prims: &Store<Primitive<'a>>, mut prim: Id<Primitive<'a>>, fill: Fill<'_>, filler: &mut FillFn<'a, '_, F>) {
            loop {
                match prims[prim].0 {
                    Node::Union(a, b) => {
                        fill_inner(prims, a, fill, filler);
                        prim = b;
                    },
                    Node::UnionAll(xs) => return xs.into_iter().for_each(|x| fill_inner(prims, *x, fill, filler)),
                    _ => /* self.fills.borrow_mut().push((prim, fill)), */return filler.fill(prims, prim, fill),
                }
            }
        }
        // return filler.fill(&prims, prim.into(), fill);
        fill_inner(&prims, prim, fill, filler); */
    }

    /* /// The area that the canvas is currently rendering.
    pub fn render_aabr(&self) -> Aabr<i32> { self.render_area }

    /// Spawns an entity if it is in the render_aabr, otherwise does nothing.
    pub fn spawn(&self, entity: EntityInfo) {
        if self.render_area.contains_point(entity.pos.xy().as_()) {
            self.entities.borrow_mut().push(entity)
        }
    } */
}

/// Type for codata (computations)
pub trait Negative {}

/// Type for data (values)
pub trait Positive {}

/// Types that can go either way (e.g. AABB, empty).
pub trait Neutral : Negative + Positive {}

/// Cbn primitive refs are linear.
#[derive(Clone, Copy)]
pub struct Cbn;

impl Negative for Cbn {}

/// Cbv primitive refs are copyable.
#[derive(Clone, Copy)]
pub struct Cbv/*<'a> {
    vol: &'a [Cell<u8>],
}*/;

impl Positive for Cbv {}

/// Fully reduced values are copyable and don't have any content.
#[derive(Clone, Copy)]
pub struct Red;

impl Positive for Red {}

impl Negative for Red {}

impl Neutral for Red {}

#[derive(Copy, Clone)]
/// PrimitiveRefs inherit copyability of their Kind.
pub struct PrimitiveRef<'painter, 'a, Kind> {
    id: Id<Primitive<'a>>,
    painter: &'painter Painter<'a>,
    kind: /*Kind*/core::marker::PhantomData<Kind>,
}

/* /// An Aabb is considered freely copyable (for the time being) since it is considered a "true"
/// value: there is no reduction required to determine which voxels are covered, so there's no need
/// to share the redex during call-by-need.  It also enjoys some automatic optimizations, such as
/// in-place intersections with other AABBs, immediately applied matrix transformations, and the
/// ability to "force" other primitives.
#[derive(Copy, Clone)]
pub struct PrimitiveAabb<'painter, 'a> {
    aabb: Aabb<i32>,
    painter: &'painter Painter<'a>,
}

/// A call-by-name primitive is able to have its reduction delayed until its evaluation is forced.
/// Generally speaking, it is desirable to delay forcing only until the smallest AABB known to
/// cover the primitive.  If the primitive is shared, we currently conservatively assume that the
/// whole primitive must be sampled and therefore force it immediately, so it's advised to *not*
/// share primitives in cases where, e.g., they are intersected with a disjoint union.  Thus,
/// call-by-name primitives in our formalism are linear and call-by-value primitives are nonlinear.

/// Copyable version of a primitive intended for reuse.  Since it knows it's going to be reused, it
/// implements call-by-need (rendering just once to a preallocated bitmap in integer-rotated world
/// space, which makes it efficient to sample and takes up much less space than a fill would).  In
/// order to ensure that rotation order doesn't hurt query performance, the bitmap uses the format:
/// 
/// x₀y₀z₀ x₀y₀z₁ x₀y₁z₀ x₀y₁z₁ x₁y₀z₀ x₁y₀z₁ x₁y₁z₀ x₁y₁z₁
///
/// This way, if the source is rendering using a 2×2 simd square in the xy plane, and the bitmap is
/// rotated so that xz or yz maps onto xy in world space, the square will always fall into a single
/// byte.
#[derive(Copy, Clone)]
pub struct PrimitiveRefExp<'painter, 'a> {
    vol: &'a [Cell<u8>],
    id: Id<Primitive<'a>>,
    painter: &'painter Painter<'a>,
} */

impl<'painter, 'a, Kind> From<PrimitiveRef<'painter, 'a, Kind>> for Id<Primitive<'a>> {
    fn from(r: PrimitiveRef<'painter, 'a, Kind>) -> Self { r.id }
}

impl<'painter, 'a, Kind: Neutral> PrimitiveRef<'painter, 'a, Kind> {
    /// Convert neutral primitives to any kind.
    pub fn as_kind<NewKind>(self) -> PrimitiveRef<'painter, 'a, NewKind> {
        PrimitiveRef {
            id: self.id,
            painter: self.painter,
            kind: core::marker::PhantomData,
        }
    }
}

impl<'painter, 'a, Kind> PrimitiveRef<'painter, 'a, Kind> {
    /// Joins two primitives together by returning the total of the blocks of
    /// both primitives. In boolean logic this is an `OR` operation.
    #[must_use]
    pub fn union(self, other: /*impl Into<Id<Primitive<'a>>>*/Self) -> Self {
        let (b, a) = self.painter.order_by_depth(self.id, other.into());
        self.painter.prim(Node::Union(a, b))
    }

    /// Joins two primitives together by returning only overlapping blocks. In
    /// boolean logic this is an `AND` operation.
    #[must_use]
    pub fn intersect(self, other: /*impl Into<Id<Primitive<'a>>>*/Self) -> Self {
        let (a, b) = self.painter.order_by_depth(self.id, other.into());
        self.painter.prim(Node::Intersect([a.into(), b.into()]))
    }

    /// Fills the primitive with `fill` and paints it into the world.
    pub fn fill<F: Filler>(self, fill: /*Fill<'_>*/impl Fill + Copy, filler: &mut FillFn<'a, '_, F>) {
        self.painter.fill(self, fill, filler);
    }

    /// Fills the primitive with empty blocks. This will subtract any
    /// blocks in the world that inhabit the same positions as the blocks in
    /// this primitive.
    pub fn clear<F: Filler>(self, filler: &mut FillFn<'a, '_, F>) {
        self.painter.fill(self, /*Fill::Block*/filler.block(Block::empty()), filler);
    }

    /// Rotates a primitive about it's own's bounds minimum point,
    #[must_use]
    pub fn rotate_about_min(
        self,
        /* cache: &mut BoundsMap,
         * */
        mat: Mat3<i32>
    ) -> Self {
        let mut cache = self.painter.bounds_cache.borrow_mut();
        let point = Painter::get_bounds(&mut *cache, &self.painter.prims.borrow(), self.id).min;
        self.rotate_about(mat, point)
    }
}

impl<'painter, 'a, Kind: Positive> PrimitiveRef<'painter, 'a, Kind> {
    /// Subtracts the blocks of the `other` primitive from `self`. In boolean
    /// logic this is a `NOT` operation.
    #[must_use]
    pub fn without<Ret: Negative>(self, other: /*impl Into<Id<Primitive<'a>>>*/PrimitiveRef<'painter, 'a, Ret>) -> Self {
        Node::without(self.painter, self, other)
    }
}

impl<'painter, 'a, Kind: Copy> PrimitiveRef<'painter, 'a, Kind> {
    pub fn repeat(self, offset: Vec3<i32>, count: /*u32*/u16) -> Self {
        /* let count = count - 1;
        let aabb = Self::get_bounds(cache, tree, *prim);
        let aabb_corner = {
            let min_red = aabb.min.map2(*offset, |a, b| if b < 0 { 0 } else { a });
            let max_red = aabb.max.map2(*offset, |a, b| if b < 0 { a } else { 0 });
            min_red + max_red
        };
        let diff = pos - aabb_corner;
        let min = diff
            .map2(*offset, |a, b| if b == 0 { i32::MAX } else { a / b })
            .reduce_min()
            .clamp(0, count as i32);
        let pos = pos - offset * min;
        Self::contains_at::<SubAabr>(cache, tree, *prim, pos)
        self.painter.prim(Node::repeat(self, offset, count)) */
        self.painter.union_all(
            (0..count).map(|i| self.translate(offset * i32::from(i)).into())
        )
    }
}

/// A trait to more easily manipulate groups of primitives.
pub trait PrimitiveTransform {
    /// Translates the primitive along the vector `trans`.
    #[must_use]
    fn translate(self, trans: Vec3<i32>) -> Self;
    /// Rotates the primitive about the given point of the primitive by
    /// multiplying each block position by the provided rotation matrix.
    #[must_use]
    fn rotate_about(self, rot: Mat3<i32>, point: Vec3<impl AsPrimitive<f32>>) -> Self;
    /* /// Scales the primitive along each axis by the x, y, and z components of
    /// the `scale` vector respectively.
    #[must_use]
    fn scale(self, scale: Vec3<f32>) -> Self; */
    /* /// Returns a `PrimitiveRef` of the primitive in addition to the same
    /// primitive translated by `offset` and repeated `count` times, each time
    /// translated by an additional offset.
    #[must_use]
    fn repeat(self, offset: Vec3<i32>, count: /*u32*/u16) -> Self; */
}

impl<'painter, 'a, Kind> PrimitiveTransform for PrimitiveRef<'painter, 'a, Kind> {
    fn translate(self, trans: Vec3<i32>) -> Self {
        self.painter.prim(Node::translate(self, trans))
    }

    fn rotate_about(self, rot: Mat3<i32>, point: Vec3<impl AsPrimitive<f32>>) -> Self {
        self.painter.prim(Node::rotate_about(self, rot, point))
    }

    /* fn scale(self, scale: Vec3<f32>) -> Self { self.painter.prim(Node::scale(self, scale)) } */
}

impl<'painter, 'a, const N: usize, Kind> PrimitiveTransform for [PrimitiveRef<'painter, 'a, Kind>; N] {
    fn translate(mut self, trans: Vec3<i32>) -> Self {
        for prim in &mut self {
            *prim = prim.painter.prim(Node::translate(prim.id, trans));
        }
        self
    }

    fn rotate_about(mut self, rot: Mat3<i32>, point: Vec3<impl AsPrimitive<f32>>) -> Self {
        for prim in &mut self {
            *prim = prim.painter.prim(Node::rotate_about(prim.id, rot, point));
        }
        self
    }

    /* fn scale(mut self, scale: Vec3<f32>) -> Self {
        for prim in &mut self {
            *prim = prim.scale(scale);
        }
        self
    } */

    /* fn repeat(mut self, offset: Vec3<i32>, count: /*u32*/u16) -> Self {
        for prim in &mut self {
            *prim = prim.repeat(offset, count);
        }
        self
    } */
}

/* pub trait PrimitiveGroupFill<'a, const N: usize> {
    fn fill_many(self, fills: [Fill<'a>; N]);
} */

/* impl<'painter, 'a, const N: usize, Kind> PrimitiveGroupFill<'a, N> for [PrimitiveRef<'painter, 'a, Kind>; N] {
    fn fill_many(self, fills: [Fill<'a>; N]) {
        core::iter::zip(self, fills).for_each(|(primitive, fill)| {
            primitive.fill(fill);
        });
    }
} */

pub trait Structure</*'a, */F> {
    fn render<'b>(
        &self,
        site: &Site,
        land: Land,
        painter: &Painter<'b>,
        filler: &mut FillFn<'b, '_, F>,
    );/* where F: FillFn<'a>;*/
}

/// Generate a primitive tree and fills for this structure, then render them.
pub fn render_collect<'b, 'c, F: Filler + 'c, Render: FnOnce(&Painter<'b>, &mut FillFn<'b, '_, F>)>(
    arena: &'b bumpalo::Bump,
    canvas_info: CanvasInfo<'c>,
    render_area: Aabr<i32>,
    filler: /*impl FnOnce(&'b mut Canvas<'a>) -> &'b mut F*/&'b mut F,
    render: Render,
)/* -> (
    Store<Primitive<'a>>,
    Vec<(Id<Primitive<'a>>, Fill<'a>)>,
    Vec<EntityInfo>,
)*/ {
    /* let canvas_info = canvas.info(); */
    let painter = Painter/*::<N>*/::new(/*canvas.*/arena/*, render_area*/);

    // let bounds_cache = HashMap::default();
    // let arena = canvas.arena;

    /* let filler = filler(canvas); */
    let mut fill_fn: FillFn<'b, 'c, _> = FillFn {
        // marker: core::marker::PhantomData,
        // arena,
        // bounds_cache,
        filler,
        render_area,
        canvas_info,
        // entities: Vec::new(),
    };
    /* let filler = move |prim_tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>, fill: Fill<'a>| {
        fill.sample_at(arena, &mut bounds_cache, prim, prim_tree, render_area, &info, filler)
    }; */

    render(&painter, &mut fill_fn);
    /* (
        painter.prims.into_inner(),
        painter.fills.into_inner(),
        painter.entities.into_inner(),
    )*/
    /* self.entities */
}

impl<'a, 'b, 'c, F: Filler + 'b> dyn Structure</*'a, */F>/* + 'b*/ {
    /// TODO: Land is the only part of info we really need, and is the part that *should* be passed
    /// in as it needs to be passed to the render function (and is totally divored from the
    /// rasterization strategy).  The rest of CanvasInfo is only used in one place: a prefab hack.
    /// Ideally, we should be able to fix the prefab hack and remove CanvasInfo as an argument,
    /// which hopefully would both simplify the implementation and avoid any reference to Canvas in
    /// the implementation of this structure (making it more abstract from the perspective of the
    /// filler).
    pub fn render_collect(
        &'a self,
        site: &'a Site,
        arena: &'b bumpalo::Bump,
        canvas_info: CanvasInfo<'c>,
        render_area: Aabr<i32>,
        filler: /*impl FnOnce(&'b mut Canvas<'a>) -> &'b mut F*/&'b mut F,
    )/* -> (
        Store<Primitive<'a>>,
        Vec<(Id<Primitive<'a>>, Fill<'a>)>,
        Vec<EntityInfo>,
    )*/ {
        /* /* let canvas_info = canvas.info(); */
        let painter = Painter/*::<N>*/::new(/*canvas.*/arena/*, render_area*/);

        // let bounds_cache = HashMap::default();
        // let arena = canvas.arena;

        /* let filler = filler(canvas); */
        let mut fill_fn: FillFn<'b, 'c, _> = FillFn {
            // marker: core::marker::PhantomData,
            // arena,
            // bounds_cache,
            filler,
            render_area,
            canvas_info,
            // entities: Vec::new(),
        };
        /* let filler = move |prim_tree: &Store<Primitive<'a>>, prim: Id<Primitive<'a>>, fill: Fill<'a>| {
            fill.sample_at(arena, &mut bounds_cache, prim, prim_tree, render_area, &info, filler)
        }; */ */

        render_collect(arena, canvas_info, render_area, filler, move |painter, fill_fn| {
            self.render(site, fill_fn.canvas_info.land(), &painter, fill_fn);
        });

        /* self.render(site, fill_fn.canvas_info.land(), &painter, &mut fill_fn); */
        /* (
            painter.prims.into_inner(),
            painter.fills.into_inner(),
            painter.entities.into_inner(),
        )*/
        /* self.entities */
    }
}
/// Extend a 2d AABR to a 3d AABB
pub fn aabr_with_z<T>(aabr: Aabr<T>, z: std::ops::Range<T>) -> Aabb<T> {
    Aabb {
        min: aabr.min.with_z(z.start),
        max: aabr.max.with_z(z.end),
    }
}

#[allow(dead_code)]
/// Just the corners of an AABB, good for outlining stuff when debugging
pub fn aabb_corners<'a, F: FnMut(Primitive<'a>) -> Id<Primitive<'a>>>(
    prim: &mut F,
    aabb: Aabb<i32>,
) -> Id<Primitive<'a>> {
    let f = |prim: &mut F, ret, vec| {
        let sub = prim(Primitive(Node::Aabb(Aabb {
            min: aabb.min + vec,
            max: aabb.max - vec,
        })));
        // Don't care about order for debugging.
        prim(Primitive(Node::Without(ret, sub, false)))
    };
    let mut ret = prim(Primitive(Node::Aabb(aabb)));
    ret = f(prim, ret, Vec3::new(1, 0, 0));
    ret = f(prim, ret, Vec3::new(0, 1, 0));
    ret = f(prim, ret, Vec3::new(0, 0, 1));
    ret
}
