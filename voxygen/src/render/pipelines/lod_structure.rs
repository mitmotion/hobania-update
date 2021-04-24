use super::{
    super::{Pipeline, TgtColorFmt, TgtDepthStencilFmt},
    shadow, terrain, Globals, Light, Shadow,
};
use core::fmt;
use gfx::{
    self, gfx_constant_struct_meta, gfx_defines, gfx_impl_struct_meta, gfx_pipeline,
    gfx_pipeline_inner, gfx_vertex_struct_meta, state::ColorMask,
};
use vek::*;

pub use super::sprite::Vertex;

gfx_defines! {
    /*
    vertex Vertex {
        pos: [f32; 3] = "v_pos",
        // Because we try to restrict terrain sprite data to a 128×128 block
        // we need an offset into the texture atlas.
        atlas_pos: u32 = "v_atlas_pos",
        // ____BBBBBBBBGGGGGGGGRRRRRRRR
        // col: u32 = "v_col",
        // ...AANNN
        // A = AO
        // N = Normal
        norm_ao: u32 = "v_norm_ao",
    }
    */

    constant Locals {
        nope: u32 = "nope",
    }

    vertex/*constant*/ Instance {
        pos: [f32; 3] = "inst_pos",
    }

    pipeline pipe {
        vbuf: gfx::VertexBuffer<Vertex> = (),
        ibuf: gfx::InstanceBuffer<Instance> = (),
        col_lights: gfx::TextureSampler<[f32; 4]> = "t_col_light",

        locals: gfx::ConstantBuffer<Locals> = "u_locals",
        globals: gfx::ConstantBuffer<Globals> = "u_globals",
        lights: gfx::ConstantBuffer<Light> = "u_lights",
        shadows: gfx::ConstantBuffer<Shadow> = "u_shadows",

        point_shadow_maps: gfx::TextureSampler<f32> = "t_point_shadow_maps",
        directed_shadow_maps: gfx::TextureSampler<f32> = "t_directed_shadow_maps",

        alt: gfx::TextureSampler<[f32; 2]> = "t_alt",
        horizon: gfx::TextureSampler<[f32; 4]> = "t_horizon",

        noise: gfx::TextureSampler<f32> = "t_noise",

        // Shadow stuff
        light_shadows: gfx::ConstantBuffer<shadow::Locals> = "u_light_shadows",

        tgt_color: gfx::BlendTarget<TgtColorFmt> = ("tgt_color", ColorMask::all(), gfx::preset::blend::ALPHA),
        tgt_depth_stencil: gfx::DepthTarget<TgtDepthStencilFmt> = gfx::preset::depth::LESS_EQUAL_WRITE,
        // tgt_depth_stencil: gfx::DepthStencilTarget<TgtDepthStencilFmt> = (gfx::preset::depth::LESS_EQUAL_WRITE,Stencil::new(Comparison::Always,0xff,(StencilOp::Keep,StencilOp::Keep,StencilOp::Keep))),
    }
}

/*
impl fmt::Display for Vertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Vertex")
            .field("pos", &Vec3::<f32>::from(self.pos))
            .field(
                "atlas_pos",
                &Vec2::new(self.atlas_pos & 0xFFFF, (self.atlas_pos >> 16) & 0xFFFF),
            )
            .field("norm_ao", &self.norm_ao)
            .finish()
    }
}

impl Vertex {
    // NOTE: Limit to 16 (x) × 16 (y) × 32 (z).
    #[allow(clippy::collapsible_else_if)]
    pub fn new(
        atlas_pos: Vec2<u16>,
        pos: Vec3<f32>,
        norm: Vec3<f32>, /* , col: Rgb<f32>, ao: f32 */
    ) -> Self {
        let norm_bits = if norm.x != 0.0 {
            if norm.x < 0.0 { 0 } else { 1 }
        } else if norm.y != 0.0 {
            if norm.y < 0.0 { 2 } else { 3 }
        } else {
            if norm.z < 0.0 { 4 } else { 5 }
        };

        Self {
            // pos_norm: ((pos.x as u32) & 0x003F)
            //     | ((pos.y as u32) & 0x003F) << 6
            //     | (((pos + EXTRA_NEG_Z).z.max(0.0).min((1 << 16) as f32) as u32) & 0xFFFF) << 12
            //     | if meta { 1 } else { 0 } << 28
            //     | (norm_bits & 0x7) << 29,
            pos: pos.into_array(),
            atlas_pos: ((atlas_pos.x as u32) & 0xFFFF) | ((atlas_pos.y as u32) & 0xFFFF) << 16,
            norm_ao: norm_bits,
        }
    }
}
*/

impl Instance {
    pub fn new(
        pos: Vec3<f32>,
    ) -> Self {
        Self {
            pos: pos.into_array(),
        }
    }
}

impl Default for Instance {
    fn default() -> Self { Self::new(Vec3::zero()) }
}

impl Default for Locals {
    fn default() -> Self { Self::new() }
}

impl Locals {
    pub fn new() -> Self {
        Self {
            nope: 0,
        }
    }
}

pub struct LodStructurePipeline;

impl Pipeline for LodStructurePipeline {
    type Vertex = Vertex;
}
