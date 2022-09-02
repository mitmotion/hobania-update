use super::super::{AaMode, GlobalsLayouts, Vertex as VertexTrait};
use bytemuck::{Pod, Zeroable};
use std::mem;
use vek::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    // ____BBBBBBBBGGGGGGGGRRRRRRRR
    // col: u32 = "v_col",
    // ...AANNN
    // A = AO
    // N = Normal
    norm_ao: u32,
}

impl Vertex {
    #[allow(clippy::collapsible_else_if)]
    pub fn new(pos: Vec3<f32>, norm: Vec3<f32>) -> Self {
        let norm_bits = if norm.x != 0.0 {
            if norm.x < 0.0 { 0 } else { 1 }
        } else if norm.y != 0.0 {
            if norm.y < 0.0 { 2 } else { 3 }
        } else {
            if norm.z < 0.0 { 4 } else { 5 }
        };

        Self {
            pos: pos.into_array(),
            norm_ao: norm_bits,
        }
    }

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32];
        wgpu::VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &ATTRIBUTES,
        }
    }
}

impl VertexTrait for Vertex {
    const QUADS_INDEX: Option<wgpu::IndexFormat> = Some(wgpu::IndexFormat::Uint16);
    const STRIDE: wgpu::BufferAddress = mem::size_of::<Self>() as wgpu::BufferAddress;
}

#[derive(Copy, Clone)]
pub enum ParticleMode {
    CampfireSmoke = 0,
    CampfireFire = 1,
    GunPowderSpark = 2,
    Shrapnel = 3,
    FireworkBlue = 4,
    FireworkGreen = 5,
    FireworkPurple = 6,
    FireworkRed = 7,
    FireworkWhite = 8,
    FireworkYellow = 9,
    Leaf = 10,
    Firefly = 11,
    Bee = 12,
    GroundShockwave = 13,
    EnergyHealing = 14,
    EnergyNature = 15,
    FlameThrower = 16,
    FireShockwave = 17,
    FireBowl = 18,
    Snow = 19,
    Explosion = 20,
    Ice = 21,
    LifestealBeam = 22,
    CultistFlame = 23,
    StaticSmoke = 24,
    Blood = 25,
    Enraged = 26,
    BigShrapnel = 27,
    Laser = 28,
    Bubbles = 29,
    Water = 30,
    IceSpikes = 31,
    Drip = 32,
    Tornado = 33,
    Death = 34,
    EnergyBuffing = 35,
    WebStrand = 36,
    BlackSmoke = 37,
    Lightning = 38,
    BarrelOrgan = 39,
}

impl ParticleMode {
    pub fn into_uint(self) -> u32 { self as u32 }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Zeroable, Pod)]
pub struct Instance {
    // created_at time, so we can calculate time relativity, needed for relative animation.
    // can save 32 bits per instance, for particles that are not relatively animated.
    inst_time: f32,

    // The lifespan in seconds of the particle
    inst_lifespan: f32,

    // a seed value for randomness
    // can save 32 bits per instance, for particles that don't need randomness/uniqueness.
    inst_entropy: f32,

    // modes should probably be seperate shaders, as a part of scaling and optimisation efforts.
    // can save 32 bits per instance, and have cleaner tailor made code.
    inst_mode: i32,

    // A direction for particles to move in
    inst_dir: [f32; 3],

    // a triangle is: f32 x 3 x 3 x 1  = 288 bits
    // a quad is:     f32 x 3 x 3 x 2  = 576 bits
    // a cube is:     f32 x 3 x 3 x 12 = 3456 bits
    // this vec is:   f32 x 3 x 1 x 1  = 96 bits (per instance!)
    // consider using a throw-away mesh and
    // positioning the vertex verticies instead,
    // if we have:
    // - a triangle mesh, and 3 or more instances.
    // - a quad mesh, and 6 or more instances.
    // - a cube mesh, and 36 or more instances.
    inst_pos: [f32; 3],
}

impl Instance {
    pub fn new(
        inst_time: f64,
        lifespan: f32,
        inst_mode: ParticleMode,
        inst_pos: Vec3<f32>,
    ) -> Self {
        use rand::Rng;
        Self {
            inst_time: inst_time as f32,
            inst_lifespan: lifespan,
            inst_entropy: rand::thread_rng().gen(),
            inst_mode: inst_mode as i32,
            inst_pos: inst_pos.into_array(),
            inst_dir: [0.0, 0.0, 0.0],
        }
    }

    pub fn new_directed(
        inst_time: f64,
        lifespan: f32,
        inst_mode: ParticleMode,
        inst_pos: Vec3<f32>,
        inst_pos2: Vec3<f32>,
    ) -> Self {
        use rand::Rng;
        Self {
            inst_time: inst_time as f32,
            inst_lifespan: lifespan,
            inst_entropy: rand::thread_rng().gen(),
            inst_mode: inst_mode as i32,
            inst_pos: inst_pos.into_array(),
            inst_dir: (inst_pos2 - inst_pos).into_array(),
        }
    }

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        const ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![2 => Float32, 3 => Float32, 4 => Float32, 5 => Sint32, 6 => Float32x3, 7 => Float32x3];
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Instance,
            attributes: &ATTRIBUTES,
        }
    }
}

impl Default for Instance {
    fn default() -> Self { Self::new(0.0, 0.0, ParticleMode::CampfireSmoke, Vec3::zero()) }
}

pub struct ParticlePipeline {
    pub pipeline: wgpu::RenderPipeline,
}

impl ParticlePipeline {
    pub fn new(
        device: &wgpu::Device,
        vs_module: &wgpu::ShaderModule,
        fs_module: &wgpu::ShaderModule,
        global_layout: &GlobalsLayouts,
        aa_mode: AaMode,
    ) -> Self {
        common_base::span!(_guard, "ParticlePipeline::new");
        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle pipeline layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[&global_layout.globals, &global_layout.shadow_textures],
            });

        let samples = match aa_mode {
            AaMode::None | AaMode::Fxaa => 1,
            AaMode::MsaaX4 => 4,
            AaMode::MsaaX8 => 8,
            AaMode::MsaaX16 => 16,
        };

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: vs_module,
                entry_point: "main",
                buffers: &[Vertex::desc(), Instance::desc()],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                clamp_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::GreaterEqual,
                stencil: wgpu::StencilState {
                    front: wgpu::StencilFaceState::IGNORE,
                    back: wgpu::StencilFaceState::IGNORE,
                    read_mask: !0,
                    write_mask: !0,
                },
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState {
                count: samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: fs_module,
                entry_point: "main",
                targets: &[wgpu::ColorTargetState {
                    // TODO: use a constant and/or pass in this format on pipeline construction
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrite::ALL,
                }],
            }),
        });

        Self {
            pipeline: render_pipeline,
        }
    }
}
