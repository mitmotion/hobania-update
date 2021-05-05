
use std::time::Duration;
use crate::comp;

pub struct DebugInfo {
    pub tps: f64,
    pub frame_time: Duration,
    pub ping_ms: f64,
    pub coordinates: Option<comp::Pos>,
    pub velocity: Option<comp::Vel>,
    pub ori: Option<comp::Ori>,
    pub num_chunks: u32,
    pub num_lights: u32,
    pub num_visible_chunks: u32,
    pub num_shadow_chunks: u32,
    pub num_figures: u32,
    pub num_figures_visible: u32,
    pub num_particles: u32,
    pub num_particles_visible: u32,
}