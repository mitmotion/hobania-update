use crate::{
    render::pipelines::particle::ParticleMode,
    scene::{DebugShape, DebugShapeId, Scene},
    window::Window,
};
use client::Client;
use common::comp::Pos;
use egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};
use specs::WorldExt;
use vek::Vec3;
use voxygen_egui::{
    EguiAction, EguiDebugInfo,
    EguiDebugShapeAction::{AddCylinder, RemoveShape, SetPosAndColor},
    EguiInnerState, EguiWindows,
};

pub struct EguiState {
    pub platform: Platform,
    egui_inner_state: EguiInnerState,
    egui_windows: EguiWindows,
    new_debug_shape_id: Option<u64>,
}

impl EguiState {
    pub fn new(window: &Window) -> Self {
        let platform = Platform::new(PlatformDescriptor {
            physical_width: window.window().inner_size().width as u32,
            physical_height: window.window().inner_size().height as u32,
            scale_factor: window.scale_factor(),
            font_definitions: FontDefinitions::default(),
            style: Default::default(),
        });

        Self {
            platform,
            egui_inner_state: EguiInnerState::default(),
            egui_windows: EguiWindows::default(),
            new_debug_shape_id: None,
        }
    }

    pub fn maintain(
        &mut self,
        client: &Client,
        scene: &mut Scene,
        debug_info: Option<EguiDebugInfo>,
    ) {
        let egui_actions = voxygen_egui::maintain(
            &mut self.platform,
            &mut self.egui_inner_state,
            &mut self.egui_windows,
            client,
            debug_info,
            self.new_debug_shape_id.take(),
        );

        egui_actions.actions.iter().for_each(|action| match action {
            EguiAction::DebugShape(AddCylinder { height, radius }) => {
                let shape_id = scene.debug.add_shape(DebugShape::Cylinder {
                    height: *height,
                    radius: *radius,
                });
                self.new_debug_shape_id = Some(shape_id.0);
            },
            EguiAction::DebugShape(RemoveShape(debug_shape_id)) => {
                scene.debug.remove_shape(DebugShapeId(*debug_shape_id));
            },
            EguiAction::DebugShape(SetPosAndColor { id, pos, color }) => {
                scene
                    .debug
                    .set_pos_and_color(DebugShapeId(*id), *pos, *color);
            },
            EguiAction::AddParticles {
                lifespan,
                mode,
                quantity,
                offset,
                color,
            } => {
                let time = client.state().get_time();

                let player_pos = client
                    .state()
                    .ecs()
                    .read_storage::<Pos>()
                    .get(client.entity())
                    .map_or(Vec3::zero(), |pos| {
                        Vec3::new(pos.0.x, pos.0.y, pos.0.z) + offset
                    });

                let particle_mode: ParticleMode =
                    num::FromPrimitive::from_u32(*mode).unwrap_or(ParticleMode::Firefly);
                scene.particle_mgr_mut().add_debug_particles(
                    *quantity,
                    *lifespan,
                    time,
                    particle_mode,
                    player_pos,
                    color,
                );
            },
        })
    }
}
