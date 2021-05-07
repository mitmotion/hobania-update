use crate::window::Window;
use client::Client;
use common::debug_info::DebugInfo;
use egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};
use voxygen_egui::EguiInnerState;

pub struct EguiState {
    pub platform: Platform,
    egui_inner_state: EguiInnerState,
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
            egui_inner_state: EguiInnerState {
                read_ecs: false,
                selected_entity_id: 0,
                max_entity_distance: 17000.0,
            },
        }
    }

    pub fn maintain(&mut self, client: &Client, debug_info: &Option<DebugInfo>) {
        voxygen_egui::maintain(
            &mut self.platform,
            &mut self.egui_inner_state,
            client,
            debug_info,
        );
    }
}
