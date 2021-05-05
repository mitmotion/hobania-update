use egui_winit_platform::{Platform, PlatformDescriptor};
use crate::window::Window;
use egui::FontDefinitions;
use client::Client;
use common::debug_info::DebugInfo;

pub struct EguiState {
    pub platform: Platform,
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
            platform
        }
    }

    pub fn maintain(&mut self,
                    client: &Client,
                    debug_info: &Option<DebugInfo>) {
        voxygen_egui::maintain(&mut self.platform, client, debug_info);
    }
}