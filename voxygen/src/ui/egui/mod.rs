use egui_winit_platform::{Platform, PlatformDescriptor};
use crate::window::Window;
use egui::FontDefinitions;
use client::Client;
use crate::hud::DebugInfo;

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
        self.platform.begin_frame();

        egui::Window::new("Test Window")
            .default_width(200.0)
            .default_height(200.0)
            .show(&self.platform.context(), |ui| {
                ui.heading("My egui Application");
                ui.horizontal(|ui| {
                    ui.label("Your name: ");
                    ui.text_edit_singleline(&mut "hello".to_owned());
                });
                ui.add(egui::Slider::new(&mut 99, 0..=120).text("age"));
                if ui.button("Click each year").clicked() {
                    println!("button clicked");
                }
                ui.label(format!("Hello '{}', age {}", "Ben", 99));
            });
    }
}