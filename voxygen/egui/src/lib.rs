use client::Client;
use egui_winit_platform::Platform;
use common::debug_info::DebugInfo;
#[cfg(all(feature = "be-dyn-lib", feature = "use-dyn-lib"))]
compile_error!("Can't use both \"be-dyn-lib\" and \"use-dyn-lib\" features at once");

#[cfg(feature = "use-dyn-lib")] pub mod dyn_lib;

#[cfg(feature = "use-dyn-lib")]
pub use dyn_lib::init;
use std::ffi::CStr;

#[cfg(feature = "use-dyn-lib")]
const MAINTAIN_EGUI_FN: &'static [u8] = b"maintain_egui_inner\0";

pub fn maintain(platform: &mut Platform,
                client: &Client,
                debug_info: &Option<DebugInfo>) {
    #[cfg(not(feature = "use-dyn-lib"))]
    {
        maintain_egui_inner(platform, client, debug_info);
    }

    #[cfg(feature = "use-dyn-lib")]
    {
        let lock = dyn_lib::LIB.lock().unwrap();
        let lib = &lock.as_ref().unwrap().lib;

        let maintain_fn: libloading::Symbol<
            fn(
                &mut Platform,
                &Client,
                &Option<DebugInfo>,
            )
        > = unsafe {
            //let start = std::time::Instant::now();
            // Overhead of 0.5-5 us (could use hashmap to mitigate if this is an issue)
            let f = lib.get(MAINTAIN_EGUI_FN);
            //println!("{}", start.elapsed().as_nanos());
            f
        }
            .unwrap_or_else(|e| {
                panic!(
                    "Trying to use: {} but had error: {:?}",
                    CStr::from_bytes_with_nul(MAINTAIN_EGUI_FN)
                        .map(CStr::to_str)
                        .unwrap()
                        .unwrap(),
                    e
                )
            });

        maintain_fn(platform, client, debug_info);
    }
}

#[cfg_attr(feature = "be-dyn-lib", export_name = "maintain_egui_inner")]
pub fn maintain_egui_inner(platform: &mut Platform,
                client: &Client,
                debug_info: &Option<DebugInfo>) {
    platform.begin_frame();

    egui::Window::new("Test Window X")
        .default_width(200.0)
        .default_height(200.0)
        .show(&platform.context(), |ui| {
            ui.heading("My egui Application z");
            ui.horizontal(|ui| {
                ui.label(format!("Ping: {}", debug_info.as_ref().map_or(0.0, |x| x.ping_ms)));
                ui.text_edit_singleline(&mut "hello".to_owned());
            });
            ui.add(egui::Slider::new(&mut 99, 0..=120).text("age"));
            if ui.button("Click each year").clicked() {
                println!("button clicked");
            }
            ui.label(format!("Hello '{}', age {}", "Ben", 99));
        });
}