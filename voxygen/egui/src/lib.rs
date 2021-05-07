use client::{Client, Join, WorldExt};
use common::debug_info::DebugInfo;
use egui_winit_platform::Platform;
#[cfg(all(feature = "be-dyn-lib", feature = "use-dyn-lib"))]
compile_error!("Can't use both \"be-dyn-lib\" and \"use-dyn-lib\" features at once");

#[cfg(feature = "use-dyn-lib")] pub mod dyn_lib;

use common::{
    comp,
    comp::{Poise, PoiseState},
};
#[cfg(feature = "use-dyn-lib")]
pub use dyn_lib::init;
use egui::{Color32, Grid, ScrollArea, Slider, Ui};
use std::{cmp::Ordering, ffi::CStr};

#[cfg(feature = "use-dyn-lib")]
const MAINTAIN_EGUI_FN: &'static [u8] = b"maintain_egui_inner\0";

pub fn maintain(
    platform: &mut Platform,
    egui_state: &mut EguiInnerState,
    client: &Client,
    debug_info: &Option<DebugInfo>,
) {
    #[cfg(not(feature = "use-dyn-lib"))]
    {
        maintain_egui_inner(platform, egui_state, client, debug_info);
    }

    #[cfg(feature = "use-dyn-lib")]
    {
        let lock = dyn_lib::LIB.lock().unwrap();
        let lib = &lock.as_ref().unwrap().lib;

        let maintain_fn: libloading::Symbol<
            fn(&mut Platform, &mut EguiInnerState, &Client, &Option<DebugInfo>),
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

        maintain_fn(platform, egui_state, client, debug_info);
    }
}

pub struct EguiInnerState {
    pub read_ecs: bool,
    pub selected_entity_id: u32,
    pub max_entity_distance: f32,
}

#[cfg_attr(feature = "be-dyn-lib", export_name = "maintain_egui_inner")]
pub fn maintain_egui_inner(
    platform: &mut Platform,
    egui_state: &mut EguiInnerState,
    client: &Client,
    debug_info: &Option<DebugInfo>,
) {
    platform.begin_frame();

    if egui_state.read_ecs {
        let ecs = client.state().ecs();

        let positions = client.state().ecs().read_storage::<comp::Pos>();
        let client_pos = positions.get(client.entity());
        let mut max_entity_distance = egui_state.max_entity_distance;
        egui::Window::new("ECS Entities")
            .default_width(500.0)
            .default_height(500.0)
            .show(&platform.context(), |ui| {
                ui.label(format!("Entity count: {}", &ecs.entities().join().count()));
                ui.add(
                    Slider::new(&mut max_entity_distance, 1.0..=17000.0)
                        .logarithmic(true)
                        .clamp_to_range(true)
                        .text("Max entity distance"),
                );

                let mut scroll_area = ScrollArea::from_max_height(800.0);
                let (current_scroll, max_scroll) = scroll_area.show(ui, |ui| {
                    // if scroll_top {
                    //     ui.scroll_to_cursor(Align::TOP);
                    // }
                    Grid::new("entities_grid")
                        .spacing([40.0, 4.0])
                        .max_col_width(300.0)
                        .striped(true)
                        .show(ui, |ui| {
                            ui.label("-");
                            ui.label("ID");
                            ui.label("Pos");
                            ui.label("Vel");
                            ui.label("Body");
                            ui.label("Poise");
                            ui.end_row();
                            for (entity, body, pos, ori, vel, poise) in (
                                &ecs.entities(),
                                ecs.read_storage::<comp::Body>().maybe(),
                                ecs.read_storage::<comp::Pos>().maybe(),
                                ecs.read_storage::<comp::Ori>().maybe(),
                                ecs.read_storage::<comp::Vel>().maybe(),
                                ecs.read_storage::<comp::Poise>().maybe(),
                            )
                                .join()
                                .filter(|(_, _, pos, _, _, _)| {
                                    client_pos.map_or(true, |client_pos| {
                                        pos.map_or(0.0, |pos| pos.0.distance_squared(client_pos.0))
                                            < max_entity_distance
                                    })
                                })
                            // .sorted_by(|(_, _, pos, _, _, _)| {
                            //     client_pos.map_or(Ordering::Less, |client_pos| {
                            //         pos.map_or(|| 0.0, |x| x.distance_squared(client_pos.0))
                            //     })
                            // })
                            {
                                if ui.button("View").clicked() {
                                    egui_state.selected_entity_id = entity.id();
                                }

                                ui.label(format!("{}", entity.id()));
                                if let Some(pos) = pos {
                                    ui.label(format!(
                                        "{:.3},{:.3},{:.3}",
                                        pos.0.x, pos.0.y, pos.0.z
                                    ));
                                } else {
                                    ui.label("-");
                                }
                                if let Some(vel) = vel {
                                    ui.label(format!(
                                        "{:.3},{:.3},{:.3}",
                                        vel.0.x, vel.0.y, vel.0.z
                                    ));
                                } else {
                                    ui.label("-");
                                }
                                if let Some(body) = body {
                                    ui.label(format!("{:?}", body));
                                } else {
                                    ui.label("-");
                                }

                                if let Some(poise) = poise {
                                    poise_state_label(ui, poise);
                                } else {
                                    ui.label("-");
                                }

                                ui.end_row();
                            }
                        });

                    let margin = ui.visuals().clip_rect_margin;

                    let current_scroll = ui.clip_rect().top() - ui.min_rect().top() + margin;
                    let max_scroll =
                        ui.min_rect().height() - ui.clip_rect().height() + 2.0 * margin;
                    (current_scroll, max_scroll)
                });
            });
        egui_state.max_entity_distance = max_entity_distance;
        let selected_entity = ecs.entities().entity(egui_state.selected_entity_id);
        if selected_entity.gen().is_alive() {
            egui::Window::new("Selected Entity")
                .default_width(300.0)
                .default_height(200.0)
                .show(&platform.context(), |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (entity, body, pos, ori, vel, poise, buffs) in (
                            &ecs.entities(),
                            ecs.read_storage::<comp::Body>().maybe(),
                            ecs.read_storage::<comp::Pos>().maybe(),
                            ecs.read_storage::<comp::Ori>().maybe(),
                            ecs.read_storage::<comp::Vel>().maybe(),
                            ecs.read_storage::<comp::Poise>().maybe(),
                            ecs.read_storage::<comp::Buffs>().maybe(),
                        )
                            .join()
                            .filter(|(e, _, _, _, _, _, _)| e.id() == egui_state.selected_entity_id)
                        {
                            if let Some(body) = body {
                                ui.group(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label("Body");
                                        Grid::new("selected_entity_body_grid")
                                            .spacing([40.0, 4.0])
                                            .max_col_width(100.0)
                                            .striped(true)
                                            .show(ui, |ui| {
                                                ui.label("Type");
                                                ui.label(format!("{:?}", body));
                                            });
                                    });
                                });
                            }
                            if let Some(pos) = pos {
                                ui.group(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label("Pos");
                                        Grid::new("selected_entity_pos_grid")
                                            .spacing([40.0, 4.0])
                                            .max_col_width(100.0)
                                            .striped(true)
                                            .show(ui, |ui| {
                                                ui.label("x");
                                                ui.label(format!("{}", pos.0.x));
                                                ui.end_row();
                                                ui.label("y");
                                                ui.label(format!("{}", pos.0.y));
                                                ui.end_row();
                                                ui.label("z");
                                                ui.label(format!("{}", pos.0.z));
                                                ui.end_row();
                                            });
                                    });
                                });
                            }
                            if let Some(poise) = poise {
                                ui.group(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label("Poise");
                                        Grid::new("selected_entity_poise_grid")
                                            .spacing([40.0, 4.0])
                                            .max_col_width(100.0)
                                            .striped(true)
                                            .show(ui, |ui| {
                                                ui.label("State");
                                                poise_state_label(ui, poise);
                                                ui.end_row();
                                                ui.label("Current");
                                                ui.label(format!("{}", poise.current()));
                                                ui.end_row();
                                                ui.label("Maximum");
                                                ui.label(format!("{}", poise.maximum()));
                                                ui.end_row();
                                                ui.label("Base Max");
                                                ui.label(format!("{}", poise.base_max()));
                                                ui.end_row();
                                            });
                                    });
                                });
                            }

                            if let Some(buffs) = buffs {
                                ui.group(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label("Buffs");
                                        Grid::new("selected_entity_buffs_grid")
                                            .spacing([40.0, 4.0])
                                            .max_col_width(100.0)
                                            .striped(true)
                                            .show(ui, |ui| {
                                                ui.label("Kind");
                                                ui.label("Time");
                                                ui.label("Source");
                                                ui.end_row();
                                                buffs.buffs.iter().for_each(|(k, v)| {
                                                    ui.label(format!("{:?}", v.kind));
                                                    ui.label(
                                                        v.time.map_or("-".to_string(), |time| {
                                                            format!("{:?}", time)
                                                        }),
                                                    );
                                                    ui.label(format!("{:?}", v.source));
                                                    ui.end_row();
                                                });
                                            });
                                    });
                                });
                            }
                        }
                    });
                });
        }
    }

    egui::Window::new("Test Window")
        .default_width(200.0)
        .default_height(200.0)
        .show(&platform.context(), |ui| {
            ui.heading("Debug UI");
            ui.horizontal(|ui| {
                ui.label(format!(
                    "Ping: {:.1}ms",
                    debug_info.as_ref().map_or(0.0, |x| x.ping_ms)
                ));
            });
            if ui.button("Enable ECS reading").clicked() {
                egui_state.read_ecs = true;
            }
        });
}

fn poise_state_label(ui: &mut Ui, poise: &Poise) {
    match poise.poise_state() {
        PoiseState::Normal => {
            ui.label("Normal");
        },
        PoiseState::Interrupted => {
            ui.colored_label(Color32::YELLOW, "Interrupted");
        },
        PoiseState::Stunned => {
            ui.colored_label(Color32::RED, "Stunned");
        },
        PoiseState::Dazed => {
            ui.colored_label(Color32::RED, "Dazed");
        },
        PoiseState::KnockedDown => {
            ui.colored_label(Color32::BLUE, "Knocked Down");
        },
    };
}
