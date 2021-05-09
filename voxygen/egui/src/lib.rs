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
use core::mem;
#[cfg(feature = "use-dyn-lib")]
pub use dyn_lib::init;
use egui::{
    plot::{Plot, Value},
    widgets::plot::Curve,
    Color32, Grid, ScrollArea, Slider, Ui,
};
use std::{cmp::Ordering, ffi::CStr};

#[cfg(feature = "use-dyn-lib")]
const MAINTAIN_EGUI_FN: &'static [u8] = b"maintain_egui_inner\0";

pub fn maintain(
    platform: &mut Platform,
    egui_state: &mut EguiInnerState,
    client: &Client,
    debug_info: &Option<DebugInfo>,
    added_cylinder_shape_id: Option<u64>,
) -> EguiActions {
    #[cfg(not(feature = "use-dyn-lib"))]
    {
        return maintain_egui_inner(
            platform,
            egui_state,
            client,
            debug_info,
            added_cylinder_shape_id,
        );
    }

    #[cfg(feature = "use-dyn-lib")]
    {
        let lock = dyn_lib::LIB.lock().unwrap();
        let lib = &lock.as_ref().unwrap().lib;

        let maintain_fn: libloading::Symbol<
            fn(
                &mut Platform,
                &mut EguiInnerState,
                &Client,
                &Option<DebugInfo>,
                Option<u64>,
            ) -> EguiActions,
        > = unsafe { lib.get(MAINTAIN_EGUI_FN) }.unwrap_or_else(|e| {
            panic!(
                "Trying to use: {} but had error: {:?}",
                CStr::from_bytes_with_nul(MAINTAIN_EGUI_FN)
                    .map(CStr::to_str)
                    .unwrap()
                    .unwrap(),
                e
            )
        });

        return maintain_fn(
            platform,
            egui_state,
            client,
            debug_info,
            added_cylinder_shape_id,
        );
    }
}

pub struct SelectedEntityInfo {
    entity_id: u32,
    debug_shape_id: Option<u64>,
}

impl SelectedEntityInfo {
    fn new(entity_id: u32) -> Self {
        Self {
            entity_id,
            debug_shape_id: None,
        }
    }
}

pub struct EguiInnerState {
    read_ecs: bool,
    selected_entity_info: Option<SelectedEntityInfo>,
    max_entity_distance: f32,
    selected_entity_cylinder_height: f32,
    frame_times: Vec<f32>,
}

impl EguiInnerState {
    pub fn new() -> Self {
        Self {
            read_ecs: false,
            selected_entity_info: None,
            max_entity_distance: 100000.0,
            selected_entity_cylinder_height: 10.0,
            frame_times: Vec::new(),
        }
    }
}

pub enum DebugShapeAction {
    AddCylinder {
        radius: f32,
        height: f32,
    },
    SetPosAndColor {
        id: u64,
        pos: [f32; 4],
        color: [f32; 4],
    },
    RemoveCylinder(u64),
}

#[derive(Default)]
pub struct EguiActions {
    pub actions: Vec<DebugShapeAction>,
}

#[cfg_attr(feature = "be-dyn-lib", export_name = "maintain_egui_inner")]
pub fn maintain_egui_inner(
    platform: &mut Platform,
    egui_state: &mut EguiInnerState,
    client: &Client,
    debug_info: &Option<DebugInfo>,
    added_cylinder_shape_id: Option<u64>,
) -> EguiActions {
    platform.begin_frame();
    let mut egui_actions = EguiActions::default();
    let mut previous_selected_entity: Option<SelectedEntityInfo> = None;
    let mut max_entity_distance = egui_state.max_entity_distance;
    let mut selected_entity_cylinder_height = egui_state.selected_entity_cylinder_height;

    // If a debug cylinder was added in the last frame, store it against the
    // selected entity
    if let Some(shape_id) = added_cylinder_shape_id {
        if let Some(selected_entity) = &mut egui_state.selected_entity_info {
            selected_entity.debug_shape_id = Some(shape_id);
        }
    }

    debug_info.as_ref().map(|x| {
        egui_state.frame_times.push(x.frame_time.as_nanos() as f32);
        if egui_state.frame_times.len() > 250 {
            egui_state.frame_times.remove(0);
        }
    });

    if egui_state.read_ecs {
        let ecs = client.state().ecs();

        let positions = client.state().ecs().read_storage::<comp::Pos>();
        let client_pos = positions.get(client.entity());

        egui::Window::new("ECS Entities")
            .default_width(500.0)
            .default_height(500.0)
            .show(&platform.context(), |ui| {
                ui.label(format!("Entity count: {}", &ecs.entities().join().count()));
                ui.add(
                    Slider::new(&mut max_entity_distance, 1.0..=100000.0)
                        .logarithmic(true)
                        .clamp_to_range(true)
                        .text("Max entity distance"),
                );

                ui.add(
                    Slider::new(&mut selected_entity_cylinder_height, 0.1..=100.0)
                        .logarithmic(true)
                        .clamp_to_range(true)
                        .text("Cylinder height"),
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
                            for (entity, body, stats, pos, ori, vel, poise) in (
                                &ecs.entities(),
                                ecs.read_storage::<comp::Body>().maybe(),
                                ecs.read_storage::<comp::Stats>().maybe(),
                                ecs.read_storage::<comp::Pos>().maybe(),
                                ecs.read_storage::<comp::Ori>().maybe(),
                                ecs.read_storage::<comp::Vel>().maybe(),
                                ecs.read_storage::<comp::Poise>().maybe(),
                            )
                                .join()
                                .filter(|(_, _, _, pos, _, _, _)| {
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
                                    previous_selected_entity =
                                        mem::take(&mut egui_state.selected_entity_info);

                                    if pos.is_some() {
                                        egui_actions.actions.push(DebugShapeAction::AddCylinder {
                                            radius: 1.0,
                                            height: egui_state.selected_entity_cylinder_height,
                                        });
                                    }
                                    egui_state.selected_entity_info =
                                        Some(SelectedEntityInfo::new(entity.id()));
                                }

                                ui.label(format!("{}", entity.id()));

                                if let Some(pos) = pos {
                                    ui.label(format!(
                                        "{:.0},{:.0},{:.0}",
                                        pos.0.x, pos.0.y, pos.0.z
                                    ));
                                } else {
                                    ui.label("-");
                                }

                                if let Some(vel) = vel {
                                    ui.label(format!(
                                        "{:.1},{:.1},{:.1}",
                                        vel.0.x, vel.0.y, vel.0.z
                                    ));
                                } else {
                                    ui.label("-");
                                }
                                if let Some(stats) = stats {
                                    ui.label(&stats.name);
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
        if let Some(selected_entity_info) = &mut egui_state.selected_entity_info {
            let selected_entity = ecs.entities().entity(selected_entity_info.entity_id);
            if !selected_entity.gen().is_alive() {
                previous_selected_entity = mem::take(&mut egui_state.selected_entity_info);
            } else {
                for (entity, body, stats, pos, ori, vel, poise, buffs) in (
                    &ecs.entities(),
                    ecs.read_storage::<comp::Body>().maybe(),
                    ecs.read_storage::<comp::Stats>().maybe(),
                    ecs.read_storage::<comp::Pos>().maybe(),
                    ecs.read_storage::<comp::Ori>().maybe(),
                    ecs.read_storage::<comp::Vel>().maybe(),
                    ecs.read_storage::<comp::Poise>().maybe(),
                    ecs.read_storage::<comp::Buffs>().maybe(),
                )
                    .join()
                    .filter(|(e, _, _, _, _, _, _, _)| e.id() == selected_entity_info.entity_id)
                {
                    egui::Window::new(format!(
                        "Selected Entity - {}",
                        stats.as_ref().map_or("<No Name>", |x| &x.name)
                    ))
                    .default_width(300.0)
                    .default_height(200.0)
                    .show(&platform.context(), |ui| {
                        ui.horizontal_wrapped(|ui| {
                            9-
                            if let Some(pos) = pos {
                                if let Some(shape_id) = selected_entity_info.debug_shape_id {
                                    egui_actions.actions.push(DebugShapeAction::SetPosAndColor {
                                        id: shape_id,
                                        color: [1.0, 1.0, 0.0, 0.5],
                                        pos: [pos.0.x, pos.0.y, pos.0.z + 2.0, 0.0],
                                    });
                                }
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
                        });
                    });
                }
            }
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

    egui::Window::new("Frame Time")
        .default_width(200.0)
        .default_height(200.0)
        .show(&platform.context(), |ui| {
            let plot = Plot::default().curve(Curve::from_values_iter(
                egui_state
                    .frame_times
                    .iter()
                    .enumerate()
                    .map(|(i, x)| Value::new(i as f64, *x)),
            ));
            ui.add(plot);
        });

    if let Some(previous) = previous_selected_entity {
        if let Some(debug_shape_id) = previous.debug_shape_id {
            egui_actions
                .actions
                .push(DebugShapeAction::RemoveCylinder(debug_shape_id));
        }
    };

    if let Some(selected_entity) = &egui_state.selected_entity_info {
        if let Some(debug_shape_id) = selected_entity.debug_shape_id {
            if egui_state.selected_entity_cylinder_height != selected_entity_cylinder_height {
                egui_actions
                    .actions
                    .push(DebugShapeAction::RemoveCylinder(debug_shape_id));
                egui_actions.actions.push(DebugShapeAction::AddCylinder {
                    radius: 1.0,
                    height: selected_entity_cylinder_height,
                });
            }
        }
    };

    egui_state.max_entity_distance = max_entity_distance;
    egui_state.selected_entity_cylinder_height = selected_entity_cylinder_height;
    egui_actions
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
