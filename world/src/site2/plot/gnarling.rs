use super::*;
use crate::{assets::AssetHandle, util::attempt, Land};
use common::terrain::{Structure as PrefabStructure, StructuresGroup};
use inline_tweak::tweak;
use lazy_static::lazy_static;
use rand::prelude::*;
use vek::*;

pub struct GnarlingFortification {
    name: String,
    seed: u32,
    origin: Vec2<i32>,
    radius: i32,
    wall_radius: i32,
    // Vec2 is relative position of wall relative to site origin, bool indicates whether it is a
    // corner, and thus whether a tower gets constructed
    ordered_wall_points: Vec<(Vec2<i32>, bool)>,
    gate_index: usize,
    // Structure indicates the kind of structure it is, vec2 is relative position of a hut compared
    // to origin, ori tells which way structure should face
    structure_locations: Vec<(GnarlingStructure, Vec2<i32>, Ori)>,
}

enum GnarlingStructure {
    Hut,
    Totem,
}

impl GnarlingStructure {
    fn required_separation(&self, other: &Self) -> i32 {
        match (self, other) {
            (Self::Hut, Self::Hut) => 15,
            (Self::Hut, Self::Totem) | (Self::Totem, Self::Hut) => 20,
            (Self::Totem, Self::Totem) => 50,
        }
    }
}

const SECTIONS_PER_WALL_SEGMENT: usize = 3;

impl GnarlingFortification {
    pub fn generate(wpos: Vec2<i32>, land: &Land, rng: &mut impl Rng) -> Self {
        let name = String::from("Gnarling Fortification");
        let seed = rng.gen();
        let origin = wpos;

        let wall_radius = {
            let unit_size = rng.gen_range(10..20);
            let num_units = rng.gen_range(5..10);
            let variation = rng.gen_range(0..50);
            unit_size * num_units + variation
        };

        let radius = wall_radius + 50;

        let num_points = (wall_radius / 15).max(4);
        let wall_corners = (0..num_points)
            .into_iter()
            .map(|a| {
                let angle = a as f32 / num_points as f32 * core::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()).map(|a| (a * wall_radius as f32) as i32)
            })
            .map(|point| {
                point.map(|a| {
                    let variation = wall_radius / 5;
                    a + rng.gen_range(-variation..=variation)
                })
            })
            .collect::<Vec<_>>();

        let gate_index = rng.gen_range(0..wall_corners.len());

        // This adds additional points for the wall on the line between two points,
        // allowing the wall to better handle slopes
        let ordered_wall_points = wall_corners
            .iter()
            .enumerate()
            .flat_map(|(i, point)| {
                let next_point = if let Some(point) = wall_corners.get(i + 1) {
                    *point
                } else {
                    wall_corners[0]
                };
                (0..(SECTIONS_PER_WALL_SEGMENT as i32))
                    .into_iter()
                    .map(move |a| {
                        let is_start_segment = a == 0;
                        (
                            point + (next_point - point) * a / (SECTIONS_PER_WALL_SEGMENT as i32),
                            is_start_segment,
                        )
                    })
            })
            .collect::<Vec<_>>();

        let desired_structures = wall_radius.pow(2) / 100;
        let mut structure_locations = Vec::<(GnarlingStructure, Vec2<i32>, Ori)>::new();
        for _ in 0..desired_structures {
            if let Some((hut_loc, kind)) = attempt(16, || {
                // Choose structure kind
                let structure_kind = match rng.gen_range(0..10) {
                    0 => GnarlingStructure::Totem,
                    _ => GnarlingStructure::Hut,
                };

                // Choose triangle
                let section = rng.gen_range(0..wall_corners.len());

                if section == gate_index {
                    return None;
                }

                let center = Vec2::zero();
                let corner_1 = wall_corners[section];
                let corner_2 = if let Some(corner) = wall_corners.get(section + 1) {
                    *corner
                } else {
                    wall_corners[0]
                };

                let center_weight: f32 = rng.gen_range(0.2..0.6);
                let corner_1_weight = rng.gen_range(0.0..(1.0 - center_weight));
                let corner_2_weight = 1.0 - center_weight - corner_1_weight;

                let structure_center: Vec2<i32> = (center * center_weight
                    + corner_1.as_() * corner_1_weight
                    + corner_2.as_() * corner_2_weight)
                    .as_();

                // Check that structure not too close to another structure
                if structure_locations.iter().any(|(kind, loc, _door_dir)| {
                    structure_center.distance_squared(*loc)
                        < structure_kind.required_separation(kind).pow(2)
                }) {
                    None
                } else {
                    Some((structure_center, structure_kind))
                }
            }) {
                let dir_to_center = match hut_loc {
                    pos if pos.x.abs() > pos.y.abs() && pos.x > 0 => Ori::West,
                    pos if pos.x.abs() > pos.y.abs() => Ori::East,
                    pos if pos.y < 0 => Ori::North,
                    _ => Ori::South,
                };
                let door_rng: u32 = rng.gen_range(0..9);
                let door_dir = match door_rng {
                    0..=3 => dir_to_center,
                    4..=5 => dir_to_center.cw(),
                    6..=7 => dir_to_center.ccw(),
                    // Should only be 8
                    _ => dir_to_center.opposite(),
                };
                structure_locations.push((kind, hut_loc, door_dir));
            }
        }

        Self {
            name,
            seed,
            origin,
            radius,
            wall_radius,
            ordered_wall_points,
            gate_index,
            structure_locations,
        }
    }

    pub fn name(&self) -> &str { &self.name }

    pub fn radius(&self) -> i32 { self.radius }
}

impl Structure for GnarlingFortification {
    fn render(&self, _site: &Site, land: &Land, painter: &Painter) {
        // Create outer wall
        for (i, (point, _is_tower)) in self.ordered_wall_points.iter().enumerate() {
            // If wall section is a gate, skip rendering the wall
            if ((self.gate_index * SECTIONS_PER_WALL_SEGMENT)
                ..((self.gate_index + 1) * SECTIONS_PER_WALL_SEGMENT))
                .contains(&i)
            {
                continue;
            }

            // Other point of wall segment
            let (next_point, _is_tower) = if let Some(point) = self.ordered_wall_points.get(i + 1) {
                *point
            } else {
                self.ordered_wall_points[0]
            };
            // 2d world positions of each point in wall segment
            let start_wpos = point + self.origin;
            let end_wpos = next_point + self.origin;

            // Wall base
            let wall_depth = 3.0;
            let start = start_wpos
                .as_()
                .with_z(land.get_alt_approx(start_wpos) - wall_depth);
            let end = end_wpos
                .as_()
                .with_z(land.get_alt_approx(end_wpos) - wall_depth);

            let wall_base_thickness = 3.0;
            let wall_base_height = 3.0;

            painter
                .segment_prism(
                    start,
                    end,
                    wall_base_thickness,
                    wall_base_height + wall_depth as f32,
                )
                .fill(Fill::Block(Block::new(
                    BlockKind::Wood,
                    Rgb::new(55, 25, 8),
                )));

            // Middle of wall
            let start = start_wpos.as_().with_z(land.get_alt_approx(start_wpos));
            let end = end_wpos.as_().with_z(land.get_alt_approx(end_wpos));

            let wall_mid_thickness = 1.0;
            let wall_mid_height = 5.0 + wall_base_height;

            painter
                .segment_prism(start, end, wall_mid_thickness, wall_mid_height)
                .fill(Fill::Block(Block::new(
                    BlockKind::Wood,
                    Rgb::new(55, 25, 8),
                )));

            // Top of wall
            let start = start_wpos
                .as_()
                .with_z(land.get_alt_approx(start_wpos) + wall_mid_height);
            let end = end_wpos
                .as_()
                .with_z(land.get_alt_approx(end_wpos) + wall_mid_height);

            let wall_top_thickness = 2.0;
            let wall_top_height = 1.0;

            painter
                .segment_prism(start, end, wall_top_thickness, wall_top_height)
                .fill(Fill::Block(Block::new(
                    BlockKind::Wood,
                    Rgb::new(55, 25, 8),
                )));

            // Wall parapets
            let parapet_z_offset = 1.0;

            let start = Vec3::new(
                point.x as f32 * (self.wall_radius as f32 + 1.0) / (self.wall_radius as f32)
                    + self.origin.x as f32,
                point.y as f32 * (self.wall_radius as f32 + 1.0) / (self.wall_radius as f32)
                    + self.origin.y as f32,
                land.get_alt_approx(start_wpos) + wall_mid_height + wall_top_height
                    - parapet_z_offset,
            );
            let end = Vec3::new(
                next_point.x as f32 * (self.wall_radius as f32 + 1.0) / (self.wall_radius as f32)
                    + self.origin.x as f32,
                next_point.y as f32 * (self.wall_radius as f32 + 1.0) / (self.wall_radius as f32)
                    + self.origin.y as f32,
                land.get_alt_approx(end_wpos) + wall_mid_height + wall_top_height
                    - parapet_z_offset,
            );

            let wall_par_thickness = tweak!(0.8);
            let wall_par_height = 1.0;

            painter
                .segment_prism(
                    start,
                    end,
                    wall_par_thickness,
                    wall_par_height + parapet_z_offset as f32,
                )
                .fill(Fill::Block(Block::new(
                    BlockKind::Wood,
                    Rgb::new(55, 25, 8),
                )));
        }

        // Create towers
        self.ordered_wall_points
            .iter()
            .filter_map(|(point, is_tower)| is_tower.then_some(point))
            .for_each(|point| {
                let wpos = point + self.origin;

                // Tower base
                let tower_depth = 3;
                let tower_base_pos = wpos.with_z(land.get_alt_approx(wpos) as i32 - tower_depth);
                let tower_radius = 5.;
                let tower_height = 20.0;

                painter
                    .prim(Primitive::cylinder(
                        tower_base_pos,
                        tower_radius,
                        tower_depth as f32 + tower_height,
                    ))
                    .fill(Fill::Block(Block::new(
                        BlockKind::Wood,
                        Rgb::new(55, 25, 8),
                    )));

                // Tower cylinder
                let tower_floor_pos = wpos.with_z(land.get_alt_approx(wpos) as i32);

                painter
                    .prim(Primitive::cylinder(
                        tower_floor_pos,
                        tower_radius - 1.0,
                        tower_height,
                    ))
                    .fill(Fill::Block(Block::empty()));

                // Tower top floor
                let top_floor_z = (land.get_alt_approx(wpos) + tower_height - 2.0) as i32;
                let tower_top_floor_pos = wpos.with_z(top_floor_z);

                painter
                    .prim(Primitive::cylinder(tower_top_floor_pos, tower_radius, 1.0))
                    .fill(Fill::Block(Block::new(
                        BlockKind::Wood,
                        Rgb::new(55, 25, 8),
                    )));

                // Tower roof poles
                let roof_pole_height = 5;
                let relative_pole_positions = [
                    Vec2::new(-4, -4),
                    Vec2::new(-4, 3),
                    Vec2::new(3, -4),
                    Vec2::new(3, 3),
                ];
                relative_pole_positions
                    .iter()
                    .map(|rpos| wpos + rpos)
                    .for_each(|pole_pos| {
                        painter
                            .line(
                                pole_pos.with_z(top_floor_z),
                                pole_pos.with_z(top_floor_z + roof_pole_height),
                                1.,
                            )
                            .fill(Fill::Block(Block::new(
                                BlockKind::Wood,
                                Rgb::new(55, 25, 8),
                            )));
                    });

                // Tower roof
                let roof_sphere_radius = 10;
                let roof_radius = tower_radius + 1.0;
                let roof_height = 3;

                let roof_cyl = painter.prim(Primitive::cylinder(
                    wpos.with_z(top_floor_z + roof_pole_height),
                    roof_radius,
                    roof_height as f32,
                ));

                painter
                    .prim(Primitive::sphere(
                        wpos.with_z(
                            top_floor_z + roof_pole_height + roof_height - roof_sphere_radius,
                        ),
                        roof_sphere_radius as f32,
                    ))
                    .intersect(roof_cyl)
                    .fill(Fill::Block(Block::new(
                        BlockKind::Wood,
                        Rgb::new(55, 25, 8),
                    )));
            });

        self.structure_locations
            .iter()
            .for_each(|(kind, loc, door_dir)| {
                let wpos = self.origin + loc;
                let alt = land.get_alt_approx(wpos) as i32;

                match kind {
                    GnarlingStructure::Hut => {
                        let hut_radius = 5.0;
                        let hut_wall_height = 4.0;

                        // Floor
                        let base = wpos.with_z(alt);
                        painter
                            .prim(Primitive::cylinder(base, hut_radius + 1.0, 2.0))
                            .fill(Fill::Block(Block::new(
                                BlockKind::Wood,
                                Rgb::new(55, 25, 8),
                            )));

                        // Wall
                        let floor_pos = wpos.with_z(alt + 1);
                        painter
                            .prim(Primitive::cylinder(floor_pos, hut_radius, hut_wall_height))
                            .fill(Fill::Block(Block::new(
                                BlockKind::Wood,
                                Rgb::new(55, 25, 8),
                            )));
                        painter
                            .prim(Primitive::cylinder(
                                floor_pos,
                                hut_radius - 1.0,
                                hut_wall_height,
                            ))
                            .fill(Fill::Block(Block::empty()));

                        // Door
                        let door_height = 3;

                        let aabb_min = |dir| {
                            match dir {
                                Ori::North | Ori::East => wpos - Vec2::one(),
                                Ori::South | Ori::West => wpos + Vec2::one(),
                            }
                            .with_z(alt + 1)
                        };
                        let aabb_max = |dir| {
                            (match dir {
                                Ori::North | Ori::East => wpos + Vec2::one(),
                                Ori::South | Ori::West => wpos - Vec2::one(),
                            } + dir.dir() * hut_radius as i32)
                                .with_z(alt + 1 + door_height)
                        };

                        painter
                            .prim(Primitive::Aabb(
                                Aabb {
                                    min: aabb_min(*door_dir),
                                    max: aabb_max(*door_dir),
                                }
                                .made_valid(),
                            ))
                            .fill(Fill::Block(Block::empty()));

                        // Roof
                        let roof_height = 3.0;
                        let roof_radius = hut_radius + 1.0;
                        painter
                            .prim(Primitive::cone(
                                wpos.with_z(alt + 1 + hut_wall_height as i32),
                                roof_radius,
                                roof_height,
                            ))
                            .fill(Fill::Block(Block::new(
                                BlockKind::Wood,
                                Rgb::new(55, 25, 8),
                            )));
                    },
                    GnarlingStructure::Totem => {
                        let totem_pos = wpos.with_z(alt);

                        lazy_static! {
                            pub static ref TOTEM: AssetHandle<StructuresGroup> =
                                PrefabStructure::load_group("site_structures.gnarling.totem");
                        }

                        let totem = TOTEM.read();
                        let totem = totem[self.seed as usize % totem.len()].clone();

                        painter
                            .prim(Primitive::Prefab(Box::new(totem.clone())))
                            .translate(totem_pos)
                            .fill(Fill::Prefab(Box::new(totem), totem_pos, self.seed));
                    },
                }
            });
    }
}
