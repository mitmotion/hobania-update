use specs::{Join, WorldExt};
use vek::*;

use client::{self, Client};
use common::{
    comp,
    consts::MAX_PICKUP_RANGE,
    link::Is,
    mounting::Mount,
    terrain::Block,
    util::find_dist::{Cylinder, FindDist},
    vol::ReadVol,
};
use common_base::span;

#[derive(Clone, Copy, Debug)]
pub struct Target<T> {
    pub kind: T,
    pub distance: f32,
    pub position: Vec3<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct Build(pub Vec3<f32>);

#[derive(Clone, Copy, Debug)]
pub struct Collectable;

#[derive(Clone, Copy, Debug)]
pub struct Entity(pub specs::Entity);

#[derive(Clone, Copy, Debug)]
pub struct Mine;

#[derive(Clone, Copy, Debug)]
// line of sight (if not bocked by entity). Not build/mine mode dependent.
pub struct Terrain;

impl<T> Target<T> {
    pub fn position_int(self) -> Vec3<i32> { self.position.map(|p| p.floor() as i32) }
}

/// Max distance an entity can be "targeted"
pub const MAX_TARGET_RANGE: f32 = 300.0;

/// Calculate what the cursor is pointing at within the 3d scene
pub(super) fn targets_under_cursor(
    client: &Client,
    cam_pos: Vec3<f32>,
    cam_dir: Vec3<f32>,
    can_build: bool,
    is_mining: bool,
) -> (
    Option<Target<Build>>,
    Option<Target<Collectable>>,
    Option<Target<Entity>>,
    Option<Target<Mine>>,
    Option<Target<Terrain>>,
) {
    span!(_guard, "targets_under_cursor");
    // Choose a spot above the player's head for item distance checks
    let player_entity = client.entity();
    let ecs = client.state().ecs();
    let positions = ecs.read_storage::<comp::Pos>();
    let player_pos = match positions.get(player_entity) {
        Some(pos) => pos.0,
        None => cam_pos, // Should never happen, but a safe fallback
    };
    let scales = ecs.read_storage();
    let colliders = ecs.read_storage();
    let char_states = ecs.read_storage();
    // Get the player's cylinder
    let player_cylinder = Cylinder::from_components(
        player_pos,
        scales.get(player_entity).copied(),
        colliders.get(player_entity),
        char_states.get(player_entity),
    );
    let terrain = client.state().terrain();

    let find_pos = |hit: fn(Block) -> bool| {
        let cam_ray = terrain
            .ray(cam_pos, cam_pos + cam_dir * 100.0)
            .until(|block| hit(*block))
            .cast();
        let cam_ray = (cam_ray.0, cam_ray.1.map(|x| x.copied()));
        let cam_dist = cam_ray.0;

        if matches!(
            cam_ray.1,
            Ok(Some(_)) if player_cylinder.min_distance(cam_pos + cam_dir * (cam_dist + 0.01)) <= MAX_PICKUP_RANGE
        ) {
            (
                Some(cam_pos + cam_dir * (cam_dist + 0.01)),
                Some(cam_pos + cam_dir * (cam_dist - 0.01)),
                Some(cam_ray),
            )
        } else {
            (None, None, None)
        }
    };

    let (collect_pos, _, collect_cam_ray) = find_pos(|b: Block| b.is_collectible());
    let (mine_pos, _, mine_cam_ray) = is_mining
        .then(|| find_pos(|b: Block| b.mine_tool().is_some()))
        .unwrap_or((None, None, None));
    let (solid_pos, place_block_pos, solid_cam_ray) = find_pos(|b: Block| b.is_filled());

    // See if ray hits entities
    // Don't cast through blocks, (hence why use shortest_cam_dist from non-entity
    // targets) Could check for intersection with entity from last frame to
    // narrow this down
    let cast_dist = solid_cam_ray
        .as_ref()
        .map(|(d, _)| d.min(MAX_TARGET_RANGE))
        .unwrap_or(MAX_TARGET_RANGE);

    // Need to raycast by distance to cam
    // But also filter out by distance to the player (but this only needs to be done
    // on final result)
    let mut nearby = (
        &ecs.entities(),
        &positions,
        scales.maybe(),
        &ecs.read_storage::<comp::Body>(),
        ecs.read_storage::<comp::Item>().maybe(),
        !&ecs.read_storage::<Is<Mount>>(),
    )
        .join()
        .filter(|(e, _, _, _, _, _)| *e != player_entity)
        .filter_map(|(e, p, s, b, i, _)| {
            const RADIUS_SCALE: f32 = 3.0;
            // TODO: use collider radius instead of body radius?
            let radius = s.map_or(1.0, |s| s.0) * b.max_radius() * RADIUS_SCALE;
            // Move position up from the feet
            let pos = Vec3::new(p.0.x, p.0.y, p.0.z + radius);
            // Distance squared from camera to the entity
            let dist_sqr = pos.distance_squared(cam_pos);
            // We only care about interacting with entities that contain items,
            // or are not inanimate (to trade with)
            if i.is_some() || !matches!(b, comp::Body::Object(_)) {
                Some((e, pos, radius, dist_sqr))
            } else {
                None
            }
        })
        // Roughly filter out entities farther than ray distance
        .filter(|(_, _, r, d_sqr)| *d_sqr <= cast_dist.powi(2) + 2.0 * cast_dist * r + r.powi(2))
        // Ignore entities intersecting the camera
        .filter(|(_, _, r, d_sqr)| *d_sqr > r.powi(2))
        // Substract sphere radius from distance to the camera
        .map(|(e, p, r, d_sqr)| (e, p, r, d_sqr.sqrt() - r))
        .collect::<Vec<_>>();
    // Sort by distance
    nearby.sort_unstable_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    let seg_ray = LineSegment3 {
        start: cam_pos,
        end: cam_pos + cam_dir * cast_dist,
    };
    // TODO: fuzzy borders
    let entity_target = nearby
        .iter()
        .map(|(e, p, r, _)| (e, *p, r))
        // Find first one that intersects the ray segment
        .find(|(_, p, r)| seg_ray.projected_point(*p).distance_squared(*p) < r.powi(2))
        .and_then(|(e, p, _)| {
            // Get the entity's cylinder
            let target_cylinder = Cylinder::from_components(
                p,
                scales.get(*e).copied(),
                colliders.get(*e),
                char_states.get(*e),
            );

            let dist_to_player = player_cylinder.min_distance(target_cylinder);
            if dist_to_player < MAX_TARGET_RANGE {
                Some(Target {
                    kind: Entity(*e),
                    position: p,
                    distance: dist_to_player,
                })
            } else { None }
        });

    let solid_ray_dist = solid_cam_ray.map(|r| r.0);
    let terrain_target = if let (None, Some(distance)) = (entity_target, solid_ray_dist) {
        solid_pos.map(|position| Target {
            kind: Terrain,
            distance,
            position,
        })
    } else {
        None
    };

    let build_target = if let (true, Some(distance)) = (can_build, solid_ray_dist) {
        place_block_pos
            .zip(solid_pos)
            .map(|(place_pos, position)| Target {
                kind: Build(place_pos),
                distance,
                position,
            })
    } else {
        None
    };

    let collect_target = collect_pos
        .zip(collect_cam_ray)
        .map(|(position, ray)| Target {
            kind: Collectable,
            distance: ray.0,
            position,
        });

    let mine_target = mine_pos.zip(mine_cam_ray).map(|(position, ray)| Target {
        kind: Mine,
        distance: ray.0,
        position,
    });

    // Return multiple possible targets
    // GameInput events determine which target to use.
    (
        build_target,
        collect_target,
        entity_target,
        mine_target,
        terrain_target,
    )
}
