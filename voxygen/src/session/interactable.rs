use ordered_float::OrderedFloat;
use specs::{Join, WorldExt};
use vek::*;

use super::{
    find_shortest_distance,
    target::{self, Target, MAX_TARGET_RANGE},
};
use client::Client;
use common::{
    comp,
    consts::MAX_PICKUP_RANGE,
    link::Is,
    mounting::Mount,
    terrain::Block,
    util::find_dist::{Cube, Cylinder, FindDist},
    vol::ReadVol,
};
use common_base::span;

use crate::scene::{terrain::Interaction, Scene};

// TODO: extract mining blocks (the None case in the Block variant) from this
// enum since they don't use the interaction key
#[derive(Clone, Copy, Debug)]
pub enum Interactable {
    Block(Block, Vec3<i32>, Interaction),
    Entity(specs::Entity),
}

impl Interactable {
    pub fn entity(self) -> Option<specs::Entity> {
        match self {
            Self::Entity(e) => Some(e),
            Self::Block(_, _, _) => None,
        }
    }
}

/// Select interactable to highlight, display interaction text for, and to
/// interact with if the interact key is pressed
/// Selected in the following order:
/// 1) Targeted items, in order of nearest under cursor:
///     (a) entity (if within range)
///     (b) collectable
///     (c) can be mined, and is a mine sprite (Air) not a weak rock.
/// 2) outside of targeted cam ray
///     -> closest of nearest interactable entity/block
pub(super) fn select_interactable(
    client: &Client,
    collect_target: Option<Target<target::Collectable>>,
    entity_target: Option<Target<target::Entity>>,
    mine_target: Option<Target<target::Mine>>,
    scene: &Scene,
) -> Option<Interactable> {
    span!(_guard, "select_interactable");
    use common::{spiral::Spiral2d, terrain::TerrainChunk, vol::RectRasterableVol};

    let nearest_dist = find_shortest_distance(&[
        mine_target.map(|t| t.distance),
        entity_target.map(|t| t.distance),
        collect_target.map(|t| t.distance),
    ]);

    fn get_block<T>(client: &Client, target: Target<T>) -> Option<Block> {
        client
            .state()
            .terrain()
            .get(target.position_int())
            .ok()
            .copied()
    }

    if let Some(interactable) = entity_target
        .and_then(|t| {
            if t.distance < MAX_TARGET_RANGE && Some(t.distance) == nearest_dist {
                let entity = t.kind.0;
                Some(Interactable::Entity(entity))
            } else {
                None
            }
        })
        .or_else(|| {
            collect_target.and_then(|t| {
                if Some(t.distance) == nearest_dist {
                    get_block(client, t)
                        .map(|b| Interactable::Block(b, t.position_int(), Interaction::Collect))
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            mine_target.and_then(|t| {
                if Some(t.distance) == nearest_dist {
                    get_block(client, t).and_then(|b| {
                        // Handling edge detection. sometimes the casting (in Target mod) returns a
                        // position which is actually empty, which we do not want labeled as an
                        // interactable. We are only returning the mineable air
                        // elements (e.g. minerals). The mineable weakrock are used
                        // in the terrain selected_pos, but is not an interactable.
                        if b.mine_tool().is_some() && b.is_air() {
                            Some(Interactable::Block(b, t.position_int(), Interaction::Mine))
                        } else {
                            None
                        }
                    })
                } else {
                    None
                }
            })
        })
    {
        Some(interactable)
    } else {
        // If there are no directly targeted interactables select the closest one if any
        // are in range
        let ecs = client.state().ecs();
        let player_entity = client.entity();
        let positions = ecs.read_storage::<comp::Pos>();
        let player_pos = positions.get(player_entity)?.0;

        let scales = ecs.read_storage::<comp::Scale>();
        let colliders = ecs.read_storage::<comp::Collider>();
        let char_states = ecs.read_storage::<comp::CharacterState>();
        let is_mount = ecs.read_storage::<Is<Mount>>();

        let player_cylinder = Cylinder::from_components(
            player_pos,
            scales.get(player_entity).copied(),
            colliders.get(player_entity),
            char_states.get(player_entity),
        );

        let closest_interactable_entity = (
            &ecs.entities(),
            &positions,
            scales.maybe(),
            colliders.maybe(),
            char_states.maybe(),
            !&is_mount,
        )
            .join()
            .filter(|(e, _, _, _, _, _)| *e != player_entity)
            .map(|(e, p, s, c, cs, _)| {
                let cylinder = Cylinder::from_components(p.0, s.copied(), c, cs);
                (e, cylinder)
            })
            // Roughly filter out entities farther than interaction distance
            .filter(|(_, cylinder)| player_cylinder.approx_in_range(*cylinder, MAX_PICKUP_RANGE))
            .map(|(e, cylinder)| (e, player_cylinder.min_distance(cylinder)))
            .min_by_key(|(_, dist)| OrderedFloat(*dist));

        // Only search as far as closest interactable entity
        let search_dist = closest_interactable_entity.map_or(MAX_PICKUP_RANGE, |(_, dist)| dist);
        let player_chunk = player_pos.xy().map2(TerrainChunk::RECT_SIZE, |e, sz| {
            (e.floor() as i32).div_euclid(sz as i32)
        });
        let terrain = scene.terrain();

        // Find closest interactable block
        // TODO: consider doing this one first?
        let closest_interactable_block_pos = Spiral2d::new()
            // TODO: this formula for the number to take was guessed
            // Note: assumes RECT_SIZE.x == RECT_SIZE.y
            .take(((search_dist / TerrainChunk::RECT_SIZE.x as f32).ceil() as usize * 2 + 1).pow(2))
            .flat_map(|offset| {
                let chunk_pos = player_chunk + offset;
                let chunk_voxel_pos =
                        Vec3::<i32>::from(chunk_pos * TerrainChunk::RECT_SIZE.map(|e| e as i32));
                terrain.get(chunk_pos).map(|data| (data, chunk_voxel_pos))
            })
            // TODO: maybe we could make this more efficient by putting the
            // interactables is some sort of spatial structure
            .flat_map(|(chunk_data, chunk_pos)| {
                chunk_data
                    .blocks_of_interest
                    .interactables
                    .iter()
                    .map(move |(block_offset, interaction)| (chunk_pos + block_offset, interaction))
            })
            .map(|(block_pos, interaction)| (
                    block_pos,
                    block_pos.map(|e| e as f32 + 0.5)
                        .distance_squared(player_pos),
                    interaction,
            ))
            .min_by_key(|(_, dist_sqr, _)| OrderedFloat(*dist_sqr))
            .map(|(block_pos, _, interaction)| (block_pos, interaction));

        // Return the closest of the 2 closest
        closest_interactable_block_pos
            .filter(|(block_pos, _)| {
                player_cylinder.min_distance(Cube {
                    min: block_pos.as_(),
                    side_length: 1.0,
                }) < search_dist
            })
            .and_then(|(block_pos, interaction)| {
                client
                    .state()
                    .terrain()
                    .get(block_pos)
                    .ok()
                    .copied()
                    .map(|b| Interactable::Block(b, block_pos, *interaction))
            })
            .or_else(|| closest_interactable_entity.map(|(e, _)| Interactable::Entity(e)))
    }
}
