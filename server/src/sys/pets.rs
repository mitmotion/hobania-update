use common::{
    comp::{Alignment, Pet, PhysicsState, Pos},
    terrain::TerrainGrid,
    uid::UidAllocator,
};
use common_ecs::{Job, Origin, Phase, System};
use specs::{
    saveload::MarkerAllocator, Entities, Entity, Join, Read, ReadExpect, ReadStorage, WriteStorage,
};

/// This system is responsible for handling pets
#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    type SystemData = (
        Entities<'a>,
        ReadExpect<'a, TerrainGrid>,
        WriteStorage<'a, Pos>,
        ReadStorage<'a, Alignment>,
        ReadStorage<'a, Pet>,
        ReadStorage<'a, PhysicsState>,
        Read<'a, UidAllocator>,
    );

    const NAME: &'static str = "pets";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(
        _job: &mut Job<Self>,
        (entities, terrain, mut positions, alignments, pets, physics, uid_allocator): Self::SystemData,
    ) {
        const LOST_PET_DISTANCE_THRESHOLD: f32 = 200.0;

        // Find pets that are too far away from their owner
        let lost_pets: Vec<(Entity, Pos)> = (&entities, &positions, &alignments, &pets)
            .join()
            .filter_map(|(entity, pos, alignment, _)| match alignment {
                Alignment::Owned(owner_uid) => Some((entity, pos, owner_uid)),
                _ => None,
            })
            .filter_map(|(pet_entity, pet_pos, owner_uid)| {
                uid_allocator
                    .retrieve_entity_internal(owner_uid.0)
                    .and_then(|owner_entity| {
                        match (positions.get(owner_entity), physics.get(owner_entity)) {
                            (Some(position), Some(physics)) => {
                                Some((pet_entity, position, physics, pet_pos))
                            },
                            _ => None,
                        }
                    })
            })
            .filter(|(_, owner_pos, owner_physics, pet_pos)| {
                // Don't teleport pets to the player if they're in the air, nobody wants
                // pets to go splat :(
                owner_physics.state.on_ground.is_some()
                    && owner_pos.0.distance_squared(pet_pos.0) > LOST_PET_DISTANCE_THRESHOLD.powi(2)
            })
            .map(|(entity, owner_pos, _, _)| (entity, *owner_pos))
            .collect();

        for (pet_entity, owner_pos) in lost_pets.iter() {
            if let Some(mut pet_pos) = positions.get_mut(*pet_entity) {
                // Move the pets to their owner's position
                // TODO: Create a teleportation event to handle this instead of
                // processing the entity position move here
                pet_pos.0 = terrain
                    .find_space(owner_pos.0.map(|e| e.floor() as i32))
                    .map(|e| e as f32);
            }
        }
    }
}
