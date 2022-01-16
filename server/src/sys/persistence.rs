use crate::{persistence::character_updater, presence::Presence, sys::SysScheduler};
use common::{
    comp::{
        pet::{is_tameable, Pet},
        ActiveAbilities, Alignment, Body, Inventory, SkillSet, Stats, Waypoint,
    },
    uid::Uid,
};
use common_ecs::{Job, Origin, Phase, System};
use common_net::msg::PresenceKind;
use specs::{Join, ReadStorage, Write, WriteExpect};

#[derive(Default)]
pub struct Sys;

impl<'a> System<'a> for Sys {
    #[allow(clippy::type_complexity)]
    type SystemData = (
        ReadStorage<'a, Alignment>,
        ReadStorage<'a, Body>,
        ReadStorage<'a, Presence>,
        ReadStorage<'a, SkillSet>,
        ReadStorage<'a, Inventory>,
        ReadStorage<'a, Uid>,
        ReadStorage<'a, Waypoint>,
        ReadStorage<'a, Pet>,
        ReadStorage<'a, Stats>,
        ReadStorage<'a, ActiveAbilities>,
        WriteExpect<'a, character_updater::CharacterUpdater>,
        Write<'a, SysScheduler<Self>>,
    );

    const NAME: &'static str = "persistence";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(
        _job: &mut Job<Self>,
        (
            alignments,
            bodies,
            presences,
            player_skill_set,
            player_inventories,
            uids,
            player_waypoints,
            pets,
            stats,
            active_abilities,
            mut updater,
            mut scheduler,
        ): Self::SystemData,
    ) {
        if scheduler.should_run() {
            updater.batch_update(
                (
                    &presences,
                    &player_skill_set,
                    &player_inventories,
                    &uids,
                    player_waypoints.maybe(),
                    &active_abilities,
                )
                    .join()
                    .filter_map(
                        |(
                            presence,
                            skill_set,
                            inventory,
                            player_uid,
                            waypoint,
                            active_abilities,
                        )| match presence.kind {
                            PresenceKind::Character(id) => {
                                let pets = (&alignments, &bodies, &stats, &pets)
                                    .join()
                                    .filter_map(|(alignment, body, stats, pet)| match alignment {
                                        // Don't try to persist non-tameable pets (likely spawned
                                        // using /spawn) since there isn't any code to handle
                                        // persisting them
                                        Alignment::Owned(ref pet_owner)
                                            if pet_owner == player_uid && is_tameable(body) =>
                                        {
                                            Some(((*pet).clone(), *body, stats.clone()))
                                        },
                                        _ => None,
                                    })
                                    .collect();

                                Some((id, skill_set, inventory, pets, waypoint, active_abilities))
                            },
                            PresenceKind::Spectator => None,
                        },
                    ),
            );
        }
    }
}
