use crate::ecs::comp::HpFloaterList;
use common::{
    comp::{Health, Pos},
    resources::{DeltaTime, PlayerEntity},
};
use common_ecs::{Job, Origin, Phase, System};
use specs::{Entities, Join, Read, ReadStorage, WriteStorage};

// How long floaters last (in seconds)
pub const HP_SHOWTIME: f32 = 3.0;
pub const MY_HP_SHOWTIME: f32 = 2.5;
pub const HP_ACCUMULATETIME: f32 = 1.0;

#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    type SystemData = (
        Entities<'a>,
        Read<'a, PlayerEntity>,
        Read<'a, DeltaTime>,
        ReadStorage<'a, Pos>,
        ReadStorage<'a, Health>,
        WriteStorage<'a, HpFloaterList>,
    );

    const NAME: &'static str = "floater";
    const ORIGIN: Origin = Origin::Frontend("voxygen");
    const PHASE: Phase = Phase::Create;

    #[allow(clippy::blocks_in_if_conditions)] // TODO: Pending review in #587
    fn run(
        _job: &mut Job<Self>,
        (entities, my_entity, dt, pos, healths, mut hp_floater_lists): Self::SystemData,
    ) {
        // Add hp floater lists to all entities with health and a position
        // Note: necessary in order to know last_hp
        for (entity, last_hp) in (&entities, &healths, &pos, !&hp_floater_lists)
            .join()
            .map(|(e, h, _, _)| (e, h.current()))
            .collect::<Vec<_>>()
        {
            let _ = hp_floater_lists.insert(entity, HpFloaterList {
                floaters: Vec::new(),
                last_hp,
                time_since_last_dmg_by_me: None,
            });
        }

        for hp_floater_list in (&mut hp_floater_lists).join() {
            // Increment timer for time since last damaged by me
            hp_floater_list
                .time_since_last_dmg_by_me
                .as_mut()
                .map(|t| *t += dt.0);
        }

        // Remove floater lists on entities without health or without position
        for entity in (&entities, !&healths, &hp_floater_lists)
            .join()
            .map(|(e, _, _)| e)
            .collect::<Vec<_>>()
        {
            hp_floater_lists.remove(entity);
        }
        for entity in (&entities, !&pos, &hp_floater_lists)
            .join()
            .map(|(e, _, _)| e)
            .collect::<Vec<_>>()
        {
            hp_floater_lists.remove(entity);
        }

        // Maintain existing floaters
        for (
            entity,
            HpFloaterList {
                ref mut floaters,
                ref last_hp,
                ..
            },
        ) in (&entities, &mut hp_floater_lists).join()
        {
            for mut floater in floaters.iter_mut() {
                // Increment timer
                floater.timer += dt.0;
            }
            // Clear floaters if newest floater is past show time or health runs out
            if floaters.last().map_or(false, |f| {
                f.timer
                    > if Some(entity) != my_entity.0 {
                        HP_SHOWTIME
                    } else {
                        MY_HP_SHOWTIME
                    }
                    || last_hp.abs() < Health::HEALTH_EPSILON
            }) {
                floaters.clear();
            }
        }
    }
}
