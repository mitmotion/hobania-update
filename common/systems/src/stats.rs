use common::{
    combat,
    comp::{
        self,
        item::MaterialStatManifest,
        skills::{GeneralSkill, Skill},
        Body, CharacterState, Combo, Energy, Health, Inventory, Poise, Pos, SkillSet, Stats,
        StatsModifier,
    },
    event::{EventBus, ServerEvent},
    resources::{DeltaTime, EntitiesDiedLastTick, Time},
    states,
};
use common_ecs::{Job, Origin, Phase, System};
use specs::{
    shred::ResourceId, Entities, Join, Read, ReadExpect, ReadStorage, SystemData, World, Write,
    WriteStorage,
};

const ENERGY_REGEN_ACCEL: f32 = 1.0;
const SIT_ENERGY_REGEN_ACCEL: f32 = 2.5;
const POISE_REGEN_ACCEL: f32 = 2.0;

#[derive(SystemData)]
pub struct ReadData<'a> {
    entities: Entities<'a>,
    dt: Read<'a, DeltaTime>,
    time: Read<'a, Time>,
    server_bus: Read<'a, EventBus<ServerEvent>>,
    positions: ReadStorage<'a, Pos>,
    bodies: ReadStorage<'a, Body>,
    char_states: ReadStorage<'a, CharacterState>,
    inventories: ReadStorage<'a, Inventory>,
    msm: ReadExpect<'a, MaterialStatManifest>,
}

/// This system kills players, levels them up, and regenerates energy.
#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    type SystemData = (
        ReadData<'a>,
        WriteStorage<'a, Stats>,
        WriteStorage<'a, SkillSet>,
        WriteStorage<'a, Health>,
        WriteStorage<'a, Poise>,
        WriteStorage<'a, Energy>,
        WriteStorage<'a, Combo>,
        Write<'a, EntitiesDiedLastTick>,
    );

    const NAME: &'static str = "stats";
    const ORIGIN: Origin = Origin::Common;
    const PHASE: Phase = Phase::Create;

    fn run(
        _job: &mut Job<Self>,
        (
            read_data,
            stats,
            mut skill_sets,
            mut healths,
            mut poises,
            mut energies,
            mut combos,
            mut entities_died_last_tick,
        ): Self::SystemData,
    ) {
        entities_died_last_tick.0.clear();
        let mut server_event_emitter = read_data.server_bus.emitter();
        let dt = read_data.dt.0;

        // Update stats
        for (entity, stats, mut health, pos, mut energy, inventory) in (
            &read_data.entities,
            &stats,
            &mut healths,
            &read_data.positions,
            &mut energies,
            read_data.inventories.maybe(),
        )
            .join()
        {
            let set_dead = { health.should_die() && !health.is_dead };

            if set_dead {
                let cloned_entity = (entity, *pos);
                entities_died_last_tick.0.push(cloned_entity);
                server_event_emitter.emit(ServerEvent::Destroy {
                    entity,
                    cause: health.last_change,
                });

                health.is_dead = true;
            }
            let stat = stats;

            if let Some(new_max) = health.needs_maximum_update(stat.max_health_modifiers) {
                // Only call this if we need to since mutable access will trigger sending an
                // update to the client.
                health.update_internal_integer_maximum(new_max);
            }

            // Calculates energy scaling from stats and inventory
            let energy_mods = StatsModifier {
                add_mod: stat.max_energy_modifiers.add_mod
                    + combat::compute_max_energy_mod(inventory, &read_data.msm),
                mult_mod: stat.max_energy_modifiers.mult_mod,
            };

            if let Some(new_max) = energy.needs_maximum_update(energy_mods) {
                // Only call this if we need to since mutable access will trigger sending an
                // update to the client.
                energy.update_internal_integer_maximum(new_max);
            }
        }

        // Apply effects from leveling skills
        for (mut skill_set, mut health, mut energy, body) in (
            &mut skill_sets,
            &mut healths,
            &mut energies,
            &read_data.bodies,
        )
            .join()
        {
            if skill_set.modify_health {
                let health_level = skill_set
                    .skill_level(Skill::General(GeneralSkill::HealthIncrease))
                    .unwrap_or(0);
                health.update_max_hp(*body, health_level);
                skill_set.modify_health = false;
            }
            if skill_set.modify_energy {
                let energy_level = skill_set
                    .skill_level(Skill::General(GeneralSkill::EnergyIncrease))
                    .unwrap_or(0);
                energy.update_max_energy(*body, energy_level);
                skill_set.modify_energy = false;
            }
        }

        // Update energies and poises
        for (character_state, mut energy, mut poise) in
            (&read_data.char_states, &mut energies, &mut poises).join()
        {
            match character_state {
                // Sitting accelerates recharging energy the most
                CharacterState::Sit => {
                    if energy.needs_regen() {
                        energy.regen(SIT_ENERGY_REGEN_ACCEL, dt);
                    }
                    if poise.needs_regen() {
                        poise.regen(POISE_REGEN_ACCEL, dt, *read_data.time);
                    }
                },
                // Accelerate recharging energy.
                CharacterState::Idle(_)
                | CharacterState::Talk
                | CharacterState::Dance
                | CharacterState::Glide(_)
                | CharacterState::Skate(_)
                | CharacterState::GlideWield(_)
                | CharacterState::Wielding(_)
                | CharacterState::Equipping(_)
                | CharacterState::Boost(_)
                | CharacterState::ComboMelee2(states::combo_melee2::Data {
                    stage_section: None,
                    ..
                }) => {
                    if energy.needs_regen() {
                        energy.regen(ENERGY_REGEN_ACCEL, dt);
                    }
                    if poise.needs_regen() {
                        poise.regen(POISE_REGEN_ACCEL, dt, *read_data.time);
                    }
                },
                // Ability use does not regen and sets the rate back to zero.
                CharacterState::BasicMelee(_)
                | CharacterState::DashMelee(_)
                | CharacterState::LeapMelee(_)
                | CharacterState::LeapShockwave(_)
                | CharacterState::SpinMelee(_)
                | CharacterState::ComboMelee(_)
                | CharacterState::ComboMelee2(_)
                | CharacterState::BasicRanged(_)
                | CharacterState::Music(_)
                | CharacterState::ChargedMelee(_)
                | CharacterState::ChargedRanged(_)
                | CharacterState::RepeaterRanged(_)
                | CharacterState::Shockwave(_)
                | CharacterState::BasicBeam(_)
                | CharacterState::BasicAura(_)
                | CharacterState::Blink(_)
                | CharacterState::Climb(_)
                | CharacterState::BasicSummon(_)
                | CharacterState::SelfBuff(_)
                | CharacterState::SpriteSummon(_)
                | CharacterState::FinisherMelee(_)
                | CharacterState::DiveMelee(_)
                | CharacterState::RiposteMelee(_)
                | CharacterState::RapidMelee(_) => {
                    if energy.needs_regen_rate_reset() {
                        energy.reset_regen_rate();
                    }
                },
                // Abilities that temporarily stall energy gain, but preserve regen_rate.
                CharacterState::Roll(_)
                | CharacterState::Wallrun(_)
                | CharacterState::Stunned(_)
                | CharacterState::BasicBlock(_)
                | CharacterState::UseItem(_)
                | CharacterState::SpriteInteract(_) => {},
            }
        }

        // Decay combo
        for (_, mut combo) in (&read_data.entities, &mut combos).join() {
            if combo.counter() > 0
                && read_data.time.0 - combo.last_increase() > comp::combo::COMBO_DECAY_START
            {
                combo.reset();
            }
        }
    }
}
