use common::{
    combat::{self, AttackOptions, AttackSource, AttackerInfo, TargetInfo},
    comp::{
        agent::{Sound, SoundKind},
        melee::MultiTarget,
        Alignment, Body, CharacterState, Combo, Energy, Group, Health, Inventory, Melee, Ori,
        Player, Pos, Scale, Stats,
    },
    event::{EventBus, ServerEvent},
    outcome::Outcome,
    resources::Time,
    uid::{Uid, UidAllocator},
    util::Dir,
    GroupTarget,
};
use common_ecs::{Job, Origin, Phase, System};
use itertools::Itertools;
use specs::{
    shred::ResourceId, Entities, Join, Read, ReadStorage, SystemData, World, WriteStorage,
};
use vek::*;

#[derive(SystemData)]
pub struct ReadData<'a> {
    time: Read<'a, Time>,
    uid_allocator: Read<'a, UidAllocator>,
    entities: Entities<'a>,
    players: ReadStorage<'a, Player>,
    uids: ReadStorage<'a, Uid>,
    positions: ReadStorage<'a, Pos>,
    orientations: ReadStorage<'a, Ori>,
    alignments: ReadStorage<'a, Alignment>,
    scales: ReadStorage<'a, Scale>,
    bodies: ReadStorage<'a, Body>,
    healths: ReadStorage<'a, Health>,
    energies: ReadStorage<'a, Energy>,
    inventories: ReadStorage<'a, Inventory>,
    groups: ReadStorage<'a, Group>,
    char_states: ReadStorage<'a, CharacterState>,
    server_bus: Read<'a, EventBus<ServerEvent>>,
    stats: ReadStorage<'a, Stats>,
    combos: ReadStorage<'a, Combo>,
}

/// This system is responsible for handling accepted inputs like moving or
/// attacking
#[derive(Default)]
pub struct Sys;

impl<'a> System<'a> for Sys {
    type SystemData = (
        ReadData<'a>,
        WriteStorage<'a, Melee>,
        Read<'a, EventBus<Outcome>>,
    );

    const NAME: &'static str = "melee";
    const ORIGIN: Origin = Origin::Common;
    const PHASE: Phase = Phase::Create;

    fn run(_job: &mut Job<Self>, (read_data, mut melee_attacks, outcomes): Self::SystemData) {
        let mut server_emitter = read_data.server_bus.emitter();
        let mut outcomes_emitter = outcomes.emitter();

        // Attacks
        for (attacker, uid, pos, ori, melee_attack, body) in (
            &read_data.entities,
            &read_data.uids,
            &read_data.positions,
            &read_data.orientations,
            &mut melee_attacks,
            &read_data.bodies,
        )
            .join()
        {
            if melee_attack.applied {
                continue;
            }
            server_emitter.emit(ServerEvent::Sound {
                sound: Sound::new(SoundKind::Melee, pos.0, 2.0, read_data.time.0),
            });
            melee_attack.applied = true;

            // Scales
            let eye_pos = pos.0 + Vec3::unit_z() * body.eye_height();
            let scale = read_data.scales.get(attacker).map_or(1.0, |s| s.0);
            // TODO: use Capsule Prisms instead of Cylinders
            let rad = body.max_radius() * scale;

            let melee_z = pos.0.z + 0.5 * body.height();
            let melee_z_range = (melee_z - melee_attack.range)..(melee_z + melee_attack.range);

            // Mine blocks broken by the attack
            if let Some((block_pos, tool)) = melee_attack.break_block {
                // Check distance to block
                if eye_pos.distance_squared(block_pos.map(|e| e as f32 + 0.5))
                    < (rad + scale * melee_attack.range).powi(2)
                {
                    server_emitter.emit(ServerEvent::MineBlock {
                        entity: attacker,
                        pos: block_pos,
                        tool,
                    });
                }
            }

            // Go through all other entities
            for (target, pos_b, health_b, body_b, uid_b) in (
                &read_data.entities,
                &read_data.positions,
                &read_data.healths,
                &read_data.bodies,
                &read_data.uids,
            )
                .join()
                .sorted_by_key(|(_, pos_b, _, _, _)| pos_b.0.distance_squared(pos.0) as u32)
            {
                // Unless the melee attack can hit multiple targets, stop the attack if it has
                // already hit 1 target
                if melee_attack.multi_target.is_none() && melee_attack.hit_count > 0 {
                    break;
                }

                let look_dir = *ori.look_dir();

                // 2D versions
                let pos2 = Vec2::from(pos.0);
                let pos_b2 = Vec2::<f32>::from(pos_b.0);
                let ori2 = Vec2::from(look_dir);

                // Scales
                let scale_b = read_data.scales.get(target).map_or(1.0, |s| s.0);
                let rad_b = body_b.max_radius() * scale_b;

                // Check if entity is dodging
                let target_dodging = read_data
                    .char_states
                    .get(target)
                    .map_or(false, |c_s| c_s.is_melee_dodge());

                // Check if it is a hit
                if attacker != target
                    && !health_b.is_dead
                    // Cylindrical wedge shaped attack field
                    && pos2.distance_squared(pos_b2) < (rad + rad_b + scale * melee_attack.range).powi(2)
                    // Checks if feet or head of b is contained in range of melee attack, or if melee attack origin is contained between feet and head of b (for large bodies/small melee attacks)
                    && (melee_z_range.contains(&pos_b.0.z) || melee_z_range.contains(&(pos_b.0.z + body_b.height())) || (pos_b.0.z..(pos_b.0.z + body_b.height())).contains(&melee_z))
                    && ori2.angle_between(pos_b2 - pos2) < melee_attack.max_angle + (rad_b / pos2.distance(pos_b2)).atan()
                {
                    // See if entities are in the same group
                    let same_group = read_data
                        .groups
                        .get(attacker)
                        .map(|group_a| Some(group_a) == read_data.groups.get(target))
                        .unwrap_or(false);

                    let target_group = if same_group {
                        GroupTarget::InGroup
                    } else {
                        GroupTarget::OutOfGroup
                    };

                    let dir = Dir::new((pos_b.0 - pos.0).try_normalized().unwrap_or(look_dir));

                    let attacker_info = Some(AttackerInfo {
                        entity: attacker,
                        uid: *uid,
                        group: read_data.groups.get(attacker),
                        energy: read_data.energies.get(attacker),
                        combo: read_data.combos.get(attacker),
                        inventory: read_data.inventories.get(attacker),
                    });

                    let target_info = TargetInfo {
                        entity: target,
                        uid: *uid_b,
                        inventory: read_data.inventories.get(target),
                        stats: read_data.stats.get(target),
                        health: read_data.healths.get(target),
                        pos: pos_b.0,
                        ori: read_data.orientations.get(target),
                        char_state: read_data.char_states.get(target),
                        energy: read_data.energies.get(target),
                    };

                    // PvP check
                    let may_harm = combat::may_harm(
                        &read_data.alignments,
                        &read_data.players,
                        &read_data.uid_allocator,
                        Some(attacker),
                        target,
                    );

                    let attack_options = AttackOptions {
                        target_dodging,
                        may_harm,
                        target_group,
                    };

                    let strength =
                        if let Some(MultiTarget::Scaling(scaling)) = melee_attack.multi_target {
                            1.0 + melee_attack.hit_count as f32 * scaling
                        } else {
                            1.0
                        };

                    let is_applied = melee_attack.attack.apply_attack(
                        attacker_info,
                        target_info,
                        dir,
                        attack_options,
                        strength,
                        AttackSource::Melee,
                        *read_data.time,
                        |e| server_emitter.emit(e),
                        |o| outcomes_emitter.emit(o),
                    );

                    if is_applied {
                        melee_attack.hit_count += 1;
                    }
                }
            }
        }
    }
}
