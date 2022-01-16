use specs::{world::WorldExt, Builder, Entity as EcsEntity, Join};
use tracing::error;
use vek::*;

use common::{
    assets,
    comp::{
        self,
        agent::{AgentEvent, Sound, SoundKind},
        dialogue::Subject,
        inventory::slot::EquipSlot,
        item,
        slot::Slot,
        tool::ToolKind,
        Inventory, Pos, SkillGroupKind,
    },
    consts::{MAX_MOUNT_RANGE, SOUND_TRAVEL_DIST_PER_VOLUME},
    outcome::Outcome,
    terrain::{Block, SpriteKind},
    uid::Uid,
    vol::ReadVol,
    mounting::{Mount, Rider, Mounting},
    link::Is,
};
use common_net::{msg::ServerGeneral, sync::WorldSyncExt};

use crate::{
    client::Client,
    presence::{Presence, RegionSubscription},
    state_ext::StateExt,
    Server,
};

use crate::pet::tame_pet;
use hashbrown::{HashMap, HashSet};
use lazy_static::lazy_static;
use serde::Deserialize;
use std::iter::FromIterator;

pub fn handle_lantern(server: &mut Server, entity: EcsEntity, enable: bool) {
    let ecs = server.state_mut().ecs();

    let lantern_exists = ecs
        .read_storage::<comp::LightEmitter>()
        .get(entity)
        .map_or(false, |light| light.strength > 0.0);

    if lantern_exists != enable {
        if !enable {
            server
                .state_mut()
                .ecs()
                .write_storage::<comp::LightEmitter>()
                .remove(entity);
        } else if ecs // Only enable lantern if entity is alive
            .read_storage::<comp::Health>()
            .get(entity)
            .map_or(true, |h| !h.is_dead)
        {
            let inventory_storage = ecs.read_storage::<Inventory>();
            let lantern_opt = inventory_storage
                .get(entity)
                .and_then(|inventory| inventory.equipped(EquipSlot::Lantern))
                .and_then(|item| {
                    if let comp::item::ItemKind::Lantern(l) = item.kind() {
                        Some(l)
                    } else {
                        None
                    }
                });
            if let Some(lantern) = lantern_opt {
                let _ =
                    ecs.write_storage::<comp::LightEmitter>()
                        .insert(entity, comp::LightEmitter {
                            col: lantern.color(),
                            strength: lantern.strength(),
                            flicker: 0.35,
                            animated: true,
                        });
            }
        }
    }
}

pub fn handle_npc_interaction(server: &mut Server, interactor: EcsEntity, npc_entity: EcsEntity) {
    let state = server.state_mut();
    if let Some(agent) = state
        .ecs()
        .write_storage::<comp::Agent>()
        .get_mut(npc_entity)
    {
        if agent.target.is_none() {
            if let Some(interactor_uid) = state.ecs().uid_from_entity(interactor) {
                agent
                    .inbox
                    .push_back(AgentEvent::Talk(interactor_uid, Subject::Regular));
            }
        }
    }
}

pub fn handle_mount(server: &mut Server, rider: EcsEntity, mount: EcsEntity) {
    let state = server.state_mut();

    if state
        .ecs()
        .read_storage::<Is<Rider>>()
        .get(rider)
        .is_none()
    {
        let not_mounting_yet = state
            .ecs()
            .read_storage::<Is<Mount>>()
            .get(mount)
            .is_none();

        let within_range = || {
            let positions = state.ecs().read_storage::<comp::Pos>();
            within_mounting_range(positions.get(rider), positions.get(mount))
        };
        let healths = state.ecs().read_storage::<comp::Health>();
        let alive = |e| healths.get(e).map_or(true, |h| !h.is_dead);

        if not_mounting_yet && within_range() && alive(rider) && alive(mount) {
            let uids = state.ecs().read_storage::<Uid>();
            if let (Some(rider_uid), Some(mount_uid)) =
                (uids.get(rider).copied(), uids.get(mount).copied())
            {
                let is_pet = match state.ecs().read_storage::<comp::Alignment>().get(mount) {
                    Some(comp::Alignment::Owned(owner)) if *owner == rider_uid => true,
                    _ => false,
                };

                if is_pet {
                    drop(uids);
                    drop(healths);
                    let _ = state.link(Mounting {
                        mount: mount_uid,
                        rider: rider_uid,
                    });
                }
            }
        }
    }
}

pub fn handle_unmount(server: &mut Server, rider: EcsEntity) {
    let state = server.state_mut();
    state
        .ecs()
        .write_storage::<Is<Rider>>()
        .remove(rider);
}

/// FIXME: This code is dangerous and needs to be refactored.  We can't just
/// comment it out, but it needs to be fixed for a variety of reasons.  Get rid
/// of this ASAP!
pub fn handle_possess(server: &mut Server, possessor_uid: Uid, possesse_uid: Uid) {
    let ecs = server.state.ecs();
    if let (Some(possessor), Some(possesse)) = (
        ecs.entity_from_uid(possessor_uid.into()),
        ecs.entity_from_uid(possesse_uid.into()),
    ) {
        // Check that entities still exist
        if !possessor.gen().is_alive()
            || !ecs.is_alive(possessor)
            || !possesse.gen().is_alive()
            || !ecs.is_alive(possesse)
        {
            error!(
                "Error possessing! either the possessor entity or possesse entity no longer exists"
            );
            return;
        }

        if ecs.read_storage::<Client>().get(possesse).is_some() {
            error!("can't possess other players");
            return;
        }

        match (|| -> Option<Result<(), specs::error::Error>> {
            let mut clients = ecs.write_storage::<Client>();
            let c = clients.remove(possessor)?;
            clients.insert(possesse, c).ok()?;
            let playerlist_messages = if let Some(client) = clients.get(possesse) {
                client.send_fallible(ServerGeneral::SetPlayerEntity(possesse_uid));
                // If a player is posessing non player, add possesse to playerlist as player and
                // remove old player
                if let Some(possessor_player) = ecs.read_storage::<comp::Player>().get(possessor) {
                    let admins = ecs.read_storage::<comp::Admin>();
                    let entity_possession_msg = ServerGeneral::PlayerListUpdate(
                        common_net::msg::server::PlayerListUpdate::Add(
                            possesse_uid,
                            common_net::msg::server::PlayerInfo {
                                player_alias: possessor_player.alias.clone(),
                                is_online: true,
                                is_moderator: admins.get(possessor).is_some(),
                                character: ecs.read_storage::<comp::Stats>().get(possesse).map(
                                    |s| common_net::msg::CharacterInfo {
                                        name: s.name.clone(),
                                    },
                                ),
                            },
                        ),
                    );
                    let remove_old_player_msg = ServerGeneral::PlayerListUpdate(
                        common_net::msg::server::PlayerListUpdate::Remove(possessor_uid),
                    );

                    // Send msg to new possesse client now because it is not yet considered a player
                    // and will be missed by notify_players
                    client.send_fallible(entity_possession_msg.clone());
                    client.send_fallible(remove_old_player_msg.clone());
                    Some((remove_old_player_msg, entity_possession_msg))
                } else {
                    None
                }
            } else {
                None
            };
            drop(clients);
            if let Some((remove_player, possess_entity)) = playerlist_messages {
                server.state().notify_players(possess_entity);
                server.state().notify_players(remove_player);
            }
            //optional entities
            let mut players = ecs.write_storage::<comp::Player>();
            let mut presence = ecs.write_storage::<Presence>();
            let mut subscriptions = ecs.write_storage::<RegionSubscription>();
            let mut admins = ecs.write_storage::<comp::Admin>();
            let mut waypoints = ecs.write_storage::<comp::Waypoint>();
            players
                .remove(possessor)
                .map(|p| players.insert(possesse, p).ok()?);
            presence
                .remove(possessor)
                .map(|p| presence.insert(possesse, p).ok()?);
            subscriptions
                .remove(possessor)
                .map(|s| subscriptions.insert(possesse, s).ok()?);
            admins
                .remove(possessor)
                .map(|a| admins.insert(possesse, a).ok()?);
            waypoints
                .remove(possessor)
                .map(|w| waypoints.insert(possesse, w).ok()?);

            Some(Ok(()))
        })() {
            Some(Ok(())) => (),
            Some(Err(e)) => {
                error!(?e, ?possesse, "Error inserting component during possession");
                return;
            },
            None => {
                error!(?possessor, "Error removing component during possession");
                return;
            },
        }

        // Put possess item into loadout
        let mut inventories = ecs.write_storage::<Inventory>();
        let mut inventory = inventories
            .entry(possesse)
            .expect("Nobody has &mut World, so there's no way to delete an entity.")
            .or_insert(Inventory::new_empty());

        let debug_item = comp::Item::new_from_asset_expect("common.items.debug.admin_stick");
        if let item::ItemKind::Tool(_) = debug_item.kind() {
            assert!(
                inventory
                    .swap(
                        Slot::Equip(EquipSlot::ActiveMainhand),
                        Slot::Equip(EquipSlot::InactiveMainhand),
                    )
                    .first()
                    .is_none(),
                "Swapping active and inactive mainhands never results in leftover items",
            );

            inventory.replace_loadout_item(EquipSlot::ActiveMainhand, Some(debug_item));
        }

        // Remove will of the entity
        ecs.write_storage::<comp::Agent>().remove(possesse);
        // Reset controller of former shell
        ecs.write_storage::<comp::Controller>()
            .get_mut(possessor)
            .map(|c| c.reset());
    }
}

fn within_mounting_range(player_position: Option<&Pos>, mount_position: Option<&Pos>) -> bool {
    match (player_position, mount_position) {
        (Some(ppos), Some(ipos)) => ppos.0.distance_squared(ipos.0) < MAX_MOUNT_RANGE.powi(2),
        _ => false,
    }
}

#[derive(Deserialize)]
struct ResourceExperienceManifest(HashMap<String, u32>);

impl assets::Asset for ResourceExperienceManifest {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

lazy_static! {
    static ref RESOURCE_EXPERIENCE_MANIFEST: assets::AssetHandle<ResourceExperienceManifest> =
        assets::AssetExt::load_expect("server.manifests.resource_experience_manifest");
}

pub fn handle_mine_block(
    server: &mut Server,
    entity: EcsEntity,
    pos: Vec3<i32>,
    tool: Option<ToolKind>,
) {
    let state = server.state_mut();
    if state.can_set_block(pos) {
        let block = state.terrain().get(pos).ok().copied();
        if let Some(block) = block.filter(|b| b.mine_tool().map_or(false, |t| Some(t) == tool)) {
            // Drop item if one is recoverable from the block
            if let Some(mut item) = comp::Item::try_reclaim_from_block(block) {
                if let Some(mut skillset) = state
                    .ecs()
                    .write_storage::<comp::SkillSet>()
                    .get_mut(entity)
                {
                    if let (Some(tool), Some(uid), Some(exp_reward)) = (
                        tool,
                        state.ecs().uid_from_entity(entity),
                        RESOURCE_EXPERIENCE_MANIFEST
                            .read()
                            .0
                            .get(item.item_definition_id()),
                    ) {
                        let skill_group = SkillGroupKind::Weapon(tool);
                        let mut outcomes = state.ecs().write_resource::<Vec<Outcome>>();
                        let positions = state.ecs().read_component::<comp::Pos>();
                        if let (Some(level_outcome), Some(pos)) = (
                            skillset.add_experience(skill_group, *exp_reward),
                            positions.get(entity),
                        ) {
                            outcomes.push(Outcome::SkillPointGain {
                                uid,
                                skill_tree: skill_group,
                                total_points: level_outcome,
                                pos: pos.0,
                            });
                        }
                        outcomes.push(Outcome::ExpChange {
                            uid,
                            exp: *exp_reward,
                            xp_pools: HashSet::from_iter(vec![skill_group]),
                        });
                    }
                    use common::comp::skills::{MiningSkill, Skill, SKILL_MODIFIERS};
                    use rand::Rng;
                    let mut rng = rand::thread_rng();

                    let need_double_ore = |rng: &mut rand::rngs::ThreadRng| {
                        let chance_mod = f64::from(SKILL_MODIFIERS.mining_tree.ore_gain);
                        let skill_level = skillset
                            .skill_level(Skill::Pick(MiningSkill::OreGain))
                            .unwrap_or(0);

                        rng.gen_bool(chance_mod * f64::from(skill_level))
                    };
                    let need_double_gem = |rng: &mut rand::rngs::ThreadRng| {
                        let chance_mod = f64::from(SKILL_MODIFIERS.mining_tree.gem_gain);
                        let skill_level = skillset
                            .skill_level(Skill::Pick(MiningSkill::GemGain))
                            .unwrap_or(0);

                        rng.gen_bool(chance_mod * f64::from(skill_level))
                    };

                    let double_gain = (item.item_definition_id().contains("mineral.ore.")
                        && need_double_ore(&mut rng))
                        || (item.item_definition_id().contains("mineral.gem.")
                            && need_double_gem(&mut rng));

                    if double_gain {
                        // Ignore non-stackable errors
                        let _ = item.increase_amount(1);
                    }
                }
                state
                    .create_object(Default::default(), comp::object::Body::Pouch)
                    .with(comp::Pos(pos.map(|e| e as f32) + Vec3::new(0.5, 0.5, 0.0)))
                    .with(item)
                    .build();
            }

            state.set_block(pos, block.into_vacant());
            state
                .ecs()
                .write_resource::<Vec<Outcome>>()
                .push(Outcome::BreakBlock {
                    pos,
                    color: block.get_color(),
                });
        }
    }
}

pub fn handle_sound(server: &mut Server, sound: &Sound) {
    let ecs = &server.state.ecs();
    let positions = &ecs.read_storage::<comp::Pos>();
    let agents = &mut ecs.write_storage::<comp::Agent>();

    // TODO: Reduce the complexity of this problem by using spatial partitioning
    // system
    for (agent, agent_pos) in (agents, positions).join() {
        // TODO: Use pathfinding for more dropoff around obstacles
        let agent_dist_sqrd = agent_pos.0.distance_squared(sound.pos);
        let sound_travel_dist_sqrd = (sound.vol * SOUND_TRAVEL_DIST_PER_VOLUME).powi(2);

        let vol_dropoff = agent_dist_sqrd / sound_travel_dist_sqrd * sound.vol;
        let propagated_sound = sound.with_new_vol(sound.vol - vol_dropoff);

        let can_hear_sound = propagated_sound.vol > 0.00;
        let should_hear_sound = agent_dist_sqrd < agent.psyche.listen_dist.powi(2);

        if can_hear_sound && should_hear_sound {
            agent
                .inbox
                .push_back(AgentEvent::ServerSound(propagated_sound));
        }
    }

    // Attempt to turn this sound into an outcome to be received by frontends.
    if let Some(outcome) = match sound.kind {
        SoundKind::Utterance(kind, body) => Some(Outcome::Utterance {
            kind,
            pos: sound.pos,
            body,
        }),
        _ => None,
    } {
        ecs.write_resource::<Vec<Outcome>>().push(outcome);
    }
}

pub fn handle_create_sprite(server: &mut Server, pos: Vec3<i32>, sprite: SpriteKind) {
    let state = server.state_mut();
    if state.can_set_block(pos) {
        let block = state.terrain().get(pos).ok().copied();
        if block.map_or(false, |b| (*b).is_air()) {
            let new_block = state
                .get_block(pos)
                .unwrap_or_else(|| Block::air(SpriteKind::Empty))
                .with_sprite(sprite);
            server.state.set_block(pos, new_block);
        }
    }
}

pub fn handle_tame_pet(server: &mut Server, pet_entity: EcsEntity, owner_entity: EcsEntity) {
    // TODO: Raise outcome to send to clients to play sound/render an indicator
    // showing taming success?
    tame_pet(server.state.ecs(), pet_entity, owner_entity);
}
