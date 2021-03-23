use tracing::{debug, trace};

/// assume deserialization is exactly what we hardcoded.
/// When it fails, we can generate a wrong version error
/// message and people stop annoying xMAC94x
/// see src/server/connecton_handler.rs
pub(crate) async fn validate_dump_from_server(
    mut validate_stream: network::Stream,
) -> Result<(), ()> {
    use common::{
        character::Character,
        comp::{
            inventory::Inventory,
            item::{tool::ToolKind, Reagent},
            skills::SkillGroupKind,
        },
        outcome::Outcome,
        terrain::{biome::BiomeKind, TerrainChunkMeta},
        trade::{Good, PendingTrade, Trades},
        uid::Uid,
    };
    use common_net::msg::{world_msg::EconomyInfo, ServerGeneral};
    use std::collections::HashMap;
    use vek::*;

    trace!("check Character (1)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::CharacterListUpdate(vec)) => {
            check_eq(&vec.len(), 1)?;
            let item = &vec[0];
            check_eq(&item.character, Character {
                id: Some(1337),
                alias: "foobar".to_owned(),
            })?;
            check(item.body.is_humanoid())?;
            let inv = Inventory::new_empty()
                .equipped_items()
                .cloned()
                .collect::<Vec<_>>();
            let inv_remote = item.inventory.equipped_items().cloned().collect::<Vec<_>>();
            check_eq(&inv_remote, inv)?;
        },
        _ => return Err(()),
    }

    trace!("check Outcomes (2)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::Outcomes(vec)) => {
            check_eq(&vec.len(), 3)?;
            check_eq(&vec[0], Outcome::Explosion {
                pos: Vec3::new(1.0, 2.0, 3.0),
                power: 4.0,
                radius: 5.0,
                is_attack: true,
                reagent: Some(Reagent::Blue),
            })?;
            check_eq(&vec[1], Outcome::SkillPointGain {
                uid: Uid::from(1337u64),
                pos: Vec3::new(2.0, 4.0, 6.0),
                skill_tree: SkillGroupKind::Weapon(ToolKind::Empty),
                total_points: 99,
            })?;
            check_eq(&vec[2], Outcome::BreakBlock {
                pos: Vec3::new(1, 2, 3),
                color: Some(Rgb::new(0u8, 8u8, 13u8)),
            })?;
        },
        _ => return Err(()),
    }

    trace!("check Terrain (3)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::TerrainChunkUpdate { key, chunk }) => {
            check_eq(&key, Vec2::new(42, 1337))?;
            let chunk = chunk?;
            check_eq(chunk.meta(), TerrainChunkMeta::void())?;
            check_eq(&chunk.get_min_z(), 5)?;
        },
        _ => return Err(()),
    }

    trace!("check Componity Sync (4)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::CompSync(item)) => {
            check_eq(&item.comp_updates.len(), 2)?;
            //check_eq(&item.comp_updates[0],
            // CompUpdateKind::Inserted(comp::Pos(Vec3::new(42.1337, 0.0,
            // 0.0))))?; check_eq(&item.comp_updates[1],
            // CompUpdateKind::Inserted(comp::Pos(Vec3::new(0.0, 42.1337,
            // 0.0))))?;
        },
        _ => return Err(()),
    }

    trace!("check Pending Trade (5)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::UpdatePendingTrade(tradeid, pending)) => {
            let uid = Uid::from(70);
            check_eq(&tradeid, Trades::default().begin_trade(uid, uid))?;
            check_eq(&pending, PendingTrade::new(uid, Uid::from(71)))?;
        },
        _ => return Err(()),
    }

    trace!("check Economy (6)");
    match validate_stream.recv::<ServerGeneral>().await {
        Ok(ServerGeneral::SiteEconomy(info)) => {
            check_eq(&info, EconomyInfo {
                id: 99,
                population: 55,
                stock: vec![
                    (Good::Wood, 50.0),
                    (Good::Tools, 33.3),
                    (Good::Coin, 9000.1),
                ]
                .into_iter()
                .collect::<HashMap<Good, f32>>(),
                labor_values: HashMap::new(),
                values: vec![
                    (Good::RoadSecurity, 1.0),
                    (Good::Terrain(BiomeKind::Forest), 1.0),
                ]
                .into_iter()
                .collect::<HashMap<Good, f32>>(),
                labors: Vec::new(),
                last_exports: HashMap::new(),
                resources: HashMap::new(),
            })?;
        },
        _ => return Err(()),
    }

    Ok(())
}

fn check_eq<T: PartialEq + core::fmt::Debug>(one: &T, two: T) -> Result<(), ()> {
    if one == &two {
        Ok(())
    } else {
        debug!(?one, ?two, "failed check");
        Err(())
    }
}

fn check(b: bool) -> Result<(), ()> { if b { Ok(()) } else { Err(()) } }
