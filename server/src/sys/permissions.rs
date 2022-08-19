use chrono::Utc;
use common::comp::permissions::RuleSet;
use common_ecs::{Job, Origin, Phase, System};
use specs::{Component, Entities, Join, ReadStorage, WriteStorage};
use std::sync::Arc;

pub struct Role {
    pub name: String,
    rules: RuleSet,
}

pub struct RoleAccess {
    role: Arc<Role>,
    valid_until: chrono::DateTime<chrono::Utc>,
}

pub struct UserRoles(Vec<RoleAccess>);

impl Component for UserRoles {
    type Storage = specs::VecStorage<Self>;
}
// This system manages loot that exists in the world
#[derive(Default)]
pub struct Sys;
impl<'a> System<'a> for Sys {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, RuleSet>,
        ReadStorage<'a, UserRoles>,
    );

    const NAME: &'static str = "permissions";
    const ORIGIN: Origin = Origin::Server;
    const PHASE: Phase = Phase::Create;

    fn run(_job: &mut Job<Self>, (entities, mut rule_sets, user_roles): Self::SystemData) {
        // recalculate the role_access on top of all individual roles assigned
        let now = Utc::now();

        // Add PreviousPhysCache for all relevant entities
        for entity in (&entities, !&rule_sets)
            .join()
            .map(|(e, _)| e)
            .collect::<Vec<_>>()
        {
            let _ = rule_sets.insert(entity, RuleSet::default());
        }

        for (rule_set, user_roles) in (&mut rule_sets, &user_roles).join() {
            *rule_set = RuleSet::default();
            for access in &user_roles.0 {
                if access.valid_until > now {
                    rule_set.append(access.role.rules.clone())
                }
            }
        }
    }
}
