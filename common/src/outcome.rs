use crate::{combat::DamageContributor, comp, uid::Uid};
use comp::{beam, item::Reagent, poise::PoiseState, skillset::SkillGroupKind, UtteranceKind};
use hashbrown::HashSet;
use serde::{Deserialize, Serialize};
use vek::*;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DamageInfo {
    pub amount: f32,
    pub crit: Option<bool>,
    pub target: Uid,
    pub by: Option<DamageContributor>,
}

/// An outcome represents the final result of an instantaneous event. It implies
/// that said event has already occurred. It is not a request for that event to
/// occur, nor is it something that may be cancelled or otherwise altered. Its
/// primary purpose is to act as something for frontends (both server and
/// client) to listen to in order to receive feedback about events in the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Outcome {
    Explosion {
        pos: Vec3<f32>,
        power: f32,
        radius: f32,
        is_attack: bool,
        reagent: Option<Reagent>, // How can we better define this?
    },
    ProjectileShot {
        pos: Vec3<f32>,
        body: comp::Body,
        vel: Vec3<f32>,
    },
    ProjectileHit {
        pos: Vec3<f32>,
        body: comp::Body,
        vel: Vec3<f32>,
        source: Option<Uid>,
        target: Option<Uid>,
    },
    Beam {
        pos: Vec3<f32>,
        specifier: beam::FrontendSpecifier,
    },
    ExpChange {
        uid: Uid,
        exp: u32,
        xp_pools: HashSet<SkillGroupKind>,
    },
    SkillPointGain {
        uid: Uid,
        skill_tree: comp::skillset::SkillGroupKind,
        total_points: u16,
        // TODO: Access ECS to get position from Uid to conserve bandwidth
        pos: Vec3<f32>,
    },
    ComboChange {
        uid: Uid,
        combo: u32,
    },
    BreakBlock {
        pos: Vec3<i32>,
        color: Option<Rgb<u8>>,
    },
    SummonedCreature {
        pos: Vec3<f32>,
        body: comp::Body,
    },
    Damage {
        pos: Vec3<f32>,
        info: DamageInfo,
    },
    Death {
        pos: Vec3<f32>,
    },
    Block {
        pos: Vec3<f32>,
        parry: bool,
        uid: Uid,
    },
    PoiseChange {
        pos: Vec3<f32>,
        state: PoiseState,
    },
    GroundSlam {
        pos: Vec3<f32>,
    },
    Utterance {
        pos: Vec3<f32>,
        body: comp::Body,
        kind: UtteranceKind,
    },
    Glider {
        pos: Vec3<f32>,
        wielded: bool,
    },
}

impl Outcome {
    pub fn get_pos(&self) -> Option<Vec3<f32>> {
        match self {
            Outcome::Explosion { pos, .. }
            | Outcome::ProjectileShot { pos, .. }
            | Outcome::ProjectileHit { pos, .. }
            | Outcome::Beam { pos, .. }
            | Outcome::SkillPointGain { pos, .. }
            | Outcome::SummonedCreature { pos, .. }
            | Outcome::Damage { pos, .. }
            | Outcome::Death { pos, .. }
            | Outcome::Block { pos, .. }
            | Outcome::PoiseChange { pos, .. }
            | Outcome::GroundSlam { pos }
            | Outcome::Utterance { pos, .. }
            | Outcome::Glider { pos, .. } => Some(*pos),
            Outcome::BreakBlock { pos, .. } => Some(pos.map(|e| e as f32 + 0.5)),
            Outcome::ExpChange { .. } | Outcome::ComboChange { .. } => None,
        }
    }
}
