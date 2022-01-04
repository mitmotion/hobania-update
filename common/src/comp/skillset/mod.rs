use crate::{
    assets::{self, Asset, AssetExt},
    comp::{
        item::tool::ToolKind,
        skills::{GeneralSkill, Skill},
    },
};
use hashbrown::HashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use specs::{Component, DerefFlaggedStorage};
use specs_idvs::IdvStorage;
use std::{collections::BTreeSet, hash::Hash};
use tracing::{trace, warn};

pub mod skills;

/// BTreeSet is used here to ensure that skills are ordered. This is important
/// to ensure that the hash created from it is consistent so that we don't
/// needlessly force a respec when loading skills from persistence.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillTreeMap(HashMap<SkillGroupKind, BTreeSet<Skill>>);

impl Asset for SkillTreeMap {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

pub struct SkillGroupDef {
    pub skills: BTreeSet<Skill>,
    pub total_skill_point_cost: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillLevelMap(HashMap<Skill, u16>);

impl Asset for SkillLevelMap {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkillPrerequisitesMap(HashMap<Skill, HashMap<Skill, u16>>);

impl Asset for SkillPrerequisitesMap {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

lazy_static! {
    // Determines the skills that comprise each skill group.
    //
    // This data is used to determine which of a player's skill groups a
    // particular skill should be added to when a skill unlock is requested.
    pub static ref SKILL_GROUP_DEFS: HashMap<SkillGroupKind, SkillGroupDef> = {
        let map = SkillTreeMap::load_expect_cloned(
            "common.skill_trees.skills_skill-groups_manifest",
        ).0;
        map.iter().map(|(sgk, skills)|
            (*sgk, SkillGroupDef { skills: skills.clone(),
                total_skill_point_cost: skills
                    .iter()
                    .map(|skill| {
                        let max_level = skill.max_level();
                        (1..=max_level)
                            .into_iter()
                            .map(|level| skill.skill_cost(level))
                            .sum::<u16>()
                    })
                    .sum()
            })
        )
        .collect()
    };
    // Creates a hashmap for the reverse lookup of skill groups from a skill
    pub static ref SKILL_GROUP_LOOKUP: HashMap<Skill, SkillGroupKind> = {
        let map = SkillTreeMap::load_expect_cloned(
            "common.skill_trees.skills_skill-groups_manifest",
        ).0;
        map.iter().flat_map(|(sgk, skills)| skills.into_iter().map(move |s| (*s, *sgk))).collect()
    };
    // Loads the maximum level that a skill can obtain
    pub static ref SKILL_MAX_LEVEL: HashMap<Skill, u16> = {
        SkillLevelMap::load_expect_cloned(
            "common.skill_trees.skill_max_levels",
        ).0
    };
    // Loads the prerequisite skills for a particular skill
    pub static ref SKILL_PREREQUISITES: HashMap<Skill, HashMap<Skill, u16>> = {
        SkillPrerequisitesMap::load_expect_cloned(
            "common.skill_trees.skill_prerequisites",
        ).0
    };
    pub static ref SKILL_GROUP_HASHES: HashMap<SkillGroupKind, Vec<u8>> = {
        let map = SkillTreeMap::load_expect_cloned(
            "common.skill_trees.skills_skill-groups_manifest",
        ).0;
        let mut hashes = HashMap::new();
        for (skill_group_kind, skills) in map.iter() {
            let mut hasher = Sha256::new();
            let json_input: Vec<_> = skills.iter().map(|skill| (*skill, skill.max_level())).collect();
            let hash_input = serde_json::to_string(&json_input).unwrap_or_default();
            hasher.update(hash_input.as_bytes());
            let hash_result = hasher.finalize();
            hashes.insert(*skill_group_kind, hash_result.iter().copied().collect());
        }
        hashes
    };
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Serialize, Deserialize, Ord, PartialOrd)]
pub enum SkillGroupKind {
    General,
    Weapon(ToolKind),
}

impl SkillGroupKind {
    /// Gets the cost in experience of earning a skill point
    /// Changing this is forward compatible with persistence and will
    /// automatically force a respec for skill group kinds that are affected.
    pub fn skill_point_cost(self, level: u16) -> u32 {
        const EXP_INCREMENT: f32 = 10.0;
        const STARTING_EXP: f32 = 70.0;
        const EXP_CEILING: f32 = 1000.0;
        const SCALING_FACTOR: f32 = 0.125;
        (EXP_INCREMENT
            * (EXP_CEILING
                / EXP_INCREMENT
                / (1.0
                    + std::f32::consts::E.powf(-SCALING_FACTOR * level as f32)
                        * (EXP_CEILING / STARTING_EXP - 1.0)))
                .floor()) as u32
    }

    /// Gets the total amount of skill points that can be spent in a particular
    /// skill group
    pub fn total_skill_point_cost(self) -> u16 {
        if let Some(SkillGroupDef {
            total_skill_point_cost,
            ..
        }) = SKILL_GROUP_DEFS.get(&self)
        {
            *total_skill_point_cost
        } else {
            0
        }
    }
}

/// A group of skills that have been unlocked by a player. Each skill group has
/// independent exp and skill points which are used to unlock skills in that
/// skill group.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
pub struct SkillGroup {
    pub skill_group_kind: SkillGroupKind,
    // How much exp has been used for skill points
    pub spent_exp: u32,
    // How much exp has been earned in total
    pub earned_exp: u32,
    pub available_sp: u16,
    pub earned_sp: u16,
    // Used for persistence
    pub ordered_skills: Vec<Skill>,
}

impl SkillGroup {
    fn new(skill_group_kind: SkillGroupKind) -> SkillGroup {
        SkillGroup {
            skill_group_kind,
            spent_exp: 0,
            earned_exp: 0,
            available_sp: 0,
            earned_sp: 0,
            ordered_skills: Vec::new(),
        }
    }

    /// Returns the available experience that could be used to earn another
    /// skill point in a particular skill group.
    pub fn available_experience(&self) -> u32 { self.earned_exp - self.spent_exp }

    /// Adds a skill point while subtracting the necessary amount of experience
    pub fn earn_skill_point(&mut self) -> Result<(), SpRewardError> {
        let sp_cost = self.skill_group_kind.skill_point_cost(self.earned_sp);
        if self.available_experience() >= sp_cost {
            let new_spent_exp = self
                .spent_exp
                .checked_add(sp_cost)
                .ok_or(SpRewardError::Overflow)?;
            let new_earned_sp = self
                .earned_sp
                .checked_add(1)
                .ok_or(SpRewardError::Overflow)?;
            let new_available_sp = self
                .available_sp
                .checked_add(1)
                .ok_or(SpRewardError::Overflow)?;
            self.spent_exp = new_spent_exp;
            self.earned_sp = new_earned_sp;
            self.available_sp = new_available_sp;
            Ok(())
        } else {
            Err(SpRewardError::InsufficientExp)
        }
    }
}

/// Contains all of a player's skill groups and skills. Provides methods for
/// manipulating assigned skills and skill groups including unlocking skills,
/// refunding skills etc.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SkillSet {
    skill_groups: Vec<SkillGroup>,
    skills: HashMap<Skill, u16>,
    pub modify_health: bool,
    pub modify_energy: bool,
}

impl Component for SkillSet {
    type Storage = DerefFlaggedStorage<Self, IdvStorage<Self>>;
}

impl Default for SkillSet {
    /// Instantiate a new skill set with the default skill groups with no
    /// unlocked skills in them - used when adding a skill set to a new
    /// player
    fn default() -> Self {
        Self {
            skill_groups: vec![
                SkillGroup::new(SkillGroupKind::General),
                SkillGroup::new(SkillGroupKind::Weapon(ToolKind::Pick)),
            ],
            skills: SkillSet::initial_skills(),
            modify_health: false,
            modify_energy: false,
        }
    }
}

impl SkillSet {
    pub fn initial_skills() -> HashMap<Skill, u16> {
        let mut skills = HashMap::new();
        skills.insert(Skill::UnlockGroup(SkillGroupKind::General), 1);
        skills.insert(
            Skill::UnlockGroup(SkillGroupKind::Weapon(ToolKind::Pick)),
            1,
        );
        skills
    }

    pub fn load_from_database(
        skill_groups: Vec<SkillGroup>,
        mut all_skills: HashMap<SkillGroupKind, Vec<Skill>>,
    ) -> Self {
        let mut skillset = SkillSet {
            skill_groups,
            skills: SkillSet::initial_skills(),
            modify_health: true,
            modify_energy: true,
        };

        // Loops while checking the all_skills hashmap. For as long as it can find an
        // entry where the skill group kind is unlocked, insert the skills corresponding
        // to that skill group kind. When no more skill group kinds can be found, break
        // the loop.
        while let Some(skill_group_kind) = all_skills
            .keys()
            .find(|kind| skillset.has_skill(Skill::UnlockGroup(**kind)))
            .copied()
        {
            // Remove valid skill group kind from the hash map so that loop eventually
            // terminates.
            if let Some(skills) = all_skills.remove(&skill_group_kind) {
                let backup_skillset = skillset.clone();
                // Iterate over all skills and make sure that unlocking them is successful. If
                // any fail, fall back to skillset before unlocking any to allow a full respec
                if !skills
                    .iter()
                    .all(|skill| skillset.unlock_skill(*skill).is_ok())
                {
                    skillset = backup_skillset;
                }
            }
        }

        skillset
    }

    /// Checks if a particular skill group is accessible for an entity
    pub fn skill_group_accessible(&self, skill_group_kind: SkillGroupKind) -> bool {
        self.skill_groups
            .iter()
            .any(|x| x.skill_group_kind == skill_group_kind)
            && self.has_skill(Skill::UnlockGroup(skill_group_kind))
    }

    ///  Unlocks a skill group for a player. It starts with 0 exp and 0 skill
    ///  points.
    pub fn unlock_skill_group(&mut self, skill_group_kind: SkillGroupKind) {
        if !self
            .skill_groups
            .iter()
            .any(|x| x.skill_group_kind == skill_group_kind)
        {
            self.skill_groups.push(SkillGroup::new(skill_group_kind));
        } else {
            warn!("Tried to unlock already known skill group");
        }
    }

    /// Returns an iterator over skill groups
    pub fn skill_groups(&self) -> &Vec<SkillGroup> { &self.skill_groups }

    /// Returns a reference to a particular skill group in a skillset
    fn skill_group(&self, skill_group: SkillGroupKind) -> Option<&SkillGroup> {
        self.skill_groups
            .iter()
            .find(|s_g| s_g.skill_group_kind == skill_group)
    }

    /// Returns a mutable reference to a particular skill group in a skillset
    /// Requires that skillset contains skill that unlocks the skill group
    fn skill_group_mut(&mut self, skill_group: SkillGroupKind) -> Option<&mut SkillGroup> {
        // In order to mutate skill group, we check that the prerequisite skill has been
        // acquired, as this is one of the requirements for us to consider the skill
        // group accessible.
        let skill_group_accessible = self.skill_group_accessible(skill_group);
        self.skill_groups
            .iter_mut()
            .find(|s_g| s_g.skill_group_kind == skill_group && skill_group_accessible)
    }

    /// Adds experience to the skill group within an entity's skill set
    pub fn add_experience(&mut self, skill_group_kind: SkillGroupKind, amount: u32) {
        if let Some(mut skill_group) = self.skill_group_mut(skill_group_kind) {
            skill_group.earned_exp = skill_group.earned_exp.saturating_add(amount);
        } else {
            warn!("Tried to add experience to a skill group that player does not have");
        }
    }

    /// Gets the available experience for a particular skill group
    pub fn available_experience(&self, skill_group: SkillGroupKind) -> u32 {
        self.skill_group(skill_group)
            .map_or(0, |s_g| s_g.available_experience())
    }

    /// Checks how much experience is needed for the next skill point in a tree
    pub fn skill_point_cost(&self, skill_group: SkillGroupKind) -> u32 {
        if let Some(level) = self.skill_group(skill_group).map(|sg| sg.earned_sp) {
            skill_group.skill_point_cost(level)
        } else {
            skill_group.skill_point_cost(0)
        }
    }

    /// Adds skill points to a skill group as long as the player has that skill
    /// group type.
    pub fn add_skill_points(
        &mut self,
        skill_group_kind: SkillGroupKind,
        number_of_skill_points: u16,
    ) {
        for _ in 0..number_of_skill_points {
            let exp_needed = self.skill_point_cost(skill_group_kind);
            self.add_experience(skill_group_kind, exp_needed);
            if self.earn_skill_point(skill_group_kind).is_err() {
                warn!("Failed to add skill point");
                break;
            }
        }
    }

    /// Adds a skill point while subtracting the necessary amount of experience
    pub fn earn_skill_point(
        &mut self,
        skill_group_kind: SkillGroupKind,
    ) -> Result<(), SpRewardError> {
        if let Some(skill_group) = self.skill_group_mut(skill_group_kind) {
            skill_group.earn_skill_point()
        } else {
            Err(SpRewardError::UnavailableSkillGroup)
        }
    }

    /// Gets the available points for a particular skill group
    pub fn available_sp(&self, skill_group: SkillGroupKind) -> u16 {
        self.skill_group(skill_group)
            .map_or(0, |s_g| s_g.available_sp)
    }

    /// Gets the total earned points for a particular skill group
    pub fn earned_sp(&self, skill_group: SkillGroupKind) -> u16 {
        self.skill_group(skill_group).map_or(0, |s_g| s_g.earned_sp)
    }

    /// Checks that the skill set contains all prerequisite skills of the
    /// required level for a particular skill
    pub fn prerequisites_met(&self, skill: Skill) -> bool {
        skill
            .prerequisite_skills()
            .all(|(s, l)| self.skill_level(s).map_or(false, |l_b| l_b >= l))
    }

    /// Gets skill point cost to purchase skill of next level
    pub fn skill_cost(&self, skill: Skill) -> u16 {
        let next_level = self.next_skill_level(skill);
        skill.skill_cost(next_level)
    }

    /// Checks if player has sufficient skill points to purchase a skill
    pub fn sufficient_skill_points(&self, skill: Skill) -> bool {
        if let Some(skill_group_kind) = skill.skill_group_kind() {
            if let Some(skill_group) = self.skill_group(skill_group_kind) {
                let needed_sp = self.skill_cost(skill);
                skill_group.available_sp >= needed_sp
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Checks the next level of a skill
    fn next_skill_level(&self, skill: Skill) -> u16 {
        if let Ok(level) = self.skill_level(skill) {
            // If already has skill, then level + 1
            level + 1
        } else {
            // Otherwise the next level is the first level
            1
        }
    }

    /// Unlocks a skill for a player, assuming they have the relevant skill
    /// group unlocked and available SP in that skill group.
    pub fn unlock_skill(&mut self, skill: Skill) -> Result<(), SkillUnlockError> {
        if let Some(skill_group_kind) = skill.skill_group_kind() {
            let next_level = self.next_skill_level(skill);
            let prerequisites_met = self.prerequisites_met(skill);
            // Check that skill is not yet at max level
            if !matches!(self.skills.get(&skill), Some(level) if *level == skill.max_level()) {
                if let Some(mut skill_group) = self.skill_group_mut(skill_group_kind) {
                    if prerequisites_met {
                        if let Some(new_available_sp) = skill_group
                            .available_sp
                            .checked_sub(skill.skill_cost(next_level))
                        {
                            skill_group.available_sp = new_available_sp;
                            skill_group.ordered_skills.push(skill);
                            match skill {
                                Skill::UnlockGroup(group) => {
                                    self.unlock_skill_group(group);
                                },
                                Skill::General(GeneralSkill::HealthIncrease) => {
                                    self.modify_health = true;
                                },
                                Skill::General(GeneralSkill::EnergyIncrease) => {
                                    self.modify_energy = true;
                                },
                                _ => {},
                            }
                            self.skills.insert(skill, next_level);
                            Ok(())
                        } else {
                            trace!("Tried to unlock skill for skill group with insufficient SP");
                            Err(SkillUnlockError::InsufficientSP)
                        }
                    } else {
                        trace!("Tried to unlock skill without meeting prerequisite skills");
                        Err(SkillUnlockError::MissingPrerequisites)
                    }
                } else {
                    trace!("Tried to unlock skill for a skill group that player does not have");
                    Err(SkillUnlockError::UnavailableSkillGroup)
                }
            } else {
                trace!("Tried to unlock skill the player already has");
                Err(SkillUnlockError::SkillAlreadyUnlocked)
            }
        } else {
            warn!(
                ?skill,
                "Tried to unlock skill that does not exist in any skill group!"
            );
            Err(SkillUnlockError::NoParentSkillTree)
        }
    }

    /// Checks if the player has available SP to spend
    pub fn has_available_sp(&self) -> bool {
        self.skill_groups.iter().any(|sg| {
            sg.available_sp > 0
            // Subtraction in bounds because of the invariant that available_sp <= earned_sp
                && (sg.earned_sp - sg.available_sp) < sg.skill_group_kind.total_skill_point_cost()
        })
    }

    /// Checks if the skill is at max level in a skill set
    pub fn is_at_max_level(&self, skill: Skill) -> bool {
        if let Ok(level) = self.skill_level(skill) {
            level == skill.max_level()
        } else {
            false
        }
    }

    /// Checks if skill set contains a skill
    pub fn has_skill(&self, skill: Skill) -> bool { self.skills.contains_key(&skill) }

    /// Returns the level of the skill
    pub fn skill_level(&self, skill: Skill) -> Result<u16, SkillError> {
        if let Some(level) = self.skills.get(&skill).copied() {
            Ok(level)
        } else {
            Err(SkillError::MissingSkill)
        }
    }

    /// Returns the level of the skill or passed value as default
    pub fn skill_level_or(&self, skill: Skill, default: u16) -> u16 {
        if let Ok(level) = self.skill_level(skill) {
            level
        } else {
            default
        }
    }
}

#[derive(Debug)]
pub enum SkillError {
    MissingSkill,
}

#[derive(Debug)]
pub enum SkillUnlockError {
    InsufficientSP,
    MissingPrerequisites,
    UnavailableSkillGroup,
    SkillAlreadyUnlocked,
    NoParentSkillTree,
}

pub enum SpRewardError {
    InsufficientExp,
    UnavailableSkillGroup,
    Overflow,
}
