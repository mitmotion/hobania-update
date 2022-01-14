// Note: If you changes here "break" old character saves you can change the
// version in voxygen\src\meta.rs in order to reset save files to being empty

use crate::{
    assets::{self, Asset, AssetExt},
    comp::{item::ItemKind, skills::Skill, CharacterAbility, Item},
};
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::{
    ops::{AddAssign, DivAssign, MulAssign, Sub},
    time::Duration,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum ToolKind {
    // weapons
    Sword,
    Axe,
    Hammer,
    Bow,
    Staff,
    Sceptre,
    // future weapons
    Dagger,
    Shield,
    Spear,
    Blowgun,
    // tools
    Debug,
    Farming,
    Pick,
    // npcs
    /// Intended for invisible weapons (e.g. a creature using its claws or
    /// biting)
    Natural,
    /// This is an placeholder item, it is used by non-humanoid npcs to attack
    Empty,
}

impl ToolKind {
    pub fn identifier_name(&self) -> &'static str {
        match self {
            ToolKind::Sword => "sword",
            ToolKind::Axe => "axe",
            ToolKind::Hammer => "hammer",
            ToolKind::Bow => "bow",
            ToolKind::Dagger => "dagger",
            ToolKind::Staff => "staff",
            ToolKind::Spear => "spear",
            ToolKind::Blowgun => "blowgun",
            ToolKind::Sceptre => "sceptre",
            ToolKind::Shield => "shield",
            ToolKind::Natural => "natural",
            ToolKind::Debug => "debug",
            ToolKind::Farming => "farming",
            ToolKind::Pick => "pickaxe",
            ToolKind::Empty => "empty",
        }
    }

    pub fn gains_combat_xp(&self) -> bool {
        matches!(
            self,
            ToolKind::Sword
                | ToolKind::Axe
                | ToolKind::Hammer
                | ToolKind::Bow
                | ToolKind::Dagger
                | ToolKind::Staff
                | ToolKind::Spear
                | ToolKind::Blowgun
                | ToolKind::Sceptre
                | ToolKind::Shield
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Hands {
    One,
    Two,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub equip_time_secs: f32,
    pub power: f32,
    pub effect_power: f32,
    pub speed: f32,
    pub crit_chance: f32,
    pub range: f32,
    pub energy_efficiency: f32,
    pub buff_strength: f32,
}

impl Stats {
    pub fn zeroed() -> Stats {
        Stats {
            equip_time_secs: 0.0,
            power: 0.0,
            effect_power: 0.0,
            speed: 0.0,
            crit_chance: 0.0,
            range: 0.0,
            energy_efficiency: 0.0,
            buff_strength: 0.0,
        }
    }

    #[must_use]
    pub fn clamp_speed(mut self) -> Self {
        // if a tool has 0.0 speed, that panics due to being infinite duration, so
        // enforce speed >= 0.1 on the final product (but not the intermediates)
        self.speed = self.speed.max(0.1);
        self
    }
}

impl Asset for Stats {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

impl AddAssign<Stats> for Stats {
    fn add_assign(&mut self, other: Stats) {
        self.equip_time_secs += other.equip_time_secs;
        self.power += other.power;
        self.effect_power += other.effect_power;
        self.speed += other.speed;
        self.crit_chance += other.crit_chance;
        self.range += other.range;
        self.buff_strength += other.buff_strength;
    }
}
impl MulAssign<Stats> for Stats {
    fn mul_assign(&mut self, other: Stats) {
        self.equip_time_secs *= other.equip_time_secs;
        self.power *= other.power;
        self.effect_power *= other.effect_power;
        self.speed *= other.speed;
        self.crit_chance *= other.crit_chance;
        self.range *= other.range;
        self.buff_strength *= other.buff_strength;
    }
}
impl DivAssign<usize> for Stats {
    fn div_assign(&mut self, scalar: usize) {
        self.equip_time_secs /= scalar as f32;
        // since averaging occurs when the stats are used multiplicatively, don't permit
        // multiplying an equip_time_secs by 0, since that would be overpowered
        self.equip_time_secs = self.equip_time_secs.max(0.001);
        self.power /= scalar as f32;
        self.effect_power /= scalar as f32;
        self.speed /= scalar as f32;
        self.crit_chance /= scalar as f32;
        self.range /= scalar as f32;
        self.buff_strength /= scalar as f32;
    }
}

impl Sub<Stats> for Stats {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            equip_time_secs: self.equip_time_secs - other.equip_time_secs,
            power: self.power - other.power,
            effect_power: self.effect_power - other.effect_power,
            speed: self.speed - other.speed,
            crit_chance: self.crit_chance - other.crit_chance,
            range: self.range - other.range,
            energy_efficiency: self.range - other.energy_efficiency,
            buff_strength: self.buff_strength - other.buff_strength,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaterialStatManifest(pub HashMap<String, Stats>);

// This could be a Compound that also loads the keys, but the RecipeBook
// Compound impl already does that, so checking for existence here is redundant.
impl Asset for MaterialStatManifest {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

impl Default for MaterialStatManifest {
    fn default() -> MaterialStatManifest {
        // TODO: Don't do this, loading a default should have no ability to panic
        MaterialStatManifest::load_expect_cloned("common.material_stats_manifest")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum StatKind {
    Direct(Stats),
    Modular,
}

impl StatKind {
    pub fn resolve_stats(&self, msm: &MaterialStatManifest, components: &[Item]) -> Stats {
        let mut stats = match self {
            StatKind::Direct(stats) => *stats,
            StatKind::Modular => Stats::zeroed(),
        };
        let mut multipliers: Vec<Stats> = Vec::new();
        for item in components.iter() {
            match item.kind() {
                ItemKind::ModularComponent(mc) => {
                    let inner_stats =
                        StatKind::Direct(mc.stats).resolve_stats(msm, item.components());
                    stats += inner_stats;
                },
                ItemKind::Ingredient { .. } => {
                    if let Some(mult_stats) = msm.0.get(item.item_definition_id()) {
                        multipliers.push(*mult_stats);
                    }
                },
                // TODO: add stats from enhancement slots
                _ => (),
            }
        }
        // Take the average of the material multipliers, to allow alloyed blades
        if !multipliers.is_empty() {
            let mut average_mult = Stats::zeroed();
            for stat in multipliers.iter() {
                average_mult += *stat;
            }
            average_mult /= multipliers.len();
            stats *= average_mult;
        }
        stats
    }
}

impl From<(&MaterialStatManifest, &[Item], &Tool)> for Stats {
    fn from((msm, components, tool): (&MaterialStatManifest, &[Item], &Tool)) -> Self {
        let raw_stats = tool.stats.resolve_stats(msm, components).clamp_speed();
        let (power, speed, poise) = match tool.hands {
            Hands::One => (0.67, 1.33, 0.67),
            Hands::Two => (1.5, 0.75, 1.5),
        };
        Self {
            equip_time_secs: raw_stats.equip_time_secs,
            power: raw_stats.power * power,
            effect_power: raw_stats.effect_power * poise,
            speed: raw_stats.speed * speed,
            crit_chance: raw_stats.crit_chance,
            range: raw_stats.range,
            energy_efficiency: raw_stats.energy_efficiency,
            buff_strength: raw_stats.buff_strength,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Tool {
    pub kind: ToolKind,
    pub hands: Hands,
    pub stats: StatKind,
    // TODO: item specific abilities
}

impl Tool {
    // DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING
    // Added for CSV import of stats
    pub fn new(kind: ToolKind, hands: Hands, stats: Stats) -> Self {
        Self {
            kind,
            hands,
            stats: StatKind::Direct(stats),
        }
    }

    pub fn empty() -> Self {
        Self {
            kind: ToolKind::Empty,
            hands: Hands::One,
            stats: StatKind::Direct(Stats {
                equip_time_secs: 0.0,
                power: 1.00,
                effect_power: 1.00,
                speed: 1.00,
                crit_chance: 0.1,
                range: 1.0,
                energy_efficiency: 1.0,
                buff_strength: 1.0,
            }),
        }
    }

    // Keep power between 0.5 and 2.00
    pub fn base_power(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).power
    }

    pub fn base_effect_power(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).effect_power
    }

    pub fn base_speed(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats
            .resolve_stats(msm, components)
            .clamp_speed()
            .speed
    }

    pub fn base_crit_chance(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).crit_chance
    }

    pub fn base_range(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).range
    }

    pub fn base_energy_efficiency(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).energy_efficiency
    }

    pub fn base_buff_strength(&self, msm: &MaterialStatManifest, components: &[Item]) -> f32 {
        self.stats.resolve_stats(msm, components).buff_strength
    }

    pub fn equip_time(&self, msm: &MaterialStatManifest, components: &[Item]) -> Duration {
        Duration::from_secs_f32(self.stats.resolve_stats(msm, components).equip_time_secs)
    }

    pub fn can_block(&self) -> bool {
        matches!(
            self.kind,
            ToolKind::Sword
                | ToolKind::Axe
                | ToolKind::Hammer
                | ToolKind::Shield
                | ToolKind::Dagger
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbilitySet<T> {
    pub primary: T,
    pub secondary: T,
    pub abilities: Vec<(Option<Skill>, T)>,
}

impl AbilitySet<AbilityItem> {
    #[must_use]
    pub fn modified_by_tool(
        self,
        tool: &Tool,
        msm: &MaterialStatManifest,
        components: &[Item],
    ) -> Self {
        let stats = Stats::from((msm, components, tool));
        self.map(|a| AbilityItem {
            id: a.id,
            ability: a.ability.adjusted_by_stats(stats),
        })
    }
}

impl<T> AbilitySet<T> {
    pub fn map<U, F: FnMut(T) -> U>(self, mut f: F) -> AbilitySet<U> {
        AbilitySet {
            primary: f(self.primary),
            secondary: f(self.secondary),
            abilities: self.abilities.into_iter().map(|(s, x)| (s, f(x))).collect(),
        }
    }

    pub fn map_ref<U, F: FnMut(&T) -> U>(&self, mut f: F) -> AbilitySet<U> {
        AbilitySet {
            primary: f(&self.primary),
            secondary: f(&self.secondary),
            abilities: self.abilities.iter().map(|(s, x)| (*s, f(x))).collect(),
        }
    }
}

impl Default for AbilitySet<AbilityItem> {
    fn default() -> Self {
        AbilitySet {
            primary: AbilityItem {
                id: String::new(),
                ability: CharacterAbility::default(),
            },
            secondary: AbilityItem {
                id: String::new(),
                ability: CharacterAbility::default(),
            },
            abilities: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AbilitySpec {
    Tool(ToolKind),
    Custom(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbilityItem {
    pub id: String,
    pub ability: CharacterAbility,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AbilityMap<T = AbilityItem>(HashMap<AbilitySpec, AbilitySet<T>>);

impl Default for AbilityMap {
    fn default() -> Self {
        // TODO: Revert to old default
        if let Ok(map) = Self::load_cloned("common.abilities.ability_set_manifest") {
            map
        } else {
            let mut map = HashMap::new();
            map.insert(AbilitySpec::Tool(ToolKind::Empty), AbilitySet::default());
            AbilityMap(map)
        }
    }
}

impl<T> AbilityMap<T> {
    pub fn get_ability_set(&self, key: &AbilitySpec) -> Option<&AbilitySet<T>> { self.0.get(key) }
}

impl Asset for AbilityMap<String> {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

impl assets::Compound for AbilityMap {
    fn load<S: assets::source::Source + ?Sized>(
        cache: &assets::AssetCache<S>,
        specifier: &str,
    ) -> Result<Self, assets::BoxedError> {
        let manifest = cache.load::<AbilityMap<String>>(specifier)?.read();

        Ok(AbilityMap(
            manifest
                .0
                .iter()
                .map(|(kind, set)| {
                    (
                        kind.clone(),
                        // expect cannot fail because CharacterAbility always
                        // provides a default value in case of failure
                        set.map_ref(|s| AbilityItem {
                            id: s.clone(),
                            ability: cache.load_expect(s).cloned(),
                        }),
                    )
                })
                .collect(),
        ))
    }
}
