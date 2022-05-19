use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, ops::Sub};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArmorKind {
    Shoulder,
    Chest,
    Belt,
    Hand,
    Pants,
    Foot,
    Back,
    Ring,
    Neck,
    Head,
    Tabard,
    Bag,
}

impl Armor {
    /// Determines whether two pieces of armour are superficially equivalent to
    /// one another (i.e: one may be substituted for the other in crafting
    /// recipes or item possession checks).
    pub fn superficially_eq(&self, other: &Self) -> bool {
        std::mem::discriminant(&self.kind) == std::mem::discriminant(&other.kind)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    /// Protection is non-linearly transformed (following summation) to a damage
    /// reduction using (prot / (60 + prot))
    protection: Option<Protection>,
    /// Poise protection is non-linearly transformed (following summation) to a
    /// poise damage reduction using (prot / (60 + prot))
    poise_resilience: Option<Protection>,
    /// Energy max is summed, and then applied directly to the max energy stat
    energy_max: Option<f32>,
    /// Energy recovery is summed, and then added to 1.0. When attacks reward
    /// energy, it is then multiplied by this value before the energy is
    /// rewarded.
    energy_reward: Option<f32>,
    /// Crit power is summed, and then added to the default crit multiplier of
    /// 1.25. Damage is multiplied by this value when an attack crits.
    crit_power: Option<f32>,
    /// Stealth is summed along with the base stealth bonus (2.0), and then
    /// the agent's perception distance is divided by this value
    stealth: Option<f32>,
}

impl Stats {
    // DO NOT USE UNLESS YOU KNOW WHAT YOU ARE DOING
    // Added for csv import of stats
    pub fn new(
        protection: Option<Protection>,
        poise_resilience: Option<Protection>,
        energy_max: Option<f32>,
        energy_reward: Option<f32>,
        crit_power: Option<f32>,
        stealth: Option<f32>,
    ) -> Self {
        Self {
            protection,
            poise_resilience,
            energy_max,
            energy_reward,
            crit_power,
            stealth,
        }
    }

    pub fn protection(&self) -> Option<Protection> { self.protection }

    pub fn poise_resilience(&self) -> Option<Protection> { self.poise_resilience }

    pub fn energy_max(&self) -> Option<f32> { self.energy_max }

    pub fn energy_reward(&self) -> Option<f32> { self.energy_reward }

    pub fn crit_power(&self) -> Option<f32> { self.crit_power }

    pub fn stealth(&self) -> Option<f32> { self.stealth }
}

impl Sub<Stats> for Stats {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            protection: self.protection.zip(other.protection).map(|(a, b)| a - b),
            poise_resilience: self
                .poise_resilience
                .zip(other.poise_resilience)
                .map(|(a, b)| a - b),
            energy_max: self.energy_max.zip(other.energy_max).map(|(a, b)| a - b),
            energy_reward: self
                .energy_reward
                .zip(other.energy_reward)
                .map(|(a, b)| a - b),
            crit_power: self.crit_power.zip(other.crit_power).map(|(a, b)| a - b),
            stealth: self.stealth.zip(other.stealth).map(|(a, b)| a - b),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Protection {
    Invincible,
    Normal(f32),
}

impl Default for Protection {
    fn default() -> Self { Self::Normal(0.0) }
}

impl Sub for Protection {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        let diff = match (self, other) {
            (Protection::Invincible, Protection::Normal(_)) => f32::INFINITY,
            (Protection::Invincible, Protection::Invincible) => 0_f32,
            (Protection::Normal(_), Protection::Invincible) => -f32::INFINITY,
            (Protection::Normal(a), Protection::Normal(b)) => a - b,
        };
        Protection::Normal(diff)
    }
}

impl PartialOrd for Protection {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (*self, *other) {
            (Protection::Invincible, Protection::Invincible) => Some(Ordering::Equal),
            (Protection::Invincible, _) => Some(Ordering::Greater),
            (_, Protection::Invincible) => Some(Ordering::Less),
            (Protection::Normal(a), Protection::Normal(b)) => f32::partial_cmp(&a, &b),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Armor {
    pub kind: ArmorKind,
    pub stats: Stats,
}

impl Armor {
    pub fn new(kind: ArmorKind, stats: Stats) -> Self { Self { kind, stats } }

    pub fn protection(&self) -> Option<Protection> { self.stats.protection }

    pub fn poise_resilience(&self) -> Option<Protection> { self.stats.poise_resilience }

    pub fn energy_max(&self) -> Option<f32> { self.stats.energy_max }

    pub fn energy_reward(&self) -> Option<f32> { self.stats.energy_reward }

    pub fn crit_power(&self) -> Option<f32> { self.stats.crit_power }

    pub fn stealth(&self) -> Option<f32> { self.stats.stealth }

    #[cfg(test)]
    pub fn test_armor(
        kind: ArmorKind,
        protection: Protection,
        poise_resilience: Protection,
    ) -> Armor {
        Armor {
            kind,
            stats: Stats {
                protection: Some(protection),
                poise_resilience: Some(poise_resilience),
                energy_max: None,
                energy_reward: None,
                crit_power: None,
                stealth: None,
            },
        }
    }
}
