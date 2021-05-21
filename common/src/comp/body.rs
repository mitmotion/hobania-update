pub mod biped_large;
pub mod biped_small;
pub mod bird_large;
pub mod bird_medium;
pub mod dragon;
pub mod fish_medium;
pub mod fish_small;
pub mod golem;
pub mod humanoid;
pub mod object;
pub mod quadruped_low;
pub mod quadruped_medium;
pub mod quadruped_small;
pub mod ship;
pub mod theropod;

use crate::{
    assets::{self, Asset},
    make_case_elim,
    npc::NpcKind,
};
use serde::{Deserialize, Serialize};
use specs::{Component, DerefFlaggedStorage};
use specs_idvs::IdvStorage;
use vek::*;

use super::{BuffKind, Density, Mass};

make_case_elim!(
    body,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[repr(u32)]
    pub enum Body {
        BipedLarge(body: biped_large::Body)= 8,
        BipedSmall(body: biped_small::Body)= 9,
        BirdLarge(body: bird_large::Body) = 6,
        BirdMedium(body: bird_medium::Body) = 3,
        Dragon(body: dragon::Body) = 5,
        FishMedium(body: fish_medium::Body) = 4,
        FishSmall(body: fish_small::Body) = 7,
        Golem(body: golem::Body) = 11,
        Humanoid(body: humanoid::Body) = 0,
        QuadrupedLow(body: quadruped_low::Body) = 13,
        QuadrupedMedium(body: quadruped_medium::Body) = 2,
        QuadrupedSmall(body: quadruped_small::Body) = 1,
        Theropod(body: theropod::Body) = 12,
        Object(body: object::Body) = 10,
        Ship(body: ship::Body) = 14,
    }
);

/// Data representing data generic to the body together with per-species data.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BodyData<BodyMeta, SpeciesData> {
    /// Shared metadata for this whole body type.
    pub body: BodyMeta,
    /// All the metadata for species with this body type.
    pub species: SpeciesData,
}

/// Metadata intended to be stored per-body, together with data intended to be
/// stored for each species for each body.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
///
///
/// ##############################################################################################
/// ##############################################################################################
///
/// ##############################################################################################
/// ##############################################################################################
/// TALK TO SHARP ABOUT IMPLEMENTING SERIALIZE!!!!!!
/// ##############################################################################################
/// ##############################################################################################
///
/// ##############################################################################################
/// ##############################################################################################
///
/// NOTE: If you are adding new body kind and it should be spawned via /spawn
/// please add it to `[ENTITIES](crate::cmd::ENTITIES)`
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllBodies<BodyMeta, SpeciesMeta> {
    pub humanoid: BodyData<BodyMeta, humanoid::AllSpecies<SpeciesMeta>>,
    pub quadruped_small: BodyData<BodyMeta, quadruped_small::AllSpecies<SpeciesMeta>>,
    pub quadruped_medium: BodyData<BodyMeta, quadruped_medium::AllSpecies<SpeciesMeta>>,
    pub bird_medium: BodyData<BodyMeta, bird_medium::AllSpecies<SpeciesMeta>>,
    pub fish_medium: BodyData<BodyMeta, fish_medium::AllSpecies<SpeciesMeta>>,
    pub dragon: BodyData<BodyMeta, dragon::AllSpecies<SpeciesMeta>>,
    pub bird_large: BodyData<BodyMeta, bird_large::AllSpecies<SpeciesMeta>>,
    pub fish_small: BodyData<BodyMeta, fish_small::AllSpecies<SpeciesMeta>>,
    pub biped_large: BodyData<BodyMeta, biped_large::AllSpecies<SpeciesMeta>>,
    pub biped_small: BodyData<BodyMeta, biped_small::AllSpecies<SpeciesMeta>>,
    pub object: BodyData<BodyMeta, ()>,
    pub golem: BodyData<BodyMeta, golem::AllSpecies<SpeciesMeta>>,
    pub theropod: BodyData<BodyMeta, theropod::AllSpecies<SpeciesMeta>>,
    pub quadruped_low: BodyData<BodyMeta, quadruped_low::AllSpecies<SpeciesMeta>>,
    pub ship: BodyData<BodyMeta, ()>,
}

/// Can only retrieve body metadata by direct index.
impl<BodyMeta, SpeciesMeta> core::ops::Index<NpcKind> for AllBodies<BodyMeta, SpeciesMeta> {
    type Output = BodyMeta;

    #[inline]
    fn index(&self, index: NpcKind) -> &Self::Output {
        match index {
            NpcKind::Humanoid => &self.humanoid.body,
            NpcKind::Pig => &self.quadruped_small.body,
            NpcKind::Wolf => &self.quadruped_medium.body,
            NpcKind::Duck => &self.bird_medium.body,
            NpcKind::Phoenix => &self.bird_large.body,
            NpcKind::Marlin => &self.fish_medium.body,
            NpcKind::Clownfish => &self.fish_small.body,
            NpcKind::Ogre => &self.biped_large.body,
            NpcKind::Gnome => &self.biped_small.body,
            NpcKind::StoneGolem => &self.golem.body,
            NpcKind::Archaeos => &self.theropod.body,
            NpcKind::Reddragon => &self.dragon.body,
            NpcKind::Crocodile => &self.quadruped_low.body,
        }
    }
}

/// Can only retrieve body metadata by direct index.
impl<'a, BodyMeta, SpeciesMeta> core::ops::Index<&'a Body> for AllBodies<BodyMeta, SpeciesMeta> {
    type Output = BodyMeta;

    #[inline]
    fn index(&self, index: &Body) -> &Self::Output {
        match index {
            Body::Humanoid(_) => &self.humanoid.body,
            Body::QuadrupedSmall(_) => &self.quadruped_small.body,
            Body::QuadrupedMedium(_) => &self.quadruped_medium.body,
            Body::BirdMedium(_) => &self.bird_medium.body,
            Body::BirdLarge(_) => &self.bird_large.body,
            Body::FishMedium(_) => &self.fish_medium.body,
            Body::Dragon(_) => &self.dragon.body,
            Body::FishSmall(_) => &self.fish_small.body,
            Body::BipedLarge(_) => &self.biped_large.body,
            Body::BipedSmall(_) => &self.biped_small.body,
            Body::Object(_) => &self.object.body,
            Body::Golem(_) => &self.golem.body,
            Body::Theropod(_) => &self.theropod.body,
            Body::QuadrupedLow(_) => &self.quadruped_low.body,
            Body::Ship(_) => &self.ship.body,
        }
    }
}

impl<
    BodyMeta: Send + Sync + for<'de> serde::Deserialize<'de> + 'static,
    SpeciesMeta: Send + Sync + for<'de> serde::Deserialize<'de> + 'static,
> Asset for AllBodies<BodyMeta, SpeciesMeta>
{
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

pub type BodyAggro = f32;
pub type SpeciesAggro = Option<f32>;
/// Type holding aggro data for bodies and species.
pub type AllBodiesAggro = AllBodies<BodyAggro, SpeciesAggro>;

pub type BodyDensity = f32;
pub type SpeciesDensity = Option<f32>;
/// Type holding density data for bodies and species.
pub type AllBodiesDensity = AllBodies<BodyDensity, SpeciesDensity>;

pub type BodyMass = f32;
pub type SpeciesMass = Option<f32>;
/// Type holding mass data for bodies and species.
pub type AllBodiesMass = AllBodies<BodyMass, SpeciesMass>;

pub type BodyBaseEnergy = u32;
pub type SpeciesBaseEnergy = Option<u32>;
/// Type holding mass data for bodies and species.
pub type AllBodiesBaseEnergy = AllBodies<BodyBaseEnergy, SpeciesBaseEnergy>;

pub type BodyBaseHealth = u32;
pub type SpeciesBaseHealth = Option<u32>;
/// Type holding mass data for bodies and species.
pub type AllBodiesBaseHealth = AllBodies<BodyBaseHealth, SpeciesBaseHealth>;

pub type BodyBaseHealthIncrease = u32;
pub type SpeciesBaseHealthIncrease = Option<u32>;
/// Type holding mass data for bodies and species.
pub type AllBodiesBaseHealthIncrease = AllBodies<BodyBaseHealthIncrease, SpeciesBaseHealthIncrease>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BodyAttributes {
    pub aggro: Option<AllBodiesAggro>,
    pub density: Option<AllBodiesDensity>,
    pub mass: Option<AllBodiesMass>,
    pub base_energy: Option<AllBodiesBaseEnergy>,
    pub base_health: Option<AllBodiesBaseHealth>,
    pub base_health_increase: Option<AllBodiesBaseHealthIncrease>,
    // Things to add:
    // dimensions of some kind
    //   - radius
    //   - height
    // poise
    // base_accel
    // base_ori_rate
    // swim_thrust
    // fly_thrust
    // jump_impulse
    // 
    // Maybes:
    // can_climb
    // flying_height
    // eye_height
    // default_light_offset
    // can_strafe
    // mounting_offset
    // air_accel

}

impl assets::Compound for BodyAttributes {
    fn load<S: assets::source::Source>(
        cache: &assets::AssetCache<S>,
        _: &str,
    ) -> Result<Self, assets::Error> {
        let aggro = cache.load_owned::<AllBodiesAggro>("common.body.aggro")?;
        let density = cache.load_owned::<AllBodiesDensity>("common.body.density")?;
        let mass = cache.load_owned::<AllBodiesMass>("common.body.mass")?;
        let energy = cache.load_owned::<AllBodiesBaseEnergy>("common.body.base_energy")?;
        let health = cache.load_owned::<AllBodiesBaseHealth>("common.body.base_health")?;
        let health_increase =
            cache.load_owned::<AllBodiesBaseHealthIncrease>("common.body.base_health_increase")?;

        Ok(Self {
            aggro: Some(aggro),
            density: Some(density),
            mass: Some(mass),
            base_energy: Some(energy),
            base_health: Some(health),
            base_health_increase: Some(health_increase),
        })
    }
}

pub fn get_body_f32_attribute<
    'a,
    Species,
    SpeciesData: for<'b> core::ops::Index<&'b Species, Output = Option<f32>>,
>(
    body_data: &'a BodyData<f32, SpeciesData>,
    species: Species,
) -> f32 {
    body_data.species[&species].unwrap_or_else(|| body_data.body)
}

pub fn get_body_u32_attribute<
    'a,
    Species,
    SpeciesData: for<'b> core::ops::Index<&'b Species, Output = Option<u32>>,
>(
    body_data: &'a BodyData<u32, SpeciesData>,
    species: Species,
) -> u32 {
    body_data.species[&species].unwrap_or_else(|| body_data.body)
}

pub fn get_body_attribute<
    'a,
    T: Copy,
    Species,
    SpeciesData: for<'b> core::ops::Index<&'b Species, Output = Option<T>>,
>(
    body_data: &'a BodyData<T, SpeciesData>,
    species: Species,
) -> T {
    body_data.species[&species].unwrap_or_else(|| body_data.body)
}

impl Body {
    pub fn is_humanoid(&self) -> bool { matches!(self, Body::Humanoid(_)) }

    pub fn get_attribute_value<T: Default + Copy>(&self, body_attribute: &AllBodies<T, Option<T>>) -> T {
        match self {
            Body::BipedLarge(body) => {
                get_body_attribute(&body_attribute.biped_large, body.species)
            },
            Body::BipedSmall(body) => {
                get_body_attribute(&body_attribute.biped_small, body.species)
            },
            Body::BirdMedium(body) => {
                get_body_attribute(&body_attribute.bird_medium, body.species)
            },
            Body::BirdLarge(body) => {
                get_body_attribute(&body_attribute.bird_large, body.species)
            },
            Body::Dragon(body) => get_body_attribute(&body_attribute.dragon, body.species),
            Body::FishMedium(body) => {
                get_body_attribute(&body_attribute.fish_medium, body.species)
            },
            Body::FishSmall(body) => {
                get_body_attribute(&body_attribute.fish_small, body.species)
            },
            Body::Golem(body) => get_body_attribute(&body_attribute.golem, body.species),
            Body::Humanoid(body) => get_body_attribute(&body_attribute.humanoid, body.species),
            Body::QuadrupedLow(body) => {
                get_body_attribute(&body_attribute.quadruped_low, body.species)
            },
            Body::QuadrupedMedium(body) => {
                get_body_attribute(&body_attribute.quadruped_medium, body.species)
            },
            Body::QuadrupedSmall(body) => {
                get_body_attribute(&body_attribute.quadruped_small, body.species)
            },
            Body::Theropod(body) => get_body_attribute(&body_attribute.theropod, body.species),
            // TODO fix this for ships and objects to read from the ron files
            Body::Ship(ship) => T::default(),
            Body::Object(object) => T::default(),
        }
    }

    /// Average density of the body
    /// Units are based on kg/m³
    pub fn density(&self, body_densities: &AllBodiesDensity) -> Density {
        let density_value = 1000.0;//match self {
        //    Body::BipedLarge(body) => {
        //        get_body_f32_attribute(&body_densities.biped_large, body.species)
        //    },
        //    Body::BipedSmall(body) => {
        //        get_body_f32_attribute(&body_densities.biped_small, body.species)
        //    },
        //    Body::BirdMedium(body) => {
        //        get_body_f32_attribute(&body_densities.bird_medium, body.species)
        //    },
        //    Body::BirdLarge(body) => {
        //        get_body_f32_attribute(&body_densities.bird_large, body.species)
        //    },
        //    Body::Dragon(body) => get_body_f32_attribute(&body_densities.dragon, body.species),
        //    Body::FishMedium(body) => {
        //        get_body_f32_attribute(&body_densities.fish_medium, body.species)
        //    },
        //    Body::FishSmall(body) => {
        //        get_body_f32_attribute(&body_densities.fish_small, body.species)
        //    },
        //    Body::Golem(body) => get_body_f32_attribute(&body_densities.golem, body.species),
        //    Body::Humanoid(body) => get_body_f32_attribute(&body_densities.humanoid, body.species),
        //    Body::QuadrupedLow(body) => {
        //        get_body_f32_attribute(&body_densities.quadruped_low, body.species)
        //    },
        //    Body::QuadrupedMedium(body) => {
        //        get_body_f32_attribute(&body_densities.quadruped_medium, body.species)
        //    },
        //    Body::QuadrupedSmall(body) => {
        //        get_body_f32_attribute(&body_densities.quadruped_small, body.species)
        //    },
        //    Body::Theropod(body) => get_body_f32_attribute(&body_densities.theropod, body.species),
        //    Body::Ship(body) => get_body_f32_attribute(&body_densities.ship, body.species),
        //    Body::Object(object) => object.density().0,
        //};
        Density(density_value)
    }

    /// Units are kg
    pub fn mass(&self, body_masses: &AllBodiesMass) -> Mass {
        let m = 100.0;//match self {
        //    Body::BipedLarge(body) => {
        //        get_body_f32_attribute(&body_masses.biped_large, body.species)
        //    },
        //    Body::BipedSmall(body) => {
        //        get_body_f32_attribute(&body_masses.biped_small, body.species)
        //    },
        //    Body::BirdMedium(body) => {
        //        get_body_f32_attribute(&body_masses.bird_medium, body.species)
        //    },
        //    Body::BirdLarge(body) => get_body_f32_attribute(&body_masses.bird_large, body.species),
        //    Body::Dragon(body) => get_body_f32_attribute(&body_masses.dragon, body.species),
        //    Body::FishMedium(body) => {
        //        get_body_f32_attribute(&body_masses.fish_medium, body.species)
        //    },
        //    Body::FishSmall(body) => get_body_f32_attribute(&body_masses.fish_small, body.species),
        //    Body::Golem(body) => get_body_f32_attribute(&body_masses.golem, body.species),
        //    Body::Humanoid(body) => get_body_f32_attribute(&body_masses.humanoid, body.species),
        //    Body::QuadrupedLow(body) => {
        //        get_body_f32_attribute(&body_masses.quadruped_low, body.species)
        //    },
        //    Body::QuadrupedMedium(body) => {
        //        get_body_f32_attribute(&body_masses.quadruped_medium, body.species)
        //    },
        //    Body::QuadrupedSmall(body) => {
        //        get_body_f32_attribute(&body_masses.quadruped_small, body.species)
        //    },
        //    Body::Theropod(body) => get_body_f32_attribute(&body_masses.theropod, body.species),
        //    Body::Ship(body) => get_body_f32_attribute(&body_masses.ship, body.species),
        //    Body::Object(object) => object.density().0,
        //};
        Mass(m)
    }

    pub fn aggro(&self, body_aggros: &AllBodiesAggro) -> f32 {
        match self {
            Body::BipedLarge(body) => {
                get_body_f32_attribute(&body_aggros.biped_large, body.species)
            },
            Body::BipedSmall(body) => {
                get_body_f32_attribute(&body_aggros.biped_small, body.species)
            },
            Body::BirdMedium(body) => {
                get_body_f32_attribute(&body_aggros.bird_medium, body.species)
            },
            Body::BirdLarge(body) => get_body_f32_attribute(&body_aggros.bird_large, body.species),
            Body::FishSmall(body) => get_body_f32_attribute(&body_aggros.fish_small, body.species),
            Body::FishMedium(body) => {
                get_body_f32_attribute(&body_aggros.fish_medium, body.species)
            },
            Body::Humanoid(body) => get_body_f32_attribute(&body_aggros.humanoid, body.species),
            Body::QuadrupedMedium(body) => {
                get_body_f32_attribute(&body_aggros.quadruped_medium, body.species)
            },
            Body::QuadrupedSmall(body) => {
                get_body_f32_attribute(&body_aggros.quadruped_small, body.species)
            },
            Body::Theropod(body) => get_body_f32_attribute(&body_aggros.theropod, body.species),
            Body::Dragon(body) => get_body_f32_attribute(&body_aggros.dragon, body.species),
            Body::QuadrupedLow(body) => {
                get_body_f32_attribute(&body_aggros.quadruped_low, body.species)
            },
            Body::Golem(body) => get_body_f32_attribute(&body_aggros.golem, body.species),
            Body::Ship(_) | Body::Object(_) => 0.0,
        }
    }

    /// The width (shoulder to shoulder), length (nose to tail) and height
    /// respectively
    pub fn dimensions(&self) -> Vec3<f32> {
        match self {
            Body::BipedLarge(body) => match body.species {
                biped_large::Species::Cyclops => Vec3::new(4.6, 3.0, 6.5),
                biped_large::Species::Dullahan => Vec3::new(4.6, 3.0, 5.5),
                biped_large::Species::Mightysaurok => Vec3::new(4.0, 3.0, 3.4),
                biped_large::Species::Mindflayer => Vec3::new(4.4, 3.0, 8.0),
                biped_large::Species::Minotaur => Vec3::new(6.0, 3.0, 8.0),
                biped_large::Species::Occultsaurok => Vec3::new(4.0, 3.0, 3.4),
                biped_large::Species::Slysaurok => Vec3::new(4.0, 3.0, 3.4),
                biped_large::Species::Werewolf => Vec3::new(4.0, 3.0, 3.5),

                _ => Vec3::new(4.6, 3.0, 6.0),
            },
            Body::BipedSmall(_) => Vec3::new(1.0, 0.75, 1.4),
            Body::BirdMedium(_) => Vec3::new(2.0, 1.0, 1.1),
            Body::BirdLarge(_) => Vec3::new(2.0, 5.0, 2.4),
            Body::Dragon(_) => Vec3::new(16.0, 10.0, 16.0),
            Body::FishMedium(_) => Vec3::new(0.5, 2.0, 0.8),
            Body::FishSmall(_) => Vec3::new(0.3, 1.2, 0.6),
            Body::Golem(_) => Vec3::new(5.0, 5.0, 7.5),
            Body::Humanoid(humanoid) => {
                let height = match (humanoid.species, humanoid.body_type) {
                    (humanoid::Species::Orc, humanoid::BodyType::Male) => 2.3,
                    (humanoid::Species::Orc, humanoid::BodyType::Female) => 2.2,
                    (humanoid::Species::Human, humanoid::BodyType::Male) => 2.3,
                    (humanoid::Species::Human, humanoid::BodyType::Female) => 2.2,
                    (humanoid::Species::Elf, humanoid::BodyType::Male) => 2.3,
                    (humanoid::Species::Elf, humanoid::BodyType::Female) => 2.2,
                    (humanoid::Species::Dwarf, humanoid::BodyType::Male) => 1.9,
                    (humanoid::Species::Dwarf, humanoid::BodyType::Female) => 1.8,
                    (humanoid::Species::Undead, humanoid::BodyType::Male) => 2.2,
                    (humanoid::Species::Undead, humanoid::BodyType::Female) => 2.1,
                    (humanoid::Species::Danari, humanoid::BodyType::Male) => 1.5,
                    (humanoid::Species::Danari, humanoid::BodyType::Female) => 1.4,
                };
                Vec3::new(1.5, 0.5, height)
            },
            Body::Object(object) => object.dimensions(),
            Body::QuadrupedMedium(body) => match body.species {
                quadruped_medium::Species::Barghest => Vec3::new(2.0, 3.6, 2.5),
                quadruped_medium::Species::Bear => Vec3::new(2.0, 3.6, 2.0),
                quadruped_medium::Species::Catoblepas => Vec3::new(2.0, 4.0, 2.9),
                quadruped_medium::Species::Cattle => Vec3::new(2.0, 3.6, 2.0),
                quadruped_medium::Species::Deer => Vec3::new(2.0, 3.0, 2.0),
                quadruped_medium::Species::Dreadhorn => Vec3::new(2.0, 3.0, 2.5),
                quadruped_medium::Species::Grolgar => Vec3::new(2.0, 4.0, 2.0),
                quadruped_medium::Species::Highland => Vec3::new(2.0, 3.6, 2.0),
                quadruped_medium::Species::Horse => Vec3::new(2.0, 3.0, 2.0),
                quadruped_medium::Species::Lion => Vec3::new(2.0, 3.0, 2.0),
                quadruped_medium::Species::Moose => Vec3::new(2.0, 4.0, 2.5),
                quadruped_medium::Species::Saber => Vec3::new(2.0, 4.0, 2.0),
                quadruped_medium::Species::Tarasque => Vec3::new(2.0, 4.0, 2.6),
                quadruped_medium::Species::Yak => Vec3::new(2.0, 3.6, 2.0),
                _ => Vec3::new(2.0, 3.0, 2.0),
            },
            Body::QuadrupedSmall(body) => match body.species {
                quadruped_small::Species::Dodarock => Vec3::new(1.2, 1.2, 1.5),
                quadruped_small::Species::Holladon => Vec3::new(1.2, 1.2, 1.5),
                quadruped_small::Species::Truffler => Vec3::new(1.2, 1.2, 2.0),
                _ => Vec3::new(1.2, 1.2, 1.0),
            },
            Body::QuadrupedLow(body) => match body.species {
                quadruped_low::Species::Asp => Vec3::new(1.0, 2.5, 1.3),
                quadruped_low::Species::Crocodile => Vec3::new(1.0, 2.4, 1.3),
                quadruped_low::Species::Deadwood => Vec3::new(1.0, 0.5, 1.3),
                quadruped_low::Species::Lavadrake => Vec3::new(1.0, 2.5, 1.3),
                quadruped_low::Species::Maneater => Vec3::new(1.0, 1.6, 4.0),
                quadruped_low::Species::Monitor => Vec3::new(1.0, 2.3, 1.5),
                quadruped_low::Species::Pangolin => Vec3::new(1.0, 2.0, 1.3),
                quadruped_low::Species::Rocksnapper => Vec3::new(1.0, 1.6, 2.9),
                quadruped_low::Species::Salamander => Vec3::new(1.0, 2.4, 1.3),
                quadruped_low::Species::Tortoise => Vec3::new(1.0, 1.6, 2.0),
                _ => Vec3::new(1.0, 1.6, 1.3),
            },
            Body::Ship(ship) => Vec3::new(1.0, 1.0, 1.0),//ship.dimensions(),
            Body::Theropod(body) => match body.species {
                theropod::Species::Archaeos => Vec3::new(4.0, 7.0, 8.0),
                theropod::Species::Ntouka => Vec3::new(4.0, 6.0, 8.0),
                theropod::Species::Odonto => Vec3::new(4.0, 6.5, 8.0),
                theropod::Species::Sandraptor => Vec3::new(2.0, 3.0, 2.6),
                theropod::Species::Snowraptor => Vec3::new(2.0, 3.0, 2.6),
                theropod::Species::Sunlizard => Vec3::new(2.0, 3.6, 2.5),
                theropod::Species::Woodraptor => Vec3::new(2.0, 3.0, 2.6),
                theropod::Species::Yale => Vec3::new(1.5, 3.2, 6.0),
            },
        }
    }

    // Note: This is used for collisions, but it's not very accurate for shapes that
    // are very much not cylindrical. Eventually this ought to be replaced by more
    // accurate collision shapes.
    pub fn radius(&self) -> f32 {
        let dim = self.dimensions();
        dim.x.max(dim.y) / 2.0
    }

    pub fn height(&self) -> f32 { self.dimensions().z }

    pub fn base_energy(&self, body_energies: &AllBodiesBaseEnergy) -> u32 {
        match self {
            Body::BipedLarge(body) => {
                get_body_u32_attribute(&body_energies.biped_large, body.species)
            },
            Body::BipedSmall(body) => {
                get_body_u32_attribute(&body_energies.biped_small, body.species)
            },
            Body::BirdMedium(body) => {
                get_body_u32_attribute(&body_energies.bird_medium, body.species)
            },
            Body::BirdLarge(body) => {
                get_body_u32_attribute(&body_energies.bird_large, body.species)
            },
            Body::FishSmall(body) => {
                get_body_u32_attribute(&body_energies.fish_small, body.species)
            },
            Body::FishMedium(body) => {
                get_body_u32_attribute(&body_energies.fish_medium, body.species)
            },
            Body::Humanoid(body) => get_body_u32_attribute(&body_energies.humanoid, body.species),
            Body::QuadrupedMedium(body) => {
                get_body_u32_attribute(&body_energies.quadruped_medium, body.species)
            },
            Body::QuadrupedSmall(body) => {
                get_body_u32_attribute(&body_energies.quadruped_small, body.species)
            },
            Body::Theropod(body) => get_body_u32_attribute(&body_energies.theropod, body.species),
            Body::Dragon(body) => get_body_u32_attribute(&body_energies.dragon, body.species),
            Body::QuadrupedLow(body) => {
                get_body_u32_attribute(&body_energies.quadruped_low, body.species)
            },
            Body::Golem(body) => get_body_u32_attribute(&body_energies.golem, body.species),
            Body::Ship(_) | Body::Object(_) => 1000,
        }
    }

    pub fn base_health(&self, body_healths: &AllBodiesBaseHealth) -> u32 {
        match self {
            Body::BipedLarge(body) => {
                get_body_u32_attribute(&body_healths.biped_large, body.species)
            },
            Body::BipedSmall(body) => {
                get_body_u32_attribute(&body_healths.biped_small, body.species)
            },
            Body::BirdMedium(body) => {
                get_body_u32_attribute(&body_healths.bird_medium, body.species)
            },
            Body::BirdLarge(body) => get_body_u32_attribute(&body_healths.bird_large, body.species),
            Body::FishSmall(body) => get_body_u32_attribute(&body_healths.fish_small, body.species),
            Body::FishMedium(body) => {
                get_body_u32_attribute(&body_healths.fish_medium, body.species)
            },
            Body::Humanoid(body) => get_body_u32_attribute(&body_healths.humanoid, body.species),
            Body::QuadrupedMedium(body) => {
                get_body_u32_attribute(&body_healths.quadruped_medium, body.species)
            },
            Body::QuadrupedSmall(body) => {
                get_body_u32_attribute(&body_healths.quadruped_small, body.species)
            },
            Body::Theropod(body) => get_body_u32_attribute(&body_healths.theropod, body.species),
            Body::Dragon(body) => get_body_u32_attribute(&body_healths.dragon, body.species),
            Body::QuadrupedLow(body) => {
                get_body_u32_attribute(&body_healths.quadruped_low, body.species)
            },
            Body::Golem(body) => get_body_u32_attribute(&body_healths.golem, body.species),
            Body::Ship(_) => 10000,
            Body::Object(object) => match object {
                object::Body::TrainingDummy => 10000,
                object::Body::Crossbow => 800,
                object::Body::HaniwaSentry => 600,
                _ => 10000,
            },
        }
    }

    pub fn base_health_increase(&self, body_health_increases: &AllBodiesBaseHealthIncrease) -> u32 {
        match self {
            Body::BipedLarge(body) => {
                get_body_u32_attribute(&body_health_increases.biped_large, body.species)
            },
            Body::BipedSmall(body) => {
                get_body_u32_attribute(&body_health_increases.biped_small, body.species)
            },
            Body::BirdMedium(body) => {
                get_body_u32_attribute(&body_health_increases.bird_medium, body.species)
            },
            Body::BirdLarge(body) => {
                get_body_u32_attribute(&body_health_increases.bird_large, body.species)
            },
            Body::FishSmall(body) => {
                get_body_u32_attribute(&body_health_increases.fish_small, body.species)
            },
            Body::FishMedium(body) => {
                get_body_u32_attribute(&body_health_increases.fish_medium, body.species)
            },
            Body::Humanoid(body) => {
                get_body_u32_attribute(&body_health_increases.humanoid, body.species)
            },
            Body::QuadrupedMedium(body) => {
                get_body_u32_attribute(&body_health_increases.quadruped_medium, body.species)
            },
            Body::QuadrupedSmall(body) => {
                get_body_u32_attribute(&body_health_increases.quadruped_small, body.species)
            },
            Body::Theropod(body) => {
                get_body_u32_attribute(&body_health_increases.theropod, body.species)
            },
            Body::Dragon(body) => {
                get_body_u32_attribute(&body_health_increases.dragon, body.species)
            },
            Body::QuadrupedLow(body) => {
                get_body_u32_attribute(&body_health_increases.quadruped_low, body.species)
            },
            Body::Golem(body) => get_body_u32_attribute(&body_health_increases.golem, body.species),
            Body::Ship(_) => 500,
            Body::Object(_) => 10,
        }
    }

    pub fn flying_height(&self) -> f32 {
        match self {
            Body::BirdLarge(_) => 50.0,
            Body::BirdMedium(_) => 40.0,
            Body::Dragon(_) => 60.0,
            //Body::Ship(ship::Body::DefaultAirship) => 60.0,
            Body::Ship(_) => 60.0,
            _ => 0.0,
        }
    }

    pub fn immune_to(&self, buff: BuffKind) -> bool {
        match buff {
            BuffKind::Bleeding => matches!(self, Body::Object(_) | Body::Golem(_) | Body::Ship(_)),
            BuffKind::Burning => match self {
                Body::Golem(g) => matches!(g.species, golem::Species::ClayGolem),
                Body::BipedSmall(b) => matches!(b.species, biped_small::Species::Haniwa),
                Body::Object(object::Body::HaniwaSentry) => true,
                _ => false,
            },
            _ => false,
        }
    }

    /// Returns a multiplier representing increased difficulty not accounted for
    /// due to AI or not using an actual weapon
    // TODO: Match on species
    pub fn combat_multiplier(&self) -> f32 {
        match self {
            Body::Object(_) | Body::Ship(_) => 0.0,
            Body::BipedLarge(b) => match b.species {
                biped_large::Species::Mindflayer => 4.8,
                biped_large::Species::Minotaur => 3.2,
                _ => 1.0,
            },
            Body::Golem(g) => match g.species {
                golem::Species::ClayGolem => 1.2,
                _ => 1.0,
            },
            _ => 1.0,
        }
    }

    pub fn base_poise(&self) -> u32 {
        match self {
            Body::Humanoid(_) => 100,
            Body::BipedLarge(_) => 250,
            Body::Golem(_) => 300,
            _ => 100,
        }
    }

    /// Returns the eye height for this creature.
    pub fn eye_height(&self) -> f32 { self.height() * 0.9 }

    pub fn default_light_offset(&self) -> Vec3<f32> {
        // TODO: Make this a manifest
        match self {
            Body::Object(_) => Vec3::unit_z() * 0.5,
            _ => Vec3::unit_z(),
        }
    }

    pub fn can_strafe(&self) -> bool {
        matches!(
            self,
            Body::Humanoid(_) | Body::BipedSmall(_) | Body::BipedLarge(_)
        )
    }

    pub fn mounting_offset(&self) -> Vec3<f32> {
        match self {
            //Body::Ship(ship::Body::DefaultAirship) => Vec3::from([0.0, 0.0, 10.0]),
            Body::Ship(_) => Vec3::from([0.0, 0.0, 10.0]),
            _ => Vec3::unit_z(),
        }
    }
}

impl Component for Body {
    type Storage = DerefFlaggedStorage<Self, IdvStorage<Self>>;
}
