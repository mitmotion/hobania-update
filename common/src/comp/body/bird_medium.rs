use crate::{make_case_elim, make_proj_elim};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

make_proj_elim!(
    body,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Body {
        pub species: Species,
        pub body_type: BodyType,
    }
);

impl Body {
    pub fn random() -> Self {
        let mut rng = thread_rng();
        let species = *(&ALL_SPECIES).choose(&mut rng).unwrap();
        Self::random_with(&mut rng, &species)
    }

    #[inline]
    pub fn random_with(rng: &mut impl rand::Rng, &species: &Species) -> Self {
        let body_type = *(&ALL_BODY_TYPES).choose(rng).unwrap();
        Self { species, body_type }
    }
}

impl From<Body> for super::Body {
    fn from(body: Body) -> Self { super::Body::BirdMedium(body) }
}

make_case_elim!(
    species,
    #[derive(
        Copy, Clone, Debug, Display, EnumString, PartialEq, Eq, Hash, Serialize, Deserialize,
    )]
    #[repr(u32)]
    pub enum Species {
        Duck = 0,
        Chicken = 1,
        Goose = 2,
        Peacock = 3,
        Eagle = 4,
        Owl = 5,
        Parrot = 6,
        Penguin = 7,
    }
);

/// Data representing per-species generic data.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
#[derive(Clone, Debug, Deserialize)]
pub struct AllSpecies<SpeciesMeta> {
    pub duck: SpeciesMeta,
    pub chicken: SpeciesMeta,
    pub goose: SpeciesMeta,
    pub peacock: SpeciesMeta,
    pub eagle: SpeciesMeta,
    pub owl: SpeciesMeta,
    pub parrot: SpeciesMeta,
    pub penguin: SpeciesMeta,
}

impl<'a, SpeciesMeta> core::ops::Index<&'a Species> for AllSpecies<SpeciesMeta> {
    type Output = SpeciesMeta;

    #[inline]
    fn index(&self, &index: &'a Species) -> &Self::Output {
        match index {
            Species::Duck => &self.duck,
            Species::Chicken => &self.chicken,
            Species::Goose => &self.goose,
            Species::Peacock => &self.peacock,
            Species::Eagle => &self.eagle,
            Species::Owl => &self.owl,
            Species::Parrot => &self.parrot,
            Species::Penguin => &self.penguin,
        }
    }
}

pub const ALL_SPECIES: [Species; 8] = [
    Species::Duck,
    Species::Chicken,
    Species::Goose,
    Species::Peacock,
    Species::Eagle,
    Species::Owl,
    Species::Parrot,
    Species::Penguin,
];

impl<'a, SpeciesMeta: 'a> IntoIterator for &'a AllSpecies<SpeciesMeta> {
    type IntoIter = std::iter::Copied<std::slice::Iter<'static, Self::Item>>;
    type Item = Species;

    fn into_iter(self) -> Self::IntoIter { ALL_SPECIES.iter().copied() }
}

make_case_elim!(
    body_type,
    #[derive(
        Copy, Clone, Debug, Display, EnumString, PartialEq, Eq, Hash, Serialize, Deserialize,
    )]
    #[repr(u32)]
    pub enum BodyType {
        Female = 0,
        Male = 1,
    }
);
pub const ALL_BODY_TYPES: [BodyType; 2] = [BodyType::Female, BodyType::Male];
