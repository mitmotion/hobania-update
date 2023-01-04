use crate::{make_case_elim, make_proj_elim};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};

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
        let species = *ALL_SPECIES.choose(&mut rng).unwrap();
        Self::random_with(&mut rng, &species)
    }

    #[inline]
    pub fn random_with(rng: &mut impl rand::Rng, &species: &Species) -> Self {
        let body_type = *ALL_BODY_TYPES.choose(rng).unwrap();
        Self { species, body_type }
    }
}

impl From<Body> for super::Body {
    fn from(body: Body) -> Self { super::Body::BirdLarge(body) }
}

make_case_elim!(
    species,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[repr(u32)]
    pub enum Species {
        Phoenix = 0,
        Cockatrice = 1,
        Roc = 2,
        FlameWyvern = 3,
        CloudWyvern = 4,
        FrostWyvern = 5,
        SeaWyvern = 6,
        WealdWyvern = 7,
    }
);

/// Data representing per-species generic data.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
#[derive(Clone, Debug, Deserialize)]
pub struct AllSpecies<SpeciesMeta> {
    pub phoenix: SpeciesMeta,
    pub cockatrice: SpeciesMeta,
    pub roc: SpeciesMeta,
    pub wyvern_flame: SpeciesMeta,
    pub wyvern_cloud: SpeciesMeta,
    pub wyvern_frost: SpeciesMeta,
    pub wyvern_sea: SpeciesMeta,
    pub wyvern_weald: SpeciesMeta,
}

impl<'a, SpeciesMeta> core::ops::Index<&'a Species> for AllSpecies<SpeciesMeta> {
    type Output = SpeciesMeta;

    #[inline]
    fn index(&self, &index: &'a Species) -> &Self::Output {
        match index {
            Species::Phoenix => &self.phoenix,
            Species::Cockatrice => &self.cockatrice,
            Species::Roc => &self.roc,
            Species::FlameWyvern => &self.wyvern_flame,
            Species::CloudWyvern => &self.wyvern_cloud,
            Species::FrostWyvern => &self.wyvern_frost,
            Species::SeaWyvern => &self.wyvern_sea,
            Species::WealdWyvern => &self.wyvern_weald,
        }
    }
}

pub const ALL_SPECIES: [Species; 8] = [
    Species::Phoenix,
    Species::Cockatrice,
    Species::Roc,
    Species::FlameWyvern,
    Species::CloudWyvern,
    Species::FrostWyvern,
    Species::SeaWyvern,
    Species::WealdWyvern,
];

impl<'a, SpeciesMeta: 'a> IntoIterator for &'a AllSpecies<SpeciesMeta> {
    type IntoIter = std::iter::Copied<std::slice::Iter<'static, Self::Item>>;
    type Item = Species;

    fn into_iter(self) -> Self::IntoIter { ALL_SPECIES.iter().copied() }
}

make_case_elim!(
    body_type,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[repr(u32)]
    pub enum BodyType {
        Female = 0,
        Male = 1,
    }
);

pub const ALL_BODY_TYPES: [BodyType; 2] = [BodyType::Female, BodyType::Male];
