use crate::{
    comp::{
        fluid_dynamics::{Drag, Glide, WingShape, WingState},
        Ori,
    },
    make_case_elim, make_proj_elim,
};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use vek::*;

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

    /// Dimensions of the body (wings folded)
    pub const fn dimensions(&self) -> Vec3<f32> { Vec3::new(5.0, 10.0, 16.0) }

    /// Distance from wing tip to wing tip and leading edge to trailing edge
    /// respectively
    // TODO: Check
    pub const fn wing_dimensions(&self) -> Vec2<f32> { Vec2::new(16.0, 5.0) }

    pub fn flying<'a>(&'a self, ori: &'a Ori) -> FlyingDragon<'a> {
        FlyingDragon::from((self, ori))
    }
}

impl From<Body> for super::Body {
    fn from(body: Body) -> Self { super::Body::Dragon(body) }
}

make_case_elim!(
    species,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[repr(u32)]
    pub enum Species {
        Reddragon = 0,
    }
);

/// Data representing per-species generic data.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
#[derive(Clone, Debug, Deserialize)]
pub struct AllSpecies<SpeciesMeta> {
    pub reddragon: SpeciesMeta,
}

impl<'a, SpeciesMeta> core::ops::Index<&'a Species> for AllSpecies<SpeciesMeta> {
    type Output = SpeciesMeta;

    #[inline]
    fn index(&self, &index: &'a Species) -> &Self::Output {
        match index {
            Species::Reddragon => &self.reddragon,
        }
    }
}

pub const ALL_SPECIES: [Species; 1] = [Species::Reddragon];

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

#[derive(Copy, Clone)]
pub struct FlyingDragon<'a> {
    wing_shape: WingShape,
    wing_state: WingState,
    planform_area: f32,
    body: &'a Body,
    ori: &'a Ori,
}

impl<'a> From<(&'a Body, &'a Ori)> for FlyingDragon<'a> {
    fn from((body, ori): (&'a Body, &'a Ori)) -> Self {
        let Vec2 {
            x: span_length,
            y: chord_length,
        } = body.wing_dimensions();
        let planform_area = WingShape::elliptical_planform_area(span_length, chord_length);
        FlyingDragon {
            wing_shape: WingShape::elliptical(span_length, chord_length),
            wing_state: WingState::Flapping,
            planform_area,
            body,
            ori,
        }
    }
}

impl Drag for Body {
    fn parasite_drag_coefficient(&self) -> f32 {
        let radius = self.dimensions().map(|a| a * 0.5);
        // "Field Estimates of body::Body Drag Coefficient on the Basis of
        // Dives in Passerine Birds", Anders Hedenström and Felix Liechti, 2001
        const CD: f32 = 0.2;
        CD * std::f32::consts::PI * radius.x * radius.z
    }
}

impl Drag for FlyingDragon<'_> {
    fn parasite_drag_coefficient(&self) -> f32 {
        self.body.parasite_drag_coefficient() + self.planform_area * 0.004
    }
}

impl Glide for FlyingDragon<'_> {
    fn wing_shape(&self) -> &WingShape { &self.wing_shape }

    fn is_gliding(&self) -> bool { matches!(self.wing_state, WingState::Fixed) }

    fn planform_area(&self) -> f32 { self.planform_area }

    fn ori(&self) -> &Ori { self.ori }
}
