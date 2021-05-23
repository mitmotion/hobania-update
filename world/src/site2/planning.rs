use super::{
    structure::{Cemetary, Hut, Structure},
    *,
};
use common::lottery::Lottery;

// All are weights, must be positive, 1.0 is default.
pub struct Values {
    defence: f32,
    farming: f32,
    housing: f32,
}

impl Site {
    // TODO: How long is a tick? A year?
    pub fn tick(&mut self, land: &Land, rng: &mut impl Rng) {
        let values = Values {
            defence: 1.0,
            farming: 1.0,
            housing: 1.0,
        };

        match *Lottery::from_slice(&mut [
            (10.0, 0), // Huts
        ])
        .choose_seeded(rng.gen())
        {
            0 => {
                Cemetary::choose_location((), land, self, rng)
                    .map(|cemetary| cemetary.generate(land, self, rng));
            },
            1 => {
                Hut::choose_location((), land, self, rng).map(|hut| hut.generate(land, self, rng));
            },
            _ => unreachable!(),
        }
    }
}
