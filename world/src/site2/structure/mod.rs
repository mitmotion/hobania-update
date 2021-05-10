mod hut;

pub use self::{
    hut::Hut,
};

use super::*;

pub trait Structure: Sized {
    type Config;

    /// Attempt to choose a location to place this plot in the given site.
    fn choose_location<R: Rng>(cfg: Self::Config, land: &Land, site: &Site, rng: &mut R) -> Option<Self>;

    /// Generate the plot with the given location information on the given site
    fn generate<R: Rng>(self, land: &Land, site: &mut Site, rng: &mut R);
}
