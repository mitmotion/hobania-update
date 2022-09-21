#![deny(unsafe_code)]
#![allow(incomplete_features)]
#![type_length_limit = "1664759"]
#![allow(clippy::option_map_unit_fn)]
#![deny(clippy::clone_on_ref_ptr)]
#![feature(
    associated_type_defaults,
    bool_to_option,
    fundamental,
    label_break_value,
    let_chains,
    option_zip,
    trait_alias,
    type_alias_impl_trait,
    extend_one,
    arbitrary_enum_discriminant,
    generic_associated_types,
    arbitrary_self_types
)]
#![feature(hash_drain_filter)]

macro_rules! cfg_not_wasm {
    ($($item:item)*) => {
        $(
            #[cfg(not(target_arch = "wasm32"))]
            $item
        )*
    }
}

// Re-exported crates
cfg_not_wasm! {
    pub use common_assets as assets;
    pub use uuid;
}

// Modules

pub mod combat;
pub mod comp;
pub mod consts;
pub mod resources;
pub mod uid;

// NOTE: Comment out macro to get rustfmt to re-order these as needed.
cfg_not_wasm! {
    pub mod astar;
    pub mod calendar;
    pub mod character;
    pub mod clock;
    pub mod cmd;
    pub mod depot;
    pub mod effect;
    pub mod event;
    pub mod explosion;
    pub mod figure;
    pub mod generation;
    pub mod grid;
    pub mod link;
    pub mod lod;
    pub mod lottery;
    pub mod mounting;
    pub mod npc;
    pub mod outcome;
    pub mod path;
    pub mod ray;
    pub mod recipe;
    pub mod region;
    pub mod rtsim;
    pub mod skillset_builder;
    pub mod slowjob;
    pub mod spiral;
    pub mod states;
    pub mod store;
    pub mod terrain;
    pub mod time;
    pub mod trade;
    pub mod util;
    pub mod vol;
    pub mod volumes;
    pub mod weather;

    mod cached_spatial_grid;
    mod view_distances;
}

// We declare a macro in this module so there are issues referring to it by path
// within this crate if typed module is declared in macro expansion.
#[cfg(not(target_arch = "wasm32"))] pub mod typed;

pub use combat::{DamageKind, DamageSource};

cfg_not_wasm! {
    pub use cached_spatial_grid::CachedSpatialGrid;
    pub use combat::{Damage, GroupTarget, Knockback, KnockbackDir};
    pub use comp::inventory::loadout_builder::LoadoutBuilder;
    pub use explosion::{Explosion, RadiusEffect};
    pub use skillset_builder::SkillSetBuilder;
    pub use view_distances::ViewDistances;
}
