//! Contains an "x macro" for all synced components as well as [NetSync]
//! implementations for those components.
//!
//!
//! An x macro accepts another macro as input and calls it with a list of
//! inputs. This allows adding components to the list in the x macro declaration
//! and then writing macros that will accept this list and generate code that
//! handles every synced component without further repitition of the component
//! set.
//!
//! This module also re-exports all the component types that are synced.
//!
//! A glob import from this can be used so that the component types are in scope
//! when using the x macro defined here which requires this.

/// This provides a lowercase name and the component type.
//#[rustfmt::skip]
#[macro_export]
macro_rules! synced_components {
    ($macro:ident) => {
        $macro! {
            body: Body,
            stats: Stats,
            buffs: Buffs,
            auras: Auras,
            energy: Energy,
            health: Health,
            poise: Poise,
            light_emitter: LightEmitter,
            item: Item,
            scale: Scale,
            group: Group,
            is_mount: IsMount,
            is_rider: IsRider,
            mass: Mass,
            density: Density,
            collider: Collider,
            sticky: Sticky,
            character_state: CharacterState,
            shockwave: Shockwave,
            beam_segment: BeamSegment,
            alignment: Alignment,
            // TODO: evaluate if this is used on the client,
            // and if so what it is used for
            player: Player,
            // TODO: change this to `SyncFrom::ClientEntity` and sync the bare minimum
            // from other entities (e.g. just keys needed to show appearance
            // based on their loadout). Also, it looks like this actually has
            // an alternative sync method implemented in entity_sync via
            // ServerGeneral::InventoryUpdate so we could use that instead
            // or remove the part where it clones the inventory.
            inventory: Inventory,
            // TODO: this is used in combat rating calculation in voxygen but we can probably
            // remove it from that and then see if it's used for anything else and try to move
            // to only being synced for the client's entity.
            skill_set: SkillSet,

            // Synced to the client only for its own entity

            combo: Combo,
            active_abilities: ActiveAbilities,
            can_build: CanBuild,
        }
    };
}

macro_rules! reexport_comps {
    ($($name:ident: $type:ident,)*) => {
        mod inner {
            pub use common::comp::*;
            use common::link::Is;
            use common::mounting::{Mount, Rider};

            pub type IsMount = Is<Mount>;
            pub type IsRider = Is<Rider>;
        }

        $(pub use inner::$type;)*
    }
}
synced_components!(reexport_comps);

// ===============================
// === NetSync implementations ===
// ===============================

use crate::sync::{NetSync, SyncFrom};

impl NetSync for Body {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Stats {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Buffs {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Auras {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Energy {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Health {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;

    fn pre_insert(&mut self, world: &specs::World) {
        use common::resources::Time;
        use specs::WorldExt;

        // Time isn't synced between client and server so replace the Time from the
        // server with the Client's local Time to enable accurate comparison.
        self.last_change.time = *world.read_resource::<Time>();
    }

    fn pre_modify(&mut self, world: &specs::World) {
        use common::resources::Time;
        use specs::WorldExt;

        // Time isn't synced between client and server so replace the Time from the
        // server with the Client's local Time to enable accurate comparison.
        self.last_change.time = *world.read_resource::<Time>();
    }
}

impl NetSync for Poise {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for LightEmitter {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Item {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Scale {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Group {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for IsMount {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for IsRider {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Mass {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Density {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Collider {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Sticky {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for CharacterState {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Shockwave {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for BeamSegment {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Alignment {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Player {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for Inventory {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

impl NetSync for SkillSet {
    const SYNC_FROM: SyncFrom = SyncFrom::AllEntities;
}

// SyncFrom::ClientEntity

impl NetSync for Combo {
    const SYNC_FROM: SyncFrom = SyncFrom::ClientEntity;
}

impl NetSync for ActiveAbilities {
    const SYNC_FROM: SyncFrom = SyncFrom::ClientEntity;
}

impl NetSync for CanBuild {
    const SYNC_FROM: SyncFrom = SyncFrom::ClientEntity;
}
