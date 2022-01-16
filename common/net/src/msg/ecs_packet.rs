use crate::sync;
use common::{
    comp,
    resources::Time,
    mounting::{Mount, Rider},
    link::Is,
};
use serde::{Deserialize, Serialize};
use specs::WorldExt;
use std::marker::PhantomData;
use sum_type::sum_type;

// Automatically derive From<T> for EcsCompPacket
// for each variant EcsCompPacket::T(T.)
sum_type! {
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum EcsCompPacket {
        Body(comp::Body),
        Player(comp::Player),
        CanBuild(comp::CanBuild),
        Stats(comp::Stats),
        SkillSet(comp::SkillSet),
        ActiveAbilities(comp::ActiveAbilities),
        Buffs(comp::Buffs),
        Auras(comp::Auras),
        Energy(comp::Energy),
        Combo(comp::Combo),
        Health(comp::Health),
        Poise(comp::Poise),
        LightEmitter(comp::LightEmitter),
        Inventory(comp::Inventory),
        Item(comp::Item),
        Scale(comp::Scale),
        Group(comp::Group),
        IsMount(Is<Mount>),
        IsRider(Is<Rider>),
        Mass(comp::Mass),
        Density(comp::Density),
        Collider(comp::Collider),
        Sticky(comp::Sticky),
        CharacterState(comp::CharacterState),
        Pos(comp::Pos),
        Vel(comp::Vel),
        Ori(comp::Ori),
        Shockwave(comp::Shockwave),
        BeamSegment(comp::BeamSegment),
        Alignment(comp::Alignment),
    }
}
// Automatically derive From<T> for EcsCompPhantom
// for each variant EcsCompPhantom::T(PhantomData<T>).
sum_type! {
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub enum EcsCompPhantom {
        Body(PhantomData<comp::Body>),
        Player(PhantomData<comp::Player>),
        CanBuild(PhantomData<comp::CanBuild>),
        Stats(PhantomData<comp::Stats>),
        SkillSet(PhantomData<comp::SkillSet>),
        ActiveAbilities(PhantomData<comp::ActiveAbilities>),
        Buffs(PhantomData<comp::Buffs>),
        Auras(PhantomData<comp::Auras>),
        Energy(PhantomData<comp::Energy>),
        Combo(PhantomData<comp::Combo>),
        Health(PhantomData<comp::Health>),
        Poise(PhantomData<comp::Poise>),
        LightEmitter(PhantomData<comp::LightEmitter>),
        Inventory(PhantomData<comp::Inventory>),
        Item(PhantomData<comp::Item>),
        Scale(PhantomData<comp::Scale>),
        Group(PhantomData<comp::Group>),
        IsMount(PhantomData<Is<Mount>>),
        IsRider(PhantomData<Is<Rider>>),
        Mass(PhantomData<comp::Mass>),
        Density(PhantomData<comp::Density>),
        Collider(PhantomData<comp::Collider>),
        Sticky(PhantomData<comp::Sticky>),
        CharacterState(PhantomData<comp::CharacterState>),
        Pos(PhantomData<comp::Pos>),
        Vel(PhantomData<comp::Vel>),
        Ori(PhantomData<comp::Ori>),
        Shockwave(PhantomData<comp::Shockwave>),
        BeamSegment(PhantomData<comp::BeamSegment>),
        Alignment(PhantomData<comp::Alignment>),
    }
}
impl sync::CompPacket for EcsCompPacket {
    type Phantom = EcsCompPhantom;

    fn apply_insert(self, entity: specs::Entity, world: &specs::World, force_update: bool) {
        match self {
            EcsCompPacket::Body(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Player(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::CanBuild(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Stats(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::SkillSet(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::ActiveAbilities(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Buffs(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Auras(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Energy(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Combo(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Health(mut comp) => {
                // Time isn't synced between client and server so replace the Time from the
                // server with the Client's local Time to enable accurate comparison.
                comp.last_change.time = *world.read_resource::<Time>();
                sync::handle_insert(comp, entity, world)
            },
            EcsCompPacket::Poise(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::LightEmitter(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Inventory(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Item(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Scale(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Group(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::IsMount(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::IsRider(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Mass(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Density(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Collider(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Sticky(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::CharacterState(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Pos(comp) => {
                sync::handle_interp_insert(comp, entity, world, force_update)
            },
            EcsCompPacket::Vel(comp) => {
                sync::handle_interp_insert(comp, entity, world, force_update)
            },
            EcsCompPacket::Ori(comp) => {
                sync::handle_interp_insert(comp, entity, world, force_update)
            },
            EcsCompPacket::Shockwave(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::BeamSegment(comp) => sync::handle_insert(comp, entity, world),
            EcsCompPacket::Alignment(comp) => sync::handle_insert(comp, entity, world),
        }
    }

    fn apply_modify(self, entity: specs::Entity, world: &specs::World, force_update: bool) {
        match self {
            EcsCompPacket::Body(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Player(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::CanBuild(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Stats(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::SkillSet(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::ActiveAbilities(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Buffs(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Auras(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Energy(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Combo(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Health(mut comp) => {
                // Time isn't synced between client and server so replace the Time from the
                // server with the Client's local Time to enable accurate comparison.
                comp.last_change.time = *world.read_resource::<Time>();
                sync::handle_modify(comp, entity, world)
            },
            EcsCompPacket::Poise(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::LightEmitter(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Inventory(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Item(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Scale(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Group(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::IsMount(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::IsRider(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Mass(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Density(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Collider(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Sticky(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::CharacterState(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Pos(comp) => {
                sync::handle_interp_modify(comp, entity, world, force_update)
            },
            EcsCompPacket::Vel(comp) => {
                sync::handle_interp_modify(comp, entity, world, force_update)
            },
            EcsCompPacket::Ori(comp) => {
                sync::handle_interp_modify(comp, entity, world, force_update)
            },
            EcsCompPacket::Shockwave(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::BeamSegment(comp) => sync::handle_modify(comp, entity, world),
            EcsCompPacket::Alignment(comp) => sync::handle_modify(comp, entity, world),
        }
    }

    fn apply_remove(phantom: Self::Phantom, entity: specs::Entity, world: &specs::World) {
        match phantom {
            EcsCompPhantom::Body(_) => sync::handle_remove::<comp::Body>(entity, world),
            EcsCompPhantom::Player(_) => sync::handle_remove::<comp::Player>(entity, world),
            EcsCompPhantom::CanBuild(_) => sync::handle_remove::<comp::CanBuild>(entity, world),
            EcsCompPhantom::Stats(_) => sync::handle_remove::<comp::Stats>(entity, world),
            EcsCompPhantom::SkillSet(_) => sync::handle_remove::<comp::SkillSet>(entity, world),
            EcsCompPhantom::ActiveAbilities(_) => {
                sync::handle_remove::<comp::ActiveAbilities>(entity, world)
            },
            EcsCompPhantom::Buffs(_) => sync::handle_remove::<comp::Buffs>(entity, world),
            EcsCompPhantom::Auras(_) => sync::handle_remove::<comp::Auras>(entity, world),
            EcsCompPhantom::Energy(_) => sync::handle_remove::<comp::Energy>(entity, world),
            EcsCompPhantom::Combo(_) => sync::handle_remove::<comp::Combo>(entity, world),
            EcsCompPhantom::Health(_) => sync::handle_remove::<comp::Health>(entity, world),
            EcsCompPhantom::Poise(_) => sync::handle_remove::<comp::Poise>(entity, world),
            EcsCompPhantom::LightEmitter(_) => {
                sync::handle_remove::<comp::LightEmitter>(entity, world)
            },
            EcsCompPhantom::Inventory(_) => sync::handle_remove::<comp::Inventory>(entity, world),
            EcsCompPhantom::Item(_) => sync::handle_remove::<comp::Item>(entity, world),
            EcsCompPhantom::Scale(_) => sync::handle_remove::<comp::Scale>(entity, world),
            EcsCompPhantom::Group(_) => sync::handle_remove::<comp::Group>(entity, world),
            EcsCompPhantom::IsMount(_) => sync::handle_remove::<Is<Mount>>(entity, world),
            EcsCompPhantom::IsRider(_) => sync::handle_remove::<Is<Rider>>(entity, world),
            EcsCompPhantom::Mass(_) => sync::handle_remove::<comp::Mass>(entity, world),
            EcsCompPhantom::Density(_) => sync::handle_remove::<comp::Density>(entity, world),
            EcsCompPhantom::Collider(_) => sync::handle_remove::<comp::Collider>(entity, world),
            EcsCompPhantom::Sticky(_) => sync::handle_remove::<comp::Sticky>(entity, world),
            EcsCompPhantom::CharacterState(_) => {
                sync::handle_remove::<comp::CharacterState>(entity, world)
            },
            EcsCompPhantom::Pos(_) => sync::handle_interp_remove::<comp::Pos>(entity, world),
            EcsCompPhantom::Vel(_) => sync::handle_interp_remove::<comp::Vel>(entity, world),
            EcsCompPhantom::Ori(_) => sync::handle_interp_remove::<comp::Ori>(entity, world),
            EcsCompPhantom::Shockwave(_) => sync::handle_remove::<comp::Shockwave>(entity, world),
            EcsCompPhantom::BeamSegment(_) => sync::handle_remove::<comp::BeamSegment>(entity, world),
            EcsCompPhantom::Alignment(_) => sync::handle_remove::<comp::Alignment>(entity, world),
        }
    }
}
