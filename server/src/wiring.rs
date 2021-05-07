use common::comp::ProjectileConstructor;
use core::f32;
use hashbrown::HashMap;
use specs::{Component, Entity};
use specs_idvs::IdvStorage;
use vek::Vec3;

pub struct Logic {
    pub kind: LogicKind,
    pub left: OutputFormula,
    pub right: OutputFormula,
}

pub struct WiringElement {
    pub inputs: HashMap<String, f32>,
    pub outputs: HashMap<String, OutputFormula>,
    pub actions: Vec<WiringAction>,
}

pub struct Circuit {
    pub wires: Vec<Wire>,
}

pub enum OutputFormula {
    Constant {
        value: f32,
    },
    Input {
        name: String,
    },
    Logic(Box<Logic>),

    SineWave {
        amplitude: Box<OutputFormula>,
        frequency: Box<OutputFormula>,
    },
    OnCollide {
        value: Box<OutputFormula>,
    },
    OnInteract {
        value: Box<OutputFormula>,
    },
    OnDeath {
        value: Box<OutputFormula>,
        radius: Box<OutputFormula>,
    },
}

pub enum LogicKind {
    Min, // acts like And
    Max, // acts like Or
    Sub, // `|x| { 5.0 - x }` acts like Not, depending on reference voltages
    Sum,
    Mul,
}

pub struct WiringAction {
    pub formula: OutputFormula,
    pub threshold: OutputFormula,
    pub effects: Vec<WiringActionEffect>,
}

pub enum WiringActionEffect {
    SpawnProjectile {
        constr: ProjectileConstructor,
    },
    SetBlockCollidability {
        coords: Vec3<i32>,
        collidable: bool,
    },
    SetLight {
        r: OutputFormula,
        g: OutputFormula,
        b: OutputFormula,
    },
}

pub struct Wire {
    pub input_field: String,
    pub output_field: String,
    pub input_entity: Entity,
    pub output_entity: Entity,
}

impl Component for WiringElement {
    type Storage = IdvStorage<Self>;
}

impl Component for Circuit {
    type Storage = IdvStorage<Self>;
}

pub fn BoxConst(constant_value: f32) -> Box<OutputFormula> { Box::new(Const(constant_value)) }

pub fn Const(constant_value: f32) -> OutputFormula {
    OutputFormula::Constant {
        value: constant_value,
    }
}

pub fn BoxInput(name: &str) -> Box<OutputFormula> { Box::new(Input(name)) }

pub fn Input(name: &str) -> OutputFormula {
    OutputFormula::Input {
        name: name.to_string(),
    }
}
