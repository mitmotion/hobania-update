use vek::*;
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LodZone {
    pub trees: Vec<LodTree>,
}

impl LodZone {
    /// Size in chunks
    pub const SIZE: u32 = 64;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LodTree {
    pub pos: Vec2<u16>,
    pub alt: u16,
}
