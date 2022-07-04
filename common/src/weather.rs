use std::fmt;

use serde::{Deserialize, Serialize};
use vek::Vec2;

pub const CHUNKS_PER_CELL: u32 = 16;
// Weather::default is Clear, 0 degrees C and no wind
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Weather {
    /// Clouds currently in the area between 0 and 1
    pub cloud: f32,
    /// Rain per time, between 0 and 1
    pub rain: f32,
    // Wind direction in block / second
    pub wind: Vec2<f32>,
}

impl Weather {
    pub fn new(cloud: f32, rain: f32, wind: Vec2<f32>) -> Self { Self { cloud, rain, wind } }

    pub fn get_kind(&self) -> WeatherKind {
        match (
            (self.cloud * 10.0) as i32,
            (self.rain * 10.0) as i32,
            (self.wind.magnitude() * 10.0) as i32,
        ) {
            // Over 24.5 m/s wind is a storm
            (_, _, 245..) => WeatherKind::Storm,
            (_, 1..=10, _) => WeatherKind::Rain,
            (4..=10, _, _) => WeatherKind::Cloudy,
            _ => WeatherKind::Clear,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WeatherKind {
    Clear,
    Cloudy,
    Rain,
    Storm,
}

impl fmt::Display for WeatherKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeatherKind::Clear => write!(f, "Clear"),
            WeatherKind::Cloudy => write!(f, "Cloudy"),
            WeatherKind::Rain => write!(f, "Rain"),
            WeatherKind::Storm => write!(f, "Storm"),
        }
    }
}
