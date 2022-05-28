#[cfg(not(target_arch = "wasm32"))]
use crate::comp::Pos;
use serde::{Deserialize, Serialize};
#[cfg(not(target_arch = "wasm32"))]
use specs::Entity;

/// A resource that stores the time of day.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, Default)]
pub struct TimeOfDay(pub f64);

/// A resource that stores the tick (i.e: physics) time.
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Time(pub f64);

/// A resource that stores the Server-Time, its needed to calculate the input
/// delay and adjust Time on the client keep in mind that it contains a
/// network-latency. Only filled on the client
#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct ServerTime(pub f64);

/// A resource that stores the time since the previous tick.
#[derive(Default)]
pub struct DeltaTime(pub f32);

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
pub struct EntitiesDiedLastTick(pub Vec<(Entity, Pos)>);

/// A resource that indicates what mode the local game is being played in.
#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GameMode {
    /// The game is being played in server mode (i.e: the code is running
    /// server-side)
    Server,
    /// The game is being played in client mode (i.e: the code is running
    /// client-side)
    Client,
    /// The game is being played in singleplayer mode (i.e: both client and
    /// server at once)
    // To be used later when we no longer start up an entirely new server for singleplayer
    Singleplayer,
}

/// A resource that stores the player's entity (on the client), and None on the
/// server
#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone, Default, Debug)]
pub struct PlayerEntity(pub Option<Entity>);

/// Describe how players interact with other players.
///
/// May be removed when we will discover better way
/// to handle duels and murders
#[derive(PartialEq, Eq, Copy, Clone, Debug, Deserialize, Serialize)]
pub enum BattleMode {
    PvP,
    PvE,
}
