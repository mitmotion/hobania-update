//! The following is a proposal for quests in Veloren. A server side quest system would maintain
//! the necessary conditions for the quests. Players would be given a `Quests` component which
//! would store current quests. Each quest consists of a name and a vector of quest conditions.
//! These conditions are popped off the vector until it is empty. At that point the quest is
//! complete. Here is a sample quest:
//!
//! ```
//! Quest {
//!     name: "Escort Bob safely to Thead for Joe",
//!     conditions: [
//!         Find(RtSimEntity(3, "Bob"), QuestLocation {
//!                                         kind: Site("Scrin"),
//!                                         location: None,
//!                                     }),
//!         Escort(RtSimEntity(3, "Bob"), QuestLocation {
//!                                         kind: Site("Thead"),
//!                                         location: None,
//!                                     }),
//!         CollectReward(QuestItem{
//!                             item_definition_id: "common.items.utility.coins",
//!                             quantity: 1000,
//!                         }),
//!     ],
//! }
//! ```
//!
//! The quest must be completed in order. Failure would be monitored by the server side quest
//! system. (Eg. escort conditions would fail if the escorted entity died.)

use crate::{comp::Body, rtsim::RtSimId};
use specs::{Component, DerefFlaggedStorage};
use specs_idvs::IdvStorage;
use serde::{Deserialize, Serialize};
use vek::*;

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Quests {
    pub quests: Vec<Quest>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Quest {
    name: String,
    conditions: Vec<QuestCondition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuestCondition {
    /// Kill entities. The vector allows multiple target types to be included in the same step of
    /// the quest
    Kill(Vec<QuestCharacterTarget>),
    /// Escort one or multiple targets to a destination
    Escort(Vec<QuestCharacterTarget>, QuestLocation),
    /// Take an item to a location. Similar to `Escort` but for items, not characters
    TakeItemTo(Vec<QuestItem>, QuestLocation),
    /// Find something(s) from an optional location
    Find(FindQuestQuery, Option<QuestLocation>),
    /// Go to a given location
    Goto(QuestLocation),
    /// A time at which all the previous conditions need to be completed to prevent failure
    CompleteBy(f64),
    /// Survive without dying or respawning until a given time
    SurviveUntil(f64),
    /// Collect a reward
    CollectReward(Vec<QuestItem>, QuestLocation),
}

/// An item and quantity for use as a reward, an item in or quest, etc
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestItem {
    item_definition_id: String,
    quantity: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestCharacterTarget {
    kind: CharacterTargetKind,
    quantity: u32,
}

/// Something to look for in a find quest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CharacterTargetKind {
    /// RtSim id and name
    RtSimEntity(RtSimId, String),
    Creature(Body),
}

/// Something to look for in a find quest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FindQuestQuery {
    /// RtSim id and name
    RtSimEntity(RtSimId, String),
    Site(String),
    Creature(Body),
    Item(QuestItem),
}

/// A location for a quest, either as a destination or a place to look for something.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuestLocation {
    kind: QuestLocationKind,
    location: Option<Vec2<i32>>,
}

/// Various kinds of locations and destinations for quests.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum QuestLocationKind {
    Site(String),
    RtSimEntity(RtSimId, String),
    Direction(Direction),
    Coordinates,
}

/// Cardinal directions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Direction {
    North,
    South,
    East,
    West,
    Northeast,
    Northwest,
    Southeast,
    Southwest,
}

impl Component for Quests {
    type Storage = DerefFlaggedStorage<Self, IdvStorage<Self>>;
}
