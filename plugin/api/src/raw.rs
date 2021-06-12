use super::*;

/// An action to be performed by the game.
#[derive(Serialize, Deserialize, Debug)]
pub enum RawAction<'a> {
    ServerShutdown,
    EntitySendChatMessage(Uid, Cow<'a, str>),
    EntityKill(Uid),
}

/// A request to the game for information.
#[derive(Serialize, Deserialize, Debug)]
pub enum RawRequest<'a> {
    PlayerUidByName(Cow<'a, str>),
    EntityName(Uid),
    EntityHealth(Uid),
}

/// A response to a [`RawRequest`]
#[derive(Serialize, Deserialize, Debug)]
pub enum RawResponse<'a> {
    EntityName(Cow<'a, str>),
    EntityHealth(Health),
}
