use super::*;

pub type Uuid = [u8; 16];

#[derive(Serialize, Deserialize)]
pub struct RawPlayerJoin<'a> {
    pub uuid: Uuid,
    pub alias: Cow<'a, str>,
}

pub struct PlayerJoin<'a> {
    uuid: Uuid,
    alias: Cow<'a, str>,
}

impl<'a> Event<'a> for PlayerJoin<'a> {
    type Raw = RawPlayerJoin<'a>;
    type Response = PlayerJoinResponse<'a>;

    fn from_raw(game: &'a Game, raw: Self::Raw) -> Self {
        Self {
            uuid: raw.uuid,
            alias: raw.alias,
        }
    }

    fn get_handler_name(raw: &Self::Raw) -> Cow<'_, str> { Cow::Borrowed("on_player_join") }
}

impl<'a> PlayerJoin<'a> {
    pub fn uuid(&self) -> Uuid { self.uuid }

    pub fn alias(&self) -> &str { &self.alias }
}

#[derive(Serialize, Deserialize)]
pub enum PlayerJoinResponse<'a> {
    Accept,
    Reject { reason: Cow<'a, str> },
}
