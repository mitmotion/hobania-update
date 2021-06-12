pub mod command;
pub mod init;
pub mod player_join;

pub use self::{
    command::Command,
    init::Init,
    player_join::{PlayerJoin, PlayerJoinResponse},
};

use super::*;

pub trait Event<'a>: Send + Sync {
    type Response: Serialize + DeserializeOwned + Send + Sync + 'a;
    type Raw: Serialize + Deserialize<'a> + Send + Sync + 'a;

    fn from_raw(game: &'a Game, raw: Self::Raw) -> Self;
    fn get_handler_name(raw: &Self::Raw) -> Cow<'_, str>;
}
