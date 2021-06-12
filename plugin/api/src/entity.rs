use super::*;

#[derive(Copy, Clone)]
pub struct Entity<'a> {
    pub(crate) game: &'a Game,
    pub(crate) uid: Uid,
}

impl<'a> Entity<'a> {
    pub fn uid(&self) -> Uid { self.uid }

    pub fn send_chat_msg<T: ToString>(&self, msg: T) {
        self.game.emit(RawAction::EntitySendChatMessage(
            self.uid,
            Cow::Owned(msg.to_string()),
        ))
    }

    // TODO
    pub fn get_name(&self) -> String { todo!() }

    pub fn get_health(&self) -> String { todo!() }
}
