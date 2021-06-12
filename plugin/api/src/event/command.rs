use super::*;

#[derive(Serialize, Deserialize)]
pub struct RawCommand<'a> {
    pub entity: Uid,
    pub cmd: &'a str,
    pub args: Cow<'a, [&'a str]>,
}

pub struct Command<'a> {
    entity: Entity<'a>,
    cmd: &'a str,
    args: Cow<'a, [&'a str]>,
}

impl<'a> Event<'a> for Command<'a> {
    type Raw = RawCommand<'a>;
    type Response = Result<Vec<String>, String>;

    fn from_raw(game: &'a Game, raw: Self::Raw) -> Self {
        Self {
            entity: game.entity(raw.entity),
            cmd: raw.cmd,
            args: raw.args,
        }
    }

    fn get_handler_name(raw: &Self::Raw) -> Cow<'_, str> {
        Cow::Owned(format!("on_command_{}", raw.cmd))
    }
}

impl<'a> Command<'a> {
    pub fn entity(&self) -> Entity<'_> { self.entity }

    pub fn cmd(&self) -> &str { self.cmd }

    pub fn args(&self) -> &[&str] { &self.args }
}
