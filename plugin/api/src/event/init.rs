use super::*;

#[derive(Serialize, Deserialize)]
pub struct RawInit<'a> {
    pub mode: GameMode,
    pub phantom: PhantomData<&'a ()>,
}

pub struct Init<'a> {
    mode: GameMode,
    pub phantom: PhantomData<&'a ()>,
}

impl<'a> Event<'a> for Init<'a> {
    type Raw = RawInit<'a>;
    type Response = ();

    fn from_raw(game: &'a Game, raw: Self::Raw) -> Self {
        Self {
            mode: raw.mode,
            phantom: PhantomData,
        }
    }

    fn get_handler_name(raw: &Self::Raw) -> Cow<'_, str> { Cow::Borrowed("on_init") }
}

impl<'a> Init<'a> {
    pub fn mode(&self) -> GameMode { self.mode }
}
