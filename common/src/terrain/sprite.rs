use crate::{
    assets::{Asset, AssetExt, AssetHandle, RonLoader},
    comp::tool::ToolKind,
    lottery::LootSpec,
    make_case_elim,
};
use enum_iterator::IntoEnumIterator;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fmt};

make_case_elim!(
    sprite_kind,
    #[derive(
        Copy,
        Clone,
        Debug,
        Hash,
        Eq,
        PartialEq,
        Serialize,
        Deserialize,
        IntoEnumIterator,
        FromPrimitive,
    )]
    #[repr(u8)]
    pub enum SpriteKind {
        // Note that the values of these should be linearly contiguous to allow for quick
        // bounds-checking when casting to a u8.
        Empty = 0x00,
        BarrelCactus = 0x01,
        RoundCactus = 0x02,
        ShortCactus = 0x03,
        MedFlatCactus = 0x04,
        ShortFlatCactus = 0x05,
        BlueFlower = 0x06,
        PinkFlower = 0x07,
        PurpleFlower = 0x08,
        RedFlower = 0x09,
        WhiteFlower = 0x0A,
        YellowFlower = 0x0B,
        Sunflower = 0x0C,
        LongGrass = 0x0D,
        MediumGrass = 0x0E,
        ShortGrass = 0x0F,
        Apple = 0x10,
        Mushroom = 0x11,
        Liana = 0x12,
        Velorite = 0x13,
        VeloriteFrag = 0x14,
        Chest = 0x15,
        Pumpkin = 0x16,
        Welwitch = 0x17,
        LingonBerry = 0x18,
        LeafyPlant = 0x19,
        Fern = 0x1A,
        DeadBush = 0x1B,
        Blueberry = 0x1C,
        Ember = 0x1D,
        Corn = 0x1E,
        WheatYellow = 0x1F,
        WheatGreen = 0x20,
        Cabbage = 0x21,
        Flax = 0x22,
        Carrot = 0x23,
        Tomato = 0x24,
        Radish = 0x25,
        Coconut = 0x26,
        Turnip = 0x27,
        Window1 = 0x28,
        Window2 = 0x29,
        Window3 = 0x2A,
        Window4 = 0x2B,
        Scarecrow = 0x2C,
        StreetLamp = 0x2D,
        StreetLampTall = 0x2E,
        Door = 0x2F,
        Bed = 0x30,
        Bench = 0x31,
        ChairSingle = 0x32,
        ChairDouble = 0x33,
        CoatRack = 0x34,
        Crate = 0x35,
        DrawerLarge = 0x36,
        DrawerMedium = 0x37,
        DrawerSmall = 0x38,
        DungeonWallDecor = 0x39,
        HangingBasket = 0x3A,
        HangingSign = 0x3B,
        WallLamp = 0x3C,
        Planter = 0x3D,
        Shelf = 0x3E,
        TableSide = 0x3F,
        TableDining = 0x40,
        TableDouble = 0x41,
        WardrobeSingle = 0x42,
        WardrobeDouble = 0x43,
        LargeGrass = 0x44,
        Pot = 0x45,
        Stones = 0x46,
        Twigs = 0x47,
        DropGate = 0x48,
        DropGateBottom = 0x49,
        GrassSnow = 0x4A,
        Reed = 0x4B,
        Beehive = 0x4C,
        LargeCactus = 0x4D,
        VialEmpty = 0x4E,
        PotionMinor = 0x4F,
        GrassBlue = 0x50,
        ChestBuried = 0x51,
        Mud = 0x52,
        FireBowlGround = 0x53,
        CaveMushroom = 0x54,
        Bowl = 0x55,
        SavannaGrass = 0x56,
        TallSavannaGrass = 0x57,
        RedSavannaGrass = 0x58,
        SavannaBush = 0x59,
        Amethyst = 0x5A,
        Ruby = 0x5B,
        Sapphire = 0x5C,
        Emerald = 0x5D,
        Topaz = 0x5E,
        Diamond = 0x5F,
        AmethystSmall = 0x60,
        TopazSmall = 0x61,
        DiamondSmall = 0x62,
        RubySmall = 0x63,
        EmeraldSmall = 0x64,
        SapphireSmall = 0x65,
        WallLampSmall = 0x66,
        WallSconce = 0x67,
        StonyCoral = 0x68,
        SoftCoral = 0x69,
        SeaweedTemperate = 0x6A,
        SeaweedTropical = 0x6B,
        GiantKelp = 0x6C,
        BullKelp = 0x6D,
        WavyAlgae = 0x6E,
        SeaGrapes = 0x6F,
        MermaidsFan = 0x70,
        SeaAnemone = 0x71,
        Seashells = 0x72,
        Seagrass = 0x73,
        RedAlgae = 0x74,
        UnderwaterVent = 0x75,
        Lantern = 0x76,
        CraftingBench = 0x77,
        Forge = 0x78,
        Cauldron = 0x79,
        Anvil = 0x7A,
        CookingPot = 0x7B,
        Garlic = 0x7C,
        Onion = 0x7D,
    }
);

#[derive(Debug, Deserialize)]
pub struct SpriteBehaviorManifest {
    pub solid_height: HashMap<SpriteKind, f32>,
    pub collectible_id: HashMap<SpriteKind, LootSpec>,
}

impl Asset for SpriteBehaviorManifest {
    type Loader = RonLoader;

    const EXTENSION: &'static str = "ron";
}

lazy_static! {
    pub static ref SPRITE_BEHAVIOR_MANIFEST: AssetHandle<SpriteBehaviorManifest> =
        AssetExt::load_expect("common.sprite_behavior_manifest");
}

impl SpriteKind {
    pub fn solid_height(&self) -> Option<f32> {
        SPRITE_BEHAVIOR_MANIFEST
            .read()
            .solid_height
            .get(self)
            .copied()
    }

    pub fn is_collectible(&self) -> bool {
        SPRITE_BEHAVIOR_MANIFEST
            .read()
            .collectible_id
            .get(self)
            .is_some()
            && !self.mine_tool().is_some()
    }

    /// Is the sprite a container that will emit a mystery item?
    pub fn is_container(&self) -> bool {
        matches!(
            SPRITE_BEHAVIOR_MANIFEST.read().collectible_id.get(self),
            Some(LootSpec::LootTable(_)),
        )
    }

    pub fn mine_tool(&self) -> Option<ToolKind> {
        match self {
            SpriteKind::Velorite
            | SpriteKind::VeloriteFrag
            // Gems
            | SpriteKind::Amethyst
            | SpriteKind::Ruby
            | SpriteKind::Diamond
            | SpriteKind::Sapphire
            | SpriteKind::Emerald
            | SpriteKind::Topaz
            | SpriteKind::AmethystSmall
            | SpriteKind::TopazSmall
            | SpriteKind::DiamondSmall
            | SpriteKind::RubySmall
            | SpriteKind::EmeraldSmall
            | SpriteKind::SapphireSmall => Some(ToolKind::Pick),
            _ => None,
        }
    }

    pub fn has_ori(&self) -> bool {
        matches!(
            self,
            SpriteKind::Window1
                | SpriteKind::Window2
                | SpriteKind::Window3
                | SpriteKind::Window4
                | SpriteKind::Bed
                | SpriteKind::Bench
                | SpriteKind::ChairSingle
                | SpriteKind::ChairDouble
                | SpriteKind::CoatRack
                | SpriteKind::Crate
                | SpriteKind::DrawerLarge
                | SpriteKind::DrawerMedium
                | SpriteKind::DrawerSmall
                | SpriteKind::DungeonWallDecor
                | SpriteKind::HangingBasket
                | SpriteKind::HangingSign
                | SpriteKind::WallLamp
                | SpriteKind::WallLampSmall
                | SpriteKind::WallSconce
                | SpriteKind::Planter
                | SpriteKind::Shelf
                | SpriteKind::TableSide
                | SpriteKind::TableDining
                | SpriteKind::TableDouble
                | SpriteKind::WardrobeSingle
                | SpriteKind::WardrobeDouble
                | SpriteKind::Pot
                | SpriteKind::Chest
                | SpriteKind::DropGate
                | SpriteKind::DropGateBottom
                | SpriteKind::Door
                | SpriteKind::Beehive
                | SpriteKind::PotionMinor
                | SpriteKind::Bowl
                | SpriteKind::VialEmpty
                | SpriteKind::FireBowlGround
                | SpriteKind::Lantern
                | SpriteKind::CraftingBench
                | SpriteKind::Forge
                | SpriteKind::Cauldron
                | SpriteKind::Anvil
                | SpriteKind::CookingPot
        )
    }
}

impl fmt::Display for SpriteKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{:?}", self) }
}

lazy_static! {
    pub static ref SPRITE_KINDS: HashMap<String, SpriteKind> = SpriteKind::into_enum_iter()
        .map(|sk| (sk.to_string(), sk))
        .collect();
}

impl<'a> TryFrom<&'a str> for SpriteKind {
    type Error = ();

    fn try_from(s: &'a str) -> Result<Self, Self::Error> { SPRITE_KINDS.get(s).copied().ok_or(()) }
}
