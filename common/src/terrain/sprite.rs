use crate::{
    assets::{Asset, AssetExt, AssetHandle, RonLoader},
    comp::tool::ToolKind,
    lottery::LootSpec,
    make_case_elim,
    terrain::TerrainChunk,
    vol::{IntoVolIterator, ReadVol, RectRasterableVol, WriteVol},
};
use enum_iterator::IntoEnumIterator;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use num_derive::FromPrimitive;
use serde::{Deserialize, Serialize};
use std::{convert::TryFrom, fmt};
use vek::Vec3;

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
    pub growth_specs: HashMap<SpriteKind, GrowthSpec>,
}

impl Asset for SpriteBehaviorManifest {
    type Loader = RonLoader;

    const EXTENSION: &'static str = "ron";
}

lazy_static! {
    pub static ref SPRITE_BEHAVIOR_MANIFEST: AssetHandle<SpriteBehaviorManifest> =
        AssetExt::load_expect("common.sprite_behavior_manifest");
}

#[derive(Copy, Clone, Debug, Deserialize)]
pub struct GrowthSpec {
    pub start_growth: u8,
    pub max_growth: u8,
    pub days_per_growth: f32,
}

impl SpriteKind {
    pub fn solid_height(&self) -> Option<f32> {
        // Beware: the height *must* be <= `MAX_HEIGHT` or the collision system will not
        // properly detect it!
        Some(match self {
            SpriteKind::Tomato => 1.65,
            SpriteKind::LargeCactus => 2.5,
            SpriteKind::Scarecrow => 3.0,
            SpriteKind::Turnip => 0.36,
            SpriteKind::Pumpkin => 0.81,
            SpriteKind::Cabbage => 0.45,
            SpriteKind::Chest => 1.09,
            SpriteKind::StreetLamp => 2.65,
            SpriteKind::Carrot => 0.18,
            SpriteKind::Radish => 0.18,
            SpriteKind::FireBowlGround => 0.55,
            // TODO: Uncomment this when we have a way to open doors
            // SpriteKind::Door => 3.0,
            SpriteKind::Bed => 1.54,
            SpriteKind::Bench => 0.5,
            SpriteKind::ChairSingle => 0.5,
            SpriteKind::ChairDouble => 0.5,
            SpriteKind::CoatRack => 2.36,
            SpriteKind::Crate => 0.90,
            SpriteKind::DrawerSmall => 1.0,
            SpriteKind::DrawerMedium => 2.0,
            SpriteKind::DrawerLarge => 2.0,
            SpriteKind::DungeonWallDecor => 1.0,
            SpriteKind::Planter => 1.09,
            SpriteKind::TableSide => 1.27,
            SpriteKind::TableDining => 1.45,
            SpriteKind::TableDouble => 1.45,
            SpriteKind::WardrobeSingle => 3.0,
            SpriteKind::WardrobeDouble => 3.0,
            SpriteKind::Pot => 0.90,
            SpriteKind::Mud => 0.36,
            SpriteKind::ChestBuried => 0.91,
            SpriteKind::StonyCoral => 1.4,
            // TODO => Find suitable heights.
            SpriteKind::BarrelCactus => 1.0,
            SpriteKind::RoundCactus => 1.0,
            SpriteKind::ShortCactus => 1.0,
            SpriteKind::MedFlatCactus => 1.0,
            SpriteKind::ShortFlatCactus => 1.0,
            SpriteKind::Apple => 1.0,
            SpriteKind::Velorite => 1.0,
            SpriteKind::VeloriteFrag => 1.0,
            SpriteKind::Coconut => 1.0,
            SpriteKind::StreetLampTall => 1.0,
            SpriteKind::Window1 => 1.0,
            SpriteKind::Window2 => 1.0,
            SpriteKind::Window3 => 1.0,
            SpriteKind::Window4 => 1.0,
            SpriteKind::DropGate => 1.0,
            // TODO: Figure out if this should be solid or not.
            SpriteKind::Shelf => 1.0,
            SpriteKind::Lantern => 0.9,
            SpriteKind::CraftingBench => 1.18,
            SpriteKind::Forge => 2.7,
            SpriteKind::Cauldron => 1.27,
            SpriteKind::Anvil => 1.1,
            SpriteKind::CookingPot => 1.36,
            _ => return None,
        })
    }

    pub fn is_collectible(&self) -> bool {
        SPRITE_BEHAVIOR_MANIFEST
            .read()
            .collectible_id
            .get(self)
            .is_some()
            && self.mine_tool().is_none()
    }

    pub fn get_growth_spec(&self) -> Option<GrowthSpec> {
        SPRITE_BEHAVIOR_MANIFEST
            .read()
            .growth_specs
            .get(self)
            .copied()
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

/// Per-chunk index of plant data, for efficiently growing plants, and
/// persisting plant growth when the chunk is unloaded with low memory footprint
pub struct PlantGrowthData {
    data: HashMap<SpriteKind, PlantGrowthPerKind>,
}

struct PlantGrowthPerKind {
    // TODO: if we made use of the assumption that chunks are 32x32xk voxels, we could pack
    // positions into 10+log_2(k) bits instead of using the whole 12 bytes that a Vec3<i32> uses
    // (don't make this optimization unless needed, and not without double-checking with zesterer
    // about the best place to document what needs to be changed for handling larger chunk sizes).
    positions: Vec<Vec3<i32>>,
    growth_amounts: Vec<u8>,
    last_growth_tick: f32,
}

impl PlantGrowthPerKind {
    fn new(time: f32) -> Self {
        Self {
            positions: Vec::new(),
            growth_amounts: Vec::new(),
            last_growth_tick: time,
        }
    }

    fn calculate_growth(&mut self, sprite: SpriteKind, time: f32) {
        if let Some(growth_spec) = sprite.get_growth_spec() {
            let dt = self.last_growth_tick - time;
            let dt_days = dt / (60.0 * 60.0 * 24.0);
            if dt_days > growth_spec.days_per_growth {
                self.last_growth_tick += dt;
                for growth in self.growth_amounts.iter_mut() {
                    *growth += (dt_days / growth_spec.days_per_growth) as u8;
                    *growth = (*growth).min(growth_spec.max_growth);
                }
            }
        }
    }
}

impl PlantGrowthData {
    pub fn from_fresh_chunk(chunk: &TerrainChunk, time: f32) -> Self {
        let mut ret = Self {
            data: HashMap::new(),
        };
        for (pos, block) in chunk.vol_iter(
            Vec3::new(0, 0, chunk.get_min_z()),
            <TerrainChunk as RectRasterableVol>::RECT_SIZE
                .as_()
                .with_z(chunk.get_max_z()),
        ) {
            if let Some(sprite) = block.get_sprite() {
                if let Some(growth_spec) = sprite.get_growth_spec() {
                    let entry = ret
                        .data
                        .entry(sprite)
                        .or_insert_with(|| PlantGrowthPerKind::new(time));
                    entry.positions.push(pos);
                    entry.growth_amounts.push(growth_spec.start_growth);
                }
            }
        }
        ret
    }

    pub fn calculate_growth(&mut self, time: f32) {
        for (sprite, per_kind_growth) in self.data.iter_mut() {
            per_kind_growth.calculate_growth(*sprite, time);
        }
    }

    pub fn overwrite_plants(&self, chunk: &mut TerrainChunk) {
        for (sprite, per_kind_growth) in self.data.iter() {
            for (pos, growth) in per_kind_growth
                .positions
                .iter()
                .zip(per_kind_growth.growth_amounts.iter())
            {
                if let Ok(block) = chunk.get(*pos) {
                    let block = *block;
                    let _ = chunk.set(*pos, block.with_sprite(*sprite).with_sprite_growth(*growth));
                }
            }
        }
    }
}
