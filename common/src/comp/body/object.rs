use crate::{
    comp::{item::Reagent, Density, Mass},
    consts::{IRON_DENSITY, WATER_DENSITY},
    make_proj_elim,
};
use rand::{seq::SliceRandom, thread_rng};
use serde::{Deserialize, Serialize};
use vek::Vec3;

make_proj_elim!(
    body,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct Body {
        pub species: Species,
    }
);

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum Species {
    Arrow = 0,
    Bomb = 1,
    Scarecrow = 2,
    Cauldron = 3,
    ChestVines = 4,
    Chest = 5,
    ChestDark = 6,
    ChestDemon = 7,
    ChestGold = 8,
    ChestLight = 9,
    ChestOpen = 10,
    ChestSkull = 11,
    Pumpkin = 12,
    Pumpkin2 = 13,
    Pumpkin3 = 14,
    Pumpkin4 = 15,
    Pumpkin5 = 16,
    Campfire = 17,
    LanternGround = 18,
    LanternGroundOpen = 19,
    LanternStanding2 = 20,
    LanternStanding = 21,
    PotionBlue = 22,
    PotionGreen = 23,
    PotionRed = 24,
    Crate = 25,
    Tent = 26,
    WindowSpooky = 27,
    DoorSpooky = 28,
    Anvil = 29,
    Gravestone = 30,
    Gravestone2 = 31,
    Bench = 32,
    Chair = 33,
    Chair2 = 34,
    Chair3 = 35,
    Table = 36,
    Table2 = 37,
    Table3 = 38,
    Drawer = 39,
    BedBlue = 40,
    Carpet = 41,
    Bedroll = 42,
    CarpetHumanRound = 43,
    CarpetHumanSquare = 44,
    CarpetHumanSquare2 = 45,
    CarpetHumanSquircle = 46,
    Pouch = 47,
    CraftingBench = 48,
    BoltFire = 49,
    ArrowSnake = 50,
    CampfireLit = 51,
    BoltFireBig = 52,
    TrainingDummy = 53,
    FireworkBlue = 54,
    FireworkGreen = 55,
    FireworkPurple = 56,
    FireworkRed = 57,
    FireworkWhite = 58,
    FireworkYellow = 59,
    MultiArrow = 60,
    BoltNature = 61,
    MeatDrop = 62,
    Steak = 63,
    Crossbow = 64,
    ArrowTurret = 65,
    Coins = 66,
    GoldOre = 67,
    SilverOre = 68,
    ClayRocket = 69,
    HaniwaSentry = 70,
}

impl Body {
    pub fn random() -> Self {
        let mut rng = thread_rng();
        Body {
            species: *(&ALL_SPECIES).choose(&mut rng).unwrap(),
        }
    }

    pub fn for_firework(reagent: Reagent) -> Body {
        match reagent {
            Reagent::Blue => Body { species: Species::FireworkBlue },
            Reagent::Green => Body { species: Species::FireworkGreen },
            Reagent::Purple => Body { species: Species::FireworkPurple },
            Reagent::Red => Body { species: Species::FireworkRed },
            Reagent::White => Body { species: Species::FireworkWhite },
            Reagent::Yellow => Body { species: Species::FireworkYellow },
        }
    }
}

/// Data representing per-species generic data.
///
/// NOTE: Deliberately don't (yet?) implement serialize.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AllSpecies<SpeciesMeta> {
    pub arrow: SpeciesMeta,
    pub bomb: SpeciesMeta,
    pub scarecrow: SpeciesMeta,
    pub cauldron: SpeciesMeta,
    pub chest_vines: SpeciesMeta,
    pub chest: SpeciesMeta,
    pub chest_dark: SpeciesMeta,
    pub chest_demon: SpeciesMeta,
    pub chest_gold: SpeciesMeta,
    pub chest_light: SpeciesMeta,
    pub chest_open: SpeciesMeta,
    pub chest_skull: SpeciesMeta,
    pub pumpkin: SpeciesMeta,
    pub pumpkin_2: SpeciesMeta,
    pub pumpkin_3: SpeciesMeta,
    pub pumpkin_4: SpeciesMeta,
    pub pumpkin_5: SpeciesMeta,
    pub campfire: SpeciesMeta,
    pub campfire_lit: SpeciesMeta,
    pub lantern_ground: SpeciesMeta,
    pub lantern_ground_open: SpeciesMeta,
    pub lantern_standing: SpeciesMeta,
    pub lantern_standing_2: SpeciesMeta,
    pub potion_red: SpeciesMeta,
    pub potion_blue: SpeciesMeta,
    pub potion_green: SpeciesMeta,
    pub crate_0: SpeciesMeta,
    pub tent: SpeciesMeta,
    pub window_spooky: SpeciesMeta,
    pub door_spooky: SpeciesMeta,
    pub anvil: SpeciesMeta,
    pub gravestone: SpeciesMeta,
    pub gravestone_2: SpeciesMeta,
    pub bench: SpeciesMeta,
    pub chair: SpeciesMeta,
    pub chair_2: SpeciesMeta,
    pub chair_3: SpeciesMeta,
    pub table: SpeciesMeta,
    pub table_2: SpeciesMeta,
    pub table_3: SpeciesMeta,
    pub drawer: SpeciesMeta,
    pub bed_blue: SpeciesMeta,
    pub carpet: SpeciesMeta,
    pub bedroll: SpeciesMeta,
    pub carpet_human_round: SpeciesMeta,
    pub carpet_human_square: SpeciesMeta,
    pub carpet_human_square_2: SpeciesMeta,
    pub carpet_human_squircle: SpeciesMeta,
    pub pouch: SpeciesMeta,
    pub crafting_bench: SpeciesMeta,
    pub bolt_fire: SpeciesMeta,
    pub bolt_fire_big: SpeciesMeta,
    pub arrow_snake: SpeciesMeta,
    pub training_dummy: SpeciesMeta,
    pub firework_blue: SpeciesMeta,
    pub firework_green: SpeciesMeta,
    pub firework_purple: SpeciesMeta,
    pub firework_red: SpeciesMeta,
    pub firework_white: SpeciesMeta,
    pub firework_yellow: SpeciesMeta,
    pub multi_arrow: SpeciesMeta,
    pub bolt_nature: SpeciesMeta,
    pub meat_drop: SpeciesMeta,
    pub steak: SpeciesMeta,
    pub crossbow: SpeciesMeta,
    pub arrow_turret: SpeciesMeta,
    pub coins: SpeciesMeta,
    pub silver_ore: SpeciesMeta,
    pub gold_ore: SpeciesMeta,
    pub clay_rocket: SpeciesMeta,
    pub haniwa_sentry: SpeciesMeta,
}

impl<'a, SpeciesMeta> core::ops::Index<&'a Species> for AllSpecies<SpeciesMeta> {
    type Output = SpeciesMeta;

    #[inline]
    fn index(&self, &index: &'a Species) -> &Self::Output {
        match index {
            Species::Arrow => &self.arrow,
            Species::Bomb => &self.bomb,
            Species::Scarecrow => &self.scarecrow,
            Species::Cauldron => &self.cauldron,
            Species::ChestVines => &self.chest_vines,
            Species::Chest => &self.chest,
            Species::ChestDark => &self.chest_dark,
            Species::ChestDemon => &self.chest_demon,
            Species::ChestGold => &self.chest_gold,
            Species::ChestLight => &self.chest_light,
            Species::ChestOpen => &self.chest_open,
            Species::ChestSkull => &self.chest_skull,
            Species::Pumpkin => &self.pumpkin,
            Species::Pumpkin2 => &self.pumpkin_2,
            Species::Pumpkin3 => &self.pumpkin_3,
            Species::Pumpkin4 => &self.pumpkin_4,
            Species::Pumpkin5 => &self.pumpkin_5,
            Species::Campfire => &self.campfire,
            Species::CampfireLit => &self.campfire_lit,
            Species::LanternGround => &self.lantern_ground,
            Species::LanternGroundOpen => &self.lantern_ground_open,
            Species::LanternStanding => &self.lantern_standing,
            Species::LanternStanding2 => &self.lantern_standing_2,
            Species::PotionRed => &self.potion_red,
            Species::PotionBlue => &self.potion_blue,
            Species::PotionGreen => &self.potion_green,
            Species::Crate => &self.crate_0,
            Species::Tent => &self.tent,
            Species::WindowSpooky => &self.window_spooky,
            Species::DoorSpooky => &self.door_spooky,
            Species::Anvil => &self.anvil,
            Species::Gravestone => &self.gravestone,
            Species::Gravestone2 => &self.gravestone_2,
            Species::Bench => &self.bench,
            Species::Chair => &self.chair,
            Species::Chair2 => &self.chair_2,
            Species::Chair3 => &self.chair_3,
            Species::Table => &self.table,
            Species::Table2 => &self.table_2,
            Species::Table3 => &self.table_3,
            Species::Drawer => &self.drawer,
            Species::BedBlue => &self.bed_blue,
            Species::Carpet => &self.carpet,
            Species::Bedroll => &self.bedroll,
            Species::CarpetHumanRound => &self.carpet_human_round,
            Species::CarpetHumanSquare => &self.carpet_human_square,
            Species::CarpetHumanSquare2 => &self.carpet_human_square_2,
            Species::CarpetHumanSquircle => &self.carpet_human_squircle,
            Species::Pouch => &self.pouch,
            Species::CraftingBench => &self.crafting_bench,
            Species::BoltFire => &self.bolt_fire,
            Species::BoltFireBig => &self.bolt_fire_big,
            Species::ArrowSnake => &self.arrow_snake,
            Species::TrainingDummy => &self.training_dummy,
            Species::FireworkBlue => &self.firework_blue,
            Species::FireworkGreen => &self.firework_green,
            Species::FireworkPurple => &self.firework_purple,
            Species::FireworkRed => &self.firework_red,
            Species::FireworkWhite => &self.firework_white,
            Species::FireworkYellow => &self.firework_yellow,
            Species::MultiArrow => &self.multi_arrow,
            Species::BoltNature => &self.bolt_nature,
            Species::MeatDrop => &self.meat_drop,
            Species::Steak => &self.steak,
            Species::Crossbow => &self.crossbow,
            Species::ArrowTurret => &self.arrow_turret,
            Species::Coins => &self.coins,
            Species::SilverOre => &self.silver_ore,
            Species::GoldOre => &self.gold_ore,
            Species::ClayRocket => &self.clay_rocket,
            Species::HaniwaSentry => &self.haniwa_sentry,
        }
    }
}

pub const ALL_SPECIES: [Species; 71] = [
    Species::Arrow,
    Species::Bomb,
    Species::Scarecrow,
    Species::Cauldron,
    Species::ChestVines,
    Species::Chest,
    Species::ChestDark,
    Species::ChestDemon,
    Species::ChestGold,
    Species::ChestLight,
    Species::ChestOpen,
    Species::ChestSkull,
    Species::Pumpkin,
    Species::Pumpkin2,
    Species::Pumpkin3,
    Species::Pumpkin4,
    Species::Pumpkin5,
    Species::Campfire,
    Species::CampfireLit,
    Species::LanternGround,
    Species::LanternGroundOpen,
    Species::LanternStanding,
    Species::LanternStanding2,
    Species::PotionRed,
    Species::PotionBlue,
    Species::PotionGreen,
    Species::Crate,
    Species::Tent,
    Species::WindowSpooky,
    Species::DoorSpooky,
    Species::Anvil,
    Species::Gravestone,
    Species::Gravestone2,
    Species::Bench,
    Species::Chair,
    Species::Chair2,
    Species::Chair3,
    Species::Table,
    Species::Table2,
    Species::Table3,
    Species::Drawer,
    Species::BedBlue,
    Species::Carpet,
    Species::Bedroll,
    Species::CarpetHumanRound,
    Species::CarpetHumanSquare,
    Species::CarpetHumanSquare2,
    Species::CarpetHumanSquircle,
    Species::Pouch,
    Species::CraftingBench,
    Species::BoltFire,
    Species::BoltFireBig,
    Species::ArrowSnake,
    Species::TrainingDummy,
    Species::FireworkBlue,
    Species::FireworkGreen,
    Species::FireworkPurple,
    Species::FireworkRed,
    Species::FireworkWhite,
    Species::FireworkYellow,
    Species::MultiArrow,
    Species::BoltNature,
    Species::MeatDrop,
    Species::Steak,
    Species::Crossbow,
    Species::ArrowTurret,
    Species::Coins,
    Species::SilverOre,
    Species::GoldOre,
    Species::ClayRocket,
    Species::HaniwaSentry,
];

impl<'a, SpeciesMeta: 'a> IntoIterator for &'a AllSpecies<SpeciesMeta> {
    type IntoIter = std::iter::Copied<std::slice::Iter<'static, Self::Item>>;
    type Item = Species;

    fn into_iter(self) -> Self::IntoIter { ALL_SPECIES.iter().copied() }
}

impl From<Body> for super::Body {
    fn from(body: Body) -> Self { super::Body::Object(body) }
}

impl Species {
    pub fn for_firework(reagent: Reagent) -> Species {
        match reagent {
            Reagent::Blue => Species::FireworkBlue,
            Reagent::Green => Species::FireworkGreen,
            Reagent::Purple => Species::FireworkPurple,
            Reagent::Red => Species::FireworkRed,
            Reagent::White => Species::FireworkWhite,
            Reagent::Yellow => Species::FireworkYellow,
        }
    }

    pub fn to_string(&self) -> &str {
        match self {
            Species::Arrow => "arrow",
            Species::Bomb => "bomb",
            Species::Scarecrow => "scarecrow",
            Species::Cauldron => "cauldron",
            Species::ChestVines => "chest_vines",
            Species::Chest => "chest",
            Species::ChestDark => "chest_dark",
            Species::ChestDemon => "chest_demon",
            Species::ChestGold => "chest_gold",
            Species::ChestLight => "chest_light",
            Species::ChestOpen => "chest_open",
            Species::ChestSkull => "chest_skull",
            Species::Pumpkin => "pumpkin",
            Species::Pumpkin2 => "pumpkin_2",
            Species::Pumpkin3 => "pumpkin_3",
            Species::Pumpkin4 => "pumpkin_4",
            Species::Pumpkin5 => "pumpkin_5",
            Species::Campfire => "campfire",
            Species::CampfireLit => "campfire_lit",
            Species::LanternGround => "lantern_ground",
            Species::LanternGroundOpen => "lantern_ground_open",
            Species::LanternStanding => "lantern_standing",
            Species::LanternStanding2 => "lantern_standing_2",
            Species::PotionRed => "potion_red",
            Species::PotionBlue => "potion_blue",
            Species::PotionGreen => "potion_green",
            Species::Crate => "crate_0",
            Species::Tent => "tent",
            Species::WindowSpooky => "window_spooky",
            Species::DoorSpooky => "door_spooky",
            Species::Anvil => "anvil",
            Species::Gravestone => "gravestone",
            Species::Gravestone2 => "gravestone_2",
            Species::Bench => "bench",
            Species::Chair => "chair",
            Species::Chair2 => "chair_2",
            Species::Chair3 => "chair_3",
            Species::Table => "table",
            Species::Table2 => "table_2",
            Species::Table3 => "table_3",
            Species::Drawer => "drawer",
            Species::BedBlue => "bed_blue",
            Species::Carpet => "carpet",
            Species::Bedroll => "bedroll",
            Species::CarpetHumanRound => "carpet_human_round",
            Species::CarpetHumanSquare => "carpet_human_square",
            Species::CarpetHumanSquare2 => "carpet_human_square_2",
            Species::CarpetHumanSquircle => "carpet_human_squircle",
            Species::Pouch => "pouch",
            Species::CraftingBench => "crafting_bench",
            Species::BoltFire => "bolt_fire",
            Species::BoltFireBig => "bolt_fire_big",
            Species::ArrowSnake => "arrow_snake",
            Species::TrainingDummy => "training_dummy",
            Species::FireworkBlue => "firework_blue",
            Species::FireworkGreen => "firework_green",
            Species::FireworkPurple => "firework_purple",
            Species::FireworkRed => "firework_red",
            Species::FireworkWhite => "firework_white",
            Species::FireworkYellow => "firework_yellow",
            Species::MultiArrow => "multi_arrow",
            Species::BoltNature => "bolt_nature",
            Species::MeatDrop => "meat_drop",
            Species::Steak => "steak",
            Species::Crossbow => "crossbow",
            Species::ArrowTurret => "arrow_turret",
            Species::Coins => "coins",
            Species::SilverOre => "silver_ore",
            Species::GoldOre => "gold_ore",
            Species::ClayRocket => "clay_rocket",
            Species::HaniwaSentry => "haniwa_sentry",
        }
    }

    //pub fn density(&self) -> Density {
    //    let density = match self {
    //        Body::Anvil | Body::Cauldron => IRON_DENSITY,
    //        Body::Arrow | Body::ArrowSnake | Body::ArrowTurret | Body::MultiArrow => 500.0,
    //        Body::Bomb => 2000.0, // I have no idea what it's supposed to be
    //        Body::Crate => 300.0, // let's say it's a lot of wood and maybe some contents
    //        Body::Scarecrow => 900.0,
    //        Body::TrainingDummy => 2000.0,
    //        // let them sink
    //        _ => 1.1 * WATER_DENSITY,
    //    };

    //    Density(density)
    //}

    //pub fn mass(&self) -> Mass {
    //    let m = match self {
    //        // I think MultiArrow is one of several arrows, not several arrows combined?
    //        Body::Anvil => 100.0,
    //        Body::Arrow | Body::ArrowSnake | Body::ArrowTurret | Body::MultiArrow => 0.003,
    //        Body::BedBlue => 50.0,
    //        Body::Bedroll => 3.0,
    //        Body::Bench => 100.0,
    //        Body::BoltFire | Body::BoltFireBig | Body::BoltNature => 1.0,
    //        Body::Bomb => {
    //            0.5 * IRON_DENSITY * std::f32::consts::PI / 6.0 * self.dimensions().x.powi(3)
    //        },
    //        Body::Campfire | Body::CampfireLit => 300.0,
    //        Body::Carpet
    //        | Body::CarpetHumanRound
    //        | Body::CarpetHumanSquare
    //        | Body::CarpetHumanSquare2
    //        | Body::CarpetHumanSquircle => 10.0,
    //        Body::Cauldron => 5.0,
    //        Body::Chair | Body::Chair2 | Body::Chair3 => 10.0,
    //        Body::Chest
    //        | Body::ChestDark
    //        | Body::ChestDemon
    //        | Body::ChestGold
    //        | Body::ChestLight
    //        | Body::ChestOpen
    //        | Body::ChestSkull
    //        | Body::ChestVines => 100.0,
    //        Body::Coins => 1.0,
    //        Body::CraftingBench => 100.0,
    //        Body::Crate => 50.0,
    //        Body::Crossbow => 200.0,
    //        Body::DoorSpooky => 20.0,
    //        Body::Drawer => 50.0,
    //        Body::FireworkBlue
    //        | Body::FireworkGreen
    //        | Body::FireworkPurple
    //        | Body::FireworkRed
    //        | Body::FireworkWhite
    //        | Body::FireworkYellow => 1.0,
    //        Body::Gravestone => 100.0,
    //        Body::Gravestone2 => 100.0,
    //        Body::LanternGround
    //        | Body::LanternGroundOpen
    //        | Body::LanternStanding
    //        | Body::LanternStanding2 => 3.0,
    //        Body::MeatDrop => 5.0,
    //        Body::PotionBlue | Body::PotionGreen | Body::PotionRed => 5.0,
    //        Body::Pouch => 1.0,
    //        Body::Pumpkin | Body::Pumpkin2 | Body::Pumpkin3 | Body::Pumpkin4 | Body::Pumpkin5 => {
    //            10.0
    //        },
    //        Body::Scarecrow => 50.0,
    //        Body::Steak => 2.0,
    //        Body::Table | Body::Table2 | Body::Table3 => 50.0,
    //        Body::Tent => 50.0,
    //        Body::TrainingDummy => 60.0,
    //        Body::WindowSpooky => 10.0,
    //        Body::SilverOre => 1000.0,
    //        Body::GoldOre => 1000.0,
    //        Body::ClayRocket => 50.0,
    //        Body::HaniwaSentry => 300.0,
    //    };

    //    Mass(m)
    //}

    //pub fn dimensions(&self) -> Vec3<f32> {
    //    match self {
    //        Body::Arrow | Body::ArrowSnake | Body::MultiArrow | Body::ArrowTurret => {
    //            Vec3::new(0.01, 0.8, 0.01)
    //        },
    //        Body::BoltFire => Vec3::new(0.1, 0.1, 0.1),
    //        Body::Crossbow => Vec3::new(3.0, 3.0, 1.5),
    //        Body::HaniwaSentry => Vec3::new(0.8, 0.8, 1.4),
    //        _ => Vec3::broadcast(0.2),
    //    }
    //}
}
