use crate::{
    assets::{self, AssetExt},
    comp::{
        arthropod, biped_large, biped_small, bird_large, golem,
        inventory::{
            loadout::Loadout,
            slot::{ArmorSlot, EquipSlot},
        },
        item::Item,
        object, quadruped_low, quadruped_medium, theropod, Body,
    },
    trade::SiteInformation,
};
use rand::{self, distributions::WeightedError, seq::SliceRandom, Rng};
use serde::{Deserialize, Serialize};
use strum::EnumIter;
use tracing::warn;

type Weight = u8;

#[derive(Debug)]
pub enum SpecError {
    LoadoutAssetError(assets::Error),
    ItemAssetError(assets::Error),
    ItemChoiceError(WeightedError),
    BaseChoiceError(WeightedError),
}

#[derive(Debug)]
#[cfg(test)]
pub enum ValidationError {
    ItemAssetError(assets::Error),
    LoadoutAssetError(assets::Error),
    Loop(Vec<String>),
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum ItemSpec {
    Item(String),
    Choice(Vec<(Weight, Option<ItemSpec>)>),
}

impl ItemSpec {
    fn try_to_item(&self, rng: &mut impl Rng) -> Result<Option<Item>, SpecError> {
        match self {
            ItemSpec::Item(item_asset) => {
                let item = Item::new_from_asset(item_asset).map_err(SpecError::ItemAssetError)?;
                Ok(Some(item))
            },
            ItemSpec::Choice(items) => {
                let (_, item_spec) = items
                    .choose_weighted(rng, |(weight, _)| *weight)
                    .map_err(SpecError::ItemChoiceError)?;

                let item = if let Some(item_spec) = item_spec {
                    item_spec.try_to_item(rng)?
                } else {
                    None
                };
                Ok(item)
            },
        }
    }

    // Check if ItemSpec is valid and can be turned into Item
    #[cfg(test)]
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            ItemSpec::Item(item_asset) => Item::new_from_asset(item_asset)
                .map(std::mem::drop)
                .map_err(ValidationError::ItemAssetError),
            ItemSpec::Choice(choices) => {
                // TODO: check for sanity of weigts?
                for (_weight, choice) in choices {
                    if let Some(item) = choice {
                        item.validate()?;
                    }
                }
                Ok(())
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
enum Hands {
    /// Allows to specify one pair
    InHands((Option<ItemSpec>, Option<ItemSpec>)),
    /// Allows specify range of choices
    Choice(Vec<(Weight, Hands)>),
}

impl Hands {
    fn try_to_pair(&self, rng: &mut impl Rng) -> Result<(Option<Item>, Option<Item>), SpecError> {
        match self {
            Hands::InHands((mainhand, offhand)) => {
                let mut from_spec = |i: &ItemSpec| i.try_to_item(rng);

                let mainhand = mainhand
                    .as_ref()
                    .map(|i| from_spec(i))
                    .transpose()?
                    .flatten();
                let offhand = offhand
                    .as_ref()
                    .map(|i| from_spec(i))
                    .transpose()?
                    .flatten();
                Ok((mainhand, offhand))
            },
            Hands::Choice(pairs) => {
                let (_, pair_spec) = pairs
                    .choose_weighted(rng, |(weight, _)| *weight)
                    .map_err(SpecError::ItemChoiceError)?;

                pair_spec.try_to_pair(rng)
            },
        }
    }

    // Check if items in Hand are valid and can be turned into Item
    #[cfg(test)]
    fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::InHands((left, right)) => {
                if let Some(hand) = left {
                    hand.validate()?;
                }
                if let Some(hand) = right {
                    hand.validate()?;
                }
                Ok(())
            },
            Self::Choice(choices) => {
                // TODO: check for sanity of weights?
                for (_weight, choice) in choices {
                    choice.validate()?;
                }
                Ok(())
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
enum Base {
    Asset(String),
    /// NOTE: If you have the same item in multiple configs,
    /// *first* one will have the priority
    Combine(Vec<Base>),
    Choice(Vec<(Weight, Base)>),
}

impl Base {
    // Turns Base to LoadoutSpec
    //
    // NOTE: Don't expect it to be fully evaluated, but in some cases
    // it may be so.
    fn to_spec(&self, rng: &mut impl Rng) -> Result<LoadoutSpec, SpecError> {
        match self {
            Base::Asset(asset_specifier) => {
                LoadoutSpec::load_cloned(asset_specifier).map_err(SpecError::LoadoutAssetError)
            },
            Base::Combine(bases) => {
                let bases = bases.iter().map(|b| b.to_spec(rng)?.eval(rng));
                // Get first base of combined
                let mut current = LoadoutSpec::default();
                for base in bases {
                    current = current.merge(base?);
                }

                Ok(current)
            },
            Base::Choice(choice) => {
                let (_, base) = choice
                    .choose_weighted(rng, |(weight, _)| *weight)
                    .map_err(SpecError::BaseChoiceError)?;

                base.to_spec(rng)
            },
        }
    }
}

// TODO: remove clone
/// Core struct of loadout asset.
///
/// If you want programing API of loadout creation,
/// use `LoadoutBuilder` instead.
///
/// For examples of assets, see `assets/test/ladout/ok` folder.
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(deny_unknown_fields)]
pub struct LoadoutSpec {
    // Meta fields
    inherit: Option<Base>,
    // Armor
    head: Option<ItemSpec>,
    neck: Option<ItemSpec>,
    shoulders: Option<ItemSpec>,
    chest: Option<ItemSpec>,
    gloves: Option<ItemSpec>,
    ring1: Option<ItemSpec>,
    ring2: Option<ItemSpec>,
    back: Option<ItemSpec>,
    belt: Option<ItemSpec>,
    legs: Option<ItemSpec>,
    feet: Option<ItemSpec>,
    tabard: Option<ItemSpec>,
    bag1: Option<ItemSpec>,
    bag2: Option<ItemSpec>,
    bag3: Option<ItemSpec>,
    bag4: Option<ItemSpec>,
    lantern: Option<ItemSpec>,
    glider: Option<ItemSpec>,
    // Weapons
    active_hands: Option<Hands>,
    inactive_hands: Option<Hands>,
}

impl LoadoutSpec {
    /// Merges `self` with `base`.
    /// If some field exists in `self` it will be used,
    /// if no, it will be taken from `base`.
    ///
    /// NOTE: it uses only inheritance chain from `base`
    /// without evaluating it.
    /// Inheritance chain from `self` is discarded.
    ///
    /// # Examples
    /// 1)
    /// You have your asset, let's call it "a". In this asset, you have
    /// inheritance from "b". In asset "b" you inherit from "c".
    ///
    /// If you load your "a" into LoadoutSpec A, and "b" into LoadoutSpec B,
    /// and then merge A into B, you will get new LoadoutSpec that will inherit
    /// from "c".
    ///
    /// 2)
    /// You have two assets, let's call them "a" and "b".
    /// "a" inherits from "n",
    /// "b" inherits from "m".
    ///
    /// If you load "a" into A, "b" into B and then will try to merge them
    /// you will get new LoadoutSpec that will inherit from "m".
    /// It's error, because chain to "n" is lost!!!
    ///
    /// Correct way to do this will be first evaluating at least "a" and then
    /// merge this new LoadoutSpec with "b".
    fn merge(self, base: Self) -> Self {
        Self {
            inherit: base.inherit,
            head: self.head.or(base.head),
            neck: self.neck.or(base.neck),
            shoulders: self.shoulders.or(base.shoulders),
            chest: self.chest.or(base.chest),
            gloves: self.gloves.or(base.gloves),
            ring1: self.ring1.or(base.ring1),
            ring2: self.ring2.or(base.ring2),
            back: self.back.or(base.back),
            belt: self.belt.or(base.belt),
            legs: self.legs.or(base.legs),
            feet: self.feet.or(base.feet),
            tabard: self.tabard.or(base.tabard),
            bag1: self.bag1.or(base.bag1),
            bag2: self.bag2.or(base.bag2),
            bag3: self.bag3.or(base.bag3),
            bag4: self.bag4.or(base.bag4),
            lantern: self.lantern.or(base.lantern),
            glider: self.glider.or(base.glider),
            active_hands: self.active_hands.or(base.active_hands),
            inactive_hands: self.inactive_hands.or(base.inactive_hands),
        }
    }

    /// Recursively evaluate all inheritance chain.
    /// For example with following structure.
    ///
    /// ```text
    /// A
    /// inherit: B,
    /// gloves: a,
    ///
    /// B
    /// inherit: C,
    /// ring1: b,
    ///
    /// C
    /// inherit: None,
    /// ring2: c
    /// ```
    ///
    /// result will be
    ///
    /// ```text
    /// inherit: None,
    /// gloves: a,
    /// ring1: b,
    /// ring2: c,
    /// ```
    fn eval(self, rng: &mut impl Rng) -> Result<Self, SpecError> {
        // Iherit loadout if needed
        if let Some(ref base) = self.inherit {
            let base = base.to_spec(rng)?.eval(rng);
            Ok(self.merge(base?))
        } else {
            Ok(self)
        }
    }

    // Validate loadout spec and check that it can be turned into real loadout.
    // Checks for possible loops too.
    //
    // NOTE: It is stricter than needed, it will check all items
    // even if they are overwritten.
    // We can avoid these redundant checks by building set of all possible
    // specs and then check them.
    // This algorithm will be much more complex and require more memory,
    // because if we Combine multiple Choice-s we will need to create
    // cartesian product of specs.
    //
    // Also we probably don't want garbage entries anyway, even if they are
    // unused.
    #[cfg(test)]
    pub fn validate(&self, history: Vec<String>) -> Result<(), ValidationError> {
        // Helper function to traverse base.
        //
        // Important invariant to hold.
        // Each time it finds new asset it appends it to history
        // and calls spec.validate()
        fn validate_base(base: &Base, mut history: Vec<String>) -> Result<(), ValidationError> {
            match base {
                Base::Asset(asset) => {
                    // read the spec
                    let based = LoadoutSpec::load_cloned(asset)
                        .map_err(ValidationError::LoadoutAssetError)?;

                    // expand history
                    history.push(asset.to_owned());

                    // validate our spec
                    based.validate(history)
                },
                Base::Combine(bases) => {
                    for base in bases {
                        validate_base(base, history.clone())?;
                    }
                    Ok(())
                },
                Base::Choice(choices) => {
                    // TODO: check for sanity of weights?
                    for (_weight, base) in choices {
                        validate_base(base, history.clone())?;
                    }
                    Ok(())
                },
            }
        }

        // Scarry logic
        //
        // We check for duplicates on each append, and because we append on each
        // call we can be sure we don't have any duplicates unless it's a last
        // element.
        // So we can check for duplicates by comparing
        // all elements with last element.
        // And if we found duplicate in our history we found a loop.
        if let Some((last, tail)) = history.split_last() {
            for asset in tail {
                if last == asset {
                    return Err(ValidationError::Loop(history));
                }
            }
        }

        match &self.inherit {
            Some(base) => validate_base(base, history)?,
            None => (),
        }

        self.validate_entries()
    }

    // Validate entries in loadout spec.
    //
    // NOTE: this only check for items, we assume that base
    // is validated separately.
    //
    // TODO: add some intelligent checks,
    // e.g. that `head` key corresponds to Item with ItemKind::Head(_)
    #[cfg(test)]
    fn validate_entries(&self) -> Result<(), ValidationError> {
        // Armor
        if let Some(item) = &self.head {
            item.validate()?;
        }
        if let Some(item) = &self.neck {
            item.validate()?;
        }
        if let Some(item) = &self.shoulders {
            item.validate()?;
        }
        if let Some(item) = &self.chest {
            item.validate()?;
        }
        if let Some(item) = &self.gloves {
            item.validate()?;
        }
        if let Some(item) = &self.ring1 {
            item.validate()?;
        }
        if let Some(item) = &self.ring2 {
            item.validate()?;
        }
        if let Some(item) = &self.back {
            item.validate()?;
        }
        if let Some(item) = &self.belt {
            item.validate()?;
        }
        if let Some(item) = &self.legs {
            item.validate()?;
        }
        if let Some(item) = &self.feet {
            item.validate()?;
        }
        if let Some(item) = &self.tabard {
            item.validate()?;
        }
        // Misc
        if let Some(item) = &self.bag1 {
            item.validate()?;
        }
        if let Some(item) = &self.bag2 {
            item.validate()?;
        }
        if let Some(item) = &self.bag3 {
            item.validate()?;
        }
        if let Some(item) = &self.bag4 {
            item.validate()?;
        }
        if let Some(item) = &self.lantern {
            item.validate()?;
        }
        if let Some(item) = &self.glider {
            item.validate()?;
        }
        // Hands, tools and weapons
        if let Some(hands) = &self.active_hands {
            hands.validate()?;
        }
        if let Some(hands) = &self.inactive_hands {
            hands.validate()?;
        }

        Ok(())
    }
}

impl assets::Asset for LoadoutSpec {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

#[must_use]
pub fn make_potion_bag(quantity: u32) -> Item {
    let mut bag = Item::new_from_asset_expect("common.items.armor.misc.bag.tiny_leather_pouch");
    if let Some(i) = bag.slots_mut().iter_mut().next() {
        let mut potions = Item::new_from_asset_expect("common.items.consumable.potion_big");
        if let Err(e) = potions.set_amount(quantity) {
            warn!("Failed to set potion quantity: {:?}", e);
        }
        *i = Some(potions);
    }
    bag
}

#[must_use]
pub fn make_food_bag(quantity: u32) -> Item {
    let mut bag = Item::new_from_asset_expect("common.items.armor.misc.bag.tiny_leather_pouch");
    if let Some(i) = bag.slots_mut().iter_mut().next() {
        let mut food = Item::new_from_asset_expect("common.items.food.apple_stick");
        if let Err(e) = food.set_amount(quantity) {
            warn!("Failed to set food quantity: {:?}", e);
        }
        *i = Some(food);
    }
    bag
}

#[must_use]
// We have many species so this function is long
// Also we are using default tools for un-specified species so
// it's fine to have wildcards
#[allow(clippy::too_many_lines, clippy::match_wildcard_for_single_variants)]
fn default_main_tool(body: &Body) -> Item {
    let maybe_tool = match body {
        Body::Golem(golem) => match golem.species {
            golem::Species::StoneGolem => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.stone_golems_fist",
            )),
            golem::Species::ClayGolem => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.clay_golem_fist",
            )),
            golem::Species::WoodGolem => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.wood_golem_fist",
            )),
            _ => None,
        },
        Body::QuadrupedMedium(quadruped_medium) => match quadruped_medium.species {
            quadruped_medium::Species::Wolf | quadruped_medium::Species::Grolgar => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.unique.quadmedquick"),
            ),
            quadruped_medium::Species::Donkey
            | quadruped_medium::Species::Horse
            | quadruped_medium::Species::Zebra
            | quadruped_medium::Species::Kelpie
            | quadruped_medium::Species::Hirdrasil
            | quadruped_medium::Species::Deer
            | quadruped_medium::Species::Antelope
            | quadruped_medium::Species::Llama
            | quadruped_medium::Species::Alpaca => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadmedhoof",
            )),
            quadruped_medium::Species::Saber
            | quadruped_medium::Species::Bonerattler
            | quadruped_medium::Species::Darkhound
            | quadruped_medium::Species::Lion
            | quadruped_medium::Species::Snowleopard => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadmedjump",
            )),
            quadruped_medium::Species::Tuskram
            | quadruped_medium::Species::Roshwalr
            | quadruped_medium::Species::Moose
            | quadruped_medium::Species::Dreadhorn
            | quadruped_medium::Species::Mammoth
            | quadruped_medium::Species::Ngoubou => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadmedcharge",
            )),
            quadruped_medium::Species::Highland
            | quadruped_medium::Species::Cattle
            | quadruped_medium::Species::Yak => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadmedbasicgentle",
            )),
            _ => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadmedbasic",
            )),
        },
        Body::QuadrupedLow(quadruped_low) => match quadruped_low.species {
            quadruped_low::Species::Maneater => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.maneater",
            )),
            quadruped_low::Species::Asp => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.asp",
            )),
            quadruped_low::Species::Crocodile
            | quadruped_low::Species::SeaCrocodile
            | quadruped_low::Species::Alligator
            | quadruped_low::Species::Salamander => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadlowtail",
            )),
            quadruped_low::Species::Monitor | quadruped_low::Species::Pangolin => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.unique.quadlowquick"),
            ),
            quadruped_low::Species::Lavadrake => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadlowbreathe",
            )),
            quadruped_low::Species::Deadwood => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.deadwood",
            )),
            quadruped_low::Species::Basilisk => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.basilisk",
            )),
            _ => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.quadlowbasic",
            )),
        },
        Body::QuadrupedSmall(_) => Some(Item::new_from_asset_expect(
            "common.items.npc_weapons.unique.quadsmallbasic",
        )),
        Body::Theropod(theropod) => match theropod.species {
            theropod::Species::Sandraptor
            | theropod::Species::Snowraptor
            | theropod::Species::Woodraptor => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.theropodbird",
            )),
            theropod::Species::Yale => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.theropodcharge",
            )),
            _ => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.theropodbasic",
            )),
        },
        Body::Arthropod(arthropod) => match arthropod.species {
            arthropod::Species::Hornbeetle
            | arthropod::Species::Stagbeetle
            | arthropod::Species::Antlion => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.arthropodcharge",
            )),
            arthropod::Species::Cavespider | arthropod::Species::Blackwidow => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.unique.arthropodranged"),
            ),
            arthropod::Species::Weevil | arthropod::Species::Tarantula => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.unique.arthropodleap"),
            ),
            _ => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.arthropodbasic",
            )),
        },
        Body::BipedLarge(biped_large) => match (biped_large.species, biped_large.body_type) {
            (biped_large::Species::Occultsaurok, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.staff.saurok_staff",
            )),
            (biped_large::Species::Mightysaurok, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.sword.saurok_sword",
            )),
            (biped_large::Species::Slysaurok, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.bow.saurok_bow",
            )),
            (biped_large::Species::Ogre, biped_large::BodyType::Male) => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.hammer.ogre_hammer"),
            ),
            (biped_large::Species::Ogre, biped_large::BodyType::Female) => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.staff.ogre_staff"),
            ),
            (
                biped_large::Species::Mountaintroll
                | biped_large::Species::Swamptroll
                | biped_large::Species::Cavetroll,
                _,
            ) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.hammer.troll_hammer",
            )),
            (biped_large::Species::Wendigo, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.wendigo_magic",
            )),
            (biped_large::Species::Werewolf, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.beast_claws",
            )),
            (biped_large::Species::Cyclops, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.hammer.cyclops_hammer",
            )),
            (biped_large::Species::Dullahan, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.sword.dullahan_sword",
            )),
            (biped_large::Species::Mindflayer, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.staff.mindflayer_staff",
            )),
            (biped_large::Species::Minotaur, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.axe.minotaur_axe",
            )),
            (biped_large::Species::Tidalwarrior, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.tidal_claws",
            )),
            (biped_large::Species::Yeti, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.hammer.yeti_hammer",
            )),
            (biped_large::Species::Harvester, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.hammer.harvester_scythe",
            )),
            (biped_large::Species::Blueoni, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.axe.oni_blue_axe",
            )),
            (biped_large::Species::Redoni, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.hammer.oni_red_hammer",
            )),
            (biped_large::Species::Cultistwarlord, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.sword.bipedlarge-cultist",
            )),
            (biped_large::Species::Cultistwarlock, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.staff.bipedlarge-cultist",
            )),
            (biped_large::Species::Huskbrute, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.husk_brute",
            )),
        },
        Body::Object(body) => match body {
            object::Body::Crossbow => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.turret",
            )),
            object::Body::HaniwaSentry => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.haniwa_sentry",
            )),
            object::Body::SeaLantern => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.tidal_totem",
            )),
            object::Body::Tornado => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.tornado",
            )),
            object::Body::GnarlingTotemRed => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.biped_small.gnarling.redtotem",
            )),
            object::Body::GnarlingTotemGreen => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.biped_small.gnarling.greentotem",
            )),
            object::Body::GnarlingTotemWhite => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.biped_small.gnarling.whitetotem",
            )),
            _ => None,
        },
        Body::BipedSmall(biped_small) => match (biped_small.species, biped_small.body_type) {
            (biped_small::Species::Gnome, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.biped_small.adlet.tracker",
            )),
            (biped_small::Species::Husk, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.husk",
            )),
            _ => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.biped_small.adlet.hunter",
            )),
        },
        Body::BirdLarge(bird_large) => match (bird_large.species, bird_large.body_type) {
            (bird_large::Species::Cockatrice | bird_large::Species::FlameWyvern, _) => Some(
                Item::new_from_asset_expect("common.items.npc_weapons.unique.birdlargebreathe"),
            ),
            (bird_large::Species::Phoenix, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.birdlargefire",
            )),
            (bird_large::Species::Roc, _) => Some(Item::new_from_asset_expect(
                "common.items.npc_weapons.unique.birdlargebasic",
            )),
        },
        _ => None,
    };

    maybe_tool.unwrap_or_else(Item::empty)
}

/// Builder for character Loadouts, containing weapon and armour items belonging
/// to a character, along with some helper methods for loading `Item`-s and
/// `ItemConfig`
///
/// ```
/// use veloren_common::{comp::Item, LoadoutBuilder};
///
/// // Build a loadout with character starter defaults
/// // and a specific sword with default sword abilities
/// let sword = Item::new_from_asset_expect("common.items.weapons.sword.steel-8");
/// let loadout = LoadoutBuilder::empty()
///     .defaults()
///     .active_mainhand(Some(sword))
///     .build();
/// ```
#[derive(Clone)]
pub struct LoadoutBuilder(Loadout);

#[derive(Copy, Clone, PartialEq, Deserialize, Serialize, Debug, EnumIter)]
pub enum Preset {
    HuskSummon,
}

impl LoadoutBuilder {
    #[must_use]
    pub fn empty() -> Self { Self(Loadout::new_empty()) }

    #[must_use]
    /// Construct new `LoadoutBuilder` from `asset_specifier`
    /// Will panic if asset is broken
    pub fn from_asset_expect(asset_specifier: &str, rng: &mut impl Rng) -> Self {
        Self::from_asset(asset_specifier, rng).expect("failed to load loadut config")
    }

    /// Construct new `LoadoutBuilder` from `asset_specifier`
    pub fn from_asset(asset_specifier: &str, rng: &mut impl Rng) -> Result<Self, SpecError> {
        let loadout = Self::empty();
        loadout.with_asset(asset_specifier, rng)
    }

    #[must_use]
    /// Construct new default `LoadoutBuilder` for corresponding `body`
    ///
    /// NOTE: make sure that you check what is default for this body
    /// Use it if you don't care much about it, for example in "/spawn" command
    pub fn from_default(body: &Body) -> Self {
        let loadout = Self::empty();
        loadout
            .with_default_maintool(body)
            .with_default_equipment(body)
    }

    /// Construct new `LoadoutBuilder` from `asset_specifier`
    pub fn from_loadout_spec(
        loadout_spec: LoadoutSpec,
        rng: &mut impl Rng,
    ) -> Result<Self, SpecError> {
        let loadout = Self::empty();
        loadout.with_loadout_spec(loadout_spec, rng)
    }

    #[must_use]
    /// Construct new `LoadoutBuilder` from `asset_specifier`
    ///
    /// Will panic if asset is broken
    pub fn from_loadout_spec_expect(loadout_spec: LoadoutSpec, rng: &mut impl Rng) -> Self {
        Self::from_loadout_spec(loadout_spec, rng).expect("failed to load loadout spec")
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    /// Set default active mainhand weapon based on `body`
    pub fn with_default_maintool(self, body: &Body) -> Self {
        self.active_mainhand(Some(default_main_tool(body)))
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    /// Set default equipement based on `body`
    pub fn with_default_equipment(self, body: &Body) -> Self {
        let chest = match body {
            Body::BipedLarge(body) => match body.species {
                biped_large::Species::Mindflayer => {
                    Some("common.items.npc_armor.biped_large.mindflayer")
                },
                biped_large::Species::Minotaur => {
                    Some("common.items.npc_armor.biped_large.minotaur")
                },
                biped_large::Species::Tidalwarrior => {
                    Some("common.items.npc_armor.biped_large.tidal_warrior")
                },
                biped_large::Species::Yeti => Some("common.items.npc_armor.biped_large.yeti"),
                biped_large::Species::Harvester => {
                    Some("common.items.npc_armor.biped_large.harvester")
                },
                biped_large::Species::Ogre
                | biped_large::Species::Cyclops
                | biped_large::Species::Blueoni
                | biped_large::Species::Redoni
                | biped_large::Species::Cavetroll
                | biped_large::Species::Wendigo => {
                    Some("common.items.npc_armor.biped_large.generic")
                },
                biped_large::Species::Cultistwarlord => {
                    Some("common.items.npc_armor.biped_large.warlord")
                },
                biped_large::Species::Cultistwarlock => {
                    Some("common.items.npc_armor.biped_large.warlock")
                },
                _ => None,
            },
            Body::Golem(body) => match body.species {
                golem::Species::ClayGolem => Some("common.items.npc_armor.golem.claygolem"),
                golem::Species::WoodGolem => Some("common.items.npc_armor.golem.woodgolem"),
                _ => None,
            },
            Body::QuadrupedLow(body) => match body.species {
                quadruped_low::Species::Basilisk
                | quadruped_low::Species::Asp
                | quadruped_low::Species::Lavadrake
                | quadruped_low::Species::Maneater
                | quadruped_low::Species::Rocksnapper
                | quadruped_low::Species::Sandshark => {
                    Some("common.items.npc_armor.quadruped_low.generic")
                },
                quadruped_low::Species::Tortoise => {
                    Some("common.items.npc_armor.quadruped_low.shell")
                },
                _ => None,
            },
            Body::Theropod(body) => match body.species {
                theropod::Species::Archaeos
                | theropod::Species::Yale
                | theropod::Species::Ntouka
                | theropod::Species::Odonto => Some("common.items.npc_armor.theropod.rugged"),
                _ => None,
            },
            Body::Arthropod(_) => Some("common.items.npc_armor.arthropod.generic"),
            _ => None,
        };

        if let Some(chest) = chest {
            self.chest(Some(Item::new_from_asset_expect(chest)))
        } else {
            self
        }
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn with_preset(mut self, preset: Preset) -> Self {
        let rng = &mut rand::thread_rng();
        match preset {
            Preset::HuskSummon => {
                self = self.with_asset_expect("common.loadout.dungeon.tier-5.husk", rng);
            },
        }

        self
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn with_creator(
        mut self,
        creator: fn(LoadoutBuilder, Option<&SiteInformation>) -> LoadoutBuilder,
        economy: Option<&SiteInformation>,
    ) -> LoadoutBuilder {
        self = creator(self, economy);

        self
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    fn with_loadout_spec<R: Rng>(
        mut self,
        spec: LoadoutSpec,
        rng: &mut R,
    ) -> Result<Self, SpecError> {
        // Include any inheritance
        let spec = spec.eval(rng)?;

        // Utility function to unwrap our itemspec
        let mut to_item = |maybe_item: Option<ItemSpec>| {
            if let Some(item) = maybe_item {
                item.try_to_item(rng)
            } else {
                Ok(None)
            }
        };

        let to_pair = |maybe_hands: Option<Hands>, rng: &mut R| {
            if let Some(hands) = maybe_hands {
                hands.try_to_pair(rng)
            } else {
                Ok((None, None))
            }
        };

        // Place every item
        if let Some(item) = to_item(spec.head)? {
            self = self.head(Some(item));
        }
        if let Some(item) = to_item(spec.neck)? {
            self = self.neck(Some(item));
        }
        if let Some(item) = to_item(spec.shoulders)? {
            self = self.shoulder(Some(item));
        }
        if let Some(item) = to_item(spec.chest)? {
            self = self.chest(Some(item));
        }
        if let Some(item) = to_item(spec.gloves)? {
            self = self.hands(Some(item));
        }
        if let Some(item) = to_item(spec.ring1)? {
            self = self.ring1(Some(item));
        }
        if let Some(item) = to_item(spec.ring2)? {
            self = self.ring2(Some(item));
        }
        if let Some(item) = to_item(spec.back)? {
            self = self.back(Some(item));
        }
        if let Some(item) = to_item(spec.belt)? {
            self = self.belt(Some(item));
        }
        if let Some(item) = to_item(spec.legs)? {
            self = self.pants(Some(item));
        }
        if let Some(item) = to_item(spec.feet)? {
            self = self.feet(Some(item));
        }
        if let Some(item) = to_item(spec.tabard)? {
            self = self.tabard(Some(item));
        }
        if let Some(item) = to_item(spec.bag1)? {
            self = self.bag(ArmorSlot::Bag1, Some(item));
        }
        if let Some(item) = to_item(spec.bag2)? {
            self = self.bag(ArmorSlot::Bag2, Some(item));
        }
        if let Some(item) = to_item(spec.bag3)? {
            self = self.bag(ArmorSlot::Bag3, Some(item));
        }
        if let Some(item) = to_item(spec.bag4)? {
            self = self.bag(ArmorSlot::Bag4, Some(item));
        }
        if let Some(item) = to_item(spec.lantern)? {
            self = self.lantern(Some(item));
        }
        if let Some(item) = to_item(spec.glider)? {
            self = self.glider(Some(item));
        }
        let (active_mainhand, active_offhand) = to_pair(spec.active_hands, rng)?;
        if let Some(item) = active_mainhand {
            self = self.active_mainhand(Some(item));
        }
        if let Some(item) = active_offhand {
            self = self.active_offhand(Some(item));
        }
        let (inactive_mainhand, inactive_offhand) = to_pair(spec.inactive_hands, rng)?;
        if let Some(item) = inactive_mainhand {
            self = self.inactive_mainhand(Some(item));
        }
        if let Some(item) = inactive_offhand {
            self = self.inactive_offhand(Some(item));
        }

        Ok(self)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn with_asset(self, asset_specifier: &str, rng: &mut impl Rng) -> Result<Self, SpecError> {
        let spec =
            LoadoutSpec::load_cloned(asset_specifier).map_err(SpecError::LoadoutAssetError)?;
        self.with_loadout_spec(spec, rng)
    }

    /// # Usage
    /// Creates new `LoadoutBuilder` with all field replaced from
    /// `asset_specifier` which corresponds to loadout config
    ///
    /// # Panics
    /// 1) Will panic if there is no asset with such `asset_specifier`
    /// 2) Will panic if path to item specified in loadout file doesn't exist
    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn with_asset_expect(self, asset_specifier: &str, rng: &mut impl Rng) -> Self {
        self.with_asset(asset_specifier, rng)
            .expect("failed loading loadout config")
    }

    /// Set default armor items for the loadout. This may vary with game
    /// updates, but should be safe defaults for a new character.
    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn defaults(self) -> Self {
        let rng = &mut rand::thread_rng();
        self.with_asset_expect("common.loadout.default", rng)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    fn with_equipment(mut self, equip_slot: EquipSlot, item: Option<Item>) -> Self {
        // Panic if item doesn't correspond to slot
        assert!(
            item.as_ref()
                .map_or(true, |item| equip_slot.can_hold(&item.kind))
        );

        self.0.swap(equip_slot, item);
        self
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    fn with_armor(self, armor_slot: ArmorSlot, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::Armor(armor_slot), item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn active_mainhand(self, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::ActiveMainhand, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn active_offhand(self, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::ActiveOffhand, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn inactive_mainhand(self, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::InactiveMainhand, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn inactive_offhand(self, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::InactiveOffhand, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn shoulder(self, item: Option<Item>) -> Self {
        self.with_armor(ArmorSlot::Shoulders, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn chest(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Chest, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn belt(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Belt, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn hands(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Hands, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn pants(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Legs, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn feet(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Feet, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn back(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Back, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn ring1(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Ring1, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn ring2(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Ring2, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn neck(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Neck, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn lantern(self, item: Option<Item>) -> Self {
        self.with_equipment(EquipSlot::Lantern, item)
    }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn glider(self, item: Option<Item>) -> Self { self.with_equipment(EquipSlot::Glider, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn head(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Head, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn tabard(self, item: Option<Item>) -> Self { self.with_armor(ArmorSlot::Tabard, item) }

    #[must_use = "Method consumes builder and returns updated builder."]
    pub fn bag(self, which: ArmorSlot, item: Option<Item>) -> Self { self.with_armor(which, item) }

    #[must_use]
    pub fn build(self) -> Loadout { self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp::{self, Body};
    use rand::thread_rng;
    use strum::IntoEnumIterator;

    // Testing different species
    //
    // Things that will be catched - invalid assets paths for
    // creating default main hand tool or equipement without config
    #[test]
    fn test_loadout_species() {
        macro_rules! test_species {
            // base case
            ($species:tt : $body:tt) => {
                let mut rng = thread_rng();
                for s in comp::$species::ALL_SPECIES.iter() {
                    let body = comp::$species::Body::random_with(&mut rng, s);
                    let female_body = comp::$species::Body {
                        body_type: comp::$species::BodyType::Female,
                        ..body
                    };
                    let male_body = comp::$species::Body {
                        body_type: comp::$species::BodyType::Male,
                        ..body
                    };
                    std::mem::drop(LoadoutBuilder::from_default(
                        &Body::$body(female_body),
                    ));
                    std::mem::drop(LoadoutBuilder::from_default(
                        &Body::$body(male_body),
                    ));
                }
            };
            // recursive call
            ($base:tt : $body:tt, $($species:tt : $nextbody:tt),+ $(,)?) => {
                test_species!($base: $body);
                test_species!($($species: $nextbody),+);
            }
        }

        // See `[AllBodies](crate::comp::body::AllBodies)`
        test_species!(
            humanoid: Humanoid,
            quadruped_small: QuadrupedSmall,
            quadruped_medium: QuadrupedMedium,
            quadruped_low: QuadrupedLow,
            bird_medium: BirdMedium,
            bird_large: BirdLarge,
            fish_small: FishSmall,
            fish_medium: FishMedium,
            biped_small: BipedSmall,
            biped_large: BipedLarge,
            theropod: Theropod,
            dragon: Dragon,
            golem: Golem,
            arthropod: Arthropod,
        );
    }

    // Testing all loadout presets
    //
    // Things that will be catched - invalid assets paths
    #[test]
    fn test_loadout_presets() {
        for preset in Preset::iter() {
            std::mem::drop(LoadoutBuilder::empty().with_preset(preset));
        }
    }

    // It just loads every loadout asset and tries to validate them
    //
    // TODO: optimize by caching checks
    // Because of nature of inheritance of loadout specs,
    // we will check some loadout assets at least two times.
    // One for asset itself and second if it serves as a base for other asset.
    #[test]
    fn validate_all_loadout_assets() {
        let loadouts = assets::load_dir::<LoadoutSpec>("common.loadout", true)
            .expect("failed to load loadout directory");
        for loadout_id in loadouts.ids() {
            let loadout =
                LoadoutSpec::load_cloned(loadout_id).expect("failed to load loadout asset");
            loadout
                .validate(vec![loadout_id.to_owned()])
                .unwrap_or_else(|e| panic!("{loadout_id} is broken: {e:?}"));
        }
    }

    // Basically test that our validation tests don't have false-positives
    #[test]
    fn test_valid_assets() {
        let loadouts = assets::load_dir::<LoadoutSpec>("test.loadout.ok", true)
            .expect("failed to load loadout directory");

        for loadout_id in loadouts.ids() {
            let loadout =
                LoadoutSpec::load_cloned(loadout_id).expect("failed to load loadout asset");
            loadout
                .validate(vec![loadout_id.to_owned()])
                .unwrap_or_else(|e| panic!("{loadout_id} is broken: {e:?}"));
        }
    }
}
