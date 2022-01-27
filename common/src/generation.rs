use crate::{
    assets::{self, AssetExt, Error},
    comp::{
        self, agent, humanoid,
        inventory::loadout_builder::{ItemSpec, LoadoutBuilder},
        Alignment, Body, Item,
    },
    lottery::LootSpec,
    npc::{self, NPC_NAMES},
    trade,
    trade::SiteInformation,
};
use rand::prelude::SliceRandom;
use serde::Deserialize;
use vek::*;

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub enum NameKind {
    Name(String),
    Automatic,
    Uninit,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
pub enum BodyBuilder {
    RandomWith(String),
    Exact(Body),
    Uninit,
}

#[derive(Debug, Deserialize, Clone)]
pub enum AlignmentMark {
    Alignment(Alignment),
    Uninit,
}

impl Default for AlignmentMark {
    fn default() -> Self { Self::Alignment(Alignment::Wild) }
}

/// - TwoHanded(ItemSpec) for one 2h or 1h weapon,
/// - Paired(ItemSpec) for two 1h weapons aka berserker mode,
/// - Mix { mainhand: ItemSpec, offhand: ItemSpec, }
///  for two different 1h weapons.
#[derive(Debug, Deserialize, Clone)]
pub enum Hands {
    TwoHanded(ItemSpec),
    Paired(ItemSpec),
    Mix {
        mainhand: ItemSpec,
        offhand: ItemSpec,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub enum LoadoutAsset {
    Loadout(String),
    Choice(Vec<(u32, String)>),
}

#[derive(Debug, Deserialize, Clone)]
pub enum LoadoutKind {
    FromBody,
    Asset(LoadoutAsset),
    Hands(Hands),
    Extended {
        hands: Hands,
        base_asset: LoadoutAsset,
        inventory: Vec<(u32, String)>,
    },
}

#[derive(Debug, Deserialize, Clone)]
pub enum Meta {
    SkillSetAsset(String),
}

// FIXME: currently this is used for both base definition
// and extension manifest.
// This is why all fields have Uninit kind which is means
// that this field should be either Default or Unchanged
// depending on how it is used.
//
// When we will use exension manifests more, it would be nicer to
// split EntityBase and EntityExtension to different structs.
//
// Fields which have Uninit enum kind
// should be optional (or renamed to Unchanged) in EntityExtension
// and required (or renamed to Default) in EntityBase
/// Struct for EntityInfo manifest.
///
/// Intended to use with .ron files as base definion or
/// in rare cases as extension manifest.
/// Pure data struct, doesn't do anything until evaluated with EntityInfo.
///
/// Check assets/common/entity/template.ron or other examples.
///
/// # Example
/// ```
/// use vek::Vec3;
/// use veloren_common::generation::EntityInfo;
///
/// // create new EntityInfo at dummy position
/// // and fill it with template config
/// let dummy_position = Vec3::new(0.0, 0.0, 0.0);
/// // rng is required because some elements may be randomly generated
/// let mut dummy_rng = rand::thread_rng();
/// let entity =
///     EntityInfo::at(dummy_position).with_asset_expect("common.entity.template", &mut dummy_rng);
/// ```
#[derive(Debug, Deserialize, Clone)]
pub struct EntityConfig {
    /// Name of Entity
    /// Can be Name(String) with given name
    /// or Automatic which will call automatic name depend on Body
    /// or Uninit (means it should be specified somewhere in code)
    // Hidden, because its behaviour depends on `body` field.
    name: NameKind,

    /// Body
    /// Can be Exact (Body with all fields e.g BodyType, Species, Hair color and
    /// such) or RandomWith (will generate random body or species)
    /// or Uninit (means it should be specified somewhere in code)
    pub body: BodyBuilder,

    /// Alignment, can be Uninit
    pub alignment: AlignmentMark,

    /// Loot
    /// See LootSpec in lottery
    pub loot: LootSpec<String>,

    /// Loadout & Inventory
    /// Variants:
    /// `FromBody` - will get default equipement for entity body.
    /// `Asset` - will get loadout from loadout asset.
    /// `Hands` - naked character with weapons.
    /// `Extended` - combination of Asset and Hands with ability to specify
    /// inventory of entity.
    pub loadout: LoadoutKind,

    /// Meta Info for optional fields
    /// Possible fields:
    /// SkillSetAsset(String) with asset_specifier for skillset
    #[serde(default)]
    pub meta: Vec<Meta>,
}

impl assets::Asset for EntityConfig {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

impl EntityConfig {
    pub fn from_asset_expect_owned(asset_specifier: &str) -> Self {
        Self::load_owned(asset_specifier)
            .unwrap_or_else(|e| panic!("Failed to load {}. Error: {:?}", asset_specifier, e))
    }

    #[must_use]
    pub fn with_body(mut self, body: BodyBuilder) -> Self {
        self.body = body;

        self
    }
}

/// Return all entity config specifiers
pub fn try_all_entity_configs() -> Result<Vec<String>, Error> {
    let configs = assets::load_dir::<EntityConfig>("common.entity", true)?;
    Ok(configs.ids().map(|id| id.to_owned()).collect())
}

#[derive(Clone)]
pub struct EntityInfo {
    pub pos: Vec3<f32>,
    pub is_waypoint: bool, // Edge case, overrides everything else
    // Agent
    pub has_agency: bool,
    pub alignment: Alignment,
    pub agent_mark: Option<agent::Mark>,
    // Stats
    pub body: Body,
    pub name: Option<String>,
    pub scale: f32,
    // Loot
    pub loot: LootSpec<String>,
    // Loadout
    pub inventory: Vec<(u32, Item)>,
    pub loadout: LoadoutBuilder,
    pub make_loadout: Option<fn(LoadoutBuilder, Option<&trade::SiteInformation>) -> LoadoutBuilder>,
    // Skills
    pub skillset_asset: Option<String>,

    // Not implemented
    pub pet: Option<Box<EntityInfo>>,

    // Economy
    // we can't use DHashMap, do we want to move that into common?
    pub trading_information: Option<trade::SiteInformation>,
    //Option<hashbrown::HashMap<crate::trade::Good, (f32, f32)>>, /* price and available amount */
}

impl EntityInfo {
    pub fn at(pos: Vec3<f32>) -> Self {
        Self {
            pos,
            is_waypoint: false,
            has_agency: true,
            alignment: Alignment::Wild,
            agent_mark: None,
            body: Body::Humanoid(humanoid::Body::random()),
            name: None,
            scale: 1.0,
            loot: LootSpec::Nothing,
            inventory: Vec::new(),
            loadout: LoadoutBuilder::empty(),
            make_loadout: None,
            skillset_asset: None,
            pet: None,
            trading_information: None,
        }
    }

    /// Helper function for applying config from asset
    /// with specified Rng for managing loadout.
    #[must_use]
    pub fn with_asset_expect<R>(self, asset_specifier: &str, loadout_rng: &mut R) -> Self
    where
        R: rand::Rng,
    {
        let config = EntityConfig::load_expect_cloned(asset_specifier);

        self.with_entity_config(config, Some(asset_specifier), loadout_rng)
    }

    /// Evaluate and apply EntityConfig
    #[must_use]
    pub fn with_entity_config<R>(
        mut self,
        config: EntityConfig,
        config_asset: Option<&str>,
        loadout_rng: &mut R,
    ) -> Self
    where
        R: rand::Rng,
    {
        let EntityConfig {
            name,
            body,
            alignment,
            loadout,
            loot,
            meta,
        } = config;

        match body {
            BodyBuilder::RandomWith(string) => {
                let npc::NpcBody(_body_kind, mut body_creator) =
                    string.parse::<npc::NpcBody>().unwrap_or_else(|err| {
                        panic!("failed to parse body {:?}. Err: {:?}", &string, err)
                    });
                let body = body_creator();
                self = self.with_body(body);
            },
            BodyBuilder::Exact(body) => {
                self = self.with_body(body);
            },
            BodyBuilder::Uninit => {},
        }

        // NOTE: set name after body, as it's used with automatic name
        match name {
            NameKind::Name(name) => {
                self = self.with_name(name);
            },
            NameKind::Automatic => {
                self = self.with_automatic_name();
            },
            NameKind::Uninit => {},
        }

        if let AlignmentMark::Alignment(alignment) = alignment {
            self = self.with_alignment(alignment);
        }

        self = self.with_loot_drop(loot);

        // NOTE: set loadout after body, as it's used with default equipement
        self = self.with_loadout(loadout, config_asset, loadout_rng);

        for field in meta {
            match field {
                Meta::SkillSetAsset(asset) => {
                    self = self.with_skillset_asset(asset);
                },
            }
        }

        self
    }

    /// Return EntityInfo with LoadoutBuilder overwritten
    // NOTE: helper function, think twice before exposing it
    #[must_use]
    fn with_loadout<R>(
        mut self,
        loadout: LoadoutKind,
        config_asset: Option<&str>,
        rng: &mut R,
    ) -> Self
    where
        R: rand::Rng,
    {
        match loadout {
            LoadoutKind::FromBody => {
                self = self.with_default_equip();
            },
            LoadoutKind::Asset(loadout) => {
                self = self.with_loadout_asset(loadout, rng);
            },
            LoadoutKind::Hands(hands) => {
                self = self.with_hands(hands, config_asset, rng);
            },
            LoadoutKind::Extended {
                hands,
                base_asset,
                inventory,
            } => {
                self = self.with_loadout_asset(base_asset, rng);
                self = self.with_hands(hands, config_asset, rng);
                // FIXME: this shouldn't always overwrite
                // inventory. Think about this when we get to
                // entity config inheritance.
                self.inventory = inventory
                    .into_iter()
                    .map(|(num, i)| (num, Item::new_from_asset_expect(&i)))
                    .collect();
            },
        }

        self
    }

    /// Return EntityInfo with LoadoutBuilder overwritten
    // NOTE: helper function, think twice before exposing it
    #[must_use]
    fn with_default_equip(mut self) -> Self {
        let loadout_builder = LoadoutBuilder::from_default(&self.body);
        self.loadout = loadout_builder;

        self
    }

    /// Return EntityInfo with LoadoutBuilder overwritten
    // NOTE: helper function, think twice before exposing it
    #[must_use]
    fn with_loadout_asset<R>(mut self, loadout: LoadoutAsset, rng: &mut R) -> Self
    where
        R: rand::Rng,
    {
        match loadout {
            LoadoutAsset::Loadout(asset) => {
                let loadout = LoadoutBuilder::from_asset_expect(&asset, Some(rng));
                self.loadout = loadout;
            },
            LoadoutAsset::Choice(assets) => {
                // TODO:
                // choose_weighted allocates WeightedIndex,
                // possible optimizaiton with using Lottery
                let (_p, asset) = assets
                    .choose_weighted(rng, |(p, _asset)| *p)
                    .expect("rng error");

                let loadout = LoadoutBuilder::from_asset_expect(asset, Some(rng));
                self.loadout = loadout;
            },
        }

        self
    }

    /// Return EntityInfo with weapons applied to LoadoutBuilder
    // NOTE: helper function, think twice before exposing it
    #[must_use]
    fn with_hands<R>(mut self, hands: Hands, config_asset: Option<&str>, rng: &mut R) -> Self
    where
        R: rand::Rng,
    {
        match hands {
            Hands::TwoHanded(main_tool) => {
                let tool = main_tool.try_to_item(config_asset.unwrap_or("??"), rng);
                if let Some(tool) = tool {
                    self.loadout = self.loadout.active_mainhand(Some(tool));
                }
            },
            Hands::Paired(tool) => {
                //FIXME: very stupid code, which just tries same item two times
                //figure out reasonable way to clone item
                let main_tool = tool.try_to_item(config_asset.unwrap_or("??"), rng);
                let second_tool = tool.try_to_item(config_asset.unwrap_or("??"), rng);

                if let Some(main_tool) = main_tool {
                    self.loadout = self.loadout.active_mainhand(Some(main_tool));
                }
                if let Some(second_tool) = second_tool {
                    self.loadout = self.loadout.active_offhand(Some(second_tool));
                }
            },
            Hands::Mix { mainhand, offhand } => {
                let main_tool = mainhand.try_to_item(config_asset.unwrap_or("??"), rng);
                let second_tool = offhand.try_to_item(config_asset.unwrap_or("??"), rng);

                if let Some(main_tool) = main_tool {
                    self.loadout = self.loadout.active_mainhand(Some(main_tool));
                }
                if let Some(second_tool) = second_tool {
                    self.loadout = self.loadout.active_offhand(Some(second_tool));
                }
            },
        }

        self
    }

    #[must_use]
    pub fn do_if(mut self, cond: bool, f: impl FnOnce(Self) -> Self) -> Self {
        if cond {
            self = f(self);
        }
        self
    }

    #[must_use]
    pub fn into_waypoint(mut self) -> Self {
        self.is_waypoint = true;
        self
    }

    #[must_use]
    pub fn with_alignment(mut self, alignment: Alignment) -> Self {
        self.alignment = alignment;
        self
    }

    #[must_use]
    pub fn with_body(mut self, body: Body) -> Self {
        self.body = body;
        self
    }

    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    #[must_use]
    pub fn with_agency(mut self, agency: bool) -> Self {
        self.has_agency = agency;
        self
    }

    #[must_use]
    pub fn with_agent_mark(mut self, agent_mark: agent::Mark) -> Self {
        self.agent_mark = Some(agent_mark);
        self
    }

    #[must_use]
    pub fn with_loot_drop(mut self, loot_drop: LootSpec<String>) -> Self {
        self.loot = loot_drop;
        self
    }

    #[must_use]
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    #[must_use]
    pub fn with_lazy_loadout(
        mut self,
        creator: fn(LoadoutBuilder, Option<&trade::SiteInformation>) -> LoadoutBuilder,
    ) -> Self {
        self.make_loadout = Some(creator);
        self
    }

    #[must_use]
    pub fn with_skillset_asset(mut self, asset: String) -> Self {
        self.skillset_asset = Some(asset);
        self
    }

    #[must_use]
    pub fn with_automatic_name(mut self) -> Self {
        let npc_names = NPC_NAMES.read();
        let name = match &self.body {
            Body::Humanoid(body) => Some(get_npc_name(&npc_names.humanoid, body.species)),
            Body::QuadrupedMedium(body) => {
                Some(get_npc_name(&npc_names.quadruped_medium, body.species))
            },
            Body::BirdMedium(body) => Some(get_npc_name(&npc_names.bird_medium, body.species)),
            Body::BirdLarge(body) => Some(get_npc_name(&npc_names.bird_large, body.species)),
            Body::FishSmall(body) => Some(get_npc_name(&npc_names.fish_small, body.species)),
            Body::FishMedium(body) => Some(get_npc_name(&npc_names.fish_medium, body.species)),
            Body::Theropod(body) => Some(get_npc_name(&npc_names.theropod, body.species)),
            Body::QuadrupedSmall(body) => {
                Some(get_npc_name(&npc_names.quadruped_small, body.species))
            },
            Body::Dragon(body) => Some(get_npc_name(&npc_names.dragon, body.species)),
            Body::QuadrupedLow(body) => Some(get_npc_name(&npc_names.quadruped_low, body.species)),
            Body::Golem(body) => Some(get_npc_name(&npc_names.golem, body.species)),
            Body::BipedLarge(body) => Some(get_npc_name(&npc_names.biped_large, body.species)),
            Body::Arthropod(body) => Some(get_npc_name(&npc_names.arthropod, body.species)),
            _ => None,
        };
        self.name = name.map(str::to_owned);
        self
    }

    /// map contains price+amount
    #[must_use]
    pub fn with_economy(mut self, e: &SiteInformation) -> Self {
        self.trading_information = Some(e.clone());
        self
    }
}

#[derive(Default)]
pub struct ChunkSupplement {
    pub entities: Vec<EntityInfo>,
}

impl ChunkSupplement {
    pub fn add_entity(&mut self, entity: EntityInfo) { self.entities.push(entity); }
}

pub fn get_npc_name<
    'a,
    Species,
    SpeciesData: for<'b> core::ops::Index<&'b Species, Output = npc::SpeciesNames>,
>(
    body_data: &'a comp::BodyData<npc::BodyNames, SpeciesData>,
    species: Species,
) -> &'a str {
    &body_data.species[&species].generic
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{comp::inventory::slot::EquipSlot, SkillSetBuilder};
    use hashbrown::HashMap;

    #[derive(Debug, Eq, Hash, PartialEq)]
    enum MetaId {
        SkillSetAsset,
    }

    impl Meta {
        fn id(&self) -> MetaId {
            match self {
                Meta::SkillSetAsset(_) => MetaId::SkillSetAsset,
            }
        }
    }

    #[cfg(test)]
    fn validate_body(body: &BodyBuilder, config_asset: &str) {
        match body {
            BodyBuilder::RandomWith(string) => {
                let npc::NpcBody(_body_kind, mut body_creator) =
                    string.parse::<npc::NpcBody>().unwrap_or_else(|err| {
                        panic!(
                            "failed to parse body {:?} in {}. Err: {:?}",
                            &string, config_asset, err
                        )
                    });
                let _ = body_creator();
            },
            BodyBuilder::Uninit | BodyBuilder::Exact { .. } => {},
        }
    }

    #[cfg(test)]
    fn validate_loadout(loadout: LoadoutKind, body: &BodyBuilder, config_asset: &str) {
        match loadout {
            LoadoutKind::FromBody => {
                if body.clone() == BodyBuilder::Uninit {
                    // there is a big chance to call automatic name
                    // when body is yet undefined
                    //
                    // use .with_automatic_name() in code explicitly
                    panic!("Used FromBody loadout with Uninit body in {}", config_asset);
                }
            },
            LoadoutKind::Asset(loadout) => {
                validate_loadout_asset(loadout, config_asset);
            },
            LoadoutKind::Hands(hands) => {
                validate_hands(hands, config_asset);
            },
            LoadoutKind::Extended {
                hands,
                base_asset,
                inventory,
            } => {
                validate_hands(hands, config_asset);
                validate_loadout_asset(base_asset, config_asset);
                for (num, item_str) in inventory {
                    let item = Item::new_from_asset(&item_str);
                    let mut item = item.unwrap_or_else(|err| {
                        panic!("can't load {} in {}: {:?}", item_str, config_asset, err);
                    });
                    item.set_amount(num).unwrap_or_else(|err| {
                        panic!(
                            "can't set amount {} for {} in {}: {:?}",
                            num, item_str, config_asset, err
                        );
                    });
                }
            },
        }
    }

    #[cfg(test)]
    fn validate_loadout_asset(loadout: LoadoutAsset, config_asset: &str) {
        match loadout {
            LoadoutAsset::Loadout(asset) => {
                let rng = &mut rand::thread_rng();
                // loadout is tested in loadout_builder
                // we just try to load it and check that it exists
                std::mem::drop(LoadoutBuilder::from_asset_expect(&asset, Some(rng)));
            },
            LoadoutAsset::Choice(assets) => {
                for (p, asset) in assets {
                    if p == 0 {
                        panic!("Weight of loadout asset is zero in {config_asset}");
                    }
                    validate_loadout_asset(LoadoutAsset::Loadout(asset), config_asset);
                }
            },
        }
    }

    #[cfg(test)]
    fn validate_hands(hands: Hands, _config_asset: &str) {
        match hands {
            Hands::TwoHanded(main_tool) => {
                main_tool.validate(EquipSlot::ActiveMainhand);
            },
            Hands::Paired(tool) => {
                tool.validate(EquipSlot::ActiveMainhand);
                tool.validate(EquipSlot::ActiveOffhand);
            },
            Hands::Mix { mainhand, offhand } => {
                mainhand.validate(EquipSlot::ActiveMainhand);
                offhand.validate(EquipSlot::ActiveOffhand);
            },
        }
    }

    #[cfg(test)]
    fn validate_name(name: NameKind, body: BodyBuilder, config_asset: &str) {
        if name == NameKind::Automatic && body == BodyBuilder::Uninit {
            // there is a big chance to call automatic name
            // when body is yet undefined
            //
            // use .with_automatic_name() in code explicitly
            panic!("Used Automatic name with Uninit body in {}", config_asset);
        }
    }

    #[cfg(test)]
    fn validate_loot(loot: LootSpec<String>, _config_asset: &str) {
        use crate::lottery;
        lottery::tests::validate_loot_spec(&loot);
    }

    #[cfg(test)]
    fn validate_meta(meta: Vec<Meta>, config_asset: &str) {
        let mut meta_counter = HashMap::new();
        for field in meta {
            meta_counter
                .entry(field.id())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            match field {
                Meta::SkillSetAsset(asset) => {
                    std::mem::drop(SkillSetBuilder::from_asset_expect(&asset));
                },
            }
        }
        for (meta_id, counter) in meta_counter {
            if counter > 1 {
                panic!("Duplicate {:?} in {}", meta_id, config_asset);
            }
        }
    }

    #[test]
    fn test_all_entity_assets() {
        // Get list of entity configs, load everything, validate content.
        let entity_configs =
            try_all_entity_configs().expect("Failed to access entity configs directory");
        for config_asset in entity_configs {
            // print asset name so we don't need to find errors everywhere
            // it'll be ignored by default so you'll see it only in case of errors
            //
            // TODO:
            // 1) Add try_validate() for loadout_builder::ItemSpec which will return
            // Result and we will happily panic in validate_hands() with name of
            // config_asset.
            // 2) Add try_from_asset() for LoadoutBuilder and
            // SkillSet builder which will return Result and we will happily
            // panic in validate_meta() with the name of config_asset
            println!("{}:", &config_asset);

            let EntityConfig {
                body,
                loadout,
                name,
                loot,
                meta,
                alignment: _alignment, // can't fail if serialized, it's a boring enum
            } = EntityConfig::from_asset_expect_owned(&config_asset);

            validate_body(&body, &config_asset);
            // body dependent stuff
            validate_loadout(loadout, &body, &config_asset);
            validate_name(name, body, &config_asset);
            // misc
            validate_loot(loot, &config_asset);
            validate_meta(meta, &config_asset);
        }
    }
}
