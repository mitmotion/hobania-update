use crate::{
    assets::AssetExt,
    comp::inventory::item::{
        armor::{Armor, ArmorKind},
        modular, tool, Glider, ItemDef, ItemDesc, ItemKind, Lantern, Throwable, Utility,
    },
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum ItemKey {
    Tool(String),
    ModularWeapon(modular::ModularWeaponKey),
    ModularWeaponComponent(modular::ModularWeaponComponentKey),
    Lantern(String),
    Glider(String),
    Armor(ArmorKind),
    Utility(Utility),
    Consumable(String),
    Throwable(Throwable),
    Ingredient(String),
    TagExamples(Vec<ItemKey>),
    Empty,
}

impl<T: ItemDesc> From<&T> for ItemKey {
    fn from(item_desc: &T) -> Self {
        let item_kind = item_desc.kind();
        let item_definition_id = item_desc.item_definition_id();

        match item_kind {
            ItemKind::Tool(tool) => {
                use tool::StatKind;
                match tool.stats {
                    StatKind::Direct(_) => ItemKey::Tool(item_definition_id.to_owned()),
                    StatKind::Modular => ItemKey::ModularWeapon(modular::weapon_to_key(item_desc)),
                }
            },
            ItemKind::ModularComponent(_) => {
                match modular::weapon_component_to_key(item_desc) {
                    Ok(key) => ItemKey::ModularWeaponComponent(key),
                    // TODO: Maybe use a different ItemKey?
                    Err(_) => ItemKey::Tool(item_definition_id.to_owned()),
                }
            },
            ItemKind::Lantern(Lantern { kind, .. }) => ItemKey::Lantern(kind.clone()),
            ItemKind::Glider(Glider { kind, .. }) => ItemKey::Glider(kind.clone()),
            ItemKind::Armor(Armor { kind, .. }) => ItemKey::Armor(kind.clone()),
            ItemKind::Utility { kind, .. } => ItemKey::Utility(*kind),
            ItemKind::Consumable { .. } => ItemKey::Consumable(item_definition_id.to_owned()),
            ItemKind::Throwable { kind, .. } => ItemKey::Throwable(*kind),
            ItemKind::Ingredient { kind, .. } => ItemKey::Ingredient(kind.clone()),
            ItemKind::TagExamples { item_ids } => ItemKey::TagExamples(
                item_ids
                    .iter()
                    .map(|id| ItemKey::from(&*Arc::<ItemDef>::load_expect_cloned(id)))
                    .collect(),
            ),
        }
    }
}
