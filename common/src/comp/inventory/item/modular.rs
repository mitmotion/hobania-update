use super::{
    tool::{self, Hands},
    Item, ItemKind, ItemName, ItemTag, Quality, RawItemDef, TagExampleInfo, ToolKind,
};
use crate::recipe::{RawRecipe, RawRecipeBook, RawRecipeInput};
use hashbrown::HashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModularComponentKind {
    Damage,
    Held,
}

impl ModularComponentKind {
    fn identifier_name(&self) -> &'static str {
        match self {
            ModularComponentKind::Damage => "damage",
            ModularComponentKind::Held => "held",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModularComponent {
    pub toolkind: ToolKind,
    pub modkind: ModularComponentKind,
    pub stats: tool::Stats,
    pub hand_restriction: Option<Hands>,
    pub weapon_name: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModularComponentTag {
    toolkind: ToolKind,
    modkind: ModularComponentKind,
    hands: Hands,
}

impl TagExampleInfo for ModularComponentTag {
    fn name(&self) -> Cow<'static, str> {
        Cow::Owned(format!(
            "{} {}",
            self.hands.identifier_name().to_owned(),
            match self.modkind {
                ModularComponentKind::Damage => match self.toolkind {
                    ToolKind::Sword => "sword blade",
                    ToolKind::Axe => "axe head",
                    ToolKind::Hammer => "hammer head",
                    ToolKind::Bow => "bow limbs",
                    ToolKind::Dagger => "dagger blade",
                    ToolKind::Staff => "staff head",
                    ToolKind::Sceptre => "sceptre head",
                    // TODO: naming
                    ToolKind::Spear => "spear damage component",
                    ToolKind::Blowgun => "blowgun damage component",
                    ToolKind::Shield => "shield damage component",
                    ToolKind::Debug => "debug damage component",
                    ToolKind::Farming => "farming damage component",
                    ToolKind::Pick => "pickaxe head",
                    ToolKind::Natural => "natural damage component",
                    ToolKind::Empty => "empty damage component",
                },
                ModularComponentKind::Held => match self.toolkind {
                    ToolKind::Sword => "sword hilt",
                    ToolKind::Axe => "axe shaft",
                    ToolKind::Hammer => "hammer shaft",
                    ToolKind::Bow => "bow riser",
                    ToolKind::Dagger => "dagger grip",
                    ToolKind::Staff => "staff shaft",
                    ToolKind::Sceptre => "sceptre shaft",
                    // TODO: naming
                    ToolKind::Spear => "spear held component",
                    ToolKind::Blowgun => "blowgun held component",
                    ToolKind::Shield => "shield held component",
                    ToolKind::Natural => "natural held component",
                    ToolKind::Debug => "debug held component",
                    ToolKind::Farming => "farming held component",
                    ToolKind::Pick => "pickaxe handle",
                    ToolKind::Empty => "empty held component",
                },
            }
        ))
    }

    fn exemplar_identifier(&self) -> Cow<'static, str> {
        Cow::Owned(format!(
            "{}.{}.{}.{}",
            TAG_EXAMPLES_PREFIX,
            self.modkind.identifier_name(),
            self.toolkind.identifier_name(),
            self.hands.identifier_name()
        ))
    }
}

const SUPPORTED_TOOLKINDS: [ToolKind; 6] = [
    ToolKind::Sword,
    ToolKind::Axe,
    ToolKind::Hammer,
    ToolKind::Bow,
    ToolKind::Staff,
    ToolKind::Sceptre,
];
const MODKINDS: [ModularComponentKind; 2] =
    [ModularComponentKind::Damage, ModularComponentKind::Held];

const WEAPON_PREFIX: &str = "common.items.weapons.modular";
const TAG_EXAMPLES_PREFIX: &str = "common.items.tag_examples.modular";

const HANDS: [Hands; 2] = [Hands::One, Hands::Two];

fn make_weapon_def(toolkind: ToolKind) -> (String, RawItemDef) {
    let identifier = format!("{}.{}", WEAPON_PREFIX, toolkind.identifier_name());
    let name = ItemName::Modular;
    let tool = tool::Tool {
        kind: toolkind,
        hands: tool::HandsKind::Modular,
        stats: tool::StatKind::Modular,
    };
    let kind = ItemKind::Tool(tool);
    let quality = Quality::Common;
    let item = RawItemDef {
        name,
        description: "".to_string(),
        kind,
        quality,
        tags: Vec::new(),
        slots: 0,
        ability_spec: None,
    };
    (identifier, item)
}

fn make_recipe_def(identifier: String, toolkind: ToolKind, hands: Hands) -> RawRecipe {
    let output = (identifier, 1);
    let mut inputs = Vec::new();
    for &modkind in &MODKINDS {
        let input = RawRecipeInput::Tag(ItemTag::ModularComponent(ModularComponentTag {
            toolkind,
            modkind,
            hands,
        }));
        inputs.push((input, 1, true));
    }
    RawRecipe {
        output,
        inputs,
        craft_sprite: None,
    }
}

fn make_tagexample_def(
    toolkind: ToolKind,
    modkind: ModularComponentKind,
    hands: Hands,
) -> (String, RawItemDef) {
    let identifier = format!(
        "{}.{}.{}.{}",
        TAG_EXAMPLES_PREFIX,
        modkind.identifier_name(),
        toolkind.identifier_name(),
        hands.identifier_name(),
    );
    let tag = ModularComponentTag {
        toolkind,
        modkind,
        hands,
    };
    // TODO: i18n
    let name = ItemName::Direct(format!("Any {}", tag.name()));
    let kind = ItemKind::TagExamples {
        // TODO: Iterate over components
        item_ids: Vec::new(),
    };
    let quality = Quality::Common;

    let item = RawItemDef {
        name,
        description: "".to_string(),
        kind,
        quality,
        tags: vec![ItemTag::ModularComponent(tag)],
        slots: 0,
        ability_spec: None,
    };
    (identifier, item)
}

// Checks that modular weapons should exist for a given toolkind and hands
// combination
fn exists(tool: ToolKind, hands: Hands) -> bool {
    match tool {
        // Has both 1 handed and 2 handed variants
        ToolKind::Sword | ToolKind::Axe | ToolKind::Hammer => true,
        // Has only 2 handed variants
        ToolKind::Bow | ToolKind::Staff | ToolKind::Sceptre => matches!(hands, Hands::Two),
        // Modular weapons do not yet exist
        ToolKind::Dagger
        | ToolKind::Spear
        | ToolKind::Blowgun
        | ToolKind::Shield
        | ToolKind::Natural
        | ToolKind::Debug
        | ToolKind::Farming
        | ToolKind::Pick
        | ToolKind::Empty => false,
    }
}

fn initialize_modular_assets() -> (HashMap<String, RawItemDef>, RawRecipeBook) {
    let mut itemdefs = HashMap::new();
    let mut recipes = HashMap::new();
    for &toolkind in &SUPPORTED_TOOLKINDS {
        let (identifier, item) = make_weapon_def(toolkind);
        itemdefs.insert(identifier.clone(), item);
        for &hands in &HANDS {
            if exists(toolkind, hands) {
                let recipe = make_recipe_def(identifier.clone(), toolkind, hands);
                recipes.insert(
                    format!("{}.{}", identifier.clone(), hands.identifier_name()),
                    recipe,
                );
                for &modkind in &MODKINDS {
                    let (identifier, item) = make_tagexample_def(toolkind, modkind, hands);
                    itemdefs.insert(identifier, item);
                }
            }
        }
    }
    (itemdefs, RawRecipeBook(recipes))
}

lazy_static! {
    static ref ITEM_DEFS_AND_RECIPES: (HashMap<String, RawItemDef>, RawRecipeBook) =
        initialize_modular_assets();
}

pub(crate) fn append_modular_recipes(recipes: &mut RawRecipeBook) {
    for (name, recipe) in ITEM_DEFS_AND_RECIPES.1.0.iter() {
        // avoid clobbering recipes from the filesystem, to allow overrides
        if !recipes.0.contains_key(name) {
            recipes.0.insert(name.clone(), recipe.clone());
        }
    }
}

/// Synthesize modular assets programmatically, to allow for the following:
/// - Allow the modular tag_examples to auto-update with the list of applicable
///   components
pub(super) fn synthesize_modular_asset(specifier: &str) -> Option<RawItemDef> {
    let ret = ITEM_DEFS_AND_RECIPES.0.get(specifier).cloned();
    tracing::trace!("synthesize_modular_asset({:?}) -> {:?}", specifier, ret);
    ret
}

/// Modular weapons are named as "{Material} {Weapon}" where {Weapon} is from
/// the damage component used and {Material} is from the material the damage
/// component is created from.
pub(super) fn modular_name<'a>(item: &'a Item, arg1: &'a str) -> Cow<'a, str> {
    match item.kind() {
        ItemKind::Tool(tool) => {
            let damage_components = item.components().iter().filter(|comp| {
                matches!(comp.kind(), ItemKind::ModularComponent(ModularComponent { modkind, .. })
                        if matches!(modkind, ModularComponentKind::Damage)
                )
            });
            // Last fine as there should only ever be one damage component on a weapon
            let (material_name, weapon_name) = if let Some(component) = damage_components.last() {
                let materials =
                    component
                        .components()
                        .iter()
                        .filter_map(|comp| match comp.kind() {
                            ItemKind::Ingredient { .. } => Some(comp.kind()),
                            _ => None,
                        });
                // TODO: Better handle multiple materials
                let material_name =
                    if let Some(ItemKind::Ingredient { descriptor, .. }) = materials.last() {
                        descriptor
                    } else {
                        "Modular"
                    };
                let weapon_name =
                    if let ItemKind::ModularComponent(ModularComponent { weapon_name, .. }) =
                        component.kind()
                    {
                        weapon_name
                    } else {
                        tool.kind.identifier_name()
                    };
                (material_name, weapon_name)
            } else {
                ("Modular", tool.kind.identifier_name())
            };

            Cow::Owned(format!("{} {}", material_name, weapon_name))
        },
        ItemKind::ModularComponent(comp) => {
            match comp.modkind {
                ModularComponentKind::Damage => {
                    let materials = item
                        .components()
                        .iter()
                        .filter_map(|comp| match comp.kind() {
                            ItemKind::Ingredient { .. } => Some(comp.kind()),
                            _ => None,
                        });
                    // TODO: Better handle multiple materials
                    let material_name =
                        if let Some(ItemKind::Ingredient { descriptor, .. }) = materials.last() {
                            descriptor
                        } else {
                            "Modular"
                        };
                    Cow::Owned(format!("{} {}", material_name, arg1))
                },
                ModularComponentKind::Held => Cow::Borrowed(arg1),
            }
        },
        _ => Cow::Borrowed("Modular Item"),
    }
}
