use crate::{
    assets,
    comp::{self, buff::BuffKind, inventory::item::try_all_item_defs, AdminRole as Role, Skill},
    generation::try_all_entity_configs,
    npc, terrain,
};
use assets::AssetExt;
use hashbrown::HashMap;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::{
    fmt::{self, Display},
    str::FromStr,
};
use strum::IntoEnumIterator;
use tracing::warn;

/// Struct representing a command that a user can run from server chat.
pub struct ChatCommandData {
    /// A list of arguments useful for both tab completion and parsing
    pub args: Vec<ArgumentSpec>,
    /// A one-line message that explains what the command does
    pub description: &'static str,
    /// Whether the command requires administrator permissions.
    pub needs_role: Option<Role>,
}

impl ChatCommandData {
    pub fn new(
        args: Vec<ArgumentSpec>,
        description: &'static str,
        needs_role: Option<Role>,
    ) -> Self {
        Self {
            args,
            description,
            needs_role,
        }
    }
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub enum KitSpec {
    Item(String),
    ModularWeapon {
        tool: comp::tool::ToolKind,
        material: comp::item::Material,
    },
}
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct KitManifest(pub HashMap<String, Vec<(KitSpec, u32)>>);
impl assets::Asset for KitManifest {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct SkillPresetManifest(pub HashMap<String, Vec<(Skill, u8)>>);
impl assets::Asset for SkillPresetManifest {
    type Loader = assets::RonLoader;

    const EXTENSION: &'static str = "ron";
}

pub const KIT_MANIFEST_PATH: &str = "server.manifests.kits";
pub const PRESET_MANIFEST_PATH: &str = "server.manifests.presets";

lazy_static! {
    static ref ALIGNMENTS: Vec<String> = vec!["wild", "enemy", "npc", "pet"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    static ref SKILL_TREES: Vec<String> = vec!["general", "sword", "axe", "hammer", "bow", "staff", "sceptre", "mining"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    /// TODO: Make this use hot-reloading
    static ref ENTITIES: Vec<String> = {
        let npc_names = &*npc::NPC_NAMES.read();
        let mut souls = Vec::new();
        macro_rules! push_souls {
            ($species:tt) => {
                for s in comp::$species::ALL_SPECIES.iter() {
                    souls.push(npc_names.$species.species[s].keyword.clone())
                }
            };
            ($base:tt, $($species:tt),+ $(,)?) => {
                push_souls!($base);
                push_souls!($($species),+);
            }
        }
        for npc in npc::ALL_NPCS.iter() {
            souls.push(npc_names[*npc].keyword.clone())
        }

        // See `[AllBodies](crate::comp::body::AllBodies)`
        push_souls!(
            humanoid,
            quadruped_small,
            quadruped_medium,
            quadruped_low,
            bird_medium,
            bird_large,
            fish_small,
            fish_medium,
            biped_small,
            biped_large,
            theropod,
            dragon,
            golem,
            arthropod,
        );

        souls
    };
    static ref OBJECTS: Vec<String> = comp::object::ALL_OBJECTS
        .iter()
        .map(|o| o.to_string().to_string())
        .collect();
    static ref TIMES: Vec<String> = vec![
        "midnight", "night", "dawn", "morning", "day", "noon", "dusk"
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    static ref WEATHERS: Vec<String> = vec![
        "clear", "cloudy", "rain", "wind", "storm"
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    pub static ref BUFF_PARSER: HashMap<String, BuffKind> = {
        let string_from_buff = |kind| match kind {
            BuffKind::Burning => "burning",
            BuffKind::Regeneration => "regeration",
            BuffKind::Saturation => "saturation",
            BuffKind::Bleeding => "bleeding",
            BuffKind::Cursed => "cursed",
            BuffKind::Potion => "potion",
            BuffKind::CampfireHeal => "campfire_heal",
            BuffKind::IncreaseMaxEnergy => "increase_max_energy",
            BuffKind::IncreaseMaxHealth => "increase_max_health",
            BuffKind::Invulnerability => "invulnerability",
            BuffKind::ProtectingWard => "protecting_ward",
            BuffKind::Frenzied => "frenzied",
            BuffKind::Crippled => "crippled",
            BuffKind::Frozen => "frozen",
            BuffKind::Wet => "wet",
            BuffKind::Ensnared => "ensnared",
            BuffKind::Poisoned => "poisoned",
            BuffKind::Hastened => "hastened",
        };
        let mut buff_parser = HashMap::new();
        BuffKind::iter().for_each(|kind| {buff_parser.insert(string_from_buff(kind).to_string(), kind);});
        buff_parser
    };

    pub static ref BUFF_PACK: Vec<String> = {
        let mut buff_pack: Vec<_> = BUFF_PARSER.keys().cloned().collect();
        // Remove invulnerability as it removes debuffs
        buff_pack.retain(|kind| kind != "invulnerability");
        buff_pack
    };

    static ref BUFFS: Vec<String> = {
        let mut buff_pack: Vec<_> = BUFF_PARSER.keys().cloned().collect();
        // Add all as valid command
        buff_pack.push("all".to_string());
        buff_pack
    };

    static ref BLOCK_KINDS: Vec<String> = terrain::block::BlockKind::iter()
        .map(|bk| bk.to_string())
        .collect();

    static ref SPRITE_KINDS: Vec<String> = terrain::sprite::SPRITE_KINDS
        .keys()
        .cloned()
        .collect();

    static ref ROLES: Vec<String> = ["admin", "moderator"].iter().copied().map(Into::into).collect();

    /// List of item specifiers. Useful for tab completing
    pub static ref ITEM_SPECS: Vec<String> = {
        let mut items = try_all_item_defs()
            .unwrap_or_else(|e| {
                warn!(?e, "Failed to load item specifiers");
                Vec::new()
            });
        items.sort();
        items
    };

    /// List of all entity configs. Useful for tab completing
    static ref ENTITY_CONFIGS: Vec<String> = {
        try_all_entity_configs()
            .unwrap_or_else(|e| {
                warn!(?e, "Failed to load entity configs");
                Vec::new()
            })
    };

    pub static ref KITS: Vec<String> = {
        if let Ok(kits) = KitManifest::load(KIT_MANIFEST_PATH) {
            let mut kits = kits.read().0.keys().cloned().collect::<Vec<String>>();
            kits.sort();
            kits
        } else {
            Vec::new()
        }
    };

    static ref PRESETS: HashMap<String, Vec<(Skill, u8)>> = {
        if let Ok(presets) = SkillPresetManifest::load(PRESET_MANIFEST_PATH) {
            presets.read().0.clone()
        } else {
            warn!("Error while loading presets");
            HashMap::new()
        }
    };

    static ref PRESET_LIST: Vec<String> = {
        let mut preset_list: Vec<String> = PRESETS.keys().cloned().collect();
        preset_list.push("clear".to_owned());

        preset_list
    };
}

// Please keep this sorted alphabetically :-)
#[derive(Copy, Clone, strum::EnumIter)]
pub enum ServerChatCommand {
    Adminify,
    Airship,
    Alias,
    ApplyBuff,
    Ban,
    BattleMode,
    BattleModeForce,
    Build,
    BuildAreaAdd,
    BuildAreaList,
    BuildAreaRemove,
    Campfire,
    DebugColumn,
    DisconnectAllPlayers,
    DropAll,
    Dummy,
    Explosion,
    Faction,
    GiveItem,
    Goto,
    Group,
    GroupInvite,
    GroupKick,
    GroupLeave,
    GroupPromote,
    Health,
    Help,
    Home,
    JoinFaction,
    Jump,
    Kick,
    Kill,
    KillNpcs,
    Kit,
    Lantern,
    Light,
    MakeBlock,
    MakeNpc,
    MakeSprite,
    Motd,
    Object,
    PermitBuild,
    Players,
    Region,
    ReloadChunks,
    RemoveLights,
    RevokeBuild,
    RevokeBuildAll,
    Safezone,
    Say,
    ServerPhysics,
    SetMotd,
    Ship,
    Site,
    SkillPoint,
    SkillPreset,
    Spawn,
    Sudo,
    Tell,
    Time,
    Tp,
    Unban,
    Version,
    Waypoint,
    Whitelist,
    Wiring,
    World,
    MakeVolume,
    Location,
    CreateLocation,
    DeleteLocation,
    WeatherZone,
    Lightning,
}

impl ServerChatCommand {
    pub fn data(&self) -> ChatCommandData {
        use ArgumentSpec::*;
        use Requirement::*;
        use Role::*;
        let cmd = ChatCommandData::new;
        match self {
            ServerChatCommand::Adminify => cmd(
                vec![PlayerName(Required), Enum("role", ROLES.clone(), Optional)],
                "Temporarily gives a player a restricted admin role or removes the current one \
                 (if not given)",
                Some(Admin),
            ),
            ServerChatCommand::Airship => cmd(
                vec![Float("destination_degrees_ccw_of_east", 90.0, Optional)],
                "Spawns an airship",
                Some(Admin),
            ),
            ServerChatCommand::Alias => cmd(
                vec![Any("name", Required)],
                "Change your alias",
                Some(Moderator),
            ),
            ServerChatCommand::ApplyBuff => cmd(
                vec![
                    Enum("buff", BUFFS.clone(), Required),
                    Float("strength", 0.01, Optional),
                    Float("duration", 10.0, Optional),
                ],
                "Cast a buff on player",
                Some(Admin),
            ),
            ServerChatCommand::Ban => cmd(
                vec![
                    PlayerName(Required),
                    Boolean("overwrite", "true".to_string(), Optional),
                    Any("ban duration", Optional),
                    Message(Optional),
                ],
                "Ban a player with a given username, for a given duration (if provided).  Pass \
                 true for overwrite to alter an existing ban..",
                Some(Moderator),
            ),
            #[rustfmt::skip]
            ServerChatCommand::BattleMode => cmd(
                vec![Enum(
                    "battle mode",
                    vec!["pvp".to_owned(), "pve".to_owned()],
                    Optional,
                )],
                "Set your battle mode to:\n\
                * pvp (player vs player)\n\
                * pve (player vs environment).\n\
                If called without arguments will show current battle mode.",
                None,

            ),
            ServerChatCommand::BattleModeForce => cmd(
                vec![Enum(
                    "battle mode",
                    vec!["pvp".to_owned(), "pve".to_owned()],
                    Required,
                )],
                "Change your battle mode flag without any checks",
                Some(Admin),
            ),
            ServerChatCommand::Build => cmd(vec![], "Toggles build mode on and off", None),
            ServerChatCommand::BuildAreaAdd => cmd(
                vec![
                    Any("name", Required),
                    Integer("xlo", 0, Required),
                    Integer("xhi", 10, Required),
                    Integer("ylo", 0, Required),
                    Integer("yhi", 10, Required),
                    Integer("zlo", 0, Required),
                    Integer("zhi", 10, Required),
                ],
                "Adds a new build area",
                Some(Admin),
            ),
            ServerChatCommand::BuildAreaList => cmd(vec![], "List all build areas", Some(Admin)),
            ServerChatCommand::BuildAreaRemove => cmd(
                vec![Any("name", Required)],
                "Removes specified build area",
                Some(Admin),
            ),
            ServerChatCommand::Campfire => cmd(vec![], "Spawns a campfire", Some(Admin)),
            ServerChatCommand::DebugColumn => cmd(
                vec![Integer("x", 15000, Required), Integer("y", 15000, Required)],
                "Prints some debug information about a column",
                Some(Moderator),
            ),
            ServerChatCommand::DisconnectAllPlayers => cmd(
                vec![Any("confirm", Required)],
                "Disconnects all players from the server",
                Some(Admin),
            ),
            ServerChatCommand::DropAll => cmd(
                vec![],
                "Drops all your items on the ground",
                Some(Moderator),
            ),
            ServerChatCommand::Dummy => cmd(vec![], "Spawns a training dummy", Some(Admin)),
            ServerChatCommand::Explosion => cmd(
                vec![Float("radius", 5.0, Required)],
                "Explodes the ground around you",
                Some(Admin),
            ),
            ServerChatCommand::Faction => cmd(
                vec![Message(Optional)],
                "Send messages to your faction",
                None,
            ),
            ServerChatCommand::GiveItem => cmd(
                vec![
                    Enum("item", ITEM_SPECS.clone(), Required),
                    Integer("num", 1, Optional),
                ],
                "Give yourself some items.\nFor an example or to auto complete use Tab.",
                Some(Admin),
            ),
            ServerChatCommand::Goto => cmd(
                vec![
                    Float("x", 0.0, Required),
                    Float("y", 0.0, Required),
                    Float("z", 0.0, Required),
                ],
                "Teleport to a position",
                Some(Admin),
            ),
            ServerChatCommand::Group => {
                cmd(vec![Message(Optional)], "Send messages to your group", None)
            },
            ServerChatCommand::GroupInvite => cmd(
                vec![PlayerName(Required)],
                "Invite a player to join a group",
                None,
            ),
            ServerChatCommand::GroupKick => cmd(
                vec![PlayerName(Required)],
                "Remove a player from a group",
                None,
            ),
            ServerChatCommand::GroupLeave => cmd(vec![], "Leave the current group", None),
            ServerChatCommand::GroupPromote => cmd(
                vec![PlayerName(Required)],
                "Promote a player to group leader",
                None,
            ),
            ServerChatCommand::Health => cmd(
                vec![Integer("hp", 100, Required)],
                "Set your current health",
                Some(Admin),
            ),
            ServerChatCommand::Help => ChatCommandData::new(
                vec![Command(Optional)],
                "Display information about commands",
                None,
            ),
            ServerChatCommand::Home => cmd(vec![], "Return to the home town", Some(Moderator)),
            ServerChatCommand::JoinFaction => ChatCommandData::new(
                vec![Any("faction", Optional)],
                "Join/leave the specified faction",
                None,
            ),
            ServerChatCommand::Jump => cmd(
                vec![
                    Float("x", 0.0, Required),
                    Float("y", 0.0, Required),
                    Float("z", 0.0, Required),
                ],
                "Offset your current position",
                Some(Admin),
            ),
            ServerChatCommand::Kick => cmd(
                vec![PlayerName(Required), Message(Optional)],
                "Kick a player with a given username",
                Some(Moderator),
            ),
            ServerChatCommand::Kill => cmd(vec![], "Kill yourself", None),
            ServerChatCommand::KillNpcs => cmd(vec![], "Kill the NPCs", Some(Admin)),
            ServerChatCommand::Kit => cmd(
                vec![Enum("kit_name", KITS.to_vec(), Required)],
                "Place a set of items into your inventory.",
                Some(Admin),
            ),
            ServerChatCommand::Lantern => cmd(
                vec![
                    Float("strength", 5.0, Required),
                    Float("r", 1.0, Optional),
                    Float("g", 1.0, Optional),
                    Float("b", 1.0, Optional),
                ],
                "Change your lantern's strength and color",
                Some(Admin),
            ),
            ServerChatCommand::Light => cmd(
                vec![
                    Float("r", 1.0, Optional),
                    Float("g", 1.0, Optional),
                    Float("b", 1.0, Optional),
                    Float("x", 0.0, Optional),
                    Float("y", 0.0, Optional),
                    Float("z", 0.0, Optional),
                    Float("strength", 5.0, Optional),
                ],
                "Spawn entity with light",
                Some(Admin),
            ),
            ServerChatCommand::MakeBlock => cmd(
                vec![
                    Enum("block", BLOCK_KINDS.clone(), Required),
                    Integer("r", 255, Optional),
                    Integer("g", 255, Optional),
                    Integer("b", 255, Optional),
                ],
                "Make a block at your location with a color",
                Some(Admin),
            ),
            ServerChatCommand::MakeNpc => cmd(
                vec![
                    Enum("entity_config", ENTITY_CONFIGS.clone(), Required),
                    Integer("num", 1, Optional),
                ],
                "Spawn entity from config near you.\nFor an example or to auto complete use Tab.",
                Some(Admin),
            ),
            ServerChatCommand::MakeSprite => cmd(
                vec![Enum("sprite", SPRITE_KINDS.clone(), Required)],
                "Make a sprite at your location",
                Some(Admin),
            ),
            ServerChatCommand::Motd => {
                cmd(vec![Message(Optional)], "View the server description", None)
            },
            ServerChatCommand::Object => cmd(
                vec![Enum("object", OBJECTS.clone(), Required)],
                "Spawn an object",
                Some(Admin),
            ),
            ServerChatCommand::PermitBuild => cmd(
                vec![Any("area_name", Required)],
                "Grants player a bounded box they can build in",
                Some(Admin),
            ),
            ServerChatCommand::Players => cmd(vec![], "Lists players currently online", None),
            ServerChatCommand::ReloadChunks => cmd(
                vec![],
                "Reloads all chunks loaded on the server",
                Some(Admin),
            ),
            ServerChatCommand::RemoveLights => cmd(
                vec![Float("radius", 20.0, Optional)],
                "Removes all lights spawned by players",
                Some(Admin),
            ),
            ServerChatCommand::RevokeBuild => cmd(
                vec![Any("area_name", Required)],
                "Revokes build area permission for player",
                Some(Admin),
            ),
            ServerChatCommand::RevokeBuildAll => cmd(
                vec![],
                "Revokes all build area permissions for player",
                Some(Admin),
            ),
            ServerChatCommand::Region => cmd(
                vec![Message(Optional)],
                "Send messages to everyone in your region of the world",
                None,
            ),
            ServerChatCommand::Safezone => cmd(
                vec![Float("range", 100.0, Optional)],
                "Creates a safezone",
                Some(Moderator),
            ),
            ServerChatCommand::Say => cmd(
                vec![Message(Optional)],
                "Send messages to everyone within shouting distance",
                None,
            ),
            ServerChatCommand::ServerPhysics => cmd(
                vec![
                    PlayerName(Required),
                    Boolean("enabled", "true".to_string(), Optional),
                ],
                "Set/unset server-authoritative physics for an account",
                Some(Moderator),
            ),
            ServerChatCommand::SetMotd => cmd(
                vec![Message(Optional)],
                "Set the server description",
                Some(Admin),
            ),
            ServerChatCommand::Ship => cmd(
                vec![Float("destination_degrees_ccw_of_east", 90.0, Optional)],
                "Spawns a ship",
                Some(Admin),
            ),
            // Uses Message because site names can contain spaces,
            // which would be assumed to be separators otherwise
            ServerChatCommand::Site => cmd(
                vec![SiteName(Required)],
                "Teleport to a site",
                Some(Moderator),
            ),
            ServerChatCommand::SkillPoint => cmd(
                vec![
                    Enum("skill tree", SKILL_TREES.clone(), Required),
                    Integer("amount", 1, Optional),
                ],
                "Give yourself skill points for a particular skill tree",
                Some(Admin),
            ),
            ServerChatCommand::SkillPreset => cmd(
                vec![Enum("preset_name", PRESET_LIST.to_vec(), Required)],
                "Gives your character desired skills.",
                Some(Admin),
            ),
            ServerChatCommand::Spawn => cmd(
                vec![
                    Enum("alignment", ALIGNMENTS.clone(), Required),
                    Enum("entity", ENTITIES.clone(), Required),
                    Integer("amount", 1, Optional),
                    Boolean("ai", "true".to_string(), Optional),
                ],
                "Spawn a test entity",
                Some(Admin),
            ),
            ServerChatCommand::Sudo => cmd(
                vec![PlayerName(Required), SubCommand],
                "Run command as if you were another player",
                Some(Moderator),
            ),
            ServerChatCommand::Tell => cmd(
                vec![PlayerName(Required), Message(Optional)],
                "Send a message to another player",
                None,
            ),
            ServerChatCommand::Time => cmd(
                vec![Enum("time", TIMES.clone(), Optional)],
                "Set the time of day",
                Some(Admin),
            ),
            ServerChatCommand::Tp => cmd(
                vec![PlayerName(Optional)],
                "Teleport to another player",
                Some(Moderator),
            ),
            ServerChatCommand::Unban => cmd(
                vec![PlayerName(Required)],
                "Remove the ban for the given username",
                Some(Moderator),
            ),
            ServerChatCommand::Version => cmd(vec![], "Prints server version", None),
            ServerChatCommand::Waypoint => cmd(
                vec![],
                "Set your waypoint to your current position",
                Some(Admin),
            ),
            ServerChatCommand::Wiring => cmd(vec![], "Create wiring element", Some(Admin)),
            ServerChatCommand::Whitelist => cmd(
                vec![Any("add/remove", Required), PlayerName(Required)],
                "Adds/removes username to whitelist",
                Some(Moderator),
            ),
            ServerChatCommand::World => cmd(
                vec![Message(Optional)],
                "Send messages to everyone on the server",
                None,
            ),
            ServerChatCommand::MakeVolume => {
                cmd(vec![], "Create a volume (experimental)", Some(Admin))
            },
            ServerChatCommand::Location => {
                cmd(vec![Any("name", Required)], "Teleport to a location", None)
            },
            ServerChatCommand::CreateLocation => cmd(
                vec![Any("name", Required)],
                "Create a location at the current position",
                Some(Moderator),
            ),
            ServerChatCommand::DeleteLocation => cmd(
                vec![Any("name", Required)],
                "Delete a location",
                Some(Moderator),
            ),
            ServerChatCommand::WeatherZone => cmd(
                vec![
                    Enum("weather kind", WEATHERS.clone(), Required),
                    Float("radius", 500.0, Optional),
                    Float("time", 300.0, Optional),
                ],
                "Create a weather zone",
                Some(Admin),
            ),
            ServerChatCommand::Lightning => {
                cmd(vec![], "Lightning strike at current position", Some(Admin))
            },
        }
    }

    /// The keyword used to invoke the command, omitting the prefix.
    pub fn keyword(&self) -> &'static str {
        match self {
            ServerChatCommand::Adminify => "adminify",
            ServerChatCommand::Airship => "airship",
            ServerChatCommand::Alias => "alias",
            ServerChatCommand::ApplyBuff => "buff",
            ServerChatCommand::Ban => "ban",
            ServerChatCommand::BattleMode => "battlemode",
            ServerChatCommand::BattleModeForce => "battlemode_force",
            ServerChatCommand::Build => "build",
            ServerChatCommand::BuildAreaAdd => "build_area_add",
            ServerChatCommand::BuildAreaList => "build_area_list",
            ServerChatCommand::BuildAreaRemove => "build_area_remove",
            ServerChatCommand::Campfire => "campfire",
            ServerChatCommand::DebugColumn => "debug_column",
            ServerChatCommand::DisconnectAllPlayers => "disconnect_all_players",
            ServerChatCommand::DropAll => "dropall",
            ServerChatCommand::Dummy => "dummy",
            ServerChatCommand::Explosion => "explosion",
            ServerChatCommand::Faction => "faction",
            ServerChatCommand::GiveItem => "give_item",
            ServerChatCommand::Goto => "goto",
            ServerChatCommand::Group => "group",
            ServerChatCommand::GroupInvite => "group_invite",
            ServerChatCommand::GroupKick => "group_kick",
            ServerChatCommand::GroupPromote => "group_promote",
            ServerChatCommand::GroupLeave => "group_leave",
            ServerChatCommand::Health => "health",
            ServerChatCommand::JoinFaction => "join_faction",
            ServerChatCommand::Help => "help",
            ServerChatCommand::Home => "home",
            ServerChatCommand::Jump => "jump",
            ServerChatCommand::Kick => "kick",
            ServerChatCommand::Kill => "kill",
            ServerChatCommand::Kit => "kit",
            ServerChatCommand::KillNpcs => "kill_npcs",
            ServerChatCommand::Lantern => "lantern",
            ServerChatCommand::Light => "light",
            ServerChatCommand::MakeBlock => "make_block",
            ServerChatCommand::MakeNpc => "make_npc",
            ServerChatCommand::MakeSprite => "make_sprite",
            ServerChatCommand::Motd => "motd",
            ServerChatCommand::Object => "object",
            ServerChatCommand::PermitBuild => "permit_build",
            ServerChatCommand::Players => "players",
            ServerChatCommand::Region => "region",
            ServerChatCommand::ReloadChunks => "reload_chunks",
            ServerChatCommand::RemoveLights => "remove_lights",
            ServerChatCommand::RevokeBuild => "revoke_build",
            ServerChatCommand::RevokeBuildAll => "revoke_build_all",
            ServerChatCommand::Safezone => "safezone",
            ServerChatCommand::Say => "say",
            ServerChatCommand::ServerPhysics => "server_physics",
            ServerChatCommand::SetMotd => "set_motd",
            ServerChatCommand::Ship => "ship",
            ServerChatCommand::Site => "site",
            ServerChatCommand::SkillPoint => "skill_point",
            ServerChatCommand::SkillPreset => "skill_preset",
            ServerChatCommand::Spawn => "spawn",
            ServerChatCommand::Sudo => "sudo",
            ServerChatCommand::Tell => "tell",
            ServerChatCommand::Time => "time",
            ServerChatCommand::Tp => "tp",
            ServerChatCommand::Unban => "unban",
            ServerChatCommand::Version => "version",
            ServerChatCommand::Waypoint => "waypoint",
            ServerChatCommand::Wiring => "wiring",
            ServerChatCommand::Whitelist => "whitelist",
            ServerChatCommand::World => "world",
            ServerChatCommand::MakeVolume => "make_volume",
            ServerChatCommand::Location => "location",
            ServerChatCommand::CreateLocation => "create_location",
            ServerChatCommand::DeleteLocation => "delete_location",
            ServerChatCommand::WeatherZone => "weather_zone",
            ServerChatCommand::Lightning => "lightning",
        }
    }

    /// The short keyword used to invoke the command, omitting the leading '/'.
    /// Returns None if the command doesn't have a short keyword
    pub fn short_keyword(&self) -> Option<&'static str> {
        Some(match self {
            ServerChatCommand::Faction => "f",
            ServerChatCommand::Group => "g",
            ServerChatCommand::Region => "r",
            ServerChatCommand::Say => "s",
            ServerChatCommand::Tell => "t",
            ServerChatCommand::World => "w",
            _ => return None,
        })
    }

    /// Produce an iterator over all the available commands
    pub fn iter() -> impl Iterator<Item = Self> { <Self as strum::IntoEnumIterator>::iter() }

    /// A message that explains what the command does
    pub fn help_string(&self) -> String {
        let data = self.data();
        let usage = std::iter::once(format!("/{}", self.keyword()))
            .chain(data.args.iter().map(|arg| arg.usage_string()))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{}: {}", usage, data.description)
    }

    /// Produce an iterator that first goes over all the short keywords
    /// and their associated commands and then iterates over all the normal
    /// keywords with their associated commands
    pub fn iter_with_keywords() -> impl Iterator<Item = (&'static str, Self)> {
        Self::iter()
        // Go through all the shortcuts first
        .filter_map(|c| c.short_keyword().map(|s| (s, c)))
        .chain(Self::iter().map(|c| (c.keyword(), c)))
    }

    pub fn needs_role(&self) -> Option<comp::AdminRole> { self.data().needs_role }

    /// Returns a format string for parsing arguments with scan_fmt
    pub fn arg_fmt(&self) -> String {
        self.data()
            .args
            .iter()
            .map(|arg| match arg {
                ArgumentSpec::PlayerName(_) => "{}",
                ArgumentSpec::SiteName(_) => "{/.*/}",
                ArgumentSpec::Float(_, _, _) => "{}",
                ArgumentSpec::Integer(_, _, _) => "{d}",
                ArgumentSpec::Any(_, _) => "{}",
                ArgumentSpec::Command(_) => "{}",
                ArgumentSpec::Message(_) => "{/.*/}",
                ArgumentSpec::SubCommand => "{} {/.*/}",
                ArgumentSpec::Enum(_, _, _) => "{}",
                ArgumentSpec::Boolean(_, _, _) => "{}",
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Display for ServerChatCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "{}", self.keyword())
    }
}

impl FromStr for ServerChatCommand {
    type Err = ();

    fn from_str(keyword: &str) -> Result<ServerChatCommand, ()> {
        Self::iter()
        // Go through all the shortcuts first
        .filter_map(|c| c.short_keyword().map(|s| (s, c)))
        .chain(Self::iter().map(|c| (c.keyword(), c)))
            // Find command with matching string as keyword
            .find_map(|(kwd, command)| (kwd == keyword).then(|| command))
            // Return error if not found
            .ok_or(())
    }
}

#[derive(Eq, PartialEq, Debug)]
pub enum Requirement {
    Required,
    Optional,
}

/// Representation for chat command arguments
pub enum ArgumentSpec {
    /// The argument refers to a player by alias
    PlayerName(Requirement),
    // The argument refers to a site, by name.
    SiteName(Requirement),
    /// The argument is a float. The associated values are
    /// * label
    /// * suggested tab-completion
    /// * whether it's optional
    Float(&'static str, f32, Requirement),
    /// The argument is an integer. The associated values are
    /// * label
    /// * suggested tab-completion
    /// * whether it's optional
    Integer(&'static str, i32, Requirement),
    /// The argument is any string that doesn't contain spaces
    Any(&'static str, Requirement),
    /// The argument is a command name (such as in /help)
    Command(Requirement),
    /// This is the final argument, consuming all characters until the end of
    /// input.
    Message(Requirement),
    /// This command is followed by another command (such as in /sudo)
    SubCommand,
    /// The argument is likely an enum. The associated values are
    /// * label
    /// * Predefined string completions
    /// * whether it's optional
    Enum(&'static str, Vec<String>, Requirement),
    /// The argument is likely a boolean. The associated values are
    /// * label
    /// * suggested tab-completion
    /// * whether it's optional
    Boolean(&'static str, String, Requirement),
}

impl ArgumentSpec {
    pub fn usage_string(&self) -> String {
        match self {
            ArgumentSpec::PlayerName(req) => {
                if &Requirement::Required == req {
                    "<player>".to_string()
                } else {
                    "[player]".to_string()
                }
            },
            ArgumentSpec::SiteName(req) => {
                if &Requirement::Required == req {
                    "<site>".to_string()
                } else {
                    "[site]".to_string()
                }
            },
            ArgumentSpec::Float(label, _, req) => {
                if &Requirement::Required == req {
                    format!("<{}>", label)
                } else {
                    format!("[{}]", label)
                }
            },
            ArgumentSpec::Integer(label, _, req) => {
                if &Requirement::Required == req {
                    format!("<{}>", label)
                } else {
                    format!("[{}]", label)
                }
            },
            ArgumentSpec::Any(label, req) => {
                if &Requirement::Required == req {
                    format!("<{}>", label)
                } else {
                    format!("[{}]", label)
                }
            },
            ArgumentSpec::Command(req) => {
                if &Requirement::Required == req {
                    "<[/]command>".to_string()
                } else {
                    "[[/]command]".to_string()
                }
            },
            ArgumentSpec::Message(req) => {
                if &Requirement::Required == req {
                    "<message>".to_string()
                } else {
                    "[message]".to_string()
                }
            },
            ArgumentSpec::SubCommand => "<[/]command> [args...]".to_string(),
            ArgumentSpec::Enum(label, _, req) => {
                if &Requirement::Required == req {
                    format!("<{}>", label)
                } else {
                    format!("[{}]", label)
                }
            },
            ArgumentSpec::Boolean(label, _, req) => {
                if &Requirement::Required == req {
                    format!("<{}>", label)
                } else {
                    format!("[{}]", label)
                }
            },
        }
    }
}

/// Parse a series of command arguments into values, including collecting all
/// trailing arguments.
#[macro_export]
macro_rules! parse_cmd_args {
    ($args:expr, $($t:ty),* $(, ..$tail:ty)? $(,)?) => {
        {
            let mut args = $args.into_iter().peekable();
            (
                // We only consume the input argument when parsing is successful. If this fails, we
                // will then attempt to parse it as the next argument type. This is done regardless
                // of whether the argument is optional because that information is not available
                // here. Nevertheless, if the caller only precedes to use the parsed arguments when
                // all required arguments parse successfully to `Some(val)` this should not create
                // any unexpected behavior.
                //
                // This does mean that optional arguments will be included in the trailing args or
                // that one optional arg could be interpreted as another, if the user makes a
                // mistake that causes an optional arg to fail to parse. But there is no way to
                // discern this in the current model with the optional args and trailing arg being
                // solely position based.
                $({
                    let parsed = args.peek().and_then(|s| s.parse::<$t>().ok());
                    // Consume successfully parsed arg.
                    if parsed.is_some() { args.next(); }
                    parsed
                }),*
                $(, args.map(|s| s.to_string()).collect::<$tail>())?
            )
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comp::Item;

    #[test]
    fn test_loading_skill_presets() { SkillPresetManifest::load_expect(PRESET_MANIFEST_PATH); }

    #[test]
    fn test_load_kits() {
        let kits = KitManifest::load_expect(KIT_MANIFEST_PATH).read();
        let mut rng = rand::thread_rng();
        for kit in kits.0.values() {
            for (item_id, _) in kit.iter() {
                match item_id {
                    KitSpec::Item(item_id) => {
                        Item::new_from_asset_expect(item_id);
                    },
                    KitSpec::ModularWeapon { tool, material } => {
                        comp::item::modular::random_weapon(*tool, *material, None, &mut rng)
                            .unwrap_or_else(|_| {
                                panic!(
                                    "Failed to synthesize a modular {tool:?} made of {material:?}."
                                )
                            });
                    },
                }
            }
        }
    }
}
