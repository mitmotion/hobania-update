use crate::hud;
use common::character::CharacterId;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
use tracing::warn;

/// Represents a character in the profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct CharacterProfile {
    /// Array representing a character's hotbar.
    pub hotbar_slots: [Option<hud::HotbarSlotContents>; 10],
}

const fn default_slots() -> [Option<hud::HotbarSlotContents>; 10] {
    [None, None, None, None, None, None, None, None, None, None]
}

impl Default for CharacterProfile {
    fn default() -> Self {
        CharacterProfile {
            hotbar_slots: default_slots(),
        }
    }
}

/// Represents a server in the profile.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerProfile {
    /// A map of character's by id to their CharacterProfile.
    pub characters: HashMap<CharacterId, CharacterProfile>,
    // Selected character in the chararacter selection screen
    pub selected_character: Option<CharacterId>,
}

impl Default for ServerProfile {
    fn default() -> Self {
        ServerProfile {
            characters: HashMap::new(),
            selected_character: None,
        }
    }
}

/// `Profile` contains everything that can be configured in the profile.ron
///
/// Initially it is just for persisting things that don't belong in
/// settings.ron - like the state of hotbar and any other character level
/// configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    pub servers: HashMap<String, ServerProfile>,
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            servers: HashMap::new(),
        }
    }
}

impl Profile {
    /// Load the profile.ron file from the standard path or create it.
    pub fn load(config_dir: &Path) -> Self {
        let path = Profile::get_path(config_dir);

        if let Ok(file) = fs::File::open(&path) {
            match ron::de::from_reader(file) {
                Ok(profile) => return profile,
                Err(e) => {
                    warn!(
                        ?e,
                        ?path,
                        "Failed to parse profile file! Falling back to default."
                    );
                    // Rename the corrupted profile file.
                    let new_path = path.with_extension("invalid.ron");
                    if let Err(e) = std::fs::rename(path.clone(), new_path.clone()) {
                        warn!(?e, ?path, ?new_path, "Failed to rename profile file.");
                    }
                },
            }
        }
        // This is reached if either:
        // - The file can't be opened (presumably it doesn't exist)
        // - Or there was an error parsing the file
        let default_profile = Self::default();
        default_profile.save_to_file_warn(config_dir);
        default_profile
    }

    /// Save the current profile to disk, warn on failure.
    pub fn save_to_file_warn(&self, config_dir: &Path) {
        if let Err(e) = self.save_to_file(config_dir) {
            warn!(?e, "Failed to save profile");
        }
    }

    /// Get the hotbar_slots for the requested character_id.
    ///
    /// If the server or character does not exist then the default hotbar_slots
    /// (empty) is returned.
    ///
    /// # Arguments
    ///
    /// * server - current server the character is on.
    /// * character_id - id of the character.
    pub fn get_hotbar_slots(
        &self,
        server: &str,
        character_id: CharacterId,
    ) -> [Option<hud::HotbarSlotContents>; 10] {
        self.servers
            .get(server)
            .and_then(|s| s.characters.get(&character_id))
            .map(|c| c.hotbar_slots.clone())
            .unwrap_or_else(default_slots)
    }

    /// Set the hotbar_slots for the requested character_id.
    ///
    /// If the server or character does not exist then the appropriate fields
    /// will be initialised and the slots added.
    ///
    /// # Arguments
    ///
    /// * server - current server the character is on.
    /// * character_id - id of the character.
    /// * slots - array of hotbar_slots to save.
    pub fn set_hotbar_slots(
        &mut self,
        server: &str,
        character_id: CharacterId,
        slots: [Option<hud::HotbarSlotContents>; 10],
    ) {
        self.servers
            .entry(server.to_string())
            .or_insert(ServerProfile::default())
            // Get or update the CharacterProfile.
            .characters
            .entry(character_id)
            .or_insert(CharacterProfile::default())
            .hotbar_slots = slots;
    }

    /// Get the selected_character for the provided server.
    ///
    /// if the server does not exist then the default selected_character (None)
    /// is returned.
    ///
    /// # Arguments
    ///
    /// * server - current server the character is on.
    pub fn get_selected_character(&self, server: &str) -> Option<CharacterId> {
        self.servers
            .get(server)
            .map(|s| s.selected_character)
            .unwrap_or_default()
    }

    /// Set the selected_character for the provided server.
    ///
    /// If the server does not exist then the appropriate fields
    /// will be initialised and the selected_character added.
    ///
    /// # Arguments
    ///
    /// * server - current server the character is on.
    /// * selected_character - option containing selected character ID
    pub fn set_selected_character(
        &mut self,
        server: &str,
        selected_character: Option<CharacterId>,
    ) {
        self.servers
            .entry(server.to_string())
            .or_insert(ServerProfile::default())
            .selected_character = selected_character;
    }

    /// Save the current profile to disk.
    fn save_to_file(&self, config_dir: &Path) -> std::io::Result<()> {
        let path = Profile::get_path(config_dir);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let mut config_file = fs::File::create(path)?;

        let s: &str = &ron::ser::to_string_pretty(self, ron::ser::PrettyConfig::default()).unwrap();
        config_file.write_all(s.as_bytes()).unwrap();
        Ok(())
    }

    fn get_path(config_dir: &Path) -> PathBuf { config_dir.join("profile.ron") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_slots_with_empty_profile() {
        let profile = Profile::default();
        let slots = profile.get_hotbar_slots("TestServer", 12345);
        assert_eq!(slots, [(); 10].map(|()| None))
    }

    #[test]
    fn test_set_slots_with_empty_profile() {
        let mut profile = Profile::default();
        let slots = [(); 10].map(|()| None);
        profile.set_hotbar_slots("TestServer", 12345, slots);
    }
}
