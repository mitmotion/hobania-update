#[cfg(any(feature = "bin", test))]
pub mod analysis;
#[cfg(any(feature = "bin", test))]
mod gitfragments;
mod path;
mod raw;
#[cfg(any(feature = "bin", test))] pub mod stats;
pub mod verification;

//reexport
pub use path::BasePath;

use crate::path::{LANG_EXTENSION, LANG_MANIFEST_FILE};
use common_assets::{self, source::DirEntry, AssetExt, AssetGuard, AssetHandle, ReloadWatcher};
use hashbrown::{HashMap, HashSet};
use raw::{RawFragment, RawLanguage, RawManifest};
use serde::{Deserialize, Serialize};
use std::{io, path::PathBuf};
use tracing::warn;

/// The reference language, aka the more up-to-date localization data.
/// Also the default language at first startup.
pub const REFERENCE_LANG: &str = "en";

/// How a language can be described
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LanguageMetadata {
    /// A human friendly language name (e.g. "English (US)")
    pub language_name: String,

    /// A short text identifier for this language (e.g. "en_US")
    ///
    /// On the opposite of `language_name` that can change freely,
    /// `language_identifier` value shall be stable in time as it
    /// is used by setting components to store the language
    /// selected by the user.
    pub language_identifier: String,
}

/// Store font metadata
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Font {
    /// Key to retrieve the font in the asset system
    pub asset_key: String,

    /// Scale ratio to resize the UI text dynamically
    scale_ratio: f32,
}

impl Font {
    /// Scale input size to final UI size
    pub fn scale(&self, value: u32) -> u32 { (value as f32 * self.scale_ratio).round() as u32 }
}

/// Store font metadata
pub type Fonts = HashMap<String, Font>;

/// Store internationalization data
#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Language {
    /// A map storing the localized texts
    ///
    /// Localized content can be accessed using a String key.
    pub(crate) string_map: HashMap<String, String>,

    /// A map for storing variations of localized texts, for example multiple
    /// ways of saying "Help, I'm under attack". Used primarily for npc
    /// dialogue.
    pub(crate) vector_map: HashMap<String, Vec<String>>,

    /// Whether to convert the input text encoded in UTF-8
    /// into a ASCII version by using the `deunicode` crate.
    pub(crate) convert_utf8_to_ascii: bool,

    /// Font configuration is stored here
    pub(crate) fonts: Fonts,

    pub(crate) metadata: LanguageMetadata,
}

impl Language {
    /// Get a localized text from the given key
    pub fn get(&self, key: &str) -> Option<&str> { self.string_map.get(key).map(String::as_str) }

    /// Get a variation of localized text from the given key
    ///
    /// `index` should be a random number from `0` to `u16::max()`
    pub fn get_variation(&self, key: &str, index: u16) -> Option<&str> {
        self.vector_map.get(key).and_then(|v| {
            if v.is_empty() {
                None
            } else {
                Some(v[index as usize % v.len()].as_str())
            }
        })
    }
}

impl common_assets::Compound for Language {
    fn load(
        cache: common_assets::AnyCache,
        asset_key: &str,
    ) -> Result<Self, common_assets::BoxedError> {
        let manifest = cache
            .load::<RawManifest>(&[asset_key, ".", LANG_MANIFEST_FILE].concat())?
            .cloned();

        // Walk through files in the folder, collecting localization fragment to merge
        // inside the asked_localization
        let mut fragments = HashMap::new();
        for id in cache
            .load_dir::<RawFragment<String>>(asset_key, true)?
            .ids()
        {
            // Don't try to load manifests
            if let Some(id) = id.strip_suffix(LANG_MANIFEST_FILE) {
                if id.ends_with('.') {
                    continue;
                }
            }

            match cache.load(id) {
                Ok(handle) => {
                    let fragment: &RawFragment<String> = &*handle.read();

                    fragments.insert(PathBuf::from(id), fragment.clone());
                },
                Err(e) => {
                    warn!("Unable to load asset {}, error={:?}", id, e);
                },
            }
        }

        Ok(Language::from(RawLanguage {
            manifest,
            fragments,
        }))
    }
}

/// the central data structure to handle localization in veloren
// inherit Copy+Clone from AssetHandle
#[derive(Debug, Copy, Clone)]
pub struct LocalizationHandle {
    active: AssetHandle<Language>,
    watcher: ReloadWatcher,
    fallback: Option<AssetHandle<Language>>,
    pub use_english_fallback: bool,
}

// RAII guard returned from Localization::read(), resembles AssetGuard
pub struct LocalizationGuard {
    active: AssetGuard<Language>,
    fallback: Option<AssetGuard<Language>>,
}

// arbitrary choice to minimize changing all of veloren
pub type Localization = LocalizationGuard;

impl LocalizationGuard {
    /// Get a localized text from the given key
    ///
    /// First lookup is done in the active language, second in
    /// the fallback (if present).
    pub fn get_opt(&self, key: &str) -> Option<&str> {
        self.active
            .get(key)
            .or_else(|| self.fallback.as_ref().and_then(|f| f.get(key)))
    }

    /// Get a localized text from the given key
    ///
    /// First lookup is done in the active language, second in
    /// the fallback (if present).
    /// If the key is not present in the localization object
    /// then the key is returned.
    pub fn get<'a>(&'a self, key: &'a str) -> &str { self.get_opt(key).unwrap_or(key) }

    /// Get a localized text from the given key
    ///
    /// First lookup is done in the active language, second in
    /// the fallback (if present).
    pub fn get_or(&self, key: &str, fallback_key: &str) -> Option<&str> {
        self.get_opt(key).or_else(|| self.get_opt(fallback_key))
    }

    /// Get a variation of localized text from the given key
    ///
    /// `index` should be a random number from `0` to `u16::max()`
    ///
    /// If the key is not present in the localization object
    /// then the key is returned.
    pub fn get_variation<'a>(&'a self, key: &'a str, index: u16) -> &str {
        self.active.get_variation(key, index).unwrap_or_else(|| {
            self.fallback
                .as_ref()
                .and_then(|f| f.get_variation(key, index))
                .unwrap_or(key)
        })
    }

    /// Return the missing keys compared to the reference language
    fn list_missing_entries(&self) -> (HashSet<String>, HashSet<String>) {
        if let Some(ref_lang) = &self.fallback {
            let reference_string_keys: HashSet<_> = ref_lang.string_map.keys().cloned().collect();
            let string_keys: HashSet<_> = self.active.string_map.keys().cloned().collect();
            let strings = reference_string_keys
                .difference(&string_keys)
                .cloned()
                .collect();

            let reference_vector_keys: HashSet<_> = ref_lang.vector_map.keys().cloned().collect();
            let vector_keys: HashSet<_> = self.active.vector_map.keys().cloned().collect();
            let vectors = reference_vector_keys
                .difference(&vector_keys)
                .cloned()
                .collect();

            (strings, vectors)
        } else {
            (HashSet::default(), HashSet::default())
        }
    }

    /// Log missing entries (compared to the reference language) as warnings
    pub fn log_missing_entries(&self) {
        let (missing_strings, missing_vectors) = self.list_missing_entries();
        for missing_key in missing_strings {
            warn!(
                "[{:?}] Missing string key {:?}",
                self.metadata().language_identifier,
                missing_key
            );
        }
        for missing_key in missing_vectors {
            warn!(
                "[{:?}] Missing vector key {:?}",
                self.metadata().language_identifier,
                missing_key
            );
        }
    }

    pub fn fonts(&self) -> &Fonts { &self.active.fonts }

    pub fn metadata(&self) -> &LanguageMetadata { &self.active.metadata }
}

impl LocalizationHandle {
    pub fn set_english_fallback(&mut self, use_english_fallback: bool) {
        self.use_english_fallback = use_english_fallback;
    }

    pub fn read(&self) -> LocalizationGuard {
        LocalizationGuard {
            active: self.active.read(),
            fallback: if self.use_english_fallback {
                self.fallback.map(|f| f.read())
            } else {
                None
            },
        }
    }

    pub fn load(specifier: &str) -> Result<Self, common_assets::Error> {
        let default_key = ["voxygen.i18n.", REFERENCE_LANG].concat();
        let language_key = ["voxygen.i18n.", specifier].concat();
        let is_default = language_key == default_key;
        let active = Language::load(&language_key)?;
        Ok(Self {
            active,
            watcher: active.reload_watcher(),
            fallback: if is_default {
                None
            } else {
                Language::load(&default_key).ok()
            },
            use_english_fallback: false,
        })
    }

    pub fn load_expect(specifier: &str) -> Self {
        Self::load(specifier).expect("Can't load language files")
    }

    pub fn reloaded(&mut self) -> bool { self.watcher.reloaded() }
}

struct FindManifests;

impl common_assets::DirLoadable for FindManifests {
    fn select_ids<S: common_assets::Source + ?Sized>(
        source: &S,
        specifier: &str,
    ) -> io::Result<Vec<common_assets::SharedString>> {
        let mut specifiers = Vec::new();

        source.read_dir(specifier, &mut |entry| {
            if let DirEntry::Directory(spec) = entry {
                let manifest_spec = [spec, ".", LANG_MANIFEST_FILE].concat();
                if source.exists(DirEntry::File(&manifest_spec, LANG_EXTENSION)) {
                    specifiers.push(manifest_spec.into());
                }
            }
        })?;

        Ok(specifiers)
    }
}

#[derive(Clone, Debug)]
struct LocalizationList(Vec<LanguageMetadata>);

impl common_assets::Compound for LocalizationList {
    fn load(
        cache: common_assets::AnyCache,
        specifier: &str,
    ) -> Result<Self, common_assets::BoxedError> {
        // List language directories
        let languages = common_assets::load_dir::<FindManifests>(specifier, false)
            .unwrap_or_else(|e| panic!("Failed to get manifests from {}: {:?}", specifier, e))
            .ids()
            .filter_map(|spec| cache.load::<RawManifest>(spec).ok())
            .map(|localization| localization.read().metadata.clone())
            .collect();

        Ok(LocalizationList(languages))
    }
}

/// Load all the available languages located in the voxygen asset directory
pub fn list_localizations() -> Vec<LanguageMetadata> {
    LocalizationList::load_expect_cloned("voxygen.i18n").0
}

#[cfg(test)]
mod tests {
    use crate::path::BasePath;

    // Test that localization list is loaded (not empty)
    #[test]
    fn test_localization_list() {
        let list = super::list_localizations();
        assert!(!list.is_empty());
    }

    // Test that reference language can be loaded
    #[test]
    fn test_localization_handle() {
        let _ = super::LocalizationHandle::load_expect(super::REFERENCE_LANG);
    }

    // Test to verify all languages that they are VALID and loadable, without
    // need of git just on the local assets folder
    #[test]
    fn verify_all_localizations() {
        // Generate paths
        let root_dir = common_assets::find_root().expect("Failed to discover repository root");
        crate::verification::verify_all_localizations(&BasePath::new(&root_dir));
    }

    // Test to verify all languages and print missing and faulty localisation
    #[test]
    #[ignore]
    fn test_all_localizations() {
        // Generate paths
        let root_dir = common_assets::find_root().expect("Failed to discover repository root");
        crate::analysis::test_all_localizations(&BasePath::new(&root_dir), true, true);
    }
}
