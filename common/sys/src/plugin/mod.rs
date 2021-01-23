pub mod errors;
pub mod module;

use common::assets::ASSETS_PATH;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use tracing::{error, info};

use plugin_api::Event;

use self::{
    errors::PluginError,
    module::{PluginModule, PreparedEventQuery},
};

use rayon::prelude::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PluginEnvironement {
    LOCAL,
    DISTANT,
    BOTH,
}

impl PluginEnvironement {
    pub fn is_local(&self) -> bool { matches!(self, Self::LOCAL | Self::BOTH) }

    pub fn is_distant(&self) -> bool { matches!(self, Self::DISTANT | Self::BOTH) }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PluginData {
    name: String,
    target: PluginEnvironement,
    modules: HashSet<PathBuf>,
    dependencies: HashSet<String>,
}

#[derive(Clone)]
pub enum PluginFile {
    ForDistant {
        data: PluginData,
        bytes: Vec<Vec<u8>>,
    },
    ForLocal {
        data: PluginData,
        modules: Vec<PluginModule>,
        files: HashMap<PathBuf, Vec<u8>>,
    },
    Both {
        data: PluginData,
        modules: Vec<PluginModule>,
        files: HashMap<PathBuf, Vec<u8>>,
        bytes: Vec<Vec<u8>>,
    },
}

impl std::fmt::Debug for PluginFile {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::ForDistant { data, bytes } => fmt
                .debug_struct("ForDistant")
                .field("data", data)
                .field("num modules", &bytes.len())
                .finish(),
            Self::ForLocal { data, modules, .. } => fmt
                .debug_struct("ForLocals")
                .field("data", data)
                .field("modules", modules)
                .finish(),
            Self::Both { data, modules, .. } => fmt
                .debug_struct("Both")
                .field("data", data)
                .field("modules", modules)
                .finish(),
        }
    }
}

impl PluginFile {
    pub fn from_reader<R: Read>(mut reader: R) -> Result<Self, PluginError> {
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).map_err(PluginError::Io)?;

        let mut files = tar::Archive::new(&*buf)
            .entries()
            .map_err(PluginError::Io)?
            .map(|e| {
                e.and_then(|e| {
                    Ok((e.path()?.into_owned(), {
                        let offset = e.raw_file_position() as usize;
                        buf[offset..offset + e.size() as usize].to_vec()
                    }))
                })
            })
            .collect::<Result<HashMap<_, _>, _>>()
            .map_err(PluginError::Io)?;

        let data = toml::de::from_slice::<PluginData>(
            &files
                .get(Path::new("plugin.toml"))
                .ok_or(PluginError::NoConfig)?,
        )
        .map_err(PluginError::Toml)?;

        Ok(match (data.target.is_local(), data.target.is_distant()) {
            (true, e) => {
                let mut bytes = Vec::new();
                let modules = data
                    .modules
                    .iter()
                    .map(|path| {
                        let wasm_data = files.remove(path).ok_or(PluginError::NoSuchModule)?;
                        let tmp =
                            PluginModule::new(data.name.to_owned(), &wasm_data).map_err(|e| {
                                PluginError::PluginModuleError(
                                    data.name.to_owned(),
                                    "<init>".to_owned(),
                                    e,
                                )
                            });
                        bytes.push(wasm_data);
                        tmp
                    })
                    .collect::<Result<_, _>>()?;
                if e {
                    Self::Both {
                        data,
                        modules,
                        files,
                        bytes,
                    }
                } else {
                    Self::ForLocal {
                        data,
                        modules,
                        files,
                    }
                }
            },
            (false, _) => {
                let bytes = data
                    .modules
                    .iter()
                    .map(|path| files.remove(path).ok_or(PluginError::NoSuchModule))
                    .collect::<Result<_, _>>()?;
                Self::ForDistant { data, bytes }
            },
        })
    }

    pub fn get_data(&self) -> &PluginData {
        // Wait for let-or syntax to be stable
        match self {
            Self::ForLocal { data, .. }
            | Self::Both { data, .. }
            | Self::ForDistant { data, .. } => data,
        }
    }
}

impl PluginExecutable for PluginFile {
    fn execute_prepared<T>(
        &self,
        event_name: &str,
        event: &PreparedEventQuery<T>,
    ) -> Result<Vec<T::Response>, PluginError>
    where
        T: Event,
    {
        if let Self::ForLocal { modules, data, .. } | Self::Both { modules, data, .. } = self {
            modules
                .iter()
                .flat_map(|module| {
                    module.try_execute(event_name, event).map(|x| {
                        x.map_err(|e| {
                            PluginError::PluginModuleError(
                                data.name.to_owned(),
                                event_name.to_owned(),
                                e,
                            )
                        })
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        } else {
            Ok(Vec::new())
        }
    }

    fn get_name(&self) -> &str { &self.get_data().name }
}

#[derive(Clone, Default, Debug)]
pub struct PluginMgr {
    plugins: Vec<PluginFile>,
    plugins_from_server: Vec<BinaryPlugin>,
}

impl PluginMgr {
    pub fn from_assets() -> Result<Self, PluginError> {
        println!("NEW PLUGIN MGR");
        let mut assets_path = (&*ASSETS_PATH).clone();
        assets_path.push("plugins");
        info!("Searching {:?} for plugins...", assets_path);
        let this = Self::from_dir(assets_path);
        if this.is_ok() {
            dbg!(&this.as_ref().unwrap().plugins);
        }
        this
    }

    pub fn load_server_plugins(&mut self, plugins: &Vec<(String, Vec<Vec<u8>>)>) {
        let prepared = PreparedEventQuery::new(&plugin_api::event::PluginLoadEvent {
            game_mode: plugin_api::GameMode::Client,
        })
        .unwrap();
        self.plugins_from_server
            .extend(plugins.iter().flat_map(|(name, bytes)| {
                info!(
                    "Loading {} with {} module(s) from server",
                    name,
                    bytes.len()
                );
                match BinaryPlugin::from_bytes(name.clone(), bytes) {
                    Ok(e) => {
                        if let Err(e) = e
                            .modules
                            .iter()
                            .flat_map(|x| x.try_execute("on_load", &prepared))
                            .collect::<Result<Vec<_>, _>>()
                        {
                            error!(
                                "Error while executing `on_load` on network retreived plugin: \
                                 `{}` \n{:?}",
                                name, e
                            );
                        }
                        Some(e)
                    },
                    Err(e) => {
                        tracing::error!(
                            "Error while loading distant plugin! Contact the server \
                             administrator!\n{:?}",
                            e
                        );
                        None
                    },
                }
            }));
        println!("Plugins from server: {}", self.plugins_from_server.len());
        dbg!(&self.plugins_from_server);
    }

    pub fn clear_server_plugins(&mut self) { self.plugins_from_server.clear(); }

    pub fn get_module_bytes(&self) -> Vec<(String, Vec<Vec<u8>>)> {
        self.plugins
            .iter()
            .flat_map(|x| {
                if let PluginFile::ForDistant { data, bytes, .. }
                | PluginFile::Both { data, bytes, .. } = x
                {
                    Some((data.name.clone(), bytes.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn execute_prepared<T>(
        &self,
        event_name: &str,
        event: &PreparedEventQuery<T>,
    ) -> Result<Vec<T::Response>, PluginError>
    where
        T: Event,
    {
        println!("{}", event_name);
        let mut o = self
            .plugins
            .par_iter()
            .map(|plugin| plugin.execute_prepared(event_name, event))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        println!("Event exe 2 {}", self.plugins_from_server.len());
        o.extend(
            self.plugins_from_server
                .par_iter()
                .map(|plugin| {
                    println!("Event exe 3");
                    plugin.execute_prepared(event_name, event)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten(),
        );
        Ok(o)
    }

    pub fn execute_event<T>(
        &self,
        event_name: &str,
        event: &T,
    ) -> Result<Vec<T::Response>, PluginError>
    where
        T: Event,
    {
        println!("Event exe 1");
        self.execute_prepared(event_name, &PreparedEventQuery::new(event)?)
    }

    pub fn from_dir<P: AsRef<Path>>(path: P) -> Result<Self, PluginError> {
        let plugins = fs::read_dir(path)
            .map_err(PluginError::Io)?
            .filter_map(|e| e.ok())
            .map(|entry| {
                if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                    && entry
                        .path()
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.ends_with(".plugin.tar"))
                        .unwrap_or(false)
                {
                    info!("Loading plugin at {:?}", entry.path());
                    PluginFile::from_reader(fs::File::open(entry.path()).map_err(PluginError::Io)?)
                        .map(Some)
                } else {
                    Ok(None)
                }
            })
            .filter_map(Result::transpose)
            .inspect(|p| {
                let _ = p.as_ref().map_err(|e| error!(?e, "Failed to load plugin"));
            })
            .collect::<Result<Vec<_>, _>>()?;

        for plugin in &plugins {
            match plugin {
                PluginFile::Both { data, modules, .. }
                | PluginFile::ForLocal { data, modules, .. } => {
                    info!(
                        "Loaded plugin '{}' with {} module(s)",
                        data.name,
                        modules.len()
                    );
                },
                PluginFile::ForDistant { data, bytes } => {
                    info!(
                        "Loaded plugin '{}' with {} module(s)",
                        data.name,
                        bytes.len()
                    );
                },
            }
        }

        Ok(Self {
            plugins,
            plugins_from_server: Vec::new(),
        })
    }
}

#[derive(Clone, Debug)]
pub struct BinaryPlugin {
    modules: Vec<PluginModule>,
    name: String,
}

impl BinaryPlugin {
    pub fn from_bytes(name: String, bytes: &Vec<Vec<u8>>) -> Result<Self, PluginError> {
        Ok(Self {
            modules: bytes
                .iter()
                .enumerate()
                .map(|(i, module)| {
                    PluginModule::new(format!("{}-module({})", name, i), module).map_err(|e| {
                        PluginError::PluginModuleError(name.clone(), "<init>".to_owned(), e)
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
            name,
        })
    }
}

impl PluginExecutable for BinaryPlugin {
    fn execute_prepared<T>(
        &self,
        event_name: &str,
        event: &PreparedEventQuery<T>,
    ) -> Result<Vec<T::Response>, PluginError>
    where
        T: Event,
    {
        println!("Event launched");
        self.modules
            .iter()
            .flat_map(|module| {
                println!("1: {}", event_name);
                module.try_execute(event_name, event).map(|x| {
                    x.map_err(|e| {
                        PluginError::PluginModuleError(
                            self.name.to_owned(),
                            event_name.to_owned(),
                            e,
                        )
                    })
                })
            })
            .collect::<Result<Vec<_>, _>>()
    }

    fn get_name(&self) -> &str { &self.name }
}

pub trait PluginExecutable {
    fn execute_prepared<T>(
        &self,
        event_name: &str,
        event: &PreparedEventQuery<T>,
    ) -> Result<Vec<T::Response>, PluginError>
    where
        T: Event;

    fn get_name(&self) -> &str;
}
