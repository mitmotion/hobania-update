mod diffusion;
mod erosion;
mod location;
mod map;
mod util;
mod way;

// Reexports
use self::erosion::Compute;
pub use self::{
    diffusion::diffusion,
    location::Location,
    map::{sample_pos, sample_wpos},
    util::get_horizon_map,
    way::{Cave, Path, Way},
};
pub(crate) use self::{
    erosion::{
        do_erosion, fill_sinks, get_lakes, get_multi_drainage, get_multi_rec, get_rivers, Alt,
        RiverData, RiverKind,
    },
    util::{
        cdf_irwin_hall, downhill, get_oceans, local_cells, map_edge_factor, uniform_noise, uphill,
        InverseCdf,
    },
};

use crate::{
    all::{Environment, ForestKind, TreeAttr},
    block::BlockGen,
    civ::{Place, PointOfInterest},
    column::ColumnGen,
    layer::spot::Spot,
    site::Site,
    util::{
        seed_expan, DHashSet, FastNoise, FastNoise2d, RandomField, Sampler, StructureGen2d,
        CARDINALS, LOCALITY, NEIGHBORS,
    },
    IndexRef, CONFIG,
};
use common::{
    assets::{self, AssetExt},
    calendar::Calendar,
    grid::Grid,
    lottery::Lottery,
    spiral::Spiral2d,
    store::Id,
    terrain::{
        map::MapConfig, uniform_idx_as_vec2, vec2_as_uniform_idx, BiomeKind, MapSizeLg,
        TerrainChunkSize,
    },
    vol::RectVolSize,
};
use common_net::msg::WorldMapMsg;
use enum_iterator::IntoEnumIterator;
use noise::{
    BasicMulti, Billow, Fbm, HybridMulti, MultiFractal, NoiseFn, RangeFunction, RidgedMulti,
    Seedable, SuperSimplex, Worley,
};
use num::{traits::FloatConst, Float, Signed};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaChaRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    f32, f64,
    fs::File,
    io::{BufReader, BufWriter},
    ops::{Add, Div, Mul, Neg, Sub},
    path::PathBuf,
};
use tracing::{debug, warn};
use vek::*;

/// Default base two logarithm of the world size, in chunks, per dimension.
///
/// Currently, our default map dimensions are 2^10 × 2^10 chunks,
/// mostly for historical reasons.  It is likely that we will increase this
/// default at some point.
const DEFAULT_WORLD_CHUNKS_LG: MapSizeLg =
    if let Ok(map_size_lg) = MapSizeLg::new(Vec2 { x: 10, y: 10 }) {
        map_size_lg
    } else {
        panic!("Default world chunk size does not satisfy required invariants.");
    };

/// A structure that holds cached noise values and cumulative distribution
/// functions for the input that led to those values.  See the definition of
/// InverseCdf for a description of how to interpret the types of its fields.
struct GenCdf {
    humid_base: InverseCdf,
    temp_base: InverseCdf,
    chaos: InverseCdf,
    alt: Box<[Alt]>,
    basement: Box<[Alt]>,
    water_alt: Box<[f32]>,
    dh: Box<[isize]>,
    /// NOTE: Until we hit 4096 × 4096, this should suffice since integers with
    /// an absolute value under 2^24 can be exactly represented in an f32.
    flux: Box<[Compute]>,
    pure_flux: InverseCdf<Compute>,
    alt_no_water: InverseCdf,
    rivers: Box<[RiverData]>,
}

pub(crate) struct GenCtx {
    pub turb_x_nz: SuperSimplex,
    pub turb_y_nz: SuperSimplex,
    pub chaos_nz: RidgedMulti,
    pub alt_nz: util::HybridMulti,
    pub hill_nz: SuperSimplex,
    pub temp_nz: Fbm,
    // Humidity noise
    pub humid_nz: Billow,
    // Small amounts of noise for simulating rough terrain.
    pub small_nz: BasicMulti,
    pub rock_nz: HybridMulti,
    pub tree_nz: BasicMulti,

    // TODO: unused, remove??? @zesterer
    pub _cave_0_nz: SuperSimplex,
    pub _cave_1_nz: SuperSimplex,

    pub structure_gen: StructureGen2d,
    pub _big_structure_gen: StructureGen2d,
    pub _region_gen: StructureGen2d,

    pub _fast_turb_x_nz: FastNoise,
    pub _fast_turb_y_nz: FastNoise,

    pub _town_gen: StructureGen2d,
    pub river_seed: RandomField,
    pub rock_strength_nz: Fbm,
    pub uplift_nz: Worley,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct SizeOpts {
    x_lg: u32,
    y_lg: u32,
    scale: f64,
}

impl SizeOpts {
    pub fn new(x_lg: u32, y_lg: u32, scale: f64) -> Self { Self { x_lg, y_lg, scale } }
}

impl Default for SizeOpts {
    fn default() -> Self {
        Self {
            x_lg: 10,
            y_lg: 10,
            scale: 2.0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum FileOpts {
    /// If set, generate the world map and do not try to save to or load from
    /// file (default).
    Generate(SizeOpts),
    /// If set, generate the world map and save the world file (path is created
    /// the same way screenshot paths are).
    Save(SizeOpts),
    /// Combination of Save and Load.
    /// Load map if exists or generate the world map and save the
    /// world file.
    LoadOrGenerate {
        name: String,
        #[serde(default)]
        opts: SizeOpts,
        #[serde(default)]
        overwrite: bool,
    },
    /// If set, load the world file from this path in legacy format (errors if
    /// path not found).  This option may be removed at some point, since it
    /// only applies to maps generated before map saving was merged into
    /// master.
    LoadLegacy(PathBuf),
    /// If set, load the world file from this path (errors if path not found).
    Load(PathBuf),
    /// If set, look for  the world file at this asset specifier (errors if
    /// asset is not found).
    ///
    /// NOTE: Could stand to merge this with `Load` and construct an enum that
    /// can handle either a PathBuf or an asset specifier, at some point.
    LoadAsset(String),
}

impl Default for FileOpts {
    fn default() -> Self { Self::Generate(SizeOpts::default()) }
}

impl FileOpts {
    fn load_content(&self) -> (Option<ModernMap>, MapSizeLg, f64) {
        let parsed_world_file = self.try_load_map();

        let map_size_lg = if let Some(map) = &parsed_world_file {
            MapSizeLg::new(map.map_size_lg)
                .expect("World size of loaded map does not satisfy invariants.")
        } else {
            self.map_size()
        };

        // NOTE: Change 1.0 to 4.0 for a 4x
        // improvement in world detail.  We also use this to automatically adjust
        // grid_scale (multiplying by 4.0) and multiply mins_per_sec by
        // 1.0 / (4.0 * 4.0) in ./erosion.rs, in order to get a similar rate of river
        // formation.
        //
        // FIXME: This is a hack!  At some point we will have a more principled way of
        // dealing with this.
        let default_continent_scale_hack = 2.0/*4.0*/;
        let continent_scale_hack = if let Some(map) = &parsed_world_file {
            map.continent_scale_hack
        } else {
            self.continent_scale_hack()
                .unwrap_or(default_continent_scale_hack)
        };

        (parsed_world_file, map_size_lg, continent_scale_hack)
    }

    // TODO: this should return Option so that caller can choose fallback
    fn map_size(&self) -> MapSizeLg {
        match self {
            Self::Generate(opts) | Self::Save(opts) | Self::LoadOrGenerate { opts, .. } => {
                MapSizeLg::new(Vec2 {
                    x: opts.x_lg,
                    y: opts.y_lg,
                })
                .unwrap_or_else(|e| {
                    warn!("World size does not satisfy invariants: {:?}", e);
                    DEFAULT_WORLD_CHUNKS_LG
                })
            },
            _ => DEFAULT_WORLD_CHUNKS_LG,
        }
    }

    fn continent_scale_hack(&self) -> Option<f64> {
        match self {
            Self::Generate(opts) | Self::Save(opts) | Self::LoadOrGenerate { opts, .. } => {
                Some(opts.scale)
            },
            _ => None,
        }
    }

    // TODO: This should probably return a Result, so that caller can choose
    // whether to log error
    fn try_load_map(&self) -> Option<ModernMap> {
        let map = match self {
            Self::LoadLegacy(ref path) => {
                let file = match File::open(path) {
                    Ok(file) => file,
                    Err(e) => {
                        warn!(?e, ?path, "Couldn't read path for maps");
                        return None;
                    },
                };

                let reader = BufReader::new(file);
                let map: WorldFileLegacy = match bincode::deserialize_from(reader) {
                    Ok(map) => map,
                    Err(e) => {
                        warn!(
                            ?e,
                            "Couldn't parse legacy map.  Maybe you meant to try a regular load?"
                        );
                        return None;
                    },
                };

                map.into_modern()
            },
            Self::Load(ref path) => {
                let file = match File::open(path) {
                    Ok(file) => file,
                    Err(e) => {
                        warn!(?e, ?path, "Couldn't read path for maps");
                        return None;
                    },
                };

                let reader = BufReader::new(file);
                let map: WorldFile = match bincode::deserialize_from(reader) {
                    Ok(map) => map,
                    Err(e) => {
                        warn!(
                            ?e,
                            "Couldn't parse modern map.  Maybe you meant to try a legacy load?"
                        );
                        return None;
                    },
                };

                map.into_modern()
            },
            Self::LoadAsset(ref specifier) => match WorldFile::load_owned(specifier) {
                Ok(map) => map.into_modern(),
                Err(err) => {
                    match err.reason().downcast_ref::<std::io::Error>() {
                        Some(e) => {
                            warn!(?e, ?specifier, "Couldn't read asset specifier for maps");
                        },
                        None => {
                            warn!(
                                ?err,
                                "Couldn't parse modern map.  Maybe you meant to try a legacy load?"
                            );
                        },
                    }
                    return None;
                },
            },
            Self::LoadOrGenerate {
                opts, overwrite, ..
            } => {
                // `unwrap` is safe here, because LoadOrGenerate has its path
                // always defined
                let path = self.map_path().unwrap();

                let file = match File::open(&path) {
                    Ok(file) => file,
                    Err(e) => {
                        warn!(?e, ?path, "Couldn't find needed map. Generating...");
                        return None;
                    },
                };

                let reader = BufReader::new(file);
                let map: WorldFile = match bincode::deserialize_from(reader) {
                    Ok(map) => map,
                    Err(e) => {
                        warn!(
                            ?e,
                            "Couldn't parse modern map.  Maybe you meant to try a legacy load?"
                        );
                        return None;
                    },
                };

                // FIXME:
                // We check if we need to generate new map by comparing gen opts.
                // But we also have another generation paramater that currently
                // passed outside and used for both worldsim and worldgen.
                //
                // Ideally, we need to figure out how we want to use seed, i. e.
                // moving worldgen seed to gen opts and use different sim seed from
                // server config or grab sim seed from world file.
                //
                // NOTE: we intentionally use pattern-matching here to get
                // options, so that when gen opts get another field, compiler
                // will force you to update following logic
                let SizeOpts { x_lg, y_lg, scale } = opts;
                let map = match map {
                    WorldFile::Veloren0_7_0(map) => map,
                    WorldFile::Veloren0_5_0(_) => {
                        panic!("World file v0.5.0 isn't supported with LoadOrGenerate.")
                    },
                };

                if map.continent_scale_hack != *scale || map.map_size_lg != Vec2::new(*x_lg, *y_lg)
                {
                    if *overwrite {
                        warn!(
                            "{}\n{}",
                            "Specified options don't correspond to these in loaded map.",
                            "Map will be regenerated and overwritten."
                        );
                    } else {
                        panic!(
                            "{}\n{}",
                            "Specified options don't correspond to these in loaded map.",
                            "Use 'ovewrite' option, if you wish to regenerate map."
                        );
                    }

                    return None;
                }

                map.into_modern()
            },
            Self::Generate { .. } | Self::Save { .. } => return None,
        };

        match map {
            Ok(map) => Some(map),
            Err(e) => {
                match e {
                    WorldFileError::WorldSizeInvalid => {
                        warn!("World size of map is invalid.");
                    },
                }
                None
            },
        }
    }

    fn map_path(&self) -> Option<PathBuf> {
        const MAP_DIR: &str = "./maps";
        // TODO: Work out a nice bincode file extension.
        let file_name = match self {
            Self::Save { .. } => {
                use std::time::SystemTime;

                Some(format!(
                    "map_{}.bin",
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .map(|d| d.as_millis())
                        .unwrap_or(0)
                ))
            },
            Self::LoadOrGenerate { name, .. } => Some(format!("{}.bin", name)),
            _ => None,
        };

        file_name.map(|name| std::path::Path::new(MAP_DIR).join(name))
    }

    fn save(&self, map: &WorldFile) {
        let path = if let Some(path) = self.map_path() {
            path
        } else {
            return;
        };

        // Check if folder exists and create it if it does not
        let map_dir = path.parent().expect("failed to get map directory");
        if !map_dir.exists() {
            if let Err(e) = std::fs::create_dir_all(map_dir) {
                warn!(?e, ?map_dir, "Couldn't create folder for map");
                return;
            }
        }

        let file = match File::create(path.clone()) {
            Ok(file) => file,
            Err(e) => {
                warn!(?e, ?path, "Couldn't create file for maps");
                return;
            },
        };

        let writer = BufWriter::new(file);
        if let Err(e) = bincode::serialize_into(writer, map) {
            warn!(?e, "Couldn't write map");
        }
    }
}

pub struct WorldOpts {
    /// Set to false to disable seeding elements during worldgen.
    pub seed_elements: bool,
    pub world_file: FileOpts,
    pub calendar: Option<Calendar>,
}

impl Default for WorldOpts {
    fn default() -> Self {
        Self {
            seed_elements: true,
            world_file: Default::default(),
            calendar: None,
        }
    }
}

/// LEGACY: Remove when people stop caring.
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct WorldFileLegacy {
    /// Saved altitude height map.
    pub alt: Box<[Alt]>,
    /// Saved basement height map.
    pub basement: Box<[Alt]>,
}

/// Version of the world map intended for use in Veloren 0.5.0.
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct WorldMap_0_5_0 {
    /// Saved altitude height map.
    pub alt: Box<[Alt]>,
    /// Saved basement height map.
    pub basement: Box<[Alt]>,
}

/// Version of the world map intended for use in Veloren 0.7.0.
#[derive(Serialize, Deserialize)]
#[repr(C)]
pub struct WorldMap_0_7_0 {
    /// Saved map size.
    pub map_size_lg: Vec2<u32>,
    /// Saved continent_scale hack, to try to better approximate the correct
    /// seed according to varying map size.
    ///
    /// TODO: Remove when generating new maps becomes more principled.
    pub continent_scale_hack: f64,
    /// Saved altitude height map.
    pub alt: Box<[Alt]>,
    /// Saved basement height map.
    pub basement: Box<[Alt]>,
}

/// Errors when converting a map to the most recent type (currently,
/// shared by the various map types, but at some point we might switch to
/// version-specific errors if it feels worthwhile).
#[derive(Debug)]
pub enum WorldFileError {
    /// Map size was invalid, and it can't be converted to a valid one.
    WorldSizeInvalid,
}

/// WORLD MAP.
///
/// A way to store certain components between runs of map generation.  Only
/// intended for development purposes--no attempt is made to detect map
/// invalidation or make sure that the map is synchronized with updates to
/// noise-rs, changes to other parameters, etc.
///
/// The map is versioned to enable format detection between versions of Veloren,
/// so that when we update the map format we don't break existing maps (or at
/// least, we will try hard not to break maps between versions; if we can't
/// avoid it, we can at least give a reasonable error message).
///
/// NOTE: We rely somewhat heavily on the implementation specifics of bincode
/// to make sure this is backwards compatible.  When adding new variants here,
/// Be very careful to make sure tha the old variants are preserved in the
/// correct order and with the correct names and indices, and make sure to keep
/// the #[repr(u32)]!
///
/// All non-legacy versions of world files should (ideally) fit in this format.
/// Since the format contains a version and is designed to be extensible
/// backwards-compatibly, the only reason not to use this forever would be if we
/// decided to move away from BinCode, or store data across multiple files (or
/// something else weird I guess).
///
/// Update this when you add a new map version.
#[derive(Serialize, Deserialize)]
#[repr(u32)]
pub enum WorldFile {
    Veloren0_5_0(WorldMap_0_5_0) = 0,
    Veloren0_7_0(WorldMap_0_7_0) = 1,
}

impl assets::Asset for WorldFile {
    type Loader = assets::BincodeLoader;

    const EXTENSION: &'static str = "bin";
}

/// Data for the most recent map type.  Update this when you add a new map
/// version.
pub type ModernMap = WorldMap_0_7_0;

/// The default world map.
///
/// TODO: Consider using some naming convention to automatically change this
/// with changing versions, or at least keep it in a constant somewhere that's
/// easy to change.
pub const DEFAULT_WORLD_MAP: &str = "world.map.veloren_0_9_0_0";

impl WorldFileLegacy {
    #[inline]
    /// Idea: each map type except the latest knows how to transform
    /// into the the subsequent map version, and each map type including the
    /// latest exposes an "into_modern()" method that converts this map type
    /// to the modern map type.  Thus, to migrate a map from an old format to a
    /// new format, we just need to transform the old format to the
    /// subsequent map version, and then call .into_modern() on that--this
    /// should construct a call chain that ultimately ends up with a modern
    /// version.
    pub fn into_modern(self) -> Result<ModernMap, WorldFileError> {
        // NOTE: At this point, we assume that any remaining legacy maps were 1024 ×
        // 1024.
        if self.alt.len() != self.basement.len() || self.alt.len() != 1024 * 1024 {
            return Err(WorldFileError::WorldSizeInvalid);
        }

        let map = WorldMap_0_5_0 {
            alt: self.alt,
            basement: self.basement,
        };

        map.into_modern()
    }
}

impl WorldMap_0_5_0 {
    #[inline]
    pub fn into_modern(self) -> Result<ModernMap, WorldFileError> {
        let pow_size = (self.alt.len().trailing_zeros()) / 2;
        let two_coord_size = 1 << (2 * pow_size);
        if self.alt.len() != self.basement.len() || self.alt.len() != two_coord_size {
            return Err(WorldFileError::WorldSizeInvalid);
        }

        // The recommended continent scale for maps from version 0.5.0 is (in all
        // existing cases) just 1.0 << (f64::from(pow_size) - 10.0).
        let continent_scale_hack = (f64::from(pow_size) - 10.0).exp2();

        let map = WorldMap_0_7_0 {
            map_size_lg: Vec2::new(pow_size, pow_size),
            continent_scale_hack,
            alt: self.alt,
            basement: self.basement,
        };

        map.into_modern()
    }
}

impl WorldMap_0_7_0 {
    #[inline]
    pub fn into_modern(self) -> Result<ModernMap, WorldFileError> {
        if self.alt.len() != self.basement.len()
            || self.alt.len() != (1 << (self.map_size_lg.x + self.map_size_lg.y))
            || self.continent_scale_hack <= 0.0
        {
            return Err(WorldFileError::WorldSizeInvalid);
        }

        Ok(self)
    }
}

impl WorldFile {
    /// Turns map data from the latest version into a versioned WorldFile ready
    /// for serialization. Whenever a new map is updated, just change the
    /// variant we construct here to make sure we're using the latest map
    /// version.

    pub fn new(map: ModernMap) -> Self { WorldFile::Veloren0_7_0(map) }

    #[inline]
    /// Turns a WorldFile into the latest version.  Whenever a new map version
    /// is added, just add it to this match statement.
    pub fn into_modern(self) -> Result<ModernMap, WorldFileError> {
        match self {
            WorldFile::Veloren0_5_0(map) => map.into_modern(),
            WorldFile::Veloren0_7_0(map) => map.into_modern(),
        }
    }
}

pub struct WorldSim {
    pub seed: u32,
    /// Base 2 logarithm of the map size.
    map_size_lg: MapSizeLg,
    /// Maximum height above sea level of any chunk in the map (not including
    /// post-erosion warping, cliffs, and other things like that).
    pub max_height: f32,
    pub(crate) chunks: Vec<SimChunk>,
    //TODO: remove or use this property
    pub(crate) _locations: Vec<Location>,

    pub(crate) gen_ctx: GenCtx,
    pub rng: ChaChaRng,

    pub(crate) calendar: Option<Calendar>,
}

impl WorldSim {
    pub fn generate(seed: u32, opts: WorldOpts, threadpool: &rayon::ThreadPool) -> Self {
        let calendar = opts.calendar; // separate lifetime of elements
        let world_file = opts.world_file;

        // Parse out the contents of various map formats into the values we need.
        let (parsed_world_file, map_size_lg, continent_scale_hack) = world_file.load_content();
        // Currently only used with LoadOrGenerate to know if we need to
        // overwrite world file
        let fresh = parsed_world_file.is_none();

        let mut rng = ChaChaRng::from_seed(seed_expan::rng_state(seed));
        let continent_scale = continent_scale_hack
            * 5_000.0f64
                .div(32.0)
                .mul(TerrainChunkSize::RECT_SIZE.x as f64);
        let rock_lacunarity = 2.0;
        let uplift_scale = 128.0;
        let uplift_turb_scale = uplift_scale / 4.0;

        // NOTE: Changing order will significantly change WorldGen, so try not to!
        let gen_ctx = GenCtx {
            turb_x_nz: SuperSimplex::new().set_seed(rng.gen()),
            turb_y_nz: SuperSimplex::new().set_seed(rng.gen()),
            chaos_nz: RidgedMulti::new()
                .set_octaves(7)
                .set_frequency(RidgedMulti::DEFAULT_FREQUENCY * (5_000.0 / continent_scale))
                .set_seed(rng.gen()),
            hill_nz: SuperSimplex::new().set_seed(rng.gen()),
            alt_nz: util::HybridMulti::new()
                .set_octaves(8)
                .set_frequency((10_000.0 / continent_scale) as f64)
                // persistence = lacunarity^(-(1.0 - fractal increment))
                .set_lacunarity(util::HybridMulti::DEFAULT_LACUNARITY)
                .set_persistence(util::HybridMulti::DEFAULT_LACUNARITY.powi(-1))
                .set_offset(0.0)
                .set_seed(rng.gen()),
            temp_nz: Fbm::new()
                .set_octaves(6)
                .set_persistence(0.5)
                .set_frequency(1.0 / (((1 << 6) * 64) as f64))
                .set_lacunarity(2.0)
                .set_seed(rng.gen()),

            small_nz: BasicMulti::new().set_octaves(2).set_seed(rng.gen()),
            rock_nz: HybridMulti::new().set_persistence(0.3).set_seed(rng.gen()),
            tree_nz: BasicMulti::new()
                .set_octaves(12)
                .set_persistence(0.75)
                .set_seed(rng.gen()),
            _cave_0_nz: SuperSimplex::new().set_seed(rng.gen()),
            _cave_1_nz: SuperSimplex::new().set_seed(rng.gen()),

            structure_gen: StructureGen2d::new(rng.gen(), 24, 10),
            _big_structure_gen: StructureGen2d::new(rng.gen(), 768, 512),
            _region_gen: StructureGen2d::new(rng.gen(), 400, 96),
            humid_nz: Billow::new()
                .set_octaves(9)
                .set_persistence(0.4)
                .set_frequency(0.2)
                .set_seed(rng.gen()),

            _fast_turb_x_nz: FastNoise::new(rng.gen()),
            _fast_turb_y_nz: FastNoise::new(rng.gen()),

            _town_gen: StructureGen2d::new(rng.gen(), 2048, 1024),
            river_seed: RandomField::new(rng.gen()),
            rock_strength_nz: Fbm::new()
                .set_octaves(10)
                .set_lacunarity(rock_lacunarity)
                // persistence = lacunarity^(-(1.0 - fractal increment))
                // NOTE: In paper, fractal increment is roughly 0.25.
                .set_persistence(rock_lacunarity.powf(-0.75))
                .set_frequency(
                    1.0 * (5_000.0 / continent_scale)
                        / (2.0 * TerrainChunkSize::RECT_SIZE.x as f64 * 2.0.powi(10 - 1)),
                )
                .set_seed(rng.gen()),
            uplift_nz: Worley::new()
                .set_seed(rng.gen())
                .set_frequency(1.0 / (TerrainChunkSize::RECT_SIZE.x as f64 * uplift_scale))
                .set_displacement(1.0)
                .set_range_function(RangeFunction::Euclidean),
        };

        let river_seed = &gen_ctx.river_seed;
        let rock_strength_nz = &gen_ctx.rock_strength_nz;

        // Suppose the old world has grid spacing Δx' = Δy', new Δx = Δy.
        // We define grid_scale such that Δx = height_scale * Δx' ⇒
        //  grid_scale = Δx / Δx'.
        let grid_scale = 1.0f64 / (4.0 / continent_scale_hack)/*1.0*/;

        // Now, suppose we want to generate a world with "similar" topography, defined
        // in this case as having roughly equal slopes at steady state, with the
        // simulation taking roughly as many steps to get to the point the
        // previous world was at when it finished being simulated.
        //
        // Some computations with our coupled SPL/debris flow give us (for slope S
        // constant) the following suggested scaling parameters to make this
        // work:   k_fs_scale ≡ (K𝑓 / K𝑓') = grid_scale^(-2m) =
        // grid_scale^(-2θn)
        let k_fs_scale = |theta, n| grid_scale.powf(-2.0 * (theta * n) as f64);

        //   k_da_scale ≡ (K_da / K_da') = grid_scale^(-2q)
        let k_da_scale = |q| grid_scale.powf(-2.0 * q);
        //
        // Some other estimated parameters are harder to come by and *much* more
        // dubious, not being accurate for the coupled equation. But for the SPL
        // only one we roughly find, for h the height at steady state and time τ
        // = time to steady state, with Hack's Law estimated b = 2.0 and various other
        // simplifying assumptions, the estimate:
        //   height_scale ≡ (h / h') = grid_scale^(n)
        let height_scale = |n: f32| grid_scale.powf(n as f64) as Alt;
        //   time_scale ≡ (τ / τ') = grid_scale^(n)
        let time_scale = |n: f32| grid_scale.powf(n as f64);
        //
        // Based on this estimate, we have:
        //   delta_t_scale ≡ (Δt / Δt') = time_scale
        let delta_t_scale = time_scale;
        //   alpha_scale ≡ (α / α') = height_scale^(-1)
        let alpha_scale = |n: f32| height_scale(n).recip() as f32;
        //
        // Slightly more dubiously (need to work out the math better) we find:
        //   k_d_scale ≡ (K_d / K_d') = grid_scale^2 / (/*height_scale * */ time_scale)
        let k_d_scale = |n: f32| grid_scale.powi(2) / (/* height_scale(n) * */time_scale(n));
        //   epsilon_0_scale ≡ (ε₀ / ε₀') = height_scale(n) / time_scale(n)
        let epsilon_0_scale = |n| (height_scale(n) / time_scale(n) as Alt) as f32;

        // Approximate n for purposes of computation of parameters above over the whole
        // grid (when a chunk isn't available).
        let n_approx = 1.0;
        let max_erosion_per_delta_t = 64.0 * delta_t_scale(n_approx);
        let n_steps = 100;
        let n_small_steps = 0;
        let n_post_load_steps = 0;

        // Logistic regression.  Make sure x ∈ (0, 1).
        let logit = |x: f64| x.ln() - (-x).ln_1p();
        // 0.5 + 0.5 * tanh(ln(1 / (1 - 0.1) - 1) / (2 * (sqrt(3)/pi)))
        let logistic_2_base = 3.0f64.sqrt() * f64::consts::FRAC_2_PI;
        // Assumes μ = 0, σ = 1
        let logistic_cdf = |x: f64| (x / logistic_2_base).tanh() * 0.5 + 0.5;

        let map_size_chunks_len_f64 = map_size_lg.chunks().map(f64::from).product();
        let min_epsilon = 1.0 / map_size_chunks_len_f64.max(f64::EPSILON as f64 * 0.5);
        let max_epsilon =
            (1.0 - 1.0 / map_size_chunks_len_f64).min(1.0 - f64::EPSILON as f64 * 0.5);

        // No NaNs in these uniform vectors, since the original noise value always
        // returns Some.
        let ((alt_base, _), (chaos, _)) = threadpool.join(
            || {
                uniform_noise(map_size_lg, |_, wposf| {
                    // "Base" of the chunk, to be multiplied by CONFIG.mountain_scale (multiplied
                    // value is from -0.35 * (CONFIG.mountain_scale * 1.05) to
                    // 0.35 * (CONFIG.mountain_scale * 0.95), but value here is from -0.3675 to
                    // 0.3325).
                    Some(
                        (gen_ctx
                            .alt_nz
                            .get((wposf.div(10_000.0)).into_array())
                            .min(1.0)
                            .max(-1.0))
                        .sub(0.05)
                        .mul(0.35),
                    )
                })
            },
            || {
                uniform_noise(map_size_lg, |_, wposf| {
                    // From 0 to 1.6, but the distribution before the max is from -1 and 1.6, so
                    // there is a 50% chance that hill will end up at 0.3 or
                    // lower, and probably a very high change it will be exactly
                    // 0.
                    let hill = (0.0f64
                        + gen_ctx
                            .hill_nz
                            .get(
                                (wposf
                                    .mul(32.0)
                                    .div(TerrainChunkSize::RECT_SIZE.map(|e| e as f64))
                                    .div(1_500.0))
                                .into_array(),
                            )
                            .min(1.0)
                            .max(-1.0)
                            .mul(1.0)
                        + gen_ctx
                            .hill_nz
                            .get(
                                (wposf
                                    .mul(32.0)
                                    .div(TerrainChunkSize::RECT_SIZE.map(|e| e as f64))
                                    .div(400.0))
                                .into_array(),
                            )
                            .min(1.0)
                            .max(-1.0)
                            .mul(0.3))
                    .add(0.3)
                    .max(0.0);

                    // chaos produces a value in [0.12, 1.32].  It is a meta-level factor intended
                    // to reflect how "chaotic" the region is--how much weird
                    // stuff is going on on this terrain.
                    Some(
                        ((gen_ctx
                            .chaos_nz
                            .get((wposf.div(3_000.0)).into_array())
                            .min(1.0)
                            .max(-1.0))
                        .add(1.0)
                        .mul(0.5)
                        // [0, 1] * [0.4, 1] = [0, 1] (but probably towards the lower end)
                        .mul(
                            (gen_ctx
                                .chaos_nz
                                .get((wposf.div(6_000.0)).into_array())
                                .min(1.0)
                                .max(-1.0))
                            .abs()
                            .max(0.4)
                            .min(1.0),
                        )
                        // Chaos is always increased by a little when we're on a hill (but remember
                        // that hill is 0.3 or less about 50% of the time).
                        // [0, 1] + 0.2 * [0, 1.6] = [0, 1.32]
                        .add(0.2 * hill)
                        // We can't have *no* chaos!
                        .max(0.12)) as f32,
                    )
                })
            },
        );

        // We ignore sea level because we actually want to be relative to sea level here
        // and want things in CONFIG.mountain_scale units, but otherwise this is
        // a correct altitude calculation.  Note that this is using the
        // "unadjusted" temperature.
        //
        // No NaNs in these uniform vectors, since the original noise value always
        // returns Some.
        let (alt_old, _) = uniform_noise(map_size_lg, |posi, wposf| {
            // This is the extension upwards from the base added to some extra noise from -1
            // to 1.
            //
            // The extra noise is multiplied by alt_main (the mountain part of the
            // extension) powered to 0.8 and clamped to [0.15, 1], to get a
            // value between [-1, 1] again.
            //
            // The sides then receive the sequence (y * 0.3 + 1.0) * 0.4, so we have
            // [-1*1*(1*0.3+1)*0.4, 1*(1*0.3+1)*0.4] = [-0.52, 0.52].
            //
            // Adding this to alt_main thus yields a value between -0.4 (if alt_main = 0 and
            // gen_ctx = -1, 0+-1*(0*.3+1)*0.4) and 1.52 (if alt_main = 1 and gen_ctx = 1).
            // Most of the points are above 0.
            //
            // Next, we add again by a sin of alt_main (between [-1, 1])^pow, getting
            // us (after adjusting for sign) another value between [-1, 1], and then this is
            // multiplied by 0.045 to get [-0.045, 0.045], which is added to [-0.4, 0.52] to
            // get [-0.445, 0.565].
            let alt_main = {
                // Extension upwards from the base.  A positive number from 0 to 1 curved to be
                // maximal at 0.  Also to be multiplied by CONFIG.mountain_scale.
                let alt_main = (gen_ctx
                    .alt_nz
                    .get((wposf.div(2_000.0)).into_array())
                    .min(1.0)
                    .max(-1.0))
                .abs()
                .powf(1.35);

                fn spring(x: f64, pow: f64) -> f64 { x.abs().powf(pow) * x.signum() }

                0.0 + alt_main
                    + (gen_ctx
                        .small_nz
                        .get(
                            (wposf
                                .mul(32.0)
                                .div(TerrainChunkSize::RECT_SIZE.map(|e| e as f64))
                                .div(300.0))
                            .into_array(),
                        )
                        .min(1.0)
                        .max(-1.0))
                    .mul(alt_main.powf(0.8).max(/* 0.25 */ 0.15))
                    .mul(0.3)
                    .add(1.0)
                    .mul(0.4)
                    + spring(alt_main.abs().sqrt().min(0.75).mul(60.0).sin(), 4.0).mul(0.045)
            };

            // Now we can compute the final altitude using chaos.
            // We multiply by chaos clamped to [0.1, 1.32] to get a value between [0.03,
            // 2.232] for alt_pre, then multiply by CONFIG.mountain_scale and
            // add to the base and sea level to get an adjusted value, then
            // multiply the whole thing by map_edge_factor (TODO: compute final
            // bounds).
            //
            // [-.3675, .3325] + [-0.445, 0.565] * [0.12, 1.32]^1.2
            // ~ [-.3675, .3325] + [-0.445, 0.565] * [0.07, 1.40]
            // = [-.3675, .3325] + ([-0.5785, 0.7345])
            // = [-0.946, 1.067]
            Some(
                ((alt_base[posi].1 + alt_main.mul((chaos[posi].1 as f64).powf(1.2)))
                    .mul(map_edge_factor(map_size_lg, posi) as f64)
                    .add(
                        (CONFIG.sea_level as f64)
                            .div(CONFIG.mountain_scale as f64)
                            .mul(map_edge_factor(map_size_lg, posi) as f64),
                    )
                    .sub((CONFIG.sea_level as f64).div(CONFIG.mountain_scale as f64)))
                    as f32,
            )
        });

        // Calculate oceans.
        let is_ocean = get_oceans(map_size_lg, |posi: usize| alt_old[posi].1);
        // NOTE: Uncomment if you want oceans to exclusively be on the border of the
        // map.
        /* let is_ocean = (0..map_size_lg.chunks())
        .into_par_iter()
        .map(|i| map_edge_factor(map_size_lg, i) == 0.0)
        .collect::<Vec<_>>(); */
        let is_ocean_fn = |posi: usize| is_ocean[posi];

        let turb_wposf_div = 8.0;
        let n_func = |posi| {
            if is_ocean_fn(posi) {
                return 1.0;
            }
            1.0
        };
        let old_height = |posi: usize| {
            alt_old[posi].1 * CONFIG.mountain_scale * height_scale(n_func(posi)) as f32
        };

        // NOTE: Needed if you wish to use the distance to the point defining the Worley
        // cell, not just the value within that cell.
        // let uplift_nz_dist = gen_ctx.uplift_nz.clone().enable_range(true);

        // Recalculate altitudes without oceans.
        // NaNs in these uniform vectors wherever is_ocean_fn returns true.
        let (alt_old_no_ocean, _) = uniform_noise(map_size_lg, |posi, _| {
            if is_ocean_fn(posi) {
                None
            } else {
                Some(old_height(posi))
            }
        });
        let (uplift_uniform, _) = uniform_noise(map_size_lg, |posi, _wposf| {
            if is_ocean_fn(posi) {
                None
            } else {
                let oheight = alt_old_no_ocean[posi].0 as f64 - 0.5;
                let height = (oheight + 0.5).powi(2);
                Some(height)
            }
        });

        let alt_old_min_uniform = 0.0;
        let alt_old_max_uniform = 1.0;

        let inv_func = |x: f64| x;
        let alt_exp_min_uniform = inv_func(min_epsilon);
        let alt_exp_max_uniform = inv_func(max_epsilon);

        let erosion_factor = |x: f64| {
            (inv_func(x) - alt_exp_min_uniform) / (alt_exp_max_uniform - alt_exp_min_uniform)
        };
        let rock_strength_div_factor = (2.0 * TerrainChunkSize::RECT_SIZE.x as f64) / 8.0;
        let theta_func = |_posi| 0.4;
        let kf_func = {
            |posi| {
                let kf_scale_i = k_fs_scale(theta_func(posi), n_func(posi)) as f64;
                if is_ocean_fn(posi) {
                    return 1.0e-4 * kf_scale_i;
                }

                let kf_i = // kf = 1.5e-4: high-high (plateau [fan sediment])
                // kf = 1e-4: high (plateau)
                // kf = 2e-5: normal (dike [unexposed])
                // kf = 1e-6: normal-low (dike [exposed])
                // kf = 2e-6: low (mountain)
                // --
                // kf = 2.5e-7 to 8e-7: very low (Cordonnier papers on plate tectonics)
                // ((1.0 - uheight) * (1.5e-4 - 2.0e-6) + 2.0e-6) as f32
                //
                // ACTUAL recorded values worldwide: much lower...
                1.0e-6
                ;
                kf_i * kf_scale_i
            }
        };
        let kd_func = {
            |posi| {
                let n = n_func(posi);
                let kd_scale_i = k_d_scale(n);
                if is_ocean_fn(posi) {
                    let kd_i = 1.0e-2 / 4.0;
                    return kd_i * kd_scale_i;
                }
                // kd = 1e-1: high (mountain, dike)
                // kd = 1.5e-2: normal-high (plateau [fan sediment])
                // kd = 1e-2: normal (plateau)
                let kd_i = 1.0e-2 / 4.0;
                kd_i * kd_scale_i
            }
        };
        let g_func = |posi| {
            if map_edge_factor(map_size_lg, posi) == 0.0 {
                return 0.0;
            }
            // G = d* v_s / p_0, where
            //  v_s is the settling velocity of sediment grains
            //  p_0 is the mean precipitation rate
            //  d* is the sediment concentration ratio (between concentration near riverbed
            //  interface, and average concentration over the water column).
            //  d* varies with Rouse number which defines relative contribution of bed,
            // suspended,  and washed loads.
            //
            // G is typically on the order of 1 or greater.  However, we are only guaranteed
            // to converge for G ≤ 1, so we keep it in the chaos range of [0.12,
            // 1.32].
            1.0
        };
        let epsilon_0_func = |posi| {
            // epsilon_0_scale is roughly [using Hack's Law with b = 2 and SPL without
            // debris flow or hillslopes] equal to the ratio of the old to new
            // area, to the power of -n_i.
            let epsilon_0_scale_i = epsilon_0_scale(n_func(posi));
            if is_ocean_fn(posi) {
                // marine: ε₀ = 2.078e-3
                let epsilon_0_i = 2.078e-3 / 4.0;
                return epsilon_0_i * epsilon_0_scale_i;
            }
            let wposf = (uniform_idx_as_vec2(map_size_lg, posi)
                * TerrainChunkSize::RECT_SIZE.map(|e| e as i32))
            .map(|e| e as f64);
            let turb_wposf = wposf
                .mul(5_000.0 / continent_scale)
                .div(TerrainChunkSize::RECT_SIZE.map(|e| e as f64))
                .div(turb_wposf_div);
            let turb = Vec2::new(
                gen_ctx.turb_x_nz.get(turb_wposf.into_array()),
                gen_ctx.turb_y_nz.get(turb_wposf.into_array()),
            ) * uplift_turb_scale
                * TerrainChunkSize::RECT_SIZE.map(|e| e as f64);
            let turb_wposf = wposf + turb;
            let uheight = gen_ctx
                .uplift_nz
                .get(turb_wposf.into_array())
                .min(1.0)
                .max(-1.0)
                .mul(0.5)
                .add(0.5);
            let wposf3 = Vec3::new(
                wposf.x,
                wposf.y,
                uheight * CONFIG.mountain_scale as f64 * rock_strength_div_factor,
            );
            let rock_strength = gen_ctx
                .rock_strength_nz
                .get(wposf3.into_array())
                .min(1.0)
                .max(-1.0)
                .mul(0.5)
                .add(0.5);
            let center = 0.4;
            let dmin = center - 0.05;
            let dmax = center + 0.05;
            let log_odds = |x: f64| logit(x) - logit(center);
            let ustrength = logistic_cdf(
                1.0 * logit(rock_strength.min(1.0f64 - 1e-7).max(1e-7))
                    + 1.0 * log_odds(uheight.min(dmax).max(dmin)),
            );
            // marine: ε₀ = 2.078e-3
            // San Gabriel Mountains: ε₀ = 3.18e-4
            // Oregon Coast Range: ε₀ = 2.68e-4
            // Frogs Hollow (peak production = 0.25): ε₀ = 1.41e-4
            // Point Reyes: ε₀ = 8.1e-5
            // Nunnock River (fractured granite, least weathered?): ε₀ = 5.3e-5
            let epsilon_0_i = ((1.0 - ustrength) * (2.078e-3 - 5.3e-5) + 5.3e-5) as f32 / 4.0;
            epsilon_0_i * epsilon_0_scale_i
        };
        let alpha_func = |posi| {
            let alpha_scale_i = alpha_scale(n_func(posi));
            if is_ocean_fn(posi) {
                // marine: α = 3.7e-2
                return 3.7e-2 * alpha_scale_i;
            }
            let wposf = (uniform_idx_as_vec2(map_size_lg, posi)
                * TerrainChunkSize::RECT_SIZE.map(|e| e as i32))
            .map(|e| e as f64);
            let turb_wposf = wposf
                .mul(5_000.0 / continent_scale)
                .div(TerrainChunkSize::RECT_SIZE.map(|e| e as f64))
                .div(turb_wposf_div);
            let turb = Vec2::new(
                gen_ctx.turb_x_nz.get(turb_wposf.into_array()),
                gen_ctx.turb_y_nz.get(turb_wposf.into_array()),
            ) * uplift_turb_scale
                * TerrainChunkSize::RECT_SIZE.map(|e| e as f64);
            let turb_wposf = wposf + turb;
            let uheight = gen_ctx
                .uplift_nz
                .get(turb_wposf.into_array())
                .min(1.0)
                .max(-1.0)
                .mul(0.5)
                .add(0.5);
            let wposf3 = Vec3::new(
                wposf.x,
                wposf.y,
                uheight * CONFIG.mountain_scale as f64 * rock_strength_div_factor,
            );
            let rock_strength = gen_ctx
                .rock_strength_nz
                .get(wposf3.into_array())
                .min(1.0)
                .max(-1.0)
                .mul(0.5)
                .add(0.5);
            let center = 0.4;
            let dmin = center - 0.05;
            let dmax = center + 0.05;
            let log_odds = |x: f64| logit(x) - logit(center);
            let ustrength = logistic_cdf(
                1.0 * logit(rock_strength.min(1.0f64 - 1e-7).max(1e-7))
                    + 1.0 * log_odds(uheight.min(dmax).max(dmin)),
            );
            // Frog Hollow (peak production = 0.25): α = 4.2e-2
            // San Gabriel Mountains: α = 3.8e-2
            // marine: α = 3.7e-2
            // Oregon Coast Range: α = 3e-2
            // Nunnock river (fractured granite, least weathered?): α = 2e-3
            // Point Reyes: α = 1.6e-2
            // The stronger  the rock, the faster the decline in soil production.
            let alpha_i = (ustrength * (4.2e-2 - 1.6e-2) + 1.6e-2) as f32;
            alpha_i * alpha_scale_i
        };
        let uplift_fn = |posi| {
            if is_ocean_fn(posi) {
                return 0.0;
            }
            let height = (uplift_uniform[posi].1 - alt_old_min_uniform) as f64
                / (alt_old_max_uniform - alt_old_min_uniform) as f64;

            let height = height.mul(max_epsilon - min_epsilon).add(min_epsilon);
            let height = erosion_factor(height);
            assert!(height >= 0.0);
            assert!(height <= 1.0);

            // u = 1e-3: normal-high (dike, mountain)
            // u = 5e-4: normal (mid example in Yuan, average mountain uplift)
            // u = 2e-4: low (low example in Yuan; known that lagoons etc. may have u ~
            // 0.05). u = 0: low (plateau [fan, altitude = 0.0])
            let height = height.mul(max_erosion_per_delta_t);
            height as f64
        };
        let alt_func = |posi| {
            if is_ocean_fn(posi) {
                old_height(posi)
            } else {
                (old_height(posi) as f64 / CONFIG.mountain_scale as f64) as f32 - 0.5
            }
        };

        // Perform some erosion.

        let (alt, basement) = if let Some(map) = parsed_world_file {
            (map.alt, map.basement)
        } else {
            let (alt, basement) = do_erosion(
                map_size_lg,
                max_erosion_per_delta_t as f32,
                n_steps,
                river_seed,
                // varying conditions
                &rock_strength_nz,
                // initial conditions
                alt_func,
                alt_func,
                is_ocean_fn,
                // empirical constants
                uplift_fn,
                n_func,
                theta_func,
                kf_func,
                kd_func,
                g_func,
                epsilon_0_func,
                alpha_func,
                // scaling factors
                height_scale,
                k_d_scale(n_approx),
                k_da_scale,
                threadpool,
            );

            // Quick "small scale" erosion cycle in order to lower extreme angles.
            do_erosion(
                map_size_lg,
                1.0f32,
                n_small_steps,
                river_seed,
                &rock_strength_nz,
                |posi| alt[posi] as f32,
                |posi| basement[posi] as f32,
                is_ocean_fn,
                |posi| uplift_fn(posi) * (1.0 / max_erosion_per_delta_t),
                n_func,
                theta_func,
                kf_func,
                kd_func,
                g_func,
                epsilon_0_func,
                alpha_func,
                height_scale,
                k_d_scale(n_approx),
                k_da_scale,
                threadpool,
            )
        };

        // Save map, if necessary.
        // NOTE: We wll always save a map with latest version.
        let map = WorldFile::new(ModernMap {
            continent_scale_hack,
            map_size_lg: map_size_lg.vec(),
            alt,
            basement,
        });
        if fresh {
            world_file.save(&map);
        }

        // Skip validation--we just performed a no-op conversion for this map, so it had
        // better be valid!
        let ModernMap {
            continent_scale_hack: _,
            map_size_lg: _,
            alt,
            basement,
        } = map.into_modern().unwrap();

        // Additional small-scale erosion after map load, only used during testing.
        let (alt, basement) = if n_post_load_steps == 0 {
            (alt, basement)
        } else {
            do_erosion(
                map_size_lg,
                1.0f32,
                n_post_load_steps,
                river_seed,
                &rock_strength_nz,
                |posi| alt[posi] as f32,
                |posi| basement[posi] as f32,
                is_ocean_fn,
                |posi| uplift_fn(posi) * (1.0 / max_erosion_per_delta_t),
                n_func,
                theta_func,
                kf_func,
                kd_func,
                g_func,
                epsilon_0_func,
                alpha_func,
                height_scale,
                k_d_scale(n_approx),
                k_da_scale,
                threadpool,
            )
        };

        let is_ocean = get_oceans(map_size_lg, |posi| alt[posi]);
        let is_ocean_fn = |posi: usize| is_ocean[posi];
        let mut dh = downhill(map_size_lg, |posi| alt[posi], is_ocean_fn);
        let (boundary_len, indirection, water_alt_pos, maxh) =
            get_lakes(map_size_lg, |posi| alt[posi], &mut dh);
        debug!(?maxh, "Max height");
        let (mrec, mstack, mwrec) = {
            let mut wh = vec![0.0; map_size_lg.chunks_len()];
            get_multi_rec(
                map_size_lg,
                |posi| alt[posi],
                &dh,
                &water_alt_pos,
                &mut wh,
                usize::from(map_size_lg.chunks().x),
                usize::from(map_size_lg.chunks().y),
                TerrainChunkSize::RECT_SIZE.x as Compute,
                TerrainChunkSize::RECT_SIZE.y as Compute,
                maxh,
                threadpool,
            )
        };
        let flux_old = get_multi_drainage(map_size_lg, &mstack, &mrec, &*mwrec, boundary_len);
        // let flux_rivers = get_drainage(map_size_lg, &water_alt_pos, &dh,
        // boundary_len); TODO: Make rivers work with multi-direction flux as
        // well.
        let flux_rivers = flux_old.clone();

        let water_height_initial = |chunk_idx| {
            let indirection_idx = indirection[chunk_idx];
            // Find the lake this point is flowing into.
            let lake_idx = if indirection_idx < 0 {
                chunk_idx
            } else {
                indirection_idx as usize
            };
            let chunk_water_alt = if dh[lake_idx] < 0 {
                // This is either a boundary node (dh[chunk_idx] == -2, i.e. water is at sea
                // level) or part of a lake that flows directly into the ocean.
                // In the former case, water is at sea level so we just return
                // 0.0.  In the latter case, the lake bottom must have been a
                // boundary node in the first place--meaning this node flows directly
                // into the ocean.  In that case, its lake bottom is ocean, meaning its water is
                // also at sea level.  Thus, we return 0.0 in both cases.
                0.0
            } else {
                // This chunk is draining into a body of water that isn't the ocean (i.e., a
                // lake). Then we just need to find the pass height of the
                // surrounding lake in order to figure out the initial water
                // height (which fill_sinks will then extend to make
                // sure it fills the entire basin).

                // Find the height of "our" side of the pass (the part of it that drains into
                // this chunk's lake).
                let pass_idx = -indirection[lake_idx] as usize;
                let pass_height_i = alt[pass_idx];
                // Find the pass this lake is flowing into (i.e. water at the lake bottom gets
                // pushed towards the point identified by pass_idx).
                let neighbor_pass_idx = dh[pass_idx/*lake_idx*/];
                // Find the height of the pass into which our lake is flowing.
                let pass_height_j = alt[neighbor_pass_idx as usize];
                // Find the maximum of these two heights.
                // Use the pass height as the initial water altitude.
                pass_height_i.max(pass_height_j) /*pass_height*/
            };
            // Use the maximum of the pass height and chunk height as the parameter to
            // fill_sinks.
            let chunk_alt = alt[chunk_idx];
            chunk_alt.max(chunk_water_alt)
        };

        // NOTE: If for for some reason you need to avoid the expensive `fill_sinks`
        // step here, and we haven't yet replaced it with a faster version, you
        // may comment out this line and replace it with the commented-out code
        // below; however, there are no guarantees that this
        // will work correctly.
        let water_alt = fill_sinks(map_size_lg, water_height_initial, is_ocean_fn);
        /* let water_alt = (0..map_size_lg.chunks_len())
        .into_par_iter()
        .map(|posi| water_height_initial(posi))
        .collect::<Vec<_>>(); */

        let rivers = get_rivers(
            map_size_lg,
            continent_scale_hack,
            &water_alt_pos,
            &water_alt,
            &dh,
            &indirection,
            &flux_rivers,
        );

        let water_alt = indirection
            .par_iter()
            .enumerate()
            .map(|(chunk_idx, &indirection_idx)| {
                // Find the lake this point is flowing into.
                let lake_idx = if indirection_idx < 0 {
                    chunk_idx
                } else {
                    indirection_idx as usize
                };
                if dh[lake_idx] < 0 {
                    // This is either a boundary node (dh[chunk_idx] == -2, i.e. water is at sea
                    // level) or part of a lake that flows directly into the
                    // ocean.  In the former case, water is at sea level so we
                    // just return 0.0.  In the latter case, the lake bottom must
                    // have been a boundary node in the first place--meaning this node flows
                    // directly into the ocean.  In that case, its lake bottom
                    // is ocean, meaning its water is also at sea level.  Thus,
                    // we return 0.0 in both cases.
                    0.0
                } else {
                    // This is not flowing into the ocean, so we can use the existing water_alt.
                    water_alt[chunk_idx] as f32
                }
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let is_underwater = |chunk_idx: usize| match rivers[chunk_idx].river_kind {
            Some(RiverKind::Ocean) | Some(RiverKind::Lake { .. }) => true,
            Some(RiverKind::River { .. }) => false, // TODO: inspect width
            None => false,
        };

        // Check whether any tiles around this tile are not water (since Lerp will
        // ensure that they are included).
        let pure_water = |posi: usize| {
            let pos = uniform_idx_as_vec2(map_size_lg, posi);
            for x in pos.x - 1..(pos.x + 1) + 1 {
                for y in pos.y - 1..(pos.y + 1) + 1 {
                    if x >= 0
                        && y >= 0
                        && x < map_size_lg.chunks().x as i32
                        && y < map_size_lg.chunks().y as i32
                    {
                        let posi = vec2_as_uniform_idx(map_size_lg, Vec2::new(x, y));
                        if !is_underwater(posi) {
                            return false;
                        }
                    }
                }
            }
            true
        };

        // NaNs in these uniform vectors wherever pure_water() returns true.
        let (((alt_no_water, _), (pure_flux, _)), ((temp_base, _), (humid_base, _))) = threadpool
            .join(
                || {
                    threadpool.join(
                        || {
                            uniform_noise(map_size_lg, |posi, _| {
                                if pure_water(posi) {
                                    None
                                } else {
                                    // A version of alt that is uniform over *non-water* (or
                                    // land-adjacent water) chunks.
                                    Some(alt[posi] as f32)
                                }
                            })
                        },
                        || {
                            uniform_noise(map_size_lg, |posi, _| {
                                if pure_water(posi) {
                                    None
                                } else {
                                    Some(flux_old[posi])
                                }
                            })
                        },
                    )
                },
                || {
                    threadpool.join(
                        || {
                            uniform_noise(map_size_lg, |posi, wposf| {
                                if pure_water(posi) {
                                    None
                                } else {
                                    // -1 to 1.
                                    Some(gen_ctx.temp_nz.get((wposf).into_array()) as f32)
                                }
                            })
                        },
                        || {
                            uniform_noise(map_size_lg, |posi, wposf| {
                                // Check whether any tiles around this tile are water.
                                if pure_water(posi) {
                                    None
                                } else {
                                    // 0 to 1, hopefully.
                                    Some(
                                        (gen_ctx.humid_nz.get(wposf.div(1024.0).into_array())
                                            as f32)
                                            .add(1.0)
                                            .mul(0.5),
                                    )
                                }
                            })
                        },
                    )
                },
            );

        let gen_cdf = GenCdf {
            humid_base,
            temp_base,
            chaos,
            alt,
            basement,
            water_alt,
            dh,
            flux: flux_old,
            pure_flux,
            alt_no_water,
            rivers,
        };

        let chunks = (0..map_size_lg.chunks_len())
            .into_par_iter()
            .map(|i| SimChunk::generate(map_size_lg, i, &gen_ctx, &gen_cdf))
            .collect::<Vec<_>>();

        let mut this = Self {
            seed,
            map_size_lg,
            max_height: maxh as f32,
            chunks,
            _locations: Vec::new(),
            gen_ctx,
            rng,
            calendar,
        };

        this.generate_cliffs();

        if opts.seed_elements {
            this.seed_elements();
        }

        this
    }

    #[inline(always)]
    pub const fn map_size_lg(&self) -> MapSizeLg { self.map_size_lg }

    pub fn get_size(&self) -> Vec2<u32> { self.map_size_lg().chunks().map(u32::from) }

    /// Draw a map of the world based on chunk information.  Returns a buffer of
    /// u32s.
    pub fn get_map(&self, index: IndexRef, calendar: Option<&Calendar>) -> WorldMapMsg {
        let mut map_config = MapConfig::orthographic(
            self.map_size_lg(),
            core::ops::RangeInclusive::new(CONFIG.sea_level, CONFIG.sea_level + self.max_height),
        );
        // Build a horizon map.
        let scale_angle = |angle: Alt| {
            (/* 0.0.max( */angle /* ) */
                .atan()
                * <Alt as FloatConst>::FRAC_2_PI()
                * 255.0)
                .floor() as u8
        };
        let scale_height = |height: Alt| {
            (/* 0.0.max( */height/*)*/ as Alt * 255.0 / self.max_height as Alt).floor() as u8
        };

        let samples_data = {
            let column_sample = ColumnGen::new(self);
            (0..self.map_size_lg().chunks_len())
                .into_par_iter()
                .map_init(
                    || Box::new(BlockGen::new(ColumnGen::new(self))),
                    |_block_gen, posi| {
                        let sample = column_sample.get(
                            (
                                uniform_idx_as_vec2(self.map_size_lg(), posi) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
                                index,
                                calendar,
                            )
                        )?;
                        // sample.water_level = CONFIG.sea_level.max(sample.water_level);

                        Some(sample)
                    },
                )
                /* .map(|posi| {
                    let mut sample = column_sample.get(
                        uniform_idx_as_vec2(self.map_size_lg(), posi) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
                    );
                }) */
                .collect::<Vec<_>>()
                .into_boxed_slice()
        };

        let horizons = get_horizon_map(
            self.map_size_lg(),
            Aabr {
                min: Vec2::zero(),
                max: self.map_size_lg().chunks().map(|e| e as i32),
            },
            CONFIG.sea_level,
            CONFIG.sea_level + self.max_height,
            |posi| {
                /* let chunk = &self.chunks[posi];
                chunk.alt.max(chunk.water_alt) as Alt */
                let sample = samples_data[posi].as_ref();
                sample
                    .map(|s| s.alt.max(s.water_level))
                    .unwrap_or(CONFIG.sea_level)
            },
            |a| scale_angle(a.into()),
            |h| scale_height(h.into()),
        )
        .unwrap();

        let mut v = vec![0u32; self.map_size_lg().chunks_len()];
        let mut alts = vec![0u32; self.map_size_lg().chunks_len()];
        // TODO: Parallelize again.
        map_config.is_shaded = false;

        map_config.generate(
            |pos| sample_pos(&map_config, self, Some(&samples_data), pos),
            |pos| sample_wpos(&map_config, self, pos),
            |pos, (r, g, b, _a)| {
                // We currently ignore alpha and replace it with the height at pos, scaled to
                // u8.
                let alt = sample_wpos(
                    &map_config,
                    self,
                    pos.map(|e| e as i32) * TerrainChunkSize::RECT_SIZE.map(|e| e as i32),
                );
                let a = 0; //(alt.min(1.0).max(0.0) * 255.0) as u8;

                // NOTE: Safe by invariants on map_size_lg.
                let posi = (pos.y << self.map_size_lg().vec().x) | pos.x;
                v[posi] = u32::from_le_bytes([r, g, b, a]);
                alts[posi] = (((alt.min(1.0).max(0.0) * 8191.0) as u32) & 0x1FFF) << 3;
            },
        );
        WorldMapMsg {
            dimensions_lg: self.map_size_lg().vec(),
            sea_level: CONFIG.sea_level,
            max_height: self.max_height,
            rgba: Grid::from_raw(self.get_size().map(|e| e as i32), v),
            alt: Grid::from_raw(self.get_size().map(|e| e as i32), alts),
            horizons,
            sites: Vec::new(), // Will be substituted later
            pois: Vec::new(),  // Will be substituted later
        }
    }

    pub fn generate_cliffs(&mut self) {
        let mut rng = self.rng.clone();

        for _ in 0..self.get_size().product() / 10 {
            let mut pos = self.get_size().map(|e| rng.gen_range(0..e) as i32);

            let mut cliffs = DHashSet::default();
            let mut cliff_path = Vec::new();

            for _ in 0..64 {
                if self.get_gradient_approx(pos).map_or(false, |g| g > 1.5) {
                    if !cliffs.insert(pos) {
                        break;
                    }
                    cliff_path.push((pos, 0.0));

                    pos += CARDINALS
                        .iter()
                        .copied()
                        .max_by_key(|rpos| {
                            self.get_gradient_approx(pos + rpos)
                                .map_or(0, |g| (g * 1000.0) as i32)
                        })
                        .unwrap(); // Can't fail
                } else {
                    break;
                }
            }

            for cliff in cliffs {
                Spiral2d::new()
                    .take((4usize * 2 + 1).pow(2))
                    .for_each(|rpos| {
                        let dist = rpos.map(|e| e as f32).magnitude();
                        if let Some(c) = self.get_mut(cliff + rpos) {
                            let warp = 1.0 / (1.0 + dist);
                            if !c.river.near_water() {
                                c.tree_density *= 1.0 - warp;
                                c.cliff_height = Lerp::lerp(44.0, 0.0, -1.0 + dist / 3.5);
                            }
                        }
                    });
            }
        }
    }

    /// Prepare the world for simulation
    pub fn seed_elements(&mut self) {
        let mut rng = self.rng.clone();

        let cell_size = 16;
        let grid_size = self.map_size_lg().chunks().map(usize::from) / cell_size;
        let loc_count = 100;

        let mut loc_grid = vec![None; grid_size.product()];
        let mut locations = Vec::new();

        // Seed the world with some locations
        (0..loc_count).for_each(|_| {
            let cell_pos = Vec2::new(
                self.rng.gen::<usize>() % grid_size.x,
                self.rng.gen::<usize>() % grid_size.y,
            );
            let wpos = (cell_pos * cell_size + cell_size / 2)
                .map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
                    e as i32 * sz as i32 + sz as i32 / 2
                });

            locations.push(Location::generate(wpos, &mut rng));

            loc_grid[cell_pos.y * grid_size.x + cell_pos.x] = Some(locations.len() - 1);
        });

        // Find neighbours
        let mut loc_clone = locations
            .iter()
            .map(|l| l.center)
            .enumerate()
            .collect::<Vec<_>>();
        // NOTE: We assume that usize is 8 or fewer bytes.
        (0..locations.len()).for_each(|i| {
            let pos = locations[i].center.map(|e| e as i64);

            loc_clone.sort_by_key(|(_, l)| l.map(|e| e as i64).distance_squared(pos));

            loc_clone.iter().skip(1).take(2).for_each(|(j, _)| {
                locations[i].neighbours.insert(*j as u64);
                locations[*j].neighbours.insert(i as u64);
            });
        });

        // Simulate invasion!
        let invasion_cycles = 25;
        (0..invasion_cycles).for_each(|_| {
            (0..grid_size.y).for_each(|j| {
                (0..grid_size.x).for_each(|i| {
                    if loc_grid[j * grid_size.x + i].is_none() {
                        const R_COORDS: [i32; 5] = [-1, 0, 1, 0, -1];
                        let idx = self.rng.gen::<usize>() % 4;
                        let new_i = i as i32 + R_COORDS[idx];
                        let new_j = j as i32 + R_COORDS[idx + 1];
                        if new_i >= 0 && new_j >= 0 {
                            let loc = Vec2::new(new_i as usize, new_j as usize);
                            loc_grid[j * grid_size.x + i] =
                                loc_grid.get(loc.y * grid_size.x + loc.x).cloned().flatten();
                        }
                    }
                });
            });
        });

        // Place the locations onto the world
        /*
        let gen = StructureGen2d::new(self.seed, cell_size as u32, cell_size as u32 / 2);

        self.chunks
            .par_iter_mut()
            .enumerate()
            .for_each(|(ij, chunk)| {
                let chunk_pos = uniform_idx_as_vec2(self.map_size_lg(), ij);
                let i = chunk_pos.x as usize;
                let j = chunk_pos.y as usize;
                let block_pos = Vec2::new(
                    chunk_pos.x * TerrainChunkSize::RECT_SIZE.x as i32,
                    chunk_pos.y * TerrainChunkSize::RECT_SIZE.y as i32,
                );
                let _cell_pos = Vec2::new(i / cell_size, j / cell_size);

                // Find the distance to each region
                let near = gen.get(chunk_pos);
                let mut near = near
                    .iter()
                    .map(|(pos, seed)| RegionInfo {
                        chunk_pos: *pos,
                        block_pos: pos
                            .map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e * sz as i32),
                        dist: (pos - chunk_pos).map(|e| e as f32).magnitude(),
                        seed: *seed,
                    })
                    .collect::<Vec<_>>();

                // Sort regions based on distance
                near.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());

                let nearest_cell_pos = near[0].chunk_pos;
                if nearest_cell_pos.x >= 0 && nearest_cell_pos.y >= 0 {
                    let nearest_cell_pos = nearest_cell_pos.map(|e| e as usize) / cell_size;
                    chunk.location = loc_grid
                        .get(nearest_cell_pos.y * grid_size.x + nearest_cell_pos.x)
                        .cloned()
                        .unwrap_or(None)
                        .map(|loc_idx| LocationInfo { loc_idx, near });
                }
            });
        */

        // Create waypoints
        const WAYPOINT_EVERY: usize = 16;
        let this = &self;
        let waypoints = (0..this.map_size_lg().chunks().x)
            .step_by(WAYPOINT_EVERY)
            .flat_map(|i| {
                (0..this.map_size_lg().chunks().y)
                    .step_by(WAYPOINT_EVERY)
                    .map(move |j| (i, j))
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .filter_map(|(i, j)| {
                let mut pos = Vec2::new(i as i32, j as i32);
                let mut chunk = this.get(pos)?;

                if chunk.is_underwater() {
                    return None;
                }
                // Slide the waypoints down hills
                const MAX_ITERS: usize = 64;
                for _ in 0..MAX_ITERS {
                    let downhill_pos = match chunk.downhill {
                        Some(downhill) => {
                            downhill.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / (sz as i32))
                        },
                        None => return Some(pos),
                    };

                    let new_chunk = this.get(downhill_pos)?;
                    const SLIDE_THRESHOLD: f32 = 5.0;
                    if new_chunk.river.near_water() || new_chunk.alt + SLIDE_THRESHOLD < chunk.alt {
                        break;
                    } else {
                        chunk = new_chunk;
                        pos = downhill_pos;
                    }
                }
                Some(pos)
            })
            .collect::<Vec<_>>();

        for waypoint in waypoints {
            self.get_mut(waypoint).map(|sc| sc.contains_waypoint = true);
        }

        self.rng = rng;
        self._locations = locations;
    }

    pub fn get(&self, chunk_pos: Vec2<i32>) -> Option<&SimChunk> {
        if chunk_pos
            .map2(self.map_size_lg().chunks(), |e, sz| e >= 0 && e < sz as i32)
            .reduce_and()
        {
            Some(&self.chunks[vec2_as_uniform_idx(self.map_size_lg(), chunk_pos)])
        } else {
            None
        }
    }

    pub fn get_gradient_approx(&self, chunk_pos: Vec2<i32>) -> Option<f32> {
        let a = self.get(chunk_pos)?;
        if let Some(downhill) = a.downhill {
            let b =
                self.get(downhill.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| e / (sz as i32)))?;
            Some((a.alt - b.alt).abs() / TerrainChunkSize::RECT_SIZE.x as f32)
        } else {
            Some(0.0)
        }
    }

    pub fn get_alt_approx(&self, wpos: Vec2<i32>) -> Option<f32> {
        self.get_interpolated(wpos, |chunk| chunk.alt)
    }

    pub fn get_wpos(&self, wpos: Vec2<i32>) -> Option<&SimChunk> {
        self.get(wpos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
            e.div_euclid(sz as i32)
        }))
    }

    pub fn get_mut(&mut self, chunk_pos: Vec2<i32>) -> Option<&mut SimChunk> {
        let map_size_lg = self.map_size_lg();
        if chunk_pos
            .map2(map_size_lg.chunks(), |e, sz| e >= 0 && e < sz as i32)
            .reduce_and()
        {
            Some(&mut self.chunks[vec2_as_uniform_idx(map_size_lg, chunk_pos)])
        } else {
            None
        }
    }

    pub fn get_base_z(&self, chunk_pos: Vec2<i32>) -> Option<f32> {
        let in_bounds = chunk_pos
            .map2(self.map_size_lg().chunks(), |e, sz| {
                e > 0 && e < sz as i32 - 2
            })
            .reduce_and();
        if !in_bounds {
            return None;
        }

        let chunk_idx = vec2_as_uniform_idx(self.map_size_lg(), chunk_pos);
        local_cells(self.map_size_lg(), chunk_idx)
            .flat_map(|neighbor_idx| {
                let neighbor_pos = uniform_idx_as_vec2(self.map_size_lg(), neighbor_idx);
                let neighbor_chunk = self.get(neighbor_pos);
                let river_kind = neighbor_chunk.and_then(|c| c.river.river_kind);
                let has_water = river_kind.is_some() && river_kind != Some(RiverKind::Ocean);
                if (neighbor_pos - chunk_pos).reduce_partial_max() <= 1 || has_water {
                    neighbor_chunk.map(|c| c.get_base_z())
                } else {
                    None
                }
            })
            .fold(None, |a: Option<f32>, x| a.map(|a| a.min(x)).or(Some(x)))
    }

    pub fn get_interpolated<T, F>(&self, pos: Vec2<i32>, mut f: F) -> Option<T>
    where
        T: Copy + Default + Add<Output = T> + Mul<f32, Output = T>,
        F: FnMut(&SimChunk) -> T,
    {
        let pos = pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
            e as f64 / sz as f64
        });

        let cubic = |a: T, b: T, c: T, d: T, x: f32| -> T {
            let x2 = x * x;

            // Catmull-Rom splines
            let co0 = a * -0.5 + b * 1.5 + c * -1.5 + d * 0.5;
            let co1 = a + b * -2.5 + c * 2.0 + d * -0.5;
            let co2 = a * -0.5 + c * 0.5;
            let co3 = b;

            co0 * x2 * x + co1 * x2 + co2 * x + co3
        };

        let mut x = [T::default(); 4];

        for (x_idx, j) in (-1..3).enumerate() {
            let y0 = f(self.get(pos.map2(Vec2::new(j, -1), |e, q| e.max(0.0) as i32 + q))?);
            let y1 = f(self.get(pos.map2(Vec2::new(j, 0), |e, q| e.max(0.0) as i32 + q))?);
            let y2 = f(self.get(pos.map2(Vec2::new(j, 1), |e, q| e.max(0.0) as i32 + q))?);
            let y3 = f(self.get(pos.map2(Vec2::new(j, 2), |e, q| e.max(0.0) as i32 + q))?);

            x[x_idx] = cubic(y0, y1, y2, y3, pos.y.fract() as f32);
        }

        Some(cubic(x[0], x[1], x[2], x[3], pos.x.fract() as f32))
    }

    /// M. Steffen splines.
    ///
    /// A more expensive cubic interpolation function that can preserve
    /// monotonicity between points.  This is useful if you rely on relative
    /// differences between endpoints being preserved at all interior
    /// points.  For example, we use this with riverbeds (and water
    /// height on along rivers) to maintain the invariant that the rivers always
    /// flow downhill at interior points (not just endpoints), without
    /// needing to flatten out the river.
    pub fn get_interpolated_monotone<T, F>(&self, pos: Vec2<i32>, mut f: F) -> Option<T>
    where
        T: Copy + Default + Signed + Float + Add<Output = T> + Mul<f32, Output = T>,
        F: FnMut(&SimChunk) -> T,
    {
        // See http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990A%26A...239..443S&defaultprint=YES&page_ind=0&filetype=.pdf
        //
        // Note that these are only guaranteed monotone in one dimension; fortunately,
        // that is sufficient for our purposes.
        let pos = pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
            e as f64 / sz as f64
        });

        let secant = |b: T, c: T| c - b;

        let parabola = |a: T, c: T| -a * 0.5 + c * 0.5;

        let slope = |_a: T, _b: T, _c: T, s_a: T, s_b: T, p_b: T| {
            // ((b - a).signum() + (c - b).signum()) * s
            (s_a.signum() + s_b.signum()) * (s_a.abs().min(s_b.abs()).min(p_b.abs() * 0.5))
        };

        let cubic = |a: T, b: T, c: T, d: T, x: f32| -> T {
            // Compute secants.
            let s_a = secant(a, b);
            let s_b = secant(b, c);
            let s_c = secant(c, d);
            // Computing slopes from parabolas.
            let p_b = parabola(a, c);
            let p_c = parabola(b, d);
            // Get slopes (setting distance between neighbors to 1.0).
            let slope_b = slope(a, b, c, s_a, s_b, p_b);
            let slope_c = slope(b, c, d, s_b, s_c, p_c);
            let x2 = x * x;

            // Interpolating splines.
            let co0 = slope_b + slope_c - s_b * 2.0;
            // = a * -0.5 + c * 0.5 + b * -0.5 + d * 0.5 - 2 * (c - b)
            // = a * -0.5 + b * 1.5 - c * 1.5 + d * 0.5;
            let co1 = s_b * 3.0 - slope_b * 2.0 - slope_c;
            // = (3.0 * (c - b) - 2.0 * (a * -0.5 + c * 0.5) - (b * -0.5 + d * 0.5))
            // = a + b * -2.5 + c * 2.0 + d * -0.5;
            let co2 = slope_b;
            // = a * -0.5 + c * 0.5;
            let co3 = b;

            co0 * x2 * x + co1 * x2 + co2 * x + co3
        };

        let mut x = [T::default(); 4];

        for (x_idx, j) in (-1..3).enumerate() {
            let y0 = f(self.get(pos.map2(Vec2::new(j, -1), |e, q| e.max(0.0) as i32 + q))?);
            let y1 = f(self.get(pos.map2(Vec2::new(j, 0), |e, q| e.max(0.0) as i32 + q))?);
            let y2 = f(self.get(pos.map2(Vec2::new(j, 1), |e, q| e.max(0.0) as i32 + q))?);
            let y3 = f(self.get(pos.map2(Vec2::new(j, 2), |e, q| e.max(0.0) as i32 + q))?);

            x[x_idx] = cubic(y0, y1, y2, y3, pos.y.fract() as f32);
        }

        Some(cubic(x[0], x[1], x[2], x[3], pos.x.fract() as f32))
    }

    /// Bilinear interpolation.
    ///
    /// Linear interpolation in both directions (i.e. quadratic interpolation).
    pub fn get_interpolated_bilinear<T, F>(&self, pos: Vec2<i32>, mut f: F) -> Option<T>
    where
        T: Copy + Default + Signed + Float + Add<Output = T> + Mul<f32, Output = T>,
        F: FnMut(&SimChunk) -> T,
    {
        // (i) Find downhill for all four points.
        // (ii) Compute distance from each downhill point and do linear interpolation on
        // their heights. (iii) Compute distance between each neighboring point
        // and do linear interpolation on       their distance-interpolated
        // heights.

        // See http://articles.adsabs.harvard.edu/cgi-bin/nph-iarticle_query?1990A%26A...239..443S&defaultprint=YES&page_ind=0&filetype=.pdf
        //
        // Note that these are only guaranteed monotone in one dimension; fortunately,
        // that is sufficient for our purposes.
        let pos = pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
            e as f64 / sz as f64
        });

        // Orient the chunk in the direction of the most downhill point of the four.  If
        // there is no "most downhill" point, then we don't care.
        let x0 = pos.map2(Vec2::new(0, 0), |e, q| e.max(0.0) as i32 + q);
        let p0 = self.get(x0)?;
        let y0 = f(p0);

        let x1 = pos.map2(Vec2::new(1, 0), |e, q| e.max(0.0) as i32 + q);
        let p1 = self.get(x1)?;
        let y1 = f(p1);

        let x2 = pos.map2(Vec2::new(0, 1), |e, q| e.max(0.0) as i32 + q);
        let p2 = self.get(x2)?;
        let y2 = f(p2);

        let x3 = pos.map2(Vec2::new(1, 1), |e, q| e.max(0.0) as i32 + q);
        let p3 = self.get(x3)?;
        let y3 = f(p3);

        let z0 = y0
            .mul(1.0 - pos.x.fract() as f32)
            .mul(1.0 - pos.y.fract() as f32);
        let z1 = y1.mul(pos.x.fract() as f32).mul(1.0 - pos.y.fract() as f32);
        let z2 = y2.mul(1.0 - pos.x.fract() as f32).mul(pos.y.fract() as f32);
        let z3 = y3.mul(pos.x.fract() as f32).mul(pos.y.fract() as f32);

        Some(z0 + z1 + z2 + z3)
    }

    /// Return the distance to the nearest way in blocks, along with the
    /// closest point on the way, the way metadata, and the tangent vector
    /// of that way.
    pub fn get_nearest_way<M: Clone + Lerp<Output = M>>(
        &self,
        wpos: Vec2<i32>,
        get_way: impl Fn(&SimChunk) -> Option<(Way, M)>,
    ) -> Option<(f32, Vec2<f32>, M, Vec2<f32>)> {
        let chunk_pos = wpos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
            e.div_euclid(sz as i32)
        });
        let get_chunk_centre = |chunk_pos: Vec2<i32>| {
            chunk_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz: u32| {
                e * sz as i32 + sz as i32 / 2
            })
        };

        let get_way = &get_way;
        LOCALITY
            .iter()
            .filter_map(|ctrl| {
                let (way, meta) = get_way(self.get(chunk_pos + *ctrl)?)?;
                let ctrl_pos = get_chunk_centre(chunk_pos + *ctrl).map(|e| e as f32)
                    + way.offset.map(|e| e as f32);

                let chunk_connections = way.neighbors.count_ones();
                if chunk_connections == 0 {
                    return None;
                }

                let (start_pos, _start_idx, start_meta) = if chunk_connections != 2 {
                    (ctrl_pos, None, meta.clone())
                } else {
                    let (start_idx, start_rpos) = NEIGHBORS
                        .iter()
                        .copied()
                        .enumerate()
                        .find(|(i, _)| way.neighbors & (1 << *i as u8) != 0)
                        .unwrap();
                    let start_pos_chunk = chunk_pos + *ctrl + start_rpos;
                    let (start_way, start_meta) = get_way(self.get(start_pos_chunk)?)?;
                    (
                        get_chunk_centre(start_pos_chunk).map(|e| e as f32)
                            + start_way.offset.map(|e| e as f32),
                        Some(start_idx),
                        start_meta,
                    )
                };

                Some(
                    NEIGHBORS
                        .iter()
                        .enumerate()
                        .filter(move |(i, _)| way.neighbors & (1 << *i as u8) != 0)
                        .filter_map(move |(_, end_rpos)| {
                            let end_pos_chunk = chunk_pos + *ctrl + end_rpos;
                            let (end_way, end_meta) = get_way(self.get(end_pos_chunk)?)?;
                            let end_pos = get_chunk_centre(end_pos_chunk).map(|e| e as f32)
                                + end_way.offset.map(|e| e as f32);

                            let bez = QuadraticBezier2 {
                                start: (start_pos + ctrl_pos) / 2.0,
                                ctrl: ctrl_pos,
                                end: (end_pos + ctrl_pos) / 2.0,
                            };
                            let nearest_interval = bez
                                .binary_search_point_by_steps(wpos.map(|e| e as f32), 16, 0.001)
                                .0
                                .clamped(0.0, 1.0);
                            let pos = bez.evaluate(nearest_interval);
                            let dist_sqrd = pos.distance_squared(wpos.map(|e| e as f32));
                            let meta = if nearest_interval < 0.5 {
                                Lerp::lerp(start_meta.clone(), meta.clone(), 0.5 + nearest_interval)
                            } else {
                                Lerp::lerp(meta.clone(), end_meta, nearest_interval - 0.5)
                            };
                            Some((dist_sqrd, pos, meta, move || {
                                bez.evaluate_derivative(nearest_interval).normalized()
                            }))
                        }),
                )
            })
            .flatten()
            .min_by_key(|(dist_sqrd, _, _, _)| (dist_sqrd * 1024.0) as i32)
            .map(|(dist, pos, meta, calc_tangent)| (dist.sqrt(), pos, meta, calc_tangent()))
    }

    pub fn get_nearest_path(&self, wpos: Vec2<i32>) -> Option<(f32, Vec2<f32>, Path, Vec2<f32>)> {
        self.get_nearest_way(wpos, |chunk| Some(chunk.path))
    }

    pub fn get_nearest_cave(&self, wpos: Vec2<i32>) -> Option<(f32, Vec2<f32>, Cave, Vec2<f32>)> {
        self.get_nearest_way(wpos, |chunk| Some(chunk.cave))
    }

    /// Create a [`Lottery<Option<ForestKind>>`] that generates [`ForestKind`]s
    /// according to the conditions at the given position. If no or fewer
    /// trees are appropriate for the conditions, `None` may be generated.
    pub fn make_forest_lottery(&self, wpos: Vec2<i32>) -> Lottery<Option<ForestKind>> {
        let chunk = if let Some(chunk) = self.get_wpos(wpos) {
            chunk
        } else {
            return Lottery::from(vec![(1.0, None)]);
        };
        let env = chunk.get_environment();
        Lottery::from(
            ForestKind::into_enum_iter()
                .enumerate()
                .map(|(i, fk)| {
                    const CLUSTER_SIZE: f64 = 48.0;
                    let nz = (FastNoise2d::new(i as u32 * 37)
                        .get(wpos.map(|e| e as f64) / CLUSTER_SIZE)
                        + 1.0)
                        / 2.0;
                    (fk.proclivity(&env) * nz, Some(fk))
                })
                .chain(std::iter::once((0.001, None)))
                .collect::<Vec<_>>(),
        )
    }

    /// WARNING: Not currently used by the tree layer. Needs to be reworked.
    /// Return an iterator over candidate tree positions (note that only some of
    /// these will become trees since environmental parameters may forbid
    /// them spawning).
    pub fn get_near_trees(&self, wpos: Vec2<i32>) -> impl Iterator<Item = TreeAttr> + '_ {
        // Deterministic based on wpos
        self.gen_ctx
            .structure_gen
            .get(wpos)
            .into_iter()
            .filter_map(move |(wpos, seed)| {
                let lottery = self.make_forest_lottery(wpos);
                Some(TreeAttr {
                    pos: wpos,
                    seed,
                    scale: 1.0,
                    forest_kind: *lottery.choose_seeded(seed).as_ref()?,
                    inhabited: false,
                })
            })
    }

    pub fn get_area_trees(
        &self,
        wpos_min: Vec2<i32>,
        wpos_max: Vec2<i32>,
    ) -> impl ParallelIterator<Item = TreeAttr> + '_ {
        self.gen_ctx
            .structure_gen
            .par_iter(wpos_min, wpos_max)
            .filter_map(move |(wpos, seed)| {
                let lottery = self.make_forest_lottery(wpos);
                Some(TreeAttr {
                    pos: wpos,
                    seed,
                    scale: 1.0,
                    forest_kind: *lottery.choose_seeded(seed).as_ref()?,
                    inhabited: false,
                })
            })
    }
}

#[derive(Debug)]
pub struct SimChunk {
    pub chaos: f32,
    pub alt: f32,
    pub basement: f32,
    pub water_alt: f32,
    pub downhill: Option<Vec2<i32>>,
    pub flux: f32,
    pub temp: f32,
    pub humidity: f32,
    pub rockiness: f32,
    pub tree_density: f32,
    pub forest_kind: ForestKind,
    pub spawn_rate: f32,
    pub river: RiverData,
    pub surface_veg: f32,

    pub sites: Vec<Id<Site>>,
    pub place: Option<Id<Place>>,
    pub poi: Option<Id<PointOfInterest>>,

    pub path: (Way, Path),
    pub cave: (Way, Cave),
    pub cliff_height: f32,
    pub spot: Option<Spot>,

    pub contains_waypoint: bool,
}

#[derive(Copy, Clone)]
pub struct RegionInfo {
    pub chunk_pos: Vec2<i32>,
    pub block_pos: Vec2<i32>,
    pub dist: f32,
    pub seed: u32,
}

impl SimChunk {
    fn generate(map_size_lg: MapSizeLg, posi: usize, gen_ctx: &GenCtx, gen_cdf: &GenCdf) -> Self {
        let pos = uniform_idx_as_vec2(map_size_lg, posi);
        let wposf = (pos * TerrainChunkSize::RECT_SIZE.map(|e| e as i32)).map(|e| e as f64);

        let (_, chaos) = gen_cdf.chaos[posi];
        let alt_pre = gen_cdf.alt[posi] as f32;
        let basement_pre = gen_cdf.basement[posi] as f32;
        let water_alt_pre = gen_cdf.water_alt[posi];
        let downhill_pre = gen_cdf.dh[posi];
        let flux = gen_cdf.flux[posi] as f32;
        let river = gen_cdf.rivers[posi].clone();

        // Can have NaNs in non-uniform part where pure_water returned true.  We just
        // test one of the four in order to find out whether this is the case.
        let (flux_uniform, /* flux_non_uniform */ _) = gen_cdf.pure_flux[posi];
        let (alt_uniform, _) = gen_cdf.alt_no_water[posi];
        let (temp_uniform, _) = gen_cdf.temp_base[posi];
        let (humid_uniform, _) = gen_cdf.humid_base[posi];

        /* // Vertical difference from the equator (NOTE: "uniform" with much lower granularity than
        // other uniform quantities, but hopefully this doesn't matter *too* much--if it does, we
        // can always add a small x component).
        //
        // Not clear that we want this yet, let's see.
        let latitude_uniform = (pos.y as f32 / f32::from(self.map_size_lg().chunks().y)).sub(0.5).mul(2.0);

        // Even less granular--if this matters we can make the sign affect the quantity slightly.
        let abs_lat_uniform = latitude_uniform.abs(); */

        // We also correlate temperature negatively with altitude and absolute latitude,
        // using different weighting than we use for humidity.
        const TEMP_WEIGHTS: [f32; 3] = [/* 1.5, */ 1.0, 2.0, 1.0];
        let temp = cdf_irwin_hall(
            &TEMP_WEIGHTS,
            [
                temp_uniform,
                1.0 - alt_uniform, /* 1.0 - abs_lat_uniform*/
                (gen_ctx.rock_nz.get((wposf.div(50000.0)).into_array()) as f32 * 2.5 + 1.0) * 0.5,
            ],
        )
        // Convert to [-1, 1]
        .sub(0.5)
        .mul(2.0);

        // Take the weighted average of our randomly generated base humidity, and the
        // calculated water flux over this point in order to compute humidity.
        const HUMID_WEIGHTS: [f32; 3] = [1.0, 1.0, 0.75];
        let humidity = cdf_irwin_hall(&HUMID_WEIGHTS, [humid_uniform, flux_uniform, 1.0]);
        // Moisture evaporates more in hot places
        let humidity = humidity
            * (1.0
                - (temp - CONFIG.tropical_temp)
                    .max(0.0)
                    .div(1.0 - CONFIG.tropical_temp))
            .max(0.0);

        let mut alt = CONFIG.sea_level.add(alt_pre);
        let basement = CONFIG.sea_level.add(basement_pre);
        let water_alt = CONFIG.sea_level.add(water_alt_pre);
        let downhill = if downhill_pre == -2 {
            None
        } else if downhill_pre < 0 {
            panic!("Uh... shouldn't this never, ever happen?");
        } else {
            Some(
                uniform_idx_as_vec2(map_size_lg, downhill_pre as usize)
                    * TerrainChunkSize::RECT_SIZE.map(|e| e as i32)
                    + TerrainChunkSize::RECT_SIZE.map(|e| e as i32 / 2),
            )
        };

        // Logistic regression.  Make sure x ∈ (0, 1).
        let logit = |x: f64| x.ln() - x.neg().ln_1p();
        // 0.5 + 0.5 * tanh(ln(1 / (1 - 0.1) - 1) / (2 * (sqrt(3)/pi)))
        let logistic_2_base = 3.0f64.sqrt().mul(f64::consts::FRAC_2_PI);
        // Assumes μ = 0, σ = 1
        let logistic_cdf = |x: f64| x.div(logistic_2_base).tanh().mul(0.5).add(0.5);

        let is_underwater = match river.river_kind {
            Some(RiverKind::Ocean) | Some(RiverKind::Lake { .. }) => true,
            Some(RiverKind::River { .. }) => false, // TODO: inspect width
            None => false,
        };
        let river_xy = Vec2::new(river.velocity.x, river.velocity.y).magnitude();
        let river_slope = river.velocity.z / river_xy;
        match river.river_kind {
            Some(RiverKind::River { cross_section }) => {
                if cross_section.x >= 0.5 && cross_section.y >= CONFIG.river_min_height {
                    /* println!(
                        "Big area! Pos area: {:?}, River data: {:?}, slope: {:?}",
                        wposf, river, river_slope
                    ); */
                }
                if river_slope.abs() >= 0.25 && cross_section.x >= 1.0 {
                    let pos_area = wposf;
                    let river_data = &river;
                    debug!(?pos_area, ?river_data, ?river_slope, "Big waterfall!",);
                }
            },
            Some(RiverKind::Lake { .. }) => {
                // Forces lakes to be downhill from the land around them, and adds some noise to
                // the lake bed to make sure it's not too flat.
                let lake_bottom_nz = (gen_ctx.small_nz.get((wposf.div(20.0)).into_array()) as f32)
                    .max(-1.0)
                    .min(1.0)
                    .mul(3.0);
                alt = alt.min(water_alt - 5.0) + lake_bottom_nz;
            },
            _ => {},
        }

        // No trees in the ocean, with zero humidity (currently), or directly on
        // bedrock.
        let tree_density = if is_underwater {
            0.0
        } else {
            let tree_density = (gen_ctx.tree_nz.get((wposf.div(1024.0)).into_array()))
                .mul(1.5)
                .add(1.0)
                .mul(0.5)
                .add(0.05)
                .max(0.0)
                .min(1.0);
            // Tree density should go (by a lot) with humidity.
            if humidity <= 0.0 || tree_density <= 0.0 {
                0.0
            } else if humidity >= 1.0 || tree_density >= 1.0 {
                1.0
            } else {
                // Weighted logit sum.
                logistic_cdf(logit(tree_density))
            }
            // rescale to (-0.95, 0.95)
            .sub(0.5)
            .add(0.5)
        } as f32;
        const MIN_TREE_HUM: f32 = 0.15;
        // Tree density increases exponentially with humidity...
        let tree_density = (tree_density * (humidity - MIN_TREE_HUM).max(0.0).mul(1.0 + MIN_TREE_HUM) / temp.max(0.75))
            // Places that are *too* wet (like marshes) also get fewer trees because the ground isn't stable enough for
            // them.
            //.mul((1.0 - flux * 0.05/*(humidity - 0.9).max(0.0) / 0.1*/).max(0.0))
            .mul(0.25 + flux * 0.05)
            // ...but is ultimately limited by available sunlight (and our tree generation system)
            .min(1.0);

        // Add geologically short timescale undulation to the world for various reasons
        let alt =
            // Don't add undulation to rivers, mainly because this could accidentally result in rivers flowing uphill
            if river.near_water() {
                alt
            } else {
                // Sand dunes (formed over a short period of time, so we don't care about erosion sim)
                let warp = Vec2::new(
                    gen_ctx.turb_x_nz.get(wposf.div(350.0).into_array()) as f32,
                    gen_ctx.turb_y_nz.get(wposf.div(350.0).into_array()) as f32,
                ) * 200.0;
                const DUNE_SCALE: f32 = 24.0;
                const DUNE_LEN: f32 = 96.0;
                const DUNE_DIR: Vec2<f32> = Vec2::new(1.0, 1.0);
                let dune_dist = (wposf.map(|e| e as f32) + warp)
                    .div(DUNE_LEN)
                    .mul(DUNE_DIR.normalized())
                    .sum();
                let dune_nz = 0.5 - dune_dist.sin().abs() + 0.5 * (dune_dist + 0.5).sin().abs();
                let dune = dune_nz * DUNE_SCALE * (temp - 0.75).clamped(0.0, 0.25) * 4.0;

                // Trees bind to soil and their roots result in small accumulating undulations over geologically short
                // periods of time. Forest floors are generally significantly bumpier than that of deforested areas.
                // This is particularly pronounced in high-humidity areas.
                let soil_nz = gen_ctx.hill_nz.get(wposf.div(96.0).into_array()) as f32;
                let soil_nz = (soil_nz + 1.0) * 0.5;
                const SOIL_SCALE: f32 = 16.0;
                let soil = soil_nz * SOIL_SCALE * tree_density.sqrt() * humidity.sqrt();

                let warp_factor = ((alt - CONFIG.sea_level) / 16.0).clamped(0.0, 1.0);

                let warp = (dune + soil) * warp_factor;

                // Prevent warping pushing the altitude underwater
                if alt + warp < water_alt {
                    alt
                } else {
                    alt + warp
                }
            };

        Self {
            chaos,
            flux,
            alt,
            basement: basement.min(alt),
            water_alt,
            downhill,
            temp,
            humidity,
            rockiness: if true {
                (gen_ctx.rock_nz.get((wposf.div(1024.0)).into_array()) as f32)
                    //.add(if river.near_river() { 20.0 } else { 0.0 })
                    .sub(0.1)
                    .mul(1.3)
                    .max(0.0)
            } else {
                0.0
            },
            tree_density,
            forest_kind: {
                let env = Environment {
                    humid: humidity,
                    temp,
                    near_water: if river.is_lake() || river.near_river() {
                        1.0
                    } else {
                        0.0
                    },
                };

                ForestKind::into_enum_iter()
                    .max_by_key(|fk| (fk.proclivity(&env) * 10000.0) as u32)
                    .unwrap() // Can't fail
            },
            spawn_rate: 1.0,
            river,
            surface_veg: 1.0,

            sites: Vec::new(),
            place: None,
            poi: None,
            path: Default::default(),
            cave: Default::default(),
            cliff_height: 0.0,
            spot: None,

            contains_waypoint: false,
        }
    }

    pub fn is_underwater(&self) -> bool {
        self.water_alt > self.alt || self.river.river_kind.is_some()
    }

    pub fn get_base_z(&self) -> f32 { self.alt - self.chaos * 50.0 - 16.0 }

    pub fn get_biome(&self) -> BiomeKind {
        let savannah_hum_temp = [0.05..0.55, 0.3..1.6];
        let taiga_hum_temp = [0.2..1.4, -0.7..-0.3];
        if self.river.is_ocean() {
            BiomeKind::Ocean
        } else if self.river.is_lake() {
            BiomeKind::Lake
        } else if self.temp < CONFIG.snow_temp {
            BiomeKind::Snowland
        } else if self.alt > 500.0 && self.chaos > 0.3 && self.tree_density < 0.6 {
            BiomeKind::Mountain
        } else if self.temp > CONFIG.desert_temp && self.humidity < CONFIG.desert_hum {
            BiomeKind::Desert
        } else if self.tree_density > 0.65 && self.humidity > 0.65 && self.temp > 0.45 {
            BiomeKind::Jungle
        } else if savannah_hum_temp[0].contains(&self.humidity)
            && savannah_hum_temp[1].contains(&self.temp)
        {
            BiomeKind::Savannah
        } else if taiga_hum_temp[0].contains(&self.humidity)
            && taiga_hum_temp[1].contains(&self.temp)
        {
            BiomeKind::Taiga
        } else if self.tree_density > 0.4 {
            BiomeKind::Forest
        // } else if self.humidity > 0.8 {
        //    BiomeKind::Swamp
        //      Swamps don't really exist yet.
        } else {
            BiomeKind::Grassland
        }
    }

    pub fn near_cliffs(&self) -> bool { self.cliff_height > 0.0 }

    pub fn get_environment(&self) -> Environment {
        Environment {
            humid: self.humidity,
            temp: self.temp,
            near_water: if self.river.is_lake()
                || self.river.near_river()
                || self.alt < CONFIG.sea_level + 6.0
            // Close to sea in altitude
            {
                1.0
            } else {
                0.0
            },
        }
    }
}
