use hashbrown::HashMap;
use common::{
    terrain::{Block, TerrainChunk},
    vol::RectRasterableVol,
};
use vek::*;
use tracing::{info, error, warn};
use serde::{Serialize, Deserialize};
use std::{
    io::{self, Read as _, Write as _},
    fs::File,
    path::PathBuf,
};

pub struct TerrainPersistence {
    path: PathBuf,
    chunks: HashMap<Vec2<i32>, Chunk>,
}

impl Default for TerrainPersistence {
    fn default() -> Self {
        let mut path = PathBuf::from(std::env::var("VELOREN_TERRAIN_DATA").unwrap_or_else(|_| String::new()));
        path.push("chunks");

        std::fs::create_dir_all(&path).unwrap();

        info!("Using {:?} as the terrain persistence path", path);

        Self {
            path,
            chunks: HashMap::default(),
        }
    }
}

impl TerrainPersistence {
    fn path_for(&self, key: Vec2<i32>, ext: &str) -> PathBuf {
        let mut path = self.path.clone();
        path.push(format!("chunk_{}_{}.{}", key.x, key.y, ext));
        path
    }

    pub fn load_chunk(&mut self, key: Vec2<i32>) -> &mut Chunk {
        let path = self.path_for(key, "dat");
        let backup_path = self.path_for(key, "dat.backup");
        self.chunks
            .entry(key)
            .or_insert_with(|| {
                File::open(&path)
                    .ok()
                    .map(|f| {
                        let bytes = match std::io::BufReader::new(f).bytes().collect::<Result<Vec<_>, _>>() {
                            Ok(bytes) => bytes,
                            Err(err) => {
                                error!("Failed to load chunk {:?} from disk: {:?}", key, err);
                                return Chunk::default();
                            },
                        };
                        match Chunk::deserialize_from(std::io::Cursor::new(bytes)) {
                            Some(chunk) => chunk,
                            None => {
                                error!("Failed to load chunk {:?}, moving to {:?} instead", key, backup_path);
                                if let Err(err) = std::fs::copy(&path, backup_path)
                                    .and_then(|_| std::fs::remove_file(&path))
                                {
                                    error!("{:?}", err);
                                }
                                Chunk::default()
                            },
                        }
                    })
                    .unwrap_or_else(|| Chunk::default())
            })
    }

    pub fn unload_chunk(&mut self, key: Vec2<i32>) {
        if let Some(chunk) = self.chunks.remove(&key) {
            if chunk.blocks.len() > 0 { // No need to write if no blocks have ever been written
                match File::create(self.path_for(key, "dat")) {
                    Ok(file) => {
                        let mut writer = std::io::BufWriter::new(file);
                        if let Err(err) = bincode::serialize_into::<_, version::Current>(&mut writer, &chunk.prepare())
                            .and_then(|_| Ok(writer.flush()?))
                        {
                            error!("Failed to write chunk to disk: {:?}", err);
                        }
                    },
                    Err(err) => error!("Failed to create file: {:?}", err),
                }
            }
        }
    }

    pub fn unload_all(&mut self) {
        for key in self.chunks.keys().copied().collect::<Vec<_>>() {
            self.unload_chunk(key);
        }
    }

    pub fn set_block(&mut self, pos: Vec3<i32>, block: Block) {
        let key = pos.xy().map2(TerrainChunk::RECT_SIZE, |e, sz| e.div_euclid(sz as i32));
        self.load_chunk(key).blocks.insert(pos - key * TerrainChunk::RECT_SIZE.map(|e| e as i32), block);
    }
}

#[derive(Default, Serialize, Deserialize)]
pub struct Chunk {
    blocks: HashMap<Vec3<i32>, Block>,
}

impl Chunk {
    pub fn deserialize_from<R: io::Read + Clone>(reader: R) -> Option<Self> {
        // Attempt deserialization through various versions
        if let Ok(data) = bincode::deserialize_from::<_, version::V2>(reader.clone())
            .map_err(|err| { warn!("Error when loading V2: {:?}", err); err })
        {
            Some(Chunk::from(data))
        } else if let Ok(data) = bincode::deserialize_from::<_, Chunk>(reader.clone())
            .map_err(|err| { warn!("Error when loading V1: {:?}", err); err })
        {
            Some(Chunk::from(data))
        } else {
            None
        }
    }

    fn prepare(self) -> version::Current { self.into() }

    pub fn blocks(&self) -> impl Iterator<Item=(Vec3<i32>, Block)> + '_ {
        self.blocks.iter().map(|(k, b)| (*k, *b))
    }
}

mod version {
    pub type Current = V2;

    fn version_magic(n: u16) -> u64 {
        (n as u64) | (0x3352ACEEA789 << 16)
    }

    use super::*;

    // Convert back to current

    impl From<Chunk> for Current {
        fn from(chunk: Chunk) -> Self {
            Self { version: version_magic(2), blocks: chunk.blocks
                .into_iter()
                .map(|(pos, b)| (pos.x as u8, pos.y as u8, pos.z as i16, b))
                .collect() }
        }
    }

    // V1

    #[derive(Serialize, Deserialize)]
    pub struct V1 {
        pub blocks: HashMap<Vec3<i32>, Block>,
    }

    impl From<V1> for Chunk {
        fn from(v1: V1) -> Self { Self { blocks: v1.blocks } }
    }

    // V2

    #[derive(Serialize, Deserialize)]
    pub struct V2 {
        #[serde(deserialize_with = "version::<_, 2>")]
        pub version: u64,
        pub blocks: Vec<(u8, u8, i16, Block)>,
    }

    fn version<'de, D: serde::Deserializer<'de>, const V: u16>(de: D) -> Result<u64, D::Error> {
        u64::deserialize(de).and_then(|x| if x == version_magic(V) {
            Ok(x)
        } else {
            Err(serde::de::Error::invalid_value(serde::de::Unexpected::Unsigned(x), &"correct version"))
        })
    }

    impl From<V2> for Chunk {
        fn from(v2: V2) -> Self { Self { blocks: v2.blocks
            .into_iter()
            .map(|(x, y, z, b)| (Vec3::new(x as i32, y as i32, z as i32), b))
            .collect() } }
    }
}
