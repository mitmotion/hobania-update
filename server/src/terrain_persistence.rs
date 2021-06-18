use hashbrown::HashMap;
use common::{
    terrain::{Block, TerrainChunk},
    vol::RectRasterableVol,
};
use vek::*;
use tracing::{info, error};
use serde::{Serialize, Deserialize};
use std::{
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
    fn path_for(&self, key: Vec2<i32>) -> PathBuf {
        let mut path = self.path.clone();
        path.push(format!("chunk_{}_{}.dat", key.x, key.y));
        path
    }

    pub fn load_chunk(&mut self, key: Vec2<i32>) -> &mut Chunk {
        let path = self.path_for(key);
        self.chunks
            .entry(key)
            .or_insert_with(|| {
                File::open(path)
                    .ok()
                    .and_then(|f| bincode::deserialize_from(&f).ok())
                    .unwrap_or_else(|| Chunk::default())
            })
    }

    pub fn unload_chunk(&mut self, key: Vec2<i32>) {
        if let Some(chunk) = self.chunks.remove(&key) {
            if chunk.blocks.len() > 0 {
                match File::create(self.path_for(key)) {
                    Ok(file) => { bincode::serialize_into(file, &chunk); },
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
    pub fn blocks(&self) -> impl Iterator<Item=(Vec3<i32>, Block)> + '_ {
        self.blocks.iter().map(|(k, b)| (*k, *b))
    }
}
