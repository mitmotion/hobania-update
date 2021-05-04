use common_net::msg::SerializedTerrainChunk;
use rusqlite::{Connection, ToSql};
use tracing::warn;
use vek::Vec2;

pub struct ChonkCache {
    connection: Option<Connection>,
}

impl ChonkCache {
    /// Create a new ChonkCache using an in-memory database. If database
    /// creation fails, returns a proxy that doesn't cache chunks instead of
    /// propagating the failure.
    pub fn new_in_memory() -> Self {
        let mut ret = Self {
            connection: Connection::open_in_memory().ok(),
        };
        if let Err(e) = ret.ensure_schema() {
            warn!("Couldn't ensure schema for ChonkCache: {:?}", e);
            ret.connection = None;
        }
        ret
    }

    #[rustfmt::skip]
    fn ensure_schema(&self) -> rusqlite::Result<()> {
        if let Some(connection) = &self.connection {
            connection.execute_batch("
                CREATE TABLE IF NOT EXISTS chonks (
                    pos_x INTEGER NOT NULL,
                    pos_y INTEGER NOT NULL,
                    data BLOB NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS chonk_pos ON chonks(pos_x, pos_y);
            ")
        } else {
            Ok(())
        }
    }

    /// Insert a chunk into the cache and return whether it already existed with
    /// the same byte representation on success.
    pub fn received_chonk(
        &self,
        key: Vec2<i32>,
        chonk: &SerializedTerrainChunk,
    ) -> Result<bool, Box<dyn std::error::Error>> {
        if let Some(connection) = &self.connection {
            let serialized = bincode::serialize(chonk)?;
            let values = [&key.x as &dyn ToSql, &key.y, &serialized];

            // TODO: ensure a canonical encoding for SerializedTerrainChunk/make caching
            // more granular to get more hits
            if connection
                .prepare("SELECT NULL FROM chonks WHERE pos_x = ?1 AND pos_y = ?2 AND data = ?3")?
                .query(&values)?
                .next()?
                .is_none()
            {
                connection.execute(
                    "REPLACE INTO chonks (pos_x, pos_y, data) VALUES (?1, ?2, ?3)",
                    &values,
                )?;
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }

    /// Check to see if there's a cached chonk at the specified index
    pub fn get_cached_chonk(&self, key: Vec2<i32>) -> Option<SerializedTerrainChunk> {
        self.connection.as_ref().and_then(|connection| {
            connection
                .query_row(
                    "SELECT data FROM chonks WHERE pos_x = ?1 AND pos_y = ?2",
                    &[key.x, key.y],
                    |row| row.get(0),
                )
                .ok()
                .and_then(|data: Vec<u8>| bincode::deserialize(&*data).ok())
        })
    }
}
