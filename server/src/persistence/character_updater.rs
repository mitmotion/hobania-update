use crate::comp;
use common::character::CharacterId;

use crate::persistence::{
    character_loader::{CharacterLoaderResponse, CharacterLoaderResponseKind},
    error::PersistenceError,
    establish_connection, ConnectionMode, DatabaseSettings, EditableComponents,
    PersistedComponents, VelorenConnection,
};
use crossbeam_channel::TryIter;
use rusqlite::{DropBehavior, Transaction};
use specs::Entity;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, RwLock,
    },
};
use tracing::{debug, error, info, trace, warn};

pub type CharacterUpdateData = (
    CharacterId,
    comp::SkillSet,
    comp::Inventory,
    Vec<PetPersistenceData>,
    Option<comp::Waypoint>,
    comp::ability::ActiveAbilities,
    Option<comp::MapMarker>,
);

pub type PetPersistenceData = (comp::Pet, comp::Body, comp::Stats);

#[allow(clippy::large_enum_variant)]
pub enum CharacterUpdaterEvent {
    BatchUpdate(Vec<PendingDatabaseEvent>),
    CreateCharacter {
        entity: Entity,
        player_uuid: String,
        character_alias: String,
        persisted_components: PersistedComponents,
    },
    EditCharacter {
        entity: Entity,
        player_uuid: String,
        character_id: CharacterId,
        character_alias: String,
        editable_components: EditableComponents,
    },
    DisconnectedSuccess,
}

pub enum PendingDatabaseEvent {
    LogoutUpdate(CharacterUpdateData),
    DeleteCharacter {
        requesting_player_uuid: String,
        character_id: CharacterId,
    },
}

/// A unidirectional messaging resource for saving characters in a
/// background thread.
///
/// This is used to make updates to a character and their persisted components,
/// such as inventory, loadout, etc...
pub struct CharacterUpdater {
    update_tx: Option<crossbeam_channel::Sender<CharacterUpdaterEvent>>,
    response_rx: crossbeam_channel::Receiver<CharacterLoaderResponse>,
    handle: Option<std::thread::JoinHandle<()>>,
    /// Pending actions to be performed during the next persistence batch, such
    /// as updates for recently logged out players and character deletions
    pending_database_events: HashMap<CharacterId, PendingDatabaseEvent>,
    /// Will disconnect all characters (without persistence) on the next tick if
    /// set to true
    disconnect_all_clients_requested: Arc<AtomicBool>,
}

impl CharacterUpdater {
    pub fn new(settings: Arc<RwLock<DatabaseSettings>>) -> rusqlite::Result<Self> {
        let (update_tx, update_rx) = crossbeam_channel::unbounded::<CharacterUpdaterEvent>();
        let (response_tx, response_rx) = crossbeam_channel::unbounded::<CharacterLoaderResponse>();

        let disconnect_all_clients_requested = Arc::new(AtomicBool::new(false));
        let disconnect_all_clients_requested_clone = Arc::clone(&disconnect_all_clients_requested);

        let builder = std::thread::Builder::new().name("persistence_updater".into());
        let handle = builder
            .spawn(move || {
                // Unwrap here is safe as there is no code that can panic when the write lock is
                // taken that could cause the RwLock to become poisoned.
                let mut conn =
                    establish_connection(&*settings.read().unwrap(), ConnectionMode::ReadWrite);
                while let Ok(updates) = update_rx.recv() {
                    match updates {
                        CharacterUpdaterEvent::BatchUpdate(updates) => {
                            if disconnect_all_clients_requested_clone.load(Ordering::Relaxed) {
                                debug!(
                                    "Skipping persistence due to pending disconnection of all \
                                     clients"
                                );
                                continue;
                            }
                            conn.update_log_mode(&settings);
                            if let Err(e) = execute_batch_update(updates, &mut conn) {
                                error!(
                                    "Error during character batch update, disconnecting all \
                                     clients to avoid loss of data integrity. Error: {:?}",
                                    e
                                );
                                disconnect_all_clients_requested_clone
                                    .store(true, Ordering::Relaxed);
                            };
                        },
                        CharacterUpdaterEvent::CreateCharacter {
                            entity,
                            character_alias,
                            player_uuid,
                            persisted_components,
                        } => {
                            match execute_character_create(
                                entity,
                                character_alias,
                                &player_uuid,
                                persisted_components,
                                &mut conn,
                            ) {
                                Ok(response) => {
                                    if let Err(e) = response_tx.send(response) {
                                        error!(?e, "Could not send character creation response");
                                    } else {
                                        debug!(
                                            "Processed character create for player {}",
                                            player_uuid
                                        );
                                    }
                                },
                                Err(e) => error!(
                                    "Error creating character for player {}, error: {:?}",
                                    player_uuid, e
                                ),
                            }
                        },
                        CharacterUpdaterEvent::EditCharacter {
                            entity,
                            character_id,
                            character_alias,
                            player_uuid,
                            editable_components,
                        } => {
                            match execute_character_edit(
                                entity,
                                character_id,
                                character_alias,
                                &player_uuid,
                                editable_components,
                                &mut conn,
                            ) {
                                Ok(response) => {
                                    if let Err(e) = response_tx.send(response) {
                                        error!(?e, "Could not send character edit response");
                                    } else {
                                        debug!(
                                            "Processed character edit for player {}",
                                            player_uuid
                                        );
                                    }
                                },
                                Err(e) => error!(
                                    "Error editing character for player {}, error: {:?}",
                                    player_uuid, e
                                ),
                            }
                        },
                        CharacterUpdaterEvent::DisconnectedSuccess => {
                            info!(
                                "CharacterUpdater received DisconnectedSuccess event, resuming \
                                 batch updates"
                            );
                            // Reset the disconnection request as we have had confirmation that all
                            // clients have been disconnected
                            disconnect_all_clients_requested_clone.store(false, Ordering::Relaxed);
                        },
                    }
                }
            })
            .unwrap();

        Ok(Self {
            update_tx: Some(update_tx),
            response_rx,
            handle: Some(handle),
            pending_database_events: HashMap::new(),
            disconnect_all_clients_requested,
        })
    }

    /// Adds a character to the list of characters that have recently logged out
    /// and will be persisted in the next batch update.
    pub fn add_pending_logout_update(&mut self, update_data: CharacterUpdateData) {
        if !self
            .disconnect_all_clients_requested
            .load(Ordering::Relaxed)
        {
            self.pending_database_events.insert(
                update_data.0, // CharacterId
                PendingDatabaseEvent::LogoutUpdate(update_data),
            );
        } else {
            warn!(
                "Ignoring request to add pending logout update for character ID {} as there is a \
                 disconnection of all clients in progress",
                update_data.0
            );
        }
    }

    /// Returns the character IDs of characters that have recently logged out
    /// and are awaiting persistence in the next batch update.
    pub fn characters_pending_logout(&self) -> impl Iterator<Item = CharacterId> + '_ {
        self.pending_database_events.keys().copied()
    }

    /// Returns a value indicating whether there is a pending request to
    /// disconnect all clients due to a batch update transaction failure
    pub fn disconnect_all_clients_requested(&self) -> bool {
        self.disconnect_all_clients_requested
            .load(Ordering::Relaxed)
    }

    pub fn create_character(
        &mut self,
        entity: Entity,
        requesting_player_uuid: String,
        alias: String,
        persisted_components: PersistedComponents,
    ) {
        if let Err(e) =
            self.update_tx
                .as_ref()
                .unwrap()
                .send(CharacterUpdaterEvent::CreateCharacter {
                    entity,
                    player_uuid: requesting_player_uuid,
                    character_alias: alias,
                    persisted_components,
                })
        {
            error!(?e, "Could not send character creation request");
        }
    }

    pub fn edit_character(
        &mut self,
        entity: Entity,
        requesting_player_uuid: String,
        character_id: CharacterId,
        alias: String,
        editable_components: EditableComponents,
    ) {
        if let Err(e) =
            self.update_tx
                .as_ref()
                .unwrap()
                .send(CharacterUpdaterEvent::EditCharacter {
                    entity,
                    player_uuid: requesting_player_uuid,
                    character_id,
                    character_alias: alias,
                    editable_components,
                })
        {
            error!(?e, "Could not send character edit request");
        }
    }

    pub fn delete_character(&mut self, requesting_player_uuid: String, character_id: CharacterId) {
        // Insert the delete as a pending database action - if the player has recently
        // logged out this will replace their pending update with a delete which
        // is fine, as the user has actively chosen to delete the character.
        self.pending_database_events
            .insert(character_id, PendingDatabaseEvent::DeleteCharacter {
                requesting_player_uuid,
                character_id,
            });
    }

    /// Updates a collection of characters based on their id and components
    pub fn batch_update(&mut self, updates: impl Iterator<Item = CharacterUpdateData>) {
        let pending_actions = self
            .pending_database_events
            .drain()
            .map(|(_, event)| event) // Discard CharacterId key as it was only required to check if a character had a pending database event
            .chain(updates.map(|update| PendingDatabaseEvent::LogoutUpdate(update)))
            .collect::<Vec<PendingDatabaseEvent>>();

        if let Err(e) = self
            .update_tx
            .as_ref()
            .unwrap()
            .send(CharacterUpdaterEvent::BatchUpdate(pending_actions))
        {
            error!(?e, "Could not send stats updates");
        }
    }

    /// Indicates to the batch update thread that a requested disconnection of
    /// all clients has been processed
    pub fn disconnected_success(&mut self) {
        self.update_tx
            .as_ref()
            .unwrap()
            .send(CharacterUpdaterEvent::DisconnectedSuccess)
            .expect(
                "Failed to send DisconnectedSuccess event - not sending this event will prevent \
                 future persistence batches from running",
            );
    }

    /// Returns a non-blocking iterator over CharacterLoaderResponse messages
    pub fn messages(&self) -> TryIter<CharacterLoaderResponse> { self.response_rx.try_iter() }
}

fn execute_batch_update(
    updates: Vec<PendingDatabaseEvent>,
    connection: &mut VelorenConnection,
) -> Result<(), PersistenceError> {
    let mut transaction = connection.connection.transaction()?;
    transaction.set_drop_behavior(DropBehavior::Rollback);
    trace!("Transaction started for character batch update");
    updates.into_iter().try_for_each(|event| match event {
        PendingDatabaseEvent::LogoutUpdate((
            character_id,
            stats,
            inventory,
            pets,
            waypoint,
            active_abilities,
            map_marker,
        )) => super::character::update(
            character_id,
            stats,
            inventory,
            pets,
            waypoint,
            active_abilities,
            map_marker,
            &mut transaction,
        ),
        PendingDatabaseEvent::DeleteCharacter {
            requesting_player_uuid,
            character_id,
        } => super::character::delete_character(
            &requesting_player_uuid,
            character_id,
            &mut transaction,
        ),
    })?;

    transaction.commit()?;

    trace!("Commit for character batch update completed");
    Ok(())
}

fn execute_character_create(
    entity: Entity,
    alias: String,
    requesting_player_uuid: &str,
    persisted_components: PersistedComponents,
    connection: &mut VelorenConnection,
) -> Result<CharacterLoaderResponse, PersistenceError> {
    let mut transaction = connection.connection.transaction()?;
    let result =
        CharacterLoaderResponseKind::CharacterCreation(super::character::create_character(
            requesting_player_uuid,
            &alias,
            persisted_components,
            &mut transaction,
        ));
    check_response(entity, transaction, result)
}

fn execute_character_edit(
    entity: Entity,
    character_id: CharacterId,
    alias: String,
    requesting_player_uuid: &str,
    editable_components: EditableComponents,
    connection: &mut VelorenConnection,
) -> Result<CharacterLoaderResponse, PersistenceError> {
    let mut transaction = connection.connection.transaction()?;
    let result = CharacterLoaderResponseKind::CharacterEdit(super::character::edit_character(
        editable_components,
        &mut transaction,
        character_id,
        requesting_player_uuid,
        &alias,
    ));
    check_response(entity, transaction, result)
}

fn check_response(
    entity: Entity,
    transaction: Transaction,
    result: CharacterLoaderResponseKind,
) -> Result<CharacterLoaderResponse, PersistenceError> {
    let response = CharacterLoaderResponse { entity, result };

    if !response.is_err() {
        transaction.commit()?;
    };

    Ok(response)
}

impl Drop for CharacterUpdater {
    fn drop(&mut self) {
        drop(self.update_tx.take());
        if let Err(e) = self.handle.take().unwrap().join() {
            error!(?e, "Error from joining character update thread");
        }
    }
}
