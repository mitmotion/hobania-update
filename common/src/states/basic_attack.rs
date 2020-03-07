use crate::{
    comp::{Attacking, CharacterState, ItemKind::Tool, StateUpdate},
    states::utils::*,
    sys::character_state::JoinData,
};
use std::{collections::VecDeque, time::Duration};

pub fn behavior(data: &JoinData) -> StateUpdate {
    let mut update = StateUpdate {
        pos: *data.pos,
        vel: *data.vel,
        ori: *data.ori,
        energy: *data.energy,
        character: *data.character,
        local_events: VecDeque::new(),
        server_events: VecDeque::new(),
    };

    if let CharacterState::BasicAttack {
        exhausted,
        remaining_duration,
    } = data.character
    {
        handle_move(data, &mut update);

        let tool_kind = data.stats.equipment.main.as_ref().map(|i| i.kind);
        let can_apply_damage = !*exhausted
            && if let Some(Tool(tool)) = tool_kind {
                *remaining_duration < tool.attack_recover_duration()
            } else {
                true
            };

        let mut new_exhausted = *exhausted;

        if can_apply_damage {
            if let Some(Tool(tool)) = tool_kind {
                data.updater
                    .insert(data.entity, Attacking { weapon: Some(tool) });
            } else {
                data.updater.insert(data.entity, Attacking { weapon: None });
            }
            new_exhausted = true;
        } else {
            data.updater.remove::<Attacking>(data.entity);
        }

        let new_remaining_duration = remaining_duration
            .checked_sub(Duration::from_secs_f32(data.dt.0))
            .unwrap_or_default();

        // Tick down
        update.character = CharacterState::BasicAttack {
            remaining_duration: new_remaining_duration,
            exhausted: new_exhausted,
        };

        // Check if attack duration has expired
        if new_remaining_duration == Duration::default() {
            update.character = if let Some(Tool(tool)) = tool_kind {
                CharacterState::Wielding { tool }
            } else {
                CharacterState::Idle {}
            };
            data.updater.remove::<Attacking>(data.entity);
        }

        update
    } else {
        update
    }
}
