use crate::{
    consts::MAX_PATH_DIST,
    data::{AgentData, AttackData, Path, ReadData, TargetData},
    util::entities_have_line_of_sight,
};
use common::{
    comp::{
        buff::BuffKind,
        skills::{AxeSkill, BowSkill, HammerSkill, SceptreSkill, Skill, StaffSkill, SwordSkill},
        AbilityInput, Agent, CharacterAbility, CharacterState, ControlAction, Controller,
        InputKind,
    },
    path::TraversalConfig,
    states::{self_buff, sprite_summon, utils::StageSection},
    terrain::Block,
    util::Dir,
    vol::ReadVol,
};
use rand::Rng;
use std::{f32::consts::PI, time::Duration};
use vek::*;

impl<'a> AgentData<'a> {
    // Intended for any agent that has one attack, that attack is a melee attack,
    // and the agent is able to freely walk around
    pub fn handle_simple_melee(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        if attack_data.in_min_range() && attack_data.angle < 30.0 {
            controller.push_basic_input(InputKind::Primary);
            controller.inputs.move_dir = Vec2::zero();
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Full,
                None,
            );
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.02
            {
                controller.push_basic_input(InputKind::Roll);
            }
        }
    }

    // Intended for any agent that has one attack, that attack is a melee attack,
    // the agent is able to freely walk around, and the agent is trying to attack
    // from behind its target
    pub fn handle_simple_backstab(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        // Handle attacking of agent
        if attack_data.in_min_range() && attack_data.angle < 30.0 {
            controller.push_basic_input(InputKind::Primary);
            controller.inputs.move_dir = Vec2::zero();
        }

        // Handle movement of agent
        let target_ori = agent
            .target
            .and_then(|t| read_data.orientations.get(t.target))
            .map(|ori| ori.look_vec())
            .unwrap_or_default();
        let dist = attack_data.dist_sqrd.sqrt();

        let in_front_of_target = target_ori.dot(self.pos.0 - tgt_data.pos.0) > 0.0;
        if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // If in front of the target, circle to try and get behind, else just make a
            // beeline for the back of the agent
            let vec_to_target = (tgt_data.pos.0 - self.pos.0).xy();
            if in_front_of_target {
                let theta = (PI / 2. - dist * 0.1).max(0.0);
                // Checks both CW and CCW rotation
                let potential_move_dirs = [
                    vec_to_target
                        .rotated_z(theta)
                        .try_normalized()
                        .unwrap_or_default(),
                    vec_to_target
                        .rotated_z(-theta)
                        .try_normalized()
                        .unwrap_or_default(),
                ];
                // Finds shortest path to get behind
                if let Some(move_dir) = potential_move_dirs
                    .iter()
                    .find(|move_dir| target_ori.xy().dot(**move_dir) < 0.0)
                {
                    controller.inputs.move_dir = *move_dir;
                }
            } else {
                // Aim for a point a given distance behind the target to prevent sideways
                // movement
                let move_target = tgt_data.pos.0.xy() - dist / 2. * target_ori.xy();
                controller.inputs.move_dir = (move_target - self.pos.0)
                    .try_normalized()
                    .unwrap_or_default();
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Full,
                None,
            );
        }
    }

    pub fn handle_elevated_ranged(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        let elevation = self.pos.0.z - tgt_data.pos.0.z;
        const PREF_DIST: f32 = 30_f32;
        if attack_data.angle_xy < 30.0
            && (elevation > 10.0 || attack_data.dist_sqrd > PREF_DIST.powi(2))
            && line_of_sight_with_target()
        {
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd < (PREF_DIST / 2.).powi(2) {
            // Attempt to move quickly away from target if too close
            if let Some((bearing, _)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                let flee_dir = -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero);
                let pos = self.pos.0.xy().with_z(self.pos.0.z + 1.5);
                if read_data
                    .terrain
                    .ray(pos, pos + flee_dir * 2.0)
                    .until(|b| b.is_solid() || b.get_sprite().is_none())
                    .cast()
                    .0
                    > 1.0
                {
                    // If able to flee, flee
                    controller.inputs.move_dir = flee_dir;
                    if !self.char_state.is_attack() {
                        controller.inputs.look_dir = -controller.inputs.look_dir;
                    }
                } else {
                    // Otherwise, fight to the death
                    controller.push_basic_input(InputKind::Primary);
                }
            }
        } else if attack_data.dist_sqrd < PREF_DIST.powi(2) {
            // Attempt to move away from target if too close, while still attacking
            if let Some((bearing, _)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if line_of_sight_with_target() {
                    controller.push_basic_input(InputKind::Primary);
                }
                controller.inputs.move_dir =
                    -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero);
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Full,
                None,
            );
        }
    }

    pub fn handle_axe_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        let has_leap = || self.skill_set.has_skill(Skill::Axe(AxeSkill::UnlockLeap));
        let has_energy = |need| self.energy.current() > need;
        let use_leap = |controller: &mut Controller| {
            controller.push_basic_input(InputKind::Ability(0));
        };

        if attack_data.in_min_range() && attack_data.angle < 45.0 {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timer > 5.0 {
                controller.push_cancel_input(InputKind::Secondary);
                agent.action_state.timer = 0.0;
            } else if agent.action_state.timer > 2.5 && has_energy(10.0) {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            } else if has_leap() && has_energy(45.0) && rng.gen_bool(0.5) {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timer += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );

            if attack_data.dist_sqrd < 32.0f32.powi(2)
                && has_leap()
                && has_energy(50.0)
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
            {
                use_leap(controller);
            }
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.02
            {
                controller.push_basic_input(InputKind::Roll);
            }
        }
    }

    pub fn handle_hammer_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        let has_leap = || {
            self.skill_set
                .has_skill(Skill::Hammer(HammerSkill::UnlockLeap))
        };

        let has_energy = |need| self.energy.current() > need;

        let use_leap = |controller: &mut Controller| {
            controller.push_basic_input(InputKind::Ability(0));
        };

        if attack_data.in_min_range() && attack_data.angle < 45.0 {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timer > 4.0 {
                controller.push_cancel_input(InputKind::Secondary);
                agent.action_state.timer = 0.0;
            } else if agent.action_state.timer > 3.0 {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            } else if has_leap() && has_energy(50.0) && rng.gen_bool(0.9) {
                use_leap(controller);
                agent.action_state.timer += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );

            if attack_data.dist_sqrd < 32.0f32.powi(2)
                && has_leap()
                && has_energy(50.0)
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
            {
                use_leap(controller);
            }
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.02
            {
                controller.push_basic_input(InputKind::Roll);
            }
        }
    }

    pub fn handle_sword_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        if attack_data.in_min_range() && attack_data.angle < 45.0 {
            controller.inputs.move_dir = Vec2::zero();
            if self
                .skill_set
                .has_skill(Skill::Sword(SwordSkill::UnlockSpin))
                && agent.action_state.timer < 2.0
                && self.energy.current() > 60.0
            {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timer += read_data.dt.0;
            } else if agent.action_state.timer > 2.0 {
                agent.action_state.timer = 0.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            if self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            ) && entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            ) {
                if agent.action_state.timer > 4.0 && attack_data.angle < 45.0 {
                    controller.push_basic_input(InputKind::Secondary);
                    agent.action_state.timer = 0.0;
                } else {
                    agent.action_state.timer += read_data.dt.0;
                }
            }
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.02
            {
                controller.push_basic_input(InputKind::Roll);
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_bow_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        const MIN_CHARGE_FRAC: f32 = 0.5;
        const OPTIMAL_TARGET_VELOCITY: f32 = 5.0;
        const DESIRED_ENERGY_LEVEL: f32 = 50.0;

        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        // Logic to use abilities
        if let CharacterState::ChargedRanged(c) = self.char_state {
            if !matches!(c.stage_section, StageSection::Recover) {
                // Don't even bother with this logic if in recover
                let target_speed_sqd = agent
                    .target
                    .as_ref()
                    .map(|t| t.target)
                    .and_then(|e| read_data.velocities.get(e))
                    .map_or(0.0, |v| v.0.magnitude_squared());
                if c.charge_frac() < MIN_CHARGE_FRAC
                    || (target_speed_sqd > OPTIMAL_TARGET_VELOCITY.powi(2) && c.charge_frac() < 1.0)
                {
                    // If haven't charged to desired level, or target is moving too fast and haven't
                    // fully charged, keep charging
                    controller.push_basic_input(InputKind::Primary);
                }
                // Else don't send primary input to release the shot
            }
        } else if matches!(self.char_state, CharacterState::RepeaterRanged(c) if self.energy.current() > 5.0 && !matches!(c.stage_section, StageSection::Recover))
        {
            // If in repeater ranged, have enough energy, and aren't in recovery, try to
            // keep firing
            if attack_data.dist_sqrd > attack_data.min_attack_dist.powi(2)
                && line_of_sight_with_target()
            {
                // Only keep firing if not in melee range or if can see target
                controller.push_basic_input(InputKind::Secondary);
            }
        } else if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            if self
                .skill_set
                .has_skill(Skill::Bow(BowSkill::UnlockShotgun))
                && self.energy.current() > 45.0
                && rng.gen_bool(0.5)
            {
                // Use shotgun if target close and have sufficient energy
                controller.push_basic_input(InputKind::Ability(0));
            } else if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && self.energy.current() > CharacterAbility::default_roll().get_energy_cost()
                && !matches!(self.char_state, CharacterState::BasicRanged(c) if !matches!(c.stage_section, StageSection::Recover))
            {
                // Else roll away if can roll and have enough energy, and not using shotgun
                // (other 2 attacks have interrupt handled above) unless in recover
                controller.push_basic_input(InputKind::Roll);
            } else {
                self.path_toward_target(
                    agent,
                    controller,
                    tgt_data.pos.0,
                    read_data,
                    Path::Separate,
                    None,
                );
                if attack_data.angle < 15.0 {
                    controller.push_basic_input(InputKind::Primary);
                }
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) && line_of_sight_with_target() {
            // If not really far, and can see target, attempt to shoot bow
            if self.energy.current() < DESIRED_ENERGY_LEVEL {
                // If low on energy, use primary to attempt to regen energy
                controller.push_basic_input(InputKind::Primary);
            } else {
                // Else we have enough energy, use repeater
                controller.push_basic_input(InputKind::Secondary);
            }
        }
        // Logic to move. Intentionally kept separate from ability logic so duplicated
        // work is less necessary.
        if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            // Attempt to move away from target if too close
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                controller.inputs.move_dir =
                    -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // Else attempt to circle target if neither too close nor too far
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if line_of_sight_with_target() && attack_data.angle < 45.0 {
                    controller.inputs.move_dir = bearing
                        .xy()
                        .rotated_z(rng.gen_range(0.5..1.57))
                        .try_normalized()
                        .unwrap_or_else(Vec2::zero)
                        * speed;
                } else {
                    // Unless cannot see target, then move towards them
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
            // Sometimes try to roll
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.01
            {
                controller.push_basic_input(InputKind::Roll);
            }
        } else {
            // If too far, move towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_staff_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        let extract_ability = |input: AbilityInput| {
            self.active_abilities
                .activate_ability(input, Some(self.inventory), self.skill_set, self.body)
                .unwrap_or_default()
                .0
        };
        let (flamethrower, shockwave) = (
            extract_ability(AbilityInput::Secondary),
            extract_ability(AbilityInput::Auxiliary(0)),
        );
        let flamethrower_range = match flamethrower {
            CharacterAbility::BasicBeam { range, .. } => range,
            _ => 20.0_f32,
        };
        let shockwave_cost = shockwave.get_energy_cost();
        if self.body.map_or(false, |b| b.is_humanoid())
            && attack_data.in_min_range()
            && self.energy.current() > CharacterAbility::default_roll().get_energy_cost()
            && !matches!(self.char_state, CharacterState::Shockwave(_))
        {
            // if a humanoid, have enough stamina, not in shockwave, and in melee range,
            // emergency roll
            controller.push_basic_input(InputKind::Roll);
        } else if matches!(self.char_state, CharacterState::Shockwave(_)) {
            agent.action_state.condition = false;
        } else if agent.action_state.condition
            && matches!(self.char_state, CharacterState::Wielding(_))
        {
            controller.push_basic_input(InputKind::Ability(0));
        } else if !matches!(self.char_state, CharacterState::Shockwave(c) if !matches!(c.stage_section, StageSection::Recover))
        {
            // only try to use another ability unless in shockwave or recover
            let target_approaching_speed = -agent
                .target
                .as_ref()
                .map(|t| t.target)
                .and_then(|e| read_data.velocities.get(e))
                .map_or(0.0, |v| v.0.dot(self.ori.look_vec()));
            if self
                .skill_set
                .has_skill(Skill::Staff(StaffSkill::UnlockShockwave))
                && target_approaching_speed > 12.0
                && self.energy.current() > shockwave_cost
            {
                // if enemy is closing distance quickly, use shockwave to knock back
                if matches!(self.char_state, CharacterState::Wielding(_)) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else {
                    agent.action_state.condition = true;
                }
            } else if self.energy.current()
                > shockwave_cost + CharacterAbility::default_roll().get_energy_cost()
                && attack_data.dist_sqrd < flamethrower_range.powi(2)
            {
                controller.push_basic_input(InputKind::Secondary);
            } else {
                controller.push_basic_input(InputKind::Primary);
            }
        }
        // Logic to move. Intentionally kept separate from ability logic so duplicated
        // work is less necessary.
        if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            // Attempt to move away from target if too close
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                controller.inputs.move_dir =
                    -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // Else attempt to circle target if neither too close nor too far
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                ) && attack_data.angle < 45.0
                {
                    controller.inputs.move_dir = bearing
                        .xy()
                        .rotated_z(rng.gen_range(-1.57..-0.5))
                        .try_normalized()
                        .unwrap_or_else(Vec2::zero)
                        * speed;
                } else {
                    // Unless cannot see target, then move towards them
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
            // Sometimes try to roll
            if self.body.map_or(false, |b| b.is_humanoid())
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && !matches!(self.char_state, CharacterState::Shockwave(_))
                && rng.gen::<f32>() < 0.02
            {
                controller.push_basic_input(InputKind::Roll);
            }
        } else {
            // If too far, move towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_sceptre_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        const DESIRED_ENERGY_LEVEL: f32 = 50.0;
        const DESIRED_COMBO_LEVEL: u32 = 8;

        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        // Logic to use abilities
        if attack_data.dist_sqrd > attack_data.min_attack_dist.powi(2)
            && line_of_sight_with_target()
        {
            // If far enough away, and can see target, check which skill is appropriate to
            // use
            if self.energy.current() > DESIRED_ENERGY_LEVEL
                && read_data
                    .combos
                    .get(*self.entity)
                    .map_or(false, |c| c.counter() >= DESIRED_COMBO_LEVEL)
                && !read_data.buffs.get(*self.entity).iter().any(|buff| {
                    buff.iter_kind(BuffKind::Regeneration)
                        .peekable()
                        .peek()
                        .is_some()
                })
            {
                // If have enough energy and combo to use healing aura, do so
                controller.push_basic_input(InputKind::Secondary);
            } else if self
                .skill_set
                .has_skill(Skill::Sceptre(SceptreSkill::UnlockAura))
                && self.energy.current() > DESIRED_ENERGY_LEVEL
                && !read_data.buffs.get(*self.entity).iter().any(|buff| {
                    buff.iter_kind(BuffKind::ProtectingWard)
                        .peekable()
                        .peek()
                        .is_some()
                })
            {
                // Use ward if target is far enough away, self is not buffed, and have
                // sufficient energy
                controller.push_basic_input(InputKind::Ability(0));
            } else {
                // If low on energy, use primary to attempt to regen energy
                // Or if at desired energy level but not able/willing to ward, just attack
                controller.push_basic_input(InputKind::Primary);
            }
        } else if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            if self.body.map_or(false, |b| b.is_humanoid())
                && self.energy.current() > CharacterAbility::default_roll().get_energy_cost()
                && !matches!(self.char_state, CharacterState::BasicAura(c) if !matches!(c.stage_section, StageSection::Recover))
            {
                // Else roll away if can roll and have enough energy, and not using aura or in
                // recover
                controller.push_basic_input(InputKind::Roll);
            } else if attack_data.angle < 15.0 {
                controller.push_basic_input(InputKind::Primary);
            }
        }
        // Logic to move. Intentionally kept separate from ability logic where possible
        // so duplicated work is less necessary.
        if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            // Attempt to move away from target if too close
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                controller.inputs.move_dir =
                    -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // Else attempt to circle target if neither too close nor too far
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if line_of_sight_with_target() && attack_data.angle < 45.0 {
                    controller.inputs.move_dir = bearing
                        .xy()
                        .rotated_z(rng.gen_range(0.5..1.57))
                        .try_normalized()
                        .unwrap_or_else(Vec2::zero)
                        * speed;
                } else {
                    // Unless cannot see target, then move towards them
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
            // Sometimes try to roll
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && !matches!(self.char_state, CharacterState::BasicAura(_))
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.01
            {
                controller.push_basic_input(InputKind::Roll);
            }
        } else {
            // If too far, move towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_stone_golem_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.in_min_range() && attack_data.angle < 90.0 {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Primary);
            //controller.inputs.primary.set_state(true);
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            if self.vel.0.is_approx_zero() {
                controller.push_basic_input(InputKind::Ability(0));
            }
            if self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            ) && entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            ) && attack_data.angle < 90.0
            {
                if agent.action_state.timer > 5.0 {
                    controller.push_basic_input(InputKind::Secondary);
                    agent.action_state.timer = 0.0;
                } else {
                    agent.action_state.timer += read_data.dt.0;
                }
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn handle_circle_charge_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        radius: u32,
        circle_time: u32,
        rng: &mut impl Rng,
    ) {
        if agent.action_state.counter >= circle_time as f32 {
            // if circle charge is in progress and time hasn't expired, continue charging
            controller.push_basic_input(InputKind::Secondary);
        }
        if attack_data.in_min_range() {
            if agent.action_state.counter > 0.0 {
                // set timer and rotation counter to zero if in minimum range
                agent.action_state.counter = 0.0;
                agent.action_state.int_counter = 0;
            } else {
                // melee attack
                controller.push_basic_input(InputKind::Primary);
                controller.inputs.move_dir = Vec2::zero();
            }
        } else if attack_data.dist_sqrd < (radius as f32 + attack_data.min_attack_dist).powi(2) {
            // if in range to charge, circle, then charge
            if agent.action_state.int_counter == 0 {
                // if you haven't chosen a direction to go in, choose now
                agent.action_state.int_counter = 1 + rng.gen_bool(0.5) as u8;
            }
            if agent.action_state.counter < circle_time as f32 {
                // circle if circle timer not ready
                let move_dir = match agent.action_state.int_counter {
                    1 =>
                    // circle left if counter is 1
                    {
                        (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(0.47 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::unit_y)
                    },
                    2 =>
                    // circle right if counter is 2
                    {
                        (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(-0.47 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::unit_y)
                    },
                    _ =>
                    // if some illegal value slipped in, get zero vector
                    {
                        Vec2::zero()
                    },
                };
                let obstacle = read_data
                    .terrain
                    .ray(
                        self.pos.0 + Vec3::unit_z(),
                        self.pos.0 + move_dir.with_z(0.0) * 2.0 + Vec3::unit_z(),
                    )
                    .until(Block::is_solid)
                    .cast()
                    .1
                    .map_or(true, |b| b.is_some());
                if obstacle {
                    // if obstacle detected, stop circling
                    agent.action_state.counter = circle_time as f32;
                }
                controller.inputs.move_dir = move_dir;
                // use counter as timer since timer may be modified in other parts of the code
                agent.action_state.counter += read_data.dt.0;
            }
            // activating charge once circle timer expires is handled above
        } else {
            let path = if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
                // if too far away from target, move towards them
                Path::Separate
            } else {
                Path::Partial
            };
            self.path_toward_target(agent, controller, tgt_data.pos.0, read_data, path, None);
        }
    }

    pub fn handle_quadlow_ranged_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.dist_sqrd < (3.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 90.0
        {
            controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                .xy()
                .try_normalized()
                .unwrap_or_else(Vec2::unit_y);
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if attack_data.angle < 15.0
                    && entities_have_line_of_sight(
                        self.pos,
                        self.body,
                        tgt_data.pos,
                        tgt_data.body,
                        read_data,
                    )
                {
                    if agent.action_state.timer > 5.0 {
                        agent.action_state.timer = 0.0;
                    } else if agent.action_state.timer > 2.5 {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(1.75 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timer += read_data.dt.0;
                    } else {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(0.25 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timer += read_data.dt.0;
                    }
                    controller.push_basic_input(InputKind::Secondary);
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                } else {
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            } else {
                agent.target = None;
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_tail_slap_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            if agent.action_state.timer > 4.0 {
                controller.push_cancel_input(InputKind::Primary);
                agent.action_state.timer = 0.0;
            } else if agent.action_state.timer > 1.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            }
            controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                .xy()
                .try_normalized()
                .unwrap_or_else(Vec2::unit_y)
                * 0.1;
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_quadlow_quick_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.dist_sqrd < (3.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.dist_sqrd > (2.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 90.0
        {
            controller.push_basic_input(InputKind::Primary);
            controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                .xy()
                .rotated_z(-0.47 * PI)
                .try_normalized()
                .unwrap_or_else(Vec2::unit_y);
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_quadlow_basic_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 70.0
            && attack_data.dist_sqrd < (1.3 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timer > 5.0 {
                agent.action_state.timer = 0.0;
            } else if agent.action_state.timer > 2.0 {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            }
        } else {
            let path = if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
                Path::Separate
            } else {
                Path::Partial
            };
            self.path_toward_target(agent, controller, tgt_data.pos.0, read_data, path, None);
        }
    }

    pub fn handle_quadmed_jump_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.angle < 15.0
            && attack_data.dist_sqrd < (5.0 * attack_data.min_attack_dist).powi(2)
        {
            controller.push_basic_input(InputKind::Ability(0));
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            if self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            ) && attack_data.angle < 15.0
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
            {
                controller.push_basic_input(InputKind::Primary);
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_quadmed_basic_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0 && attack_data.in_min_range() {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timer < 2.0 {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            } else if agent.action_state.timer < 3.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            } else {
                agent.action_state.timer = 0.0;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_quadlow_beam_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (2.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.dist_sqrd < (7.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 15.0
        {
            if agent.action_state.timer < 2.0 {
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .rotated_z(0.47 * PI)
                    .try_normalized()
                    .unwrap_or_else(Vec2::unit_y);
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            } else if agent.action_state.timer < 4.0 && attack_data.angle < 15.0 {
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .rotated_z(-0.47 * PI)
                    .try_normalized()
                    .unwrap_or_else(Vec2::unit_y);
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            } else if agent.action_state.timer < 6.0 && attack_data.angle < 15.0 {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timer += read_data.dt.0;
            } else {
                agent.action_state.timer = 0.0;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_organ_aura_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        _tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const ORGAN_AURA_DURATION: f32 = 34.75;
        if attack_data.dist_sqrd < (7.0 * attack_data.min_attack_dist).powi(2) {
            if agent.action_state.timer > ORGAN_AURA_DURATION {
                agent.action_state.timer = 0.0;
            } else if agent.action_state.timer < 1.0 {
                controller
                    .actions
                    .push(ControlAction::basic_input(InputKind::Primary));
                agent.action_state.timer += read_data.dt.0;
            } else {
                agent.action_state.timer += read_data.dt.0;
            }
        } else {
            agent.target = None;
        }
    }

    pub fn handle_theropod_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if attack_data.angle < 90.0 && attack_data.in_min_range() {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_turret_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        if entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
            && attack_data.angle < 15.0
        {
            controller.push_basic_input(InputKind::Primary);
        } else {
            agent.target = None;
        }
    }

    pub fn handle_fixed_turret_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        controller.inputs.look_dir = self.ori.look_dir();
        if entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
            && attack_data.angle < 15.0
        {
            controller.push_basic_input(InputKind::Primary);
        } else {
            agent.target = None;
        }
    }

    pub fn handle_rotating_turret_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        controller.inputs.look_dir = Dir::new(
            Quaternion::from_xyzw(self.ori.look_dir().x, self.ori.look_dir().y, 0.0, 0.0)
                .rotated_z(6.0 * read_data.dt.0 as f32)
                .into_vec3()
                .try_normalized()
                .unwrap_or_default(),
        );
        if entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        {
            controller.push_basic_input(InputKind::Primary);
        } else {
            agent.target = None;
        }
    }

    pub fn handle_radial_turret_attack(&self, controller: &mut Controller) {
        controller.push_basic_input(InputKind::Primary);
    }

    pub fn handle_mindflayer_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        const MINDFLAYER_ATTACK_DIST: f32 = 16.0;
        const MINION_SUMMON_THRESHOLD: f32 = 0.20;
        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already initialized
        if !agent.action_state.condition {
            agent.action_state.counter = 1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.condition = true;
        }

        if agent.action_state.counter > health_fraction {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(2));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counter -= MINION_SUMMON_THRESHOLD;
            }
        } else if attack_data.dist_sqrd < MINDFLAYER_ATTACK_DIST.powi(2) {
            if entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            ) {
                // If close to target, use either primary or secondary ability
                if matches!(self.char_state, CharacterState::BasicBeam(c) if c.timer < Duration::from_secs(10) && !matches!(c.stage_section, StageSection::Recover))
                {
                    // If already using primary, keep using primary until 10 consecutive seconds
                    controller.push_basic_input(InputKind::Primary);
                } else if matches!(self.char_state, CharacterState::SpinMelee(c) if c.consecutive_spins < 50 && !matches!(c.stage_section, StageSection::Recover))
                {
                    // If already using secondary, keep using secondary until 10 consecutive
                    // seconds
                    controller.push_basic_input(InputKind::Secondary);
                } else if rng.gen_bool(health_fraction.into()) {
                    // Else if at high health, use primary
                    controller.push_basic_input(InputKind::Primary);
                } else {
                    // Else use secondary
                    controller.push_basic_input(InputKind::Secondary);
                }
            } else {
                self.path_toward_target(
                    agent,
                    controller,
                    tgt_data.pos.0,
                    read_data,
                    Path::Separate,
                    None,
                );
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // If too far from target, throw a random number of necrotic spheres at them and
            // then blink to them.
            let num_fireballs = &mut agent.action_state.int_counter;
            if *num_fireballs == 0 {
                controller.push_action(ControlAction::StartInput {
                    input: InputKind::Ability(0),
                    target_entity: agent
                        .target
                        .as_ref()
                        .and_then(|t| read_data.uids.get(t.target))
                        .copied(),
                    select_pos: None,
                });
                if matches!(self.char_state, CharacterState::Blink(_)) {
                    *num_fireballs = rand::random::<u8>() % 4;
                }
            } else if matches!(self.char_state, CharacterState::Wielding(_)) {
                *num_fireballs -= 1;
                controller.push_action(ControlAction::StartInput {
                    input: InputKind::Ability(1),
                    target_entity: agent
                        .target
                        .as_ref()
                        .and_then(|t| read_data.uids.get(t.target))
                        .copied(),
                    select_pos: None,
                });
            }
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_birdlarge_fire_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        if attack_data.dist_sqrd > 30.0_f32.powi(2) {
            let small_chance = rng.gen_bool(0.05);

            if small_chance
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
                && attack_data.angle < 15.0
            {
                // Fireball
                controller.push_basic_input(InputKind::Primary);
            }
            // If some target
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                // Walk to target
                controller.inputs.move_dir =
                    bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                // If less than 20 blocks higher than target
                if (self.pos.0.z - tgt_data.pos.0.z) < 20.0 {
                    // Fly upward
                    controller.push_basic_input(InputKind::Fly);
                    controller.inputs.move_z = 1.0;
                } else {
                    // Jump
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
        }
        // If higher than 2 blocks
        else if !read_data
            .terrain
            .ray(self.pos.0, self.pos.0 - (Vec3::unit_z() * 2.0))
            .until(Block::is_solid)
            .cast()
            .1
            .map_or(true, |b| b.is_some())
        {
            // Do not increment the timer during this movement
            // The next stage shouldn't trigger until the entity
            // is on the ground
            // Fly to target
            controller.push_basic_input(InputKind::Fly);
            let move_dir = tgt_data.pos.0 - self.pos.0;
            controller.inputs.move_dir =
                move_dir.xy().try_normalized().unwrap_or_else(Vec2::zero) * 2.0;
            controller.inputs.move_z = move_dir.z - 0.5;
            // If further than 4 blocks and random chance
            if rng.gen_bool(0.05)
                && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
                && attack_data.angle < 15.0
            {
                // Fireball
                controller.push_basic_input(InputKind::Primary);
            }
        }
        // If further than 4 blocks and random chance
        else if rng.gen_bool(0.05)
            && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 15.0
        {
            // Fireball
            controller.push_basic_input(InputKind::Primary);
        }
        // If random chance and less than 20 blocks higher than target and further than 4
        // blocks
        else if rng.gen_bool(0.5)
            && (self.pos.0.z - tgt_data.pos.0.z) < 15.0
            && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
        {
            controller.push_basic_input(InputKind::Fly);
            controller.inputs.move_z = 1.0;
        }
        // If further than 2.5 blocks and random chance
        else if attack_data.dist_sqrd > (2.5 * attack_data.min_attack_dist).powi(2) {
            // Walk to target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        }
        // If energy higher than 600 and random chance
        else if self.energy.current() > 60.0 && rng.gen_bool(0.4) {
            // Shockwave
            controller.push_basic_input(InputKind::Ability(0));
        } else if attack_data.angle < 90.0 {
            // Triple strike
            controller.push_basic_input(InputKind::Secondary);
        } else {
            // Target is behind us. Turn around and chase target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        }
    }

    pub fn handle_birdlarge_breathe_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        // Set fly to false
        controller.push_cancel_input(InputKind::Fly);
        if attack_data.dist_sqrd > 30.0_f32.powi(2) {
            if rng.gen_bool(0.05)
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
                && attack_data.angle < 15.0
            {
                controller.push_basic_input(InputKind::Primary);
            }
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                controller.inputs.move_dir =
                    bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                if (self.pos.0.z - tgt_data.pos.0.z) < 20.0 {
                    controller.push_basic_input(InputKind::Fly);
                    controller.inputs.move_z = 1.0;
                } else {
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
        } else if !read_data
            .terrain
            .ray(self.pos.0, self.pos.0 - (Vec3::unit_z() * 2.0))
            .until(Block::is_solid)
            .cast()
            .1
            .map_or(true, |b| b.is_some())
        {
            // Do not increment the timer during this movement
            // The next stage shouldn't trigger until the entity
            // is on the ground
            controller.push_basic_input(InputKind::Fly);
            let move_dir = tgt_data.pos.0 - self.pos.0;
            controller.inputs.move_dir =
                move_dir.xy().try_normalized().unwrap_or_else(Vec2::zero) * 2.0;
            controller.inputs.move_z = move_dir.z - 0.5;
            if rng.gen_bool(0.05)
                && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
                && attack_data.angle < 15.0
            {
                controller.push_basic_input(InputKind::Primary);
            }
        } else if rng.gen_bool(0.05)
            && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 15.0
        {
            controller.push_basic_input(InputKind::Primary);
        } else if rng.gen_bool(0.5)
            && (self.pos.0.z - tgt_data.pos.0.z) < 15.0
            && attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2)
        {
            controller.push_basic_input(InputKind::Fly);
            controller.inputs.move_z = 1.0;
        } else if attack_data.dist_sqrd > (3.0 * attack_data.min_attack_dist).powi(2) {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        } else if self.energy.current() > 60.0
            && agent.action_state.timer < 3.0
            && attack_data.angle < 15.0
        {
            // Fire breath attack
            controller.push_basic_input(InputKind::Ability(0));
            // Move towards the target slowly
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                Some(0.5),
            );
            agent.action_state.timer += read_data.dt.0;
        } else if agent.action_state.timer < 6.0
            && attack_data.angle < 90.0
            && attack_data.in_min_range()
        {
            // Triple strike
            controller.push_basic_input(InputKind::Secondary);
            agent.action_state.timer += read_data.dt.0;
        } else {
            // Reset timer
            agent.action_state.timer = 0.0;
            // Target is behind us or the timer needs to be reset. Chase target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Separate,
                None,
            );
        }
    }

    pub fn handle_birdlarge_basic_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const BIRD_ATTACK_RANGE: f32 = 4.0;
        const BIRD_CHARGE_DISTANCE: f32 = 15.0;
        let bird_attack_distance = self.body.map_or(0.0, |b| b.max_radius()) + BIRD_ATTACK_RANGE;
        // Increase action timer
        agent.action_state.timer += read_data.dt.0;
        // If higher than 2 blocks
        if !read_data
            .terrain
            .ray(self.pos.0, self.pos.0 - (Vec3::unit_z() * 2.0))
            .until(Block::is_solid)
            .cast()
            .1
            .map_or(true, |b| b.is_some())
        {
            // Fly to target and land
            controller.push_basic_input(InputKind::Fly);
            let move_dir = tgt_data.pos.0 - self.pos.0;
            controller.inputs.move_dir =
                move_dir.xy().try_normalized().unwrap_or_else(Vec2::zero) * 2.0;
            controller.inputs.move_z = move_dir.z - 0.5;
        } else if agent.action_state.timer > 8.0 {
            // If action timer higher than 8, make bird summon tornadoes
            controller.push_basic_input(InputKind::Secondary);
            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                // Reset timer
                agent.action_state.timer = 0.0;
            }
        } else if matches!(self.char_state, CharacterState::DashMelee(c) if !matches!(c.stage_section, StageSection::Recover))
        {
            // If already in dash, keep dashing if not in recover
            controller.push_basic_input(InputKind::Ability(0));
        } else if matches!(self.char_state, CharacterState::ComboMelee(c) if matches!(c.stage_section, StageSection::Recover))
        {
            // If already in combo keep comboing if not in recover
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd > BIRD_CHARGE_DISTANCE.powi(2) {
            // Charges at target if they are far enough away
            if attack_data.angle < 60.0 {
                controller.push_basic_input(InputKind::Ability(0));
            }
        } else if attack_data.dist_sqrd < bird_attack_distance.powi(2) {
            // Combo melee target
            controller.push_basic_input(InputKind::Primary);
            agent.action_state.condition = true;
        }
        // Make bird move towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Separate,
            None,
        );
    }

    pub fn handle_arthropod_ranged_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        agent.action_state.timer += read_data.dt.0;
        if agent.action_state.timer > 6.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
            // Reset timer
            if matches!(self.char_state,
            CharacterState::SpriteSummon(sprite_summon::Data { stage_section, .. })
            | CharacterState::SelfBuff(self_buff::Data { stage_section, .. })
            if matches!(stage_section, StageSection::Recover))
            {
                agent.action_state.timer = 0.0;
            }
        } else if attack_data.dist_sqrd < (2.5 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 90.0
        {
            controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                .xy()
                .try_normalized()
                .unwrap_or_else(Vec2::unit_y);
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if attack_data.angle < 15.0
                    && entities_have_line_of_sight(
                        self.pos,
                        self.body,
                        tgt_data.pos,
                        tgt_data.body,
                        read_data,
                    )
                {
                    if agent.action_state.timer > 5.0 {
                        agent.action_state.timer = 0.0;
                    } else if agent.action_state.timer > 2.5 {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(1.75 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timer += read_data.dt.0;
                    } else {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(0.25 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timer += read_data.dt.0;
                    }
                    controller.push_basic_input(InputKind::Ability(0));
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                } else {
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            } else {
                agent.target = None;
            }
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_arthropod_ambush_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        agent.action_state.timer += read_data.dt.0;
        if agent.action_state.timer > 12.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
            // Reset timer
            if matches!(self.char_state,
            CharacterState::SpriteSummon(sprite_summon::Data { stage_section, .. })
            | CharacterState::SelfBuff(self_buff::Data { stage_section, .. })
            if matches!(stage_section, StageSection::Recover))
            {
                agent.action_state.timer = 0.0;
            }
        } else if attack_data.angle < 90.0
            && attack_data.dist_sqrd < attack_data.min_attack_dist.powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Primary);
        } else if rng.gen_bool(0.01)
            && attack_data.angle < 60.0
            && attack_data.dist_sqrd > (2.0 * attack_data.min_attack_dist).powi(2)
        {
            controller.push_basic_input(InputKind::Ability(0));
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_arthropod_melee_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        agent.action_state.timer += read_data.dt.0;
        if matches!(self.char_state, CharacterState::DashMelee(c) if !matches!(c.stage_section, StageSection::Recover))
        {
            // If already charging, keep charging if not in recover
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.dist_sqrd > (2.5 * attack_data.min_attack_dist).powi(2) {
            // Charges at target if they are far enough away
            if attack_data.angle < 60.0 {
                controller.push_basic_input(InputKind::Secondary);
            }
        } else if attack_data.angle < 90.0
            && attack_data.dist_sqrd < attack_data.min_attack_dist.powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Primary);
        } else {
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_minotaur_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const MINOTAUR_FRENZY_THRESHOLD: f32 = 0.5;
        const MINOTAUR_ATTACK_RANGE: f32 = 5.0;
        const MINOTAUR_CHARGE_DISTANCE: f32 = 15.0;
        let minotaur_attack_distance =
            self.body.map_or(0.0, |b| b.max_radius()) + MINOTAUR_ATTACK_RANGE;
        let health_fraction = self.health.map_or(1.0, |h| h.fraction());
        // Sets action counter at start of combat
        if agent.action_state.counter < MINOTAUR_FRENZY_THRESHOLD
            && health_fraction > MINOTAUR_FRENZY_THRESHOLD
        {
            agent.action_state.counter = MINOTAUR_FRENZY_THRESHOLD;
        }
        if health_fraction < agent.action_state.counter {
            // Makes minotaur buff itself with frenzy
            controller.push_basic_input(InputKind::Ability(1));
            if matches!(self.char_state, CharacterState::SelfBuff(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counter = 0.0;
            }
        } else if matches!(self.char_state, CharacterState::DashMelee(c) if !matches!(c.stage_section, StageSection::Recover))
        {
            // If already charging, keep charging if not in recover
            controller.push_basic_input(InputKind::Ability(0));
        } else if matches!(self.char_state, CharacterState::ChargedMelee(c) if matches!(c.stage_section, StageSection::Charge) && c.timer < c.static_data.charge_duration)
        {
            // If already charging a melee attack, keep charging it if charging
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd > MINOTAUR_CHARGE_DISTANCE.powi(2) {
            // Charges at target if they are far enough away
            if attack_data.angle < 60.0 {
                controller.push_basic_input(InputKind::Ability(0));
            }
        } else if attack_data.dist_sqrd < minotaur_attack_distance.powi(2) {
            if agent.action_state.condition && !self.char_state.is_attack() {
                // Cripple target if not just used cripple
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.condition = false;
            } else if !self.char_state.is_attack() {
                // Cleave target if not just used cleave
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.condition = true;
            }
        }
        // Make minotaur move towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Separate,
            None,
        );
    }

    pub fn handle_clay_golem_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const GOLEM_MELEE_RANGE: f32 = 4.0;
        const GOLEM_LASER_RANGE: f32 = 30.0;
        const GOLEM_LONG_RANGE: f32 = 50.0;
        const GOLEM_TARGET_SPEED: f32 = 8.0;
        let golem_melee_range = self.body.map_or(0.0, |b| b.max_radius()) + GOLEM_MELEE_RANGE;
        // Fraction of health, used for activation of shockwave
        // If golem don't have health for some reason, assume it's full
        let health_fraction = self.health.map_or(1.0, |h| h.fraction());
        // Magnitude squared of cross product of target velocity with golem orientation
        let target_speed_cross_sqd = agent
            .target
            .as_ref()
            .map(|t| t.target)
            .and_then(|e| read_data.velocities.get(e))
            .map_or(0.0, |v| v.0.cross(self.ori.look_vec()).magnitude_squared());
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        if attack_data.dist_sqrd < golem_melee_range.powi(2) {
            if agent.action_state.counter < 7.5 {
                // If target is close, whack them
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.counter += read_data.dt.0;
            } else {
                // If whacked for too long, nuke them
                controller.push_basic_input(InputKind::Ability(1));
                if matches!(self.char_state, CharacterState::BasicRanged(c) if matches!(c.stage_section, StageSection::Recover))
                {
                    agent.action_state.counter = 0.0;
                }
            }
        } else if attack_data.dist_sqrd < GOLEM_LASER_RANGE.powi(2) {
            if matches!(self.char_state, CharacterState::BasicBeam(c) if c.timer < Duration::from_secs(5))
                || target_speed_cross_sqd < GOLEM_TARGET_SPEED.powi(2)
                    && line_of_sight_with_target()
                    && attack_data.angle < 45.0
            {
                // If target in range threshold and haven't been lasering for more than 5
                // seconds already or if target is moving slow-ish, laser them
                controller.push_basic_input(InputKind::Secondary);
            } else if health_fraction < 0.7 {
                // Else target moving too fast for laser, shockwave time.
                // But only if damaged enough
                controller.push_basic_input(InputKind::Ability(0));
            }
        } else if attack_data.dist_sqrd < GOLEM_LONG_RANGE.powi(2) {
            if target_speed_cross_sqd < GOLEM_TARGET_SPEED.powi(2) && line_of_sight_with_target() {
                // If target is far-ish and moving slow-ish, rocket them
                controller.push_basic_input(InputKind::Ability(1));
            } else if health_fraction < 0.7 {
                // Else target moving too fast for laser, shockwave time.
                // But only if damaged enough
                controller.push_basic_input(InputKind::Ability(0));
            }
        }
        // Make clay golem move towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Separate,
            None,
        );
    }

    pub fn handle_tidal_warrior_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const SCUTTLE_RANGE: f32 = 40.0;
        const BUBBLE_RANGE: f32 = 20.0;
        const MINION_SUMMON_THRESHOLD: f32 = 0.20;
        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already initialized
        if !agent.action_state.condition {
            agent.action_state.counter = 1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.condition = true;
        }

        if agent.action_state.counter > health_fraction {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(1));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counter -= MINION_SUMMON_THRESHOLD;
            }
        } else if attack_data.dist_sqrd < SCUTTLE_RANGE.powi(2) {
            if matches!(self.char_state, CharacterState::DashMelee(c) if !matches!(c.stage_section, StageSection::Recover))
            {
                // Keep scuttling if already in dash melee and not in recover
                controller.push_basic_input(InputKind::Secondary);
            } else if attack_data.dist_sqrd < BUBBLE_RANGE.powi(2) {
                if matches!(self.char_state, CharacterState::BasicBeam(c) if !matches!(c.stage_section, StageSection::Recover) && c.timer < Duration::from_secs(10))
                {
                    // Keep shooting bubbles at them if already in basic beam and not in recover and
                    // have not been bubbling too long
                    controller.push_basic_input(InputKind::Ability(0));
                } else if attack_data.in_min_range() && attack_data.angle < 60.0 {
                    // Pincer them if they're in range and angle
                    controller.push_basic_input(InputKind::Primary);
                } else if attack_data.angle < 30.0 && line_of_sight_with_target() {
                    // Start bubbling them if not close enough to do something else and in angle and
                    // can see target
                    controller.push_basic_input(InputKind::Ability(0));
                }
            } else if attack_data.angle < 90.0 && line_of_sight_with_target() {
                // Start scuttling if not close enough to do something else and in angle and can
                // see target
                controller.push_basic_input(InputKind::Secondary);
            }
        }
        // Always attempt to path towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Partial,
            None,
        );
    }

    pub fn handle_yeti_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const ICE_SPIKES_RANGE: f32 = 15.0;
        const ICE_BREATH_RANGE: f32 = 10.0;
        const ICE_BREATH_TIMER: f32 = 10.0;
        const SNOWBALL_MAX_RANGE: f32 = 50.0;

        agent.action_state.counter += read_data.dt.0;

        if attack_data.dist_sqrd < ICE_BREATH_RANGE.powi(2) {
            if matches!(self.char_state, CharacterState::BasicBeam(c) if c.timer < Duration::from_secs(2))
            {
                // Keep using ice breath for 2 second
                controller.push_basic_input(InputKind::Ability(0));
            } else if agent.action_state.counter > ICE_BREATH_TIMER {
                // Use ice breath if timer has gone for long enough
                controller.push_basic_input(InputKind::Ability(0));

                if matches!(self.char_state, CharacterState::BasicBeam(_)) {
                    // Resets action counter when using beam
                    agent.action_state.counter = 0.0;
                }
            } else if attack_data.in_min_range() {
                // Basic attack if on top of them
                controller.push_basic_input(InputKind::Primary);
            } else {
                // Use ice spikes if too far for other abilities
                controller.push_basic_input(InputKind::Secondary);
            }
        } else if attack_data.dist_sqrd < ICE_SPIKES_RANGE.powi(2) && attack_data.angle < 60.0 {
            // Use ice spikes if in range
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.dist_sqrd < SNOWBALL_MAX_RANGE.powi(2) && attack_data.angle < 60.0 {
            // Otherwise, chuck all the snowballs
            controller.push_basic_input(InputKind::Ability(1));
        }

        // Always attempt to path towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Partial,
            None,
        );
    }

    pub fn handle_harvester_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const VINE_CREATION_THRESHOLD: f32 = 0.50;
        const FIRE_BREATH_RANGE: f32 = 20.0;
        const MAX_PUMPKIN_RANGE: f32 = 50.0;
        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        if health_fraction < VINE_CREATION_THRESHOLD && !agent.action_state.condition {
            // Summon vines when reach threshold of health
            controller.push_basic_input(InputKind::Ability(0));

            if matches!(self.char_state, CharacterState::SpriteSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.condition = true;
            }
        } else if attack_data.dist_sqrd < FIRE_BREATH_RANGE.powi(2) {
            if matches!(self.char_state, CharacterState::BasicBeam(c) if c.timer < Duration::from_secs(5))
                && line_of_sight_with_target()
            {
                // Keep breathing fire if close enough, can see target, and have not been
                // breathing for more than 5 seconds
                controller.push_basic_input(InputKind::Secondary);
            } else if attack_data.in_min_range() && attack_data.angle < 60.0 {
                // Scythe them if they're in range and angle
                controller.push_basic_input(InputKind::Primary);
            } else if attack_data.angle < 30.0 && line_of_sight_with_target() {
                // Start breathing fire at them if close enough, in angle, and can see target
                controller.push_basic_input(InputKind::Secondary);
            }
        } else if attack_data.dist_sqrd < MAX_PUMPKIN_RANGE.powi(2) && line_of_sight_with_target() {
            // Throw a pumpkin at them if close enough and can see them
            controller.push_basic_input(InputKind::Ability(1));
        }
        // Always attempt to path towards target
        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Partial,
            None,
        );
    }

    pub fn handle_cardinal_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        const DESIRED_ENERGY_LEVEL: f32 = 50.0;
        const DESIRED_COMBO_LEVEL: u32 = 8;
        const MINION_SUMMON_THRESHOLD: f32 = 0.10;
        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already intitialized
        if !agent.action_state.condition {
            agent.action_state.counter = 1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.condition = true;
        }

        if agent.action_state.counter > health_fraction {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(1));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counter -= MINION_SUMMON_THRESHOLD;
            }
        }
        // Logic to use abilities
        else if attack_data.dist_sqrd > attack_data.min_attack_dist.powi(2)
            && entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            )
        {
            // If far enough away, and can see target, check which skill is appropriate to
            // use
            if self.energy.current() > DESIRED_ENERGY_LEVEL
                && read_data
                    .combos
                    .get(*self.entity)
                    .map_or(false, |c| c.counter() >= DESIRED_COMBO_LEVEL)
                && !read_data.buffs.get(*self.entity).iter().any(|buff| {
                    buff.iter_kind(BuffKind::Regeneration)
                        .peekable()
                        .peek()
                        .is_some()
                })
            {
                // If have enough energy and combo to use healing aura, do so
                controller.push_basic_input(InputKind::Secondary);
            } else if self
                .skill_set
                .has_skill(Skill::Sceptre(SceptreSkill::UnlockAura))
                && self.energy.current() > DESIRED_ENERGY_LEVEL
                && !read_data.buffs.get(*self.entity).iter().any(|buff| {
                    buff.iter_kind(BuffKind::ProtectingWard)
                        .peekable()
                        .peek()
                        .is_some()
                })
            {
                // Use ward if target is far enough away, self is not buffed, and have
                // sufficient energy
                controller.push_basic_input(InputKind::Ability(0));
            } else {
                // If low on energy, use primary to attempt to regen energy
                // Or if at desired energy level but not able/willing to ward, just attack
                controller.push_basic_input(InputKind::Primary);
            }
        } else if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            if self.body.map_or(false, |b| b.is_humanoid())
                && self.energy.current() > CharacterAbility::default_roll().get_energy_cost()
                && !matches!(self.char_state, CharacterState::BasicAura(c) if !matches!(c.stage_section, StageSection::Recover))
            {
                // Else roll away if can roll and have enough energy, and not using aura or in
                // recover
                controller.push_basic_input(InputKind::Roll);
            } else if attack_data.angle < 15.0 {
                controller.push_basic_input(InputKind::Primary);
            }
        }
        // Logic to move. Intentionally kept separate from ability logic where possible
        // so duplicated work is less necessary.
        if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            // Attempt to move away from target if too close
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                controller.inputs.move_dir =
                    -bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
            }
        } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            // Else attempt to circle target if neither too close nor too far
            if let Some((bearing, speed)) = agent.chaser.chase(
                &*read_data.terrain,
                self.pos.0,
                self.vel.0,
                tgt_data.pos.0,
                TraversalConfig {
                    min_tgt_dist: 1.25,
                    ..self.traversal_config
                },
            ) {
                if entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                ) && attack_data.angle < 45.0
                {
                    controller.inputs.move_dir = bearing
                        .xy()
                        .rotated_z(rng.gen_range(0.5..1.57))
                        .try_normalized()
                        .unwrap_or_else(Vec2::zero)
                        * speed;
                } else {
                    // Unless cannot see target, then move towards them
                    controller.inputs.move_dir =
                        bearing.xy().try_normalized().unwrap_or_else(Vec2::zero) * speed;
                    self.jump_if(bearing.z > 1.5, controller);
                    controller.inputs.move_z = bearing.z;
                }
            }
            // Sometimes try to roll
            if self.body.map(|b| b.is_humanoid()).unwrap_or(false)
                && !matches!(self.char_state, CharacterState::BasicAura(_))
                && attack_data.dist_sqrd < 16.0f32.powi(2)
                && rng.gen::<f32>() < 0.01
            {
                controller.push_basic_input(InputKind::Roll);
            }
        } else {
            // If too far, move towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_dagon_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        // if close to target, shoot dagon bombs and lay out sea urchins
        if attack_data.angle < 70.0
            && attack_data.dist_sqrd < (1.3 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timer > 1.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timer += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timer += read_data.dt.0;
            }
        } else if attack_data.angle < 30.0
            && entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            )
        {
            // if in range, angle and sight, shoot dagon bombs at target
            controller.push_basic_input(InputKind::Primary);
        }
        // chase
        let path = if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2) {
            Path::Separate
        } else {
            Path::Partial
        };
        self.path_toward_target(agent, controller, tgt_data.pos.0, read_data, path, None);
    }

    pub fn handle_deadwood(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const BEAM_RANGE: f32 = 20.0;
        const BEAM_TIME: Duration = Duration::from_secs(3);
        // action_state.condition controls whether or not deadwood should beam or dash
        if matches!(self.char_state, CharacterState::DashMelee(s) if s.stage_section != StageSection::Recover)
        {
            // If already dashing, keep dashing and have move_dir set to forward
            controller.push_basic_input(InputKind::Secondary);
            controller.inputs.move_dir = self.ori.look_vec().xy();
        } else if attack_data.in_min_range() && attack_data.angle_xy < 10.0 {
            // If near target, dash at them and through them to get away
            controller.push_basic_input(InputKind::Secondary);
        } else if matches!(self.char_state, CharacterState::BasicBeam(s) if s.stage_section != StageSection::Recover && s.timer < BEAM_TIME)
        {
            // If already beaming, keep beaming if not beaming for over 5 seconds
            controller.push_basic_input(InputKind::Primary);
        } else if attack_data.dist_sqrd < BEAM_RANGE.powi(2) {
            // Else if in beam range, beam them
            if attack_data.angle_xy < 5.0 {
                controller.push_basic_input(InputKind::Primary);
            } else {
                // If not in angle, apply slight movement so deadwood orients itself correctly
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .try_normalized()
                    .unwrap_or_else(Vec2::zero)
                    * 0.01;
            }
        } else {
            // Otherwise too far, move towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_mandragora(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const SCREAM_RANGE: f32 = 10.0;

        if !agent.action_state.initialized {
            agent.action_state.counter = self.health.map_or(0.0, |h| h.maximum());
            agent.action_state.initialized = true;
        }

        if !agent.action_state.condition {
            // If mandragora is still "sleeping" and hasn't screamed yet, do nothing until
            // target in range or until it's taken damage
            if self
                .health
                .map_or(false, |h| h.current() < agent.action_state.counter)
                || attack_data.dist_sqrd < SCREAM_RANGE.powi(2)
            {
                agent.action_state.condition = true;
                controller.push_basic_input(InputKind::Secondary);
            }
        } else {
            // Once mandragora has woken, move towards target and attack
            if attack_data.in_min_range() {
                controller.push_basic_input(InputKind::Primary);
            } else if attack_data.dist_sqrd < MAX_PATH_DIST.powi(2)
                && entities_have_line_of_sight(
                    self.pos,
                    self.body,
                    tgt_data.pos,
                    tgt_data.body,
                    read_data,
                )
            {
                // If in pathing range and can see target, move towards them
                self.path_toward_target(
                    agent,
                    controller,
                    tgt_data.pos.0,
                    read_data,
                    Path::Partial,
                    None,
                );
            } else {
                // Otherwise, go back to sleep
                agent.action_state.condition = false;
                agent.action_state.counter = self.health.map_or(0.0, |h| h.maximum());
            }
        }
    }

    pub fn handle_wood_golem(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const SHOCKWAVE_RANGE: f32 = 25.0;
        const SHOCKWAVE_WAIT_TIME: f32 = 7.5;
        const SPIN_WAIT_TIME: f32 = 3.0;

        // After spinning, reset timer
        if matches!(self.char_state, CharacterState::SpinMelee(s) if s.stage_section == StageSection::Recover)
        {
            agent.action_state.timer = 0.0;
        }

        if attack_data.in_min_range() {
            // If in minimum range
            if agent.action_state.timer > SPIN_WAIT_TIME {
                // If it's been too long since able to hit target, spin
                controller.push_basic_input(InputKind::Secondary);
            } else if attack_data.angle < 30.0 {
                // Else if in angle to strike, strike
                controller.push_basic_input(InputKind::Primary);
            } else {
                // Else increment spin timer
                agent.action_state.timer += read_data.dt.0;
                // If not in angle, apply slight movement so golem orients itself correctly
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .try_normalized()
                    .unwrap_or_else(Vec2::zero)
                    * 0.01;
            }
        } else {
            // Else if too far for melee
            if attack_data.dist_sqrd < SHOCKWAVE_RANGE.powi(2) && attack_data.angle < 45.0 {
                // Shockwave if close enough and haven't shockwaved too recently
                if agent.action_state.counter > SHOCKWAVE_WAIT_TIME {
                    controller.push_basic_input(InputKind::Ability(0));
                }
                if matches!(self.char_state, CharacterState::Shockwave(_)) {
                    agent.action_state.counter = 0.0;
                } else {
                    agent.action_state.counter += read_data.dt.0;
                }
            }
            // And always try to path towards target
            self.path_toward_target(
                agent,
                controller,
                tgt_data.pos.0,
                read_data,
                Path::Partial,
                None,
            );
        }
    }

    pub fn handle_gnarling_chieftain(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        rng: &mut impl Rng,
    ) {
        const TOTEM_TIMER: f32 = 10.0;
        const HEAVY_ATTACK_WAIT_TIME: f32 = 15.0;

        // Handle timers
        agent.action_state.timer += read_data.dt.0;
        match self.char_state {
            CharacterState::BasicSummon(_) => agent.action_state.timer = 0.0,
            CharacterState::Shockwave(_) | CharacterState::BasicRanged(_) => {
                agent.action_state.counter = 0.0
            },
            _ => {},
        }

        if !agent.action_state.initialized {
            // If not initialized yet, start out by summoning green totem
            controller.push_basic_input(InputKind::Ability(2));
            if matches!(self.char_state, CharacterState::BasicSummon(s) if s.stage_section == StageSection::Recover)
            {
                agent.action_state.initialized = true;
            }
        } else if agent.action_state.timer > TOTEM_TIMER {
            // If time to summon a totem, do it
            let input = rng.gen_range(1..=3);
            let buff_kind = match input {
                2 => Some(BuffKind::Regeneration),
                3 => Some(BuffKind::Hastened),
                _ => None,
            };
            if buff_kind.map_or(true, |b| self.has_buff(read_data, b))
                && matches!(self.char_state, CharacterState::Wielding { .. })
            {
                // If already under effects of buff from totem that would be summoned, don't
                // summon totem (doesn't work for red totems since that applies debuff to
                // enemies instead)
                agent.action_state.timer = 0.0;
            } else {
                controller.push_basic_input(InputKind::Ability(input));
            }
        } else if agent.action_state.counter > HEAVY_ATTACK_WAIT_TIME {
            // Else if time for a heavy attack
            if attack_data.in_min_range() {
                // If in range, shockwave
                controller.push_basic_input(InputKind::Ability(0));
            } else if entities_have_line_of_sight(
                self.pos,
                self.body,
                tgt_data.pos,
                tgt_data.body,
                read_data,
            ) {
                // Else if in sight, barrage
                controller.push_basic_input(InputKind::Secondary);
            }
        } else if attack_data.in_min_range() {
            // Else if not time to use anything fancy, if in range and angle, strike them
            if attack_data.angle < 20.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.counter += read_data.dt.0;
            } else {
                // If not in angle, charge heavy attack faster
                agent.action_state.counter += read_data.dt.0 * 5.0;
            }
        } else {
            // If not in range, charge heavy attack faster
            agent.action_state.counter += read_data.dt.0 * 3.3;
        }

        self.path_toward_target(
            agent,
            controller,
            tgt_data.pos.0,
            read_data,
            Path::Full,
            None,
        );
    }
}
