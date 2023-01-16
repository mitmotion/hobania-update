use crate::{consts::MAX_PATH_DIST, data::*, util::entities_have_line_of_sight};
use common::{
    comp::{
        ability::{self, Ability, AbilityKind, ActiveAbilities, AuxiliaryAbility, Capability},
        buff::BuffKind,
        item::tool::AbilityContext,
        skills::{AxeSkill, BowSkill, HammerSkill, SceptreSkill, Skill, StaffSkill, SwordSkill},
        AbilityInput, Agent, CharacterAbility, CharacterState, ControlAction, ControlEvent,
        Controller, InputKind,
    },
    path::TraversalConfig,
    states::{self_buff, sprite_summon, utils::StageSection},
    terrain::Block,
    util::Dir,
    vol::ReadVol,
};
use rand::{prelude::SliceRandom, Rng};
use specs::saveload::MarkerAllocator;
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
    // and the agent is able to freely fly around
    pub fn handle_simple_flying_melee(
        &self,
        _agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
        _rng: &mut impl Rng,
    ) {
        // Fly to target
        let dir_to_target = ((tgt_data.pos.0 + Vec3::unit_z() * 1.5) - self.pos.0)
            .try_normalized()
            .unwrap_or_else(Vec3::zero);
        let speed = 1.0;
        controller.inputs.move_dir = dir_to_target.xy() * speed;

        // Always fly! If the floor can't touch you, it can't hurt you...
        controller.push_basic_input(InputKind::Fly);
        // Flee from the ground! The internet told me it was lava!
        // If on the ground, jump with every last ounce of energy, holding onto
        // all that is dear in life and straining for the wide open skies.
        if self.physics_state.on_ground.is_some() {
            controller.push_basic_input(InputKind::Jump);
        } else {
            // Use a proportional controller with a coefficient of 1.0 to
            // maintain altidude at the the provided set point
            let mut maintain_altitude = |set_point| {
                let alt = read_data
                    .terrain
                    .ray(self.pos.0, self.pos.0 - (Vec3::unit_z() * 7.0))
                    .until(Block::is_solid)
                    .cast()
                    .0;
                let error = set_point - alt;
                controller.inputs.move_z = error;
            };
            if (tgt_data.pos.0 - self.pos.0).xy().magnitude_squared() > (5.0_f32).powi(2) {
                maintain_altitude(5.0);
            } else {
                maintain_altitude(2.0);

                // Attack if in range
                if attack_data.dist_sqrd < 3.5_f32.powi(2) && attack_data.angle < 150.0 {
                    controller.push_basic_input(InputKind::Primary);
                }
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
        enum ActionStateTimers {
            TimerHandleAxeAttack = 0,
        }
        let has_leap = || self.skill_set.has_skill(Skill::Axe(AxeSkill::UnlockLeap));
        let has_energy = |need| self.energy.current() > need;
        let use_leap = |controller: &mut Controller| {
            controller.push_basic_input(InputKind::Ability(0));
        };

        if attack_data.in_min_range() && attack_data.angle < 45.0 {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize] > 5.0 {
                controller.push_cancel_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize] = 0.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize]
                > 2.5
                && has_energy(10.0)
            {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize] +=
                    read_data.dt.0;
            } else if has_leap() && has_energy(45.0) && rng.gen_bool(0.5) {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize] +=
                    read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerHandleAxeAttack as usize] +=
                    read_data.dt.0;
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
        enum ActionStateTimers {
            TimerHandleHammerAttack = 0,
        }
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
            if agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize] > 4.0
            {
                controller.push_cancel_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize] =
                    0.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize]
                > 3.0
            {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize] +=
                    read_data.dt.0;
            } else if has_leap() && has_energy(50.0) && rng.gen_bool(0.9) {
                use_leap(controller);
                agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize] +=
                    read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerHandleHammerAttack as usize] +=
                    read_data.dt.0;
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
        const INT_COUNTER_STANCE: usize = 0;
        use ability::SwordStance;
        let stance = |stance| match stance {
            1 => SwordStance::Offensive,
            2 => SwordStance::Defensive,
            3 => SwordStance::Mobility,
            4 => SwordStance::Crippling,
            5 => SwordStance::Cleaving,
            6 => SwordStance::Parrying,
            7 => SwordStance::Heavy,
            8 => SwordStance::Reaching,
            _ => SwordStance::Balanced,
        };
        if !agent.action_state.initialized {
            // TODO: Don't always assume that if they have skill checked for, they have
            // entire set of necessary skills to take full advantage of AI. Make sure to
            // change this to properly query for all required skills when AI dynamically get
            // new skills and don't have pre-created skill sets.
            agent.action_state.int_counters[INT_COUNTER_STANCE] = if self
                .skill_set
                .has_skill(Skill::Sword(SwordSkill::CripplingFinisher))
            {
                // Hack to make cleaving stance come up less often because agents cannot use it
                // effectively due to only considering a single target
                // Remove when agents properly consider multiple targets
                // (stance, weight)
                let options = [(4, 3), (5, 1), (6, 3), (7, 3), (8, 3)];
                options.choose_weighted(rng, |x| x.1).map_or(0, |x| x.0)
                // rng.gen_range(4..9)
            } else if self
                .skill_set
                .has_skill(Skill::Sword(SwordSkill::OffensiveFinisher))
            {
                rng.gen_range(1..4)
            } else {
                0
            };
            let auxiliary_key = ActiveAbilities::active_auxiliary_key(Some(self.inventory));
            let mut set_sword_ability = |slot, skill| {
                controller.push_event(ControlEvent::ChangeAbility {
                    slot,
                    auxiliary_key,
                    new_ability: AuxiliaryAbility::MainWeapon(skill),
                })
            };
            match stance(agent.action_state.int_counters[INT_COUNTER_STANCE]) {
                SwordStance::Balanced => {
                    // Balanced finisher
                    set_sword_ability(0, 8);
                },
                SwordStance::Offensive => {
                    // Offensive combo
                    set_sword_ability(0, 0);
                    // Offensive advance
                    set_sword_ability(1, 9);
                    // Offensive finisher
                    set_sword_ability(2, 8);
                },
                SwordStance::Defensive => {
                    // Defensive combo
                    set_sword_ability(0, 3);
                    // Defensive retreat
                    set_sword_ability(1, 9);
                    // Defensive bulwark
                    set_sword_ability(2, 10);
                },
                SwordStance::Mobility => {
                    // Mobility combo
                    set_sword_ability(0, 6);
                    // Mobility feint
                    set_sword_ability(1, 9);
                    // Mobility agility
                    set_sword_ability(2, 10);
                },
                SwordStance::Crippling => {
                    // Crippling combo
                    set_sword_ability(0, 1);
                    // Crippling finisher
                    set_sword_ability(1, 8);
                    // Crippling strike
                    set_sword_ability(2, 9);
                    // Crippling gouge
                    set_sword_ability(3, 10);
                },
                SwordStance::Cleaving => {
                    // Cleaving combo
                    set_sword_ability(0, 2);
                    // Cleaving finisher
                    set_sword_ability(1, 8);
                    // Cleaving spin
                    set_sword_ability(2, 10);
                    // Cleaving dive
                    set_sword_ability(3, 9);
                },
                SwordStance::Parrying => {
                    // Parrying combo
                    set_sword_ability(0, 4);
                    // Parrying parry
                    set_sword_ability(1, 10);
                    // Parrying riposte
                    set_sword_ability(2, 9);
                    // Parrying counter
                    set_sword_ability(3, 8);
                },
                SwordStance::Heavy => {
                    // Heavy combo
                    set_sword_ability(0, 5);
                    // Heavy finisher
                    set_sword_ability(1, 8);
                    // Heavy pommelstrike
                    set_sword_ability(2, 10);
                    // Heavy fortitude
                    set_sword_ability(3, 9);
                },
                SwordStance::Reaching => {
                    // Reaching combo
                    set_sword_ability(0, 7);
                    // Reaching charge
                    set_sword_ability(1, 9);
                    // Reaching flurry
                    set_sword_ability(2, 8);
                    // Reaching skewer
                    set_sword_ability(3, 10);
                },
            }
            agent.action_state.initialized = true;
        }

        let advance = |agent: &mut Agent, controller: &mut Controller, dist: f32, angle: f32| {
            if attack_data.dist_sqrd > (dist + self.body.map_or(0.0, |b| b.max_radius())).powi(2)
                || attack_data.angle > angle
            {
                self.path_toward_target(
                    agent,
                    controller,
                    tgt_data.pos.0,
                    read_data,
                    Path::Separate,
                    None,
                );
            }
        };

        const AVERAGE_ROLL_FREQUENCY: f32 = 12.0;
        const TIMER_LAST_ROLL: usize = 0;
        const CONDITION_RANDOM_ROLL: usize = 0;
        const MIN_ENERGY_FOR_ROLL: f32 = 30.0;
        if self.energy.current() > MIN_ENERGY_FOR_ROLL {
            agent.action_state.timers[TIMER_LAST_ROLL] += read_data.dt.0;
        }
        if (-agent.action_state.timers[TIMER_LAST_ROLL] / AVERAGE_ROLL_FREQUENCY).exp()
            < rng.gen::<f32>() / std::f32::consts::E
        {
            agent.action_state.conditions[CONDITION_RANDOM_ROLL] = true;
        }
        if agent.action_state.conditions[CONDITION_RANDOM_ROLL] {
            controller.push_basic_input(InputKind::Roll);
            advance(agent, controller, 1.0, 30.0);
            let random_angle = rng.gen_range(-PI..PI) / 4.0;
            controller.inputs.move_dir.rotated_z(random_angle);
        }
        if matches!(self.char_state, CharacterState::Roll(c) if c.stage_section == StageSection::Recover)
        {
            agent.action_state.timers[TIMER_LAST_ROLL] = 0.0;
            agent.action_state.conditions[CONDITION_RANDOM_ROLL] = false;
        }

        // Called when out of energy, or the situation is not right to use another
        // ability. Only contains tactics for using M1 and M2
        let fallback_rng_1 = rng.gen::<f32>() < 0.67;
        let fallback_tactics = |agent: &mut Agent, controller: &mut Controller| {
            const BALANCED_COMBO: ComboMeleeData = ComboMeleeData {
                min_range: 0.0,
                max_range: 2.5,
                angle: 40.0,
                energy: 0.0,
            };
            const BALANCED_THRUST: ComboMeleeData = ComboMeleeData {
                min_range: 0.0,
                max_range: 4.0,
                angle: 8.0,
                energy: 0.0,
            };

            let balanced_thrust_targetable = BALANCED_THRUST.could_use(attack_data, self);

            if (matches!(self.char_state, CharacterState::ChargedMelee(c) if c.charge_amount < 0.95)
                || matches!(
                    tgt_data.char_state.and_then(|cs| cs.stage_section()),
                    Some(StageSection::Buildup)
                ))
                && balanced_thrust_targetable
            {
                // If target in buildup (and therefore likely to still be around for an attack
                // that needs charging), thrust
                // Or if already thrusting, keep thrussting
                controller.push_basic_input(InputKind::Secondary);
            } else if BALANCED_COMBO.could_use(attack_data, self) && fallback_rng_1 {
                controller.push_basic_input(InputKind::Primary);
            } else if balanced_thrust_targetable {
                controller.push_basic_input(InputKind::Secondary);
            }

            advance(agent, controller, 1.5, BALANCED_COMBO.angle);
        };

        let in_stance = |stance| {
            if let CharacterState::ComboMelee2(c) = self.char_state {
                c.static_data.is_stance
                    && c.static_data
                        .ability_info
                        .ability_meta
                        .and_then(|meta| meta.kind)
                        .map_or(false, |kind| AbilityKind::Sword(stance) == kind)
            } else {
                false
            }
        };

        match stance(agent.action_state.int_counters[INT_COUNTER_STANCE]) {
            SwordStance::Balanced => {
                const BALANCED_FINISHER: FinisherMeleeData = FinisherMeleeData {
                    range: 2.5,
                    angle: 12.5,
                    energy: 30.0,
                    combo: 10,
                };

                if self
                    .skill_set
                    .has_skill(Skill::Sword(SwordSkill::BalancedFinisher))
                    && BALANCED_FINISHER.could_use(attack_data, self)
                {
                    controller.push_basic_input(InputKind::Ability(0));
                    advance(
                        agent,
                        controller,
                        BALANCED_FINISHER.range,
                        BALANCED_FINISHER.angle,
                    );
                } else {
                    fallback_tactics(agent, controller);
                }
            },
            SwordStance::Offensive => {
                const OFFENSIVE_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.0,
                    angle: 30.0,
                    energy: 4.0,
                };
                const OFFENSIVE_ADVANCE: ComboMeleeData = ComboMeleeData {
                    min_range: 6.0,
                    max_range: 10.0,
                    angle: 20.0,
                    energy: 10.0,
                };
                const OFFENSIVE_FINISHER: FinisherMeleeData = FinisherMeleeData {
                    range: 2.0,
                    angle: 10.0,
                    energy: 40.0,
                    combo: 10,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Offensive) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else {
                    let used_finisher = if OFFENSIVE_FINISHER.could_use(attack_data, self)
                        && OFFENSIVE_FINISHER.use_desirable(tgt_data, self)
                    {
                        controller.push_basic_input(InputKind::Ability(2));
                        advance(
                            agent,
                            controller,
                            OFFENSIVE_FINISHER.range,
                            OFFENSIVE_FINISHER.angle,
                        );
                        true
                    } else {
                        false
                    };

                    if !used_finisher {
                        if OFFENSIVE_COMBO.could_use(attack_data, self) {
                            controller.push_basic_input(InputKind::Primary);
                            advance(
                                agent,
                                controller,
                                OFFENSIVE_COMBO.max_range,
                                OFFENSIVE_COMBO.angle,
                            );
                        } else if OFFENSIVE_ADVANCE.could_use(attack_data, self) {
                            controller.push_basic_input(InputKind::Ability(1));
                            advance(
                                agent,
                                controller,
                                OFFENSIVE_ADVANCE.max_range,
                                OFFENSIVE_ADVANCE.angle,
                            );
                        } else {
                            advance(
                                agent,
                                controller,
                                OFFENSIVE_COMBO.max_range,
                                OFFENSIVE_COMBO.angle,
                            );
                        }
                    }
                }
            },
            SwordStance::Defensive => {
                const DEFENSIVE_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 3.0,
                };
                const DEFENSIVE_RETREAT: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 3.0,
                    angle: 35.0,
                    energy: 10.0,
                };
                const DEFENSIVE_BULWARK: SelfBuffData = SelfBuffData {
                    buff: BuffKind::ProtectingWard,
                    energy: 40.0,
                };
                const BASIC_BLOCK: BlockData = BlockData {
                    angle: 55.0,
                    range: 5.0,
                    energy: 2.5,
                };
                const DESIRED_ENERGY: f32 = 50.0;
                const CONDITION_HOLD: usize = 1;
                const COUNTER_ANGLE: usize = 1;
                const TIMER_HOLD_TIMEOUT: usize = 1;
                const HOLD_TIMEOUT: f32 = 3.0;

                let mut try_block = || {
                    if let Some(char_state) = tgt_data.char_state {
                        matches!(char_state.stage_section(), Some(StageSection::Buildup))
                            && char_state.is_melee_attack()
                            && BASIC_BLOCK.could_use(attack_data, self)
                            && char_state
                                .timer()
                                .map_or(false, |timer| rng.gen::<f32>() < timer.as_secs_f32() * 4.0)
                            && self
                                .char_state
                                .ability_info()
                                .and_then(|a| a.ability_meta)
                                .map(|m| m.capabilities)
                                .map_or(false, |c| c.contains(Capability::BLOCK_INTERRUPT))
                    } else {
                        false
                    }
                };

                if agent.action_state.conditions[CONDITION_HOLD] {
                    agent.action_state.timers[TIMER_HOLD_TIMEOUT] += read_data.dt.0;
                }

                if read_data.time.0
                    - self
                        .health
                        .map_or(read_data.time.0, |h| h.last_change.time.0)
                    < read_data.dt.0 as f64 * 2.0
                    || agent.action_state.timers[TIMER_HOLD_TIMEOUT] > HOLD_TIMEOUT
                {
                    // If attacked in last couple ticks, stop standing still
                    agent.action_state.conditions[CONDITION_HOLD] = false;
                    agent.action_state.timers[TIMER_HOLD_TIMEOUT] = 0.0;
                } else if matches!(
                    self.char_state.ability_info().and_then(|info| info.input),
                    Some(InputKind::Ability(1))
                ) {
                    // If used defensive retreat, stand still for a little bit to bait people
                    // forward
                    agent.action_state.conditions[CONDITION_HOLD] = true;
                };

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Defensive) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if DEFENSIVE_BULWARK.could_use(self) && DEFENSIVE_BULWARK.use_desirable(self)
                {
                    controller.push_basic_input(InputKind::Ability(2));
                } else if try_block() {
                    controller.push_basic_input(InputKind::Block);
                } else if DEFENSIVE_RETREAT.could_use(attack_data, self) && rng.gen::<f32>() < 0.2 {
                    controller.push_basic_input(InputKind::Ability(1));
                    if !agent.action_state.conditions[CONDITION_HOLD] {
                        advance(
                            agent,
                            controller,
                            DEFENSIVE_RETREAT.max_range,
                            DEFENSIVE_RETREAT.angle,
                        );
                    }
                } else if DEFENSIVE_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    if !agent.action_state.conditions[CONDITION_HOLD] {
                        advance(
                            agent,
                            controller,
                            DEFENSIVE_COMBO.max_range,
                            DEFENSIVE_COMBO.angle,
                        );
                    }
                } else if agent.action_state.conditions[CONDITION_HOLD] {
                    agent.action_state.counters[COUNTER_ANGLE] += rng.gen::<f32>() * read_data.dt.0;
                    controller.inputs.move_dir =
                        Vec2::unit_x().rotated_z(agent.action_state.counters[COUNTER_ANGLE]) * 0.25;
                } else {
                    advance(
                        agent,
                        controller,
                        DEFENSIVE_COMBO.max_range,
                        DEFENSIVE_COMBO.angle,
                    );
                }
            },
            SwordStance::Mobility => {
                const MOBILITY_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 3.0,
                };
                const MOBILITY_FEINT: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 3.0,
                    angle: 35.0,
                    energy: 10.0,
                };
                const MOBILITY_AGILITY: SelfBuffData = SelfBuffData {
                    buff: BuffKind::Hastened,
                    energy: 40.0,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                let mut try_roll = |controller: &mut Controller| {
                    if let Some(char_state) = tgt_data.char_state {
                        let try_roll =
                            matches!(char_state.stage_section(), Some(StageSection::Buildup))
                                && char_state.is_melee_attack()
                                && char_state.timer().map_or(false, |timer| {
                                    rng.gen::<f32>() < timer.as_secs_f32() * 4.0
                                })
                                && matches!(
                                    self.char_state.stage_section(),
                                    Some(StageSection::Recover)
                                )
                                && self
                                    .char_state
                                    .ability_info()
                                    .and_then(|a| a.ability_meta)
                                    .map(|m| m.capabilities)
                                    .map_or(false, |c| c.contains(Capability::ROLL_INTERRUPT));
                        if try_roll {
                            controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                                .xy()
                                .try_normalized()
                                .unwrap_or_default();
                            controller.push_basic_input(InputKind::Roll);
                        }
                        try_roll
                    } else {
                        false
                    }
                };

                if matches!(self.char_state, CharacterState::Roll(_)) {
                    controller.push_basic_input(InputKind::Jump);
                }

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Mobility) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if MOBILITY_AGILITY.could_use(self) && MOBILITY_AGILITY.use_desirable(self) {
                    controller.push_basic_input(InputKind::Ability(2));
                } else if try_roll(controller) {
                    // Nothing here, rolling handled in try_roll function
                } else if MOBILITY_FEINT.could_use(attack_data, self) && rng.gen::<f32>() < 0.3 {
                    controller.push_basic_input(InputKind::Ability(1));
                    advance(
                        agent,
                        controller,
                        MOBILITY_FEINT.max_range,
                        MOBILITY_FEINT.angle,
                    );
                } else if MOBILITY_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(
                        agent,
                        controller,
                        MOBILITY_COMBO.max_range,
                        MOBILITY_COMBO.angle,
                    );
                } else {
                    advance(
                        agent,
                        controller,
                        MOBILITY_COMBO.max_range,
                        MOBILITY_COMBO.angle,
                    );
                }

                const CONDITION_FEINT_DIR: usize = 1;

                if rng.gen::<f32>() < read_data.dt.0 {
                    agent.action_state.conditions[CONDITION_FEINT_DIR] =
                        !agent.action_state.conditions[CONDITION_FEINT_DIR];
                }
                let dir = if agent.action_state.conditions[CONDITION_FEINT_DIR] {
                    1.0
                } else {
                    -1.0
                };
                controller.inputs.move_dir.rotated_z(PI / 4.0 * dir);
            },
            SwordStance::Crippling => {
                const CRIPPLING_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 5.0,
                };
                const CRIPPLING_GOUGE: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 3.0,
                    angle: 35.0,
                    energy: 25.0,
                };
                const CRIPPLING_STRIKE: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 3.0,
                    angle: 35.0,
                    energy: 25.0,
                };
                const CRIPPLING_FINISHER: FinisherMeleeData = FinisherMeleeData {
                    range: 2.5,
                    angle: 10.0,
                    energy: 40.0,
                    combo: 10,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Crippling) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if CRIPPLING_FINISHER.could_use(attack_data, self)
                    && CRIPPLING_FINISHER.use_desirable(tgt_data, self)
                {
                    controller.push_basic_input(InputKind::Ability(1));
                    advance(
                        agent,
                        controller,
                        CRIPPLING_FINISHER.range,
                        CRIPPLING_FINISHER.angle,
                    );
                } else if tgt_data
                    .buffs
                    .map_or(false, |buffs| !buffs.contains(BuffKind::Crippled))
                    && CRIPPLING_STRIKE.could_use(attack_data, self)
                {
                    controller.push_basic_input(InputKind::Ability(2));
                    advance(
                        agent,
                        controller,
                        CRIPPLING_STRIKE.max_range,
                        CRIPPLING_STRIKE.angle,
                    );
                } else if CRIPPLING_GOUGE.could_use(attack_data, self) && rng.gen::<f32>() < 0.3 {
                    controller.push_basic_input(InputKind::Ability(3));
                    advance(
                        agent,
                        controller,
                        CRIPPLING_GOUGE.max_range,
                        CRIPPLING_GOUGE.angle,
                    );
                } else if CRIPPLING_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(
                        agent,
                        controller,
                        CRIPPLING_COMBO.max_range,
                        CRIPPLING_COMBO.angle,
                    );
                } else {
                    advance(
                        agent,
                        controller,
                        CRIPPLING_COMBO.max_range,
                        CRIPPLING_COMBO.angle,
                    );
                }
            },
            SwordStance::Cleaving => {
                // TODO: Rewrite cleaving stance tactics when agents can consider multiple
                // targets at once. Remove hack to make cleaving AI appear less frequently above
                // when doing so.
                const CLEAVING_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 6.0,
                };
                const CLEAVING_FINISHER: FinisherMeleeData = FinisherMeleeData {
                    range: 2.0,
                    angle: 10.0,
                    energy: 40.0,
                    combo: 10,
                };
                const CLEAVING_SPIN: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 5.0,
                    angle: 360.0,
                    energy: 20.0,
                };
                const CLEAVING_DIVE: DiveMeleeData = DiveMeleeData {
                    range: 5.0,
                    angle: 10.0,
                    energy: 25.0,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                // TODO: Remove when agents actually have multiple targets
                // Hacky check for multiple nearby melee range targets
                let are_nearby_targets = || {
                    if let Some(health) = self.health {
                        health
                            .recent_damagers()
                            .filter(|(_, time)| read_data.time.0 - time.0 < 5.0)
                            .filter_map(|(uid, _)| {
                                read_data.uid_allocator.retrieve_entity_internal(uid.0)
                            })
                            .filter(|e| {
                                read_data.positions.get(*e).map_or(false, |pos| {
                                    pos.0.distance_squared(self.pos.0) < 10_f32.powi(2)
                                })
                            })
                            .count()
                            > 1
                    } else {
                        false
                    }
                };

                const CONDITION_SELF_ROLLING: usize = 1;
                const TIMER_DIVE_TIMEOUT: usize = 1;

                if matches!(self.char_state, CharacterState::Roll(_)) {
                    agent.action_state.conditions[CONDITION_SELF_ROLLING] = true;
                }

                if agent.action_state.conditions[CONDITION_SELF_ROLLING] {
                    if self.physics_state.on_ground.is_some() {
                        controller.push_basic_input(InputKind::Jump);
                    } else {
                        controller.push_basic_input(InputKind::Ability(3));
                    }
                    agent.action_state.timers[TIMER_DIVE_TIMEOUT] += read_data.dt.0;
                    if agent.action_state.timers[TIMER_DIVE_TIMEOUT] > 2.0 {
                        agent.action_state.timers[TIMER_DIVE_TIMEOUT] = 0.0;
                        agent.action_state.conditions[CONDITION_SELF_ROLLING] = false;
                    }
                    advance(agent, controller, CLEAVING_DIVE.range, CLEAVING_DIVE.angle);
                } else if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Cleaving) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if CLEAVING_FINISHER.could_use(attack_data, self)
                    && CLEAVING_FINISHER.use_desirable(tgt_data, self)
                {
                    controller.push_basic_input(InputKind::Ability(1));
                    advance(
                        agent,
                        controller,
                        CLEAVING_FINISHER.range,
                        CLEAVING_FINISHER.angle,
                    );
                } else if CLEAVING_SPIN.could_use(attack_data, self) && are_nearby_targets() {
                    controller.push_basic_input(InputKind::Ability(2));
                    advance(
                        agent,
                        controller,
                        CLEAVING_SPIN.max_range,
                        CLEAVING_SPIN.angle,
                    );
                } else if CLEAVING_DIVE.npc_should_use_hack(self, tgt_data) {
                    controller.push_basic_input(InputKind::Roll);
                    advance(agent, controller, CLEAVING_DIVE.range, CLEAVING_DIVE.angle);
                } else if CLEAVING_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(
                        agent,
                        controller,
                        CLEAVING_COMBO.max_range,
                        CLEAVING_COMBO.angle,
                    );
                } else {
                    advance(
                        agent,
                        controller,
                        CLEAVING_COMBO.max_range,
                        CLEAVING_COMBO.angle,
                    );
                }
            },
            SwordStance::Parrying => {
                const PARRYING_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 6.0,
                };
                const PARRYING_PARRY: BlockData = BlockData {
                    angle: 40.0,
                    range: 5.0,
                    energy: 10.0,
                };
                const PARRYING_RIPOSTE: BlockData = BlockData {
                    angle: 15.0,
                    range: 3.5,
                    energy: 20.0,
                };
                const PARRYING_COUNTER: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 25.0,
                    energy: 15.0,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                let try_parry = |pregen_rng_1: f32| {
                    if let Some(char_state) = tgt_data.char_state {
                        matches!(char_state.stage_section(), Some(StageSection::Buildup))
                            && char_state.is_melee_attack()
                            && char_state
                                .timer()
                                .map_or(false, |timer| pregen_rng_1 < timer.as_secs_f32() * 4.0)
                    } else {
                        false
                    }
                };

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Parrying) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if matches!(
                    tgt_data.char_state.and_then(|cs| cs.stage_section()),
                    Some(StageSection::Buildup)
                ) && PARRYING_COUNTER.could_use(attack_data, self)
                    && rng.gen::<f32>() < self.health.map_or(0.0, |h| h.fraction())
                {
                    controller.push_basic_input(InputKind::Ability(3));
                    advance(
                        agent,
                        controller,
                        PARRYING_COUNTER.max_range,
                        PARRYING_COUNTER.angle,
                    );
                } else if try_parry(rng.gen::<f32>()) {
                    if PARRYING_RIPOSTE.could_use(attack_data, self)
                        && rng.gen::<f32>() > self.health.map_or(0.0, |h| h.fraction())
                    {
                        controller.push_basic_input(InputKind::Ability(2));
                        advance(
                            agent,
                            controller,
                            PARRYING_RIPOSTE.range,
                            PARRYING_RIPOSTE.angle,
                        );
                    } else if PARRYING_PARRY.could_use(attack_data, self) {
                        controller.push_basic_input(InputKind::Ability(1));
                        advance(
                            agent,
                            controller,
                            PARRYING_PARRY.range,
                            PARRYING_PARRY.angle,
                        );
                    } else if PARRYING_COMBO.could_use(attack_data, self) {
                        controller.push_basic_input(InputKind::Primary);
                        advance(
                            agent,
                            controller,
                            PARRYING_COMBO.max_range,
                            PARRYING_COMBO.angle,
                        );
                    } else {
                        advance(
                            agent,
                            controller,
                            PARRYING_COMBO.max_range,
                            PARRYING_COMBO.angle,
                        );
                    }
                } else if PARRYING_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(
                        agent,
                        controller,
                        PARRYING_COMBO.max_range,
                        PARRYING_COMBO.angle,
                    );
                } else {
                    advance(
                        agent,
                        controller,
                        PARRYING_COMBO.max_range,
                        PARRYING_COMBO.angle,
                    );
                }
            },
            SwordStance::Heavy => {
                const HEAVY_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 2.5,
                    angle: 35.0,
                    energy: 5.0,
                };
                const HEAVY_FINISHER: FinisherMeleeData = FinisherMeleeData {
                    range: 2.5,
                    angle: 10.0,
                    energy: 40.0,
                    combo: 10,
                };
                const HEAVY_POMMELSTRIKE: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 3.0,
                    angle: 3.0,
                    energy: 15.0,
                };
                const HEAVY_FORTITUDE: SelfBuffData = SelfBuffData {
                    buff: BuffKind::Fortitude,
                    energy: 40.0,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                const CONDITION_POISE_DMG: usize = 1;
                const TIMER_POMMELSTRIKE: usize = 1;

                agent.action_state.conditions[CONDITION_POISE_DMG] = self
                    .poise
                    .map_or(false, |p| p.current() < p.maximum() * 0.8);
                if matches!(
                    self.char_state.ability_info().and_then(|info| info.ability),
                    Some(Ability::MainWeaponAux(21))
                ) {
                    agent.action_state.timers[TIMER_POMMELSTRIKE] = 0.0;
                }

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Heavy) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if HEAVY_FORTITUDE.could_use(self)
                    && HEAVY_FORTITUDE.use_desirable(self)
                    && agent.action_state.conditions[CONDITION_POISE_DMG]
                {
                    controller.push_basic_input(InputKind::Ability(3));
                } else if HEAVY_FINISHER.could_use(attack_data, self)
                    && HEAVY_FINISHER.use_desirable(tgt_data, self)
                {
                    controller.push_basic_input(InputKind::Ability(1));
                    advance(
                        agent,
                        controller,
                        HEAVY_FINISHER.range,
                        HEAVY_FINISHER.angle,
                    );
                } else if HEAVY_POMMELSTRIKE.could_use(attack_data, self)
                    && rng.gen::<f32>()
                        < agent.action_state.timers[TIMER_POMMELSTRIKE]
                            * self.poise.map_or(0.5, |p| p.fraction())
                {
                    controller.push_basic_input(InputKind::Ability(2));
                    advance(
                        agent,
                        controller,
                        HEAVY_POMMELSTRIKE.max_range,
                        HEAVY_POMMELSTRIKE.angle,
                    );
                } else if HEAVY_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(agent, controller, HEAVY_COMBO.max_range, HEAVY_COMBO.angle);
                    agent.action_state.timers[TIMER_POMMELSTRIKE] += read_data.dt.0;
                } else {
                    advance(agent, controller, HEAVY_COMBO.max_range, HEAVY_COMBO.angle);
                }
            },
            SwordStance::Reaching => {
                const REACHING_COMBO: ComboMeleeData = ComboMeleeData {
                    min_range: 0.0,
                    max_range: 4.5,
                    angle: 6.0,
                    energy: 5.0,
                };
                const REACHING_CHARGE: DashMeleeData = DashMeleeData {
                    range: 3.5,
                    angle: 12.5,
                    initial_energy: 10.0,
                    energy_drain: 20.0,
                    speed: 3.0,
                    charge_dur: 1.0,
                };
                const REACHING_FLURRY: RapidMeleeData = RapidMeleeData {
                    range: 4.0,
                    angle: 7.5,
                    energy: 10.0,
                    strikes: 6,
                };
                const REACHING_SKEWER: ComboMeleeData = ComboMeleeData {
                    min_range: 1.0,
                    max_range: 7.5,
                    angle: 7.5,
                    energy: 15.0,
                };
                const DESIRED_ENERGY: f32 = 50.0;

                if self.energy.current() < DESIRED_ENERGY {
                    fallback_tactics(agent, controller);
                } else if !in_stance(SwordStance::Reaching) {
                    controller.push_basic_input(InputKind::Ability(0));
                } else if matches!(self.char_state, CharacterState::DashMelee(s) if s.stage_section != StageSection::Recover)
                    || (REACHING_CHARGE.could_use(attack_data, self)
                        && REACHING_CHARGE.use_desirable(attack_data, self))
                {
                    controller.push_basic_input(InputKind::Ability(1));
                    advance(
                        agent,
                        controller,
                        REACHING_CHARGE.range,
                        REACHING_CHARGE.angle,
                    );
                } else if REACHING_FLURRY.could_use(attack_data, self)
                    && matches!(
                        tgt_data.char_state.and_then(|cs| cs.stage_section()),
                        Some(StageSection::Buildup)
                    )
                    && rng.gen::<f32>() < 0.5
                {
                    controller.push_basic_input(InputKind::Ability(2));
                    advance(
                        agent,
                        controller,
                        REACHING_FLURRY.range,
                        REACHING_FLURRY.angle,
                    );
                } else if REACHING_SKEWER.could_use(attack_data, self) && rng.gen::<f32>() < 0.33 {
                    controller.push_basic_input(InputKind::Ability(3));
                    advance(
                        agent,
                        controller,
                        REACHING_SKEWER.max_range,
                        REACHING_SKEWER.angle,
                    );
                } else if REACHING_COMBO.could_use(attack_data, self) {
                    controller.push_basic_input(InputKind::Primary);
                    advance(
                        agent,
                        controller,
                        REACHING_COMBO.max_range,
                        REACHING_COMBO.angle,
                    );
                } else {
                    advance(
                        agent,
                        controller,
                        REACHING_COMBO.max_range,
                        REACHING_COMBO.angle,
                    );
                }
            },
        }

        if self.active_abilities.auxiliary_sets.is_empty() {
            agent.action_state.initialized = false;
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
                && self.energy.current() > CharacterAbility::default_roll().energy_cost()
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
        enum ActionStateConditions {
            ConditionStaffCanShockwave = 0,
        }
        let context = AbilityContext::try_from(Some(self.char_state));
        let extract_ability = |input: AbilityInput| {
            self.active_abilities
                .activate_ability(
                    input,
                    Some(self.inventory),
                    self.skill_set,
                    self.body,
                    context,
                )
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
        let shockwave_cost = shockwave.energy_cost();
        if self.body.map_or(false, |b| b.is_humanoid())
            && attack_data.in_min_range()
            && self.energy.current() > CharacterAbility::default_roll().energy_cost()
            && !matches!(self.char_state, CharacterState::Shockwave(_))
        {
            // if a humanoid, have enough stamina, not in shockwave, and in melee range,
            // emergency roll
            controller.push_basic_input(InputKind::Roll);
        } else if matches!(self.char_state, CharacterState::Shockwave(_)) {
            agent.action_state.conditions
                [ActionStateConditions::ConditionStaffCanShockwave as usize] = false;
        } else if agent.action_state.conditions
            [ActionStateConditions::ConditionStaffCanShockwave as usize]
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
                    agent.action_state.conditions
                        [ActionStateConditions::ConditionStaffCanShockwave as usize] = true;
                }
            } else if self.energy.current()
                > shockwave_cost + CharacterAbility::default_roll().energy_cost()
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
                && self.energy.current() > CharacterAbility::default_roll().energy_cost()
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
        enum ActionStateTimers {
            TimerHandleStoneGolemAttack = 0, //Timer 0
        }

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
                if agent.action_state.timers
                    [ActionStateTimers::TimerHandleStoneGolemAttack as usize]
                    > 5.0
                {
                    controller.push_basic_input(InputKind::Secondary);
                    agent.action_state.timers
                        [ActionStateTimers::TimerHandleStoneGolemAttack as usize] = 0.0;
                } else {
                    agent.action_state.timers
                        [ActionStateTimers::TimerHandleStoneGolemAttack as usize] += read_data.dt.0;
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
        enum ActionStateCountersF {
            CounterFHandleCircleChargeAttack = 0,
        }

        enum ActionStateCountersI {
            CounterIHandleCircleChargeAttack = 0,
        }

        if agent.action_state.counters
            [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize]
            >= circle_time as f32
        {
            // if circle charge is in progress and time hasn't expired, continue charging
            controller.push_basic_input(InputKind::Secondary);
        }
        if attack_data.in_min_range() {
            if agent.action_state.counters
                [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize]
                > 0.0
            {
                // set timer and rotation counter to zero if in minimum range
                agent.action_state.counters
                    [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize] = 0.0;
                agent.action_state.int_counters
                    [ActionStateCountersI::CounterIHandleCircleChargeAttack as usize] = 0;
            } else {
                // melee attack
                controller.push_basic_input(InputKind::Primary);
                controller.inputs.move_dir = Vec2::zero();
            }
        } else if attack_data.dist_sqrd < (radius as f32 + attack_data.min_attack_dist).powi(2) {
            // if in range to charge, circle, then charge
            if agent.action_state.int_counters
                [ActionStateCountersI::CounterIHandleCircleChargeAttack as usize]
                == 0
            {
                // if you haven't chosen a direction to go in, choose now
                agent.action_state.int_counters
                    [ActionStateCountersI::CounterIHandleCircleChargeAttack as usize] =
                    1 + rng.gen_bool(0.5) as u8;
            }
            if agent.action_state.counters
                [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize]
                < circle_time as f32
            {
                // circle if circle timer not ready
                let move_dir = match agent.action_state.int_counters
                    [ActionStateCountersI::CounterIHandleCircleChargeAttack as usize]
                {
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
                    agent.action_state.counters
                        [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize] =
                        circle_time as f32;
                }
                controller.inputs.move_dir = move_dir;
                // use counter as timer since timer may be modified in other parts of the code
                agent.action_state.counters
                    [ActionStateCountersF::CounterFHandleCircleChargeAttack as usize] +=
                    read_data.dt.0;
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
        enum ActionStateTimers {
            TimerHandleQuadLowRanged = 0,
        }

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
                    if agent.action_state.timers
                        [ActionStateTimers::TimerHandleQuadLowRanged as usize]
                        > 5.0
                    {
                        agent.action_state.timers
                            [ActionStateTimers::TimerHandleQuadLowRanged as usize] = 0.0;
                    } else if agent.action_state.timers
                        [ActionStateTimers::TimerHandleQuadLowRanged as usize]
                        > 2.5
                    {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(1.75 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timers
                            [ActionStateTimers::TimerHandleQuadLowRanged as usize] +=
                            read_data.dt.0;
                    } else {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(0.25 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timers
                            [ActionStateTimers::TimerHandleQuadLowRanged as usize] +=
                            read_data.dt.0;
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
        enum ActionStateTimers {
            TimerTailSlap = 0,
        }

        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (1.5 * attack_data.min_attack_dist).powi(2)
        {
            if agent.action_state.timers[ActionStateTimers::TimerTailSlap as usize] > 4.0 {
                controller.push_cancel_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerTailSlap as usize] = 0.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerTailSlap as usize] > 1.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerTailSlap as usize] +=
                    read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerTailSlap as usize] +=
                    read_data.dt.0;
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
        enum ActionStateTimers {
            TimerQuadLowBasic = 0,
        }

        if attack_data.angle < 70.0
            && attack_data.dist_sqrd < (1.3 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timers[ActionStateTimers::TimerQuadLowBasic as usize] > 5.0 {
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBasic as usize] = 0.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerQuadLowBasic as usize] > 2.0
            {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBasic as usize] +=
                    read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBasic as usize] +=
                    read_data.dt.0;
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
        enum ActionStateTimers {
            TimerQuadMedBasic = 0,
        }

        if attack_data.angle < 90.0 && attack_data.in_min_range() {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timers[ActionStateTimers::TimerQuadMedBasic as usize] < 2.0 {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerQuadMedBasic as usize] +=
                    read_data.dt.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerQuadMedBasic as usize] < 3.0
            {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerQuadMedBasic as usize] +=
                    read_data.dt.0;
            } else {
                agent.action_state.timers[ActionStateTimers::TimerQuadMedBasic as usize] = 0.0;
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

    pub fn handle_quadmed_hoof_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const HOOF_ATTACK_RANGE: f32 = 1.0;
        const HOOF_ATTACK_ANGLE: f32 = 50.0;

        if attack_data.angle < HOOF_ATTACK_ANGLE
            && attack_data.dist_sqrd
                < (HOOF_ATTACK_RANGE + self.body.map_or(0.0, |b| b.max_radius())).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Primary);
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

    pub fn handle_quadlow_beam_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        enum ActionStateTimers {
            TimerQuadLowBeam = 0,
        }
        if attack_data.angle < 90.0
            && attack_data.dist_sqrd < (2.5 * attack_data.min_attack_dist).powi(2)
        {
            controller.inputs.move_dir = Vec2::zero();
            controller.push_basic_input(InputKind::Secondary);
        } else if attack_data.dist_sqrd < (7.0 * attack_data.min_attack_dist).powi(2)
            && attack_data.angle < 15.0
        {
            if agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] < 2.0 {
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .rotated_z(0.47 * PI)
                    .try_normalized()
                    .unwrap_or_else(Vec2::unit_y);
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] +=
                    read_data.dt.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] < 4.0
                && attack_data.angle < 15.0
            {
                controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                    .xy()
                    .rotated_z(-0.47 * PI)
                    .try_normalized()
                    .unwrap_or_else(Vec2::unit_y);
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] +=
                    read_data.dt.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] < 6.0
                && attack_data.angle < 15.0
            {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] +=
                    read_data.dt.0;
            } else {
                agent.action_state.timers[ActionStateTimers::TimerQuadLowBeam as usize] = 0.0;
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
        enum ActionStateTimers {
            TimerOrganAura = 0,
        }

        const ORGAN_AURA_DURATION: f32 = 34.75;
        if attack_data.dist_sqrd < (7.0 * attack_data.min_attack_dist).powi(2) {
            if agent.action_state.timers[ActionStateTimers::TimerOrganAura as usize]
                > ORGAN_AURA_DURATION
            {
                agent.action_state.timers[ActionStateTimers::TimerOrganAura as usize] = 0.0;
            } else if agent.action_state.timers[ActionStateTimers::TimerOrganAura as usize] < 1.0 {
                controller
                    .actions
                    .push(ControlAction::basic_input(InputKind::Primary));
                agent.action_state.timers[ActionStateTimers::TimerOrganAura as usize] +=
                    read_data.dt.0;
            } else {
                agent.action_state.timers[ActionStateTimers::TimerOrganAura as usize] +=
                    read_data.dt.0;
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
                .rotated_z(6.0 * read_data.dt.0)
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
        enum ActionStateFCounters {
            FCounterHealthThreshold = 0,
        }

        enum ActionStateICounters {
            ICounterNumFireballs = 0,
        }

        enum ActionStateConditions {
            ConditionCounterInit = 0,
        }

        const MINDFLAYER_ATTACK_DIST: f32 = 16.0;
        const MINION_SUMMON_THRESHOLD: f32 = 0.20;
        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already initialized
        if !agent.action_state.conditions[ActionStateConditions::ConditionCounterInit as usize] {
            agent.action_state.counters[ActionStateFCounters::FCounterHealthThreshold as usize] =
                1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.conditions[ActionStateConditions::ConditionCounterInit as usize] =
                true;
        }

        if agent.action_state.counters[ActionStateFCounters::FCounterHealthThreshold as usize]
            > health_fraction
        {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(2));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counters
                    [ActionStateFCounters::FCounterHealthThreshold as usize] -=
                    MINION_SUMMON_THRESHOLD;
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
            let num_fireballs = &mut agent.action_state.int_counters
                [ActionStateICounters::ICounterNumFireballs as usize];
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
        enum ActionStateTimers {
            TimerBirdLargeBreathe = 0,
        }

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
            && agent.action_state.timers[ActionStateTimers::TimerBirdLargeBreathe as usize] < 3.0
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
            agent.action_state.timers[ActionStateTimers::TimerBirdLargeBreathe as usize] +=
                read_data.dt.0;
        } else if agent.action_state.timers[ActionStateTimers::TimerBirdLargeBreathe as usize] < 6.0
            && attack_data.angle < 90.0
            && attack_data.in_min_range()
        {
            // Triple strike
            controller.push_basic_input(InputKind::Secondary);
            agent.action_state.timers[ActionStateTimers::TimerBirdLargeBreathe as usize] +=
                read_data.dt.0;
        } else {
            // Reset timer
            agent.action_state.timers[ActionStateTimers::TimerBirdLargeBreathe as usize] = 0.0;
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
        enum ActionStateTimers {
            TimerBirdLargeBasic = 0,
        }

        enum ActionStateConditions {
            ConditionBirdLargeBasic = 0, /* FIXME: Not sure what this represents. This name
                                          * should be reflective of the condition... */
        }

        const BIRD_ATTACK_RANGE: f32 = 4.0;
        const BIRD_CHARGE_DISTANCE: f32 = 15.0;
        let bird_attack_distance = self.body.map_or(0.0, |b| b.max_radius()) + BIRD_ATTACK_RANGE;
        // Increase action timer
        agent.action_state.timers[ActionStateTimers::TimerBirdLargeBasic as usize] +=
            read_data.dt.0;
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
        } else if agent.action_state.timers[ActionStateTimers::TimerBirdLargeBasic as usize] > 8.0 {
            // If action timer higher than 8, make bird summon tornadoes
            controller.push_basic_input(InputKind::Secondary);
            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                // Reset timer
                agent.action_state.timers[ActionStateTimers::TimerBirdLargeBasic as usize] = 0.0;
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
            agent.action_state.conditions
                [ActionStateConditions::ConditionBirdLargeBasic as usize] = true;
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
        enum ActionStateTimers {
            TimerArthropodRanged = 0,
        }

        agent.action_state.timers[ActionStateTimers::TimerArthropodRanged as usize] +=
            read_data.dt.0;
        if agent.action_state.timers[ActionStateTimers::TimerArthropodRanged as usize] > 6.0
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
                agent.action_state.timers[ActionStateTimers::TimerArthropodRanged as usize] = 0.0;
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
                    if agent.action_state.timers[ActionStateTimers::TimerArthropodRanged as usize]
                        > 5.0
                    {
                        agent.action_state.timers
                            [ActionStateTimers::TimerArthropodRanged as usize] = 0.0;
                    } else if agent.action_state.timers
                        [ActionStateTimers::TimerArthropodRanged as usize]
                        > 2.5
                    {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(1.75 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timers
                            [ActionStateTimers::TimerArthropodRanged as usize] += read_data.dt.0;
                    } else {
                        controller.inputs.move_dir = (tgt_data.pos.0 - self.pos.0)
                            .xy()
                            .rotated_z(0.25 * PI)
                            .try_normalized()
                            .unwrap_or_else(Vec2::zero)
                            * speed;
                        agent.action_state.timers
                            [ActionStateTimers::TimerArthropodRanged as usize] += read_data.dt.0;
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
        enum ActionStateTimers {
            TimersArthropodAmbush = 0,
        }

        agent.action_state.timers[ActionStateTimers::TimersArthropodAmbush as usize] +=
            read_data.dt.0;
        if agent.action_state.timers[ActionStateTimers::TimersArthropodAmbush as usize] > 12.0
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
                agent.action_state.timers[ActionStateTimers::TimersArthropodAmbush as usize] = 0.0;
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
        enum ActionStateTimers {
            TimersArthropodMelee = 0,
        }
        agent.action_state.timers[ActionStateTimers::TimersArthropodMelee as usize] +=
            read_data.dt.0;
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

        enum ActionStateFCounters {
            FCounterMinotaurAttack = 0,
        }

        enum ActionStateConditions {
            ConditionJustCrippledOrCleaved = 0,
        }

        let minotaur_attack_distance =
            self.body.map_or(0.0, |b| b.max_radius()) + MINOTAUR_ATTACK_RANGE;
        let health_fraction = self.health.map_or(1.0, |h| h.fraction());
        // Sets action counter at start of combat
        if agent.action_state.counters[ActionStateFCounters::FCounterMinotaurAttack as usize]
            < MINOTAUR_FRENZY_THRESHOLD
            && health_fraction > MINOTAUR_FRENZY_THRESHOLD
        {
            agent.action_state.counters[ActionStateFCounters::FCounterMinotaurAttack as usize] =
                MINOTAUR_FRENZY_THRESHOLD;
        }
        if health_fraction
            < agent.action_state.counters[ActionStateFCounters::FCounterMinotaurAttack as usize]
        {
            // Makes minotaur buff itself with frenzy
            controller.push_basic_input(InputKind::Ability(1));
            if matches!(self.char_state, CharacterState::SelfBuff(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counters
                    [ActionStateFCounters::FCounterMinotaurAttack as usize] = 0.0;
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
            if agent.action_state.conditions
                [ActionStateConditions::ConditionJustCrippledOrCleaved as usize]
                && !self.char_state.is_attack()
            {
                // Cripple target if not just used cripple
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.conditions
                    [ActionStateConditions::ConditionJustCrippledOrCleaved as usize] = false;
            } else if !self.char_state.is_attack() {
                // Cleave target if not just used cleave
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.conditions
                    [ActionStateConditions::ConditionJustCrippledOrCleaved as usize] = true;
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

        enum ActionStateFCounters {
            FCounterGlayGolemAttack = 0,
        }

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
            if agent.action_state.counters[ActionStateFCounters::FCounterGlayGolemAttack as usize]
                < 7.5
            {
                // If target is close, whack them
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.counters
                    [ActionStateFCounters::FCounterGlayGolemAttack as usize] += read_data.dt.0;
            } else {
                // If whacked for too long, nuke them
                controller.push_basic_input(InputKind::Ability(1));
                if matches!(self.char_state, CharacterState::BasicRanged(c) if matches!(c.stage_section, StageSection::Recover))
                {
                    agent.action_state.counters
                        [ActionStateFCounters::FCounterGlayGolemAttack as usize] = 0.0;
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

        enum ActionStateConditions {
            ConditionCounterInitialized = 0,
        }

        enum ActionStateFCounters {
            FCounterMinionSummonThreshold = 0,
        }

        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already initialized
        if !agent.action_state.conditions
            [ActionStateConditions::ConditionCounterInitialized as usize]
        {
            agent.action_state.counters
                [ActionStateFCounters::FCounterMinionSummonThreshold as usize] =
                1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.conditions
                [ActionStateConditions::ConditionCounterInitialized as usize] = true;
        }

        if agent.action_state.counters[ActionStateFCounters::FCounterMinionSummonThreshold as usize]
            > health_fraction
        {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(1));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counters
                    [ActionStateFCounters::FCounterMinionSummonThreshold as usize] -=
                    MINION_SUMMON_THRESHOLD;
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

        enum ActionStateFCounters {
            FCounterYetiAttack = 0,
        }

        agent.action_state.counters[ActionStateFCounters::FCounterYetiAttack as usize] +=
            read_data.dt.0;

        if attack_data.dist_sqrd < ICE_BREATH_RANGE.powi(2) {
            if matches!(self.char_state, CharacterState::BasicBeam(c) if c.timer < Duration::from_secs(2))
            {
                // Keep using ice breath for 2 second
                controller.push_basic_input(InputKind::Ability(0));
            } else if agent.action_state.counters[ActionStateFCounters::FCounterYetiAttack as usize]
                > ICE_BREATH_TIMER
            {
                // Use ice breath if timer has gone for long enough
                controller.push_basic_input(InputKind::Ability(0));

                if matches!(self.char_state, CharacterState::BasicBeam(_)) {
                    // Resets action counter when using beam
                    agent.action_state.counters
                        [ActionStateFCounters::FCounterYetiAttack as usize] = 0.0;
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

    pub fn handle_roshwalr_attack(
        &self,
        agent: &mut Agent,
        controller: &mut Controller,
        attack_data: &AttackData,
        tgt_data: &TargetData,
        read_data: &ReadData,
    ) {
        const SLOW_CHARGE_RANGE: f32 = 12.5;
        const SHOCKWAVE_RANGE: f32 = 12.5;
        const SHOCKWAVE_TIMER: f32 = 15.0;
        const MELEE_RANGE: f32 = 4.0;

        enum ActionStateFCounters {
            FCounterRoshwalrAttack = 0,
        }

        agent.action_state.counters[ActionStateFCounters::FCounterRoshwalrAttack as usize] +=
            read_data.dt.0;
        if matches!(self.char_state, CharacterState::DashMelee(c) if !matches!(c.stage_section, StageSection::Recover))
        {
            // If already charging, keep charging if not in recover
            controller.push_basic_input(InputKind::Ability(0));
        } else if attack_data.dist_sqrd < SHOCKWAVE_RANGE.powi(2) && attack_data.angle < 270.0 {
            if agent.action_state.counters[ActionStateFCounters::FCounterRoshwalrAttack as usize]
                > SHOCKWAVE_TIMER
            {
                // Use shockwave if timer has gone for long enough
                controller.push_basic_input(InputKind::Ability(0));

                if matches!(self.char_state, CharacterState::Shockwave(_)) {
                    // Resets action counter when using shockwave
                    agent.action_state.counters
                        [ActionStateFCounters::FCounterRoshwalrAttack as usize] = 0.0;
                }
            } else if attack_data.dist_sqrd < MELEE_RANGE.powi(2) && attack_data.angle < 135.0 {
                // Basic attack if in melee range
                controller.push_basic_input(InputKind::Primary);
            }
        } else if attack_data.dist_sqrd > SLOW_CHARGE_RANGE.powi(2) {
            // Use slow charge if outside the range
            controller.push_basic_input(InputKind::Secondary);
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

        enum ActionStateConditions {
            ConditionHasSummonedVines = 0,
        }

        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        let line_of_sight_with_target = || {
            entities_have_line_of_sight(self.pos, self.body, tgt_data.pos, tgt_data.body, read_data)
        };

        if health_fraction < VINE_CREATION_THRESHOLD
            && !agent.action_state.conditions
                [ActionStateConditions::ConditionHasSummonedVines as usize]
        {
            // Summon vines when reach threshold of health
            controller.push_basic_input(InputKind::Ability(0));

            if matches!(self.char_state, CharacterState::SpriteSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.conditions
                    [ActionStateConditions::ConditionHasSummonedVines as usize] = true;
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

        enum ActionStateConditions {
            ConditionCounterInitialized = 0,
        }

        enum ActionStateFCounters {
            FCounterHealthThreshold = 0,
        }

        let health_fraction = self.health.map_or(0.5, |h| h.fraction());
        // Sets counter at start of combat, using `condition` to keep track of whether
        // it was already intitialized
        if !agent.action_state.conditions
            [ActionStateConditions::ConditionCounterInitialized as usize]
        {
            agent.action_state.counters[ActionStateFCounters::FCounterHealthThreshold as usize] =
                1.0 - MINION_SUMMON_THRESHOLD;
            agent.action_state.conditions
                [ActionStateConditions::ConditionCounterInitialized as usize] = true;
        }

        if agent.action_state.counters[ActionStateFCounters::FCounterHealthThreshold as usize]
            > health_fraction
        {
            // Summon minions at particular thresholds of health
            controller.push_basic_input(InputKind::Ability(1));

            if matches!(self.char_state, CharacterState::BasicSummon(c) if matches!(c.stage_section, StageSection::Recover))
            {
                agent.action_state.counters
                    [ActionStateFCounters::FCounterHealthThreshold as usize] -=
                    MINION_SUMMON_THRESHOLD;
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
                // Use steam beam if target is far enough away, self is not buffed, and have
                // sufficient energy
                controller.push_basic_input(InputKind::Ability(0));
            } else {
                // If low on energy, use primary to attempt to regen energy
                // Or if at desired energy level but not able/willing to ward, just attack
                controller.push_basic_input(InputKind::Primary);
            }
        } else if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            if self.body.map_or(false, |b| b.is_humanoid())
                && self.energy.current() > CharacterAbility::default_roll().energy_cost()
                && !matches!(self.char_state, CharacterState::BasicAura(c) if !matches!(c.stage_section, StageSection::Recover))
            {
                // Else use steam beam
                controller.push_basic_input(InputKind::Ability(0));
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
        enum ActionStateTimers {
            TimerDagon = 0,
        }
        if agent.action_state.timers[ActionStateTimers::TimerDagon as usize] > 2.5 {
            agent.action_state.timers[ActionStateTimers::TimerDagon as usize] = 0.0;
        }
        // if target gets very close, shoot dagon bombs and lay out sea urchins
        if attack_data.dist_sqrd < (2.0 * attack_data.min_attack_dist).powi(2) {
            if agent.action_state.timers[ActionStateTimers::TimerDagon as usize] > 1.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Secondary);
                agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
            }
        // if target in close range use steambeam and shoot dagon bombs
        } else if attack_data.dist_sqrd < (3.0 * attack_data.min_attack_dist).powi(2) {
            controller.inputs.move_dir = Vec2::zero();
            if agent.action_state.timers[ActionStateTimers::TimerDagon as usize] > 2.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Ability(1));
            }
        } else if attack_data.dist_sqrd > (4.0 * attack_data.min_attack_dist).powi(2) {
            // if enemy is far, heal
            controller.push_basic_input(InputKind::Ability(2));
            agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
        } else if entities_have_line_of_sight(
            self.pos,
            self.body,
            tgt_data.pos,
            tgt_data.body,
            read_data,
        ) {
            // if enemy in mid range shoot dagon bombs and steamwave
            if agent.action_state.timers[ActionStateTimers::TimerDagon as usize] > 1.0 {
                controller.push_basic_input(InputKind::Primary);
                agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
            } else {
                controller.push_basic_input(InputKind::Ability(0));
                agent.action_state.timers[ActionStateTimers::TimerDagon as usize] += read_data.dt.0;
            }
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

        enum ActionStateFCounters {
            FCounterHealthThreshold = 0,
        }

        enum ActionStateConditions {
            ConditionHasScreamed = 0,
        }

        if !agent.action_state.initialized {
            agent.action_state.counters[ActionStateFCounters::FCounterHealthThreshold as usize] =
                self.health.map_or(0.0, |h| h.maximum());
            agent.action_state.initialized = true;
        }

        if !agent.action_state.conditions[ActionStateConditions::ConditionHasScreamed as usize] {
            // If mandragora is still "sleeping" and hasn't screamed yet, do nothing until
            // target in range or until it's taken damage
            if self.health.map_or(false, |h| {
                h.current()
                    < agent.action_state.counters
                        [ActionStateFCounters::FCounterHealthThreshold as usize]
            }) || attack_data.dist_sqrd < SCREAM_RANGE.powi(2)
            {
                agent.action_state.conditions
                    [ActionStateConditions::ConditionHasScreamed as usize] = true;
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
                agent.action_state.conditions
                    [ActionStateConditions::ConditionHasScreamed as usize] = false;
                agent.action_state.counters
                    [ActionStateFCounters::FCounterHealthThreshold as usize] =
                    self.health.map_or(0.0, |h| h.maximum());
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

        enum ActionStateTimers {
            TimerSpinWait = 0,
            TimerShockwaveWait,
        }

        // After spinning, reset timer
        if matches!(self.char_state, CharacterState::SpinMelee(s) if s.stage_section == StageSection::Recover)
        {
            agent.action_state.timers[ActionStateTimers::TimerSpinWait as usize] = 0.0;
        }

        if attack_data.in_min_range() {
            // If in minimum range
            if agent.action_state.timers[ActionStateTimers::TimerSpinWait as usize] > SPIN_WAIT_TIME
            {
                // If it's been too long since able to hit target, spin
                controller.push_basic_input(InputKind::Secondary);
            } else if attack_data.angle < 30.0 {
                // Else if in angle to strike, strike
                controller.push_basic_input(InputKind::Primary);
            } else {
                // Else increment spin timer
                agent.action_state.timers[ActionStateTimers::TimerSpinWait as usize] +=
                    read_data.dt.0;
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
                if agent.action_state.timers[ActionStateTimers::TimerSpinWait as usize]
                    > SHOCKWAVE_WAIT_TIME
                {
                    controller.push_basic_input(InputKind::Ability(0));
                }
                if matches!(self.char_state, CharacterState::Shockwave(_)) {
                    agent.action_state.timers[ActionStateTimers::TimerShockwaveWait as usize] = 0.0;
                } else {
                    agent.action_state.timers[ActionStateTimers::TimerShockwaveWait as usize] +=
                        read_data.dt.0;
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

        enum ActionStateTimers {
            TimerSummonTotem = 0,
            TimerShockwave,
        }
        // Handle timers
        agent.action_state.timers[ActionStateTimers::TimerSummonTotem as usize] += read_data.dt.0;
        match self.char_state {
            CharacterState::BasicSummon(_) => {
                agent.action_state.timers[ActionStateTimers::TimerSummonTotem as usize] = 0.0
            },
            CharacterState::Shockwave(_) | CharacterState::BasicRanged(_) => {
                agent.action_state.counters[ActionStateTimers::TimerShockwave as usize] = 0.0
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
        } else if agent.action_state.timers[ActionStateTimers::TimerSummonTotem as usize]
            > TOTEM_TIMER
        {
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
                agent.action_state.timers[ActionStateTimers::TimerSummonTotem as usize] = 0.0;
            } else {
                controller.push_basic_input(InputKind::Ability(input));
            }
        } else if agent.action_state.counters[ActionStateTimers::TimerShockwave as usize]
            > HEAVY_ATTACK_WAIT_TIME
        {
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
                agent.action_state.counters[ActionStateTimers::TimerShockwave as usize] +=
                    read_data.dt.0;
            } else {
                // If not in angle, charge heavy attack faster
                agent.action_state.counters[ActionStateTimers::TimerShockwave as usize] +=
                    read_data.dt.0 * 5.0;
            }
        } else {
            // If not in range, charge heavy attack faster
            agent.action_state.counters[ActionStateTimers::TimerShockwave as usize] +=
                read_data.dt.0 * 3.3;
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
