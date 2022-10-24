use super::{
    super::{vek::*, Animation},
    CharacterSkeleton, SkeletonAttr,
};
use common::{
    comp::item::{AbilitySpec, Hands, ToolKind},
    states::utils::AbilityInfo,
};
use std::{f32::consts::PI, ops::Mul};

pub struct MusicAnimation;

type MusicAnimationDependency<'a> = (
    (Option<Hands>, Option<Hands>),
    (Option<AbilityInfo>, Option<&'a AbilitySpec>, f32),
    Vec3<f32>,
);
impl Animation for MusicAnimation {
    type Dependency<'a> = MusicAnimationDependency<'a>;
    type Skeleton = CharacterSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"character_music\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "character_music")]
    fn update_skeleton_inner<'a>(
        skeleton: &Self::Skeleton,
        (_hands, (ability_info, active_tool_spec, global_time), rel_vel): Self::Dependency<'a>,
        anim_time: f32,
        rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        *rate = 2.0;

        let lab: f32 = 1.0;
        let short = ((5.0 / (3.0 + 2.0 * ((anim_time * lab * 6.0).sin()).powi(2))).sqrt())
            * ((anim_time * lab * 6.0).sin());
        let noisea = (anim_time * 11.0 + PI / 6.0).sin();
        let noiseb = (anim_time * 19.0 + PI / 4.0).sin();

        let shorte = (anim_time * lab * 6.0).sin();

        let shortealt = (anim_time * lab * 6.0 + PI / 2.0).sin();

        let foot = ((0.1 / (1.0 + (4.0) * ((anim_time * lab * 8.0).sin()).powi(2))).sqrt())
            * ((anim_time * lab * 8.0).sin());

        // common animations
        next.chest.position = Vec3::new(0.0, s_a.chest.0, s_a.chest.1 + shortealt * 1.5);
        next.chest.orientation = Quaternion::rotation_z(short * 0.35)
            * Quaternion::rotation_y(shorte * 0.08)
            * Quaternion::rotation_x(foot * 0.07);

        next.belt.position = Vec3::new(0.0, s_a.belt.0, s_a.belt.1);
        next.belt.orientation = Quaternion::rotation_z(shorte * 0.25);

        next.back.position = Vec3::new(0.0, s_a.back.0, s_a.back.1);
        next.back.orientation =
            Quaternion::rotation_x(-0.25 + shorte * 0.1 + noisea * 0.1 + noiseb * 0.1);

        next.shorts.position = Vec3::new(0.0, s_a.shorts.0, s_a.shorts.1);
        next.shorts.orientation = Quaternion::rotation_z(foot * 0.35);

        // don't override run animation when instruments are played moving
        if rel_vel.magnitude() < 0.1 {
            next.foot_l.position = Vec3::new(
                -s_a.foot.0 + foot * 0.8,
                1.5 + -s_a.foot.1 + foot * -4.0,
                s_a.foot.2,
            );
            next.foot_l.orientation =
                Quaternion::rotation_x(foot * -0.3) * Quaternion::rotation_z(short * -0.15);

            next.foot_r.position = Vec3::new(
                s_a.foot.0 + foot * 0.8,
                1.5 + -s_a.foot.1 + foot * 4.0,
                s_a.foot.2,
            );
            next.foot_r.orientation =
                Quaternion::rotation_x(foot * 0.3) * Quaternion::rotation_z(short * 0.15);
        };
        next.shoulder_l.position = Vec3::new(-s_a.shoulder.0, s_a.shoulder.1, s_a.shoulder.2);
        next.shoulder_l.orientation = Quaternion::rotation_x(shorte * 0.15);

        next.shoulder_r.position = Vec3::new(s_a.shoulder.0, s_a.shoulder.1, s_a.shoulder.2);
        next.shoulder_r.orientation = Quaternion::rotation_x(shorte * -0.15);

        next.lantern.orientation =
            Quaternion::rotation_x(shorte * 0.7 + 0.4) * Quaternion::rotation_y(shorte * 0.4);

        next.torso.position = Vec3::new(0.0, -3.3, 0.0);
        next.torso.orientation = Quaternion::rotation_z(short * -0.2);

        let head_look = Vec2::new(
            (global_time + anim_time / 6.0).floor().mul(7331.0).sin() * 0.3,
            (global_time + anim_time / 6.0).floor().mul(1337.0).sin() * 0.15,
        );

        // TODO: create Instrument SubToolKinds to distinguish instruments
        match ability_info.and_then(|a| a.tool) {
            Some(ToolKind::Instrument) => {
                if let Some(AbilitySpec::Custom(spec)) = active_tool_spec {
                    // instrument specific head_bop
                    let head_bop = match spec.as_str() {
                        "Flute" | "Melodica" => 0.2,
                        "Guitar" | "Lute" | "Sitar" => 0.5,
                        "Harp" | "Kalimba" => 0.3,
                        _ => 1.0,
                    };
                    next.head.position = Vec3::new(0.0, s_a.head.0, s_a.head.1);
                    next.head.orientation = Quaternion::rotation_z((short * head_bop) * -0.6)
                        * Quaternion::rotation_x(
                            0.2 + head_look.y.max(0.0) + (shorte * head_bop).abs() * -0.2,
                        );
                    // instrument specific hand and instrument animations
                    match spec.as_str() {
                        "Bass" => {
                            next.hand_l.position = Vec3::new(
                                3.5 - s_a.hand.0,
                                7.0 + s_a.hand.1 + shortealt * -3.0,
                                8.0 + s_a.hand.2 + shortealt * -0.75,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(2.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.5);

                            next.hand_r.position = Vec3::new(
                                -2.0 + s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * 6.0,
                                4.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.4);

                            next.main.position = Vec3::new(-4.0, 6.0, 16.0);
                            next.main.orientation = Quaternion::rotation_x(0.1)
                                * Quaternion::rotation_y(3.0)
                                * Quaternion::rotation_z(PI / -3.0);
                        },
                        "Flute" => {
                            next.hand_l.position = Vec3::new(
                                4.0 - s_a.hand.0,
                                6.0 + s_a.hand.1 + shortealt * -0.5,
                                4.0 + s_a.hand.2 + shortealt * -0.75,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(2.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.9);

                            next.hand_r.position = Vec3::new(
                                -4.5 + s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * 2.0,
                                2.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.6);

                            next.main.position = Vec3::new(-2.5, 10.0, -11.0);
                            next.main.orientation = Quaternion::rotation_x(3.5)
                                * Quaternion::rotation_y(PI)
                                * Quaternion::rotation_z(0.05);
                        },
                        "Guitar" => {
                            next.hand_l.position = Vec3::new(
                                1.0 - s_a.hand.0,
                                6.0 + s_a.hand.1 + shortealt * -1.0,
                                2.0 + s_a.hand.2 + shortealt * -1.5,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.8 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6)
                                * Quaternion::rotation_z(0.8);

                            next.hand_r.position = Vec3::new(
                                -2.0 + s_a.hand.0 - shortealt * 1.25,
                                6.0 + s_a.hand.1 + shortealt * 2.0,
                                3.0 + s_a.hand.2 + shortealt * 0.25,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.0 + foot * -0.15)
                                * Quaternion::rotation_y(0.6);

                            next.main.position = Vec3::new(-14.0, 6.0, 5.0);
                            next.main.orientation = Quaternion::rotation_x(0.1)
                                * Quaternion::rotation_y(2.0)
                                * Quaternion::rotation_z(PI / -3.0);
                        },
                        "Harp" => {
                            next.hand_l.position = Vec3::new(
                                3.0 - s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * -0.1,
                                1.0 + s_a.hand.2 + shortealt * -0.2,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6);

                            next.hand_r.position = Vec3::new(
                                -4.0 + s_a.hand.0 + shortealt * 2.0,
                                5.0 + s_a.hand.1 - shortealt * 3.0,
                                2.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.9);

                            next.main.position = Vec3::new(8.0, 14.0, -6.0);
                            next.main.orientation = Quaternion::rotation_x(0.2)
                                * Quaternion::rotation_y(-0.75)
                                * Quaternion::rotation_z(0.20);
                        },

                        "Kalimba" => {
                            next.hand_l.position = Vec3::new(
                                3.0 - s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * -0.1,
                                1.0 + s_a.hand.2 + shortealt * -0.2,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6);

                            next.hand_r.position = Vec3::new(
                                -2.0 + s_a.hand.0 + shortealt * 2.0,
                                5.0 + s_a.hand.1 - shortealt * 3.0,
                                2.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.9);

                            next.main.position = Vec3::new(8.0, 12.0, -8.0);
                            next.main.orientation = Quaternion::rotation_x(0.2)
                                * Quaternion::rotation_y(-0.75)
                                * Quaternion::rotation_z(PI - 0.2);
                        },
                        "Lute" => {
                            next.hand_l.position = Vec3::new(
                                2.0 - s_a.hand.0,
                                5.0 + s_a.hand.1 + shortealt * -1.0,
                                2.0 + s_a.hand.2 + shortealt * -1.5,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.8 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6)
                                * Quaternion::rotation_z(0.8);

                            next.hand_r.position = Vec3::new(
                                -1.0 + s_a.hand.0 - shortealt * 1.25,
                                6.0 + s_a.hand.1 + shortealt * 2.0,
                                2.0 + s_a.hand.2 + shortealt * 0.25,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.0 + foot * -0.15)
                                * Quaternion::rotation_y(0.6);

                            next.main.position = Vec3::new(-14.0, 6.0, 4.0);
                            next.main.orientation = Quaternion::rotation_x(0.1)
                                * Quaternion::rotation_y(2.0)
                                * Quaternion::rotation_z(PI / -3.0);
                        },
                        "Melodica" => {
                            next.hand_l.position = Vec3::new(
                                4.0 - s_a.hand.0,
                                6.0 + s_a.hand.1 + shortealt * -0.5,
                                4.0 + s_a.hand.2 + shortealt * -0.75,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(2.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.9);

                            next.hand_r.position = Vec3::new(
                                -3.5 + s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * 2.0,
                                2.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.6);

                            next.main.position = Vec3::new(-1.0, 2.0, 16.0);
                            next.main.orientation = Quaternion::rotation_x(0.3)
                                * Quaternion::rotation_y(PI)
                                * Quaternion::rotation_z(PI / -2.0);
                        },
                        "Perc" => {
                            next.hand_l.position = Vec3::new(
                                3.0 - s_a.hand.0,
                                4.0 + s_a.hand.1 + shortealt * -0.1,
                                1.0 + s_a.hand.2 + shortealt * -0.2,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.4 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6);

                            next.hand_r.position = Vec3::new(
                                -4.0 + s_a.hand.0 + shortealt * 2.0,
                                5.0 + s_a.hand.1 - shortealt * 3.0,
                                2.0 + s_a.hand.2 + shortealt * 0.75,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.4 + foot * -0.15)
                                * Quaternion::rotation_y(0.9);

                            next.main.position = Vec3::new(8.0, 14.0, -6.0);
                            next.main.orientation = Quaternion::rotation_x(0.2)
                                * Quaternion::rotation_y(-0.75)
                                * Quaternion::rotation_z(0.20);
                        },
                        "Sitar" => {
                            next.hand_l.position = Vec3::new(
                                2.0 - s_a.hand.0,
                                6.0 + s_a.hand.1 + shortealt * -1.0,
                                1.0 + s_a.hand.2 + shortealt * -1.5,
                            );
                            next.hand_l.orientation = Quaternion::rotation_x(1.8 + foot * 0.15)
                                * Quaternion::rotation_y(-0.6)
                                * Quaternion::rotation_z(0.8);

                            next.hand_r.position = Vec3::new(
                                -2.0 + s_a.hand.0 - shortealt * 1.25,
                                6.0 + s_a.hand.1 + shortealt * 2.0,
                                2.0 + s_a.hand.2 + shortealt * 0.25,
                            );
                            next.hand_r.orientation = Quaternion::rotation_x(1.0 + foot * -0.15)
                                * Quaternion::rotation_y(0.6);

                            next.main.position = Vec3::new(-14.0, 6.0, 4.0);
                            next.main.orientation = Quaternion::rotation_x(0.1)
                                * Quaternion::rotation_y(2.0)
                                * Quaternion::rotation_z(PI / -3.0);
                        },
                        _ => {},
                    }
                }
            },
            _ => {},
        }

        if skeleton.holding_lantern {
            next.hand_r.position = Vec3::new(s_a.hand.0, s_a.hand.1 + 5.0, s_a.hand.2 + 12.0);
            next.hand_r.orientation = Quaternion::rotation_x(2.25) * Quaternion::rotation_z(0.9);

            next.lantern.position = Vec3::new(-0.5, -0.5, 5.5);
            next.lantern.orientation = next.hand_r.orientation.inverse();
        }

        next
    }
}
