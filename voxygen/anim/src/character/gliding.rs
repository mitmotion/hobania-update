use super::{
    super::{vek::*, Animation},
    CharacterSkeleton, SkeletonAttr,
};
use common::comp::item::ToolKind;
use std::{f32::consts::PI, ops::Mul};

pub struct GlidingAnimation;

type GlidingAnimationDependency = (
    Option<ToolKind>,
    Option<ToolKind>,
    Vec3<f32>,
    Quaternion<f32>,
    Quaternion<f32>,
    Quaternion<f32>,
    f32,
    f32,
);

impl Animation for GlidingAnimation {
    type Dependency = GlidingAnimationDependency;
    type Skeleton = CharacterSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"character_gliding\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "character_gliding")]

    fn update_skeleton_inner(
        skeleton: &Self::Skeleton,
        (
            _active_tool_kind,
            _second_tool_kind,
            velocity,
            orientation,
            last_ori,
            glider_orientation,
            global_time,
            acc_vel,
        ): Self::Dependency,
        anim_time: f32,
        _rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        let speednorm = velocity.xy().magnitude().min(30.0) / 30.0;
        let speedxyznorm = velocity.magnitude().min(30.0) / 30.0;

        let slow = (acc_vel * 0.5).sin();
        let slowa = (acc_vel * 0.5 + PI / 2.0).sin();

        let head_look = Vec2::new(
            ((global_time + anim_time) as f32 / 4.0)
                .floor()
                .mul(7331.0)
                .sin()
                * 0.5,
            ((global_time + anim_time) as f32 / 4.0)
                .floor()
                .mul(1337.0)
                .sin()
                * 0.25,
        );

        let tilt = {
            let ori: Vec2<f32> = Vec2::from(orientation * Vec3::unit_y());
            let last_ori: Vec2<f32> = Vec2::from(last_ori * Vec3::unit_y());
            if ::vek::Vec2::new(ori, last_ori)
                .map(|o| o.magnitude_squared())
                .map(|m| m > 0.001 && m.is_finite())
                .reduce_and()
                && ori.angle_between(last_ori).is_finite()
            {
                ori.angle_between(last_ori).min(0.2)
                    * last_ori.determine_side(Vec2::zero(), ori).signum()
                    * 1.3
            } else {
                0.0
            }
        };
        let torso_ori = Quaternion::slerp(
            Quaternion::rotation_x(-0.06 * speednorm.max(5.0) + slow * 0.04)
                * Quaternion::rotation_y(speednorm * tilt * 2.0 / speednorm.max(0.2))
                * Quaternion::rotation_z(speednorm * tilt * 3.0 * speednorm),
            orientation.inverse() * glider_orientation,
            0.3,
        );
        let chest_ori = Quaternion::rotation_z(slowa * 0.01);
        let chest_global_inv = (orientation * torso_ori * chest_ori).inverse();
        let glider_pos = Vec3::new(0.0, -5.0, 18.0);
        let glider_ori = chest_global_inv * glider_orientation;
        let center_of_rot = glider_pos * 0.8;

        next.head.orientation = Quaternion::rotation_x(head_look.y + speednorm.min(28.0) * 0.03)
            * Quaternion::rotation_z(head_look.x);

        next.torso.position =
            (center_of_rot - orientation.inverse() * glider_orientation * center_of_rot) / 11.0
                * s_a.scaler;
        next.torso.orientation = torso_ori;

        next.chest.orientation = chest_ori;

        next.belt.orientation = Quaternion::rotation_z(slowa * 0.1);

        next.shorts.position = Vec3::new(s_a.shorts.0, 0.0, s_a.shorts.1);
        next.shorts.orientation = chest_ori.inverse() * Quaternion::rotation_z(slowa * 0.12);

        next.hand_l.position = glider_pos
            + glider_ori * Vec3::new(-s_a.hand.0 + -2.0, s_a.hand.1 + 8.0, s_a.hand.2 + -5.0);
        next.hand_l.orientation = Quaternion::rotation_x(3.35) * Quaternion::rotation_y(0.2);

        next.hand_r.position = glider_pos
            + glider_ori * Vec3::new(s_a.hand.0 + 2.0, s_a.hand.1 + 8.0, s_a.hand.2 + -5.0);
        next.hand_r.orientation = Quaternion::rotation_x(3.35) * Quaternion::rotation_y(-0.2);

        next.foot_l.position = Vec3::new(-s_a.foot.0, s_a.foot.1, s_a.foot.2);
        next.foot_l.orientation =
            Quaternion::rotation_x(-0.8 * speedxyznorm + slow * -0.5 * speedxyznorm);

        next.foot_r.position = Vec3::new(s_a.foot.0, s_a.foot.1, s_a.foot.2);
        next.foot_r.orientation =
            Quaternion::rotation_x(-0.8 * speedxyznorm + slow * 0.5 * speedxyznorm);

        next.glider.position = glider_pos;
        next.glider.orientation = glider_ori;
        next.glider.scale = Vec3::one();

        next
    }
}
