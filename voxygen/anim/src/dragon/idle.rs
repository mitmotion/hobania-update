use super::{
    super::{vek::*, Animation},
    DragonSkeleton, SkeletonAttr,
};
use std::{f32::consts::PI, ops::Mul};

pub struct IdleAnimation;

impl Animation for IdleAnimation {
    type Dependency<'a> = f32;
    type Skeleton = DragonSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"dragon_idle\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "dragon_idle")]
    fn update_skeleton_inner(
        skeleton: &Self::Skeleton,
        global_time: Self::Dependency<'_>,
        anim_time: f32,
        _rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        let ultra_slow = (anim_time * 1.0).sin();
        let slow = (anim_time * 2.5).sin();
        let slowalt = (anim_time * 2.5 + PI / 2.0).sin();

        let dragon_look = Vec2::new(
            (global_time / 2.0 + anim_time / 8.0)
                .floor()
                .mul(7331.0)
                .sin()
                * 0.5,
            (global_time / 2.0 + anim_time / 8.0)
                .floor()
                .mul(1337.0)
                .sin()
                * 0.25,
        );

        next.head_upper.scale = Vec3::one() * 1.05;
        next.head_lower.scale = Vec3::one() * 1.05;
        next.jaw.scale = Vec3::one() * 1.05;
        next.tail_front.scale = Vec3::one() * 0.98;
        next.tail_rear.scale = Vec3::one() * 0.98;

        next.head_upper.position =
            Vec3::new(0.0, s_a.head_upper.0, s_a.head_upper.1 + ultra_slow * 0.20);
        next.head_upper.orientation = Quaternion::rotation_z(0.8 * dragon_look.x)
            * Quaternion::rotation_x(0.8 * dragon_look.y);

        next.head_lower.position =
            Vec3::new(0.0, s_a.head_lower.0, s_a.head_lower.1 + ultra_slow * 0.20);
        next.head_lower.orientation = Quaternion::rotation_z(0.8 * dragon_look.x)
            * Quaternion::rotation_x(-0.2 + 0.8 * dragon_look.y);

        next.jaw.position = Vec3::new(0.0, s_a.jaw.0, s_a.jaw.1);
        next.jaw.orientation = Quaternion::rotation_x(slow * 0.04);

        next.chest_front.position = Vec3::new(0.0, s_a.chest_front.0, s_a.chest_front.1);
        next.chest_front.orientation = Quaternion::rotation_y(0.0);

        next.chest_rear.position = Vec3::new(0.0, s_a.chest_rear.0, s_a.chest_rear.1);
        next.chest_rear.orientation = Quaternion::rotation_y(0.0);

        next.tail_front.position = Vec3::new(0.0, s_a.tail_front.0, s_a.tail_front.1);
        next.tail_front.orientation =
            Quaternion::rotation_z(slowalt * 0.10) * Quaternion::rotation_x(0.1);

        next.tail_rear.position = Vec3::new(0.0, s_a.tail_rear.0, s_a.tail_rear.1);
        next.tail_rear.orientation =
            Quaternion::rotation_z(slowalt * 0.12) * Quaternion::rotation_x(0.05);

        next.foot_fl.position = Vec3::new(-s_a.feet_f.0, s_a.feet_f.1, s_a.feet_f.2);

        next.foot_fr.position = Vec3::new(s_a.feet_f.0, s_a.feet_f.1, s_a.feet_f.2);

        next.foot_bl.position = Vec3::new(-s_a.feet_b.0, s_a.feet_b.1, s_a.feet_b.2);

        next.foot_br.position = Vec3::new(s_a.feet_b.0, s_a.feet_b.1, s_a.feet_b.2);

        next.wing_in_l.position = Vec3::new(-s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);
        next.wing_in_l.orientation = Quaternion::rotation_y(0.8 + slow * 0.02);

        next.wing_in_r.position = Vec3::new(s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);
        next.wing_in_r.orientation = Quaternion::rotation_y(-0.8 - slow * 0.02);

        next.wing_out_l.position = Vec3::new(-s_a.wing_out.0, s_a.wing_out.1, s_a.wing_out.2);
        next.wing_out_l.orientation = Quaternion::rotation_y(-2.0 + slow * 0.02);

        next.wing_out_r.position = Vec3::new(s_a.wing_out.0, s_a.wing_out.1, s_a.wing_out.2);
        next.wing_out_r.orientation = Quaternion::rotation_y(2.0 - slow * 0.02);

        next
    }
}
