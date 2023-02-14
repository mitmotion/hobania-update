use super::{
    super::{vek::*, Animation},
    BirdMediumSkeleton, SkeletonAttr,
};
use std::{f32::consts::PI, ops::Mul};
pub struct IdleAnimation;

impl Animation for IdleAnimation {
    type Dependency<'a> = f32;
    type Skeleton = BirdMediumSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"bird_medium_idle\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "bird_medium_idle")]
    fn update_skeleton_inner<'a>(
        skeleton: &Self::Skeleton,
        global_time: Self::Dependency<'_>,
        anim_time: f32,
        _rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        let duck_head_look = Vec2::new(
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
        let wave_slow_cos = (anim_time * 4.5).cos();

        next.head.scale = Vec3::one() * 0.99;
        next.leg_l.scale = Vec3::one();
        next.leg_r.scale = Vec3::one();
        next.chest.scale = Vec3::one() * s_a.scaler * 0.99;
        next.tail.scale = Vec3::one() * 1.01;

        next.leg_l.position = Vec3::new(-s_a.leg.0, s_a.leg.1, s_a.leg.2);
        next.leg_l.orientation = Quaternion::rotation_x(0.0);
        next.leg_r.position = Vec3::new(s_a.leg.0, s_a.leg.1, s_a.leg.2);
        next.leg_r.orientation = Quaternion::rotation_x(0.0);

        next.chest.position =
            Vec3::new(0.0, s_a.chest.0, s_a.chest.1 + wave_slow_cos * 0.06 + 1.5) * s_a.scaler;

        next.head.position = Vec3::new(0.0, s_a.head.0, s_a.head.1);
        next.head.orientation = Quaternion::rotation_z(duck_head_look.x)
            * Quaternion::rotation_x(-duck_head_look.y.abs() + wave_slow_cos * 0.01);

        next.tail.position = Vec3::new(0.0, s_a.tail.0, s_a.tail.1);
        next.tail.orientation = Quaternion::rotation_x(0.6);

        next.wing_in_l.position = Vec3::new(-s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);
        next.wing_in_r.position = Vec3::new(s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);

        next.wing_in_l.orientation = Quaternion::rotation_x(-PI / 2.0)
            * Quaternion::rotation_y(-PI / 2.5 + wave_slow_cos * 0.06)
            * Quaternion::rotation_z(-PI / 4.0);
        next.wing_in_r.orientation = Quaternion::rotation_x(-PI / 2.0)
            * Quaternion::rotation_y(PI / 2.5 - wave_slow_cos * 0.06)
            * Quaternion::rotation_z(PI / 4.0);

        next.wing_out_l.position = Vec3::new(-s_a.wing_out.0 / 2.5, s_a.wing_out.1, s_a.wing_out.2);
        next.wing_out_r.position = Vec3::new(s_a.wing_out.0 / 2.5, s_a.wing_out.1, s_a.wing_out.2);
        next.wing_out_l.orientation = Quaternion::rotation_y(0.1) * Quaternion::rotation_z(0.2);
        next.wing_out_r.orientation = Quaternion::rotation_y(-0.1) * Quaternion::rotation_z(-0.2);

        next
    }
}
