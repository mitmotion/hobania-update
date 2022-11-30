use super::{super::Animation, SkeletonAttr, TheropodSkeleton};
//use std::{f32::consts::PI, ops::Mul};
use super::super::vek::*;
use std::ops::Mul;

pub struct IdleAnimation;

impl Animation for IdleAnimation {
    type Dependency<'a> = f32;
    type Skeleton = TheropodSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"theropod_idle\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "theropod_idle")]
    fn update_skeleton_inner(
        skeleton: &Self::Skeleton,
        global_time: Self::Dependency<'_>,
        anim_time: f32,
        _rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        let breathe = (anim_time * 0.8).sin();
        let head_look = Vec2::new(
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

        next.head.scale = Vec3::one() * 1.02;
        next.neck.scale = Vec3::one() * 0.98;
        next.jaw.scale = Vec3::one() * 0.98;
        next.foot_l.scale = Vec3::one() * 0.96;
        next.foot_r.scale = Vec3::one() * 0.96;
        next.leg_l.scale = Vec3::one() * 1.02;
        next.leg_r.scale = Vec3::one() * 1.02;
        next.hand_l.scale = Vec3::one() * 0.98;
        next.hand_r.scale = Vec3::one() * 0.98;
        next.tail_front.scale = Vec3::one() * 1.02;
        next.tail_back.scale = Vec3::one() * 0.98;

        next.head.position = Vec3::new(0.0, s_a.head.0, s_a.head.1 + breathe * 0.3);
        next.head.orientation = Quaternion::rotation_x(head_look.y + breathe * 0.1 - 0.1)
            * Quaternion::rotation_z(head_look.x);

        next.jaw.position = Vec3::new(0.0, s_a.jaw.0, s_a.jaw.1);
        next.jaw.orientation = Quaternion::rotation_x(breathe * 0.05 - 0.05);

        next.neck.position = Vec3::new(0.0, s_a.neck.0, s_a.neck.1 + breathe * 0.2);
        next.neck.orientation = Quaternion::rotation_x(-0.1);

        next.chest_front.position =
            Vec3::new(0.0, s_a.chest_front.0, s_a.chest_front.1 + breathe * 0.3);
        next.chest_front.orientation = Quaternion::rotation_x(breathe * 0.04);

        next.chest_back.position = Vec3::new(0.0, s_a.chest_back.0, s_a.chest_back.1);
        next.chest_back.orientation = Quaternion::rotation_x(breathe * -0.04);

        next.tail_front.position = Vec3::new(0.0, s_a.tail_front.0, s_a.tail_front.1);
        next.tail_front.orientation = Quaternion::rotation_x(0.1);

        next.tail_back.position = Vec3::new(0.0, s_a.tail_back.0, s_a.tail_back.1);
        next.tail_back.orientation = Quaternion::rotation_x(0.1);

        next.hand_l.position = Vec3::new(-s_a.hand.0, s_a.hand.1, s_a.hand.2);
        next.hand_l.orientation = Quaternion::rotation_x(breathe * 0.2);

        next.hand_r.position = Vec3::new(s_a.hand.0, s_a.hand.1, s_a.hand.2);
        next.hand_r.orientation = Quaternion::rotation_x(breathe * 0.2);

        next.leg_l.position = Vec3::new(-s_a.leg.0, s_a.leg.1, s_a.leg.2 + breathe * 0.05);
        next.leg_l.orientation = Quaternion::rotation_z(0.0);

        next.leg_r.position = Vec3::new(s_a.leg.0, s_a.leg.1, s_a.leg.2 + breathe * 0.05);
        next.leg_r.orientation = Quaternion::rotation_z(0.0);

        next.foot_l.position = Vec3::new(-s_a.foot.0, s_a.foot.1, s_a.foot.2 + breathe * -0.15);
        next.foot_l.orientation = Quaternion::rotation_z(0.0);

        next.foot_r.position = Vec3::new(s_a.foot.0, s_a.foot.1, s_a.foot.2 + breathe * -0.15);
        next.foot_r.orientation = Quaternion::rotation_z(0.0);

        next
    }
}
