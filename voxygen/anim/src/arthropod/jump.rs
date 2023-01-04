use super::{
    super::{vek::*, Animation},
    ArthropodSkeleton, SkeletonAttr,
};

pub struct JumpAnimation;

impl Animation for JumpAnimation {
    type Dependency<'a> = (f32, Vec3<f32>, Vec3<f32>, f32, Vec3<f32>);
    type Skeleton = ArthropodSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"arthropod_jump\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "arthropod_jump")]
    fn update_skeleton_inner(
        skeleton: &Self::Skeleton,
        (_velocity, _orientation, _last_ori, _global_time, _avg_vel): Self::Dependency<'_>,
        _anim_time: f32,
        _rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        let mut next = (*skeleton).clone();

        next.chest.scale = Vec3::one() / s_a.scaler;
        next.wing_bl.scale = Vec3::one() * 0.98;
        next.wing_br.scale = Vec3::one() * 0.98;

        next.head.position = Vec3::new(0.0, s_a.head.0, s_a.head.1);

        next.chest.position = Vec3::new(0.0, s_a.chest.0, s_a.chest.1);

        next.mandible_l.position = Vec3::new(-s_a.mandible.0, s_a.mandible.1, s_a.mandible.2);
        next.mandible_r.position = Vec3::new(s_a.mandible.0, s_a.mandible.1, s_a.mandible.2);

        next.wing_fl.position = Vec3::new(-s_a.wing_f.0, s_a.wing_f.1, s_a.wing_f.2);
        next.wing_fr.position = Vec3::new(s_a.wing_f.0, s_a.wing_f.1, s_a.wing_f.2);

        next.wing_bl.position = Vec3::new(-s_a.wing_b.0, s_a.wing_b.1, s_a.wing_b.2);
        next.wing_br.position = Vec3::new(s_a.wing_b.0, s_a.wing_b.1, s_a.wing_b.2);

        next.leg_fl.position = Vec3::new(-s_a.leg_f.0, s_a.leg_f.1, s_a.leg_f.2);
        next.leg_fr.position = Vec3::new(s_a.leg_f.0, s_a.leg_f.1, s_a.leg_f.2);
        next.leg_fl.orientation =
            Quaternion::rotation_z(s_a.leg_ori.0) * Quaternion::rotation_y(0.6);
        next.leg_fr.orientation =
            Quaternion::rotation_z(-s_a.leg_ori.0) * Quaternion::rotation_y(-0.6);

        next.leg_fcl.position = Vec3::new(-s_a.leg_fc.0, s_a.leg_fc.1, s_a.leg_fc.2);
        next.leg_fcr.position = Vec3::new(s_a.leg_fc.0, s_a.leg_fc.1, s_a.leg_fc.2);
        next.leg_fcl.orientation =
            Quaternion::rotation_z(s_a.leg_ori.1) * Quaternion::rotation_y(0.6);
        next.leg_fcr.orientation =
            Quaternion::rotation_z(-s_a.leg_ori.1) * Quaternion::rotation_y(-0.6);

        next.leg_bcl.position = Vec3::new(-s_a.leg_bc.0, s_a.leg_bc.1, s_a.leg_bc.2);
        next.leg_bcr.position = Vec3::new(s_a.leg_bc.0, s_a.leg_bc.1, s_a.leg_bc.2);
        next.leg_bcl.orientation =
            Quaternion::rotation_z(s_a.leg_ori.2) * Quaternion::rotation_y(0.6);
        next.leg_bcr.orientation =
            Quaternion::rotation_z(-s_a.leg_ori.2) * Quaternion::rotation_y(-0.6);

        next.leg_bl.position = Vec3::new(-s_a.leg_b.0, s_a.leg_b.1, s_a.leg_b.2);
        next.leg_br.position = Vec3::new(s_a.leg_b.0, s_a.leg_b.1, s_a.leg_b.2);
        next.leg_bl.orientation =
            Quaternion::rotation_z(s_a.leg_ori.3) * Quaternion::rotation_y(0.6);
        next.leg_br.orientation =
            Quaternion::rotation_z(-s_a.leg_ori.3) * Quaternion::rotation_y(-0.6);

        next
    }
}
