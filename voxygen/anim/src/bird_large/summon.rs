use super::{
    super::{vek::*, Animation},
    BirdLargeSkeleton, SkeletonAttr,
};
use common::{states::utils::StageSection, util::Dir};

pub struct SummonAnimation;

type SummonAnimationDependency = (f32, Option<StageSection>, f32, Dir, bool);

impl Animation for SummonAnimation {
    type Dependency<'a> = SummonAnimationDependency;
    type Skeleton = BirdLargeSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"bird_large_summon\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "bird_large_summon")]
    fn update_skeleton_inner(
        skeleton: &Self::Skeleton,
        (global_time, stage_section, timer, look_dir, on_ground): Self::Dependency<'_>,
        anim_time: f32,
        rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        *rate = 1.0;
        let mut next = (*skeleton).clone();

        let (movement1base, movement2base, movement3, twitch) = match stage_section {
            Some(StageSection::Buildup) => (anim_time.powf(0.25), 0.0, 0.0, 0.0),
            Some(StageSection::Action) => (1.0, anim_time.min(1.0).powf(0.1), 0.0, anim_time),
            Some(StageSection::Recover) => (1.0, 1.0, anim_time.min(1.0).powi(2), 1.0),
            _ => (0.0, 0.0, 0.0, 0.0),
        };

        let pullback = 1.0 - movement3;
        let subtract = global_time - timer;
        let check = subtract - subtract.trunc();
        let mirror = (check - 0.5).signum();
        let twitch2 = mirror * (twitch * 20.0).sin() * pullback;

        let movement1abs = movement1base * pullback;
        let movement2abs = movement2base * pullback;

        let wave_slow_cos = (anim_time * 4.5).cos();

        next.head.scale = Vec3::one() * 0.98;
        next.neck.scale = Vec3::one() * 1.02;
        next.leg_l.scale = Vec3::one() * 0.98;
        next.leg_r.scale = Vec3::one() * 0.98;
        next.foot_l.scale = Vec3::one() * 1.02;
        next.foot_r.scale = Vec3::one() * 1.02;

        next.chest.position = Vec3::new(
            0.0,
            s_a.chest.0,
            s_a.chest.1 + wave_slow_cos * 0.06 + twitch2 * 0.1,
        );

        next.head.position = Vec3::new(0.0, s_a.head.0, s_a.head.1);
        next.head.orientation =
            Quaternion::rotation_x(movement1abs * -1.0 - movement2abs * 0.1 + look_dir.z * 0.4);

        next.beak.position = Vec3::new(0.0, s_a.beak.0, s_a.beak.1);
        next.beak.orientation = Quaternion::rotation_x(movement1abs * -0.7 + twitch2 * 0.1);

        if on_ground {
            next.neck.position = Vec3::new(0.0, s_a.neck.0, s_a.neck.1);
            next.neck.orientation = Quaternion::rotation_x(movement1abs * 0.5 - movement2abs * 0.5);

            next.chest.orientation =
                Quaternion::rotation_x(movement1abs * 1.1 - movement2abs * 0.1);

            next.wing_in_l.position = Vec3::new(-s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);
            next.wing_in_r.position = Vec3::new(s_a.wing_in.0, s_a.wing_in.1, s_a.wing_in.2);

            next.wing_in_l.orientation =
                Quaternion::rotation_x(movement1abs * 0.4 - movement2abs * 0.4)
                    * Quaternion::rotation_y(-1.0 + movement1abs * 1.6 - movement2abs * 1.8)
                    * Quaternion::rotation_z(0.2 - movement1abs * 1.8 + movement2abs * 0.4);
            next.wing_in_r.orientation =
                Quaternion::rotation_x(movement1abs * 0.4 - movement2abs * 0.4)
                    * Quaternion::rotation_y(1.0 - movement1abs * 1.6 + movement2abs * 1.8)
                    * Quaternion::rotation_z(-0.2 + movement1abs * 1.8 - movement2abs * 0.4);

            next.wing_mid_l.position = Vec3::new(-s_a.wing_mid.0, s_a.wing_mid.1, s_a.wing_mid.2);
            next.wing_mid_r.position = Vec3::new(s_a.wing_mid.0, s_a.wing_mid.1, s_a.wing_mid.2);
            next.wing_mid_l.orientation =
                Quaternion::rotation_y(-0.1 - movement2abs * 0.4) * Quaternion::rotation_z(0.7);
            next.wing_mid_r.orientation =
                Quaternion::rotation_y(0.1 + movement2abs * 0.4) * Quaternion::rotation_z(-0.7);

            next.wing_out_l.position = Vec3::new(-s_a.wing_out.0, s_a.wing_out.1, s_a.wing_out.2);
            next.wing_out_r.position = Vec3::new(s_a.wing_out.0, s_a.wing_out.1, s_a.wing_out.2);
            next.wing_out_l.orientation =
                Quaternion::rotation_y(-0.2 - movement2abs * 0.4) * Quaternion::rotation_z(0.2);
            next.wing_out_r.orientation =
                Quaternion::rotation_y(0.2 + movement2abs * 0.4) * Quaternion::rotation_z(-0.2);

            next.tail_front.position = Vec3::new(0.0, s_a.tail_front.0, s_a.tail_front.1);
            next.tail_front.orientation =
                Quaternion::rotation_x(-movement1abs * 0.1 + movement2abs * 0.1 + twitch2 * 0.02);
            next.tail_rear.position = Vec3::new(0.0, s_a.tail_rear.0, s_a.tail_rear.1);
            next.tail_rear.orientation =
                Quaternion::rotation_x(-movement1abs * 0.1 + movement2abs * 0.1 + twitch2 * 0.02);
        } else {
        }

        next
    }
}
