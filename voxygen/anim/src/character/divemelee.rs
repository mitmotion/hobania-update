use super::{
    super::{vek::*, Animation},
    CharacterSkeleton, SkeletonAttr,
};
use common::states::utils::StageSection;

pub struct DiveMeleeAnimation;
impl Animation for DiveMeleeAnimation {
    type Dependency<'a> = (Option<&'a str>, Option<StageSection>, f32);
    type Skeleton = CharacterSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"character_dive_melee\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "character_dive_melee")]
    fn update_skeleton_inner<'a>(
        skeleton: &Self::Skeleton,
        (ability_id, stage_section, ground_dist): Self::Dependency<'a>,
        anim_time: f32,
        rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        *rate = 1.0;
        let mut next = (*skeleton).clone();

        next.main.position = Vec3::new(0.0, 0.0, 0.0);
        next.main.orientation = Quaternion::rotation_z(0.0);
        next.main_weapon_trail = true;

        let ground_dist = ground_dist.clamp(0.0, 0.5) * 2.0;
        let ground_dist = if ground_dist.is_nan() {
            0.0
        } else {
            ground_dist
        };

        match ability_id {
            Some("common.abilities.sword.cleaving_dive") => {
                let (move1, move2, move3, move4) = match stage_section {
                    Some(StageSection::Movement) => (
                        anim_time.min(1.0).powi(2),
                        (1.0 - ground_dist).powi(2),
                        0.0,
                        0.0,
                    ),
                    Some(StageSection::Action) => (1.0, 1.0, anim_time.powf(0.25), 0.0),
                    Some(StageSection::Recover) => (1.0, 1.0, 1.0, anim_time.powi(4)),
                    _ => (0.0, 0.0, 0.0, 0.0),
                };
                let pullback = 1.0 - move4;
                let move1 = move1 * pullback;
                let move2 = move2 * pullback;
                let move3 = move3 * pullback;

                next.hand_l.position = Vec3::new(s_a.shl.0, s_a.shl.1, s_a.shl.2);
                next.hand_l.orientation =
                    Quaternion::rotation_x(s_a.shl.3) * Quaternion::rotation_y(s_a.shl.4);
                next.hand_r.position =
                    Vec3::new(-s_a.sc.0 + 6.0 + move1 * -12.0, -4.0 + move1 * 3.0, -2.0);
                next.hand_r.orientation = Quaternion::rotation_x(0.9 + move1 * 0.5);
                next.control.position = Vec3::new(s_a.sc.0, s_a.sc.1, s_a.sc.2);
                next.control.orientation = Quaternion::rotation_x(s_a.sc.3);

                next.torso.orientation.rotate_x(move1 * -0.8);
                next.control.orientation.rotate_x(move1 * 1.5);
                next.control.position += Vec3::new(move1 * 7.0, move1.powi(4) * -6.0, move1 * 20.0);

                next.torso.orientation.rotate_x(move2 * 0.8);
                next.chest.orientation = Quaternion::rotation_x(move2 * -0.4);
                next.control.orientation.rotate_x(move2 * -1.2);
                next.control.position += Vec3::new(0.0, move2 * 12.0, move2 * -8.0);

                next.control.orientation.rotate_x(move3 * -1.2);
                next.control.position += Vec3::new(0.0, move3 * 4.0, move3 * -8.0);
            },
            _ => {},
        }

        next
    }
}
