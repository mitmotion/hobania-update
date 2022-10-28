use super::{
    super::{vek::*, Animation},
    CharacterSkeleton, SkeletonAttr,
};
use common::states::utils::StageSection;
use core::f32::consts::PI;

pub struct RiposteMeleeAnimation;
impl Animation for RiposteMeleeAnimation {
    type Dependency<'a> = (Option<&'a str>, Option<StageSection>);
    type Skeleton = CharacterSkeleton;

    #[cfg(feature = "use-dyn-lib")]
    const UPDATE_FN: &'static [u8] = b"character_riposte_melee\0";

    #[cfg_attr(feature = "be-dyn-lib", export_name = "character_riposte_melee")]
    fn update_skeleton_inner<'a>(
        skeleton: &Self::Skeleton,
        (ability_id, stage_section): Self::Dependency<'a>,
        anim_time: f32,
        rate: &mut f32,
        s_a: &SkeletonAttr,
    ) -> Self::Skeleton {
        *rate = 1.0;
        let mut next = (*skeleton).clone();

        next.main.position = Vec3::new(0.0, 0.0, 0.0);
        next.main.orientation = Quaternion::rotation_z(0.0);

        match ability_id {
            Some("common.abilities.sword.parrying_riposte") => {
                let (move1, move2, move3) = match stage_section {
                    Some(StageSection::Buildup) => (anim_time.powf(0.25), 0.0, 0.0),
                    Some(StageSection::Action) => (1.0, anim_time.powi(2), 0.0),
                    Some(StageSection::Recover) => (1.0, 1.0, anim_time.powi(4)),
                    _ => (0.0, 0.0, 0.0),
                };

                let move2_slow = move2.powi(4);

                next.hand_l.position = Vec3::new(s_a.shl.0, s_a.shl.1, s_a.shl.2);
                next.hand_l.orientation =
                    Quaternion::rotation_x(s_a.shl.3) * Quaternion::rotation_y(s_a.shl.4);
                next.hand_r.position =
                    Vec3::new(-s_a.sc.0 + 6.0 + move1 * -12.0, -4.0 + move1 * 3.0, -2.0);
                next.hand_r.orientation = Quaternion::rotation_x(0.9 + move1 * 0.5);
                next.control.position = Vec3::new(s_a.sc.0, s_a.sc.1, s_a.sc.2);
                next.control.orientation = Quaternion::rotation_x(s_a.sc.3)
                    * Quaternion::rotation_z(move1 * 1.3 + move2 * -0.7);

                next.chest.orientation = Quaternion::rotation_z(move1 * 0.8);
                next.head.orientation = Quaternion::rotation_z(move1 * -0.4);
                next.belt.orientation = Quaternion::rotation_z(move1 * -0.2);
                next.shorts.orientation = Quaternion::rotation_z(move1 * -0.5);
                next.control.orientation.rotate_x(move1 * 0.5);
                next.control.orientation.rotate_y(move1 * 2.1);
                next.control.orientation.rotate_z(move1 * -0.5);
                next.control.position += Vec3::new(0.0, move1 * 5.0, move1 * 8.0);

                next.chest.orientation.rotate_z(move2 * -1.4);
                next.head.orientation.rotate_z(move2 * 0.9);
                next.belt.orientation.rotate_z(move2 * -0.3);
                next.shorts.orientation.rotate_z(move2 * 0.6);
                next.control.orientation.rotate_y(move2 * -4.0);
                next.control
                    .orientation
                    .rotate_z(move2_slow * -3.0 + move2.powf(0.25) * 1.0);
                next.control.position +=
                    Vec3::new(move2_slow * 11.0, move2_slow * -4.0, move2_slow * -6.0);
                // next.control.orientation.rotate_y(move2.min(0.5) * 2.0 *
                // -2.4); next.control.orientation.
                // rotate_x((move2.max(0.5) - 0.5) * 2.0 * 1.8);
            },
            _ => {},
        }

        next
    }
}
