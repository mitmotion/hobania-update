use crate::{
    comp::{Collider, Density, Mass},
    consts::{AIR_DENSITY, WATER_DENSITY},
    make_case_elim,
    terrain::{Block, BlockKind, SpriteKind},
};
use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use vek::*;

pub const ALL_BODIES: [Body; 4] = [
    Body::DefaultAirship,
    Body::AirBalloon,
    Body::SailBoat,
    Body::Galleon,
];

pub const ALL_AIRSHIPS: [Body; 2] = [Body::DefaultAirship, Body::AirBalloon];
pub const ALL_SHIPS: [Body; 2] = [Body::SailBoat, Body::Galleon];

make_case_elim!(
    body,
    #[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    #[repr(u32)]
    pub enum Body {
        DefaultAirship = 0,
        AirBalloon = 1,
        SailBoat = 2,
        Galleon = 3,
        Volume = 4,
    }
);

impl From<Body> for super::Body {
    fn from(body: Body) -> Self { super::Body::Ship(body) }
}

impl Body {
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        Self::random_with(&mut rng)
    }

    pub fn random_with(rng: &mut impl rand::Rng) -> Self { *(&ALL_BODIES).choose(rng).unwrap() }

    pub fn random_airship_with(rng: &mut impl rand::Rng) -> Self {
        *(&ALL_AIRSHIPS).choose(rng).unwrap()
    }

    pub fn random_ship_with(rng: &mut impl rand::Rng) -> Self { *(&ALL_SHIPS).choose(rng).unwrap() }

    /// Return the structure manifest that this ship uses. `None` means that it
    /// should be derived from the collider.
    pub fn manifest_entry(&self) -> Option<&'static str> {
        match self {
            Body::DefaultAirship => Some("airship_human.structure"),
            Body::AirBalloon => Some("air_balloon.structure"),
            Body::SailBoat => Some("sail_boat.structure"),
            Body::Galleon => Some("galleon.structure"),
            Body::Volume => None,
        }
    }

    pub fn dimensions(&self) -> Vec3<f32> {
        match self {
            Body::DefaultAirship | Body::Volume => Vec3::new(25.0, 50.0, 40.0),
            Body::AirBalloon => Vec3::new(25.0, 50.0, 40.0),
            Body::SailBoat => Vec3::new(13.0, 31.0, 3.0),
            Body::Galleon => Vec3::new(13.0, 32.0, 3.0),
        }
    }

    fn balloon_vol(&self) -> f32 {
        match self {
            Body::DefaultAirship | Body::AirBalloon | Body::Volume => {
                let spheroid_vol = |equat_d: f32, polar_d: f32| -> f32 {
                    (std::f32::consts::PI / 6.0) * equat_d.powi(2) * polar_d
                };
                let dim = self.dimensions();
                spheroid_vol(dim.z, dim.y)
            },
            _ => 0.0,
        }
    }

    fn hull_vol(&self) -> f32 {
        // height from bottom of keel to deck
        let deck_height = 10_f32;
        let dim = self.dimensions();
        (std::f32::consts::PI / 6.0) * (deck_height * 1.5).powi(2) * dim.y
    }

    pub fn hull_density(&self) -> Density {
        let oak_density = 600_f32;
        let ratio = 0.1;
        Density(ratio * oak_density + (1.0 - ratio) * AIR_DENSITY)
    }

    pub fn density(&self) -> Density {
        match self {
            Body::DefaultAirship | Body::AirBalloon | Body::Volume => Density(AIR_DENSITY),
            _ => Density(AIR_DENSITY * 0.8 + WATER_DENSITY * 0.2), // Most boats should be buoyant
        }
    }

    pub fn mass(&self) -> Mass { Mass((self.hull_vol() + self.balloon_vol()) * self.density().0) }

    pub fn can_fly(&self) -> bool {
        matches!(self, Body::DefaultAirship | Body::AirBalloon | Body::Volume)
    }

    pub fn has_water_thrust(&self) -> bool {
        !self.can_fly() // TODO: Differentiate this more carefully
    }

    pub fn make_collider(&self) -> Collider {
        match self.manifest_entry() {
            Some(manifest_entry) => Collider::Voxel {
                id: manifest_entry.to_string(),
            },
            None => {
                use rand::prelude::*;
                let sz = Vec3::broadcast(11);
                Collider::Volume(Arc::new(figuredata::VoxelCollider::from_fn(sz, |_pos| {
                    if thread_rng().gen_bool(0.25) {
                        Block::new(BlockKind::Rock, Rgb::new(255, 0, 0))
                    } else {
                        Block::air(SpriteKind::Empty)
                    }
                })))
            },
        }
    }
}

/// Terrain is 11.0 scale relative to small-scale voxels,
/// airship scale is multiplied by 11 to reach terrain scale.
pub const AIRSHIP_SCALE: f32 = 11.0;

/// Duplicate of some of the things defined in `voxygen::scene::figure::load` to
/// avoid having to refactor all of that to `common` for using voxels as
/// collider geometry
pub mod figuredata {
    use crate::{
        assets::{self, AssetExt, AssetHandle, DotVoxAsset, Ron},
        figure::cell::Cell,
        terrain::{
            block::{Block, BlockKind},
            sprite::SpriteKind,
        },
        volumes::dyna::{ColumnAccess, Dyna},
    };
    use hashbrown::HashMap;
    use lazy_static::lazy_static;
    use serde::{Deserialize, Serialize};
    use vek::Vec3;

    #[derive(Deserialize)]
    pub struct VoxSimple(pub String);

    #[derive(Deserialize)]
    pub struct ShipCentralSpec(pub HashMap<super::Body, SidedShipCentralVoxSpec>);

    #[derive(Deserialize)]
    pub struct SidedShipCentralVoxSpec {
        pub bone0: ShipCentralSubSpec,
        pub bone1: ShipCentralSubSpec,
        pub bone2: ShipCentralSubSpec,
        pub bone3: ShipCentralSubSpec,
    }

    #[derive(Deserialize)]
    pub struct ShipCentralSubSpec {
        pub offset: [f32; 3],
        pub phys_offset: [f32; 3],
        pub central: VoxSimple,
    }

    /// manual instead of through `make_vox_spec!` so that it can be in `common`
    #[derive(Clone)]
    pub struct ShipSpec {
        pub central: AssetHandle<Ron<ShipCentralSpec>>,
        pub colliders: HashMap<String, VoxelCollider>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct VoxelCollider {
        pub(super) dyna: Dyna<Block, (), ColumnAccess>,
        pub translation: Vec3<f32>,
        /// This value should be incremented every time the volume is mutated
        /// and can be used to keep track of volume changes.
        pub mut_count: usize,
    }

    impl VoxelCollider {
        pub fn from_fn<F: FnMut(Vec3<i32>) -> Block>(sz: Vec3<u32>, f: F) -> Self {
            Self {
                dyna: Dyna::from_fn(sz, (), f),
                translation: -sz.map(|e| e as f32) / 2.0,
                mut_count: 0,
            }
        }

        pub fn volume(&self) -> &Dyna<Block, (), ColumnAccess> { &self.dyna }
    }

    impl assets::Compound for ShipSpec {
        fn load<S: assets::source::Source + ?Sized>(
            cache: &assets::AssetCache<S>,
            _: &str,
        ) -> Result<Self, assets::BoxedError> {
            let manifest: AssetHandle<Ron<ShipCentralSpec>> =
                AssetExt::load("common.manifests.ship_manifest")?;
            let mut colliders = HashMap::new();
            for (_, spec) in (manifest.read().0).0.iter() {
                for bone in [&spec.bone0, &spec.bone1, &spec.bone2].iter() {
                    // TODO: Currently both client and server load models and manifests from
                    // "common.voxel.". In order to support CSG procedural airships, we probably
                    // need to load them in the server and sync them as an ECS resource.
                    let vox =
                        cache.load::<DotVoxAsset>(&["common.voxel.", &bone.central.0].concat())?;
                    let dyna = Dyna::<Cell, (), ColumnAccess>::from_vox(&vox.read().0, false);
                    let dyna = dyna.map_into(|cell| {
                        if let Some(rgb) = cell.get_color() {
                            Block::new(BlockKind::Misc, rgb)
                        } else {
                            Block::air(SpriteKind::Empty)
                        }
                    });
                    let collider = VoxelCollider {
                        dyna,
                        translation: Vec3::from(bone.offset) + Vec3::from(bone.phys_offset),
                        mut_count: 0,
                    };
                    colliders.insert(bone.central.0.clone(), collider);
                }
            }
            Ok(ShipSpec {
                central: manifest,
                colliders,
            })
        }
    }

    lazy_static! {
        // TODO: Load this from the ECS as a resource, and maybe make it more general than ships
        // (although figuring out how to keep the figure bones in sync with the terrain offsets seems
        // like a hard problem if they're not the same manifest)
        pub static ref VOXEL_COLLIDER_MANIFEST: AssetHandle<ShipSpec> = AssetExt::load_expect("common.manifests.ship_manifest");
    }

    #[test]
    fn test_ship_manifest_entries() {
        for body in super::ALL_BODIES {
            if let Some(entry) = body.manifest_entry() {
                assert!(
                    VOXEL_COLLIDER_MANIFEST
                        .read()
                        .colliders
                        .get(entry)
                        .is_some()
                );
            }
        }
    }
}
