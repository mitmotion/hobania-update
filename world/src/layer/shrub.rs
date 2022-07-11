use crate::{
    all::ForestKind,
    site2::{self, PrimitiveTransform},
    util::{gen_cache::StructureGenCache, seed_expan, Sampler, StructureGen2d, UnitChooser},
    Canvas,
};
use common::{
    assets::AssetHandle,
    terrain::structure::{Structure, StructuresGroup},
};
use lazy_static::lazy_static;
use rand::prelude::*;
use rand_chacha::ChaChaRng;
use vek::*;

lazy_static! {
    static ref JUNGLE_SHRUBS: AssetHandle<StructuresGroup> = Structure::load_group("shrubs.jungle");
    static ref SAVANNAH_SHRUBS: AssetHandle<StructuresGroup> =
        Structure::load_group("shrubs.savannah");
    static ref TEMPERATE_SHRUBS: AssetHandle<StructuresGroup> =
        Structure::load_group("shrubs.temperate");
    static ref TAIGA_SHRUBS: AssetHandle<StructuresGroup> = Structure::load_group("shrubs.taiga");
}

struct Shrub {
    wpos: Vec3<i32>,
    seed: u32,
    kind: ForestKind,
    rng: ChaChaRng,
}

pub fn apply_shrubs_to(canvas: &mut Canvas, _dynamic_rng: &mut impl Rng) {
    // let mut shrub_gen = StructureGenCache::new(StructureGen2d::new(canvas.index().seed, 8, 4));

    let info = canvas.info();

    let area_size = Vec2::from(info.area().size().map(|e| e as i32));
    let render_area = Aabr {
        min: info.wpos(),
        max: info.wpos() + area_size,
    };

    let mut arena = bumpalo::Bump::new();

    info.chunks()
        .gen_ctx
        .shrub_gen
        .iter(render_area.min, render_area.max)
        .filter_map(|(wpos, seed)| {
            let lottery = info.chunks().make_forest_lottery(wpos);
            let kind = *lottery.choose_seeded(seed).as_ref()?;

            let mut rng = ChaChaRng::from_seed(seed_expan::rng_state(seed));
            if rng.gen_bool(kind.shrub_density_factor() as f64) {
                info.col_or_gen(wpos)
                    .and_then(move |col| {
                        const BASE_SHRUB_DENSITY: f64 = 0.15;
                        if rng.gen_bool((BASE_SHRUB_DENSITY * col.tree_density as f64).clamped(0.0, 1.0))
                            && col.water_dist.map_or(true, |d| d > 8.0)
                            && col.alt > col.water_level
                            && col.spawn_rate > 0.9
                            && col.path.map_or(true, |(d, _, _, _)| d > 6.0)
                        {
                            Some(Shrub {
                                wpos: wpos.with_z(col.alt as i32),
                                seed,
                                kind,
                                rng,
                            })
                        } else {
                            None
                        }
                    })
            } else {
                None
            }
        })
        .for_each(|mut shrub| {
    /* canvas.foreach_col(|_, wpos2d, _| {
        shrub_gen.get(wpos2d, |wpos, seed| {
            let col = info.col_or_gen(wpos)?;

            let mut rng = ChaChaRng::from_seed(seed_expan::rng_state(seed));

            const BASE_SHRUB_DENSITY: f64 = 0.15;
            if rng.gen_bool((BASE_SHRUB_DENSITY * col.tree_density as f64).clamped(0.0, 1.0))
                && col.water_dist.map_or(true, |d| d > 8.0)
                && col.alt > col.water_level
                && col.spawn_rate > 0.9
                && col.path.map_or(true, |(d, _, _, _)| d > 6.0)
            {
                let kind = *info
                    .chunks()
                    .make_forest_lottery(wpos)
                    .choose_seeded(seed)
                    .as_ref()?;
                if rng.gen_bool(kind.shrub_density_factor() as f64) {
                    Some(Shrub {
                        wpos: wpos.with_z(col.alt as i32),
                        seed,
                        kind,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        });
    });

    for shrub in shrub_gen.generated() {
        let mut rng = ChaChaRng::from_seed(seed_expan::rng_state(shrub.seed)); */

        // let units = UnitChooser::new(shrub.seed).get(shrub.seed).into();

        let shrubs = match shrub.kind {
            ForestKind::Mangrove => &*JUNGLE_SHRUBS,
            ForestKind::Acacia | ForestKind::Baobab => &*SAVANNAH_SHRUBS,
            ForestKind::Oak | ForestKind::Chestnut => &*TEMPERATE_SHRUBS,
            ForestKind::Pine => &*TAIGA_SHRUBS,
            _ => return, // TODO: Add more shrub varieties
        }
        ./*read*/get();

        let structure = shrubs.choose(&mut shrub.rng).unwrap();

        site2::render_collect(
            &arena,
            info,
            render_area,
            canvas,
            |painter, filler| {
                painter
                    .prefab(structure)
                    .translate(/*totem_pos*/shrub.wpos)
                    .fill(filler.prefab(structure, shrub.wpos, shrub.seed), filler);
            },
        );

        arena.reset();

        // canvas.blit_structure(shrub.wpos, structure, shrub.seed, units, true);
    /* } */
        });
}
