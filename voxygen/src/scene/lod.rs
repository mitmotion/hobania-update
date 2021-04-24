use crate::{
    render::{
        pipelines::lod_terrain::{Locals, LodData, Vertex},
        Consts, GlobalModel, LodTerrainPipeline, Mesh, Model, Quad, Renderer,
        LodStructurePipeline, LodStructureLocals, LodStructureInstance, Instances, Texture, ColLightFmt, ShadowPipeline,
    },
    settings::Settings,
    scene::SceneData,
};
use client::Client;
use common::{
    lod::LodZone,
    terrain::TerrainChunkSize,
    spiral::Spiral2d,
    util::srgba_to_linear,
    vol::RectVolSize,
};
use vek::*;
use hashbrown::HashMap;

pub struct Lod {
    model: Option<(u32, Model<LodTerrainPipeline>)>,
    locals: Consts<Locals>,
    data: LodData,

    structure_col_lights: Texture<ColLightFmt>,
    tree_locals: Consts<LodStructureLocals>,
    tree_model: Model<LodStructurePipeline>,
    zones: HashMap<Vec2<i32>, Zone>,
}

// TODO: Make constant when possible.
pub fn water_color() -> Rgba<f32> {
    /* Rgba::new(0.2, 0.5, 1.0, 0.0) */
    srgba_to_linear(Rgba::new(0.0, 0.25, 0.5, 0.0))
}

impl Lod {
    pub fn new(renderer: &mut Renderer, client: &Client, settings: &Settings) -> Self {
        let (tree_model, structure_col_lights) = {
            use crate::{mesh::{Meshable, greedy::GreedyMesh}, render::SpritePipeline};
            use common::{figure::Segment, assets::{DotVoxAsset, AssetExt}};

            let max_texture_size = renderer.max_texture_size();
            let max_size = guillotiere::Size::new(i32::from(max_texture_size), i32::from(max_texture_size));
            let mut greedy = GreedyMesh::new(max_size);
            let mut mesh = Mesh::new();
            Meshable::<LodStructurePipeline, &mut GreedyMesh>::generate_mesh(
                Segment::from(&DotVoxAsset::load_expect("voxygen.voxel.mini_tree").read().0),
                (&mut greedy, &mut mesh, false),
            );
            println!("Tree mesh has {} vertices", mesh.vertices().len());
            (
                renderer.create_model(&mesh).expect("Failed to upload sprite model data to the GPU!"),
                ShadowPipeline::create_col_lights(renderer, &greedy.finalize())
                    .expect("Failed to upload sprite color and light data to the GPU!"),
            )
        };

        Self {
            model: None,
            locals: renderer.create_consts(&[Locals::default()]).unwrap(),
            data: LodData::new(
                renderer,
                client.world_data().chunk_size(),
                client.world_data().lod_base.raw(),
                client.world_data().lod_alt.raw(),
                client.world_data().lod_horizon.raw(),
                settings.graphics.lod_detail.max(100).min(2500),
                water_color().into_array().into(),
            ),
            tree_model,
            structure_col_lights,
            tree_locals: renderer.create_consts(&[LodStructureLocals::default()]).unwrap(),
            zones: HashMap::default(),
        }
    }

    pub fn get_data(&self) -> &LodData { &self.data }

    pub fn set_detail(&mut self, detail: u32) {
        // Make sure the recorded detail is even.
        self.data.tgt_detail = (detail - detail % 2).max(100).min(2500);
    }

    pub fn maintain(
        &mut self,
        renderer: &mut Renderer,
        scene_data: &SceneData,
    ) {
        if self
            .model
            .as_ref()
            .map(|(detail, _)| *detail != self.data.tgt_detail)
            .unwrap_or(true)
        {
            self.model = Some((
                self.data.tgt_detail,
                renderer
                    .create_model(&create_lod_terrain_mesh(self.data.tgt_detail))
                    .unwrap(),
            ));
        }

        if let Some(zones) = scene_data.client.nearby_zones() {
            for zone_pos in zones {
                if let Some(Some(zone)) = scene_data.client.get_zone(zone_pos) {
                    self.zones
                        .entry(zone_pos)
                        .or_insert_with(|| {
                            let zone_wpos = zone_pos.map2(TerrainChunkSize::RECT_SIZE, |e, sz| e * (sz * LodZone::SIZE) as i32);
                            println!("Zone {:?} ({:?}) has {} trees", zone_pos, zone_wpos, zone.trees.len());
                            Zone {
                                tree_instances: renderer
                                    .create_instances(&zone.trees
                                        .iter()
                                        .map(|tree| LodStructureInstance::new(
                                            (zone_wpos + tree.pos.map(|e| e as i32)).map(|e| e as f32).with_z(tree.alt as f32)
                                        ))
                                        .collect::<Vec<_>>())
                                    .expect("Failed to upload lod tree instances to the GPU!"),
                            }
                        });
                }
            }
        }
    }

    pub fn render(
        &self,
        renderer: &mut Renderer,
        global: &GlobalModel,
        lod: &LodData,
        scene_data: &SceneData,
    ) {
        if let Some((_, model)) = self.model.as_ref() {
            renderer.render_lod_terrain(&model, global, &self.locals, &self.data);
        }

        if let Some(zones) = scene_data.client.nearby_zones() {
            for zone_pos in zones {
                if let Some(zone) = self.zones.get(&zone_pos) {
                    renderer.render_lod_structures(
                        &self.tree_model,
                        &self.structure_col_lights,
                        global,
                        &self.tree_locals,
                        &zone.tree_instances,
                        lod,
                    );
                }
            }
        }
    }
}

fn create_lod_terrain_mesh(detail: u32) -> Mesh<LodTerrainPipeline> {
    // detail is even, so we choose odd detail (detail + 1) to create two even
    // halves with an empty hole.
    let detail = detail + 1;
    Spiral2d::new()
        .take((detail * detail) as usize)
        .skip(1)
        .map(|pos| {
            let x = pos.x + detail as i32 / 2;
            let y = pos.y + detail as i32 / 2;

            let transform = |x| (2.0 * x as f32) / detail as f32 - 1.0;

            Quad::new(
                Vertex::new(Vec2::new(x, y).map(transform)),
                Vertex::new(Vec2::new(x + 1, y).map(transform)),
                Vertex::new(Vec2::new(x + 1, y + 1).map(transform)),
                Vertex::new(Vec2::new(x, y + 1).map(transform)),
            )
            .rotated_by(if (x > detail as i32 / 2) ^ (y > detail as i32 / 2) {
                0
            } else {
                1
            })
        })
        .collect()
}

struct Zone {
    tree_instances: Instances<LodStructureInstance>,
}
