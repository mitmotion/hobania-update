use crate::render::pipelines::rain_occlusion;

use std::sync::Arc;
use super::{
    super::{
        pipelines::{
            debug, figure, lod_terrain, shadow, sprite, terrain, ui, ColLights, GlobalModel,
            GlobalsBindGroup,
        },
        texture::Texture,
    },
    Consts,
    Renderer,
};

impl Renderer {
    pub fn bind_globals(
        &self,
        global_model: &GlobalModel,
        lod_data: &lod_terrain::LodData,
    ) -> GlobalsBindGroup {
        self.layouts
            .global
            .bind(&self.device, global_model, lod_data, &self.noise_tex)
    }

    pub fn bind_sprite_globals(
        &self,
        global_model: &GlobalModel,
        lod_data: &lod_terrain::LodData,
        sprite_verts: &sprite::SpriteVerts,
    ) -> sprite::SpriteGlobalsBindGroup {
        self.layouts.sprite.bind_globals(
            &self.device,
            global_model,
            lod_data,
            &self.noise_tex,
            sprite_verts,
        )
    }

    pub fn create_debug_bound_locals(&mut self, vals: &[debug::Locals]) -> debug::BoundLocals {
        let locals = self.create_consts(vals);
        self.layouts.debug.bind_locals(&self.device, locals)
    }

    pub fn create_ui_bound_locals(&mut self, vals: &[ui::Locals]) -> ui::BoundLocals {
        let locals = self.create_consts(vals);
        self.layouts.ui.bind_locals(&self.device, locals)
    }

    pub fn ui_bind_texture(&self, texture: &Texture) -> ui::TextureBindGroup {
        self.layouts.ui.bind_texture(&self.device, texture)
    }

    pub fn create_figure_bound_locals(
        &mut self,
        locals: &[figure::Locals],
        bone_data: &[figure::BoneData],
    ) -> figure::BoundLocals {
        let locals = self.create_consts(locals);
        let bone_data = self.create_consts(bone_data);
        self.layouts
            .figure
            .bind_locals(&self.device, locals, bone_data)
    }

    /* /// Create a new set of constants with the provided values, lazily (so this can be instantiated
    /// from another thread).
    pub fn create_consts_lazy<T: Copy + bytemuck::Pod>(&mut self) ->
        impl for<'a> Fn(&'a [T]) -> Consts<T> + Send + Sync
    {
        let device = Arc::clone(&self.device);
        move |vals| Self::create_consts_inner(&device, vals)
    } */

    /// NOTE: Locals are mapped at creation, so you still have to memory map and bind them in order
    /// before use.
    pub fn create_terrain_bound_locals(
        &mut self,
        locals: /*Arc<*/&Consts<terrain::Locals>/*>*/,
        offset: usize,
    ) -> /*for<'a> Fn(&'a [terrain::Locals]) -> terrain::BoundLocals + Send + Sync*//* impl Fn() -> terrain::BoundLocals + Send + Sync */terrain::BoundLocals {
        /* let device = Arc::clone(&self.device);
        let immutable = Arc::clone(&self.layouts.immutable);
        move || {
            let locals = Consts::new_mapped(&device, 1);
            immutable.terrain.bind_locals(&device, locals)
        } */
        self.layouts.immutable.terrain.bind_locals(&self.device, locals, offset)
    }

    pub fn create_shadow_bound_locals(&mut self, locals: &[shadow::Locals]) -> shadow::BoundLocals {
        let locals = self.create_consts(locals);
        self.layouts.shadow.bind_locals(&self.device, locals)
    }

    pub fn create_rain_occlusion_bound_locals(
        &mut self,
        locals: &[rain_occlusion::Locals],
    ) -> rain_occlusion::BoundLocals {
        let locals = self.create_consts(locals);
        self.layouts
            .rain_occlusion
            .bind_locals(&self.device, locals)
    }

    pub fn figure_bind_col_light(&self, col_light: Texture) -> ColLights<figure::Locals> {
        self.layouts.global.bind_col_light(&self.device, col_light)
    }

    pub fn terrain_bind_col_light(&self, col_light: Texture) -> ColLights<terrain::Locals> {
        self.layouts.global.bind_col_light(&self.device, col_light)
    }

    pub fn sprite_bind_col_light(&self, col_light: Texture) -> ColLights<sprite::Locals> {
        self.layouts.global.bind_col_light(&self.device, col_light)
    }
}
