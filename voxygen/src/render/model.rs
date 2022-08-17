use super::{
    buffer::{Buffer, DynamicBuffer},
    mesh::Mesh,
    Vertex,
};
use std::ops::Range;

/// Represents a mesh that has been sent to the GPU.
pub struct SubModel<'a, V: Vertex> {
    pub vertex_range: Range<u32>,
    buf: &'a wgpu::Buffer,
    phantom_data: std::marker::PhantomData<V>,
}

impl<'a, V: Vertex> SubModel<'a, V> {
    pub(super) fn buf(&self) -> wgpu::BufferSlice<'a> {
        let start = self.vertex_range.start as wgpu::BufferAddress * V::STRIDE;
        let end = self.vertex_range.end as wgpu::BufferAddress * V::STRIDE;
        self.buf.slice(start..end)
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u32 { self.vertex_range.end - self.vertex_range.start }
}

/// Represents a mesh that has been sent to the GPU.
pub struct Model<V: Vertex> {
    vbuf: Buffer<V>,
}

impl<V: Vertex> Model<V> {
    /// Returns None if the provided mesh is empty
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, usage: wgpu::BufferUsage, mesh: &Mesh<V>) -> Option<Self> {
        if mesh.vertices().is_empty() {
            return None;
        }

        Some(Self {
            vbuf: Buffer::new(device, queue, /*wgpu::BufferUsage::VERTEX*/usage, mesh.vertices()),
        })
    }

    /// Create a new `Const<T>` that is mapped at creation.  Returns None if the mesh is empty.
    ///
    /// Warning: buffer must be unmapped before attempting to use this buffer on the GPU!
    pub fn new_mapped(device: &wgpu::Device, len: usize, usage: wgpu::BufferUsage) -> Option<Self> {
        if len == 0 {
            return None;
        }

        Some(Self {
            vbuf: Buffer::new_mapped(device, len, /*wgpu::BufferUsage::VERTEX*/usage/*, mesh.vertices()*/),
        })
    }

    /// Get the GPU-side mapped slice represented by this model handle, if it was previously
    /// memory mapped.
    pub fn get_mapped_mut(&self, offset: usize, len: usize) -> /* &mut [T] */wgpu::BufferViewMut<'_> {
        self.vbuf.get_mapped_mut(offset, len)
    }

    /// Unmaps the GPU-side handle represented by this model handle, if it was previously
    /// memory-mapped.
    pub fn unmap(&self, queue: &wgpu::Queue) {
        self.vbuf.unmap(queue);
    }

    /// Create a model with a slice of a portion of this model to send to the
    /// renderer.
    pub fn submodel(&self, vertex_range: Range<u32>) -> SubModel<V> {
        SubModel {
            vertex_range,
            buf: self.buf(),
            phantom_data: std::marker::PhantomData,
        }
    }

    pub fn buf(&self) -> &wgpu::Buffer { &self.vbuf.buf }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize { self.vbuf.len() }
}

/// Represents a mesh that has been sent to the GPU.
pub struct DynamicModel<V: Vertex> {
    vbuf: DynamicBuffer<V>,
}

impl<V: Vertex> DynamicModel<V> {
    pub fn new(device: &wgpu::Device, size: usize) -> Self {
        Self {
            vbuf: DynamicBuffer::new(device, size, wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST),
        }
    }

    pub fn update(&self, queue: &wgpu::Queue, mesh: &Mesh<V>, offset: usize) {
        self.vbuf.update(queue, mesh.vertices(), offset)
    }

    /// Create a model with a slice of a portion of this model to send to the
    /// renderer.
    pub fn submodel(&self, vertex_range: Range<u32>) -> SubModel<V> {
        SubModel {
            vertex_range,
            buf: self.buf(),
            phantom_data: std::marker::PhantomData,
        }
    }

    pub fn buf(&self) -> &wgpu::Buffer { &self.vbuf.buf }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize { self.vbuf.len() }
}
