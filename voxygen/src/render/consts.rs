use super::buffer::DynamicBuffer;
use bytemuck::Pod;

/// A handle to a series of constants sitting on the GPU. This is used to hold
/// information used in the rendering process that does not change throughout a
/// single render pass.
pub struct Consts<T: Copy + Pod> {
    buf: DynamicBuffer<T>,
}

impl<T: Copy + Pod> Consts<T> {
    /// Create a new `Const<T>`.
    pub fn new(device: &wgpu::Device, len: usize) -> Self {
        Self {
            // TODO: examine if all our consts need to be updatable
            buf: DynamicBuffer::new(device, len, wgpu::BufferUsage::UNIFORM),
        }
    }

    pub fn new_with_data(device: &wgpu::Device, data: &[T]) -> Self {
        Self {
            // TODO: examine if all our consts need to be updatable
            buf: DynamicBuffer::new_with_data(device, wgpu::BufferUsage::UNIFORM, data),
        }
    }

    /// Create a new `Const<T>` that is mapped at creation.
    ///
    /// Warning: buffer must be unmapped before attempting to use this buffer on the GPU!
    pub fn new_mapped(device: &wgpu::Device, len: usize) -> Self {
        Self {
            // TODO: examine if all our consts need to be updatable
            buf: DynamicBuffer::new_mapped(device, len, wgpu::BufferUsage::UNIFORM),
        }
    }

    /// Update the GPU-side value represented by this constant handle.
    pub fn update(&mut self, queue: &wgpu::Queue, vals: &[T], offset: usize) {
        self.buf.update(queue, vals, offset)
    }

    /// Get the GPU-side mapped slice represented by this constant handle, if it was previously
    /// memory mapped.
    pub fn get_mapped_mut(&self, offset: usize, len: usize) -> /* &mut [T] */wgpu::BufferViewMut<'_> {
        self.buf.get_mapped_mut(offset, len)
    }

    /// Unmaps the GPU-side handle represented by this constant handle, if it was previously
    /// memory-mapped.
    pub fn unmap(&self, queue: &wgpu::Queue) {
        self.buf.unmap(queue);
    }

    pub fn buf(&self) -> &wgpu::Buffer { &self.buf.buf }

    pub fn len(&self) -> usize { self.buf.len() }
}
