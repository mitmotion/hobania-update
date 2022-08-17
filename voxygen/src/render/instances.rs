use super::buffer::Buffer;
use bytemuck::Pod;

/// Represents a mesh that has been sent to the GPU.
pub struct Instances<T: Copy + Pod> {
    buf: Buffer<T>,
}

impl<T: Copy + Pod> Instances<T> {
    pub fn new_mapped(device: &wgpu::Device, len: usize) -> Self {
        Self {
            buf: Buffer::new_mapped(device, len, wgpu::BufferUsage::VERTEX),
        }
    }

    pub fn new_with_data(device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) -> Self {
        Self {
            buf: Buffer::new(device, queue, wgpu::BufferUsage::VERTEX, data),
        }
    }

    /// Get the GPU-side mapped slice represented by this instances buffer, if it was previously
    /// memory mapped.
    pub fn get_mapped_mut(&self, offset: usize, len: usize) -> /* &mut [T] */wgpu::BufferViewMut<'_> {
        self.buf.get_mapped_mut(offset, len)
    }

    /// Unmaps the GPU-side handle represented by this instances buffer, if it was previously
    /// memory-mapped.
    pub fn unmap(&self, queue: &wgpu::Queue) {
        self.buf.unmap(queue);
    }

    // TODO: count vs len naming scheme??
    pub fn count(&self) -> usize { self.buf.len() }

    pub fn buf(&self) -> &wgpu::Buffer { &self.buf.buf }
}
