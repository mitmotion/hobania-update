use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub struct Buffer<T: Copy + Pod> {
    pub(super) buf: wgpu::Buffer,
    // Size in number of elements
    // TODO: determine if this is a good name
    len: usize,
    phantom_data: std::marker::PhantomData<T>,
}

impl<T: Copy + Pod> Buffer<T> {
    pub fn new_mapped(device: &wgpu::Device, len: usize, usage: wgpu::BufferUsage) -> Self {
        Self {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                mapped_at_creation: true,
                size: len as u64 * std::mem::size_of::<T>() as u64,
                usage: usage | wgpu::BufferUsage::COPY_DST,
            }),
            len,
            phantom_data: std::marker::PhantomData,
        }
    }

    pub fn new(device: &wgpu::Device, usage: wgpu::BufferUsage, data: &[T]) -> Self {
        let contents = bytemuck::cast_slice(data);

        Self {
            buf: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage,
            }),
            len: data.len(),
            phantom_data: std::marker::PhantomData,
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize { self.len }
}

pub struct DynamicBuffer<T: Copy + Pod>(Buffer<T>);

impl<T: Copy + Pod> DynamicBuffer<T> {
    pub fn new(device: &wgpu::Device, len: usize, usage: wgpu::BufferUsage) -> Self {
        let buffer = Buffer {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                mapped_at_creation: false,
                size: len as u64 * std::mem::size_of::<T>() as u64,
                usage: usage | wgpu::BufferUsage::COPY_DST,
            }),
            len,
            phantom_data: std::marker::PhantomData,
        };
        Self(buffer)
    }

    pub fn new_with_data(device: &wgpu::Device, usage: wgpu::BufferUsage, data: &[T]) -> Self {
        Self(Buffer::new(device, usage | wgpu::BufferUsage::COPY_DST, data))
    }

    pub fn new_mapped(device: &wgpu::Device, len: usize, usage: wgpu::BufferUsage) -> Self {
        Self(Buffer::new_mapped(device, len, usage | wgpu::BufferUsage::COPY_DST))
    }

    pub fn update(&self, queue: &wgpu::Queue, vals: &[T], offset: usize) {
        if !vals.is_empty() {
            queue.write_buffer(
                &self.buf,
                offset as u64 * std::mem::size_of::<T>() as u64,
                bytemuck::cast_slice(vals),
            )
        }
    }

    /// Update the GPU-side value represented by this constant handle, if it was previously memory
    /// mapped, and then unmaps it.
    ///
    /// NOTE: Will panic if the buffer was not explicitly mapped before this (without being
    /// unmapped), either directly or via [Buffer::new_mapped].
    ///
    /// NOTE: Queue is not *explicitly* used here, but it is implicitly used during the unmap
    /// (within wgpu internals) and requires acquiring a lock on it, so it's left in the API to
    /// deter people from using it when the queue isn't available.
    pub fn update_mapped(&mut self, _queue: &wgpu::Queue, vals: &[T], offset: usize) {
        if !vals.is_empty() {
            let contents = bytemuck::cast_slice(vals);

            let size_ty = std::mem::size_of::<T>() as u64;
            let offset = offset as u64 * size_ty;
            let size = vals.len() as u64 * size_ty;
            self.buf.slice(offset..offset + size)
                .get_mapped_range_mut()
                .copy_from_slice(contents);
        }
        self.buf.unmap();
    }
}

impl<T: Copy + Pod> std::ops::Deref for DynamicBuffer<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target { &self.0 }
}
