use super::super::pipelines::blit;

pub type ScreenshotFn = Box<dyn FnMut(image::DynamicImage) + Send>;

pub struct ScreenshotDownloader {
    buffer: wgpu::Buffer,
    screenshot_fn: ScreenshotFn,
}

pub struct TakeScreenshot {
    bind_group: blit::BindGroup,
    view: wgpu::TextureView,
    texture: wgpu::Texture,
    // /// Option so that we can pass ownership of the contents of this field around (and eventually to a new thread) without
    // /// taking ownership of this whole struct
    downloader: ScreenshotDownloader,
    // Dimensions used for copying from the screenshot texture to a buffer
    width: u32,
    height: u32,
    bytes_per_pixel: u8,
}

//pub struct TakingScreenshot<'a> {
//    pub bind_group: &'a blit::BindGroup,
//    pub tex: &'a wgpu::TextureView,
//    downloader: Option<ScreenshotDownloader>
//}

impl TakeScreenshot {
pub fn new(
    device: &wgpu::Device,
    blit_layout: &blit::BlitLayout,
    sampler: &wgpu::Sampler,
    /// Used to determine the resolution and texture format
    sc_desc: &wgpu::SwapChainDescriptor,
    /// Function that is given the image after downloading it from the GPU
    /// This is executed in a background thread
    screenshot_fn: ScreenshotFn,
) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot tex"),
            size: wgpu::Extent3d {
                width: sc_desc.width,
                height: sc_desc.height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::COPY_SRC
                | wgpu::TextureUsage::SAMPLED
                | wgpu::TextureUsage::RENDER_ATTACHEMENT,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("screenshot tex view"),
            format: Some(sc_desc.format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let bind_group = blit_layout.bind(device, &taking_tex, sampler);

        let bytes_per_pixel = sc_desc.format.describe().block_size;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("screenshot download buffer"),
            size: padded_bytes_per_row * 
        }
        )
        let downloader = ScreenshotDownloader { screenshot_fn, buffer };

        Self {
            bind_group,
            texture,
            view,
            downloader: Some(Downloader),

        }

    /// NOTE: spawns thread
    /// Call this after rendering to the screenshot texture
    pub fn download_and_handle(
        self,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        // Calculate padded bytes per row
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let unpadded_bytes_per_row = 
        // Copy image to a buffer
        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &self.texture,
                mip_level: 0,
                orgin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &self.buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: padded_bytes_per_row(self.width, self.bytes_per_pixel),
                    rows_per_image: 0,
                }
            },
        )
        // Send buffer to another thread for async mapping, downloading, and passing to the given
        // handler function (which probably saves it to the disk)
        std::thread::Builder::new().name("screenshot".into()).spawn(|| {
                     
        });
        .expect("Failed to spawn screenshot thread");
    }
}

fn padded_bytes_per_row(width: u32, bytes_per_pixel: u8) -> u32 {
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padding = (align - unpadded_bytes_per_row % align) % align;
    unpadded_bytes_per_row + padding
}
