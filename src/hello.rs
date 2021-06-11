#![allow(unused)]
#![allow(unused_imports)]

use log::{debug, error, info, log_enabled, Level};

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use winit::dpi::LogicalSize;
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkano::buffer::{
    BufferAccess, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, ImmutableBuffer,
    TypedBufferAccess,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, DynamicState,
    PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor::descriptor::{
    DescriptorBufferDesc, DescriptorDesc, DescriptorDescTy, ShaderStages,
};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSetsPool, PersistentDescriptorSet};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::{
    swapchain::SwapchainImage, view::ImageView, ImageDimensions, ImageUsage, ImmutableImage,
    MipmapsCount,
};
use vulkano::impl_vertex;
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{
    layers_list, ApplicationInfo, Instance, InstanceExtensions, PhysicalDevice, Version,
};
use vulkano::pipeline::{viewport::Viewport, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::single_pass_renderpass;
use vulkano::swapchain::{
    acquire_next_image, AcquireError, Capabilities, ColorSpace, CompositeAlpha,
    FullscreenExclusive, PresentMode, SupportedPresentModes, Surface, Swapchain,
};
use vulkano::sync::{self, GpuFuture, NowFuture, SharingMode};
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode};

use vulkano_win::VkSurfaceBuild;

use cgmath::{Deg, Matrix4, Point3, Rad, Vector2, Vector3};

use image::io::Reader as ImageReader;
use image::DynamicImage::*;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;
const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

/// Required device extensions
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    }
}

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;

#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self {
            graphics_family: -1,
            present_family: -1,
        }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

#[derive(Default, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
    uv: [f32; 2],
}

impl Vertex {
    fn new(position: [f32; 2], uv: [f32; 2]) -> Self {
        Self { position, uv }
    }
}

#[allow(clippy::ref_in_deref)]
impl_vertex!(Vertex, position, uv);

#[allow(dead_code)]
#[derive(Copy, Clone)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

fn vertices() -> [Vertex; 4] {
    [
        Vertex::new([-0.5, -0.5], [0.0, 0.0]),
        Vertex::new([0.5, -0.5], [1.0, 0.0]),
        Vertex::new([0.5, 0.5], [1.0, 1.0]),
        Vertex::new([-0.5, 0.5], [0.0, 1.0]),
    ]
}

fn indices() -> [u16; 6] {
    [0, 1, 2, 2, 3, 0]
}

struct MouseState {
    delta: [f64; 2],
    position: [f64; 2],

    right_click: bool,
}

impl MouseState {
    pub fn new() -> Self {
        Self {
            delta: [0.0, 0.0],
            position: [0.0, 0.0],
            right_click: false,
        }
    }
}

struct InputState {
    mouse: MouseState,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            mouse: MouseState::new(),
        }
    }
}

pub struct HelloWorldApplication {
    event_loop: Option<EventLoop<()>>,
    surface: Arc<Surface<Window>>,
    instance: Arc<Instance>,

    #[allow(unused)]
    debug_callback: Option<DebugCallback>,

    physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<RenderPass>,

    graphics_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    index_buffer: Arc<dyn TypedBufferAccess<Content = [u16]> + Send + Sync>,

    uniform_buffers: Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>>,

    descriptor_sets_pool: FixedSizeDescriptorSetsPool,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    #[allow(dead_code)]
    start_time: Instant,

    input: InputState,

    image_view: Arc<ImageView<Arc<ImmutableImage>>>,
    image_sampler: Arc<Sampler>,
}

impl HelloWorldApplication {
    pub fn new() -> Self {
        let instance = Self::create_instance();
        let (event_loop, surface) = Self::create_surface(&instance);
        let debug_callback = Self::setup_debug_callback(&instance);
        let physical_device_index = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) =
            Self::create_logical_device(&instance, &surface, physical_device_index);

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(
            &instance,
            &surface,
            physical_device_index,
            &device,
            &graphics_queue,
            &present_queue,
            None,
        );

        let render_pass = Self::create_render_pass(&device, swap_chain.format());

        let graphics_pipeline =
            Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_framebuffers(&swap_chain_images, &render_pass);

        let start_time = Instant::now();

        let vertex_buffer = Self::create_vertex_buffer(&graphics_queue);
        let index_buffer = Self::create_index_buffer(&graphics_queue);
        let uniform_buffers = Self::create_uniform_buffers(
            &device,
            swap_chain_images.len(),
            start_time,
            swap_chain.dimensions(),
        );

        let descriptor_sets_pool = FixedSizeDescriptorSetsPool::new(
            graphics_pipeline.descriptor_set_layout(0).unwrap().clone(),
        );

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let (image_view, image_future) = Self::load_image(&graphics_queue);

        let image_sampler = Sampler::new(device.clone(),
            Filter::Linear,
            Filter::Linear, MipmapMode::Linear,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0,
            1.0,
            0.0,
            0.0
        ).unwrap();

        Self {
            event_loop: Some(event_loop),
            surface,
            instance,
            debug_callback,
            physical_device_index,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,

            render_pass,

            graphics_pipeline,

            swap_chain_framebuffers,

            vertex_buffer,
            index_buffer,
            uniform_buffers,

            descriptor_sets_pool,

            previous_frame_end,
            recreate_swap_chain: false,

            start_time,

            input: InputState::new(),

            image_view,
            image_sampler,
        }
    }

    fn load_image(
        graphics_queue: &Arc<Queue>,
    ) -> (
        Arc<ImageView<Arc<ImmutableImage>>>,
        CommandBufferExecFuture<NowFuture, PrimaryAutoCommandBuffer>,
    ) {
        let img = ImageReader::open("src/img/grass.jpg")
            .unwrap()
            .decode()
            .unwrap();

        let rgb = match img {
            ImageRgb8(rgb_image) => rgb_image,
            _ => panic!("Not Rgb8"),
        };
        let image_data = rgb.as_raw();
        let dimensions = rgb.dimensions();
        let image_dimensions = ImageDimensions::Dim2d {
            width: dimensions.0,
            height: dimensions.1,
            array_layers: 1,
        };

        let (image, future) = ImmutableImage::from_iter(
            image_data.iter().cloned(),
            image_dimensions,
            MipmapsCount::One,
            Format::R8G8B8Srgb,
            graphics_queue.clone(),
        )
        .unwrap();

        (ImageView::new(image).unwrap(), future)
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layers_support() {
            error!("Validation layers requested, but not available!");
        }

        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("Failed to retrieve supported extensions");
        info!("Supported extensions: {:?}", supported_extensions);
        let app_info = ApplicationInfo {
            application_name: Some("Vulkan Renderer".into()),
            application_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version {
                major: 1,
                minor: 0,
                patch: 0,
            }),
        };
        let required_extensions = Self::get_required_extensions();

        Instance::new(Some(&app_info), &required_extensions, None)
            .expect("Failed to create Vulkan instance")
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_types = MessageType::all();

        let msg_severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };

        DebugCallback::new(&instance, msg_severity, msg_types, |msg| {
            info!("Validation layer: {:?}", msg.description);
        })
        .ok()
    }

    fn check_validation_layers_support() -> bool {
        let layers: Vec<_> = layers_list()
            .unwrap()
            .map(|l| l.name().to_owned())
            .collect();

        VALIDATION_LAYERS
            .iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();

        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_utils = true;
        }

        extensions
    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("Failed to find a suitable GPU!")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        let extensions_supported = Self::check_device_extension_support(device);
        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface
                .capabilities(*device)
                .expect("Failed to get surface capabilities");
            !capabilities.supported_formats.is_empty()
                && capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();

        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(
        available_formats: &[(Format, ColorSpace)],
    ) -> (Format, ColorSpace) {
        *available_formats
            .iter()
            .find(|(format, color_space)| {
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            })
            .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            current_extent
        } else {
            let mut actual_extent = [WIDTH, HEIGHT];

            actual_extent[0] =
                actual_extent[0].clamp(actual_extent[0], capabilities.max_image_extent[0]);
            actual_extent[1] =
                actual_extent[1].clamp(actual_extent[1], capabilities.max_image_extent[1]);

            actual_extent
        }
    }

    fn create_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_devince_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
        old_swapchain: Option<Arc<Swapchain<Window>>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_devince_index).unwrap();
        let capabilities = surface
            .capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some()
            && image_count > capabilities.max_image_count.unwrap()
        {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let indices = Self::find_queue_families(&surface, &physical_device);

        let sharing: SharingMode = if indices.graphics_family != indices.present_family {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };

        let swapchain_builder = if let Some(old) = old_swapchain {
            old.recreate()
        } else {
            Swapchain::start(device.clone(), surface.clone())
        };

        swapchain_builder
            .num_images(image_count)
            .format(surface_format.0)
            .dimensions(extent)
            .layers(1)
            .usage(image_usage)
            .sharing_mode(sharing)
            .transform(capabilities.current_transform)
            .present_mode(present_mode)
            .fullscreen_exclusive(FullscreenExclusive::Default)
            .clipped(true)
            .color_space(surface_format.1)
            .build()
            .expect("Failed to create swap chain")
    }

    fn create_render_pass(device: &Arc<Device>, color_format: Format) -> Arc<RenderPass> {
        Arc::new(
            single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            })
            .unwrap(),
        )
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<RenderPass>,
    ) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        mod vertex_shader {
            vulkano_shaders::shader! {
                ty: "vertex",
                path: "src/bin/shader_base.vert"
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "src/bin/shader_base.frag"
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone())
            .expect("failed to create vertex shader module");
        let frag_shader_module = fragment_shader::Shader::load(device.clone())
            .expect("failed to create fragment shader module");

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vert_shader_module.main_entry_point(), ())
                .triangle_list()
                .primitive_restart(false)
                .viewports(vec![viewport]) // NOTE: Also sets scissor to cover whole viewport
                .fragment_shader(frag_shader_module.main_entry_point(), ())
                .depth_clamp(false)
                .polygon_mode_fill()
                .line_width(1.0)
                .cull_mode_disabled()
                .front_face_clockwise()
                .blend_pass_through()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(device.clone())
                .unwrap(),
        )
    }

    fn create_vertex_buffer(graphics_queue: &Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            vertices().iter().cloned(),
            BufferUsage::vertex_buffer(),
            graphics_queue.clone(),
        )
        .unwrap();

        future.flush().unwrap();

        buffer
    }

    fn create_index_buffer(
        graphics_queue: &Arc<Queue>,
    ) -> Arc<dyn TypedBufferAccess<Content = [u16]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            indices().iter().cloned(),
            BufferUsage::index_buffer(),
            graphics_queue.clone(),
        )
        .unwrap();
        future.flush().unwrap();
        buffer
    }

    fn create_uniform_buffers(
        device: &Arc<Device>,
        num_buffers: usize,
        start_time: Instant,
        dimensions_u32: [u32; 2],
    ) -> Vec<Arc<CpuAccessibleBuffer<UniformBufferObject>>> {
        let mut buffers = Vec::new();

        let dimensions = [dimensions_u32[0] as f32, dimensions_u32[1] as f32];
        let uniform_buffer = Self::update_uniform_buffer(start_time, dimensions);
        for _ in 0..num_buffers {
            let buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false,
                uniform_buffer,
            )
            .unwrap();

            buffers.push(buffer);
        }

        buffers
    }

    fn create_framebuffers(
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<RenderPass>,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images
            .iter()
            .map(|image| {
                let view = ImageView::new(image.clone()).unwrap();
                let fba: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(
                    Framebuffer::start(render_pass.clone())
                        .add(view)
                        .unwrap()
                        .build()
                        .unwrap(),
                );
                fba
            })
            .collect::<Vec<_>>()
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        sync::now(device.clone()).boxed()
    }

    fn find_queue_families(
        surface: &Arc<Surface<Window>>,
        device: &PhysicalDevice,
    ) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();

        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_idx: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(instance, physical_device_idx).unwrap();
        let indices = Self::find_queue_families(&surface, &physical_device);
        let families = [indices.graphics_family, indices.present_family];

        use std::iter::FromIterator;

        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;

        let queue_families = unique_queue_families.iter().map(|i| {
            (
                physical_device.queue_families().nth(**i as usize).unwrap(),
                queue_priority,
            )
        });

        // NOTE: the tutorial recommends passing the validation layers as well
        // for legacy reasons (if ENABLE_VALIDATION_LAYERS is true). Vulkano handles that
        // for us internally.

        let (device, mut queues) = Device::new(
            physical_device,
            &Features::none(),
            &device_extensions(),
            queue_families,
        )
        .expect("Failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn create_surface(instance: &Arc<Instance>) -> (EventLoop<()>, Arc<Surface<Window>>) {
        let event_loop = EventLoop::new();

        let surface = WindowBuilder::new()
            .with_title("Vulkan Renderer")
            .with_inner_size(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
            .build_vk_surface(&event_loop, instance.clone())
            .unwrap();

        (event_loop, surface)
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            info!("Recreating swapchain");
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, is_suboptimal, acquire_future) =
            match acquire_next_image(self.swap_chain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swap_chain = true;
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if is_suboptimal {
            self.recreate_swap_chain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.graphics_queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        self.uniform_buffers = Self::create_uniform_buffers(
            &self.device,
            self.swap_chain_images.len(),
            self.start_time,
            self.swap_chain.dimensions(),
        );

        let layout = self.graphics_pipeline.descriptor_set_layout(0).unwrap();
        let set = Arc::new(
            self.descriptor_sets_pool
                .next()
                .add_buffer(self.uniform_buffers[0].clone())
                .unwrap()
                .add_sampled_image(self.image_view.clone(), self.image_sampler.clone())
                .unwrap()
                .build()
                .unwrap(),
        );

        builder
            .begin_render_pass(
                self.swap_chain_framebuffers[image_index].clone(),
                SubpassContents::Inline,
                vec![[0.0, 0.0, 0.0, 1.0].into()],
            )
            .unwrap()
            .draw_indexed(
                self.graphics_pipeline.clone(),
                &DynamicState::none(),
                vec![self.vertex_buffer.clone()],
                self.index_buffer.clone(),
                set,
                (),
                vec![],
            )
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.present_queue.clone(),
                self.swap_chain.clone(),
                image_index,
            )
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end = Some(vulkano::sync::now(self.device.clone()).boxed());
            }
        }
    }

    fn update_uniform_buffer(start_time: Instant, dimensions: [f32; 2]) -> UniformBufferObject {
        let duration = Instant::now().duration_since(start_time);
        let elapsed = (duration.as_secs() * 1000) + u64::from(duration.subsec_millis());

        let model = Matrix4::from_angle_z(Rad::from(Deg(elapsed as f32 * 0.180)));
        let view = Matrix4::look_at_rh(
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        );
        let mut proj = cgmath::perspective(
            Rad::from(Deg(45.0)),
            dimensions[0] / dimensions[1],
            0.1,
            10.0,
        );

        proj.y.y *= -1.0;

        UniformBufferObject { model, view, proj }
    }

    fn recreate_swap_chain(&mut self) {
        let (swap_chain, images) = Self::create_swap_chain(
            &self.instance,
            &self.surface,
            self.physical_device_index,
            &self.device,
            &self.graphics_queue,
            &self.present_queue,
            Some(self.swap_chain.clone()),
        );
        self.swap_chain = swap_chain;
        self.swap_chain_images = images;
        self.render_pass = Self::create_render_pass(&self.device, self.swap_chain.format());
        self.graphics_pipeline = Self::create_graphics_pipeline(
            &self.device,
            self.swap_chain.dimensions(),
            &self.render_pass,
        );
        self.swap_chain_framebuffers =
            Self::create_framebuffers(&self.swap_chain_images, &self.render_pass);
    }

    fn update_cursor(&mut self, position: [f64; 2]) {
        let prev_position = self.input.mouse.position;
        self.input.mouse.position = position;

        let prev_vec: Vector2<_> = prev_position.into();
        let curr_vec: Vector2<_> = self.input.mouse.position.into();
        let dir = prev_vec - curr_vec;

        self.input.mouse.delta = dir.into();
    }

    fn update(&mut self) {
        if self.input.mouse.right_click {
            println!("Right Click is being pressed");
        }
    }

    pub fn main_loop(mut self) {
        let event_loop = &mut self.event_loop;
        let surface_window_id = self.surface.window().id();

        self.event_loop
            .take()
            .unwrap()
            .run(move |event, _, control_flow| match event {
                Event::WindowEvent {
                    event, window_id, ..
                } => match event {
                    WindowEvent::CloseRequested { .. } if window_id == surface_window_id => {
                        *control_flow = ControlFlow::Exit
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::CursorMoved { position, .. } => {
                        self.update_cursor(position.into())
                    }
                    WindowEvent::MouseInput {
                        state,
                        button: MouseButton::Right,
                        ..
                    } => self.input.mouse.right_click = state == ElementState::Pressed,
                    _ => (),
                },

                Event::MainEventsCleared => {
                    self.update();
                    self.draw_frame();
                }
                _ => (),
            });
    }
}
