extern crate winit;

use log::{debug, error, log_enabled, info, Level};

use std::collections::HashSet;
use std::sync::Arc;

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::Format;
use vulkano::image::{swapchain::SwapchainImage, ImageUsage};
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::instance::{
    layers_list, ApplicationInfo, Instance, InstanceExtensions, PhysicalDevice, Version,
};
use vulkano::swapchain::{
    Capabilities, ColorSpace, CompositeAlpha, PresentMode, SupportedPresentModes, Surface,
    Swapchain, FullscreenExclusive,
};
use vulkano::sync::SharingMode;

use vulkano_win::VkSurfaceBuild;

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

pub struct HelloWorldApplication {
    event_loop: EventLoop<()>,
    surface: Arc<Surface<Window>>,
    instance: Arc<Instance>,

    debug_callback: Option<DebugCallback>,

    physical_device_index: usize,
    device: Arc<Device>,

    graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,
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
        );

        Self {
            event_loop,
            surface,
            instance,
            debug_callback,
            physical_device_index,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,
        }
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

        let (swap_chain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            image_count,
            surface_format.0,
            extent,
            1,
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            FullscreenExclusive::Default,
            true,
            surface_format.1,
        )
        .expect("Failed to create swap chain");

        (swap_chain, images)
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

    pub fn main_loop(self) {
        let event_loop = self.event_loop;
        let surface_window_id = self.surface.window().id();

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == surface_window_id => *control_flow = ControlFlow::Exit,
                _ => (),
            }
        });
    }
}
