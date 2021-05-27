mod hello;

use log::{debug, error, log_enabled, info, Level};

fn main() {
    env_logger::init();

    info!("Staring Vulkan Renderer...");

    let mut app = hello::HelloWorldApplication::new();
    app.main_loop();
}
