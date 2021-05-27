mod hello;

use log::{debug, error, info, log_enabled, Level};

fn main() {
    env_logger::init();

    info!("Staring Vulkan Renderer...");

    let mut app = hello::HelloWorldApplication::new();
    app.main_loop();
}
