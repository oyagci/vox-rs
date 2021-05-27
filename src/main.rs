mod hello;

fn main() {
    let mut app = hello::HelloWorldApplication::new();
    app.main_loop();
}
