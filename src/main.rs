#![windows_subsystem = "windows"]

mod app;
mod audio;
mod captions;
mod headless;
mod segmenter;
mod settings;
mod streaming;
mod topmost;

use eframe::egui;
use std::path::PathBuf;

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    // --input <file.wav>: play a 16-bit PCM WAV in real time instead of the microphone
    let input = args
        .iter()
        .position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    // --headless: no window, print caption changes to stdout (for testing)
    if args.iter().any(|a| a == "--headless") {
        headless::run(input);
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([3840.0, 500.0])
            .with_min_inner_size([200.0, 40.0])
            .with_position([0.0, 0.0])
            .with_decorations(false)
            .with_transparent(true)
            .with_resizable(true)
            .with_always_on_top(),
        // Accessory apps (no Dock icon) can show windows over other apps' fullscreen Spaces
        #[cfg(target_os = "macos")]
        event_loop_builder: Some(Box::new(
            |builder: &mut eframe::EventLoopBuilder<eframe::UserEvent>| {
                use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
                builder.with_activation_policy(ActivationPolicy::Accessory);
            },
        )),
        ..Default::default()
    };

    eframe::run_native(
        "LiveCapTran",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc, input)))),
    )
}
