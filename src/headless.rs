// Runs the pipeline without a window and prints caption changes, for testing without a GUI.
// Each line shows newly fixed words (+) and the current tentative tail (~).

use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

use crate::audio::{read_wav, start_audio_and_transcription};
use crate::captions::{Caption, Captions};
use crate::settings::Settings;

struct Seen {
    fixed_len: usize,
    tentative: String,
}

fn report(label: &str, caption: &Caption, seen: &mut Seen, elapsed: f32) {
    let (fixed, tentative) = caption.parts();
    // Older text may be trimmed from the front, so fall back to printing the whole thing
    let added = fixed.get(seen.fixed_len..).unwrap_or(&fixed).trim();
    if !added.is_empty() || tentative != seen.tentative {
        println!("[{elapsed:6.1}s] {label} + {added}  ~ {tentative}");
    }
    seen.fixed_len = fixed.len();
    seen.tentative = tentative;
}

pub fn run(input: Option<PathBuf>) {
    let settings = Arc::new(Mutex::new(Settings::load()));
    let captions = Arc::new(Mutex::new(Captions::default()));
    let running = Arc::new(AtomicBool::new(true));
    let session_active = Arc::new(AtomicBool::new(true));

    // With a file, stop a while after it ends; with the microphone, run until killed
    let deadline = input
        .as_deref()
        .and_then(|p| read_wav(p).ok())
        .map(|(samples, rate)| samples.len() as f32 / rate as f32 + 20.0);

    start_audio_and_transcription(
        captions.clone(),
        running.clone(),
        settings,
        session_active,
        input,
    );

    let start = Instant::now();
    let mut source = Seen { fixed_len: 0, tentative: String::new() };
    let mut translation = Seen { fixed_len: 0, tentative: String::new() };
    loop {
        thread::sleep(Duration::from_millis(100));
        let elapsed = start.elapsed().as_secs_f32();
        {
            let mut caps = captions.lock().unwrap();
            caps.tick(f32::MAX); // reveal everything at once
            report("SRC", &caps.source, &mut source, elapsed);
            report("TR ", &caps.translation, &mut translation, elapsed);
        }
        if deadline.is_some_and(|d| elapsed > d) {
            break;
        }
    }
    running.store(false, Ordering::Relaxed);

    let caps = captions.lock().unwrap();
    println!("\n=== Source ===\n{}", caps.source.parts().0);
    println!("\n=== Translation ===\n{}", caps.translation.parts().0);
}
