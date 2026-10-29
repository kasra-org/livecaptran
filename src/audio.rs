use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use crate::captions::Captions;
use crate::segmenter::Segmenter;
use crate::settings::Settings;
use crate::streaming::Agreement;

const PARTIAL_STEP_SECS: f32 = 1.0; // re-transcribe the growing phrase this often
const PROMPT_CHARS: usize = 200; // recent committed text sent as the Whisper prompt
const TRANSLATION_HISTORY: usize = 5; // finished sentences given to the translator as context
const MAX_SENTENCE_WORDS: usize = 30; // finish long unpunctuated runs as a sentence anyway

/// Newly committed source text for the translator.
struct TranslateUpdate {
    text: String,
    sentence_end: bool,
}

type SessionLog = Arc<Mutex<Option<std::fs::File>>>;

fn log_line(log: &SessionLog, line: &str) {
    use std::io::Write;
    if let Some(file) = log.lock().unwrap().as_mut() {
        let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(file, "[{now}] {line}");
        let _ = file.flush();
    }
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len();
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;
    let byte_rate = sample_rate * 2;

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let i = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

fn chat(
    client: &reqwest::blocking::Client,
    chat_api_url: &str,
    chat_api_key: &str,
    chat_model: &str,
    messages: Vec<serde_json::Value>,
) -> Option<String> {
    let body = serde_json::json!({
        "model": chat_model,
        "messages": messages
    });

    let mut req = client
        .post(chat_api_url)
        .header("Content-Type", "application/json");
    if !chat_api_key.is_empty() {
        req = req.bearer_auth(chat_api_key);
    }
    match req.body(body.to_string()).send() {
        Ok(resp) => {
            let body = resp.text().ok()?;
            let json: serde_json::Value = serde_json::from_str(&body).ok()?;
            Some(json["choices"][0]["message"]["content"].as_str()?.trim().to_string())
        }
        Err(e) => {
            eprintln!("Translation error: {e}");
            None
        }
    }
}

/// Translates the sentence so far, asking the model to keep `on_screen` (what viewers are
/// already reading) word for word wherever it can. Returns None if the request failed.
fn translate_sentence(
    client: &reqwest::blocking::Client,
    chat_api_url: &str,
    chat_api_key: &str,
    chat_model: &str,
    target_language: &str,
    history: &VecDeque<(String, String)>,
    source: &str,
    on_screen: &str,
    complete: bool,
) -> Option<String> {
    let system = format!(
        "You are a real-time translator for a scientific presentation. Speech arrives in pieces, so you translate each sentence while it is still being spoken. Translate into {}. Preserve technical and scientific terminology accurately. You receive the source sentence so far and the translation currently on screen. Reply with the full translation of the source so far. Viewers are already reading the on-screen translation, so keep it word for word as the start of your reply whenever possible and change a word only if the new source makes it wrong. Do not guess what the speaker has not said yet. Print only the translation and absolutely nothing else—no alternatives, explanations, notes, or quotation marks.",
        target_language
    );

    let mut user = String::new();
    if !history.is_empty() {
        user.push_str("Earlier sentences (context):\n");
        for (src, tr) in history {
            user.push_str(&format!("{src} => {tr}\n"));
        }
        user.push('\n');
    }
    let status = if complete { " (sentence complete)" } else { "" };
    user.push_str(&format!("Source so far{status}: {source}\nOn screen: {on_screen}"));

    chat(
        client,
        chat_api_url,
        chat_api_key,
        chat_model,
        vec![
            serde_json::json!({"role": "system", "content": system}),
            serde_json::json!({"role": "user", "content": user}),
        ],
    )
}

fn transcribe(
    client: &reqwest::blocking::Client,
    samples: &[f32],
    rate: u32,
    api_url: &str,
    api_key: &str,
    transcribe_model: &str,
    language: &str,
    prompt: &str,
) -> Option<String> {
    let wav = encode_wav(samples, rate);
    let mut form = reqwest::blocking::multipart::Form::new()
        .part(
            "file",
            reqwest::blocking::multipart::Part::bytes(wav)
                .file_name("audio.wav")
                .mime_str("audio/wav")
                .unwrap(),
        )
        .text("model", transcribe_model.to_string())
        .text("language", language.to_string());
    if !prompt.is_empty() {
        form = form.text("prompt", prompt.to_string());
    }

    let mut req = client.post(api_url);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }
    match req.multipart(form).send() {
        Ok(resp) => {
            let body = resp.text().ok()?;
            let json: serde_json::Value = serde_json::from_str(&body).ok()?;
            Some(json["text"].as_str()?.trim().to_string())
        }
        Err(e) => {
            eprintln!("Transcription error: {e}");
            None
        }
    }
}

/// Appends `text` to `buf` with a separating space.
fn append(buf: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if !buf.is_empty() && !buf.ends_with(char::is_whitespace) {
        buf.push(' ');
    }
    buf.push_str(text);
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '?', '!', '。', '？', '！'])
}

// Streams the translation of committed source text, off the transcription thread so it
// never delays it. The current sentence is re-translated on each update; translated words
// two consecutive results agree on are fixed, and only the dimmed tail may still change.
fn spawn_translator(
    updates: mpsc::Receiver<TranslateUpdate>,
    captions: Arc<Mutex<Captions>>,
    settings: Arc<Mutex<Settings>>,
    session_active: Arc<AtomicBool>,
    log: SessionLog,
) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");
        let mut history: VecDeque<(String, String)> = VecDeque::new();
        let mut source = String::new(); // current sentence so far
        let mut on_screen = String::new(); // its latest translation (fixed + tentative)
        let mut agreement = Agreement::default();

        while let Ok(mut update) = updates.recv() {
            // Merge queued updates so a slow API doesn't fall behind
            while let Ok(next) = updates.try_recv() {
                append(&mut update.text, &next.text);
                update.sentence_end = next.sentence_end;
            }
            if !session_active.load(Ordering::Relaxed) {
                history.clear();
                source.clear();
                on_screen.clear();
                agreement.reset();
                continue;
            }
            let (chat_api_url, chat_api_key, chat_model, target_language) = {
                let s = settings.lock().unwrap();
                (
                    s.chat_api_url.clone(),
                    s.chat_api_key.clone(),
                    s.chat_model.clone(),
                    s.target_language.clone(),
                )
            };
            if target_language.is_empty() {
                continue;
            }

            append(&mut source, &update.text);
            if source.is_empty() {
                continue;
            }
            let sentence_end =
                update.sentence_end || source.split_whitespace().count() >= MAX_SENTENCE_WORDS;
            let reply = translate_sentence(
                &client,
                &chat_api_url,
                &chat_api_key,
                &chat_model,
                &target_language,
                &history,
                &source,
                &on_screen,
                sentence_end,
            );
            if let Some(full) = reply.as_deref().filter(|t| !t.is_empty()) {
                on_screen = full.to_string();
                if !sentence_end {
                    let (new, tentative) = agreement.partial(full);
                    let mut caps = captions.lock().unwrap();
                    caps.translation.push(&new);
                    caps.translation.set_tentative(&tentative);
                }
            }

            if sentence_end {
                let mut caps = captions.lock().unwrap();
                if on_screen.is_empty() {
                    // Fall back to the original if translation failed
                    if reply.is_none() {
                        caps.translation.push(&source);
                    }
                } else {
                    caps.translation.push(&agreement.finalize(&on_screen, false));
                }
                caps.translation.set_tentative("");
                drop(caps);
                if !on_screen.is_empty() {
                    log_line(&log, &format!("→ {on_screen}"));
                    history.push_back((source.clone(), on_screen.clone()));
                    if history.len() > TRANSLATION_HISTORY {
                        history.pop_front();
                    }
                }
                source.clear();
                on_screen.clear();
                agreement.reset();
            }
        }
    });
}

pub fn start_audio_and_transcription(
    captions: Arc<Mutex<Captions>>,
    running: Arc<AtomicBool>,
    settings: Arc<Mutex<Settings>>,
    session_active: Arc<AtomicBool>,
) {
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let sample_rate: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // Audio capture thread
    {
        let buf = audio_buffer.clone();
        let sr = sample_rate.clone();
        let run = running.clone();
        let input_device_name = settings.lock().unwrap().input_device.clone();
        thread::spawn(move || {
            let host = cpal::default_host();
            let device = if input_device_name.is_empty() {
                host.default_input_device()
            } else {
                use cpal::traits::HostTrait;
                host.input_devices()
                    .ok()
                    .and_then(|mut devs| {
                        devs.find(|d| {
                            d.name().map(|n| n == input_device_name).unwrap_or(false)
                        })
                    })
                    .or_else(|| {
                        eprintln!("Device '{}' not found, using default", input_device_name);
                        host.default_input_device()
                    })
            };
            let device = match device {
                Some(d) => d,
                None => {
                    eprintln!("No audio input device found");
                    return;
                }
            };
            let supported = match device.default_input_config() {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("No input config: {e}");
                    return;
                }
            };

            *sr.lock().unwrap() = supported.sample_rate().0;
            let channels = supported.channels() as usize;
            let fmt = supported.sample_format();
            let config: cpal::StreamConfig = supported.into();

            let _stream = match fmt {
                cpal::SampleFormat::F32 => {
                    let buf = buf.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                                let mut b = buf.lock().unwrap();
                                if channels > 1 {
                                    for chunk in data.chunks(channels) {
                                        b.push(chunk.iter().sum::<f32>() / channels as f32);
                                    }
                                } else {
                                    b.extend_from_slice(data);
                                }
                            },
                            |e| eprintln!("Audio error: {e}"),
                            None,
                        )
                        .expect("Failed to build input stream")
                }
                cpal::SampleFormat::I16 => {
                    let buf = buf.clone();
                    device
                        .build_input_stream(
                            &config,
                            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                                let mut b = buf.lock().unwrap();
                                if channels > 1 {
                                    for chunk in data.chunks(channels) {
                                        let sum: f32 =
                                            chunk.iter().map(|&s| s as f32 / 32768.0).sum();
                                        b.push(sum / channels as f32);
                                    }
                                } else {
                                    for &s in data {
                                        b.push(s as f32 / 32768.0);
                                    }
                                }
                            },
                            |e| eprintln!("Audio error: {e}"),
                            None,
                        )
                        .expect("Failed to build input stream")
                }
                fmt => panic!("Unsupported sample format: {fmt:?}"),
            };

            _stream.play().expect("Failed to start audio stream");

            while run.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
            }
        });
    }

    // Streaming transcription thread: while a phrase is being spoken it is re-sent every
    // PARTIAL_STEP_SECS; words two consecutive results agree on are shown right away
    {
        let buf = audio_buffer;
        let sr = sample_rate;
        let run = running;
        let log: SessionLog = Arc::new(Mutex::new(None));
        let (translate_tx, translate_rx) = mpsc::channel::<TranslateUpdate>();
        spawn_translator(
            translate_rx,
            captions.clone(),
            settings.clone(),
            session_active.clone(),
            log.clone(),
        );
        thread::spawn(move || {
            let client = reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .expect("Failed to build HTTP client");

            thread::sleep(Duration::from_secs(1));

            let mut segmenter: Option<Segmenter> = None;
            let mut agreement = Agreement::default();
            let mut last_partial_len = 0usize;
            let mut context = String::new();
            let mut was_session_active = false;

            while run.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(50));

                let rate = *sr.lock().unwrap();
                if rate == 0 {
                    continue;
                }

                let new_samples: Vec<f32> = {
                    let mut b = buf.lock().unwrap();
                    std::mem::take(&mut *b)
                };

                if new_samples.is_empty() {
                    continue;
                }

                // Session state transitions
                let is_active = session_active.load(Ordering::Relaxed);
                if is_active && !was_session_active {
                    let dir = crate::settings::sessions_dir();
                    let filename = format!(
                        "session_{}.txt",
                        chrono::Local::now().format("%Y-%m-%d_%H-%M-%S")
                    );
                    match std::fs::File::create(dir.join(&filename)) {
                        Ok(f) => *log.lock().unwrap() = Some(f),
                        Err(e) => eprintln!("Failed to create session log: {e}"),
                    }
                    was_session_active = true;
                } else if !is_active && was_session_active {
                    *log.lock().unwrap() = None;
                    captions.lock().unwrap().clear();
                    if let Some(seg) = &mut segmenter {
                        seg.reset();
                    }
                    agreement.reset();
                    last_partial_len = 0;
                    context.clear();
                    // Lets the translator drop its unfinished sentence (it sees the session ended)
                    let _ = translate_tx.send(TranslateUpdate {
                        text: String::new(),
                        sentence_end: false,
                    });
                    was_session_active = false;
                }

                if !is_active {
                    continue;
                }

                let (threshold, min_phrase_secs, max_phrase_secs, api_url, api_key, transcribe_model, language, translating) = {
                    let s = settings.lock().unwrap();
                    (
                        s.silence_threshold,
                        s.min_phrase_secs,
                        s.max_phrase_secs,
                        s.api_url.clone(),
                        s.api_key.clone(),
                        s.transcribe_model.clone(),
                        s.language.clone(),
                        !s.target_language.is_empty(),
                    )
                };
                let transcribe = |samples: &[f32], prompt: &str| {
                    transcribe(
                        &client,
                        samples,
                        rate,
                        &api_url,
                        &api_key,
                        &transcribe_model,
                        &language,
                        prompt,
                    )
                };
                let commit = |text: &str, sentence_end: bool, context: &mut String| {
                    if !text.trim().is_empty() {
                        captions.lock().unwrap().source.push(text);
                        append(context, text);
                        if context.len() > PROMPT_CHARS {
                            let mut cut = context.len() - PROMPT_CHARS;
                            while !context.is_char_boundary(cut) {
                                cut += 1;
                            }
                            context.drain(..cut);
                        }
                    }
                    if translating && (!text.trim().is_empty() || sentence_end) {
                        let _ = translate_tx.send(TranslateUpdate {
                            text: text.to_string(),
                            sentence_end,
                        });
                    }
                };

                let seg = segmenter.get_or_insert_with(|| Segmenter::new(rate));

                // Finished phrases: final transcription commits whatever is left
                for phrase in seg.push(&new_samples, threshold, min_phrase_secs, max_phrase_secs) {
                    let text = transcribe(&phrase.samples, &context)
                        .unwrap_or_else(|| agreement.last());
                    log_line(&log, &text);
                    let rest = agreement.finalize(&text, phrase.continues);
                    // A pause ends the sentence for the translator; a mid-speech cut doesn't
                    commit(&rest, !phrase.continues || ends_sentence(&rest), &mut context);
                    captions.lock().unwrap().source.set_tentative("");
                    last_partial_len = 0;
                }

                // Phrase in progress: re-transcribe once enough new audio has arrived
                let step = (rate as f32 * PARTIAL_STEP_SECS) as usize;
                if let Some(current) = seg.current() {
                    if current.len() >= last_partial_len + step {
                        last_partial_len = current.len();
                        if let Some(text) = transcribe(current, &context) {
                            let (new, tentative) = agreement.partial(&text);
                            commit(&new, ends_sentence(&new), &mut context);
                            captions.lock().unwrap().source.set_tentative(&tentative);
                        }
                    }
                }
            }
        });
    }
}
