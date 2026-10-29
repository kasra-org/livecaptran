use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::{HashSet, VecDeque};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use crate::captions::Captions;
use crate::segmenter::Segmenter;
use crate::settings::Settings;
use crate::streaming::Agreement;

const PARTIAL_STEP_SECS: f32 = 1.0; // re-transcribe the growing phrase this often
const FIRST_PARTIAL_SECS: f32 = 2.0; // shorter clips make Whisper guess stock phrases ("감사합니다.")
const TRANSLATION_HISTORY: usize = 5; // finished sentences given to the translator as context
const MAX_SENTENCE_WORDS: usize = 30; // finish long unpunctuated runs as a sentence anyway
const IDLE_FLUSH_SECS: u64 = 2; // after this long without new words, fix the translation so far

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

/// Reads a 16-bit PCM WAV file as mono samples.
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a WAV file".into());
    }
    let (mut channels, mut rate, mut bits) = (0usize, 0u32, 0u16);
    let mut pos = 12;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &bytes[pos + 8..(pos + 8 + size).min(bytes.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                channels = u16::from_le_bytes([body[2], body[3]]) as usize;
                rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                bits = u16::from_le_bytes([body[14], body[15]]);
            }
            b"data" => {
                if bits != 16 || channels == 0 {
                    return Err("only 16-bit PCM WAV is supported".into());
                }
                let samples = body
                    .chunks_exact(2 * channels)
                    .map(|frame| {
                        frame
                            .chunks_exact(2)
                            .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                            .sum::<f32>()
                            / channels as f32
                    })
                    .collect();
                return Ok((samples, rate));
            }
            _ => {}
        }
        pos += 8 + size + (size & 1);
    }
    Err("no audio data in WAV file".into())
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
    vocabulary: &str,
    history: &VecDeque<(String, String)>,
    source: &str,
    on_screen: &str,
    complete: bool,
) -> Option<String> {
    let mut system = format!(
        "You are a real-time translator for a scientific presentation. Speech arrives in pieces, so you translate each sentence while it is still being spoken. Translate into {}. Preserve technical and scientific terminology accurately. You receive the source sentence so far and the translation currently on screen. Reply with the full translation of the source so far. Viewers are already reading the on-screen translation, so keep it word for word as the start of your reply whenever possible and change a word only if the new source makes it wrong. Do not guess what the speaker has not said yet. Print only the translation and absolutely nothing else—no alternatives, explanations, notes, or quotation marks.",
        target_language
    );
    if !vocabulary.trim().is_empty() {
        system.push_str(&format!(" Terms that may appear: {}.", vocabulary.trim()));
    }

    let mut user = String::new();
    if !history.is_empty() {
        user.push_str("Earlier sentences, already on screen (context only, do not translate them again):\n");
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
    vocabulary: &str,
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
    // A fixed term list biases recognition without the echo a rolling transcript causes
    if !vocabulary.trim().is_empty() {
        form = form.text("prompt", vocabulary.trim().to_string());
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

/// Whisper sometimes loops on short or unclear audio ("네. 네. 네. …", counting up).
fn looks_hallucinated(text: &str, secs: f32) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    // Far more words than anyone says in that time
    if words.len() as f32 > secs * 6.0 + 3.0 {
        return true;
    }
    // Mostly the same words over and over
    let unique: HashSet<&str> = words.iter().copied().collect();
    if words.len() >= 6 && (unique.len() as f32) < words.len() as f32 * 0.5 {
        return true;
    }
    // A short pattern looping, possibly without spaces ("블렉블렉블렉…", "100,100,100,…")
    let c: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    (1..=6).any(|len| {
        (0..c.len()).any(|i| {
            let pattern = &c[i..(i + len).min(c.len())];
            let repeats = (1..)
                .take_while(|k| {
                    let j = i + k * len;
                    j + len <= c.len() && c[j..j + len] == *pattern
                })
                .count();
            repeats >= 5 && pattern.iter().any(|ch| ch.is_alphanumeric())
        })
    })
}

/// With a vocabulary prompt, Whisper may recite the terms on unclear audio
/// ("사이브, 도너, 샘플, …"), so drop output made up mostly of the terms themselves.
fn echoes_vocabulary(text: &str, vocabulary: &str) -> bool {
    let terms: HashSet<String> = vocabulary.split(',').map(normalize).filter(|t| !t.is_empty()).collect();
    let words: Vec<String> = text.split_whitespace().map(normalize).filter(|w| !w.is_empty()).collect();
    if terms.is_empty() || words.len() < 3 {
        return false;
    }
    let exact = words.iter().filter(|w| terms.contains(*w)).count();
    // A comma-separated list of terms or their variants ("아카이브, 도너스, 아카이브,")
    let listed = text.matches(',').count() + 1 >= words.len();
    let variants = words
        .iter()
        .filter(|w| terms.iter().any(|t| w.starts_with(t.as_str())))
        .count();
    exact as f32 >= words.len() as f32 * 0.7 || (listed && variants * 2 >= words.len())
}

fn normalize(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()
}

/// Some models start the reply with the previous sentence's translation, which they
/// were given only as context; strip it when most of it is repeated word for word.
fn strip_repeated_context(reply: &str, previous: &str) -> String {
    let previous: Vec<String> = previous.split_whitespace().map(normalize).collect();
    let words: Vec<&str> = reply.split_whitespace().collect();
    let matched = previous
        .iter()
        .zip(&words)
        .take_while(|(p, w)| **p == normalize(w))
        .count();
    if matched >= 4 && matched * 2 >= previous.len() {
        words[matched..].join(" ")
    } else {
        reply.to_string()
    }
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
        let mut flushed = false; // idle flush already done since the last update

        loop {
            // A pause mid-sentence fixes what has been translated so far, but keeps the
            // sentence open: closing it would make the next clause re-translate its subject
            let (mut update, idle) =
                match updates.recv_timeout(Duration::from_secs(IDLE_FLUSH_SECS)) {
                    Ok(update) => (update, false),
                    Err(mpsc::RecvTimeoutError::Timeout) if !source.is_empty() && !flushed => {
                        let update = TranslateUpdate {
                            text: String::new(),
                            sentence_end: false,
                        };
                        (update, true)
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                };
            flushed = idle;
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
            let (chat_api_url, chat_api_key, chat_model, target_language, vocabulary) = {
                let s = settings.lock().unwrap();
                (
                    s.chat_api_url.clone(),
                    s.chat_api_key.clone(),
                    s.chat_model.clone(),
                    s.target_language.clone(),
                    s.vocabulary.clone(),
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
                &vocabulary,
                &history,
                &source,
                &on_screen,
                sentence_end || idle,
            );
            let reply = match (reply, history.back()) {
                (Some(r), Some((_, previous))) => Some(strip_repeated_context(&r, previous)),
                (r, _) => r,
            };
            if let Some(full) = reply.as_deref().filter(|t| !t.is_empty()) {
                on_screen = full.to_string();
                if !sentence_end {
                    let (mut new, mut tentative) = agreement.partial(full);
                    if idle {
                        // Feeding the same result again agrees with it, fixing the tail
                        new.push_str(&agreement.partial(full).0);
                        tentative.clear();
                    }
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
                    caps.translation.push(&agreement.finalize(Some(&on_screen), false));
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
    input: Option<PathBuf>,
) {
    let audio_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let sample_rate: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    if let Some(path) = input {
        // Feed a WAV file in real time instead of the microphone, starting with the session
        let buf = audio_buffer.clone();
        let sr = sample_rate.clone();
        let run = running.clone();
        let active = session_active.clone();
        thread::spawn(move || {
            let (samples, rate) = match read_wav(&path) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Failed to read {}: {e}", path.display());
                    return;
                }
            };
            *sr.lock().unwrap() = rate;
            while !active.load(Ordering::Relaxed) {
                if !run.load(Ordering::Relaxed) {
                    return;
                }
                buf.lock().unwrap().extend(std::iter::repeat_n(0.0, rate as usize / 10));
                thread::sleep(Duration::from_millis(100));
            }
            let chunk = rate as usize / 20;
            let start = Instant::now();
            // Trailing silence lets the last phrase finish
            let silence = vec![0.0; rate as usize * 2];
            for (i, c) in samples.chunks(chunk).chain(silence.chunks(chunk)).enumerate() {
                if !run.load(Ordering::Relaxed) {
                    return;
                }
                buf.lock().unwrap().extend_from_slice(c);
                let due = Duration::from_secs_f64((i + 1) as f64 * chunk as f64 / rate as f64);
                if let Some(wait) = due.checked_sub(start.elapsed()) {
                    thread::sleep(wait);
                }
            }
        });
    } else {
        // Audio capture thread
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

                let (threshold, min_phrase_secs, max_phrase_secs, api_url, api_key, transcribe_model, language, vocabulary, translating) = {
                    let s = settings.lock().unwrap();
                    (
                        s.silence_threshold,
                        s.min_phrase_secs,
                        s.max_phrase_secs,
                        s.api_url.clone(),
                        s.api_key.clone(),
                        s.transcribe_model.clone(),
                        s.language.clone(),
                        s.vocabulary.clone(),
                        !s.target_language.is_empty(),
                    )
                };
                let transcribe = |samples: &[f32]| {
                    let text = transcribe(
                        &client,
                        samples,
                        rate,
                        &api_url,
                        &api_key,
                        &transcribe_model,
                        &language,
                        &vocabulary,
                    )?;
                    let secs = samples.len() as f32 / rate as f32;
                    if looks_hallucinated(&text, secs) || echoes_vocabulary(&text, &vocabulary) {
                        eprintln!("Dropped likely hallucination: {text}");
                        return None;
                    }
                    Some(text)
                };
                let commit = |text: &str, sentence_end: bool| {
                    if !text.trim().is_empty() {
                        captions.lock().unwrap().source.push(text);
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
                    let text = transcribe(&phrase.samples);
                    if let Some(text) = &text {
                        log_line(&log, text);
                    }
                    let rest = agreement.finalize(text.as_deref(), phrase.continues);
                    // Phrases often end mid-sentence, so only punctuation ends one here
                    commit(&rest, ends_sentence(text.as_deref().unwrap_or(&rest)));
                    captions.lock().unwrap().source.set_tentative("");
                    last_partial_len = 0;
                }

                // Phrase in progress: re-transcribe once enough new audio has arrived
                let step = (rate as f32 * PARTIAL_STEP_SECS) as usize;
                let first = (rate as f32 * FIRST_PARTIAL_SECS) as usize;
                if let Some(current) = seg.current() {
                    if current.len() >= (last_partial_len + step).max(first) {
                        last_partial_len = current.len();
                        if let Some(text) = transcribe(current) {
                            let (new, tentative) = agreement.partial(&text);
                            commit(&new, ends_sentence(&new));
                            captions.lock().unwrap().source.set_tentative(&tentative);
                        }
                    }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_hallucinations() {
        assert!(looks_hallucinated(&"네. ".repeat(27), 2.0));
        assert!(looks_hallucinated(&(1..=100).map(|n| n.to_string()).collect::<Vec<_>>().join(", "), 8.0));
        assert!(looks_hallucinated("tml블렉블렉블렉블렉블렉블렉블렉블렉", 2.0));
        assert!(!looks_hallucinated("사이노스 아카이트 샘플이 22년도에 모은 샘플들하고 24년도에 모은 샘플들하고", 5.0));
        assert!(!looks_hallucinated("네 맞습니다... 지금 회사로 보낸", 2.0));
        assert!(!looks_hallucinated("We produced the data while looking at what was sent to the company.", 4.0));
    }

    #[test]
    fn detects_vocabulary_echo() {
        let vocab = "TMA, 사이노스, 도너, 샘플, 블록";
        assert!(echoes_vocabulary("사이브, 도너, 샘플, 샘플.", vocab));
        assert!(echoes_vocabulary("TMA, 도너, 블록", vocab));
        assert!(echoes_vocabulary("아카이브, 도너스, 아카이브,", "아카이브, 도너"));
        assert!(!echoes_vocabulary("TMA 블록이 있는데", vocab));
        assert!(!echoes_vocabulary("TMA 블록이 총 두 가지가 있는데", vocab));
        assert!(!echoes_vocabulary("TMA 블록", vocab));
        assert!(!echoes_vocabulary("사이브, 도너, 샘플", ""));
    }

    #[test]
    fn strips_repeated_previous_translation() {
        let previous = "This is not data we created ourselves, and when producing those CosMx...";
        assert_eq!(
            strip_repeated_context(
                "This is not data we created ourselves, and when producing those CosMx, it is unclear.",
                previous
            ),
            "it is unclear."
        );
        // A reply that merely starts similarly is kept
        assert_eq!(
            strip_repeated_context("This is not what we expected.", previous),
            "This is not what we expected."
        );
    }
}
