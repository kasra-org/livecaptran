# LiveCapTran

Live Korean speech caption/transcription overlay app.

## Project Structure

```
livecaptran/
├── assets/           # Icons and fonts
│   ├── NotoSansKR-Regular.ttf
│   ├── cog.png
│   └── close.png
├── src/              # Rust egui desktop app (frameless overlay window)
│   ├── main.rs
│   ├── app.rs
│   ├── audio.rs      # mic capture + transcription/translation requests
│   ├── captions.rs   # rolling word-by-word captions
│   ├── headless.rs   # --headless: print caption changes instead of a window (testing)
│   ├── segmenter.rs  # splits speech into 2–8s phrases at pauses
│   ├── settings.rs
│   ├── streaming.rs  # commits words two consecutive partial transcriptions agree on
│   └── topmost.rs    # one overlay per monitor, kept above fullscreen presentations (macOS/Windows)
├── Cargo.toml
└── .github/workflows/
    └── build.yml     # CI: builds for Linux, Windows, macOS
```

## Transcription API

- **Type**: OpenAI-compatible Whisper API (faster-whisper-server)
- Configured via `settings.yml` (defaults to OpenAI endpoints)

### Endpoints

#### GET /v1/models
Lists available Whisper models.

**Available models**: tiny, small, base, medium, large, large-v2, large-v3, distil-small.en, distil-medium.en, distil-large-v2, distil-large-v3

#### POST /v1/audio/transcriptions
Transcribes audio file to text.

**Request**: `multipart/form-data`
- `file`: audio file (WAV, etc.)
- `model`: model id (use `large-v3` for Korean)
- `language`: language code (`ko` for Korean)

**Response**: `{"text": "transcribed text"}`

## GUI Spec

- Frameless overlay window, screen-wide (left=0), height=500px (resizable down to 40px; font shrinks to fit one line)
- Black semi-transparent background
- White text, 60pt, rolling captions revealed word by word (older lines scroll off the top)
- One overlay per monitor (macOS/Windows), each snapped to the bottom of its monitor at full width; windows titled "LiveCapTran", "LiveCapTran 2", …
- Stays above fullscreen presentations: macOS joins all Spaces (app runs as Accessory, no Dock icon); Windows re-asserts topmost
- Small X close button at top-right
- Draggable via mouse drag
- Captures live microphone audio, sends chunks to transcription API, displays results in real-time
- Phrases are cut at pauses between min/max length (settings), forced at the quietest point when max is reached
- Streaming: the phrase in progress is re-transcribed every 1s; new words flow in right away, dimmed until two consecutive transcriptions agree, then turn solid in place (nothing already shown is hidden and re-revealed)
- Vocabulary hints (settings) go to Whisper as `prompt` and to the translator; output that just recites the terms is dropped
- Translation streams too: the sentence so far is re-translated on each update, asking the model to keep the on-screen words; translated words two consecutive results agree on are fixed, only the dimmed tail can change

## Testing

- `--input file.wav` plays a 16-bit PCM WAV in real time instead of the mic; `--headless` prints caption changes to stdout
- Test servers: transcription `http://10.125.208.42:8000/v1/audio/transcriptions` (model `large-v3`), chat `http://10.125.208.42:8080/v1/chat/completions` (model `gemma-4-26b-a4b-it`)

## Build

```sh
cargo build --release
```

Binary name: `livecaptran`
