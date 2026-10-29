# LiveCapTran

Live Korean speech transcription/translation overlay app. Captures microphone audio, detects phrases via voice activity detection, sends them to an OpenAI-compatible Whisper API, optionally translates via a Chat API, and displays results in a frameless always-on-top window.

## Features

- Real-time speech transcription using Whisper API
- Optional translation via Chat completions API (with scientific terminology preservation)
- Configurable via `settings.yml` (created next to the binary on first run)
- Separate settings window with API URLs, API keys, language selection, and more
- Frameless, draggable, resizable overlay window
- Rolling captions that flow in word by word
- Stays visible over fullscreen presentations (macOS Spaces / Windows topmost). On macOS the app has no Dock icon.
- Supports 10 source languages (transcription) and 50+ target languages (translation)

## Prerequisites

- [Rust](https://rustup.rs/) (1.85+ for edition 2024)
- An OpenAI-compatible Whisper API endpoint

### Linux

```bash
sudo apt-get install -y pkg-config libasound2-dev libssl-dev \
  libgtk-3-dev libxcb-render0-dev libxcb-shape0-dev libxcb-xfixes0-dev libxkbcommon-dev
```

### macOS

No extra dependencies. CoreAudio and Security Framework ship with the OS.

### Windows

No extra dependencies. WASAPI and SChannel are built-in.

## Build

```bash
cargo build --release
```

Binary outputs to `target/release/livecaptran` (or `livecaptran.exe` on Windows).

## Configuration

On first run, a `settings.yml` file is created next to the binary with default settings (OpenAI API endpoints). Edit this file or use the in-app settings window (gear icon) to configure:

- **Transcribe API URL / Key / Model** - Whisper-compatible transcription endpoint (`whisper-1` for OpenAI, e.g. `large-v3` for faster-whisper-server)
- **Chat API URL / Key / Model** - Chat completions endpoint for translation
- **Source language** - Language being spoken
- **Target language** - Translation target (or "None" to disable)
- **Display mode** - Show both transcription + translation, or translation only
- **Font size** and **VAD sensitivity**
- **Min / Max phrase length** - continuous speech is cut at short pauses within this range

## CI

GitHub Actions builds for all three platforms on push to `main`. See `.github/workflows/build.yml`. Download artifacts from the Actions tab (kept for 90 days).

Pushing a `v*` tag also publishes the archives as a GitHub Release:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

| Platform | Target | Archive |
|----------|--------|---------|
| Linux | `x86_64-unknown-linux-gnu` | `livecaptran-linux-x86_64.tar.gz` |
| Windows | `x86_64-pc-windows-msvc` | `livecaptran-windows-x86_64.zip` |
| macOS | universal (`aarch64` + `x86_64`) | `livecaptran-macos-universal.tar.gz` |

The macOS binary is ad-hoc signed only. On first launch, right-click → Open, or run `xattr -d com.apple.quarantine livecaptran`.
