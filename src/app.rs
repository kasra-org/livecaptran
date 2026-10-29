use cpal::traits::{DeviceTrait, HostTrait};
use eframe::egui;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use crate::audio::start_audio_and_transcription;
use crate::captions::Captions;
use crate::settings::{DisplayMode, Settings, SOURCE_LANGUAGES, TARGET_LANGUAGES};

fn setup_korean_fonts(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "noto_sans_kr".to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!(
            "../assets/NotoSansKR-Regular.ttf"
        ))),
    );
    fonts
        .families
        .entry(egui::FontFamily::Proportional)
        .or_default()
        .insert(0, "noto_sans_kr".to_owned());
    fonts
        .families
        .entry(egui::FontFamily::Monospace)
        .or_default()
        .insert(0, "noto_sans_kr".to_owned());
    ctx.set_fonts(fonts);
}

fn load_icon(ctx: &egui::Context, name: &str, png_bytes: &[u8]) -> egui::TextureHandle {
    let img = image::load_from_memory(png_bytes).expect("Failed to decode icon");
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba
        .pixels()
        .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
        .collect();
    ctx.load_texture(
        name,
        egui::ColorImage { size, pixels },
        egui::TextureOptions::LINEAR,
    )
}

const RESIZE_BORDER: f32 = 8.0;

/// Draws wrapped text in `rect`, bottom-aligned, keeping only the newest lines that fit
/// so older lines scroll off the top. Unconfirmed `tentative` words follow, dimmed.
fn paint_rolling(
    ui: &egui::Ui,
    rect: egui::Rect,
    (text, tentative): (&str, &str),
    font: egui::FontId,
    color: egui::Color32,
) {
    let text = if text.is_empty() && tentative.is_empty() { "..." } else { text };
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = rect.width();
    job.append(text, 0.0, egui::TextFormat::simple(font.clone(), color));
    if !tentative.is_empty() {
        let sep = if text.is_empty() || text.ends_with(char::is_whitespace) { "" } else { " " };
        job.append(
            &format!("{sep}{tentative}"),
            0.0,
            egui::TextFormat::simple(font, color.gamma_multiply(0.5)),
        );
    }
    let galley = ui.fonts(|f| f.layout_job(job));
    let excess = galley.size().y - rect.height();
    // Scroll by whole lines so the top line isn't cut in half
    let shift = if excess > 0.0 {
        galley
            .rows
            .iter()
            .map(|r| r.rect.top())
            .find(|&top| top >= excess)
            .unwrap_or(excess)
    } else {
        0.0
    };
    let pos = egui::pos2(rect.left(), rect.bottom() - (galley.size().y - shift));
    ui.painter().with_clip_rect(rect).galley(pos, galley, color);
}

fn detect_resize_direction(ctx: &egui::Context) -> Option<egui::ResizeDirection> {
    let rect = ctx.input(|i| i.screen_rect());
    let pos = ctx.input(|i| i.pointer.hover_pos())?;

    let at_left = pos.x <= rect.left() + RESIZE_BORDER;
    let at_right = pos.x >= rect.right() - RESIZE_BORDER;
    let at_top = pos.y <= rect.top() + RESIZE_BORDER;
    let at_bottom = pos.y >= rect.bottom() - RESIZE_BORDER;

    match (at_left, at_right, at_top, at_bottom) {
        (true, _, true, _) => Some(egui::ResizeDirection::NorthWest),
        (true, _, _, true) => Some(egui::ResizeDirection::SouthWest),
        (_, true, true, _) => Some(egui::ResizeDirection::NorthEast),
        (_, true, _, true) => Some(egui::ResizeDirection::SouthEast),
        (true, _, _, _) => Some(egui::ResizeDirection::West),
        (_, true, _, _) => Some(egui::ResizeDirection::East),
        (_, _, true, _) => Some(egui::ResizeDirection::North),
        (_, _, _, true) => Some(egui::ResizeDirection::South),
        _ => None,
    }
}

fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devs| {
            devs.filter_map(|d| d.name().ok())
                .collect()
        })
        .unwrap_or_default()
}

pub struct App {
    captions: Arc<Mutex<Captions>>,
    last_frame: Instant,
    settings: Arc<Mutex<Settings>>,
    running: Arc<AtomicBool>,
    session_active: Arc<AtomicBool>,
    positioned: bool,
    show_settings: bool,
    edit_api_url: String,
    edit_api_key: String,
    edit_transcribe_model: String,
    edit_threshold: f32,
    edit_min_phrase: f32,
    edit_max_phrase: f32,
    edit_language: String,
    edit_font_size: f32,
    edit_chat_api_url: String,
    edit_chat_api_key: String,
    edit_chat_model: String,
    edit_target_language: String,
    edit_display_mode: DisplayMode,
    edit_opacity: u8,
    edit_input_device: String,
    input_devices: Vec<String>,
    cog_icon: egui::TextureHandle,
    close_icon: egui::TextureHandle,
}

impl App {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        setup_korean_fonts(&cc.egui_ctx);

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = egui::Color32::TRANSPARENT;
        cc.egui_ctx.set_visuals(visuals);

        let loaded = Settings::load();

        let captions = Arc::new(Mutex::new(Captions::default()));
        let running = Arc::new(AtomicBool::new(true));
        let session_active = Arc::new(AtomicBool::new(false));

        let edit_api_url = loaded.api_url.clone();
        let edit_api_key = loaded.api_key.clone();
        let edit_transcribe_model = loaded.transcribe_model.clone();
        let edit_threshold = loaded.silence_threshold;
        let edit_min_phrase = loaded.min_phrase_secs;
        let edit_max_phrase = loaded.max_phrase_secs;
        let edit_language = loaded.language.clone();
        let edit_font_size = loaded.font_size;
        let edit_chat_api_url = loaded.chat_api_url.clone();
        let edit_chat_api_key = loaded.chat_api_key.clone();
        let edit_chat_model = loaded.chat_model.clone();
        let edit_target_language = loaded.target_language.clone();
        let edit_display_mode = loaded.display_mode.clone();
        let edit_opacity = loaded.opacity;
        let edit_input_device = loaded.input_device.clone();

        let input_devices = list_input_devices();

        let settings = Arc::new(Mutex::new(loaded));

        start_audio_and_transcription(
            captions.clone(),
            running.clone(),
            settings.clone(),
            session_active.clone(),
        );

        let cog_icon = load_icon(
            &cc.egui_ctx,
            "cog",
            include_bytes!("../assets/cog.png"),
        );
        let close_icon = load_icon(
            &cc.egui_ctx,
            "close",
            include_bytes!("../assets/close.png"),
        );

        Self {
            captions,
            last_frame: Instant::now(),
            settings,
            running,
            session_active,
            positioned: false,
            show_settings: false,
            edit_api_url,
            edit_api_key,
            edit_transcribe_model,
            edit_threshold,
            edit_min_phrase,
            edit_max_phrase,
            edit_language,
            edit_font_size,
            edit_chat_api_url,
            edit_chat_api_key,
            edit_chat_model,
            edit_target_language,
            edit_display_mode,
            edit_opacity,
            edit_input_device,
            input_devices,
            cog_icon,
            close_icon,
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

impl eframe::App for App {
    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.0, 0.0, 0.0, 0.0]
    }

    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        crate::topmost::keep_on_top(frame, self.show_settings);

        // Position at bottom of screen on first frame
        if !self.positioned {
            if let Some(monitor) = ctx.input(|i| i.viewport().monitor_size) {
                let window_h = 500.0;
                ctx.send_viewport_cmd(egui::ViewportCommand::OuterPosition(
                    egui::pos2(0.0, monitor.y - window_h),
                ));
                self.positioned = true;
            }
        }

        // Edge resize detection
        let resize_dir = detect_resize_direction(ctx);
        if let Some(dir) = resize_dir {
            ctx.set_cursor_icon(match dir {
                egui::ResizeDirection::North | egui::ResizeDirection::South => {
                    egui::CursorIcon::ResizeVertical
                }
                egui::ResizeDirection::East | egui::ResizeDirection::West => {
                    egui::CursorIcon::ResizeHorizontal
                }
                egui::ResizeDirection::NorthWest | egui::ResizeDirection::SouthEast => {
                    egui::CursorIcon::ResizeNwSe
                }
                egui::ResizeDirection::NorthEast | egui::ResizeDirection::SouthWest => {
                    egui::CursorIcon::ResizeNeSw
                }
            });
            if ctx.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary)) {
                ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(dir));
            }
        }

        // Settings window (separate OS window)
        if self.show_settings {
            let close_req = std::cell::Cell::new(false);

            let edit_api_url = &mut self.edit_api_url;
            let edit_api_key = &mut self.edit_api_key;
            let edit_transcribe_model = &mut self.edit_transcribe_model;
            let edit_threshold = &mut self.edit_threshold;
            let edit_min_phrase = &mut self.edit_min_phrase;
            let edit_max_phrase = &mut self.edit_max_phrase;
            let edit_language = &mut self.edit_language;
            let edit_font_size = &mut self.edit_font_size;
            let edit_chat_api_url = &mut self.edit_chat_api_url;
            let edit_chat_api_key = &mut self.edit_chat_api_key;
            let edit_chat_model = &mut self.edit_chat_model;
            let edit_target_language = &mut self.edit_target_language;
            let edit_display_mode = &mut self.edit_display_mode;
            let edit_opacity = &mut self.edit_opacity;
            let edit_input_device = &mut self.edit_input_device;
            let input_devices = &self.input_devices;

            ctx.show_viewport_immediate(
                egui::ViewportId::from_hash_of("settings"),
                egui::ViewportBuilder::default()
                    .with_title("LiveCapTran Settings")
                    .with_inner_size([550.0, 620.0])
                    .with_resizable(false)
                    .with_minimize_button(false)
                    .with_maximize_button(false)
                    .with_always_on_top(),
                |ctx, _class| {
                    if ctx.input(|i| i.viewport().close_requested()) {
                        close_req.set(true);
                    }
                    egui::CentralPanel::default().show(ctx, |ui| {
                        egui::Grid::new("settings_grid")
                            .num_columns(2)
                            .spacing([10.0, 8.0])
                            .show(ui, |ui| {
                                ui.label("Transcribe API URL:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_api_url)
                                        .desired_width(400.0),
                                );
                                ui.end_row();

                                ui.label("Transcribe API Key:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_api_key)
                                        .desired_width(400.0)
                                        .password(true),
                                );
                                ui.end_row();

                                ui.label("Transcribe Model:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_transcribe_model)
                                        .desired_width(400.0),
                                );
                                ui.end_row();

                                ui.label("Language:");
                                egui::ComboBox::from_id_salt("language_combo")
                                    .selected_text(
                                        SOURCE_LANGUAGES
                                            .iter()
                                            .find(|(code, _)| *code == edit_language.as_str())
                                            .map(|(_, name)| *name)
                                            .unwrap_or(edit_language),
                                    )
                                    .show_ui(ui, |ui| {
                                        for &(code, name) in SOURCE_LANGUAGES {
                                            ui.selectable_value(
                                                edit_language,
                                                code.to_string(),
                                                name,
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Silence Threshold:");
                                ui.add(
                                    egui::Slider::new(edit_threshold, 0.0005..=0.05)
                                        .logarithmic(true),
                                );
                                ui.end_row();

                                ui.label("Min Phrase (s):");
                                ui.add(egui::Slider::new(edit_min_phrase, 0.5..=5.0));
                                ui.end_row();

                                ui.label("Max Phrase (s):");
                                ui.add(egui::Slider::new(edit_max_phrase, 1.5..=10.0));
                                ui.end_row();

                                ui.label("Font Size:");
                                ui.add(egui::Slider::new(edit_font_size, 20.0..=120.0));
                                ui.end_row();

                                ui.label("");
                                ui.separator();
                                ui.end_row();

                                ui.label("Chat API URL:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_chat_api_url)
                                        .desired_width(400.0),
                                );
                                ui.end_row();

                                ui.label("Chat API Key:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_chat_api_key)
                                        .desired_width(400.0)
                                        .password(true),
                                );
                                ui.end_row();

                                ui.label("Chat Model:");
                                ui.add(
                                    egui::TextEdit::singleline(edit_chat_model)
                                        .desired_width(400.0),
                                );
                                ui.end_row();

                                ui.label("Translate To:");
                                egui::ComboBox::from_id_salt("target_language_combo")
                                    .selected_text(if edit_target_language.is_empty() {
                                        "None"
                                    } else {
                                        TARGET_LANGUAGES
                                            .iter()
                                            .find(|(code, _)| {
                                                *code == edit_target_language.as_str()
                                            })
                                            .map(|(_, name)| *name)
                                            .unwrap_or(edit_target_language)
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            edit_target_language,
                                            String::new(),
                                            "None",
                                        );
                                        for &(code, name) in TARGET_LANGUAGES {
                                            ui.selectable_value(
                                                edit_target_language,
                                                code.to_string(),
                                                name,
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label("Display:");
                                ui.horizontal(|ui| {
                                    ui.selectable_value(
                                        edit_display_mode,
                                        DisplayMode::Both,
                                        "Both",
                                    );
                                    ui.selectable_value(
                                        edit_display_mode,
                                        DisplayMode::TranslationOnly,
                                        "Translation only",
                                    );
                                });
                                ui.end_row();

                                ui.label("");
                                ui.separator();
                                ui.end_row();

                                ui.label("Opacity:");
                                let mut opacity_f32 = *edit_opacity as f32;
                                if ui.add(egui::Slider::new(&mut opacity_f32, 0.0..=255.0)).changed() {
                                    *edit_opacity = opacity_f32 as u8;
                                }
                                ui.end_row();

                                ui.label("Input Device:");
                                egui::ComboBox::from_id_salt("input_device_combo")
                                    .selected_text(if edit_input_device.is_empty() {
                                        "Default"
                                    } else {
                                        edit_input_device.as_str()
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            edit_input_device,
                                            String::new(),
                                            "Default",
                                        );
                                        for name in input_devices {
                                            ui.selectable_value(
                                                edit_input_device,
                                                name.clone(),
                                                name.as_str(),
                                            );
                                        }
                                    });
                                ui.end_row();
                            });
                    });
                },
            );

            if close_req.get() {
                self.show_settings = false;
            }

            // Write back to shared settings and save to file
            let mut s = self.settings.lock().unwrap();
            s.api_url = self.edit_api_url.clone();
            s.api_key = self.edit_api_key.clone();
            s.transcribe_model = self.edit_transcribe_model.clone();
            s.silence_threshold = self.edit_threshold;
            s.min_phrase_secs = self.edit_min_phrase;
            s.max_phrase_secs = self.edit_max_phrase;
            s.language = self.edit_language.clone();
            s.font_size = self.edit_font_size;
            s.chat_api_url = self.edit_chat_api_url.clone();
            s.chat_api_key = self.edit_chat_api_key.clone();
            s.chat_model = self.edit_chat_model.clone();
            s.target_language = self.edit_target_language.clone();
            s.display_mode = self.edit_display_mode.clone();
            s.opacity = self.edit_opacity;
            s.input_device = self.edit_input_device.clone();
            s.save();
        }

        let font_size = self.edit_font_size;

        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f32().min(0.5);
        self.last_frame = now;
        let revealing = self.captions.lock().unwrap().tick(dt);

        let panel_frame = egui::Frame::new()
            .fill(egui::Color32::from_black_alpha(self.edit_opacity))
            .inner_margin(10.0);

        egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ctx, |ui| {
                let panel_rect = ui.max_rect();

                // Drag anywhere to move the window (buttons added later take precedence)
                let drag = ui.interact(panel_rect, ui.id().with("drag"), egui::Sense::drag());
                if drag.drag_started() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }

                // Rolling captions, leaving a column on the right for the buttons
                let area = egui::Rect::from_min_max(
                    panel_rect.left_top(),
                    egui::pos2(panel_rect.right() - 120.0, panel_rect.bottom()),
                );
                // In a short window, shrink the font so at least one line fits
                let row_h = ui.fonts(|f| f.row_height(&egui::FontId::proportional(font_size)));
                let size = (font_size * (area.height() / row_h).min(1.0)).max(10.0);
                let big = egui::FontId::proportional(size);
                let big_row_h = ui.fonts(|f| f.row_height(&big));
                let caps = self.captions.lock().unwrap();
                let translating = !self.edit_target_language.is_empty();
                let main_text = if translating {
                    caps.translation.parts()
                } else {
                    caps.source.parts()
                };
                let small = egui::FontId::proportional(size * 0.5);
                let source_h = ui.fonts(|f| f.row_height(&small)) * 2.0;
                // Original in two smaller gray lines above the translation, if there's room
                if translating
                    && self.edit_display_mode == DisplayMode::Both
                    && area.height() >= source_h + big_row_h
                {
                    let split_y = area.top() + source_h;
                    let top = egui::Rect::from_min_max(area.left_top(), egui::pos2(area.right(), split_y));
                    let bottom = egui::Rect::from_min_max(egui::pos2(area.left(), split_y), area.right_bottom());
                    paint_rolling(ui, top, caps.source.parts(), small, egui::Color32::from_gray(180));
                    paint_rolling(ui, bottom, main_text, big, egui::Color32::WHITE);
                } else {
                    paint_rolling(ui, area, main_text, big, egui::Color32::WHITE);
                }
                drop(caps);

                // Session start/stop button: gray "● REC" when idle, red with blinking dot while recording
                let session_rect = egui::Rect::from_min_size(
                    egui::pos2(panel_rect.right() - 112.0, panel_rect.top()),
                    egui::vec2(48.0, 32.0),
                );
                let is_active = self.session_active.load(Ordering::Relaxed);
                let rec_color = if is_active {
                    egui::Color32::from_rgb(255, 60, 60)
                } else {
                    egui::Color32::from_gray(160)
                };
                let dot_visible = !is_active || (ctx.input(|i| i.time) * 2.0) as i64 % 2 == 0;
                let mut rec_label = egui::text::LayoutJob::default();
                rec_label.append(
                    "● ",
                    0.0,
                    egui::TextFormat::simple(
                        egui::FontId::proportional(12.0),
                        if dot_visible { rec_color } else { egui::Color32::TRANSPARENT },
                    ),
                );
                rec_label.append(
                    "REC",
                    0.0,
                    egui::TextFormat::simple(egui::FontId::proportional(12.0), rec_color),
                );
                let session_btn = ui
                    .put(session_rect, egui::Button::new(rec_label).frame(false))
                    .on_hover_text(if is_active { "Stop recording" } else { "Start recording" });
                if session_btn.clicked() {
                    let new_state = !is_active;
                    self.session_active.store(new_state, Ordering::Relaxed);
                    if !new_state {
                        self.captions.lock().unwrap().clear();
                    }
                }

                // Settings button overlaid to the left of close button
                let settings_rect = egui::Rect::from_min_size(
                    egui::pos2(panel_rect.right() - 64.0, panel_rect.top()),
                    egui::vec2(32.0, 32.0),
                );
                let settings_btn = ui.put(
                    settings_rect,
                    egui::ImageButton::new(
                        egui::Image::new(&self.cog_icon)
                            .fit_to_exact_size(egui::vec2(20.0, 20.0)),
                    )
                    .frame(false),
                );
                if settings_btn.clicked() {
                    self.show_settings = !self.show_settings;
                    if self.show_settings {
                        let s = self.settings.lock().unwrap();
                        self.edit_api_url = s.api_url.clone();
                        self.edit_api_key = s.api_key.clone();
                        self.edit_transcribe_model = s.transcribe_model.clone();
                        self.edit_threshold = s.silence_threshold;
                        self.edit_min_phrase = s.min_phrase_secs;
                        self.edit_max_phrase = s.max_phrase_secs;
                        self.edit_language = s.language.clone();
                        self.edit_font_size = s.font_size;
                        self.edit_chat_api_url = s.chat_api_url.clone();
                        self.edit_chat_api_key = s.chat_api_key.clone();
                        self.edit_chat_model = s.chat_model.clone();
                        self.edit_target_language = s.target_language.clone();
                        self.edit_display_mode = s.display_mode.clone();
                        self.edit_opacity = s.opacity;
                        self.edit_input_device = s.input_device.clone();
                        drop(s);
                        self.input_devices = list_input_devices();
                    }
                }

                // Close button overlaid at top-right
                let btn_rect = egui::Rect::from_min_size(
                    egui::pos2(panel_rect.right() - 32.0, panel_rect.top()),
                    egui::vec2(32.0, 32.0),
                );
                let btn = ui.put(
                    btn_rect,
                    egui::ImageButton::new(
                        egui::Image::new(&self.close_icon)
                            .fit_to_exact_size(egui::vec2(20.0, 20.0)),
                    )
                    .frame(false),
                );
                if btn.clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });

        // Repaint faster while words are still being revealed
        ctx.request_repaint_after(Duration::from_millis(if revealing { 50 } else { 200 }));
    }
}
