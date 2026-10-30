// One overlay per monitor, kept visible above fullscreen presentations (Keynote,
// PowerPoint, etc.). Overlay windows are found by title: "LiveCapTran" sits on the
// primary monitor, "LiveCapTran 2", "LiveCapTran 3", … on the others.

pub fn overlay_title(index: usize) -> String {
    if index == 0 {
        "LiveCapTran".to_string()
    } else {
        format!("LiveCapTran {}", index + 1)
    }
}

/// Overlay index from a window title; None for other windows such as settings.
#[cfg_attr(not(any(target_os = "macos", target_os = "windows")), allow(dead_code))]
pub fn overlay_index(title: &str) -> Option<usize> {
    match title.strip_prefix("LiveCapTran")? {
        "" => Some(0),
        rest => rest
            .strip_prefix(' ')?
            .parse::<usize>()
            .ok()
            .filter(|&n| n >= 2)
            .map(|n| n - 1),
    }
}

/// Number of monitors (1 where not supported).
pub fn monitor_count() -> usize {
    imp::monitor_count()
}

/// Moves each overlay to the bottom of its monitor at full width, keeping its height.
pub fn place_overlays() {
    imp::place_overlays();
}

/// Call every frame.
pub fn keep_on_top(settings_open: bool) {
    imp::keep_on_top(settings_open);
}

#[cfg(target_os = "macos")]
mod imp {
    use super::overlay_index;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSScreen, NSWindowCollectionBehavior};
    use objc2_foundation::{NSPoint, NSRect, NSSize};

    // kCGScreenSaverWindowLevel: above slideshow windows, which sit higher than "floating"
    const LEVEL: isize = 1000;

    pub fn monitor_count() -> usize {
        MainThreadMarker::new()
            .map(|mtm| NSScreen::screens(mtm).count())
            .unwrap_or(1)
            .max(1)
    }

    pub fn place_overlays() {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        // The first screen is the primary (menu bar) one
        let screens = NSScreen::screens(mtm).to_vec();
        for window in NSApplication::sharedApplication(mtm).windows().to_vec() {
            let Some(screen) = overlay_index(&window.title().to_string()).and_then(|i| screens.get(i))
            else {
                continue;
            };
            let s = screen.frame();
            let height = window.frame().size.height;
            // AppKit's origin is bottom-left, so the screen's y is its bottom edge
            let frame = NSRect::new(NSPoint::new(s.origin.x, s.origin.y), NSSize::new(s.size.width, height));
            window.setFrame_display(frame, true);
        }
    }

    pub fn keep_on_top(_settings_open: bool) {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        // Show on every Space, including other apps' fullscreen Spaces
        let behavior = NSWindowCollectionBehavior::CanJoinAllSpaces
            | NSWindowCollectionBehavior::FullScreenAuxiliary
            | NSWindowCollectionBehavior::Stationary;
        // All windows, so the settings window isn't buried under the overlays
        for window in NSApplication::sharedApplication(mtm).windows().to_vec() {
            if window.collectionBehavior() != behavior {
                window.setCollectionBehavior(behavior);
            }
            if window.level() != LEVEL {
                window.setLevel(LEVEL);
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod imp {
    use super::overlay_index;
    use windows_sys::Win32::Foundation::{BOOL, HWND, LPARAM, RECT};
    use windows_sys::Win32::Graphics::Gdi::{EnumDisplayMonitors, HDC, HMONITOR};
    use windows_sys::Win32::System::Threading::GetCurrentThreadId;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        EnumThreadWindows, GetWindowRect, GetWindowTextW, SetWindowPos, HWND_TOPMOST,
        SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
    };

    /// This app's overlay windows with their overlay index.
    fn overlays() -> Vec<(usize, HWND)> {
        unsafe extern "system" fn collect(hwnd: HWND, found: LPARAM) -> BOOL {
            let found = unsafe { &mut *(found as *mut Vec<(usize, HWND)>) };
            let mut title = [0u16; 64];
            let len = unsafe { GetWindowTextW(hwnd, title.as_mut_ptr(), title.len() as i32) };
            if let Some(i) = overlay_index(&String::from_utf16_lossy(&title[..len.max(0) as usize])) {
                found.push((i, hwnd));
            }
            1
        }
        let mut found: Vec<(usize, HWND)> = Vec::new();
        unsafe { EnumThreadWindows(GetCurrentThreadId(), Some(collect), &mut found as *mut _ as LPARAM) };
        found
    }

    /// Monitor rectangles in physical pixels, primary first. Working in physical pixels
    /// sidesteps logical-coordinate mixups between monitors with different scaling.
    fn monitors() -> Vec<RECT> {
        unsafe extern "system" fn collect(_: HMONITOR, _: HDC, rect: *mut RECT, found: LPARAM) -> BOOL {
            unsafe { (*(found as *mut Vec<RECT>)).push(*rect) };
            1
        }
        let mut found: Vec<RECT> = Vec::new();
        unsafe {
            EnumDisplayMonitors(
                std::ptr::null_mut(),
                std::ptr::null(),
                Some(collect),
                &mut found as *mut _ as LPARAM,
            )
        };
        // The primary monitor contains the desktop origin
        let primary = |r: &RECT| r.left <= 0 && 0 < r.right && r.top <= 0 && 0 < r.bottom;
        found.sort_by_key(|r| (!primary(r), r.left, r.top));
        found
    }

    pub fn monitor_count() -> usize {
        monitors().len().max(1)
    }

    pub fn place_overlays() {
        let monitors = monitors();
        for (i, hwnd) in overlays() {
            let Some(m) = monitors.get(i) else {
                continue;
            };
            let mut current = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            unsafe { GetWindowRect(hwnd, &mut current) };
            let height = current.bottom - current.top;
            unsafe {
                SetWindowPos(
                    hwnd,
                    HWND_TOPMOST,
                    m.left,
                    m.bottom - height,
                    m.right - m.left,
                    height,
                    SWP_NOACTIVATE,
                )
            };
        }
    }

    pub fn keep_on_top(settings_open: bool) {
        // Re-raising the overlays would cover the settings window
        if settings_open {
            return;
        }
        // Another topmost window (e.g. a slideshow) can take the top spot, so re-claim it
        for (_, hwnd) in overlays() {
            unsafe {
                SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)
            };
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod imp {
    pub fn monitor_count() -> usize {
        1
    }

    pub fn place_overlays() {}

    pub fn keep_on_top(_settings_open: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_titles_round_trip() {
        for i in 0..4 {
            assert_eq!(overlay_index(&overlay_title(i)), Some(i));
        }
        assert_eq!(overlay_index("LiveCapTran Settings"), None);
        assert_eq!(overlay_index("LiveCapTran 1"), None);
        assert_eq!(overlay_index("Other"), None);
    }
}
