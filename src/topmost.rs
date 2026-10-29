// Keeps the overlay visible above fullscreen presentations (Keynote, PowerPoint, etc.)

use raw_window_handle::HasWindowHandle;

/// Call every frame with the root (overlay) window.
pub fn keep_on_top(window: &impl HasWindowHandle, settings_open: bool) {
    imp::keep_on_top(window, settings_open);
}

#[cfg(target_os = "macos")]
mod imp {
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSWindowCollectionBehavior};
    use raw_window_handle::HasWindowHandle;

    // kCGScreenSaverWindowLevel: above slideshow windows, which sit higher than "floating"
    const LEVEL: isize = 1000;

    pub fn keep_on_top(_window: &impl HasWindowHandle, _settings_open: bool) {
        let Some(mtm) = MainThreadMarker::new() else {
            return;
        };
        // Show on every Space, including other apps' fullscreen Spaces
        let behavior = NSWindowCollectionBehavior::CanJoinAllSpaces
            | NSWindowCollectionBehavior::FullScreenAuxiliary
            | NSWindowCollectionBehavior::Stationary;
        // All windows, so the settings window isn't buried under the overlay
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
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        SetWindowPos, HWND_TOPMOST, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
    };

    pub fn keep_on_top(window: &impl HasWindowHandle, settings_open: bool) {
        // Re-raising the overlay would cover the settings window
        if settings_open {
            return;
        }
        let Ok(handle) = window.window_handle() else {
            return;
        };
        let RawWindowHandle::Win32(h) = handle.as_raw() else {
            return;
        };
        // Another topmost window (e.g. a slideshow) can take the top spot, so re-claim it
        unsafe {
            SetWindowPos(
                h.hwnd.get() as _,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
            );
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
mod imp {
    use raw_window_handle::HasWindowHandle;

    pub fn keep_on_top(_window: &impl HasWindowHandle, _settings_open: bool) {}
}
