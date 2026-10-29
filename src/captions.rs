// Rolling captions revealed word by word, so text flows on screen instead of
// replacing a whole phrase at once.

use std::collections::VecDeque;

const WORDS_PER_SEC: f32 = 3.0; // normal reveal speed
const CATCH_UP_SECS: f32 = 3.0; // with a backlog, speed up to show it within this time
const MAX_CHARS: usize = 2000; // only the newest lines fit on screen anyway

// Chinese/Japanese have no spaces, so each character is its own unit
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3000..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF | 0xFF00..=0xFFEF)
}

/// Splits text into words (CJK: characters), each carrying its trailing space.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    for c in text.trim().chars() {
        if c.is_whitespace() {
            if !cur.is_empty() && !cur.ends_with(' ') {
                cur.push(' ');
            }
            continue;
        }
        let boundary = cur.ends_with(' ') || is_cjk(c) || cur.chars().last().is_some_and(is_cjk);
        if boundary && !cur.is_empty() {
            tokens.push(std::mem::take(&mut cur));
        }
        cur.push(c);
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

#[derive(Default)]
pub struct Caption {
    shown: String,
    pending: VecDeque<String>,
    owed: f32, // fractional words carried between ticks
    tentative: String, // unconfirmed words shown after the revealed text
}

impl Caption {
    pub fn push(&mut self, text: &str) {
        let tokens = tokenize(text);
        let Some(first) = tokens.first().and_then(|t| t.chars().next()) else {
            return;
        };
        let prev = self.pending.back().unwrap_or(&self.shown);
        if let Some(last) = prev.chars().last() {
            if !last.is_whitespace() && !(is_cjk(last) && is_cjk(first)) {
                match self.pending.back_mut() {
                    Some(t) => t.push(' '),
                    None => self.shown.push(' '),
                }
            }
        }
        if self.pending.is_empty() {
            self.owed = 1.0; // show the first word right away
        }
        self.pending.extend(tokens);
    }

    /// Reveals words due after `dt` seconds. Returns true while words are still pending.
    pub fn tick(&mut self, dt: f32) -> bool {
        if self.pending.is_empty() {
            return false;
        }
        let rate = WORDS_PER_SEC.max(self.pending.len() as f32 / CATCH_UP_SECS);
        self.owed += rate * dt;
        while self.owed >= 1.0 {
            let Some(token) = self.pending.pop_front() else {
                break;
            };
            self.shown.push_str(&token);
            self.owed -= 1.0;
        }
        if self.shown.len() > MAX_CHARS {
            let mut cut = self.shown.len() - MAX_CHARS;
            while !self.shown.is_char_boundary(cut) {
                cut += 1;
            }
            if let Some(space) = self.shown[cut..].find(' ') {
                cut += space + 1;
            }
            self.shown.drain(..cut);
        }
        !self.pending.is_empty()
    }

    pub fn set_tentative(&mut self, text: &str) {
        self.tentative = text.trim().to_string();
    }

    /// Revealed text and the tentative tail, which waits until all committed words are shown.
    pub fn parts(&self) -> (&str, &str) {
        let tentative = if self.pending.is_empty() { self.tentative.as_str() } else { "" };
        (&self.shown, tentative)
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

#[derive(Default)]
pub struct Captions {
    pub source: Caption,
    pub translation: Caption,
}

impl Captions {
    pub fn tick(&mut self, dt: f32) -> bool {
        let source = self.source.tick(dt);
        let translation = self.translation.tick(dt);
        source || translation
    }

    pub fn clear(&mut self) {
        self.source.clear();
        self.translation.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizes_words_and_cjk_characters() {
        assert_eq!(tokenize(" 안녕하세요  여러분\n반갑습니다 "), ["안녕하세요 ", "여러분 ", "반갑습니다"]);
        assert_eq!(tokenize("Hello, world!"), ["Hello, ", "world!"]);
        assert_eq!(tokenize("你好。OK"), ["你", "好", "。", "OK"]);
    }

    #[test]
    fn reveals_word_by_word_and_joins_phrases() {
        let mut c = Caption::default();
        c.push("one two three");
        assert!(c.tick(0.0));
        assert_eq!(c.parts().0, "one ");
        c.tick(1.0 / WORDS_PER_SEC);
        assert_eq!(c.parts().0, "one two ");
        c.push("four");
        while c.tick(0.1) {}
        assert_eq!(c.parts().0, "one two three four");
        c.push("five");
        c.tick(0.0);
        assert_eq!(c.parts().0, "one two three four five");
    }

    #[test]
    fn speeds_up_with_backlog() {
        let mut c = Caption::default();
        c.push(&"word ".repeat(30));
        let mut secs = 0.0;
        while c.tick(0.05) {
            secs += 0.05;
        }
        // Clearly faster than the base rate alone (10s)
        assert!(secs < 30.0 / WORDS_PER_SEC * 0.75, "{secs}");
    }

    #[test]
    fn trims_old_text() {
        let mut c = Caption::default();
        for _ in 0..500 {
            c.push("가나다라 마바사");
            while c.tick(1.0) {}
        }
        assert!(c.parts().0.len() <= MAX_CHARS);
        assert!(c.parts().0.starts_with("가나다라") || c.parts().0.starts_with("마바사"));
    }
}
