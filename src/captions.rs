// Rolling captions revealed word by word, so text flows on screen instead of
// replacing a whole phrase at once.

use std::collections::VecDeque;

const WORDS_PER_SEC: f32 = 3.0; // normal reveal speed
const CATCH_UP_SECS: f32 = 2.0; // with a backlog, speed up to show it within this time
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
    fixed: VecDeque<String>, // committed words
    fixed_len: usize,        // their total length in bytes
    tentative: Vec<String>,  // unconfirmed words after them, which may still change
    revealed: usize,         // words of fixed + tentative shown so far
    owed: f32,               // fractional words carried between ticks
}

impl Caption {
    fn total(&self) -> usize {
        self.fixed.len() + self.tentative.len()
    }

    /// Appends newly committed text and replaces the tentative tail. Newly committed words
    /// are normally the start of the previous tail, so words already on screen stay put
    /// (they only stop being dimmed) and only words not shown yet flow in.
    pub fn update(&mut self, committed: &str, tentative: &str) {
        let caught_up = self.revealed >= self.total();
        let tokens = tokenize(committed);
        if let (Some(last), Some(first)) = (self.fixed.back_mut(), tokens.first()) {
            let (end, start) = (last.chars().last().unwrap(), first.chars().next().unwrap());
            if !end.is_whitespace() && !(is_cjk(end) && is_cjk(start)) {
                last.push(' ');
                self.fixed_len += 1;
            }
        }
        self.fixed_len += tokens.iter().map(String::len).sum::<usize>();
        self.fixed.extend(tokens);
        while self.fixed_len > MAX_CHARS {
            let Some(old) = self.fixed.pop_front() else {
                break;
            };
            self.fixed_len -= old.len();
            self.revealed = self.revealed.saturating_sub(1);
        }
        self.tentative = tokenize(tentative);
        self.revealed = self.revealed.min(self.total());
        if caught_up {
            self.owed = 1.0; // show the first new word right away
        }
    }

    /// Reveals words due after `dt` seconds. Returns true while words are still hidden.
    pub fn tick(&mut self, dt: f32) -> bool {
        let hidden = self.total() - self.revealed;
        if hidden == 0 {
            return false;
        }
        let rate = WORDS_PER_SEC.max(hidden as f32 / CATCH_UP_SECS);
        self.owed += rate * dt;
        let n = (self.owed as usize).min(hidden);
        self.revealed += n;
        self.owed -= n as f32;
        self.revealed < self.total()
    }

    /// Revealed committed text and revealed tentative text (drawn dimmed after it).
    pub fn parts(&self) -> (String, String) {
        let fixed_shown = self.revealed.min(self.fixed.len());
        let fixed = self.fixed.iter().take(fixed_shown).map(String::as_str).collect();
        (fixed, self.tentative[..self.revealed - fixed_shown].concat())
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

    // "fixed|tentative"
    fn shown(c: &Caption) -> String {
        let (fixed, tentative) = c.parts();
        format!("{fixed}|{tentative}")
    }

    #[test]
    fn reveals_word_by_word_and_joins_phrases() {
        let mut c = Caption::default();
        c.update("one two three", "");
        assert!(c.tick(0.0));
        assert_eq!(shown(&c), "one |");
        c.tick(1.0 / WORDS_PER_SEC);
        assert_eq!(shown(&c), "one two |");
        c.update("four", "");
        while c.tick(0.1) {}
        assert_eq!(shown(&c), "one two three four|");
        c.update("five", "");
        c.tick(0.0);
        assert_eq!(shown(&c), "one two three four five|");
    }

    #[test]
    fn committing_shown_words_keeps_them_on_screen() {
        let mut c = Caption::default();
        c.update("", "오늘은 단백질");
        while c.tick(0.1) {}
        assert_eq!(shown(&c), "|오늘은 단백질");
        // The committed word turns solid in place; nothing disappears
        c.update("오늘은", "단백질 구조를");
        assert_eq!(shown(&c), "오늘은|단백질 ");
        // Only the new word flows in
        c.tick(0.0);
        assert_eq!(shown(&c), "오늘은|단백질 구조를");
        // A revised tentative word changes in place
        c.update("", "단백질 구조가");
        assert_eq!(shown(&c), "오늘은|단백질 구조가");
        // Fewer words: the extra ones are gone, and new ones flow in again
        c.update("", "단백");
        assert_eq!(shown(&c), "오늘은|단백");
        c.update("단백질 구조를", "");
        assert_eq!(shown(&c), "오늘은 단백질 |");
    }

    #[test]
    fn speeds_up_with_backlog() {
        let mut c = Caption::default();
        c.update(&"word ".repeat(30), "");
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
            c.update("가나다라 마바사", "");
            while c.tick(1.0) {}
        }
        let (fixed, _) = c.parts();
        assert!(fixed.len() <= MAX_CHARS);
        assert!(fixed.starts_with("가나다라") || fixed.starts_with("마바사"));
    }
}
