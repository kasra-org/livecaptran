// Turns repeated transcriptions of a growing phrase into a stable word stream.
// A word is committed once two consecutive hypotheses agree on it (LocalAgreement-2);
// the rest of the latest hypothesis is shown as tentative.

use crate::captions::tokenize;

fn same(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    norm(a) == norm(b)
}

#[derive(Default)]
pub struct Agreement {
    committed: usize, // words of the current phrase already committed
    prev: Vec<String>,
}

impl Agreement {
    /// Feeds a transcription of the phrase so far. Returns (newly committed text, tentative text).
    pub fn partial(&mut self, text: &str) -> (String, String) {
        let words = tokenize(text);
        let agreed = self
            .prev
            .iter()
            .zip(&words)
            .take_while(|(a, b)| same(a, b))
            .count();
        let new = if agreed > self.committed {
            words[self.committed..agreed].concat()
        } else {
            String::new()
        };
        self.committed = self.committed.max(agreed);
        let tentative = words.get(self.committed..).map(|w| w.concat()).unwrap_or_default();
        self.prev = words;
        (new, tentative)
    }

    /// Latest partial hypothesis, used when the final transcription fails.
    pub fn last(&self) -> String {
        self.prev.concat()
    }

    /// Feeds the final transcription of a finished phrase. Returns the text not yet committed.
    pub fn finalize(&mut self, text: &str, continues: bool) -> String {
        let words = tokenize(text);
        let rest = words.get(self.committed..).map(|w| w.concat()).unwrap_or_default();
        // On a mid-speech cut, words committed past this phrase's end belong to the
        // audio carried into the next phrase, so skip them there
        self.committed = if continues {
            self.committed.saturating_sub(words.len())
        } else {
            0
        };
        self.prev.clear();
        rest
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commits_words_two_hypotheses_agree_on() {
        let mut a = Agreement::default();
        assert_eq!(a.partial("오늘은 단백"), (String::new(), "오늘은 단백".into()));
        assert_eq!(a.partial("오늘은 단백질 구"), ("오늘은 ".into(), "단백질 구".into()));
        assert_eq!(a.partial("오늘은 단백질 구조를,"), ("단백질 ".into(), "구조를,".into()));
        assert_eq!(a.finalize("오늘은 단백질 구조를 보겠습니다.", false), "구조를 보겠습니다.");
    }

    #[test]
    fn ignores_case_and_punctuation_when_comparing() {
        let mut a = Agreement::default();
        a.partial("Hello world.");
        assert_eq!(a.partial("hello, world and"), ("hello, world ".into(), "and".into()));
    }

    #[test]
    fn revised_words_stay_tentative() {
        let mut a = Agreement::default();
        a.partial("one two three");
        a.partial("one two four");
        // "two" is committed; "three" -> "four" disagreed, so only later agreement commits it
        assert_eq!(a.partial("one two four five"), ("four ".into(), "five".into()));
    }

    #[test]
    fn carries_committed_words_across_mid_speech_cut() {
        let mut a = Agreement::default();
        a.partial("a b c d");
        a.partial("a b c d e"); // commits a b c d
        // The cut phrase only contained "a b"; "c d" were in the carried-over audio
        assert_eq!(a.finalize("a b", true), "");
        assert_eq!(a.partial("c d e f"), (String::new(), "e f".into()));
        assert_eq!(a.partial("c d e f g"), ("e f ".into(), "g".into()));
    }

    #[test]
    fn pause_cut_does_not_carry() {
        let mut a = Agreement::default();
        a.partial("a b c");
        a.partial("a b c d");
        assert_eq!(a.finalize("abc", false), "");
        assert_eq!(a.partial("next"), (String::new(), "next".into()));
    }
}
