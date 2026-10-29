// Turns repeated transcriptions of a growing phrase into a stable word stream.
// A word is committed once two consecutive hypotheses agree on it (LocalAgreement-2);
// the rest of the latest hypothesis is shown as tentative.

use crate::captions::tokenize;

const ANCHOR_SEARCH: usize = 3; // how far the committed point may shift between hypotheses

fn same(a: &str, b: &str) -> bool {
    let norm = |s: &str| s.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
    norm(a) == norm(b)
}

#[derive(Default)]
pub struct Agreement {
    committed: usize,          // expected index just past the committed words
    last_word: Option<String>, // last committed word, used to re-find that point
    prev_tail: Vec<String>,    // uncommitted part of the previous hypothesis
}

impl Agreement {
    /// Index in `words` just past the committed text. Hypotheses may merge or split
    /// words, so look for the last committed word near where it is expected.
    fn boundary(&self, words: &[String]) -> usize {
        let k = self.committed;
        if let Some(last) = &self.last_word {
            for d in 0..=ANCHOR_SEARCH {
                for j in [k + d, k.wrapping_sub(d)] {
                    if (1..=words.len()).contains(&j) && same(&words[j - 1], last) {
                        return j;
                    }
                }
            }
        }
        k.min(words.len())
    }

    /// Feeds a transcription of the phrase so far. Returns (newly committed text, tentative text).
    pub fn partial(&mut self, text: &str) -> (String, String) {
        let words = tokenize(text);
        let start = self.boundary(&words);
        let tail = &words[start..];
        let agreed = self
            .prev_tail
            .iter()
            .zip(tail)
            .take_while(|(a, b)| same(a, b))
            .count();
        self.committed = start + agreed;
        if agreed > 0 {
            self.last_word = Some(tail[agreed - 1].clone());
        }
        self.prev_tail = tail[agreed..].to_vec();
        (tail[..agreed].concat(), tail[agreed..].concat())
    }

    /// Feeds the final transcription of a finished phrase (None if it failed, in which
    /// case the last tentative words are used). Returns the text not yet committed.
    pub fn finalize(&mut self, text: Option<&str>, continues: bool) -> String {
        let rest = match text {
            Some(text) => {
                let words = tokenize(text);
                let rest = words[self.boundary(&words)..].concat();
                // On a mid-speech cut, words committed past this phrase's end belong to
                // the audio carried into the next phrase, so skip them there
                self.committed = if continues {
                    self.committed.saturating_sub(words.len())
                } else {
                    0
                };
                rest
            }
            None => {
                self.committed = 0;
                self.prev_tail.concat()
            }
        };
        if self.committed == 0 {
            self.last_word = None;
        }
        self.prev_tail.clear();
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
        assert_eq!(a.finalize(Some("오늘은 단백질 구조를 보겠습니다."), false), "구조를 보겠습니다.");
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
        // "three" -> "four" disagreed, so only the next agreement commits it
        assert_eq!(a.partial("one two four five"), ("four ".into(), "five".into()));
    }

    #[test]
    fn realigns_when_earlier_words_merge_or_split() {
        let mut a = Agreement::default();
        a.partial("단백질 구조를 보면");
        a.partial("단백질 구조를 보면 이렇게"); // commits all three
        // Earlier words merged: "단백질구조를" — still continue after "보면"
        assert_eq!(a.partial("단백질구조를 보면 이렇게 됩니다"), ("이렇게 ".into(), "됩니다".into()));
        // An extra word appeared earlier: still continue after "이렇게"
        assert_eq!(a.finalize(Some("음 단백질구조를 보면 이렇게 됩니다."), false), "됩니다.");
    }

    #[test]
    fn carries_committed_words_across_mid_speech_cut() {
        let mut a = Agreement::default();
        a.partial("a b c d");
        a.partial("a b c d e"); // commits a b c d
        // The cut phrase only contained "a b"; "c d" were in the carried-over audio
        assert_eq!(a.finalize(Some("a b"), true), "");
        assert_eq!(a.partial("c d e f"), (String::new(), "e f".into()));
        assert_eq!(a.partial("c d e f g"), ("e f ".into(), "g".into()));
    }

    #[test]
    fn pause_cut_does_not_carry() {
        let mut a = Agreement::default();
        a.partial("a b c");
        a.partial("a b c d");
        assert_eq!(a.finalize(Some("abc"), false), "");
        assert_eq!(a.partial("next"), (String::new(), "next".into()));
    }

    #[test]
    fn failed_final_commits_tentative_tail() {
        let mut a = Agreement::default();
        a.partial("a b");
        a.partial("a b c");
        assert_eq!(a.finalize(None, false), "c");
    }
}
