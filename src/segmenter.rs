// Splits the mic stream into phrases for transcription.
//
// Continuous speakers rarely leave long pauses, so the pause needed to cut shrinks
// as the phrase grows:
//   - 500ms of quiet: speaker stopped, always cut
//   - past min length: a 200ms pause cuts
//   - past the midpoint of min and max: a 100ms dip cuts
//   - at max length: force a cut at the quietest point of the last 2s
// "Quiet" is below the silence threshold, or well below the phrase's own speech
// level, since gaps between words in fast speech seldom reach true silence.
//
// Noise handling: the threshold rises with the background level (the quietest moment
// of the last 10s), and only voiced frames (with a pitch) can start a phrase, so fans,
// hiss, clicks and breaths don't. Phrases with too little voiced audio are dropped.

use std::collections::VecDeque;

const FRAMES_PER_SEC: usize = 20; // 50ms frames
const END_FRAMES: usize = 10; // 500ms
const PAUSE_FRAMES: usize = 4; // 200ms
const DIP_FRAMES: usize = 2; // 100ms
const DIP_RATIO: f32 = 0.25; // quiet if below 25% of the phrase's mean speech energy
const TAIL_FRAMES: usize = 2; // quiet frames kept at the end so the last syllable isn't clipped
const PRE_ROLL_FRAMES: usize = 4; // frames kept before onset so the first syllable isn't clipped
const FORCED_CUT_WINDOW_SECS: f32 = 2.0;
const MIN_SEND_SECS: f32 = 0.5; // shorter phrases are dropped as noise
const NOISE_WINDOW_FRAMES: usize = 200; // 10s; background = quietest frame in this window
const NOISE_WARMUP_FRAMES: usize = 40; // 2s of history before trusting the estimate
const NOISE_MARGIN: f32 = 3.0; // speech must be this much louder than the background
const MAX_AUTO_THRESHOLD: f32 = 0.1;
const SPEECH_QUIET_RATIO: f32 = 0.4; // within a phrase, never call frames above 40% of its level quiet
const VOICED_CORR: f32 = 0.6; // pitch periodicity that counts as voiced
const MIN_VOICED_FRAMES: usize = 4; // phrases with less voiced audio (200ms) are noise

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Peak normalized autocorrelation over 70–400 Hz pitch lags: near 1 for voiced
/// speech, low for hiss, hum-free noise, breaths and clicks.
fn voicing(frame: &[f32], sample_rate: u32) -> f32 {
    // ~8 kHz is plenty for pitch
    let step = (sample_rate as usize / 8000).max(1);
    let x: Vec<f32> = frame
        .chunks(step)
        .map(|c| c.iter().sum::<f32>() / c.len() as f32)
        .collect();
    let mean = x.iter().sum::<f32>() / x.len() as f32;
    let x: Vec<f32> = x.iter().map(|v| v - mean).collect();
    let rate = sample_rate as usize / step;
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
    (rate / 400..(rate / 70).min(x.len() / 2))
        .map(|lag| {
            let (a, b) = (&x[..x.len() - lag], &x[lag..]);
            dot(a, b) / (dot(a, a) * dot(b, b)).sqrt().max(1e-12)
        })
        .fold(0.0, f32::max)
}

pub struct Phrase {
    pub samples: Vec<f32>,
    pub continues: bool, // cut at max length while speech goes on
}

pub struct Segmenter {
    sample_rate: u32,
    frame_len: usize,
    partial: Vec<f32>,
    pre_roll: VecDeque<Vec<f32>>,
    phrase: Vec<f32>,
    energies: Vec<f32>, // one per frame of `phrase`
    voiced: Vec<bool>,  // one per frame of `phrase`
    history: VecDeque<f32>, // recent frame energies, for the background level
    speaking: bool,
    quiet_run: usize,
}

impl Segmenter {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            frame_len: (sample_rate as usize / FRAMES_PER_SEC).max(1),
            partial: Vec::new(),
            pre_roll: VecDeque::new(),
            phrase: Vec::new(),
            energies: Vec::new(),
            voiced: Vec::new(),
            history: VecDeque::new(),
            speaking: false,
            quiet_run: 0,
        }
    }

    pub fn reset(&mut self) {
        self.partial.clear();
        self.pre_roll.clear();
        self.phrase.clear();
        self.energies.clear();
        self.voiced.clear();
        self.speaking = false;
        self.quiet_run = 0;
    }

    /// The phrase being spoken right now, once it has enough voiced audio.
    pub fn current(&self) -> Option<&[f32]> {
        let voiced = self.voiced.iter().filter(|&&v| v).count();
        (self.speaking && voiced >= MIN_VOICED_FRAMES).then_some(self.phrase.as_slice())
    }

    /// The user's threshold, raised above the background level once it is known.
    fn effective_threshold(&self, threshold: f32) -> f32 {
        if self.history.len() < NOISE_WARMUP_FRAMES {
            return threshold;
        }
        let floor = self.history.iter().copied().fold(f32::MAX, f32::min);
        threshold.max((floor * NOISE_MARGIN).min(MAX_AUTO_THRESHOLD))
    }

    fn is_voiced(&self, frame: &[f32]) -> bool {
        voicing(frame, self.sample_rate) > VOICED_CORR
    }

    /// Feeds samples and returns any phrases completed by them.
    pub fn push(
        &mut self,
        samples: &[f32],
        threshold: f32,
        min_secs: f32,
        max_secs: f32,
    ) -> Vec<Phrase> {
        let min_frames = ((min_secs * FRAMES_PER_SEC as f32) as usize).max(1);
        let max_frames = ((max_secs * FRAMES_PER_SEC as f32) as usize).max(min_frames + 1);

        let mut buf = std::mem::take(&mut self.partial);
        buf.extend_from_slice(samples);
        let mut out = Vec::new();
        let mut frames = buf.chunks_exact(self.frame_len);
        for frame in &mut frames {
            self.process_frame(frame, threshold, min_frames, max_frames, &mut out);
        }
        self.partial = frames.remainder().to_vec();
        out
    }

    fn process_frame(
        &mut self,
        frame: &[f32],
        threshold: f32,
        min_frames: usize,
        max_frames: usize,
        out: &mut Vec<Phrase>,
    ) {
        let energy = rms(frame);
        self.history.push_back(energy);
        if self.history.len() > NOISE_WINDOW_FRAMES {
            self.history.pop_front();
        }
        // Pitch decides what is speech; only starting a phrase also needs the level
        // to clear the background
        let voiced = energy > threshold && self.is_voiced(frame);
        let threshold = self.effective_threshold(threshold);

        if !self.speaking {
            if voiced && energy > threshold {
                self.speaking = true;
                self.quiet_run = 0;
                for f in std::mem::take(&mut self.pre_roll) {
                    self.append(&f, rms(&f), false);
                }
                self.append(frame, energy, true);
            } else {
                self.pre_roll.push_back(frame.to_vec());
                if self.pre_roll.len() > PRE_ROLL_FRAMES {
                    self.pre_roll.pop_front();
                }
            }
            return;
        }

        self.append(frame, energy, voiced);
        // In long unbroken speech the background estimate can creep up to soft speech;
        // the phrase's own level keeps such frames from counting as pauses
        let level = self.speech_level();
        let quiet_threshold = if level > 0.0 {
            threshold.min(level * SPEECH_QUIET_RATIO)
        } else {
            threshold
        };
        if energy <= quiet_threshold || energy < level * DIP_RATIO {
            self.quiet_run += 1;
        } else {
            self.quiet_run = 0;
        }

        let n = self.energies.len();
        let speech = n - self.quiet_run; // length before the current pause
        let soft_frames = (min_frames + max_frames) / 2;
        let pause_cut = self.quiet_run >= END_FRAMES
            || (speech >= min_frames && self.quiet_run >= PAUSE_FRAMES)
            || (speech >= soft_frames && self.quiet_run >= DIP_FRAMES);

        if pause_cut {
            let end = n - self.quiet_run + self.quiet_run.min(TAIL_FRAMES);
            self.emit(end, false, out);
            // Leftover quiet becomes pre-roll for the next phrase
            let rest = self.phrase.split_off(0);
            self.energies.clear();
            self.voiced.clear();
            for f in rest.chunks_exact(self.frame_len) {
                self.pre_roll.push_back(f.to_vec());
            }
            while self.pre_roll.len() > PRE_ROLL_FRAMES {
                self.pre_roll.pop_front();
            }
            self.speaking = false;
            self.quiet_run = 0;
        } else if n >= max_frames {
            let window = (FORCED_CUT_WINDOW_SECS * FRAMES_PER_SEC as f32) as usize;
            let lo = n.saturating_sub(window).max(min_frames);
            let cut = (lo..n)
                .min_by(|&a, &b| self.energies[a].total_cmp(&self.energies[b]))
                .unwrap_or(n);
            // Speech continues: the remainder after the cut starts the next phrase
            self.emit(cut, true, out);
        }
    }

    fn append(&mut self, frame: &[f32], energy: f32, voiced: bool) {
        self.phrase.extend_from_slice(frame);
        self.energies.push(energy);
        self.voiced.push(voiced);
    }

    /// Mean energy of the phrase's voiced frames.
    fn speech_level(&self) -> f32 {
        let (sum, count) = self
            .energies
            .iter()
            .zip(&self.voiced)
            .filter(|(_, v)| **v)
            .fold((0.0, 0usize), |(s, c), (&e, _)| (s + e, c + 1));
        if count == 0 { 0.0 } else { sum / count as f32 }
    }

    /// Sends the first `frames` frames and keeps the rest as the current phrase.
    fn emit(&mut self, frames: usize, continues: bool, out: &mut Vec<Phrase>) {
        let split = frames * self.frame_len;
        let rest = self.phrase.split_off(split);
        let head = std::mem::replace(&mut self.phrase, rest);
        self.energies.drain(..frames);
        let voiced = self.voiced.drain(..frames).filter(|&v| v).count();
        if voiced >= MIN_VOICED_FRAMES
            && head.len() as f32 >= MIN_SEND_SECS * (self.frame_len * FRAMES_PER_SEC) as f32
        {
            out.push(Phrase { samples: head, continues });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 16000;
    const THRESHOLD: f32 = 0.003;
    const MIN: f32 = 1.0;
    const MAX: f32 = 3.0;

    // Speech-like signal: 200ms syllables at varying loudness, plus `gaps` as (start_sec, len_sec, amplitude)
    fn signal(secs: f32, gaps: &[(f32, f32, f32)]) -> Vec<f32> {
        let n = (secs * RATE as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / RATE as f32;
                let syllable = (t / 0.2) as usize;
                let mut amp = [0.10, 0.06, 0.12, 0.04, 0.08][syllable % 5];
                for &(start, len, a) in gaps {
                    if t >= start && t < start + len {
                        amp = a;
                    }
                }
                amp * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
            })
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * RATE as f32) as usize]
    }

    fn run(audio: &[f32]) -> Vec<f32> {
        let mut seg = Segmenter::new(RATE);
        let mut lens = Vec::new();
        // Feed in uneven chunks, like the real capture loop with API backlog
        for chunk in audio.chunks(3001) {
            for p in seg.push(chunk, THRESHOLD, MIN, MAX) {
                lens.push(p.samples.len() as f32 / RATE as f32);
            }
        }
        lens
    }

    #[test]
    fn continuous_speech_is_cut_between_min_and_max() {
        let mut audio = signal(30.0, &[]);
        audio.extend(silence(1.0));
        let lens = run(&audio);
        assert!(lens.len() >= 5, "{lens:?}");
        for &l in &lens[..lens.len() - 1] {
            assert!(l >= MIN && l <= MAX + 0.05, "{lens:?}");
        }
        let total: f32 = lens.iter().sum();
        assert!(total > 29.0, "audio lost: {lens:?}");
    }

    #[test]
    fn cuts_land_on_short_dips() {
        // 100ms near-silent dip every 1.5s, never reaching the threshold
        let gaps: Vec<_> = (1..20).map(|k| (k as f32 * 1.5, 0.1, 0.005)).collect();
        let mut audio = signal(30.0, &gaps);
        audio.extend(silence(1.0));
        let lens = run(&audio);
        // Every cut falls on a dip, so phrases are whole multiples of 1.5s
        for &l in &lens[..lens.len() - 1] {
            assert!(l >= MIN && l <= MAX + 0.05, "{lens:?}");
            let off = (l / 1.5).round() * 1.5 - l;
            assert!(off.abs() <= 0.1, "{lens:?}");
        }
    }

    #[test]
    fn pauses_cut_after_min() {
        // 250ms silence every 2.5s
        let gaps: Vec<_> = (1..12).map(|k| (k as f32 * 2.5, 0.25, 0.0)).collect();
        let mut audio = signal(30.0, &gaps);
        audio.extend(silence(1.0));
        let lens = run(&audio);
        assert!(lens.len() >= 10, "{lens:?}");
        for &l in &lens[..lens.len() - 1] {
            assert!(l >= MIN && l <= 3.0, "{lens:?}");
        }
    }

    #[test]
    fn short_utterance_sent_after_silence() {
        let mut audio = silence(1.0);
        audio.extend(signal(1.0, &[]));
        audio.extend(silence(1.0));
        let lens = run(&audio);
        assert_eq!(lens.len(), 1, "{lens:?}");
        assert!(lens[0] >= 1.0 && lens[0] <= 1.4, "{lens:?}");
    }

    // Deterministic white noise with the given RMS
    fn noise(secs: f32, level: f32, seed: u32) -> Vec<f32> {
        let mut state = seed;
        (0..(secs * RATE as f32) as usize)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state as f32 / u32::MAX as f32 * 2.0 - 1.0) * level * 3f32.sqrt()
            })
            .collect()
    }

    fn mix(a: &mut [f32], b: &[f32]) {
        for (x, y) in a.iter_mut().zip(b) {
            *x += y;
        }
    }

    #[test]
    fn background_noise_neither_starts_nor_holds_phrases() {
        // Steady hiss far above the 0.003 threshold, with two 1.5s utterances in it
        let mut audio = noise(4.0, 0.02, 1);
        let mut utterance = signal(1.5, &[]);
        mix(&mut utterance, &noise(1.5, 0.02, 2));
        audio.extend(&utterance);
        audio.extend(noise(2.0, 0.02, 3));
        audio.extend(&utterance);
        audio.extend(noise(2.0, 0.02, 4));
        let lens = run(&audio);
        assert_eq!(lens.len(), 2, "{lens:?}");
        for &l in &lens {
            assert!(l >= 1.4 && l <= 2.0, "{lens:?}");
        }
    }

    #[test]
    fn loud_unvoiced_noise_is_dropped() {
        // Bursts of hiss and a click: loud, but no pitch
        let mut audio = silence(1.0);
        audio.extend(noise(1.0, 0.1, 5));
        audio.extend(silence(1.0));
        audio.extend(noise(0.05, 0.5, 6));
        audio.extend(silence(1.0));
        assert!(run(&audio).is_empty());
    }

    #[test]
    fn silence_and_blips_are_dropped() {
        let mut audio = silence(2.0);
        audio.extend(signal(0.1, &[]));
        audio.extend(silence(2.0));
        assert!(run(&audio).is_empty());
    }
}
