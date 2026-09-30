//! Published dual-rate waveform processing. Both branches start at original PCM;
//! the sinc kernel follows torchaudio's default Hann resampler. Spectrograms use
//! the export's actual Slaney filters, HF float64 FFT -> complex64 convention.
use super::manifest::AudioProcessor;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::f64::consts::PI;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Spectrum {
    pub sampling_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_samples: usize,
    pub n_frames: usize,
    pub mel_filters: Vec<Vec<f64>>,
    pub window: String,
    pub center: bool,
    pub pad_mode: String,
    pub power: f64,
    pub log: String,
    pub floor: f64,
    pub range: Option<f32>,
    pub affine: Option<[f32; 2]>,
    pub reference: Option<f64>,
    pub min_value: Option<f64>,
    pub db_range: Option<f64>,
    pub padding: Option<String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    pub format_version: u32,
    pub whisper: Spectrum,
    pub clap: Spectrum,
    pub windows: String,
}
impl AudioConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.format_version == 1 && self.windows == "endpoint_cover_v1",
            "unsupported audio feature contract"
        );
        for (s, rate, fft, hop, samples, frames, mels) in [
            (&self.whisper, 16000, 400, 160, 480000, 3000, 80),
            (&self.clap, 48000, 1024, 480, 480000, 1001, 64),
        ] {
            anyhow::ensure!(
                s.sampling_rate == rate
                    && s.n_fft == fft
                    && s.hop_length == hop
                    && s.n_samples == samples
                    && s.n_frames == frames
                    && s.window == "periodic_hann"
                    && s.center
                    && s.pad_mode == "reflect"
                    && s.power == 2.0
                    && s.floor == 1e-10,
                "unsupported spectrogram contract"
            );
            anyhow::ensure!(
                s.mel_filters.len() == fft / 2 + 1
                    && s.mel_filters
                        .iter()
                        .all(|v| v.len() == mels && v.iter().all(|f| f.is_finite() && *f >= 0.0)),
                "invalid mel filter bank"
            );
            anyhow::ensure!(
                (0..mels).all(|m| s.mel_filters.iter().any(|row| row[m] > 0.0)),
                "empty mel filter"
            );
        }
        anyhow::ensure!(
            self.whisper.log == "log10"
                && self.whisper.range == Some(8.0)
                && self.whisper.affine == Some([0.25, 1.0])
                && self.whisper.padding.is_none()
                && self.whisper.reference.is_none()
                && self.whisper.min_value.is_none()
                && self.whisper.db_range.is_none(),
            "invalid Whisper readout"
        );
        anyhow::ensure!(
            self.clap.log == "db"
                && self.clap.reference == Some(1.0)
                && self.clap.min_value == Some(1e-10)
                && self.clap.db_range.is_none()
                && self.clap.range.is_none()
                && self.clap.affine.is_none()
                && self.clap.padding.as_deref() == Some("repeatpad"),
            "invalid CLAP readout"
        );
        Ok(())
    }
}
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}
/// Direct sparse polyphase convolution avoids allocating orig_rate*new_rate
/// kernels for valid, relatively prime source rates.
fn resample(input: &[f32], source: usize, target: usize) -> Vec<f32> {
    if source == target {
        return input.to_vec();
    }
    let divisor = gcd(source, target);
    let (original, new) = (source / divisor, target / divisor);
    let base = (original.min(new) as f64) * 0.99;
    let width = (6.0 * original as f64 / base).ceil() as isize;
    let length = (input.len() * new).div_ceil(original);
    let mut output = Vec::with_capacity(length);
    for index in 0..length {
        let group = (index / new) * original;
        // torch arange's integer phase division uses the default float32 dtype.
        let phase = -((index % new) as f32) / (new as f32);
        let center = -(phase as f64) * original as f64;
        let lower = (center - 6.0 * original as f64 / base).floor() as isize;
        let upper = (center + 6.0 * original as f64 / base).ceil() as isize;
        let mut value = 0f32;
        for offset in lower.max(-width)..=upper.min(width + original as isize - 1) {
            let position = group as isize + offset;
            if !(0..input.len() as isize).contains(&position) {
                continue;
            }
            let t = ((phase as f64 + offset as f64 / original as f64) * base).clamp(-6.0, 6.0);
            let window = (t * PI / 6.0 / 2.0).cos().powi(2);
            let angle = t * PI;
            let sinc = if angle == 0.0 {
                1.0
            } else {
                angle.sin() / angle
            };
            let weight = (sinc * window * (base / original as f64)) as f32;
            value += input[position as usize] * weight;
        }
        output.push(value);
    }
    output
}
pub fn native_rate(pcm: &[f32], source: usize, channels: usize, target: usize) -> Vec<f32> {
    let frames = pcm.len() / channels;
    let mut result = vec![0f32; (frames * target).div_ceil(source).min(target * 30)];
    for channel in pcm.chunks_exact(frames) {
        let values = resample(channel, source, target);
        for (dst, value) in result.iter_mut().zip(values) {
            *dst += value;
        }
    }
    for value in &mut result {
        *value /= channels as f32;
    }
    result
}
pub fn validate_pcm(
    pcm: &[f32],
    rate: usize,
    channels: usize,
    contract: &AudioProcessor,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        rate > 0
            && rate <= contract.max_sample_rate
            && (contract.sample_rates.is_empty() || contract.sample_rates.contains(&rate)),
        "unsupported original audio sampling rate"
    );
    anyhow::ensure!(
        (1..=8).contains(&channels)
            && !pcm.is_empty()
            && pcm.len().is_multiple_of(channels)
            && pcm.iter().all(|v| v.is_finite()),
        "require finite nonempty channels-first PCM with 1..8 channels"
    );
    anyhow::ensure!(
        pcm.len() / channels <= rate * contract.max_seconds,
        "audio input exceeds 30 seconds"
    );
    Ok(())
}
pub fn windows(samples: usize) -> Vec<(usize, usize)> {
    if samples <= 480000 {
        return vec![(0, samples)];
    }
    let count = samples.div_ceil(480000);
    let last = samples - 480000;
    (0..count)
        .map(|i| {
            let start = i * last / (count - 1);
            (start, start + 480000)
        })
        .collect()
}
fn reflect(index: isize, length: usize) -> usize {
    let period = 2 * (length - 1) as isize;
    let value = index.rem_euclid(period);
    if value < length as isize {
        value as usize
    } else {
        (period - value) as usize
    }
}
pub fn features(wave: &[f32], spec: &Spectrum, clap: bool) -> Vec<f32> {
    let mut padded = vec![0f32; spec.n_samples];
    if clap {
        let repeated = (spec.n_samples / wave.len()) * wave.len();
        for (i, value) in padded.iter_mut().take(repeated).enumerate() {
            *value = wave[i % wave.len()];
        }
    } else {
        padded[..wave.len()].copy_from_slice(wave);
    }
    let mels = spec.mel_filters[0].len();
    let sparse: Vec<Vec<(usize, f64)>> = (0..mels)
        .map(|m| {
            spec.mel_filters
                .iter()
                .enumerate()
                .filter_map(|(i, row)| (row[m] != 0.0).then_some((i, row[m])))
                .collect()
        })
        .collect();
    let window: Vec<f64> = (0..spec.n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / spec.n_fft as f64).cos())
        .collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(spec.n_fft);
    let mut buffer = vec![Complex::new(0f64, 0f64); spec.n_fft];
    let mut scratch = vec![Complex::new(0f64, 0f64); fft.get_inplace_scratch_len()];
    let mut power = vec![0f64; spec.n_fft / 2 + 1];
    let mut output = vec![0f32; spec.n_frames * mels];
    for frame in 0..spec.n_frames {
        for (i, value) in buffer.iter_mut().enumerate() {
            let index =
                frame as isize * spec.hop_length as isize + i as isize - (spec.n_fft / 2) as isize;
            *value = Complex::new(
                f64::from(padded[reflect(index, padded.len())]) * window[i],
                0.0,
            );
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for (dst, value) in power.iter_mut().zip(&buffer) {
            // np spectrogram storage rounds FFT to complex64 before abs(dtype=f64).
            *dst = f64::from(value.re as f32).powi(2) + f64::from(value.im as f32).powi(2);
        }
        for (mel, filter) in sparse.iter().enumerate() {
            let energy = filter
                .iter()
                .map(|(bin, w)| power[*bin] * w)
                .sum::<f64>()
                .max(spec.floor);
            let log = if clap {
                10.0 * energy.log10()
            } else {
                energy.log10()
            };
            let index = if clap {
                frame * mels + mel
            } else {
                mel * spec.n_frames + frame
            };
            output[index] = log as f32;
        }
    }
    if !clap {
        let maximum = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for value in &mut output {
            *value = ((*value).max(maximum - 8.0) + 4.0) / 4.0;
        }
    }
    output
}
pub fn normalize(values: &mut [f32]) -> anyhow::Result<()> {
    anyhow::ensure!(values.iter().all(|x| x.is_finite()), "nonfinite embedding");
    let norm = values.iter().map(|x| x * x).sum::<f32>().sqrt();
    anyhow::ensure!(norm > 1e-12, "zero embedding");
    for value in values {
        *value /= norm;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn endpoint_windows_cover_original_without_random_crops() {
        assert_eq!(windows(480000), vec![(0, 480000)]);
        assert_eq!(windows(480001), vec![(0, 480000), (1, 480001)]);
        assert_eq!(
            windows(1440000),
            vec![(0, 480000), (480000, 960000), (960000, 1440000)]
        );
    }
    #[test]
    fn resampling_length_and_original_channels_are_preserved() {
        let a = vec![0.25; 441];
        let b = vec![-0.25; 441];
        let pcm = [a, b].concat();
        let output = native_rate(&pcm, 44100, 2, 48000);
        assert_eq!(output.len(), 480);
        assert!(output.iter().all(|v| v.abs() < 1e-7));
        assert_eq!(resample(&[1., 2., 3.], 16000, 16000), vec![1., 2., 3.]);
    }
    #[test]
    fn reflected_center_matches_numpy() {
        assert_eq!(
            (-3..8).map(|i| reflect(i, 4)).collect::<Vec<_>>(),
            vec![3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]
        );
    }
}
