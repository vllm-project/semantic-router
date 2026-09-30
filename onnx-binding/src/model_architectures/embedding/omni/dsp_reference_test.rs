use super::{audio, image, manifest::ImageProcessor};
use serde_json::Value;
use std::path::{Path, PathBuf};
fn floats(root: &Path, record: &Value) -> Vec<f32> {
    let data = std::fs::read(root.join(record["file"].as_str().unwrap())).unwrap();
    assert_eq!(data.len() % 4, 0);
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}
fn assert_probes(name: &str, actual: &[f32], probe: &Value, tolerance: f32) {
    let mut worst = 0f32;
    for (index, value) in probe["indices"]
        .as_array()
        .unwrap()
        .iter()
        .zip(probe["values"].as_array().unwrap())
    {
        let i = index.as_u64().unwrap() as usize;
        let reference = value.as_f64().unwrap() as f32;
        let difference = (actual[i] - reference).abs();
        worst = worst.max(difference);
        assert!(
            difference
                <= tolerance
                    + if name.contains("/clap") {
                        2e-5 * reference.abs()
                    } else {
                        0.0
                    },
            "{name}[{i}]: actual={} expected={reference} delta={difference} > {tolerance}",
            actual[i]
        );
    }
    eprintln!("{name}: max probe error {worst}");
}
fn compare_full(name: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    let (index, error) = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap();
    assert!(
        error
            <= tolerance
                + if name.contains("/clap") {
                    2e-5 * expected[index].abs()
                } else {
                    0.0
                },
        "{name}: max error {error} at {index}: {} vs {}",
        actual[index],
        expected[index]
    );
    eprintln!("{name}: max full-array error {error}");
}

fn verify_jpeg(root: &Path, index: &Value, processor: &ImageProcessor, full: bool) {
    let variants = index["jpeg_variants"].as_array().into_iter().flatten();
    for record in std::iter::once(&index["jpeg"])
        .chain(variants)
        .filter(|r| !r.is_null())
    {
        let file = record["file"].as_str().unwrap();
        let bytes = std::fs::read(root.join(file)).unwrap();
        let actual = image::pixels(&bytes, processor).unwrap();
        assert_probes(file, &actual, &record["probes"], 1e-7);
        if full {
            let rgb = image::decode_rgb(&bytes).unwrap();
            let decoded: Vec<f32> = rgb.as_raw().iter().map(|p| f32::from(*p)).collect();
            compare_full(
                &format!("{file}/decoded RGB"),
                &decoded,
                &floats(root, &record["decoded_rgb"]),
                0.0,
            );
            compare_full(
                &format!("{file}/processor"),
                &actual,
                &floats(root, &record["pixels"]),
                1e-7,
            );
        }
    }
}

fn verify(root: &Path, config: &audio::AudioConfig, full: bool) {
    let index: Value =
        serde_json::from_slice(&std::fs::read(root.join("index.json")).unwrap()).unwrap();
    for (n, record) in index["audio"].as_array().unwrap().iter().enumerate() {
        let pcm = floats(root, &record["pcm"]);
        let shape = record["pcm"]["shape"].as_array().unwrap();
        let channels = if shape.len() == 2 {
            shape[0].as_u64().unwrap() as usize
        } else {
            1
        };
        let rate = record["sampling_rate"].as_u64().unwrap() as usize;
        let mut outputs = Vec::new();
        let native16 = audio::native_rate(&pcm, rate, channels, 16000);
        let native48 = audio::native_rate(&pcm, rate, channels, 48000);
        let whisper = audio::features(&native16, &config.whisper, false);
        let clap: Vec<f32> = audio::windows(native48.len())
            .into_iter()
            .flat_map(|(s, e)| audio::features(&native48[s..e], &config.clap, true))
            .collect();
        if full {
            // Isolate spectrogram math from cross-library f32 resampling ULPs,
            // whose tiny waveform differences are amplified in quiet log bins.
            let reference48 = floats(root, &record["audio48"]);
            let clap_from_reference: Vec<f32> = audio::windows(reference48.len())
                .into_iter()
                .flat_map(|(s, e)| audio::features(&reference48[s..e], &config.clap, true))
                .collect();
            compare_full(
                "CLAP from exact reference waveform",
                &clap_from_reference,
                &floats(root, &record["clap_features"]),
                1e-5,
            );
        }
        outputs.push(("audio16", "audio16", native16, 2e-6));
        outputs.push(("audio48", "audio48", native48, 2e-6));
        outputs.push(("whisper", "whisper_features", whisper, 1e-4));
        outputs.push(("clap", "clap_features", clap, 1e-4));
        for (probe, key, actual, tolerance) in outputs {
            let name = format!("{rate}Hz case{n}/{probe}");
            assert_probes(&name, &actual, &record["probes"][probe], tolerance);
            if full {
                compare_full(&name, &actual, &floats(root, &record[key]), tolerance);
            }
        }
    }
    let variant = index["variant"].as_str().unwrap();
    let processor = ImageProcessor {
        size: if variant == "nano" { 512 } else { 384 },
        mean: [0.5; 3],
        std: [0.5; 3],
        resample: if variant == "nano" {
            "bicubic"
        } else {
            "bilinear"
        }
        .into(),
    };
    let record = &index["image"];
    let pixels = image::pixels(
        &std::fs::read(root.join(record["file"].as_str().unwrap())).unwrap(),
        &processor,
    )
    .unwrap();
    assert_probes(
        &format!("{variant} image"),
        &pixels,
        &record["probes"],
        1e-7,
    );
    if full {
        compare_full(
            &format!("{variant} image"),
            &pixels,
            &floats(root, &record["pixels"]),
            1e-7,
        );
    }
    verify_jpeg(root, &index, &processor, full);
}
#[test]
fn published_processors_match_offline_numeric_probes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/vela_omni_dsp");
    let config: audio::AudioConfig =
        serde_json::from_slice(&std::fs::read(root.join("audio.json")).unwrap()).unwrap();
    config.validate().unwrap();
    for variant in ["nano", "mini"] {
        verify(&root.join(variant), &config, false);
    }
}
#[test]
#[ignore = "explicit full reference artifacts, produced without model weights"]
fn full_reference_dsp() {
    let root = PathBuf::from(std::env::var("VELA_OMNI_DSP_DIR").expect("set VELA_OMNI_DSP_DIR"));
    for variant in ["nano", "mini"] {
        let path = root.join(variant);
        let config: audio::AudioConfig =
            serde_json::from_slice(&std::fs::read(path.join("processors/audio.json")).unwrap())
                .unwrap();
        config.validate().unwrap();
        verify(&path, &config, true);
    }
}
