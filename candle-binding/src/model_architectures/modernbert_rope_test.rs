use super::*;
use candle_core::IndexOp;
use serde_json::json;

fn resolve(raw: Value) -> Result<[Parameters; 2]> {
    let options: RopeOptions =
        serde_json::from_value(raw.clone()).map_err(candle_core::Error::wrap)?;
    let thetas = resolve_thetas(&raw, [160000.0, 10000.0])?;
    options.resolve(thetas, 32768, 128, 2, 4, 2)
}

#[test]
fn tf4_and_tf5_preserve_both_attention_types() -> Result<()> {
    let yarn = json!({"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192,"truncate":true});
    let old = resolve(
        json!({"global_rope_theta":160000.0,"local_rope_theta":10000.0,"rope_scaling":yarn}),
    )?;
    let mut global = yarn.clone();
    global["rope_theta"] = json!(160000.0);
    let mut local = yarn.clone();
    local["rope_theta"] = json!(10000.0);
    let new = resolve(
        json!({"rope_parameters":{"full_attention":global,"sliding_attention":local},"layer_types":["full_attention","sliding_attention","full_attention","sliding_attention"]}),
    )?;
    assert_eq!(old, new);
    assert_ne!(old[0].frequencies(64)?.0, old[1].frequencies(64)?.0);
    let mixed = resolve(
        json!({"rope_parameters":{"full_attention":global,"sliding_attention":{"rope_type":"default","rope_theta":10000.0}}}),
    )?;
    assert!(matches!(mixed[0].scaling, Scaling::Yarn { .. }));
    assert_eq!(mixed[1].scaling, Scaling::Default);
    Ok(())
}

#[test]
fn rejects_conflicts_and_unsupported_semantics() {
    let yarn = json!({"rope_type":"yarn","factor":4.0});
    for raw in [
        json!({"rope_scaling":{"rope_type":"dynamic","factor":4.0}}),
        json!({"rope_scaling":{"rope_type":"default","factor":1.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","type":"default","factor":4.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":0.5}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"truncate":false}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"mscale":1.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"attention_factor":0.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"beta_slow":0.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"beta_fast":1.0,"beta_slow":32.0}}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":0}}),
        json!({"rope_scaling":true}),
        json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"attention_factor":1e-100}}),
        json!({"local_rope_theta":10000.0,"rope_parameters":{"full_attention":{"rope_theta":10000.0}}}),
        json!({"rope_parameters":{"full_attention":{"rope_theta":10000.0},"sliding_attention":null}}),
        json!({"rope_parameters":{"rope_type":"yarn","factor":4.0}}),
        json!({"partial_rotary_factor":0.5}),
        json!({"rope_scaling":yarn,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"]}),
        json!({"rope_scaling":yarn,"layer_types":["full_attention"]}),
        json!({"global_rope_theta":10000.0,"rope_parameters":{"full_attention":{"rope_theta":20000.0},"sliding_attention":{"rope_theta":10000.0}}}),
        json!({"rope_parameters":{"full_attention":{"rope_theta":10000.0}}}),
        json!({"rope_scaling":yarn,"rope_parameters":{"full_attention":{"rope_theta":160000.0},"sliding_attention":{"rope_theta":10000.0}}}),
    ] {
        assert!(resolve(raw.clone()).is_err(), "unexpectedly accepted {raw}");
    }
}

#[test]
fn geometry_and_capacity_are_not_inferred_from_factor() -> Result<()> {
    let options: RopeOptions = serde_json::from_value(json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":8192}})).unwrap();
    for (hidden, heads, layers, cadence, positions) in [
        (0, 2, 4, 2, 32768),
        (128, 0, 4, 2, 32768),
        (127, 2, 4, 2, 32768),
        (6, 2, 4, 2, 32768),
        (128, 2, 4, 0, 32768),
        (128, 2, 4, 2, 0),
    ] {
        assert!(options
            .resolve([10000.0; 2], positions, hidden, heads, layers, cadence)
            .is_err());
    }
    // Layer-free backbones remain useful for token-admission fixtures; RoPE
    // geometry is still valid, and no attention layer consumes the cache.
    assert!(options.resolve([10000.0; 2], 17, 128, 2, 0, 2).is_ok());
    let params = options.resolve([10000.0; 2], 17, 128, 2, 4, 2)?;
    let cache = RotaryEmbedding::new(DType::F32, 64, 17, &params[0], &Device::Cpu)?;
    assert_eq!(cache.cos.dims(), &[17, 32]);
    Ok(())
}

#[test]
fn default_cache_is_bitwise_identical_to_previous_fp32_algorithm() -> Result<()> {
    let device = Device::Cpu;
    for theta in [10000.0f64, 160000.0] {
        let frequencies: Vec<_> = (0..64)
            .step_by(2)
            .map(|i| 1f32 / theta.powf(i as f64 / 64.0) as f32)
            .collect();
        let frequencies = Tensor::from_vec(frequencies, (1, 32), &device)?;
        let positions = Tensor::arange(0u32, 32768u32, &device)?
            .to_dtype(DType::F32)?
            .reshape((32768, 1))?;
        let angles = positions.matmul(&frequencies)?;
        let expected_sin = angles.sin()?.flatten_all()?.to_vec1::<f32>()?;
        let expected_cos = angles.cos()?.flatten_all()?.to_vec1::<f32>()?;
        let actual =
            RotaryEmbedding::new(DType::F32, 64, 32768, &Parameters::unscaled(theta), &device)?;
        for (actual, expected) in [(actual.sin, expected_sin), (actual.cos, expected_cos)] {
            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
            assert!(actual
                .iter()
                .zip(expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
    Ok(())
}

#[test]
fn attention_factor_is_applied_before_dtype_cast() -> Result<()> {
    let params =
        resolve(json!({"rope_scaling":{"rope_type":"yarn","factor":4.0,"attention_factor":1.25}}))?;
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let cache = RotaryEmbedding::new(dtype, 64, 3, &params[0], &Device::Cpu)?;
        assert!(cache
            .cos
            .i(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| *value == 1.25));
        assert!(cache
            .sin
            .i(0)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
            .iter()
            .all(|value| *value == 0.0));
    }
    Ok(())
}

pub(crate) fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test_data/modernbert_rope")
}

fn read_fixture(mode: &str, name: &str) -> Value {
    let bytes = std::fs::read(fixture_dir().join(mode).join(name)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn read_reference(name: &str) -> Value {
    let bytes = std::fs::read(fixture_dir().join(name)).unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn floats(value: &Value) -> Vec<f32> {
    let bytes = std::fs::read(fixture_dir().join("reference.safetensors.fixture")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let tensor = tensors.tensor(value["tensor"].as_str().unwrap()).unwrap();
    let shape: Vec<usize> = serde_json::from_value(value["shape"].clone()).unwrap();
    assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
    assert_eq!(tensor.shape(), shape);
    tensor
        .data()
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32, scope: &str) {
    assert_eq!(actual.len(), expected.len(), "{scope}");
    let mut max_diff = 0f32;
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let diff = (actual - expected).abs();
        max_diff = max_diff.max(diff);
        assert!(
            actual.is_finite() && expected.is_finite() && diff <= atol + rtol * expected.abs(),
            "{scope}[{index}]: actual={actual}, expected={expected}, diff={diff}"
        );
    }
    eprintln!("{scope}: max_abs_diff={max_diff}");
}

#[test]
fn fixture_identity_is_frozen() {
    use sha2::{Digest, Sha256};
    let weights = std::fs::read(fixture_dir().join("weights.safetensors.fixture")).unwrap();
    for mode in ["tf4", "tf5"] {
        let manifest = read_fixture(mode, "manifest.json");
        assert_eq!(
            format!("{:x}", Sha256::digest(&weights)),
            manifest["weights_sha256"]
        );
        for (name, expected) in manifest["files"].as_object().unwrap() {
            let bytes = std::fs::read(fixture_dir().join(mode).join(name)).unwrap();
            assert_eq!(format!("{:x}", Sha256::digest(bytes)), *expected);
        }
    }
    let bytes = std::fs::read(fixture_dir().join("reference.safetensors.fixture")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut referenced = std::collections::BTreeSet::new();
    for name in ["rope.json", "tiny-output.json"] {
        for case in read_reference(name)["cases"].as_array().unwrap() {
            for value in case.as_object().unwrap().values() {
                if let Some(key) = value.get("tensor").and_then(Value::as_str) {
                    assert!(
                        referenced.insert(key.to_owned()),
                        "duplicate tensor reference"
                    );
                    assert!(floats(value).iter().all(|value| value.is_finite()));
                }
            }
        }
    }
    assert_eq!(
        referenced,
        tensors
            .names()
            .into_iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        "all reference tensors must be consumed"
    );
}

#[test]
fn official_yarn_frequency_cache_and_rotated_qk_goldens() -> Result<()> {
    let device = Device::Cpu;
    for mode in ["tf4", "tf5"] {
        for (case_index, case) in read_reference("rope.json")["cases"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let theta = case["theta"].as_f64().unwrap();
            let params = parse_parameters(Some(&case["recipe"]), theta, 32768)?;
            let (frequencies, factor) = params.frequencies(64)?;
            assert_close(
                &frequencies,
                &floats(&case["inv_freq"]),
                1e-7,
                1e-6,
                &format!("{mode}/{case_index}/frequency"),
            );
            assert!((factor - case["attention_factor"].as_f64().unwrap()).abs() <= 1e-7);
            let positions: Vec<u32> = serde_json::from_value(case["positions"].clone()).unwrap();
            let indices = Tensor::from_vec(positions.clone(), positions.len(), &device)?;
            for (dtype, tolerance) in [
                (DType::F32, 0.0025),
                (DType::F16, 0.003),
                (DType::BF16, 0.0081),
            ] {
                let cache = RotaryEmbedding::new(dtype, 64, 32768, &params, &device)?;
                for (name, values) in [("cos", &cache.cos), ("sin", &cache.sin)] {
                    let actual = values
                        .index_select(&indices, 0)?
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_close(
                        &actual,
                        &floats(&case[name]),
                        tolerance,
                        0.0,
                        &format!("{mode}/{case_index}/{dtype:?}/{name}"),
                    );
                }
                if dtype == DType::F32 {
                    let selected = RotaryEmbedding {
                        cos: cache.cos.index_select(&indices, 0)?,
                        sin: cache.sin.index_select(&indices, 0)?,
                    };
                    let shape = (1, 1, positions.len(), 64);
                    let q = Tensor::from_vec(floats(&case["q"]), shape, &device)?;
                    let k = Tensor::from_vec(floats(&case["k"]), shape, &device)?;
                    let (q, k) = selected.apply_rotary_emb_qkv(&q, &k)?;
                    for (name, values) in [("rotated_q", q), ("rotated_k", k)] {
                        assert_close(
                            &values.flatten_all()?.to_vec1::<f32>()?,
                            &floats(&case[name]),
                            0.01,
                            0.0,
                            &format!("{mode}/{case_index}/{name}"),
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn assert_tiny_fixture(
    mode: &str,
    mut forward: impl FnMut(&Tensor, &Tensor) -> Result<Tensor>,
) -> Result<Vec<f32>> {
    let device = Device::Cpu;
    let mut all_outputs = Vec::new();
    for (index, case) in read_reference("tiny-output.json")["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let ids: Vec<Vec<u32>> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        let mask: Vec<Vec<u32>> = serde_json::from_value(case["attention_mask"].clone()).unwrap();
        let shape = (ids.len(), ids[0].len());
        let input = Tensor::from_vec(ids.concat(), shape, &device)?;
        let attention = Tensor::from_vec(mask.concat(), shape, &device)?;
        let actual = forward(&input, &attention)?;
        assert_eq!(actual.dtype(), DType::F32);
        let hidden = actual.to_vec3::<f32>()?;
        let mut valid = Vec::new();
        let mut means = Vec::new();
        for (row, visible) in hidden.iter().zip(&mask) {
            let mut mean = vec![0f32; row[0].len()];
            let mut count = 0f32;
            for (token, valid_token) in row.iter().zip(visible) {
                if *valid_token != 0 {
                    valid.extend_from_slice(token);
                    for (mean, value) in mean.iter_mut().zip(token) {
                        *mean += value;
                    }
                    count += 1.0;
                }
            }
            means.extend(mean.into_iter().map(|value| value / count));
        }
        assert_close(
            &valid,
            &floats(&case["valid_hidden"]),
            5e-5,
            5e-5,
            &format!("{mode}/tiny{index}/valid-hidden"),
        );
        assert_close(
            &means,
            &floats(&case["masked_mean"]),
            5e-5,
            5e-5,
            &format!("{mode}/tiny{index}/masked-mean"),
        );
        all_outputs.extend(valid);
    }
    Ok(all_outputs)
}

pub(crate) fn legacy_default_cache(
    dim: usize,
    positions: usize,
    theta: f64,
) -> Result<RotaryEmbedding> {
    let device = Device::Cpu;
    let frequencies: Vec<_> = (0..dim)
        .step_by(2)
        .map(|i| 1f32 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let frequencies = Tensor::from_vec(frequencies, (1, dim / 2), &device)?;
    let positions = Tensor::arange(0u32, positions as u32, &device)?
        .to_dtype(DType::F32)?
        .reshape((positions, 1))?;
    let angles = positions.matmul(&frequencies)?;
    Ok(RotaryEmbedding {
        sin: angles.sin()?,
        cos: angles.cos()?,
    })
}

pub(crate) fn assert_default_forward_unchanged(
    mut current: impl FnMut(&Tensor, &Tensor) -> Result<Tensor>,
    mut previous: impl FnMut(&Tensor, &Tensor) -> Result<Tensor>,
) -> Result<()> {
    for case in read_reference("tiny-output.json")["cases"]
        .as_array()
        .unwrap()
    {
        let ids: Vec<Vec<u32>> = serde_json::from_value(case["input_ids"].clone()).unwrap();
        let mask: Vec<Vec<u32>> = serde_json::from_value(case["attention_mask"].clone()).unwrap();
        let shape = (ids.len(), ids[0].len());
        let ids = Tensor::from_vec(ids.concat(), shape, &Device::Cpu)?;
        let mask = Tensor::from_vec(mask.concat(), shape, &Device::Cpu)?;
        let actual = current(&ids, &mask)?.flatten_all()?.to_vec1::<f32>()?;
        let expected = previous(&ids, &mask)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), expected.len());
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()));
    }
    Ok(())
}
