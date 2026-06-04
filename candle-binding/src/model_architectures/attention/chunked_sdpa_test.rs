//! Equivalence tests for the chunked-SDPA kernel.
//!
//! These exercise the free function in isolation (random `q`/`k`/`v`, no model) and
//! assert it is numerically identical to a dense reference that materializes the full
//! `(b, heads, seq, seq)` score matrix. Causal masking is reserved and not yet
//! implemented, so it is not tested here (it lands with the generative migration).

use super::chunked_sdpa::*;
use candle_core::{DType, Device, Tensor, D};

/// Dense reference attention: materializes the full `(b, heads, seq, seq)` score
/// matrix and applies the same additive masks as [`chunked_sdpa`].
fn dense_sdpa_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pad_mask: Option<&Tensor>,
    window: Option<usize>,
    scale: f64,
) -> Tensor {
    let (_b, _h, seq_len, _hd) = q.dims4().unwrap();
    let device = q.device();
    let q = (q * scale).unwrap().contiguous().unwrap();
    let k_t = k
        .transpose(D::Minus2, D::Minus1)
        .unwrap()
        .contiguous()
        .unwrap();
    let mut att = q.matmul(&k_t).unwrap(); // (b, heads, seq, seq)
    if let Some(pad_mask) = pad_mask {
        att = att.broadcast_add(pad_mask).unwrap();
    }
    if let Some(window) = window {
        let band = build_local_band_mask(0, seq_len, 0, seq_len, window, device)
            .unwrap()
            .to_dtype(att.dtype())
            .unwrap();
        att = att.broadcast_add(&band).unwrap();
    }
    let att = candle_nn::ops::softmax(&att, D::Minus1).unwrap();
    let v = v.contiguous().unwrap();
    att.matmul(&v).unwrap() // (b, heads, seq, hd)
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    a.broadcast_sub(b)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

/// Random `(b=1, heads, seq, head_dim)` q/k/v with a fixed shape.
fn random_qkv(
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &Device,
) -> (Tensor, Tensor, Tensor) {
    let shape = (1, heads, seq_len, head_dim);
    (
        Tensor::randn(0f32, 1f32, shape, device).unwrap(),
        Tensor::randn(0f32, 1f32, shape, device).unwrap(),
        Tensor::randn(0f32, 1f32, shape, device).unwrap(),
    )
}

#[test]
fn test_chunked_sdpa_matches_dense() {
    let device = Device::Cpu;
    let heads = 4;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    let window_size = 4;

    // Cover global + local paths, several block sizes (divisor, non-divisor,
    // single-block, block smaller than window, block larger than seq).
    for window in [None, Some(window_size)] {
        for &seq_len in &[1usize, 5, 16, 40] {
            let (q, k, v) = random_qkv(heads, seq_len, head_dim, &device);
            let reference = dense_sdpa_reference(&q, &k, &v, None, window, scale);

            for &block in &[1usize, 3, 8, 16, 512] {
                let cfg = ChunkedSdpaConfig {
                    block_size: block,
                    window,
                    causal: false,
                    scale,
                };
                let chunked = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
                let diff = max_abs_diff(&chunked, &reference);
                assert!(
                    diff < 1e-4,
                    "window={:?} seq={} block={}: max|Δ|={}",
                    window,
                    seq_len,
                    block,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_chunked_sdpa_matches_dense_with_padding() {
    let device = Device::Cpu;
    let heads = 4;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    let window_size = 4;
    let seq_len = 24;

    // Last 7 positions are padding.
    let mut mask_vec = vec![1u32; seq_len];
    for m in mask_vec.iter_mut().skip(seq_len - 7) {
        *m = 0;
    }
    let raw_mask = Tensor::from_vec(mask_vec, (1, seq_len), &device).unwrap();
    let pad = prepare_padding_mask(&raw_mask, DType::F32).unwrap();

    let (q, k, v) = random_qkv(heads, seq_len, head_dim, &device);

    for window in [None, Some(window_size)] {
        let reference = dense_sdpa_reference(&q, &k, &v, Some(&pad), window, scale);
        for &block in &[3usize, 8, 512] {
            let cfg = ChunkedSdpaConfig {
                block_size: block,
                window,
                causal: false,
                scale,
            };
            let chunked = chunked_sdpa(&q, &k, &v, Some(&pad), &cfg).unwrap();
            let diff = max_abs_diff(&chunked, &reference);
            assert!(
                diff < 1e-4,
                "padding window={:?} block={}: max|Δ|={}",
                window,
                block,
                diff
            );
        }
    }
}

#[test]
fn test_chunked_sdpa_matches_dense_f64() {
    // The kernel is dtype-agnostic: an F64 path (e.g. Qwen3 embedding) must work
    // because the band mask is converted to the score dtype before being added.
    let device = Device::Cpu;
    let heads = 2;
    let head_dim = 8;
    let seq_len = 20;
    let scale = (head_dim as f64).powf(-0.5);
    let window = Some(4usize);

    let shape = (1, heads, seq_len, head_dim);
    let q = Tensor::randn(0f64, 1f64, shape, &device).unwrap();
    let k = Tensor::randn(0f64, 1f64, shape, &device).unwrap();
    let v = Tensor::randn(0f64, 1f64, shape, &device).unwrap();

    let reference = dense_sdpa_reference(&q, &k, &v, None, window, scale);
    for &block in &[3usize, 8, 512] {
        let cfg = ChunkedSdpaConfig {
            block_size: block,
            window,
            causal: false,
            scale,
        };
        let chunked = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
        assert_eq!(chunked.dtype(), DType::F64);
        let diff = max_abs_diff(&chunked, &reference);
        assert!(diff < 1e-4, "f64 block={}: max|Δ|={}", block, diff);
    }
}

#[test]
fn test_band_mask_window_semantics() {
    let device = Device::Cpu;
    // Offset block: queries [10,14), keys [6,20), window 4.
    let band = build_local_band_mask(10, 4, 6, 14, 4, &device).unwrap();
    assert_eq!(band.dims(), &[4, 14]);
    let data: Vec<f32> = band.flatten_all().unwrap().to_vec1().unwrap();
    for a in 0..4usize {
        let i = 10 + a as i64;
        for c in 0..14usize {
            let j = 6 + c as i64;
            let v = data[a * 14 + c];
            if (i - j).abs() > 4 {
                assert!(
                    v.is_infinite() && v.is_sign_negative(),
                    "expected -inf at ({a},{c})"
                );
            } else {
                assert_eq!(v, 0.0, "expected 0 at ({a},{c})");
            }
        }
    }
}
