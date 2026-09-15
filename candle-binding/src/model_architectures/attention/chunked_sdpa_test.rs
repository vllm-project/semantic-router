//! Equivalence tests for the chunked-SDPA kernel.
//!
//! These exercise the free function in isolation (random `q`/`k`/`v`, no model) and
//! assert it is numerically identical to a dense reference that materializes the full
//! `(b, heads, seq, seq)` score matrix — across global, sliding-window, and causal
//! masking (and their combinations with padding).

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
    causal: bool,
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
    if causal {
        let tri = build_causal_mask(0, seq_len, 0, seq_len, device)
            .unwrap()
            .to_dtype(att.dtype())
            .unwrap();
        att = att.broadcast_add(&tri).unwrap();
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

    // Cover global + local paths, several block sizes (zero-coercion, divisor,
    // non-divisor, single-block, block smaller than window, block larger than seq).
    // `block == 0` exercises the "no chunking" coercion path (coerced to seq_len).
    for window in [None, Some(window_size)] {
        for &seq_len in &[1usize, 5, 16, 40] {
            let (q, k, v) = random_qkv(heads, seq_len, head_dim, &device);
            let reference = dense_sdpa_reference(&q, &k, &v, None, window, false, scale);

            for &block in &[0usize, 1, 3, 8, 16, 512] {
                let cfg = ChunkedSdpaConfig {
                    block_size: block,
                    window,
                    causal: false,
                    scale,
                    q_offset: 0,
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
        let reference = dense_sdpa_reference(&q, &k, &v, Some(&pad), window, false, scale);
        for &block in &[3usize, 8, 512] {
            let cfg = ChunkedSdpaConfig {
                block_size: block,
                window,
                causal: false,
                scale,
                q_offset: 0,
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

    let reference = dense_sdpa_reference(&q, &k, &v, None, window, false, scale);
    for &block in &[3usize, 8, 512] {
        let cfg = ChunkedSdpaConfig {
            block_size: block,
            window,
            causal: false,
            scale,
            q_offset: 0,
        };
        let chunked = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
        assert_eq!(chunked.dtype(), DType::F64);
        let diff = max_abs_diff(&chunked, &reference);
        assert!(diff < 1e-4, "f64 block={}: max|Δ|={}", block, diff);
    }
}

#[test]
fn test_chunked_sdpa_matches_dense_causal() {
    let device = Device::Cpu;
    let heads = 4;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    let window_size = 4;

    // Causal global + causal sliding-window (intersection of the triangle and the
    // band), across the same block/seq sweep as the non-causal test.
    for window in [None, Some(window_size)] {
        for &seq_len in &[1usize, 5, 16, 40] {
            let (q, k, v) = random_qkv(heads, seq_len, head_dim, &device);
            let reference = dense_sdpa_reference(&q, &k, &v, None, window, true, scale);

            for &block in &[0usize, 1, 3, 8, 16, 512] {
                let cfg = ChunkedSdpaConfig {
                    block_size: block,
                    window,
                    causal: true,
                    scale,
                    q_offset: 0,
                };
                let chunked = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
                let diff = max_abs_diff(&chunked, &reference);
                assert!(
                    diff < 1e-4,
                    "causal window={:?} seq={} block={}: max|Δ|={}",
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
fn test_chunked_sdpa_matches_dense_causal_with_padding() {
    let device = Device::Cpu;
    let heads = 4;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    let window_size = 4;
    let seq_len = 24;

    // Trailing padding: a causal query at a real position still attends to real
    // earlier keys, so no row is all -inf (no softmax NaN).
    let mut mask_vec = vec![1u32; seq_len];
    for m in mask_vec.iter_mut().skip(seq_len - 7) {
        *m = 0;
    }
    let raw_mask = Tensor::from_vec(mask_vec, (1, seq_len), &device).unwrap();
    let pad = prepare_padding_mask(&raw_mask, DType::F32).unwrap();

    let (q, k, v) = random_qkv(heads, seq_len, head_dim, &device);

    for window in [None, Some(window_size)] {
        let reference = dense_sdpa_reference(&q, &k, &v, Some(&pad), window, true, scale);
        for &block in &[3usize, 8, 512] {
            let cfg = ChunkedSdpaConfig {
                block_size: block,
                window,
                causal: true,
                scale,
                q_offset: 0,
            };
            let chunked = chunked_sdpa(&q, &k, &v, Some(&pad), &cfg).unwrap();
            let diff = max_abs_diff(&chunked, &reference);
            assert!(
                diff < 1e-4,
                "causal+padding window={:?} block={}: max|Δ|={}",
                window,
                block,
                diff
            );
        }
    }
}

#[test]
fn test_causal_mask_semantics() {
    let device = Device::Cpu;
    // Offset block: queries [10,14), keys [6,20).
    let mask = build_causal_mask(10, 4, 6, 14, &device).unwrap();
    assert_eq!(mask.dims(), &[4, 14]);
    let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
    for a in 0..4usize {
        let i = 10 + a as i64;
        for c in 0..14usize {
            let j = 6 + c as i64;
            let v = data[a * 14 + c];
            if j > i {
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

/// Dense reference for a query block that trails cached keys: `q` holds `q_len`
/// queries at absolute positions `[offset, offset + q_len)`, `k`/`v` hold
/// `offset + q_len` keys. Mirrors the decoder's `causal_mask` (`j <= i + offset`,
/// optionally `i + offset - j <= window`).
fn dense_offset_reference(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    offset: usize,
    window: Option<usize>,
    causal: bool,
    scale: f64,
) -> Tensor {
    let (_b, _h, q_len, _hd) = q.dims4().unwrap();
    let k_len = k.dim(2).unwrap();
    let device = q.device();
    let q = (q * scale).unwrap().contiguous().unwrap();
    let k_t = k
        .transpose(D::Minus2, D::Minus1)
        .unwrap()
        .contiguous()
        .unwrap();
    let mut att = q.matmul(&k_t).unwrap(); // (b, heads, q_len, k_len)
    let mut mask = vec![0f32; q_len * k_len];
    for a in 0..q_len {
        let i = (offset + a) as i64;
        for c in 0..k_len {
            let j = c as i64;
            let past_ok = !causal || j <= i;
            let band_ok = window.is_none_or(|w| (i - j).abs() <= w as i64);
            if !(past_ok && band_ok) {
                mask[a * k_len + c] = f32::NEG_INFINITY;
            }
        }
    }
    let mask = Tensor::from_slice(&mask, (q_len, k_len), device)
        .unwrap()
        .to_dtype(att.dtype())
        .unwrap();
    att = att.broadcast_add(&mask).unwrap();
    let att = candle_nn::ops::softmax(&att, D::Minus1).unwrap();
    att.matmul(&v.contiguous().unwrap()).unwrap()
}

#[test]
fn test_chunked_sdpa_single_query_over_many_keys() {
    // A pooling probe: one query attends to every key (SigLIP attention pooling).
    let device = Device::Cpu;
    let heads = 2;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    for &k_len in &[1usize, 7, 40] {
        let q = Tensor::randn(0f32, 1f32, (1, heads, 1, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, heads, k_len, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, heads, k_len, head_dim), &device).unwrap();
        let reference = dense_offset_reference(&q, &k, &v, 0, None, false, scale);
        for &block in &[0usize, 1, 512] {
            let cfg = ChunkedSdpaConfig {
                block_size: block,
                window: None,
                causal: false,
                scale,
                q_offset: 0,
            };
            let out = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
            assert_eq!(out.dims(), &[1, heads, 1, head_dim]);
            let diff = max_abs_diff(&out, &reference);
            assert!(
                diff < 1e-4,
                "k_len={} block={}: max|Δ|={}",
                k_len,
                block,
                diff
            );
        }
    }
}

#[test]
fn test_chunked_sdpa_matches_dense_with_decode_offset() {
    // A decode step: `q_len` new queries at absolute positions `[offset, offset+q_len)`
    // over `offset + q_len` cached keys, causal, with and without a window.
    let device = Device::Cpu;
    let heads = 4;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    for window in [None, Some(3usize)] {
        for &(offset, q_len) in &[(0usize, 9usize), (5, 1), (5, 4), (20, 3), (17, 8)] {
            let k_len = offset + q_len;
            let q = Tensor::randn(0f32, 1f32, (1, heads, q_len, head_dim), &device).unwrap();
            let k = Tensor::randn(0f32, 1f32, (1, heads, k_len, head_dim), &device).unwrap();
            let v = Tensor::randn(0f32, 1f32, (1, heads, k_len, head_dim), &device).unwrap();
            let reference = dense_offset_reference(&q, &k, &v, offset, window, true, scale);
            for &block in &[0usize, 1, 2, 512] {
                let cfg = ChunkedSdpaConfig {
                    block_size: block,
                    window,
                    causal: true,
                    scale,
                    q_offset: offset,
                };
                let out = chunked_sdpa(&q, &k, &v, None, &cfg).unwrap();
                let diff = max_abs_diff(&out, &reference);
                assert!(
                    diff < 1e-4,
                    "window={:?} offset={} q_len={} block={}: max|Δ|={}",
                    window,
                    offset,
                    q_len,
                    block,
                    diff
                );
            }
        }
    }
}

#[test]
fn test_chunked_sdpa_offset_prefill_equals_split_prefill() {
    // Prefilling 12 positions at once must equal prefilling 7 and then 5 more with
    // `q_offset = 7` over the full key set — the invariant a KV cache relies on.
    let device = Device::Cpu;
    let heads = 2;
    let head_dim = 8;
    let scale = (head_dim as f64).powf(-0.5);
    let (q, k, v) = random_qkv(heads, 12, head_dim, &device);
    let cfg = |offset: usize| ChunkedSdpaConfig {
        block_size: 4,
        window: None,
        causal: true,
        scale,
        q_offset: offset,
    };
    let whole = chunked_sdpa(&q, &k, &v, None, &cfg(0)).unwrap();
    let head = chunked_sdpa(
        &q.narrow(2, 0, 7).unwrap(),
        &k.narrow(2, 0, 7).unwrap(),
        &v.narrow(2, 0, 7).unwrap(),
        None,
        &cfg(0),
    )
    .unwrap();
    let tail = chunked_sdpa(&q.narrow(2, 7, 5).unwrap(), &k, &v, None, &cfg(7)).unwrap();
    let split = Tensor::cat(&[head, tail], 2).unwrap();
    let diff = max_abs_diff(&whole, &split);
    assert!(diff < 1e-5, "split prefill diverges: max|Δ|={}", diff);
}

#[test]
fn test_chunked_sdpa_rejects_a_block_with_no_keys() {
    // A windowed query block that starts past every key has nothing to attend to;
    // that is a caller bug and must not become a NaN row.
    let device = Device::Cpu;
    let q = Tensor::randn(0f32, 1f32, (1, 1, 2, 4), &device).unwrap();
    let k = Tensor::randn(0f32, 1f32, (1, 1, 3, 4), &device).unwrap();
    let cfg = ChunkedSdpaConfig {
        block_size: 0,
        window: Some(1),
        causal: false,
        scale: 0.5,
        q_offset: 10,
    };
    assert!(chunked_sdpa(&q, &k, &k, None, &cfg).is_err());
}
