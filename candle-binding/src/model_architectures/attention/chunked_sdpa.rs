//! Chunked scaled-dot-product attention (memory-bounded SDPA).
//!
//! A single, reusable attention kernel that processes the query dimension in
//! fixed-size blocks so the full `(b, heads, seq, seq)` score matrix is never
//! materialized — ~3.2 GB at 8K tokens and ~51 GB at 32K *per layer*, which is what
//! crashed the router on long inputs (issue #1957, fixed for mmBERT in #2007).
//!
//! [`chunked_sdpa`] operates on already-projected `q`/`k`/`v` (RoPE applied, GQA
//! repeated) so each model keeps its own projection quirks (RoPE theta, MQA/GQA
//! repeat, dtype) and only shares the memory-bounded loop. The output is numerically
//! identical to dense attention: every query attends to exactly the same keys with
//! the same pre-softmax scores; only the memory layout and arithmetic grouping differ.
//!
//! The kernel is dtype-agnostic — masks are converted to the score tensor's dtype
//! before being added, so an F64 path (e.g. Qwen3 embedding) works unchanged.

use candle_core::{DType, Device, Tensor, D};

/// Query-block size for chunked (memory-bounded) attention.
///
/// Attention is computed in blocks of this many query positions so the full
/// `seq×seq` score matrix is never materialized (~3.2 GB at 8K tokens), which is
/// what crashed the router on long inputs (issue #1957). 512 is a good CPU
/// memory/throughput trade-off; short inputs fit in a single block.
pub const ATTN_QUERY_BLOCK: usize = 512;

/// Configuration for [`chunked_sdpa`].
pub struct ChunkedSdpaConfig {
    /// Query block size. `0` means "no chunking" (a single block over the whole
    /// sequence); a zero is coerced to the sequence length so the loop always
    /// makes progress.
    pub block_size: usize,
    /// `Some(w)` = local sliding-window band (keep only `|i - j| <= w`),
    /// `None` = global attention (every query attends to every key).
    pub window: Option<usize>,
    /// Decoder causal masking: when `true`, a query at absolute index `i` attends
    /// only to keys `j <= i`. Composes with `window` (intersection of the causal
    /// triangle and the sliding-window band). Implemented for the same-length case
    /// (`q_len == k_len`); the KV-cache/decode-offset path (`k_len > q_len`) lands
    /// with the generative-model migration.
    pub causal: bool,
    /// Softmax scale applied to the queries, typically `head_dim^-0.5`.
    pub scale: f64,
}

/// Memory-bounded scaled-dot-product attention.
///
/// `q`, `k`, `v`: `(b, heads, seq, head_dim)` — RoPE already applied, GQA already
/// repeated. `pad_mask`: optional `(b, 1, 1, seq)` additive mask (`0` for real
/// tokens, large negative for padding) that broadcasts over query positions.
///
/// Returns `(b, heads, seq, head_dim)`. Numerically identical to dense SDPA; the
/// caller is responsible for the final transpose/reshape and output projection.
pub fn chunked_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pad_mask: Option<&Tensor>,
    cfg: &ChunkedSdpaConfig,
) -> candle_core::Result<Tensor> {
    let (_b, _heads, seq_len, _head_dim) = q.dims4()?;
    let device = q.device();

    // Fold the scale into the queries once (cheap, O(seq*d)) before chunking.
    let q = (q * cfg.scale)?.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    // A non-positive block size means "no chunking" (single block over the whole
    // sequence); guard against a zero so the loop always makes progress.
    let block = if cfg.block_size == 0 {
        seq_len.max(1)
    } else {
        cfg.block_size
    };

    let mut out_blocks: Vec<Tensor> = Vec::new();
    let mut qs = 0usize;
    while qs < seq_len {
        let blk = block.min(seq_len - qs);
        let qe = qs + blk;

        // Key/value range for this query block: a local layer only needs the
        // `±window` band around the block; a global layer needs every key. A causal
        // query at absolute index `i` never attends to keys `j > i`, so keys beyond
        // the block end (`qe`) are always masked — cap the upper bound there (a
        // correctness-preserving memory win, since those keys would softmax to zero).
        let (ks, ke) = match (cfg.window, cfg.causal) {
            (Some(w), false) => (qs.saturating_sub(w), (qe + w).min(seq_len)),
            (Some(w), true) => (qs.saturating_sub(w), qe),
            (None, false) => (0, seq_len),
            (None, true) => (0, qe),
        };
        let kw = ke - ks;

        let q_blk = q.narrow(2, qs, blk)?.contiguous()?; // (b, heads, blk, hd)
        let k_win = k.narrow(2, ks, kw)?;
        let v_win = v.narrow(2, ks, kw)?.contiguous()?; // (b, heads, kw, hd)
        let k_t = k_win.transpose(D::Minus2, D::Minus1)?.contiguous()?; // (b, heads, hd, kw)

        // (b, heads, blk, kw)
        let mut scores = q_blk.matmul(&k_t)?;

        // Padding mask slice over the key range: (b, 1, 1, kw), broadcasts.
        if let Some(pad_mask) = pad_mask {
            let pad_slice = pad_mask.narrow(D::Minus1, ks, kw)?;
            scores = scores.broadcast_add(&pad_slice)?;
        }

        // Sliding-window band: keep only |i - j| <= window within the key superset.
        if let Some(window) = cfg.window {
            let band =
                build_local_band_mask(qs, blk, ks, kw, window, device)?.to_dtype(scores.dtype())?;
            scores = scores.broadcast_add(&band)?;
        }

        // Causal triangle: keep only keys j <= i. Composes with the window band and
        // the padding mask by addition (-inf + anything = -inf).
        if cfg.causal {
            let causal = build_causal_mask(qs, blk, ks, kw, device)?.to_dtype(scores.dtype())?;
            scores = scores.broadcast_add(&causal)?;
        }

        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        out_blocks.push(probs.matmul(&v_win)?); // (b, heads, blk, hd)

        qs = qe;
    }

    Tensor::cat(&out_blocks, 2) // (b, heads, seq, hd)
}

/// Build the additive padding mask in `(b, 1, 1, seq)` form.
///
/// Real tokens map to `0`, padding tokens to a large negative value, so that
/// `scores.broadcast_add(pad_mask)` zeroes out padding after softmax. Unlike a
/// `(b, 1, seq, seq)` expansion, this keeps the mask O(seq) — the value depends only
/// on the key position and broadcasts over query positions.
pub fn prepare_padding_mask(mask: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    let expanded_mask = mask.unsqueeze(1)?.unsqueeze(2)?.to_dtype(dtype)?; // (b, 1, 1, seq)
    let inverted_mask = (1.0 - expanded_mask)?;
    (inverted_mask * f32::MIN as f64)?.to_dtype(dtype)
}

/// Build the additive sliding-window band mask for one query block.
///
/// Returns a `(q_len, k_len)` tensor (broadcasts over batch/heads) where entry
/// `(a, c)` — query at absolute index `q_start + a`, key at `k_start + c` — is `0`
/// when within the window (`|i - j| <= window`) and `-inf` otherwise. Called only
/// for local-attention layers; the key range is the per-block superset `[k_start,
/// k_start + k_len)`, so this mask removes the edge keys that fall outside an
/// individual query's window.
pub fn build_local_band_mask(
    q_start: usize,
    q_len: usize,
    k_start: usize,
    k_len: usize,
    window: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let window = window as i64;
    let mut mask = vec![0f32; q_len * k_len];
    for a in 0..q_len {
        let i = (q_start + a) as i64;
        for c in 0..k_len {
            let j = (k_start + c) as i64;
            if (i - j).abs() > window {
                mask[a * k_len + c] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_slice(&mask, (q_len, k_len), device)
}

/// Build the additive causal mask for one query block.
///
/// Returns a `(q_len, k_len)` tensor (broadcasts over batch/heads) where entry
/// `(a, c)` — query at absolute index `q_start + a`, key at `k_start + c` — is `0`
/// when the key is at or before the query (`j <= i`) and `-inf` otherwise. Taking
/// absolute `q_start`/`k_start` keeps this correct under a future decode offset
/// where the query window trails the cached keys (`k_start != q_start`).
pub fn build_causal_mask(
    q_start: usize,
    q_len: usize,
    k_start: usize,
    k_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut mask = vec![0f32; q_len * k_len];
    for a in 0..q_len {
        let i = (q_start + a) as i64;
        for c in 0..k_len {
            let j = (k_start + c) as i64;
            if j > i {
                mask[a * k_len + c] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_slice(&mask, (q_len, k_len), device)
}
