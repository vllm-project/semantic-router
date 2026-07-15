//! Shared attention mechanisms for candle models.
//!
//! [`chunked_sdpa`] is a memory-bounded scaled-dot-product attention kernel that
//! processes the query dimension in fixed-size blocks so the full
//! `(b, heads, seq, seq)` score matrix is never materialized (issue #1957 / #2007).
//! Models call it after their own QKV projection / RoPE / GQA-repeat, sharing only
//! the memory-bounded loop while keeping their architecture-specific quirks.

pub mod chunked_sdpa;

pub use chunked_sdpa::{
    build_causal_mask, build_local_band_mask, chunked_sdpa, prepare_padding_mask,
    ChunkedSdpaConfig, ATTN_QUERY_BLOCK,
};

#[cfg(test)]
mod chunked_sdpa_test;
