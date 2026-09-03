//! Unit tests for FFI embedding functions
//!
//! Following the repository's Rust test conventions:
//! - Test framework: rstest (parameterized testing)
//! - Concurrency control: serial_test (#[serial] for serial execution)
//! - File naming: embedding.rs → embedding_test.rs
//! - Location: Same directory as source file
//!
//! Note: These tests require the global ModelFactory to be initialized.
//! Use the `setup_embedding_models` fixture to initialize models before testing.

use super::embedding::*;
use crate::ffi::types::EmbeddingResult;
use crate::test_fixtures::fixtures::{
    GEMMA_EMBEDDING_300M, MODELS_BASE_PATH, QWEN3_EMBEDDING_0_6B,
};
use rstest::*;
use serial_test::serial;
use std::ffi::CString;
use std::sync::Once;

/// Global initializer to ensure ModelFactory is initialized once
static INIT: Once = Once::new();

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|v| v * v).sum::<f32>().sqrt()
}

#[test]
fn test_truncate_embedding_renormalizes_prefix() {
    let full = vec![0.6, 0.0, 0.8];

    let truncated = truncate_embedding_to_dimension(full, Some(1));

    assert_eq!(truncated, vec![1.0]);
    assert!(
        (l2_norm(&truncated) - 1.0).abs() < 1e-6,
        "truncated embedding must stay unit-normalized"
    );
}

/// Setup fixture: Initialize embedding models before tests
///
/// This fixture initializes the global ModelFactory with both Qwen3 and Gemma models.
/// It uses Once to ensure initialization happens only once across all tests.
#[fixture]
fn setup_embedding_models() {
    INIT.call_once(|| {
        let qwen3_path = format!("{}/{}", MODELS_BASE_PATH, QWEN3_EMBEDDING_0_6B);
        let gemma_path = format!("{}/{}", MODELS_BASE_PATH, GEMMA_EMBEDDING_300M);

        let qwen3_cstr = CString::new(qwen3_path.as_str()).unwrap();
        let gemma_cstr = CString::new(gemma_path.as_str()).unwrap();

        let success = init_embedding_models(qwen3_cstr.as_ptr(), gemma_cstr.as_ptr(), true);

        if !success {
            panic!("Failed to initialize embedding models for FFI tests");
        }

        println!("ModelFactory initialized for FFI tests");
    });
}

/// Test get_embedding_smart with valid medium text
#[rstest]
#[serial]
fn test_get_embedding_smart_medium_text(_setup_embedding_models: ()) {
    let text = CString::new("This is a medium length text with enough words to exceed 512 tokens when tokenized properly. Let's add more words to make sure we're in the medium range. More text here, and more, and even more to be safe.").unwrap();
    let mut result = EmbeddingResult {
        data: std::ptr::null_mut(),
        length: 0,
        error: false,
        model_type: -1,
        sequence_length: 0,
        processing_time_ms: 0.0,
    };

    let status = get_embedding_smart(text.as_ptr(), 0.5, 0.5, &mut result);

    assert_eq!(status, 0, "Should succeed");
    assert!(!result.error, "Should not have error");

    // Embedding dimension should be either 768 (Gemma) or 1024 (Qwen3)
    assert!(
        result.length == 768 || result.length == 1024,
        "Embedding dimension should be 768 (Gemma) or 1024 (Qwen3), got {}",
        result.length
    );

    assert!(!result.data.is_null(), "Data pointer should not be null");
    assert!(result.model_type >= 0, "Should have valid model_type");
    assert!(
        result.sequence_length > 0,
        "Should have valid sequence_length"
    );
    assert!(
        result.processing_time_ms >= 0.0,
        "Should have valid processing_time_ms"
    );

    // Cleanup
    if !result.data.is_null() && result.length > 0 {
        crate::ffi::memory::free_embedding(result.data, result.length);
    }
}

/// Test get_embedding_smart with different priority combinations
#[rstest]
#[case(0.9, 0.2)] // High quality priority
#[case(0.2, 0.9)] // High latency priority
#[case(0.5, 0.5)] // Balanced
#[serial]
fn test_get_embedding_smart_priority_combinations(
    _setup_embedding_models: (),
    #[case] quality_priority: f32,
    #[case] latency_priority: f32,
) {
    let text = CString::new("Test text").unwrap();
    let mut result = EmbeddingResult {
        data: std::ptr::null_mut(),
        length: 0,
        error: false,
        model_type: -1,
        sequence_length: 0,
        processing_time_ms: 0.0,
    };

    let status = get_embedding_smart(
        text.as_ptr(),
        quality_priority,
        latency_priority,
        &mut result,
    );

    assert_eq!(status, 0, "Should succeed with any valid priority");
    assert!(!result.error);

    // Embedding dimension should be either 768 (Gemma) or 1024 (Qwen3)
    assert!(
        result.length == 768 || result.length == 1024,
        "Embedding dimension should be 768 (Gemma) or 1024 (Qwen3), got {} for quality={}, latency={}",
        result.length, quality_priority, latency_priority
    );

    // Cleanup
    if !result.data.is_null() && result.length > 0 {
        crate::ffi::memory::free_embedding(result.data, result.length);
    }
}

/// The multimodal text encoder holds a fixed number of position embeddings, so
/// a longer sequence has to be clamped before it reaches the position lookup.
/// Regression for a long input failing with
/// "index-select invalid index 512 with dim size 512".
#[rstest]
#[case(0, 512, 0)]
#[case(1, 512, 1)]
#[case(511, 512, 511)]
#[case(512, 512, 512)]
#[case(513, 512, 512)]
#[case(4096, 512, 512)]
fn test_clamp_to_position_limit(
    #[case] input_len: usize,
    #[case] max_position: usize,
    #[case] want_len: usize,
) {
    let mut ids: Vec<u32> = (0..input_len as u32).collect();
    let mut mask: Vec<u32> = vec![1; input_len];

    clamp_to_position_limit(&mut ids, &mut mask, max_position);

    assert_eq!(ids.len(), want_len, "clamped ids length");
    assert_eq!(
        mask.len(),
        want_len,
        "mask must stay the same length as ids"
    );
    // The kept tokens are the start of the sequence, not an arbitrary window.
    assert!(ids.iter().enumerate().all(|(i, id)| *id == i as u32));
}
