//! Unit tests for FFI embedding functions
//!
//! Following .cursorrules Line 20-25 specifications:
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

        println!("✅ ModelFactory initialized for FFI tests");
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
    assert_eq!(result.error, false, "Should not have error");

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
    assert_eq!(result.error, false);

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
