//! Tests for pooling implementations
//!
//! This test file validates the three pooling methods:
//! - mean_pool: Mean pooling with attention mask
//! - last_token_pool: Last token pooling (Qwen3)
//! - cls_pool: CLS token pooling

use super::pooling::*;
use candle_core::{DType, IndexOp, Tensor};
use rstest::*;
use serial_test::serial;

// Import test fixture
use crate::test_fixtures::fixtures::test_device;

/// Test mean pooling with normal case
#[rstest]
#[serial]
fn test_mean_pool_normal() {
    let device = test_device();

    // Create dummy hidden states: [2, 10, 768]
    let hidden = Tensor::randn(0f32, 1.0, (2, 10, 768), &device).unwrap();

    // All tokens are valid
    let mask = Tensor::ones((2, 10), DType::F32, &device).unwrap();

    let pooled = mean_pool(&hidden, &mask).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[2, 768]);
}

/// Test mean pooling with partial masking
#[rstest]
#[serial]
fn test_mean_pool_with_masking() {
    let device = test_device();

    // Create dummy hidden states: [2, 5, 8]
    let hidden = Tensor::randn(0f32, 1.0, (2, 5, 8), &device).unwrap();

    // First sequence: 3 valid tokens, second: 5 valid tokens
    let mask_data = vec![
        vec![1.0f32, 1.0, 1.0, 0.0, 0.0],
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0],
    ];
    let mask = Tensor::new(mask_data, &device).unwrap();

    let pooled = mean_pool(&hidden, &mask).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[2, 8]);
}

/// Test mean pooling edge case: single token
#[rstest]
#[serial]
fn test_mean_pool_single_token() {
    let device = test_device();

    // Single token per sequence
    let hidden = Tensor::randn(0f32, 1.0, (2, 1, 768), &device).unwrap();
    let mask = Tensor::ones((2, 1), DType::F32, &device).unwrap();

    let pooled = mean_pool(&hidden, &mask).unwrap();

    // Output should match input (no averaging needed)
    assert_eq!(pooled.dims(), &[2, 768]);
}

/// Test last token pooling with parametrized masks
#[rstest]
#[case(vec![1.0, 1.0, 1.0, 0.0, 0.0], 2)] // Should select index 2
#[case(vec![1.0, 1.0, 1.0, 1.0, 1.0], 4)] // Should select index 4
#[case(vec![1.0, 0.0, 0.0, 0.0, 0.0], 0)] // Should select index 0
#[serial]
fn test_last_token_pool_single(#[case] mask_values: Vec<f32>, #[case] expected_idx: usize) {
    let device = test_device();

    // Create hidden states: [1, 5, 8]
    let hidden_data: Vec<f32> = (0..40).map(|i| i as f32 / 10.0).collect();
    let hidden = Tensor::from_vec(hidden_data, (1, 5, 8), &device).unwrap();

    // Create mask from vector
    let mask = Tensor::from_vec(mask_values, (1, 5), &device).unwrap();

    let pooled = last_token_pool(&hidden, &mask).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[1, 8]);

    // Verify we extracted the correct token
    let expected_token = hidden.i((0, expected_idx)).unwrap();
    let pooled_data = pooled.i(0).unwrap().to_vec1::<f32>().unwrap();
    let expected_data = expected_token.to_vec1::<f32>().unwrap();

    for (p, e) in pooled_data.iter().zip(expected_data.iter()) {
        assert!((p - e).abs() < 1e-6, "Mismatch: got {}, expected {}", p, e);
    }
}

/// Test last token pooling with batch and different lengths
#[rstest]
#[serial]
fn test_last_token_pool_batch() {
    let device = test_device();

    // Create hidden states: [2, 10, 768]
    let hidden = Tensor::randn(0f32, 1.0, (2, 10, 768), &device).unwrap();

    // First sequence: 5 valid tokens (last at index 4)
    // Second sequence: 8 valid tokens (last at index 7)
    let mask_data = vec![
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
    ];
    let mask = Tensor::new(mask_data, &device).unwrap();

    let pooled = last_token_pool(&hidden, &mask).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[2, 768]);
}

/// Test last token pooling edge case: all tokens valid
#[rstest]
#[serial]
fn test_last_token_pool_all_valid() {
    let device = test_device();

    let hidden = Tensor::randn(0f32, 1.0, (3, 20, 512), &device).unwrap();
    let mask = Tensor::ones((3, 20), DType::F32, &device).unwrap();

    let pooled = last_token_pool(&hidden, &mask).unwrap();

    // Should extract index 19 (last token) for all batches
    assert_eq!(pooled.dims(), &[3, 512]);
}

/// Test CLS token pooling
#[rstest]
#[serial]
fn test_cls_pool_normal() {
    let device = test_device();

    // Create hidden states: [2, 10, 768]
    let hidden = Tensor::randn(0f32, 1.0, (2, 10, 768), &device).unwrap();

    let pooled = cls_pool(&hidden).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[2, 768]);
}

/// Test CLS token pooling - verify it extracts first token
#[rstest]
#[serial]
fn test_cls_pool_extracts_first_token() {
    let device = test_device();

    // Create known hidden states: [1, 5, 4]
    let hidden_data = vec![
        // Token 0 (CLS)
        1.0f32, 2.0, 3.0, 4.0, // Token 1
        5.0, 6.0, 7.0, 8.0, // Token 2
        9.0, 10.0, 11.0, 12.0, // Token 3
        13.0, 14.0, 15.0, 16.0, // Token 4
        17.0, 18.0, 19.0, 20.0,
    ];
    let hidden = Tensor::from_vec(hidden_data, (1, 5, 4), &device).unwrap();

    let pooled = cls_pool(&hidden).unwrap();

    // Check output shape
    assert_eq!(pooled.dims(), &[1, 4]);

    // Verify we extracted the first token (CLS)
    let pooled_data = pooled.to_vec2::<f32>().unwrap();
    assert_eq!(pooled_data[0], vec![1.0, 2.0, 3.0, 4.0]);
}

/// Test CLS pooling with batch
#[rstest]
#[serial]
fn test_cls_pool_batch() {
    let device = test_device();

    let hidden = Tensor::randn(0f32, 1.0, (4, 15, 512), &device).unwrap();

    let pooled = cls_pool(&hidden).unwrap();

    // Should extract first token for all batches
    assert_eq!(pooled.dims(), &[4, 512]);
}

/// Performance test: 32K sequence length (Qwen3 use case)
#[rstest]
#[serial]
#[ignore] // Run with --ignored flag for performance testing
fn test_last_token_pool_32k_sequence() {
    let device = test_device();

    // Simulate 32K context (Qwen3 max length)
    let seq_len = 32768;
    let batch_size = 2;
    let hidden_size = 768;

    println!("Testing last_token_pool with 32K sequence length...");
    let start = std::time::Instant::now();

    let hidden = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let mask = Tensor::ones((batch_size, seq_len), DType::F32, &device).unwrap();

    let pooled = last_token_pool(&hidden, &mask).unwrap();

    let duration = start.elapsed();
    println!("32K sequence pooling took: {:?}", duration);

    // Check output shape
    assert_eq!(pooled.dims(), &[batch_size, hidden_size]);

    // Performance expectation: CPU performance (without GPU acceleration)
    // Real-world: Flash Attention 2 on GPU would be much faster
    assert!(
        duration.as_secs() < 30,
        "32K pooling too slow: {:?}",
        duration
    );
}

/// Performance test: Mean pooling with large batch
#[rstest]
#[serial]
#[ignore] // Run with --ignored flag for performance testing
fn test_mean_pool_large_batch() {
    let device = test_device();

    let batch_size = 64;
    let seq_len = 512;
    let hidden_size = 768;

    println!("Testing mean_pool with large batch (64 × 512)...");
    let start = std::time::Instant::now();

    let hidden = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), &device).unwrap();
    let mask = Tensor::ones((batch_size, seq_len), DType::F32, &device).unwrap();

    let pooled = mean_pool(&hidden, &mask).unwrap();

    let duration = start.elapsed();
    println!("Large batch mean pooling took: {:?}", duration);

    // Check output shape
    assert_eq!(pooled.dims(), &[batch_size, hidden_size]);

    // Performance expectation: CPU performance
    // Should complete in reasonable time even on CPU
    assert!(
        duration.as_secs() < 30,
        "Mean pooling too slow: {:?}",
        duration
    );
}
