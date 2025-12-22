//! Unit tests for Gemma3 Transformer Backbone
//!
//! This module tests the core components of the Gemma3 model:
//! - RmsNorm
//! - RotaryEmbeddingCache (RoPE with local base frequency)
//! - Gemma3Attention (MQA with mixed attention pattern)
//! - Gemma3MLP (gelu_pytorch_tanh activation)
//! - Gemma3Layer (pre-norm architecture)
//! - Gemma3Model (complete transformer backbone)
//!
//! ## Test Conventions
//! - Framework: `rstest` for parameterized tests
//! - Concurrency: `serial_test` for model loading tests
//! - Device: Uses `Device::Cpu` for unit tests
//! - Model Loading: Will use cached model from `test_fixtures` after full implementation

use crate::model_architectures::embedding::{
    AttentionLayerType, Gemma3Model, Gemma3RmsNorm as RmsNorm, Gemma3RoPE as RotaryEmbeddingCache,
    GemmaEmbeddingConfig,
};
use candle_core::{DType, Tensor};
use rstest::*;
use serial_test::serial;
use std::sync::Arc;

// Import test fixtures
use crate::test_fixtures::fixtures::{gemma3_model_only, test_device};

// ============================================================================
// Test Fixtures
// ============================================================================

/// Create a test GemmaEmbeddingConfig
#[fixture]
fn gemma_config() -> GemmaEmbeddingConfig {
    GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        num_hidden_layers: 24,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        intermediate_size: 1152,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![
            AttentionLayerType::SlidingAttention, // 0
            AttentionLayerType::SlidingAttention, // 1
            AttentionLayerType::SlidingAttention, // 2
            AttentionLayerType::SlidingAttention, // 3
            AttentionLayerType::SlidingAttention, // 4
            AttentionLayerType::FullAttention,    // 5
            AttentionLayerType::SlidingAttention, // 6
            AttentionLayerType::SlidingAttention, // 7
            AttentionLayerType::SlidingAttention, // 8
            AttentionLayerType::SlidingAttention, // 9
            AttentionLayerType::SlidingAttention, // 10
            AttentionLayerType::FullAttention,    // 11
            AttentionLayerType::SlidingAttention, // 12
            AttentionLayerType::SlidingAttention, // 13
            AttentionLayerType::SlidingAttention, // 14
            AttentionLayerType::SlidingAttention, // 15
            AttentionLayerType::SlidingAttention, // 16
            AttentionLayerType::FullAttention,    // 17
            AttentionLayerType::SlidingAttention, // 18
            AttentionLayerType::SlidingAttention, // 19
            AttentionLayerType::SlidingAttention, // 20
            AttentionLayerType::SlidingAttention, // 21
            AttentionLayerType::SlidingAttention, // 22
            AttentionLayerType::FullAttention,    // 23
        ],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    }
}

// ============================================================================
// RmsNorm Tests
// ============================================================================

#[rstest]
#[case(768, "Gemma hidden_size")]
#[case(1024, "Qwen3 hidden_size")]
#[serial]
fn test_rmsnorm_output_shape(#[case] hidden_size: usize, #[case] description: &str) {
    let device = test_device();
    let eps = 1e-6;

    // Create weight tensor
    let weight = Tensor::ones((hidden_size,), DType::F32, &device).unwrap();
    let rms_norm = RmsNorm::new(weight, eps);

    // Test input
    let input = Tensor::randn(0f32, 1f32, (2, 128, hidden_size), &device).unwrap();

    // Forward pass
    let output = rms_norm.forward(&input).unwrap();

    // Validate shape
    assert_eq!(
        output.dims(),
        &[2, 128, hidden_size],
        "Failed for {}",
        description
    );
    assert_eq!(output.dtype(), DType::F32);
}

#[rstest]
#[serial]
fn test_rmsnorm_zero_mean() {
    let device = test_device();
    let hidden_size = 768;
    let eps = 1e-6;

    // Create weight tensor (all zeros, because Gemma3 uses (1.0 + weight) scaling)
    let weight = Tensor::zeros((hidden_size,), DType::F32, &device).unwrap();
    let rms_norm = RmsNorm::new(weight, eps);

    // Test input with known values
    let input = Tensor::randn(0f32, 1f32, (1, 1, hidden_size), &device).unwrap();

    // Forward pass
    let output = rms_norm.forward(&input).unwrap();

    // RmsNorm should normalize the input such that RMS ≈ 1
    // Compute RMS of output: sqrt(mean(output^2))
    let output_squared = output.sqr().unwrap();
    let mean_squared = output_squared
        .mean_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap();
    let rms = mean_squared.sqrt();

    // RMS should be close to 1.0
    assert!(
        (rms - 1.0).abs() < 0.1,
        "RMS should be close to 1.0, got {}",
        rms
    );
}

#[rstest]
#[serial]
fn test_rmsnorm_numerical_properties() {
    let device = test_device();
    let hidden_size = 64;

    // Weight = 0.0 because Gemma3 uses (1.0 + weight) scaling
    let weight = Tensor::zeros((hidden_size,), DType::F32, &device).unwrap();
    let rms_norm = RmsNorm::new(weight, 1e-6);

    // Create input with known values
    let input = Tensor::ones((1, 1, hidden_size), DType::F32, &device).unwrap();

    let output = rms_norm.forward(&input).unwrap();

    // For input = [1, 1, ..., 1]:
    // mean(x^2) = 1
    // rms = sqrt(1 + eps) ≈ 1
    // output = input / rms * weight ≈ [1, 1, ..., 1]

    let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    // Check that output values are close to 1.0
    for (i, &val) in output_vec.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 0.01,
            "Output[{}] = {}, expected ~1.0",
            i,
            val
        );
    }
}

// ============================================================================
// RoPE (RotaryEmbeddingCache) Tests
// ============================================================================

#[rstest]
#[case(
    256,
    512,
    10000.0,
    "Gemma3: head_dim=256, max_len=512, local_base=10000"
)]
#[case(
    256,
    2048,
    10000.0,
    "Gemma3: head_dim=256, max_len=2048, local_base=10000"
)]
#[case(
    128,
    1024,
    10000.0,
    "Qwen3-like: head_dim=128, max_len=1024, local_base=10000"
)]
#[serial]
fn test_rope_cache_creation(
    #[case] head_dim: usize,
    #[case] max_seq_len: usize,
    #[case] rope_local_base_freq: f32,
    #[case] description: &str,
) {
    let device = test_device();

    // Create RoPE cache
    let result = RotaryEmbeddingCache::new(head_dim, max_seq_len, rope_local_base_freq, &device);

    // Validate that cache was created successfully
    assert!(
        result.is_ok(),
        "Failed for {}: {:?}",
        description,
        result.err()
    );
}

#[rstest]
#[serial]
fn test_rope_cache_odd_head_dim_fails() {
    let device = test_device();
    let head_dim = 127; // Odd number
    let max_seq_len = 512;
    let rope_local_base_freq = 10000.0;

    // Should fail with ValidationError
    let result = RotaryEmbeddingCache::new(head_dim, max_seq_len, rope_local_base_freq, &device);

    assert!(result.is_err(), "RoPE should reject odd head_dim");
}

#[rstest]
#[case(
    1,
    3,
    10,
    256,
    "Gemma3: batch=1, num_heads=3, seq_len=10, head_dim=256"
)]
#[case(
    2,
    3,
    50,
    256,
    "Gemma3: batch=2, num_heads=3, seq_len=50, head_dim=256"
)]
#[case(
    4,
    8,
    128,
    128,
    "Qwen3-like: batch=4, num_heads=8, seq_len=128, head_dim=128"
)]
#[serial]
fn test_rope_apply_output_shape(
    #[case] batch_size: usize,
    #[case] num_heads: usize,
    #[case] seq_len: usize,
    #[case] head_dim: usize,
    #[case] description: &str,
) {
    let device = test_device();
    let max_seq_len = 2048;
    let rope_local_base_freq = 10000.0;

    // Create RoPE cache
    let rope_cache =
        RotaryEmbeddingCache::new(head_dim, max_seq_len, rope_local_base_freq, &device).unwrap();

    // Create test input: [batch, num_heads, seq_len, head_dim]
    let q = Tensor::randn(
        0f32,
        1f32,
        (batch_size, num_heads, seq_len, head_dim),
        &device,
    )
    .unwrap();

    // Create position IDs: [batch, seq_len]
    let positions: Vec<u32> = (0..seq_len as u32).collect();
    let position_tensor = Tensor::from_vec(positions, (seq_len,), &device).unwrap();
    let position_ids = position_tensor
        .unsqueeze(0)
        .unwrap()
        .repeat(&[batch_size, 1])
        .unwrap();

    // Apply RoPE
    let q_rope = rope_cache.apply_rotary_emb(&q, &position_ids).unwrap();

    // Validate shape
    assert_eq!(
        q_rope.dims(),
        &[batch_size, num_heads, seq_len, head_dim],
        "Failed for {}",
        description
    );
    assert_eq!(q_rope.dtype(), DType::F32);
}

// ============================================================================
// Config and Attention Type Tests
// ============================================================================

#[rstest]
#[case(0, AttentionLayerType::SlidingAttention)]
#[case(5, AttentionLayerType::FullAttention)]
#[case(11, AttentionLayerType::FullAttention)]
#[case(17, AttentionLayerType::FullAttention)]
#[case(23, AttentionLayerType::FullAttention)]
#[serial]
fn test_gemma_attention_layer_type(
    gemma_config: GemmaEmbeddingConfig,
    #[case] layer_idx: usize,
    #[case] expected_type: AttentionLayerType,
) {
    let actual_type = gemma_config.get_layer_type(layer_idx);
    assert_eq!(actual_type, Some(expected_type));
}

#[rstest]
#[serial]
fn test_gemma_config_validates_mqa(gemma_config: GemmaEmbeddingConfig) {
    // Validate that config has MQA (num_key_value_heads = 1)
    assert_eq!(gemma_config.num_key_value_heads, 1);

    // Validate head_dim
    assert_eq!(gemma_config.head_dim, 256);

    // Validate sliding_window
    assert_eq!(gemma_config.sliding_window, 512);
}

// ============================================================================
// GemmaEmbeddingConfig Loading Test
// ============================================================================

/// Test loading actual GemmaEmbedding config
///
/// This test verifies loading the embeddinggemma-300m config
#[rstest]
#[serial]
fn test_load_gemma_config_valid() {
    let config = GemmaEmbeddingConfig::from_pretrained("../models/mom-embedding-flash").unwrap();

    // Validate critical parameters
    assert_eq!(config.vocab_size, 262144, "vocab_size should be 262144");
    assert_eq!(config.hidden_size, 768, "hidden_size should be 768");
    assert_eq!(
        config.num_hidden_layers, 24,
        "num_hidden_layers should be 24"
    );
    assert_eq!(
        config.num_attention_heads, 3,
        "num_attention_heads should be 3"
    );
    assert_eq!(
        config.num_key_value_heads, 1,
        "num_key_value_heads should be 1 (MQA)"
    );
    assert_eq!(config.head_dim, 256, "head_dim should be 256");
    assert_eq!(
        config.intermediate_size, 1152,
        "intermediate_size should be 1152"
    );
    assert_eq!(
        config.max_position_embeddings, 2048,
        "max_position_embeddings should be 2048"
    );
    assert_eq!(
        config.rope_theta, 1000000.0,
        "rope_theta should be 1000000.0"
    );
    assert_eq!(
        config.rope_local_base_freq, 10000.0,
        "rope_local_base_freq should be 10000.0"
    );
    assert_eq!(config.sliding_window, 512, "sliding_window should be 512");
    assert_eq!(
        config.layer_types.len(),
        24,
        "layer_types should have 24 elements"
    );

    // Validate that layer_types match the mixed attention pattern
    // Full attention layers: 5, 11, 17, 23
    assert!(config.is_full_attention_layer(5));
    assert!(config.is_full_attention_layer(11));
    assert!(config.is_full_attention_layer(17));
    assert!(config.is_full_attention_layer(23));

    // Sliding attention layers: all others
    assert!(!config.is_full_attention_layer(0));
    assert!(!config.is_full_attention_layer(1));
    assert!(!config.is_full_attention_layer(10));
    assert!(!config.is_full_attention_layer(12));
}

// ============================================================================
// Integration Test Placeholders (for future model loading)
// ============================================================================

/// Test loading the actual Gemma3 model
#[rstest]
#[serial(gemma3_model)]
fn test_gemma3_model_load(gemma3_model_only: Arc<Gemma3Model>) {
    println!("\n{}", "=".repeat(80));
    println!("Gemma3Model Load Test (using cached fixture)");
    println!("{}\n", "=".repeat(80));

    println!("  ✅ Gemma3Model loaded successfully via fixture");
    println!(
        "  Model config: {} layers, {} attention heads",
        gemma3_model_only.config().num_hidden_layers,
        gemma3_model_only.config().num_attention_heads
    );
    println!("  Device: {:?}", gemma3_model_only.device());
}

/// Test Gemma3 model forward pass
#[rstest]
#[serial(gemma3_model)]
fn test_gemma3_model_forward(gemma3_model_only: Arc<Gemma3Model>) {
    use candle_core::{DType, Tensor};

    println!("\n{}", "=".repeat(80));
    println!("Gemma3Model Forward Pass Test (using cached fixture)");
    println!("{}\n", "=".repeat(80));

    // Get device from model
    let device = gemma3_model_only.device();
    println!("  Using model device: {:?}", device);

    // Create test input
    let batch_size = 2;
    let seq_len = 128;

    println!(
        "  Creating test input: batch={}, seq_len={}",
        batch_size, seq_len
    );

    let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)
        .expect("Failed to create input_ids");
    let attention_mask = Tensor::ones((batch_size, seq_len), DType::U32, &device)
        .expect("Failed to create attention_mask");

    // Forward pass
    println!("  Running forward pass...");
    let output = gemma3_model_only
        .forward(&input_ids, Some(&attention_mask))
        .expect("Forward pass failed");

    println!("  Output shape: {:?}", output.dims());

    // Validate output shape: [batch, seq_len, hidden_size]
    assert_eq!(
        output.dims(),
        &[batch_size, seq_len, 768],
        "Output shape should be [batch={}, seq_len={}, hidden_size=768]",
        batch_size,
        seq_len
    );

    println!("  ✅ Forward pass test passed");
}
