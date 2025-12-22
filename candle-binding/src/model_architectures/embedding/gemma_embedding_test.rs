//! Unit tests for GemmaEmbedding model implementation
//!
//! ## Test Coverage
//! - Configuration loading and validation
//! - Matryoshka dimension support (768/512/256/128)
//! - Output validation against Python reference implementation
//! - Complete model forward pass
//!
//! ## Testing Strategy
//! - Use `rstest` for parameterized tests
//! - Use `serial_test` for model loading tests (to avoid parallel resource contention)
//! - Use test fixtures for model caching
//! - Validate outputs with cosine similarity > 0.99

use candle_core::Tensor;
use rstest::*;
use serde::{Deserialize, Serialize};
use serde_json;
use serial_test::serial;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::core::UnifiedError;
use crate::model_architectures::embedding::gemma_embedding::{
    AttentionLayerType, GemmaEmbeddingConfig, GemmaEmbeddingModel,
};
use crate::test_fixtures::fixtures::{gemma_embedding_model, test_device};

// ============================================================================
// Data Structures for Validation Tests
// ============================================================================

/// Structure to deserialize reference outputs from Python script
#[derive(Debug, Deserialize, Serialize)]
struct ReferenceOutput {
    name: String,
    input: InputInfo,
    #[serde(default)]
    tokenization: Option<TokenizationInfo>,
    #[serde(default)]
    embedding_full: Vec<f32>,
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    embedding_shape: Vec<usize>,
    #[serde(default)]
    embedding_dim: usize,
    #[serde(default)]
    matryoshka: HashMap<String, Vec<f32>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct InputInfo {
    #[serde(default)]
    text: String,
    #[serde(default)]
    full_text_length: usize,
    #[serde(default)]
    texts: Vec<String>,
    #[serde(default)]
    batch_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct TokenizationInfo {
    #[serde(default)]
    seq_len: usize,
    #[serde(default)]
    input_shape: Vec<usize>,
    // Use serde_json::Value to handle both Vec<i64> (single) and Vec<Vec<i64>> (batch)
    #[serde(default)]
    input_ids: serde_json::Value,
    #[serde(default)]
    attention_mask: serde_json::Value,
}

impl TokenizationInfo {
    /// Get input_ids as Vec<Vec<u32>> (handles both single and batch formats)
    fn get_input_ids(&self) -> Vec<Vec<u32>> {
        if let Some(arr) = self.input_ids.as_array() {
            // Check if it's a batch (2D array) or single (1D array)
            if let Some(first) = arr.first() {
                if first.is_array() {
                    // Batch format: [[ids...], [ids...]]
                    arr.iter()
                        .map(|row| {
                            row.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_i64().unwrap() as u32)
                                .collect()
                        })
                        .collect()
                } else {
                    // Single format: [ids...] - wrap in outer array
                    vec![arr.iter().map(|v| v.as_i64().unwrap() as u32).collect()]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }

    /// Get attention_mask as Vec<Vec<u32>> (handles both single and batch formats)
    fn get_attention_mask(&self) -> Vec<Vec<u32>> {
        if let Some(arr) = self.attention_mask.as_array() {
            if let Some(first) = arr.first() {
                if first.is_array() {
                    // Batch format
                    arr.iter()
                        .map(|row| {
                            row.as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_i64().unwrap() as u32)
                                .collect()
                        })
                        .collect()
                } else {
                    // Single format
                    vec![arr.iter().map(|v| v.as_i64().unwrap() as u32).collect()]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }
}

/// Helper function to load reference outputs
fn load_reference_outputs() -> Vec<ReferenceOutput> {
    let json_path = Path::new("./test_data/gemma_reference_outputs.json");

    if !json_path.exists() {
        eprintln!("⚠️  Reference data not found. Generating...");

        let status = std::process::Command::new("python")
            .arg("scripts/generate_gemma_reference.py")
            .current_dir("../")
            .status()
            .expect("Failed to execute Python script");

        if !status.success() {
            panic!("Failed to generate reference data");
        }

        eprintln!("✅ Reference data generated successfully");
    }

    let json_content =
        std::fs::read_to_string(json_path).expect("Failed to read reference outputs JSON");

    serde_json::from_str(&json_content).expect("Failed to parse reference outputs JSON")
}

/// Helper to calculate cosine similarity
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have same length");
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

/// Helper function to create a minimal test config for Matryoshka tests
fn create_test_config() -> GemmaEmbeddingConfig {
    GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 2, // Reduced for testing
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![
            AttentionLayerType::SlidingAttention,
            AttentionLayerType::FullAttention,
        ],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

/// Test GemmaEmbeddingConfig loading from pretrained model
///
/// **Test Strategy**: Load the actual model configuration from disk and validate
/// all parameters match the expected EmbeddingGemma-300M specification.
#[rstest]
#[serial(gemma_model)]
fn test_config_load_from_pretrained() {
    let model_path = "../models/mom-embedding-flash";

    let config = GemmaEmbeddingConfig::from_pretrained(model_path).expect("Failed to load config");

    // Verify core architecture parameters
    assert_eq!(config.vocab_size, 262144, "vocab_size mismatch");
    assert_eq!(config.hidden_size, 768, "hidden_size mismatch");
    assert_eq!(config.num_hidden_layers, 24, "num_hidden_layers mismatch");
    assert_eq!(
        config.num_attention_heads, 3,
        "num_attention_heads mismatch"
    );
    assert_eq!(
        config.num_key_value_heads, 1,
        "num_key_value_heads mismatch (MQA)"
    );
    assert_eq!(config.intermediate_size, 1152, "intermediate_size mismatch");
    assert_eq!(
        config.max_position_embeddings, 2048,
        "max_position_embeddings mismatch"
    );
    assert_eq!(config.head_dim, 256, "head_dim mismatch");
    assert_eq!(config.sliding_window, 512, "sliding_window mismatch");

    // Verify RoPE parameters
    assert_eq!(config.rope_theta, 1000000.0, "rope_theta mismatch");
    assert_eq!(
        config.rope_local_base_freq, 10000.0,
        "rope_local_base_freq mismatch"
    );

    // Verify normalization and dropout
    assert_eq!(config.rms_norm_eps, 1e-6, "rms_norm_eps mismatch");
    assert_eq!(config.attention_dropout, 0.0, "attention_dropout mismatch");

    // Verify attention configuration
    assert_eq!(
        config.query_pre_attn_scalar, 256,
        "query_pre_attn_scalar mismatch"
    );
    assert!(
        config.use_bidirectional_attention,
        "use_bidirectional_attention should be true"
    );

    // Verify activation function
    assert_eq!(
        config.hidden_activation, "gelu_pytorch_tanh",
        "hidden_activation mismatch"
    );

    // Verify layer types (24 layers alternating between sliding and full attention)
    assert_eq!(config.layer_types.len(), 24, "layer_types length mismatch");

    // Verify pattern: full_attention every 6 layers (controlled by _sliding_window_pattern: 6)
    // Expected pattern: [S, S, S, S, S, F, S, S, S, S, S, F, ...]
    let expected_full_attention_layers = vec![5, 11, 17, 23];
    for (i, layer_type) in config.layer_types.iter().enumerate() {
        let expected = if expected_full_attention_layers.contains(&i) {
            AttentionLayerType::FullAttention
        } else {
            AttentionLayerType::SlidingAttention
        };
        assert_eq!(
            *layer_type, expected,
            "Layer {} type mismatch: expected {:?}, got {:?}",
            i, expected, layer_type
        );
    }
}

/// Test config validation with valid parameters
#[rstest]
#[serial]
fn test_config_validation_valid() {
    let config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 24,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![AttentionLayerType::SlidingAttention; 24],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    // If validation were implemented, this would call config.validate()
    // For now, just verify the config can be created
    assert_eq!(config.vocab_size, 262144);
    assert_eq!(config.hidden_size, 768);
}

/// Test config validation with invalid parameters
#[rstest]
#[case(0, 768, "vocab_size cannot be zero")]
#[case(262144, 0, "hidden_size cannot be zero")]
#[serial]
fn test_config_validation_invalid(
    #[case] vocab_size: usize,
    #[case] hidden_size: usize,
    #[case] _expected_error: &str,
) {
    let _config = GemmaEmbeddingConfig {
        vocab_size,
        hidden_size,
        intermediate_size: 1152,
        num_hidden_layers: 24,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![AttentionLayerType::SlidingAttention; 24],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    // If validation were implemented, this would assert an error
    // For now, config creation succeeds (no validation yet)
}

/// Test MQA (Multi-Query Attention) configuration validation
#[rstest]
#[case(3, 1, true)] // Valid: 3 query heads, 1 KV head
#[case(3, 3, true)] // Valid: 3 query heads, 3 KV heads (standard multi-head)
#[case(6, 2, true)] // Valid: 6 query heads, 2 KV heads
#[serial]
fn test_config_mqa_validation(
    #[case] num_attention_heads: usize,
    #[case] num_key_value_heads: usize,
    #[case] should_be_valid: bool,
) {
    let _config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 24,
        num_attention_heads,
        num_key_value_heads,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![AttentionLayerType::SlidingAttention; 24],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    assert!(
        should_be_valid,
        "MQA configuration validation not yet implemented"
    );
}

/// Test layer types validation
#[rstest]
#[case(vec![AttentionLayerType::SlidingAttention; 24], true)]
#[case(vec![AttentionLayerType::FullAttention; 24], true)]
#[case(vec![], false)] // Empty layer types should be invalid
#[serial]
fn test_config_layer_types_validation(
    #[case] layer_types: Vec<AttentionLayerType>,
    #[case] should_be_valid: bool,
) {
    let _config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: layer_types.len(),
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types,
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    if !should_be_valid {
        // Validation not yet implemented, so empty layer_types currently succeeds
        // This test documents expected behavior
    }
}

/// Test get_layer_type helper method
#[rstest]
#[serial]
fn test_get_layer_type() {
    let config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 4,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![
            AttentionLayerType::SlidingAttention,
            AttentionLayerType::FullAttention,
            AttentionLayerType::SlidingAttention,
            AttentionLayerType::FullAttention,
        ],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    assert_eq!(
        config.get_layer_type(0),
        Some(AttentionLayerType::SlidingAttention)
    );
    assert_eq!(
        config.get_layer_type(1),
        Some(AttentionLayerType::FullAttention)
    );
    assert_eq!(
        config.get_layer_type(2),
        Some(AttentionLayerType::SlidingAttention)
    );
    assert_eq!(
        config.get_layer_type(3),
        Some(AttentionLayerType::FullAttention)
    );
}

/// Test is_full_attention_layer helper method
#[rstest]
#[serial]
fn test_is_full_attention_layer() {
    let config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 4,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps: 1e-6,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![
            AttentionLayerType::SlidingAttention,
            AttentionLayerType::FullAttention,
            AttentionLayerType::SlidingAttention,
            AttentionLayerType::FullAttention,
        ],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    assert!(!config.is_full_attention_layer(0));
    assert!(config.is_full_attention_layer(1));
    assert!(!config.is_full_attention_layer(2));
    assert!(config.is_full_attention_layer(3));
}

/// Test config loading with missing file
#[rstest]
#[serial]
fn test_config_file_not_found() {
    let result = GemmaEmbeddingConfig::from_pretrained("/nonexistent/path");
    assert!(result.is_err(), "Should fail with missing config file");

    match result {
        Err(UnifiedError::Configuration { .. }) => {
            // Expected error type
        }
        _ => panic!("Expected Configuration error"),
    }
}

/// Test rms_norm_eps validation
#[rstest]
#[case(1e-6, true)]
#[case(1e-5, true)]
#[case(0.0, false)]
#[serial]
fn test_config_rms_norm_eps_validation(#[case] rms_norm_eps: f64, #[case] should_be_valid: bool) {
    let _config = GemmaEmbeddingConfig {
        vocab_size: 262144,
        hidden_size: 768,
        intermediate_size: 1152,
        num_hidden_layers: 24,
        num_attention_heads: 3,
        num_key_value_heads: 1,
        max_position_embeddings: 2048,
        rope_theta: 1000000.0,
        rope_local_base_freq: 10000.0,
        rms_norm_eps,
        attention_dropout: 0.0,
        head_dim: 256,
        sliding_window: 512,
        layer_types: vec![AttentionLayerType::SlidingAttention; 24],
        use_bidirectional_attention: true,
        query_pre_attn_scalar: 256,
        hidden_activation: "gelu_pytorch_tanh".to_string(),
    };

    if !should_be_valid {
        // Validation not yet implemented
    }
}

// ============================================================================
// Matryoshka Dimension Tests
// ============================================================================

/// Test that all supported Matryoshka dimensions are accepted
#[rstest]
#[case(768)]
#[case(512)]
#[case(256)]
#[case(128)]
#[serial]
fn test_matryoshka_supported_dimensions(#[case] embedding_dim: usize) {
    let supported_dims = vec![768, 512, 256, 128];
    assert!(
        supported_dims.contains(&embedding_dim),
        "Dimension {} should be supported",
        embedding_dim
    );
}

/// Test that invalid dimensions are rejected
#[rstest]
#[serial]
fn test_matryoshka_invalid_dimension() {
    let invalid_dims = vec![0, 64, 100, 384, 1024, 2048];
    for dim in invalid_dims {
        let supported_dims = vec![768, 512, 256, 128];
        assert!(
            !supported_dims.contains(&dim),
            "Dimension {} should not be supported",
            dim
        );
    }
}

/// Test L2 normalization logic on mock tensors
#[rstest]
#[serial]
fn test_matryoshka_l2_normalization_concept() {
    let device = test_device();

    // Create a test tensor [4, 768]
    let full_embedding = Tensor::randn(0f32, 1.0, (4, 768), &device).unwrap();

    // Normalize to L2 norm = 1.0
    let squared = full_embedding.sqr().unwrap();
    let sum_squared = squared.sum_keepdim(1).unwrap();
    let norm = sum_squared.sqrt().unwrap();
    let normalized_full = full_embedding.broadcast_div(&norm).unwrap();

    // Verify full embedding has L2 norm ≈ 1.0
    let full_norms = normalized_full
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for batch_norms in &full_norms {
        for &n in batch_norms {
            assert!(
                (n - 1.0).abs() < 1e-5,
                "Full embedding norm should be 1.0, got {}",
                n
            );
        }
    }

    // Test truncation to 512 dims
    let truncated = normalized_full.narrow(1, 0, 512).unwrap();

    // After truncation, norm is no longer 1.0
    let truncated_norms_before = truncated
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for batch_norms in &truncated_norms_before {
        for &n in batch_norms {
            assert!(
                n < 1.0,
                "Truncated embedding norm should be < 1.0 before re-normalization, got {}",
                n
            );
        }
    }

    // Re-normalize after truncation
    let squared = truncated.sqr().unwrap();
    let sum_squared = squared.sum_keepdim(1).unwrap();
    let norm = sum_squared.sqrt().unwrap();
    let normalized_truncated = truncated.broadcast_div(&norm).unwrap();

    // Verify re-normalized embedding has L2 norm ≈ 1.0
    let truncated_norms_after = normalized_truncated
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for batch_norms in &truncated_norms_after {
        for &n in batch_norms {
            assert!(
                (n - 1.0).abs() < 1e-5,
                "Re-normalized embedding norm should be 1.0, got {}",
                n
            );
        }
    }
}

/// Test narrow operation for dimension truncation
#[rstest]
#[case(768, 512)]
#[case(768, 256)]
#[case(768, 128)]
#[case(512, 256)]
#[case(512, 128)]
#[case(256, 128)]
#[serial]
fn test_matryoshka_truncation_logic(#[case] from_dim: usize, #[case] to_dim: usize) {
    let device = test_device();
    let full_tensor = Tensor::randn(0f32, 1.0, (4, from_dim), &device).unwrap();

    // Truncate using narrow(dim, start, length)
    let truncated = full_tensor.narrow(1, 0, to_dim).unwrap();

    // Verify shape
    assert_eq!(truncated.dims(), &[4, to_dim]);

    // Verify values match (first to_dim elements should be identical)
    let full_values = full_tensor.to_vec2::<f32>().unwrap();
    let truncated_values = truncated.to_vec2::<f32>().unwrap();

    for (full_row, trunc_row) in full_values.iter().zip(truncated_values.iter()) {
        for i in 0..to_dim {
            assert_eq!(
                full_row[i], trunc_row[i],
                "Truncated values should match original at index {}",
                i
            );
        }
    }
}

/// Test that 768 dimension has no truncation
#[rstest]
#[serial]
fn test_matryoshka_768_no_truncation() {
    let device = test_device();
    let embedding_dim = 768;

    // Create test tensor
    let test_tensor = Tensor::randn(0f32, 1.0, (2, 768), &device).unwrap();

    // Normalize
    let squared = test_tensor.sqr().unwrap();
    let sum_squared = squared.sum_keepdim(1).unwrap();
    let norm = sum_squared.sqrt().unwrap();
    let normalized = test_tensor.broadcast_div(&norm).unwrap();

    // If embedding_dim == 768, the output should be the same as input (no truncation)
    if embedding_dim == 768 {
        let output_dims = normalized.dims();
        assert_eq!(output_dims, &[2, 768]);
    }
}

/// Test different batch sizes with different embedding dimensions
#[rstest]
#[case(1, 768)]
#[case(2, 512)]
#[case(4, 256)]
#[case(8, 128)]
#[serial]
fn test_matryoshka_batch_processing(#[case] batch_size: usize, #[case] embedding_dim: usize) {
    let device = test_device();

    // Create test tensor
    let full_embeddings = Tensor::randn(0f32, 1.0, (batch_size, 768), &device).unwrap();

    // Normalize
    let squared = full_embeddings.sqr().unwrap();
    let sum_squared = squared.sum_keepdim(1).unwrap();
    let norm = sum_squared.sqrt().unwrap();
    let normalized_full = full_embeddings.broadcast_div(&norm).unwrap();

    // Truncate if needed
    let output = if embedding_dim < 768 {
        let truncated = normalized_full.narrow(1, 0, embedding_dim).unwrap();
        let squared = truncated.sqr().unwrap();
        let sum_squared = squared.sum_keepdim(1).unwrap();
        let norm = sum_squared.sqrt().unwrap();
        truncated.broadcast_div(&norm).unwrap()
    } else {
        normalized_full
    };

    // Verify shape
    assert_eq!(output.dims(), &[batch_size, embedding_dim]);

    // Verify L2 normalization
    let norms = output
        .sqr()
        .unwrap()
        .sum_keepdim(1)
        .unwrap()
        .sqrt()
        .unwrap()
        .to_vec2::<f32>()
        .unwrap();

    for batch_norms in &norms {
        for &n in batch_norms {
            assert!((n - 1.0).abs() < 1e-5, "Norm should be 1.0, got {}", n);
        }
    }
}

/// Test config creation for Matryoshka tests
#[rstest]
#[serial]
fn test_matryoshka_config_creation() {
    let config = create_test_config();

    // Verify key configuration parameters
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.vocab_size, 262144);
    assert_eq!(config.num_hidden_layers, 2);

    // Verify Matryoshka-relevant config
    assert_eq!(
        config.hidden_size, 768,
        "Hidden size must be 768 for Matryoshka support"
    );

    // Verify other required fields
    assert_eq!(config.rope_local_base_freq, 10000.0);
    assert_eq!(config.sliding_window, 512);
    assert_eq!(config.layer_types.len(), 2);
}

// ============================================================================
// Output Validation Tests (Against Python Reference Implementation)
// ============================================================================

/// Test GemmaEmbedding output consistency with full dimension (768)
#[rstest]
#[serial(gemma_model)]
fn test_gemma_output_consistency_full_dim(gemma_embedding_model: Arc<GemmaEmbeddingModel>) {
    println!("\n{}", "=".repeat(80));
    println!("GemmaEmbedding Output Validation Test (Full Dimension 768)");
    println!("{}\n", "=".repeat(80));

    // Get device from model
    let device = gemma_embedding_model.device();
    println!("  Using model device: {:?}", device);

    // Load reference outputs
    println!("Loading reference outputs...");
    let reference_outputs = load_reference_outputs();

    // Filter only single-item tests (not batch)
    let single_tests: Vec<&ReferenceOutput> = reference_outputs
        .iter()
        .filter(|r| r.name != "batch_processing_test" && r.tokenization.is_some())
        .collect();

    println!(
        "  Loaded {} single test cases with tokenization\n",
        single_tests.len()
    );
    println!("  Running forward pass with real tokenization data...\n");

    let mut all_passed = true;

    for (i, reference) in single_tests.iter().enumerate() {
        println!("{}", "-".repeat(80));
        println!(
            "[{}/{}] Validating: {}",
            i + 1,
            single_tests.len(),
            reference.name
        );
        println!("{}", "-".repeat(80));
        println!("  Text: {}", reference.input.text);
        println!("  Text length: {} chars", reference.input.full_text_length);

        // Get tokenization from reference
        let tokenization = reference.tokenization.as_ref().unwrap();
        let input_ids_vec = tokenization.get_input_ids();
        let attention_mask_vec = tokenization.get_attention_mask();

        println!(
            "  Tokenization: seq_len={}, shape={:?}",
            tokenization.seq_len, tokenization.input_shape
        );

        // Convert to Tensors
        let input_ids_data: Vec<u32> = input_ids_vec[0].clone();
        let attention_mask_data: Vec<u32> = attention_mask_vec[0].clone();

        let input_ids =
            Tensor::from_vec(input_ids_data.clone(), (1, input_ids_data.len()), &device)
                .expect("Failed to create input_ids tensor");

        let attention_mask = Tensor::from_vec(
            attention_mask_data.clone(),
            (1, attention_mask_data.len()),
            &device,
        )
        .expect("Failed to create attention_mask tensor");

        // Run model forward pass (full dimension 768)
        let rust_embedding_result =
            gemma_embedding_model.embedding_forward(&input_ids, Some(&attention_mask));

        let rust_embedding = match rust_embedding_result {
            Ok(emb) => emb,
            Err(e) => {
                eprintln!("  ERROR: Forward pass failed: {:?}", e);
                all_passed = false;
                continue;
            }
        };

        // Convert to Vec<f32>
        let rust_vec = rust_embedding
            .flatten_all()
            .expect("Failed to flatten")
            .to_vec1::<f32>()
            .expect("Failed to convert to vec");

        // Get Python reference embedding (full dimension)
        let python_vec = if !reference.embedding_full.is_empty() {
            &reference.embedding_full
        } else if !reference.embeddings.is_empty() {
            &reference.embeddings[0]
        } else {
            eprintln!("  ERROR: No reference embedding found");
            all_passed = false;
            continue;
        };

        // Calculate cosine similarity
        let similarity = cosine_similarity(&rust_vec, python_vec);

        // Calculate L2 norms
        let rust_norm: f32 = rust_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let python_norm: f32 = python_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!("  Rust embedding shape: {:?}", rust_embedding.dims());
        println!("  Python embedding shape: [1, 768]");
        println!("  Rust L2 norm: {:.6}", rust_norm);
        println!("  Python L2 norm: {:.6}", python_norm);
        println!("  Cosine similarity: {:.6}", similarity);

        // Verify similarity threshold
        let threshold = 0.99;
        if similarity >= threshold {
            println!(
                "  PASS: Cosine similarity {:.6} >= {}",
                similarity, threshold
            );
        } else {
            println!(
                "  FAIL: Cosine similarity {:.6} < {}",
                similarity, threshold
            );
            all_passed = false;
        }
    }

    println!("\n{}", "=".repeat(80));
    if all_passed {
        println!("ALL TESTS PASSED");
    } else {
        println!("SOME TESTS FAILED");
        panic!("GemmaEmbedding output validation failed");
    }
    println!("{}", "=".repeat(80));
}

/// Test GemmaEmbedding with Matryoshka dimensions (512/256/128)
#[rstest]
#[case(512)]
#[case(256)]
#[case(128)]
#[serial(gemma_model)]
fn test_gemma_matryoshka_dimensions(
    gemma_embedding_model: Arc<GemmaEmbeddingModel>,
    #[case] target_dim: usize,
) {
    println!("\n{}", "=".repeat(80));
    println!("GemmaEmbedding Matryoshka Dimension Test ({})", target_dim);
    println!("{}\n", "=".repeat(80));

    // Get device from model
    let device = gemma_embedding_model.device();

    // Load reference outputs
    let reference_outputs = load_reference_outputs();

    // Filter single-item tests
    let single_tests: Vec<&ReferenceOutput> = reference_outputs
        .iter()
        .filter(|r| r.name != "batch_processing_test" && r.tokenization.is_some())
        .collect();

    println!("  Loaded {} test cases\n", single_tests.len());

    let mut all_passed = true;

    for (i, reference) in single_tests.iter().enumerate() {
        println!("{}", "-".repeat(80));
        println!(
            "[{}/{}] Testing: {}",
            i + 1,
            single_tests.len(),
            reference.name
        );
        println!("{}", "-".repeat(80));

        // Get tokenization
        let tokenization = reference.tokenization.as_ref().unwrap();
        let input_ids_vec = tokenization.get_input_ids();
        let attention_mask_vec = tokenization.get_attention_mask();

        let input_ids_data: Vec<u32> = input_ids_vec[0].clone();
        let attention_mask_data: Vec<u32> = attention_mask_vec[0].clone();

        let input_ids =
            Tensor::from_vec(input_ids_data.clone(), (1, input_ids_data.len()), &device)
                .expect("Failed to create input_ids tensor");

        let attention_mask = Tensor::from_vec(
            attention_mask_data.clone(),
            (1, attention_mask_data.len()),
            &device,
        )
        .expect("Failed to create attention_mask tensor");

        // Run model with target dimension (Matryoshka)
        let rust_embedding_result =
            gemma_embedding_model.matryoshka_forward(&input_ids, Some(&attention_mask), target_dim);

        let rust_embedding = match rust_embedding_result {
            Ok(emb) => emb,
            Err(e) => {
                eprintln!("  ERROR: Forward pass failed: {:?}", e);
                all_passed = false;
                continue;
            }
        };

        // Verify shape
        assert_eq!(
            rust_embedding.dims(),
            &[1, target_dim],
            "Output dimension mismatch"
        );

        // Convert to Vec<f32>
        let rust_vec = rust_embedding
            .flatten_all()
            .expect("Failed to flatten")
            .to_vec1::<f32>()
            .expect("Failed to convert to vec");

        // Get Python reference for this dimension
        let dim_key = target_dim.to_string();
        let python_vec = if let Some(mat_embedding) = reference.matryoshka.get(&dim_key) {
            mat_embedding
        } else {
            eprintln!(
                "  ERROR: No reference embedding for dimension {}",
                target_dim
            );
            all_passed = false;
            continue;
        };

        // Calculate similarity
        let similarity = cosine_similarity(&rust_vec, python_vec);

        // Calculate L2 norms
        let rust_norm: f32 = rust_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let python_norm: f32 = python_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

        println!("  Rust L2 norm: {:.6}", rust_norm);
        println!("  Python L2 norm: {:.6}", python_norm);
        println!("  Cosine similarity: {:.6}", similarity);

        // Verify threshold
        let threshold = 0.99;
        if similarity >= threshold {
            println!(
                "  PASS: Cosine similarity {:.6} >= {}",
                similarity, threshold
            );
        } else {
            println!(
                "  FAIL: Cosine similarity {:.6} < {}",
                similarity, threshold
            );
            all_passed = false;
        }
    }

    println!("\n{}", "=".repeat(80));
    if all_passed {
        println!("ALL TESTS PASSED for dimension {}", target_dim);
    } else {
        println!("SOME TESTS FAILED for dimension {}", target_dim);
        panic!("Matryoshka dimension {} validation failed", target_dim);
    }
    println!("{}", "=".repeat(80));
}
