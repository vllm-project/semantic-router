//! Tests for unified classifier functionality

use crate::test_fixtures::fixtures::*;
use rstest::*;
use std::path::Path;

/// Test unified classifier model path validation
#[rstest]
fn test_unified_unified_classifier_model_path_validation(
    traditional_model_path: String,
    lora_model_path: String,
) {
    // Test unified classifier model path validation logic
    println!("Testing unified classifier model path validation");

    // Test traditional model path validation
    if Path::new(&traditional_model_path).exists() {
        println!(
            "Traditional model path validated: {}",
            traditional_model_path
        );
        assert!(!traditional_model_path.is_empty());
        assert!(traditional_model_path.contains("models"));
    } else {
        println!(
            "Traditional model path not found: {}",
            traditional_model_path
        );
    }

    // Test LoRA model path validation
    if Path::new(&lora_model_path).exists() {
        println!("LoRA model path validated: {}", lora_model_path);
        assert!(!lora_model_path.is_empty());
        assert!(lora_model_path.contains("models"));
    } else {
        println!("LoRA model path not found: {}", lora_model_path);
    }

    // Test unified path validation logic
    let model_paths = vec![&traditional_model_path, &lora_model_path];
    for (i, path) in model_paths.iter().enumerate() {
        assert!(!path.is_empty(), "Model path {} should not be empty", i);

        // Test path format validation
        if path.contains("models") {
            println!("Path {} format validation passed: {}", i, path);
        }
    }

    println!("Unified classifier model path validation test completed");
}

use crate::classifiers::unified::{DualPathUnifiedClassifier, EmbeddingRequirements};
use crate::model_architectures::config::{
    DevicePreference, DualPathConfig, EmbeddingConfig, GlobalConfig, LoRAConfig, OptimizationLevel,
    PathSelectionStrategy, TraditionalConfig,
};
use crate::model_architectures::ModelType;
use serial_test::serial;

/// Helper function to create a test classifier
fn create_test_classifier() -> DualPathUnifiedClassifier {
    let config = DualPathConfig {
        global: GlobalConfig {
            device_preference: DevicePreference::CPU,
            path_selection: PathSelectionStrategy::Automatic,
            optimization_level: OptimizationLevel::Balanced,
            enable_monitoring: false,
        },
        traditional: TraditionalConfig::default(),
        lora: LoRAConfig::default(),
        embedding: EmbeddingConfig::default(),
    };

    DualPathUnifiedClassifier::new(config).expect("Failed to create test classifier")
}

/// Test short sequence routing with high latency priority
#[rstest]
#[serial]
fn test_select_embedding_model_short_sequence_high_latency() {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: 256,
        quality_priority: 0.3,
        latency_priority: 0.8, // High latency priority
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ModelType::GemmaEmbedding,
        "Short sequences with high latency priority (> 0.7) should use GemmaEmbedding (fastest embedding model)");
}

/// Test short sequence routing with low latency priority
#[rstest]
#[serial]
fn test_select_embedding_model_short_sequence_low_latency() {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: 512,
        quality_priority: 0.8,
        latency_priority: 0.3, // Low latency priority
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), ModelType::Qwen3Embedding,
        "Short sequences with high quality priority (latency_priority <= 0.7) should use Qwen3Embedding");
}

/// Test medium sequence routing
#[rstest]
#[case(513)] // Lower bound
#[case(1024)] // Middle
#[case(2048)] // Upper bound
#[serial]
fn test_select_embedding_model_medium_sequences(#[case] seq_len: usize) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: seq_len,
        quality_priority: 0.5,
        latency_priority: 0.5,
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        ModelType::GemmaEmbedding,
        "Medium sequences (513-2048) should always use GemmaEmbedding (optimal for this range)"
    );
}

/// Test long sequence routing
#[rstest]
#[case(2049)] // Lower bound
#[case(16384)] // Middle (16K)
#[case(32768)] // Upper bound (32K)
#[serial]
fn test_select_embedding_model_long_sequences(#[case] seq_len: usize) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: seq_len,
        quality_priority: 0.5,
        latency_priority: 0.5,
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        ModelType::Qwen3Embedding,
        "Long sequences (2049-32768) should always use Qwen3Embedding (only model supporting 32K)"
    );
}

/// Test ultra-long sequence error handling
#[rstest]
#[case(32769)] // Just over limit
#[case(40000)] // Far over limit
#[case(100000)] // Very far over limit
#[serial]
fn test_select_embedding_model_ultra_long_sequences_error(#[case] seq_len: usize) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: seq_len,
        quality_priority: 0.5,
        latency_priority: 0.5,
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(
        result.is_err(),
        "Ultra-long sequences (>32768) should return error"
    );

    let error_msg = result.unwrap_err().to_string();
    assert!(
        error_msg.contains("exceeds maximum"),
        "Error message should indicate exceeding maximum length"
    );
    assert!(
        error_msg.contains(&seq_len.to_string()),
        "Error message should contain the actual sequence length"
    );
}

/// Test boundary conditions
#[rstest]
#[case(0, ModelType::GemmaEmbedding)] // Zero length (high latency priority > 0.7)
#[case(1, ModelType::GemmaEmbedding)] // Minimum length (high latency priority)
#[case(512, ModelType::GemmaEmbedding)] // Short-medium boundary (high latency priority)
#[case(513, ModelType::GemmaEmbedding)] // Medium lower bound (always Gemma)
#[case(2048, ModelType::GemmaEmbedding)] // Medium upper bound (always Gemma)
#[case(2049, ModelType::Qwen3Embedding)] // Long lower bound (only Qwen3 supports)
#[case(32768, ModelType::Qwen3Embedding)] // Maximum supported (only Qwen3)
#[serial]
fn test_select_embedding_model_boundary_conditions(
    #[case] seq_len: usize,
    #[case] expected_type: ModelType,
) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: seq_len,
        quality_priority: 0.5,
        latency_priority: 0.8, // High latency for short sequences
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        expected_type,
        "Boundary condition for sequence length {} failed",
        seq_len
    );
}

/// Test priority influence on short sequences
#[rstest]
#[case(0.9, 0.2, ModelType::Qwen3Embedding)] // High quality priority (latency <= 0.7)
#[case(0.2, 0.9, ModelType::GemmaEmbedding)] // High latency priority (> 0.7)
#[case(0.5, 0.5, ModelType::Qwen3Embedding)] // Balanced (latency <= 0.7, defaults to quality)
#[case(0.5, 0.6, ModelType::Qwen3Embedding)] // Slightly latency-focused (still <= 0.7)
#[case(0.5, 0.75, ModelType::GemmaEmbedding)] // Clearly latency-focused (> 0.7)
#[serial]
fn test_select_embedding_model_priority_influence(
    #[case] quality_priority: f32,
    #[case] latency_priority: f32,
    #[case] expected_type: ModelType,
) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: 256, // Short sequence
        quality_priority,
        latency_priority,
        target_dimension: None,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap(),
        expected_type,
        "Priority (quality={}, latency={}) should route to {:?}",
        quality_priority,
        latency_priority,
        expected_type
    );
}

/// Test with Matryoshka dimension hints
#[rstest]
#[case(Some(768))]
#[case(Some(512))]
#[case(Some(256))]
#[case(Some(128))]
#[case(None)]
#[serial]
fn test_select_embedding_model_with_matryoshka_dimensions(#[case] target_dim: Option<usize>) {
    let classifier = create_test_classifier();

    let requirements = EmbeddingRequirements {
        sequence_length: 1024,
        quality_priority: 0.5,
        latency_priority: 0.5,
        target_dimension: target_dim,
    };

    let result = classifier.select_embedding_model(&requirements);
    assert!(result.is_ok());
    // This test documents the current behavior: medium sequences always use Gemma
    assert_eq!(result.unwrap(), ModelType::GemmaEmbedding);
}
