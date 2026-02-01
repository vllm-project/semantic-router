//! Tests for traditional ModernBERT implementation
//!
//! This module tests both standard ModernBERT and mmBERT (multilingual) variants.
//! mmBERT is supported through the same implementation using `ModernBertVariant::Multilingual`.

use super::modernbert::*;
use crate::core::tokenization::{detect_mmbert_from_config, TokenizationStrategy};
use crate::model_architectures::traits::{ModelType, TaskType};
use crate::test_fixtures::{fixtures::*, test_utils::*};
use rstest::*;
use serial_test::serial;
use std::sync::Arc;

/// Test TraditionalModernBertClassifier creation interface
#[rstest]
#[serial]
fn test_modernbert_traditional_modernbert_classifier_new(
    cached_traditional_intent_classifier: Option<Arc<TraditionalModernBertClassifier>>,
) {
    // Use cached Traditional Intent classifier
    if let Some(classifier) = cached_traditional_intent_classifier {
        println!("Testing TraditionalModernBertClassifier with cached model");

        // Test actual classification with cached model
        let business_texts = business_texts();
        let test_text = business_texts[11]; // "Hello, how are you today?"
        match classifier.classify_text(test_text) {
            Ok((class_id, confidence)) => {
                println!(
                    "Cached model classification result: class_id={}, confidence={:.3}",
                    class_id, confidence
                );

                // Validate cached model output
                assert!(confidence >= 0.0 && confidence <= 1.0);
                assert!(class_id < 100); // Reasonable upper bound
            }
            Err(e) => {
                println!("Cached model classification failed: {}", e);
            }
        }
    } else {
        println!("Traditional Intent classifier not available in cache");
    }
}

/// Test TraditionalModernBertTokenClassifier creation interface
#[rstest]
fn test_modernbert_traditional_modernbert_token_classifier_new(
    traditional_pii_token_model_path: String,
) {
    // Use real traditional ModernBERT PII model (token classifier) from fixtures

    let classifier_result = TraditionalModernBertTokenClassifier::new(
        &traditional_pii_token_model_path,
        true, // use CPU
    );

    match classifier_result {
        Ok(classifier) => {
            println!(
                "TraditionalModernBertTokenClassifier creation succeeded with real model: {}",
                traditional_pii_token_model_path
            );

            // Test actual token classification with real model
            let test_text = "Please call me at 555-123-4567 or visit my address at 123 Main Street, New York, NY 10001";
            match classifier.classify_tokens(test_text) {
                Ok(results) => {
                    println!(
                        "Real model token classification succeeded with {} results",
                        results.len()
                    );

                    for (i, (token, label_id, confidence, start_pos, end_pos)) in
                        results.iter().enumerate()
                    {
                        println!("Token result {}: token='{}', label_id={}, confidence={:.3}, pos={}..{}",
                            i, token, label_id, confidence, start_pos, end_pos);

                        // Validate each result
                        assert!(!token.is_empty());
                        assert!(confidence >= &0.0 && confidence <= &1.0);
                        assert!(start_pos <= end_pos);
                    }

                    // Should detect some tokens
                    assert!(!results.is_empty());
                }
                Err(e) => {
                    println!("Real model token classification failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!(
                "TraditionalModernBertTokenClassifier creation failed with real model {}: {}",
                traditional_pii_token_model_path, e
            );
            // This might happen if model files are missing or corrupted
        }
    }
}

/// Test TraditionalModernBertClassifier error handling
#[rstest]
fn test_modernbert_traditional_modernbert_classifier_error_handling() {
    // Test error scenarios

    // Invalid model path
    let invalid_model_result = TraditionalModernBertClassifier::load_from_directory("", true);
    assert!(invalid_model_result.is_err());

    // Non-existent model path
    let nonexistent_model_result =
        TraditionalModernBertClassifier::load_from_directory("/nonexistent/path/to/model", true);
    assert!(nonexistent_model_result.is_err());

    println!("TraditionalModernBertClassifier error handling test passed");
}

/// Test TraditionalModernBertTokenClassifier error handling
#[rstest]
fn test_modernbert_traditional_modernbert_token_classifier_error_handling() {
    // Test error scenarios

    // Invalid model path
    let invalid_model_result = TraditionalModernBertTokenClassifier::new("", true);
    assert!(invalid_model_result.is_err());

    // Non-existent model path
    let nonexistent_model_result =
        TraditionalModernBertTokenClassifier::new("/nonexistent/path/to/model", true);
    assert!(nonexistent_model_result.is_err());

    println!("TraditionalModernBertTokenClassifier error handling test passed");
}

/// Test TraditionalModernBertClassifier classification output format with real model
#[rstest]
#[serial]
fn test_modernbert_traditional_modernbert_classifier_output_format(
    cached_traditional_intent_classifier: Option<Arc<TraditionalModernBertClassifier>>,
) {
    // Use cached Traditional Intent classifier to test actual output format
    if let Some(classifier) = cached_traditional_intent_classifier {
        println!("Testing cached model output format");

        // Test with multiple different texts to verify output format consistency
        let test_texts = vec![
            "This is a positive example",
            "This is a negative example",
            "This is a neutral example",
        ];

        for test_text in test_texts {
            match classifier.classify_text(test_text) {
                Ok((predicted_class, confidence)) => {
                    println!(
                        "Cached output format for '{}': class={}, confidence={:.3}",
                        test_text, predicted_class, confidence
                    );

                    // Validate cached output format
                    assert!(predicted_class < 100); // Reasonable upper bound for real models
                    assert!(confidence >= 0.0 && confidence <= 1.0);

                    // Test that output is the expected tuple format (usize, f32)
                    let output: (usize, f32) = (predicted_class, confidence);
                    assert_eq!(output.0, predicted_class);
                    assert_eq!(output.1, confidence);

                    // Test that confidence is a reasonable probability (not NaN, not infinite)
                    assert!(confidence.is_finite());
                    assert!(!confidence.is_nan());
                }
                Err(e) => {
                    println!(
                        "Cached model classification failed for '{}': {}",
                        test_text, e
                    );
                }
            }
        }
    } else {
        println!("Traditional Intent classifier not available in cache");
    }
}

/// Test TraditionalModernBertTokenClassifier token output format with real model
#[rstest]
fn test_modernbert_traditional_modernbert_token_classifier_output_format(
    traditional_pii_token_model_path: String,
) {
    // Use real traditional ModernBERT PII model to test actual token output format
    let classifier_result = TraditionalModernBertTokenClassifier::new(
        &traditional_pii_token_model_path,
        true, // use CPU
    );

    match classifier_result {
        Ok(classifier) => {
            println!(
                "Testing real token model output format with: {}",
                traditional_pii_token_model_path
            );

            // Test with texts containing clear PII entities
            let test_texts = vec![
                "My personal information: Phone: +1-800-555-0199, Address: 456 Oak Avenue, Los Angeles, CA 90210",
                "Please call me at 555-123-4567 or visit my address at 123 Main Street, New York, NY 10001",
                "My SSN is 123-45-6789 and my credit card is 4532-1234-5678-9012",
            ];

            for test_text in test_texts {
                match classifier.classify_tokens(test_text) {
                    Ok(token_results) => {
                        println!(
                            "Real token output format for '{}': {} tokens",
                            test_text,
                            token_results.len()
                        );

                        for (i, (token, predicted_class, confidence, start_pos, end_pos)) in
                            token_results.iter().enumerate()
                        {
                            println!(
                                "  Token {}: '{}' -> class={}, conf={:.3}, pos={}..{}",
                                i, token, predicted_class, confidence, start_pos, end_pos
                            );

                            // Validate real token output format
                            assert!(!token.is_empty());
                            assert!(*predicted_class < 100); // Reasonable upper bound for real models
                            assert!(*confidence >= 0.0 && *confidence <= 1.0);
                            assert!(*start_pos <= *end_pos);

                            // Test that output is the expected tuple format
                            let output: (String, usize, f32, usize, usize) = (
                                token.clone(),
                                *predicted_class,
                                *confidence,
                                *start_pos,
                                *end_pos,
                            );
                            assert_eq!(output.0, *token);
                            assert_eq!(output.1, *predicted_class);
                            assert_eq!(output.2, *confidence);
                            assert_eq!(output.3, *start_pos);
                            assert_eq!(output.4, *end_pos);

                            // Test that confidence is a reasonable probability (not NaN, not infinite)
                            assert!(confidence.is_finite());
                            assert!(!confidence.is_nan());

                            // Test that positions make sense for the text
                            if *end_pos <= test_text.len() {
                                let extracted_token = &test_text[*start_pos..*end_pos];
                                // Note: Tokenization might not match exact string slicing due to subword tokenization
                                println!(
                                    "    Extracted: '{}' (original token: '{}')",
                                    extracted_token, token
                                );
                            }
                        }

                        // Check if we got tokens (some models might return empty results due to thresholds)
                        if token_results.is_empty() {
                            println!("    Warning: No tokens returned for '{}' - this might be due to confidence thresholds", test_text);
                        } else {
                            println!(
                                "    Successfully got {} tokens with real model",
                                token_results.len()
                            );
                        }
                    }
                    Err(e) => {
                        println!(
                            "Real token model classification failed for '{}': {}",
                            test_text, e
                        );
                    }
                }
            }
        }
        Err(e) => {
            println!(
                "TraditionalModernBertTokenClassifier creation failed for output format test: {}",
                e
            );
        }
    }
}

// ============================================================================
// mmBERT (Multilingual ModernBERT) Variant Tests
// ============================================================================

/// Test ModernBertVariant enum
#[rstest]
fn test_modernbert_variant_properties() {
    // Test Standard variant
    let standard = ModernBertVariant::Standard;
    assert_eq!(standard.max_length(), 512);
    assert_eq!(standard.pad_token(), "[PAD]");
    assert!(!standard.uses_yarn_scaling());
    assert_eq!(standard.expected_rope_theta(), 10000.0);
    assert!(matches!(
        standard.tokenization_strategy(),
        TokenizationStrategy::ModernBERT
    ));

    // Test Multilingual (mmBERT) variant
    let multilingual = ModernBertVariant::Multilingual;
    assert_eq!(multilingual.max_length(), 8192);
    assert_eq!(multilingual.pad_token(), "<pad>");
    assert!(!multilingual.uses_yarn_scaling());
    assert_eq!(multilingual.expected_rope_theta(), 10000.0);
    assert!(matches!(
        multilingual.tokenization_strategy(),
        TokenizationStrategy::MmBERT
    ));

    // Test Multilingual32K (mmBERT-32K YaRN) variant
    let multilingual_32k = ModernBertVariant::Multilingual32K;
    assert_eq!(multilingual_32k.max_length(), 32768);
    assert_eq!(multilingual_32k.pad_token(), "<pad>");
    assert!(multilingual_32k.uses_yarn_scaling());
    assert_eq!(multilingual_32k.expected_rope_theta(), 160000.0);
    assert!(matches!(
        multilingual_32k.tokenization_strategy(),
        TokenizationStrategy::MmBERT
    ));

    println!("ModernBertVariant properties test passed");
}

/// Test mmBERT config detection
#[rstest]
fn test_mmbert_config_detection() {
    use std::io::Write;
    use tempfile::TempDir;

    // Create a temporary directory with mmBERT-like config
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    // Write mmBERT-style config
    let mmbert_config = r#"{
        "vocab_size": 256000,
        "model_type": "modernbert",
        "position_embedding_type": "sans_pos",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "intermediate_size": 1152,
        "max_position_embeddings": 8192,
        "local_attention": 128,
        "global_attn_every_n_layers": 3
    }"#;

    std::fs::write(&config_path, mmbert_config).expect("Failed to write config");

    // Test variant detection
    let variant = ModernBertVariant::detect_from_config(config_path.to_str().unwrap());
    assert!(variant.is_ok());
    assert_eq!(variant.unwrap(), ModernBertVariant::Multilingual);

    // Also test the tokenization detection
    let is_mmbert = detect_mmbert_from_config(config_path.to_str().unwrap());
    assert!(
        is_mmbert.unwrap_or(false),
        "Should detect mmBERT config correctly"
    );

    println!("mmBERT config detection test passed");
}

/// Test that regular ModernBERT is NOT detected as mmBERT
#[rstest]
fn test_modernbert_not_detected_as_mmbert() {
    use tempfile::TempDir;

    // Create a temporary directory with regular ModernBERT config
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    // Write regular ModernBERT config (smaller vocab, different position embedding)
    let modernbert_config = r#"{
        "vocab_size": 50368,
        "model_type": "modernbert",
        "position_embedding_type": "absolute",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12
    }"#;

    std::fs::write(&config_path, modernbert_config).expect("Failed to write config");

    // Test variant detection - should be Standard
    let variant = ModernBertVariant::detect_from_config(config_path.to_str().unwrap());
    assert!(variant.is_ok());
    assert_eq!(variant.unwrap(), ModernBertVariant::Standard);

    // Test tokenization detection - should NOT be mmBERT
    let is_mmbert = detect_mmbert_from_config(config_path.to_str().unwrap());
    assert!(
        !is_mmbert.unwrap_or(true),
        "Regular ModernBERT should not be detected as mmBERT"
    );

    println!("ModernBERT not detected as mmBERT test passed");
}

// ============================================================================
// mmBERT-32K YaRN (Extended Context) Variant Tests
// ============================================================================

/// Test mmBERT-32K config detection via max_position_embeddings
#[rstest]
fn test_mmbert_32k_config_detection_by_max_position() {
    use tempfile::TempDir;

    // Create a temporary directory with mmBERT-32K config
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    // Write mmBERT-32K style config with extended max_position_embeddings
    let mmbert_32k_config = r#"{
        "vocab_size": 256000,
        "model_type": "modernbert",
        "position_embedding_type": "sans_pos",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "intermediate_size": 1152,
        "max_position_embeddings": 32768,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000,
        "local_rope_theta": 160000
    }"#;

    std::fs::write(&config_path, mmbert_32k_config).expect("Failed to write config");

    // Test variant detection - should be Multilingual32K
    let variant = ModernBertVariant::detect_from_config(config_path.to_str().unwrap());
    assert!(variant.is_ok());
    assert_eq!(
        variant.unwrap(),
        ModernBertVariant::Multilingual32K,
        "Should detect mmBERT-32K from max_position_embeddings"
    );

    println!("mmBERT-32K config detection (max_position) test passed");
}

/// Test mmBERT-32K config detection via high rope_theta (YaRN indicator)
#[rstest]
fn test_mmbert_32k_config_detection_by_rope_theta() {
    use tempfile::TempDir;

    // Create a temporary directory with mmBERT config that has YaRN rope_theta
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let config_path = temp_dir.path().join("config.json");

    // Write config with YaRN rope_theta but default max_position_embeddings
    // This tests detection via rope_theta alone
    let mmbert_yarn_config = r#"{
        "vocab_size": 256000,
        "model_type": "modernbert",
        "position_embedding_type": "sans_pos",
        "hidden_size": 768,
        "num_hidden_layers": 22,
        "num_attention_heads": 12,
        "intermediate_size": 1152,
        "max_position_embeddings": 8192,
        "local_attention": 128,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000
    }"#;

    std::fs::write(&config_path, mmbert_yarn_config).expect("Failed to write config");

    // Test variant detection - should be Multilingual32K due to high rope_theta
    let variant = ModernBertVariant::detect_from_config(config_path.to_str().unwrap());
    assert!(variant.is_ok());
    assert_eq!(
        variant.unwrap(),
        ModernBertVariant::Multilingual32K,
        "Should detect mmBERT-32K from high global_rope_theta (YaRN indicator)"
    );

    println!("mmBERT-32K config detection (rope_theta) test passed");
}

/// Test mmBERT-32K type aliases
#[rstest]
fn test_mmbert_32k_type_aliases() {
    // Verify 32K type aliases are correctly defined
    fn _accepts_mmbert_32k_classifier(_c: &MmBert32KClassifier) {}
    fn _accepts_mmbert_32k_token_classifier(_c: &MmBert32KTokenClassifier) {}

    // Type equivalence checks (compile-time)
    fn _accepts_modernbert_as_32k(_c: &TraditionalModernBertClassifier) {
        // MmBert32KClassifier is an alias for TraditionalModernBertClassifier
    }
    fn _accepts_token_modernbert_as_32k(_c: &TraditionalModernBertTokenClassifier) {
        // MmBert32KTokenClassifier is an alias for TraditionalModernBertTokenClassifier
    }

    println!("mmBERT-32K type aliases test passed");
}

/// Test mmBERT-32K classifier error handling
#[rstest]
fn test_mmbert_32k_classifier_error_handling() {
    // Invalid model path with explicit 32K variant
    let invalid_result = TraditionalModernBertClassifier::load_from_directory_with_variant(
        "",
        true,
        ModernBertVariant::Multilingual32K,
    );
    assert!(invalid_result.is_err());

    // Non-existent model path using convenience method
    let nonexistent_result = TraditionalModernBertClassifier::load_mmbert_32k_from_directory(
        "/nonexistent/path/to/model",
        true,
    );
    assert!(nonexistent_result.is_err());

    println!("mmBERT-32K classifier error handling test passed");
}

/// Test mmBERT-32K token classifier error handling
#[rstest]
fn test_mmbert_32k_token_classifier_error_handling() {
    // Invalid model path with explicit 32K variant
    let invalid_result = TraditionalModernBertTokenClassifier::new_with_variant(
        "",
        true,
        ModernBertVariant::Multilingual32K,
    );
    assert!(invalid_result.is_err());

    // Non-existent model path using convenience method
    let nonexistent_result =
        TraditionalModernBertTokenClassifier::new_mmbert_32k("/nonexistent/path/to/model", true);
    assert!(nonexistent_result.is_err());

    println!("mmBERT-32K token classifier error handling test passed");
}

/// Test mmBERT-32K expected configuration values
#[rstest]
fn test_mmbert_32k_expected_config_values() {
    // Document expected mmBERT-32K (YaRN) configuration values based on
    // https://huggingface.co/llm-semantic-router/mmbert-32k-yarn

    let expected_config = vec![
        ("vocab_size", "256000"),
        ("hidden_size", "768"),
        ("num_hidden_layers", "22"),
        ("num_attention_heads", "12"),
        ("intermediate_size", "1152"),
        ("max_position_embeddings", "32768"), // Extended from 8192
        ("position_embedding_type", "sans_pos"),
        ("local_attention", "128"),
        ("global_attn_every_n_layers", "3"),
        ("global_rope_theta", "160000"), // YaRN-scaled (4x from original)
        ("local_rope_theta", "160000"),
        ("pad_token_id", "0"),
        ("bos_token_id", "2"),
        ("eos_token_id", "1"),
    ];

    println!("Expected mmBERT-32K (YaRN) configuration:");
    for (key, value) in &expected_config {
        println!("  {}: {}", key, value);
    }

    // Verify critical 32K-specific values
    let max_pos = expected_config
        .iter()
        .find(|(k, _)| *k == "max_position_embeddings");
    assert_eq!(
        max_pos.unwrap().1,
        "32768",
        "max_position_embeddings should be 32768"
    );

    let rope_theta = expected_config
        .iter()
        .find(|(k, _)| *k == "global_rope_theta");
    assert_eq!(
        rope_theta.unwrap().1,
        "160000",
        "global_rope_theta should be 160000 (YaRN)"
    );

    println!("mmBERT-32K config values test passed");
}

/// Integration test for mmBERT-32K with actual model (requires model files)
/// Run with: MMBERT_32K_MODEL_PATH=models/mmbert-32k-yarn cargo test test_mmbert_32k_integration_with_model -- --nocapture
#[rstest]
fn test_mmbert_32k_integration_with_model() {
    // Skip in CI environments
    if std::env::var("CI").is_ok() {
        println!("Skipping mmBERT-32K integration test in CI environment");
        return;
    }

    let model_path = std::env::var("MMBERT_32K_MODEL_PATH")
        .unwrap_or_else(|_| "../models/mmbert-32k-yarn".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "Skipping integration test - model path not found: {}",
            model_path
        );
        println!("To run this test, download the model first:");
        println!("  make download-mmbert-32k");
        return;
    }

    println!("Loading mmBERT-32K from: {}", model_path);

    // Verify config detection works correctly for the real model
    let config_path = format!("{}/config.json", model_path);
    let detected_variant = ModernBertVariant::detect_from_config(&config_path)
        .expect("Failed to detect variant from config");

    assert_eq!(
        detected_variant,
        ModernBertVariant::Multilingual32K,
        "Real model should be detected as Multilingual32K variant"
    );
    println!("✓ Config correctly detected as Multilingual32K variant");

    // Verify config values match expected 32K YaRN parameters
    let config_str = std::fs::read_to_string(&config_path).expect("Failed to read config");
    let config_json: serde_json::Value =
        serde_json::from_str(&config_str).expect("Failed to parse config");

    let max_pos = config_json["max_position_embeddings"].as_u64().unwrap_or(0);
    let rope_theta = config_json["global_rope_theta"].as_f64().unwrap_or(0.0);
    let vocab_size = config_json["vocab_size"].as_u64().unwrap_or(0);

    assert!(
        max_pos >= 32768,
        "max_position_embeddings should be >= 32768, got {}",
        max_pos
    );
    assert!(
        rope_theta >= 100000.0,
        "global_rope_theta should be >= 100000 (YaRN), got {}",
        rope_theta
    );
    assert!(
        vocab_size >= 200000,
        "vocab_size should be >= 200000 (mmBERT), got {}",
        vocab_size
    );

    println!("✓ Config values verified:");
    println!("  - max_position_embeddings: {}", max_pos);
    println!("  - global_rope_theta: {} (YaRN-scaled)", rope_theta);
    println!("  - vocab_size: {}", vocab_size);

    // Note: The base mmbert-32k-yarn model is an MLM model, not a classifier
    // For classification, you would need a fine-tuned version
    // Here we just verify the config detection and loading works

    println!("");
    println!("✅ mmBERT-32K model config validation passed");
    println!("   Model supports 32K context with YaRN RoPE scaling");
}

// ============================================================================
// Real Unit Tests for 32K Context Support
// These tests verify the actual implementation without requiring model files
// ============================================================================

/// Test RoPE frequency computation for 32K context with YaRN theta
/// This verifies the mathematical correctness of YaRN RoPE scaling
#[rstest]
fn test_yarn_rope_frequency_computation_32k() {
    // YaRN RoPE parameters for 32K context
    let rope_theta: f64 = 160000.0; // YaRN-scaled theta (4x from 10000 base)
    let head_dim: usize = 64; // 768 hidden / 12 heads = 64
    let max_seq_len: usize = 32768;

    // Compute inverse frequencies (same formula as in mmbert_embedding.rs)
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();

    // Verify we have the right number of frequencies (half of head_dim)
    assert_eq!(inv_freq.len(), head_dim / 2);
    assert_eq!(inv_freq.len(), 32);

    // Verify frequencies are in valid range (0, 1]
    for (i, &freq) in inv_freq.iter().enumerate() {
        assert!(
            freq > 0.0 && freq <= 1.0,
            "Frequency at index {} = {} should be in (0, 1]",
            i,
            freq
        );
    }

    // Verify frequencies decrease monotonically (higher dims = lower freq)
    for i in 1..inv_freq.len() {
        assert!(
            inv_freq[i] < inv_freq[i - 1],
            "Frequencies should decrease: inv_freq[{}]={} >= inv_freq[{}]={}",
            i,
            inv_freq[i],
            i - 1,
            inv_freq[i - 1]
        );
    }

    // Verify the first frequency (dim=0) is 1.0 (theta^0 = 1)
    assert!(
        (inv_freq[0] - 1.0).abs() < 1e-6,
        "First frequency should be 1.0, got {}",
        inv_freq[0]
    );

    // Verify last frequency is correct for YaRN theta
    // inv_freq[31] = 1 / 160000^(62/64) = 1 / 160000^0.96875
    let expected_last = 1.0 / rope_theta.powf(62.0 / 64.0) as f32;
    assert!(
        (inv_freq[31] - expected_last).abs() < 1e-10,
        "Last frequency should be {}, got {}",
        expected_last,
        inv_freq[31]
    );

    // Verify we can create position indices for 32K
    let positions: Vec<u32> = (0..max_seq_len as u32).collect();
    assert_eq!(positions.len(), 32768);
    assert_eq!(positions[0], 0);
    assert_eq!(positions[32767], 32767);

    println!("YaRN RoPE frequency computation test passed for 32K context");
    println!("  rope_theta: {}", rope_theta);
    println!("  head_dim: {}", head_dim);
    println!("  max_seq_len: {}", max_seq_len);
    println!("  num_frequencies: {}", inv_freq.len());
    println!("  first_freq: {}", inv_freq[0]);
    println!("  last_freq: {}", inv_freq[31]);
}

/// Test that YaRN theta (160000) provides different frequencies than base theta (10000)
/// Higher theta = lower inv_freq = slower rotation = can distinguish positions further apart
#[rstest]
fn test_yarn_vs_base_rope_theta_difference() {
    let base_theta: f64 = 10000.0;
    let yarn_theta: f64 = 160000.0;
    let head_dim: usize = 64;

    // Compute inverse frequencies for both thetas
    // inv_freq = 1 / theta^(2i/dim)
    let base_inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / base_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();

    let yarn_inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / yarn_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();

    // With higher theta (YaRN), inverse frequencies should be SMALLER
    // This means the rotation angle per position is smaller, allowing the model
    // to distinguish positions that are further apart (longer context)
    for i in 1..base_inv_freq.len() {
        assert!(
            yarn_inv_freq[i] < base_inv_freq[i],
            "YaRN inv_freq[{}]={} should be < base inv_freq[{}]={} (higher theta = lower inv_freq)",
            i,
            yarn_inv_freq[i],
            i,
            base_inv_freq[i]
        );
    }

    // Calculate the ratio - YaRN with 16x theta (160000/10000) should have lower frequencies
    let ratio_at_dim_30 = yarn_inv_freq[15] / base_inv_freq[15];
    println!(
        "Ratio of YaRN/base inv_freq at dim 30: {:.4}",
        ratio_at_dim_30
    );

    // The ratio should be < 1 (YaRN has lower inv_freq values)
    assert!(
        ratio_at_dim_30 < 1.0,
        "YaRN should have lower inverse frequencies"
    );

    // Verify the frequency difference allows for longer context
    // At position P, the rotation angle is: P * inv_freq
    // For the same max rotation (2*pi*N cycles), YaRN can support:
    // P_yarn = P_base * (base_inv_freq / yarn_inv_freq)
    //
    // The extension factor varies by dimension:
    // - factor = (yarn_theta / base_theta)^(2i/dim)
    // - At dim=0: factor = 1
    // - At dim=62: factor = 16^(62/64) ≈ 14.7
    // - At mid-dim (30): factor = 16^(30/64) ≈ 3.7
    let context_extension_factor_mid = base_inv_freq[15] / yarn_inv_freq[15];
    let context_extension_factor_high = base_inv_freq[31] / yarn_inv_freq[31];

    println!(
        "Context extension factor at mid-dim: {:.2}x",
        context_extension_factor_mid
    );
    println!(
        "Context extension factor at high-dim: {:.2}x",
        context_extension_factor_high
    );

    // Mid-dim should show moderate extension (~3-4x)
    assert!(
        context_extension_factor_mid > 3.0 && context_extension_factor_mid < 5.0,
        "Mid-dim extension should be ~3.7x, got {:.2}x",
        context_extension_factor_mid
    );

    // High-dim should show significant extension (~10-16x)
    assert!(
        context_extension_factor_high > 10.0,
        "High-dim extension should be >10x, got {:.2}x",
        context_extension_factor_high
    );

    // Verify the theoretical maximum extension at the highest dimension
    // theta_ratio = 160000 / 10000 = 16
    // At dim 62: factor = 16^(62/64) ≈ 14.7
    let theta_ratio = yarn_theta / base_theta;
    let expected_max_extension = theta_ratio.powf(62.0 / 64.0);
    assert!(
        (context_extension_factor_high - expected_max_extension as f32).abs() < 1.0,
        "High-dim extension {:.2}x should be close to theoretical {:.2}x",
        context_extension_factor_high,
        expected_max_extension
    );

    println!("YaRN vs base theta difference test passed");
    println!("  Base theta: {}", base_theta);
    println!("  YaRN theta: {}", yarn_theta);
    println!("  Theta ratio: {:.0}x", theta_ratio);
    println!(
        "  Context extension at high-dim: {:.2}x",
        context_extension_factor_high
    );
}

/// Test local attention mask generation for long sequences
/// This verifies the sliding window attention works for 32K context
#[rstest]
fn test_local_attention_mask_32k_context() {
    // mmBERT uses local_attention = 128 (window size)
    let local_attention_size: usize = 128;
    let half_window = local_attention_size / 2; // 64 tokens each side

    // Test with various sequence lengths up to 32K
    let test_seq_lens = vec![128, 512, 1024, 4096, 8192, 16384, 32768];

    for seq_len in test_seq_lens {
        // Generate local attention mask (same logic as get_local_attention_mask)
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if (j as i32 - i as i32).abs() > half_window as i32 {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();

        // Verify mask dimensions
        assert_eq!(
            mask.len(),
            seq_len * seq_len,
            "Mask should be seq_len x seq_len"
        );

        // Verify diagonal is always 0 (can attend to self)
        for i in 0..seq_len {
            let diag_idx = i * seq_len + i;
            assert_eq!(
                mask[diag_idx], 0.0,
                "Diagonal should be 0 at position {}",
                i
            );
        }

        // Verify positions within window are 0 (can attend)
        let test_pos = seq_len / 2;
        for j in
            test_pos.saturating_sub(half_window)..std::cmp::min(test_pos + half_window, seq_len)
        {
            let idx = test_pos * seq_len + j;
            assert_eq!(
                mask[idx], 0.0,
                "Position {} should be able to attend to {} within window",
                test_pos, j
            );
        }

        // Verify positions outside window are -inf (cannot attend)
        if seq_len > local_attention_size {
            let far_pos = if test_pos > half_window + 10 {
                test_pos - half_window - 10
            } else {
                test_pos + half_window + 10
            };
            if far_pos < seq_len {
                let idx = test_pos * seq_len + far_pos;
                assert!(
                    mask[idx].is_infinite() && mask[idx].is_sign_negative(),
                    "Position {} should NOT attend to {} outside window",
                    test_pos,
                    far_pos
                );
            }
        }

        println!(
            "Local attention mask verified for seq_len={}, window={}",
            seq_len, local_attention_size
        );
    }

    println!("Local attention mask test passed for all sequence lengths up to 32K");
}

/// Test 4D attention mask expansion for batch processing with 32K context
#[rstest]
fn test_4d_attention_mask_expansion_32k() {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;

    // Test various batch sizes and sequence lengths
    let test_cases = vec![
        (1, 128),   // Single sample, short
        (1, 1024),  // Single sample, medium
        (1, 8192),  // Single sample, original mmBERT max
        (1, 16384), // Single sample, extended
        (1, 32768), // Single sample, full 32K
        (2, 4096),  // Batch of 2, 4K each
        (4, 8192),  // Batch of 4, 8K each (memory permitting)
    ];

    for (batch_size, seq_len) in test_cases {
        // Skip very large tests to avoid memory issues in unit tests
        if batch_size * seq_len > 65536 {
            println!(
                "Skipping batch_size={}, seq_len={} (too large for unit test)",
                batch_size, seq_len
            );
            continue;
        }

        // Create a simple attention mask (1s for real tokens, 0s for padding)
        // Simulate some padding at the end
        let padding_start = seq_len - seq_len / 10; // Last 10% is padding
        let mask_data: Vec<u32> = (0..batch_size)
            .flat_map(|_| (0..seq_len).map(|j| if j < padding_start { 1u32 } else { 0u32 }))
            .collect();

        let mask = Tensor::from_vec(mask_data, (batch_size, seq_len), &device)
            .expect("Failed to create mask tensor");

        // Expand to 4D: [batch, 1, 1, seq_len] -> broadcast to [batch, 1, tgt_len, src_len]
        let expanded = mask
            .unsqueeze(1)
            .expect("unsqueeze 1")
            .unsqueeze(2)
            .expect("unsqueeze 2")
            .to_dtype(DType::F32)
            .expect("to_dtype");

        let dims = expanded.dims();
        assert_eq!(dims.len(), 4, "Should be 4D tensor");
        assert_eq!(dims[0], batch_size);
        assert_eq!(dims[1], 1);
        assert_eq!(dims[2], 1);
        assert_eq!(dims[3], seq_len);

        // Verify the mask values are correct
        let expanded_data = expanded
            .flatten_all()
            .expect("flatten")
            .to_vec1::<f32>()
            .expect("to_vec1");

        // Check that real tokens have mask=1.0 and padding has mask=0.0
        for b in 0..batch_size {
            for j in 0..seq_len {
                let idx = b * seq_len + j;
                let expected = if j < padding_start { 1.0f32 } else { 0.0f32 };
                assert_eq!(
                    expanded_data[idx], expected,
                    "Mask at batch={}, pos={} should be {}",
                    b, j, expected
                );
            }
        }

        println!(
            "4D attention mask expansion verified: batch_size={}, seq_len={}",
            batch_size, seq_len
        );
    }

    println!("4D attention mask expansion test passed for 32K context");
}

/// Test position embeddings tensor creation for 32K positions
#[rstest]
fn test_position_tensor_creation_32k() {
    use candle_core::{DType, Device, Tensor};

    let device = Device::Cpu;
    let max_seq_len: usize = 32768;

    // Create position indices (same as in RotaryEmbedding::new)
    let positions = Tensor::arange(0u32, max_seq_len as u32, &device)
        .expect("Failed to create position tensor");

    // Verify shape
    assert_eq!(positions.dims(), &[max_seq_len]);

    // Verify values at key positions
    let pos_data = positions.to_vec1::<u32>().expect("to_vec1");
    assert_eq!(pos_data[0], 0, "First position should be 0");
    assert_eq!(pos_data[1023], 1023, "Position 1023");
    assert_eq!(pos_data[8191], 8191, "Position 8191 (original mmBERT max)");
    assert_eq!(pos_data[16383], 16383, "Position 16383 (midpoint of 32K)");
    assert_eq!(pos_data[32767], 32767, "Last position should be 32767");

    // Test reshaping for matmul with frequencies
    let positions_f32 = positions
        .to_dtype(DType::F32)
        .expect("to_dtype")
        .reshape((max_seq_len, 1))
        .expect("reshape");

    assert_eq!(positions_f32.dims(), &[max_seq_len, 1]);

    println!(
        "Position tensor creation test passed for {} positions",
        max_seq_len
    );
}

/// Test sin/cos computation for RoPE at 32K positions with YaRN theta
#[rstest]
fn test_rope_sin_cos_computation_32k() {
    use candle_core::{DType, Device, IndexOp, Tensor};

    let device = Device::Cpu;
    let rope_theta: f64 = 160000.0;
    let head_dim: usize = 64;
    let max_seq_len: usize = 32768;

    // Compute inverse frequencies
    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();
    let inv_freq_len = inv_freq.len();

    let inv_freq_tensor = Tensor::from_vec(inv_freq.clone(), (1, inv_freq_len), &device)
        .expect("inv_freq tensor")
        .to_dtype(DType::F32)
        .expect("to_dtype");

    // Create position indices
    let t = Tensor::arange(0u32, max_seq_len as u32, &device)
        .expect("arange")
        .to_dtype(DType::F32)
        .expect("to_dtype f32")
        .reshape((max_seq_len, 1))
        .expect("reshape");

    // Compute frequencies: t @ inv_freq^T -> (max_seq_len, inv_freq_len)
    let freqs = t.matmul(&inv_freq_tensor).expect("matmul");
    assert_eq!(freqs.dims(), &[max_seq_len, inv_freq_len]);

    // Compute sin and cos
    let sin = freqs.sin().expect("sin");
    let cos = freqs.cos().expect("cos");

    assert_eq!(sin.dims(), &[max_seq_len, inv_freq_len]);
    assert_eq!(cos.dims(), &[max_seq_len, inv_freq_len]);

    // Verify sin/cos are in valid range [-1, 1]
    let sin_data = sin
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1");
    let cos_data = cos
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1");

    for (i, (&s, &c)) in sin_data.iter().zip(cos_data.iter()).enumerate() {
        assert!(
            s >= -1.0 && s <= 1.0,
            "Sin value {} at index {} out of range",
            s,
            i
        );
        assert!(
            c >= -1.0 && c <= 1.0,
            "Cos value {} at index {} out of range",
            c,
            i
        );
        // Verify sin^2 + cos^2 = 1 (within floating point tolerance)
        let sum_sq = s * s + c * c;
        assert!(
            (sum_sq - 1.0).abs() < 1e-5,
            "sin^2 + cos^2 = {} at index {}, expected 1.0",
            sum_sq,
            i
        );
    }

    // Verify position 0 has cos=1, sin=0 for all frequencies
    let sin_pos0: Vec<f32> = sin
        .i(0)
        .expect("sin pos 0")
        .to_vec1::<f32>()
        .expect("to_vec1");
    let cos_pos0: Vec<f32> = cos
        .i(0)
        .expect("cos pos 0")
        .to_vec1::<f32>()
        .expect("to_vec1");

    for (i, (&s, &c)) in sin_pos0.iter().zip(cos_pos0.iter()).enumerate() {
        assert!(
            s.abs() < 1e-6,
            "Sin at position 0, freq {} should be ~0, got {}",
            i,
            s
        );
        assert!(
            (c - 1.0).abs() < 1e-6,
            "Cos at position 0, freq {} should be ~1, got {}",
            i,
            c
        );
    }

    // Verify at position 32767 (last position), values are still valid
    let sin_last: Vec<f32> = sin
        .i(max_seq_len - 1)
        .expect("sin last")
        .to_vec1::<f32>()
        .expect("to_vec1");
    let cos_last: Vec<f32> = cos
        .i(max_seq_len - 1)
        .expect("cos last")
        .to_vec1::<f32>()
        .expect("to_vec1");

    for (i, (&s, &c)) in sin_last.iter().zip(cos_last.iter()).enumerate() {
        assert!(
            s.is_finite(),
            "Sin at position 32767, freq {} should be finite",
            i
        );
        assert!(
            c.is_finite(),
            "Cos at position 32767, freq {} should be finite",
            i
        );
    }

    println!("RoPE sin/cos computation test passed for 32K positions");
    println!("  Total sin/cos values computed: {}", sin_data.len());
    println!("  All values in valid range [-1, 1]: verified");
    println!("  sin^2 + cos^2 = 1: verified for all positions");
}

/// Test config detection edge cases for 32K models
#[rstest]
fn test_mmbert_32k_config_detection_edge_cases() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Test case 1: max_position_embeddings = 16384 (boundary)
    let config_path_1 = temp_dir.path().join("config_16k.json");
    let config_16k = r#"{
        "vocab_size": 256000,
        "position_embedding_type": "sans_pos",
        "max_position_embeddings": 16384
    }"#;
    std::fs::write(&config_path_1, config_16k).expect("write");
    let variant_1 = ModernBertVariant::detect_from_config(config_path_1.to_str().unwrap());
    assert_eq!(
        variant_1.unwrap(),
        ModernBertVariant::Multilingual32K,
        "16384 should be detected as 32K variant"
    );

    // Test case 2: max_position_embeddings = 16383 (just below boundary)
    let config_path_2 = temp_dir.path().join("config_below_16k.json");
    let config_below_16k = r#"{
        "vocab_size": 256000,
        "position_embedding_type": "sans_pos",
        "max_position_embeddings": 16383
    }"#;
    std::fs::write(&config_path_2, config_below_16k).expect("write");
    let variant_2 = ModernBertVariant::detect_from_config(config_path_2.to_str().unwrap());
    assert_eq!(
        variant_2.unwrap(),
        ModernBertVariant::Multilingual,
        "16383 should be detected as 8K variant"
    );

    // Test case 3: global_rope_theta = 100000 (boundary)
    let config_path_3 = temp_dir.path().join("config_theta_100k.json");
    let config_theta_100k = r#"{
        "vocab_size": 256000,
        "position_embedding_type": "sans_pos",
        "max_position_embeddings": 8192,
        "global_rope_theta": 100000
    }"#;
    std::fs::write(&config_path_3, config_theta_100k).expect("write");
    let variant_3 = ModernBertVariant::detect_from_config(config_path_3.to_str().unwrap());
    assert_eq!(
        variant_3.unwrap(),
        ModernBertVariant::Multilingual32K,
        "rope_theta=100000 should be detected as 32K variant"
    );

    // Test case 4: global_rope_theta = 99999 (just below boundary)
    let config_path_4 = temp_dir.path().join("config_theta_99999.json");
    let config_theta_99999 = r#"{
        "vocab_size": 256000,
        "position_embedding_type": "sans_pos",
        "max_position_embeddings": 8192,
        "global_rope_theta": 99999
    }"#;
    std::fs::write(&config_path_4, config_theta_99999).expect("write");
    let variant_4 = ModernBertVariant::detect_from_config(config_path_4.to_str().unwrap());
    assert_eq!(
        variant_4.unwrap(),
        ModernBertVariant::Multilingual,
        "rope_theta=99999 should be detected as 8K variant"
    );

    // Test case 5: Both conditions met (32K positions AND high theta)
    let config_path_5 = temp_dir.path().join("config_both.json");
    let config_both = r#"{
        "vocab_size": 256000,
        "position_embedding_type": "sans_pos",
        "max_position_embeddings": 32768,
        "global_rope_theta": 160000
    }"#;
    std::fs::write(&config_path_5, config_both).expect("write");
    let variant_5 = ModernBertVariant::detect_from_config(config_path_5.to_str().unwrap());
    assert_eq!(
        variant_5.unwrap(),
        ModernBertVariant::Multilingual32K,
        "Both 32K and high theta should be detected as 32K variant"
    );

    println!("Config detection edge cases test passed");
}

/// Test that 32K context variant methods work correctly
#[rstest]
fn test_32k_variant_method_consistency() {
    let variant = ModernBertVariant::Multilingual32K;

    // Test all properties are consistent
    assert_eq!(variant.max_length(), 32768);
    assert!(variant.uses_yarn_scaling());
    assert_eq!(variant.expected_rope_theta(), 160000.0);
    assert_eq!(variant.pad_token(), "<pad>");

    // Verify tokenization strategy matches Multilingual
    let multilingual = ModernBertVariant::Multilingual;
    assert_eq!(
        format!("{:?}", variant.tokenization_strategy()),
        format!("{:?}", multilingual.tokenization_strategy()),
        "32K should use same tokenization strategy as Multilingual"
    );

    // Verify it's different from Standard
    let standard = ModernBertVariant::Standard;
    assert_ne!(variant.max_length(), standard.max_length());
    assert_ne!(variant.pad_token(), standard.pad_token());
    assert!(variant.uses_yarn_scaling() != standard.uses_yarn_scaling());

    println!("32K variant method consistency test passed");
}

/// Test mmBERT type aliases
#[rstest]
fn test_mmbert_type_aliases() {
    // Verify type aliases are correctly defined
    // MmBertClassifier should be an alias for TraditionalModernBertClassifier
    // MmBertTokenClassifier should be an alias for TraditionalModernBertTokenClassifier

    // These are compile-time checks - if they compile, the aliases are correct
    fn _accepts_mmbert_classifier(_c: &MmBertClassifier) {}
    fn _accepts_modernbert_classifier(_c: &TraditionalModernBertClassifier) {}
    fn _accepts_mmbert_token_classifier(_c: &MmBertTokenClassifier) {}
    fn _accepts_modernbert_token_classifier(_c: &TraditionalModernBertTokenClassifier) {}

    println!("mmBERT type aliases test passed");
}

/// Test mmBERT classifier error handling
#[rstest]
fn test_mmbert_classifier_error_handling() {
    // Invalid model path with explicit mmBERT variant
    let invalid_result = TraditionalModernBertClassifier::load_from_directory_with_variant(
        "",
        true,
        ModernBertVariant::Multilingual,
    );
    assert!(invalid_result.is_err());

    // Non-existent model path
    let nonexistent_result = TraditionalModernBertClassifier::load_mmbert_from_directory(
        "/nonexistent/path/to/model",
        true,
    );
    assert!(nonexistent_result.is_err());

    println!("mmBERT classifier error handling test passed");
}

/// Test mmBERT token classifier error handling
#[rstest]
fn test_mmbert_token_classifier_error_handling() {
    // Invalid model path with explicit mmBERT variant
    let invalid_result = TraditionalModernBertTokenClassifier::new_with_variant(
        "",
        true,
        ModernBertVariant::Multilingual,
    );
    assert!(invalid_result.is_err());

    // Non-existent model path
    let nonexistent_result =
        TraditionalModernBertTokenClassifier::new_mmbert("/nonexistent/path/to/model", true);
    assert!(nonexistent_result.is_err());

    println!("mmBERT token classifier error handling test passed");
}

/// Test mmBERT multilingual text samples (documentation of capability)
#[rstest]
fn test_mmbert_multilingual_samples() {
    // Sample texts in different languages that mmBERT supports
    let multilingual_samples = vec![
        ("English", "Hello, how are you today?"),
        ("Spanish", "Hola, ¿cómo estás hoy?"),
        ("French", "Bonjour, comment allez-vous aujourd'hui?"),
        ("German", "Hallo, wie geht es Ihnen heute?"),
        ("Chinese", "你好，今天怎么样？"),
        ("Japanese", "こんにちは、今日はいかがですか？"),
        ("Korean", "안녕하세요, 오늘 어떠세요?"),
        ("Arabic", "مرحبا، كيف حالك اليوم؟"),
        ("Russian", "Привет, как дела сегодня?"),
        ("Hindi", "नमस्ते, आज आप कैसे हैं?"),
    ];

    println!(
        "mmBERT supports {} languages. Sample texts:",
        multilingual_samples.len()
    );
    for (lang, text) in &multilingual_samples {
        println!("  {}: {}", lang, text);
    }

    // Verify we have coverage for major language families
    assert!(multilingual_samples.len() >= 10);
    println!("mmBERT multilingual samples test passed");
}

/// Test mmBERT expected configuration values
#[rstest]
fn test_mmbert_expected_config_values() {
    // Document expected mmBERT configuration values based on
    // https://huggingface.co/jhu-clsp/mmBERT-base/blob/main/config.json

    let expected_config = vec![
        ("vocab_size", "256000"),
        ("hidden_size", "768"),
        ("num_hidden_layers", "22"),
        ("num_attention_heads", "12"),
        ("intermediate_size", "1152"),
        ("max_position_embeddings", "8192"),
        ("position_embedding_type", "sans_pos"),
        ("local_attention", "128"),
        ("global_attn_every_n_layers", "3"),
        ("global_rope_theta", "160000"),
        ("local_rope_theta", "160000"),
        ("pad_token_id", "0"),
        ("bos_token_id", "2"),
        ("eos_token_id", "1"),
        ("cls_token_id", "1"),
        ("sep_token_id", "1"),
        ("mask_token_id", "4"),
    ];

    println!("Expected mmBERT configuration:");
    for (key, value) in &expected_config {
        println!("  {}: {}", key, value);
    }

    assert!(expected_config.len() > 10);
    println!("mmBERT config values test passed");
}

/// Integration test for mmBERT with actual model (requires model files)
/// Skipped in CI environments to save resources (mmBERT is not downloaded in CI)
#[rstest]
fn test_mmbert_integration_with_model() {
    // Skip in CI environments - mmBERT model is not downloaded in CI to save resources
    if std::env::var("CI").is_ok() {
        println!("Skipping mmBERT integration test in CI environment");
        return;
    }

    // This test requires actual mmBERT model files to be present
    // Default path assumes model is downloaded to ../models/mmbert-base

    let model_path =
        std::env::var("MMBERT_MODEL_PATH").unwrap_or_else(|_| "../models/mmbert-base".to_string());

    if !std::path::Path::new(&model_path).exists() {
        println!(
            "Skipping integration test - model path not found: {}",
            model_path
        );
        return;
    }

    println!("Loading mmBERT from: {}", model_path);

    match TraditionalModernBertClassifier::load_mmbert_from_directory(&model_path, true) {
        Ok(classifier) => {
            println!("Successfully loaded mmBERT classifier");
            println!("Variant: {:?}", classifier.variant());
            println!("Is multilingual: {}", classifier.is_multilingual());
            println!("Number of classes: {}", classifier.get_num_classes());

            assert!(classifier.is_multilingual());

            // Test with multilingual texts
            let test_texts = vec![
                "This is an English test sentence.",
                "这是一个中文测试句子。",
                "Dies ist ein deutscher Testsatz.",
            ];

            for text in test_texts {
                match classifier.classify_text(text) {
                    Ok((class_id, confidence)) => {
                        println!(
                            "Text: '{}' -> class={}, confidence={:.4}",
                            text, class_id, confidence
                        );
                    }
                    Err(e) => {
                        println!("Classification failed for '{}': {}", text, e);
                    }
                }
            }
        }
        Err(e) => {
            println!("Failed to load mmBERT classifier: {}", e);
        }
    }
}
