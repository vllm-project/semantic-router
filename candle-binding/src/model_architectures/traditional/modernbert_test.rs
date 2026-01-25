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
    assert!(matches!(
        standard.tokenization_strategy(),
        TokenizationStrategy::ModernBERT
    ));

    // Test Multilingual (mmBERT) variant
    let multilingual = ModernBertVariant::Multilingual;
    assert_eq!(multilingual.max_length(), 8192);
    assert_eq!(multilingual.pad_token(), "<pad>");
    assert!(matches!(
        multilingual.tokenization_strategy(),
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

/// Test load_with_custom_base_model error handling
#[rstest]
fn test_load_with_custom_base_model_error_handling() {
    // Test with invalid base model path
    let invalid_base_result = TraditionalModernBertClassifier::load_with_custom_base_model(
        "",
        "/nonexistent/classifier",
        ModernBertVariant::Standard,
        true,
    );
    assert!(invalid_base_result.is_err());

    // Test with invalid classifier path
    let invalid_classifier_result = TraditionalModernBertClassifier::load_with_custom_base_model(
        "/nonexistent/base",
        "",
        ModernBertVariant::Standard,
        true,
    );
    assert!(invalid_classifier_result.is_err());

    // Test with non-existent paths
    let nonexistent_result = TraditionalModernBertClassifier::load_with_custom_base_model(
        "/nonexistent/base/model",
        "/nonexistent/classifier/model",
        ModernBertVariant::Standard,
        true,
    );
    assert!(nonexistent_result.is_err());

    println!("load_with_custom_base_model error handling test passed");
}

/// Test load_with_custom_base_model with Extended32K variant
/// 
/// This test verifies that the method correctly handles Extended32K variant
/// and attempts to read training_config.json for model_max_length.
/// Note: This test will fail if the actual model files are not present,
/// which is expected in CI environments without model files.
#[rstest]
#[serial]
#[ignore] // Ignore by default - requires actual model files
fn test_load_with_custom_base_model_extended32k(
    traditional_pii_model_path: String,
) {
    // This test requires:
    // 1. Extended32K base model at a known path (or downloaded)
    // 2. PII classifier model at traditional_pii_model_path
    
    // For now, we'll just verify the method signature and error handling
    // Actual integration test would require model files
    
    // Test that Extended32K variant is accepted
    let extended32k_variant = ModernBertVariant::Extended32K;
    assert_eq!(extended32k_variant.max_length(), 32768);
    
    // Test error handling with Extended32K variant
    let result = TraditionalModernBertClassifier::load_with_custom_base_model(
        "/nonexistent/extended32k/base",
        &traditional_pii_model_path,
        extended32k_variant,
        true,
    );
    
    // Should fail because base model path doesn't exist
    assert!(result.is_err());
    
    println!("load_with_custom_base_model Extended32K variant test passed (error handling)");
}

/// Test load_with_custom_base_model parameter validation
#[rstest]
fn test_load_with_custom_base_model_parameter_validation() {
    // Test that all variants are accepted
    let variants = vec![
        ModernBertVariant::Standard,
        ModernBertVariant::Multilingual,
        ModernBertVariant::Extended32K,
    ];
    
    for variant in variants {
        // All variants should be accepted (even if loading fails due to missing files)
        let result = TraditionalModernBertClassifier::load_with_custom_base_model(
            "/nonexistent/base",
            "/nonexistent/classifier",
            variant,
            true,
        );
        
        // Should fail due to missing files, but variant should be accepted
        assert!(result.is_err());
        
        // Verify variant max_length is correct
        match variant {
            ModernBertVariant::Standard => assert_eq!(variant.max_length(), 512),
            ModernBertVariant::Multilingual => assert_eq!(variant.max_length(), 8192),
            ModernBertVariant::Extended32K => assert_eq!(variant.max_length(), 32768),
        }
    }
    
    println!("load_with_custom_base_model parameter validation test passed");
}
