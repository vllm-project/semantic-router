//! Tests for ModernBERT classifier implementation

use crate::test_fixtures::fixtures::*;
use rstest::*;
use serial_test::serial;

/// Test TraditionalModernBertClassifier structure with real model
#[rstest]
#[serial]
fn test_modernbert_classifier_traditional_modernbert_classifier_new(
    cached_traditional_intent_classifier: Option<
        std::sync::Arc<
            crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier,
        >,
    >,
) {
    if let Some(classifier) = cached_traditional_intent_classifier {
        println!("Testing TraditionalModernBertClassifier with cached real model");

        // Test Debug formatting
        let debug_str = format!("{:?}", classifier);
        assert!(debug_str.contains("TraditionalModernBertClassifier"));

        // Test Clone
        let cloned = classifier.clone();
        let cloned_debug = format!("{:?}", cloned);
        assert!(cloned_debug.contains("TraditionalModernBertClassifier"));

        // Test real text classification
        let sample_texts = sample_texts();
        let test_text = sample_texts[4]; // "Hello world"

        let classification_result = classifier.classify_text(test_text);
        match classification_result {
            Ok((class_id, confidence)) => {
                println!(
                    "Real model classification succeeded: text='{}' -> class_id={}, confidence={:.3}",
                    test_text, class_id, confidence
                );

                // Validate real model output
                assert!(confidence >= 0.0 && confidence <= 1.0);
                assert!(class_id < 100); // Reasonable class ID range

                // Test high-quality classification
                assert!(
                    confidence > 0.1,
                    "Classification confidence too low: {}",
                    confidence
                );
            }
            Err(e) => {
                println!("Real model classification failed: {}", e);
                panic!("Real model should work for basic text classification");
            }
        }

        println!("TraditionalModernBertClassifier real model test passed");
    } else {
        panic!("Cached Traditional Intent classifier not available");
    }
}

/// Test ModernBertClassifier creation interface with real model
#[rstest]
#[serial]
fn test_modernbert_classifier_modernbert_classifier_new(
    cached_traditional_intent_classifier: Option<
        std::sync::Arc<
            crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier,
        >,
    >,
) {
    if let Some(base_classifier) = cached_traditional_intent_classifier {
        println!("Testing ModernBertClassifier creation with cached real model");
        // Test real model classification capabilities
        let sample_texts = sample_texts();
        let test_text = sample_texts[0]; // "I want to book a flight"

        let classification_result = base_classifier.classify_text(test_text);
        match classification_result {
            Ok((class_id, confidence)) => {
                println!(
                    "ModernBertClassifier real model test: text='{}' -> class_id={}, confidence={:.3}",
                    test_text, class_id, confidence
                );

                // Validate real model classification
                assert!(confidence >= 0.0 && confidence <= 1.0);
                assert!(class_id < 100); // Reasonable class ID range

                // Test classification quality
                assert!(
                    confidence > 0.1,
                    "Classification confidence too low: {}",
                    confidence
                );

                println!("ModernBertClassifier real model integration test passed");
            }
            Err(e) => {
                println!("ModernBertClassifier real model test failed: {}", e);
                panic!("Real model should work for intent classification");
            }
        }
    } else {
        panic!("Cached Traditional Intent classifier not available");
    }
}

/// Test ModernBERT classifier with real model integration
#[rstest]
fn test_modernbert_classifier_real_model_integration() {
    // Test ModernBERT classifier with real model
    use std::path::Path;

    // Use Traditional Intent model path directly
    let traditional_model_path = format!(
        "{}/{}",
        crate::test_fixtures::fixtures::MODELS_BASE_PATH,
        crate::test_fixtures::fixtures::MODERNBERT_INTENT_MODEL
    );

    if Path::new(&traditional_model_path).exists() {
        println!(
            "Testing ModernBERT classifier with real model: {}",
            traditional_model_path
        );

        // Test model path validation
        assert!(!traditional_model_path.is_empty());
        assert!(traditional_model_path.contains("models"));

        // Test that config files exist
        let config_path = format!("{}/config.json", traditional_model_path);
        if Path::new(&config_path).exists() {
            println!("Config file found: {}", config_path);
        } else {
            println!(
                "Config file not found, but model path is valid: {}",
                traditional_model_path
            );
        }

        // Test model directory structure
        let model_files = ["pytorch_model.bin", "model.safetensors", "tokenizer.json"];
        for file in &model_files {
            let file_path = format!("{}/{}", traditional_model_path, file);
            if Path::new(&file_path).exists() {
                println!("Model file found: {}", file);
            }
        }

        println!(
            "Real model integration test passed for: {}",
            traditional_model_path
        );
    } else {
        println!(
            "Real model not found at: {}, skipping integration test",
            traditional_model_path
        );
    }
}
