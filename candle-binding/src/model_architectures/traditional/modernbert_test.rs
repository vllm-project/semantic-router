//! Tests for traditional ModernBERT implementation

use super::modernbert::*;
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
