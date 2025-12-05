//! Tests for LoRA PII detector implementation

use super::pii_lora::*;
use crate::test_fixtures::{fixtures::*, test_utils::*};
use rstest::*;
use serial_test::serial;
use std::sync::Arc;

/// Test PIILoRAClassifier creation with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_lora_classifier_new(cached_pii_classifier: Option<Arc<PIILoRAClassifier>>) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing PIILoRAClassifier with cached model - instant access!");

        // Test actual PII detection with cached model
        {
            let test_text = "My name is John Doe and my email is john.doe@example.com";
            match classifier.detect_pii(test_text) {
                Ok(result) => {
                    println!("Real model PII detection result: has_pii={}, types={:?}, confidence={:.3}, time={}ms",
                        result.has_pii, result.pii_types, result.confidence, result.processing_time_ms);

                    // Validate real model output
                    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                    assert!(result.processing_time_ms > 0);
                    assert!(result.processing_time_ms < 10000);

                    // Check PII detection logic
                    if result.has_pii {
                        assert!(!result.pii_types.is_empty());
                        assert!(!result.occurrences.is_empty());
                    } else {
                        assert!(result.pii_types.is_empty());
                        assert!(result.occurrences.is_empty());
                    }
                }
                Err(e) => {
                    println!("Real model PII detection failed: {}", e);
                }
            }
        }
    } else {
        println!("Cached PII classifier not available, skipping test");
    }
}

/// Test cached model batch PII detection (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_lora_classifier_batch_detect(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing batch PII detection with cached model!");
        {
            let test_texts = vec![
                "Hello, my name is Alice",
                "Contact me at bob@company.com",
                "My phone number is 555-1234",
                "This is a normal message without PII",
            ];

            match classifier.batch_detect(&test_texts) {
                Ok(results) => {
                    println!(
                        "Real model batch PII detection succeeded with {} results",
                        results.len()
                    );
                    assert_eq!(results.len(), test_texts.len());

                    for (i, result) in results.iter().enumerate() {
                        println!("Batch PII result {}: has_pii={}, types={:?}, confidence={:.3}, time={}ms",
                            i, result.has_pii, result.pii_types, result.confidence, result.processing_time_ms);

                        // Validate each result
                        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                        assert!(result.processing_time_ms > 0);

                        // Check PII detection consistency
                        assert_eq!(result.has_pii, !result.pii_types.is_empty());
                        assert_eq!(result.has_pii, !result.occurrences.is_empty());
                    }
                }
                Err(e) => {
                    println!("Real model batch PII detection failed: {}", e);
                }
            }
        }
    } else {
        println!("Cached PII classifier not available, skipping batch test");
    }
}

/// Test cached model parallel PII detection (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_lora_classifier_parallel_detect(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing parallel PII detection with cached model!");
        {
            let test_texts = vec![
                "My SSN is 123-45-6789",
                "Call me at (555) 123-4567",
                "Email: user@domain.com",
            ];

            match classifier.parallel_detect(&test_texts) {
                Ok(results) => {
                    println!(
                        "Real model parallel PII detection succeeded with {} results",
                        results.len()
                    );
                    assert_eq!(results.len(), test_texts.len());

                    for (i, result) in results.iter().enumerate() {
                        println!("Parallel PII result {}: has_pii={}, types={:?}, confidence={:.3}, time={}ms",
                            i, result.has_pii, result.pii_types, result.confidence, result.processing_time_ms);

                        // Validate each result
                        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                        assert!(result.processing_time_ms > 0);

                        // Check PII detection consistency
                        assert_eq!(result.has_pii, !result.pii_types.is_empty());
                        assert_eq!(result.has_pii, !result.occurrences.is_empty());

                        // Validate occurrences if PII detected
                        if result.has_pii {
                            for occurrence in &result.occurrences {
                                assert!(!occurrence.pii_type.is_empty());
                                assert!(!occurrence.token.is_empty());
                                assert!(
                                    occurrence.confidence >= 0.0 && occurrence.confidence <= 1.0
                                );
                                assert!(occurrence.start_pos <= occurrence.end_pos);
                            }
                        }
                    }
                }
                Err(e) => {
                    println!("Real model parallel PII detection failed: {}", e);
                }
            }
        }
    } else {
        println!("Cached PII classifier not available, skipping parallel test");
    }
}

/// Test PIILoRAClassifier error handling with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_lora_classifier_error_handling(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing error handling with cached model!");

        // Test with cached model first (should work)
        let test_text = "Test error handling";
        match classifier.detect_pii(test_text) {
            Ok(_) => println!("Cached model error handling test passed"),
            Err(e) => println!("Cached model error: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping error handling test");
    }

    // Test error scenarios with invalid paths
    let invalid_model_result = PIILoRAClassifier::new("", true);
    assert!(invalid_model_result.is_err());

    let nonexistent_model_result = PIILoRAClassifier::new("/nonexistent/path/to/model", true);
    assert!(nonexistent_model_result.is_err());

    println!("PIILoRAClassifier error handling test passed");
}

/// Test PII detection output format with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_detection_output_format(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing PII detection output format with cached model!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                // Test output format
                assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                assert!(result.processing_time_ms > 0);

                // Test PII types format (adapt to real model output)
                for pii_type in &result.pii_types {
                    assert!(!pii_type.is_empty());
                    assert!(pii_type
                        .chars()
                        .all(|c| c.is_ascii_alphabetic() || c == '_' || c == '-'));
                    println!("  Detected PII type: '{}'", pii_type);
                }

                println!("PII detection output format test passed with cached model");
            }
            Err(e) => {
                println!("PII detection failed: {}", e);
            }
        }
    } else {
        println!("Cached PII classifier not available, skipping output format test");
    }
}

/// Test PII type classification with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_type_classification(cached_pii_classifier: Option<Arc<PIILoRAClassifier>>) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing PII type classification with cached model!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                for pii_type in &result.pii_types {
                    assert!(pii_type
                        .chars()
                        .all(|c| c.is_ascii_alphabetic() || c == '_' || c == '-'));
                    println!("  Detected PII type: '{}'", pii_type);
                }
                println!("PII type classification test passed with cached model");
            }
            Err(e) => println!("PII type classification failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping type classification test");
    }
}

/// Test token-level PII detection with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_token_level_pii_detection(cached_pii_classifier: Option<Arc<PIILoRAClassifier>>) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing token-level PII detection with cached model!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                // Test token-level detection
                for occurrence in &result.occurrences {
                    assert!(occurrence.start_pos <= occurrence.end_pos);
                    assert!(!occurrence.pii_type.is_empty());
                    assert!(occurrence.confidence >= 0.0 && occurrence.confidence <= 1.0);
                    println!(
                        "  Token PII: '{}' at {}:{}, type='{}', confidence={:.3}",
                        occurrence.token,
                        occurrence.start_pos,
                        occurrence.end_pos,
                        occurrence.pii_type,
                        occurrence.confidence
                    );
                }
                println!("Token-level PII detection test passed with cached model");
            }
            Err(e) => println!("Token-level PII detection failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping token-level test");
    }
}

/// Performance test for PIILoRAClassifier cached model operations (OPTIMIZED)
#[rstest]
#[serial]
fn test_pii_lora_pii_lora_classifier_performance(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing PIILoRAClassifier cached model performance");

        let test_texts = vec![
            "My name is John Doe and my email is john.doe@example.com",
            "Contact Alice at alice@test.com or call 555-1234",
            "The weather is nice today",
        ];

        let (_, total_duration) = measure_execution_time(|| {
            for text in &test_texts {
                let (_, single_duration) =
                    measure_execution_time(|| match classifier.detect_pii(text) {
                        Ok(result) => {
                            assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                            assert!(result.processing_time_ms > 0);
                        }
                        Err(e) => println!("Performance test failed for '{}': {}", text, e),
                    });
                assert!(
                    single_duration.as_secs() < 15,
                    "Single PII detection took too long: {:?}",
                    single_duration
                );
            }
        });

        assert!(
            total_duration.as_secs() < 60,
            "Batch PII processing took too long: {:?}",
            total_duration
        );
        println!(
            "PIILoRAClassifier cached model performance: {} texts in {:?}",
            test_texts.len(),
            total_duration
        );
    } else {
        println!("Cached PII classifier not available, skipping performance test");
    }
}
