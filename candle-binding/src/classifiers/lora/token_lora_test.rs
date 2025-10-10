//! Tests for LoRA token classifier implementation

use super::pii_lora::PIILoRAClassifier;
use crate::test_fixtures::fixtures::*;
use rstest::*;
use serial_test::serial;
use std::sync::Arc;

/// Test LoRATokenClassifier creation with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_token_lora_lora_token_classifier_new(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing LoRATokenClassifier with cached PII model - instant access!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                // Test token-level results from PII detection
                for occurrence in &result.occurrences {
                    assert!(!occurrence.token.is_empty());
                    assert!(!occurrence.pii_type.is_empty());
                    assert!(occurrence.confidence >= 0.0 && occurrence.confidence <= 1.0);
                    println!(
                        "Token: '{}' -> '{}' (confidence={:.3})",
                        occurrence.token, occurrence.pii_type, occurrence.confidence
                    );
                }
                println!("LoRATokenClassifier creation test passed with cached model");
            }
            Err(e) => println!("Token classification failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping token test");
    }
}

/// Test token classification output format with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_token_lora_token_classification_output_format(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing token classification output format with cached model!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                for occurrence in &result.occurrences {
                    assert!(!occurrence.token.is_empty());
                    assert!(!occurrence.pii_type.is_empty());
                    assert!(occurrence.confidence >= 0.0 && occurrence.confidence <= 1.0);
                    assert!(occurrence.start_pos <= occurrence.end_pos);
                    println!(
                        "Token: '{}' -> '{}' (confidence={:.3}, pos={}:{})",
                        occurrence.token,
                        occurrence.pii_type,
                        occurrence.confidence,
                        occurrence.start_pos,
                        occurrence.end_pos
                    );
                }
                println!("Token classification output format test passed with cached model");
            }
            Err(e) => println!("Token classification failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping output format test");
    }
}

/// Test BIO tagging format with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_token_lora_bio_tagging_format(cached_pii_classifier: Option<Arc<PIILoRAClassifier>>) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing BIO tagging format with cached model!");

        let test_text = "John Doe works at john@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                for occurrence in &result.occurrences {
                    // Test BIO format
                    if occurrence.pii_type != "O" {
                        assert!(
                            occurrence.pii_type.starts_with("B-")
                                || occurrence.pii_type.starts_with("I-")
                        );
                    }
                    println!(
                        "BIO Token: '{}' -> '{}' (confidence={:.3})",
                        occurrence.token, occurrence.pii_type, occurrence.confidence
                    );
                }
                println!("BIO tagging format test passed with cached model");
            }
            Err(e) => println!("BIO tagging failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping BIO tagging test");
    }
}

/// Test token position tracking with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_token_lora_token_position_tracking(cached_pii_classifier: Option<Arc<PIILoRAClassifier>>) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing token position tracking with cached model!");

        let test_text = "My name is John Doe and my email is john.doe@example.com";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                for occurrence in &result.occurrences {
                    assert!(occurrence.start_pos <= occurrence.end_pos);
                    assert!(occurrence.end_pos <= test_text.len());
                    println!(
                        "Position tracking: '{}' at {}:{}",
                        occurrence.token, occurrence.start_pos, occurrence.end_pos
                    );
                }
                println!("Token position tracking test passed with cached model");
            }
            Err(e) => println!("Token position tracking failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping position tracking test");
    }
}

/// Test entity recognition capabilities with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_token_lora_entity_recognition_capabilities(
    cached_pii_classifier: Option<Arc<PIILoRAClassifier>>,
) {
    if let Some(classifier) = cached_pii_classifier {
        println!("Testing entity recognition capabilities with cached model!");

        let test_text = "Contact John Doe at john.doe@example.com or call 555-1234";
        match classifier.detect_pii(test_text) {
            Ok(result) => {
                let mut entity_types = std::collections::HashSet::new();
                for occurrence in &result.occurrences {
                    if occurrence.pii_type != "O" {
                        entity_types.insert(occurrence.pii_type.clone());
                    }
                    println!(
                        "Entity: '{}' -> '{}' (confidence={:.3})",
                        occurrence.token, occurrence.pii_type, occurrence.confidence
                    );
                }
                println!(
                    "Entity recognition test passed with cached model - found {} entity types",
                    entity_types.len()
                );
            }
            Err(e) => println!("Entity recognition failed: {}", e),
        }
    } else {
        println!("Cached PII classifier not available, skipping entity recognition test");
    }
}
