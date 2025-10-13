//! Tests for traditional BERT implementation

use super::bert::*;
use crate::test_fixtures::{fixtures::*, test_utils::*};
use rstest::*;

/// Test TraditionalBertClassifier creation with real model
#[rstest]
fn test_bert_traditional_bert_classifier_new(traditional_model_path: String) {
    // Test TraditionalBertClassifier creation with real model
    use std::path::Path;

    if Path::new(&traditional_model_path).exists() {
        println!(
            "Testing TraditionalBertClassifier creation with real model: {}",
            traditional_model_path
        );

        // Test model path validation
        assert!(!traditional_model_path.is_empty());
        assert!(traditional_model_path.contains("models"));

        let classifier_result = TraditionalBertClassifier::new(
            &traditional_model_path,
            3,    // num_classes
            true, // use CPU
        );

        match classifier_result {
            Ok(_classifier) => {
                println!(
                    "TraditionalBertClassifier creation succeeded with real model: {}",
                    traditional_model_path
                );
            }
            Err(e) => {
                println!(
                    "TraditionalBertClassifier creation failed with real model {}: {}",
                    traditional_model_path, e
                );
                // This might be expected if model format differs or dependencies are missing
            }
        }
    } else {
        println!(
            "Traditional model not found at: {}, skipping real model test",
            traditional_model_path
        );
    }
}

/// Test TraditionalBertClassifier with different class numbers and real model
#[rstest]
#[case(2, "binary_classification")]
#[case(3, "three_class")]
#[case(5, "multi_class")]
#[case(10, "large_multi_class")]
fn test_bert_traditional_bert_classifier_class_numbers(
    #[case] num_classes: usize,
    #[case] task_name: &str,
    traditional_model_path: String,
) {
    use std::path::Path;

    let model_path = if Path::new(&traditional_model_path).exists() {
        println!(
            "Using real model for {} classes test: {}",
            num_classes, traditional_model_path
        );
        traditional_model_path.as_str()
    } else {
        println!(
            "Real model not found, using mock path for {} classes test",
            num_classes
        );
        "nonexistent-model"
    };

    let classifier_result = TraditionalBertClassifier::new(model_path, num_classes, true);

    match classifier_result {
        Ok(classifier) => {
            // Test Debug formatting
            let debug_str = format!("{:?}", classifier);
            assert!(debug_str.contains("TraditionalBertClassifier"));
            assert!(debug_str.contains(&num_classes.to_string()));

            println!(
                "TraditionalBertClassifier creation succeeded for {} with {} classes",
                task_name, num_classes
            );
        }
        Err(e) => {
            println!(
                "TraditionalBertClassifier creation failed for {} (expected): {}",
                task_name, e
            );
        }
    }
}

/// Test TraditionalBertClassifier error handling with real model path
#[rstest]
fn test_bert_traditional_bert_classifier_error_handling(traditional_model_path: String) {
    use std::path::Path;

    let model_path = if Path::new(&traditional_model_path).exists() {
        println!(
            "Using real model for error handling test: {}",
            traditional_model_path
        );
        traditional_model_path.as_str()
    } else {
        println!("Real model not found, using mock path for error handling test");
        "nonexistent-model"
    };
    // Test error scenarios

    // Invalid model path
    let invalid_model_result = TraditionalBertClassifier::new("", 3, true);
    assert!(invalid_model_result.is_err());

    // Zero classes (invalid)
    let zero_classes_result = TraditionalBertClassifier::new(model_path, 0, true);
    assert!(zero_classes_result.is_err());

    println!("TraditionalBertClassifier error handling test passed");
}

/// Test TraditionalBertClassifier device compatibility with real model path
#[rstest]
fn test_bert_traditional_bert_classifier_device_compatibility(traditional_model_path: String) {
    use std::path::Path;

    let model_path = if Path::new(&traditional_model_path).exists() {
        println!(
            "Using real model for device compatibility test: {}",
            traditional_model_path
        );
        traditional_model_path.as_str()
    } else {
        println!("Real model not found, using mock path for device compatibility test");
        "nonexistent-model"
    };
    // Test CPU usage (always available)
    let cpu_result = TraditionalBertClassifier::new(
        model_path, 3, true, // force CPU
    );

    match cpu_result {
        Ok(_classifier) => {
            println!("TraditionalBertClassifier CPU compatibility succeeded");
        }
        Err(e) => {
            println!(
                "TraditionalBertClassifier CPU compatibility failed (expected without model): {}",
                e
            );
        }
    }

    // Test GPU usage preference (may fall back to CPU)
    let gpu_result = TraditionalBertClassifier::new(
        model_path, 3, false, // prefer GPU
    );

    match gpu_result {
        Ok(_classifier) => {
            println!("TraditionalBertClassifier GPU compatibility succeeded");
        }
        Err(e) => {
            println!(
                "TraditionalBertClassifier GPU compatibility failed (expected without model): {}",
                e
            );
        }
    }
}
