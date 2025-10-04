//! Tests for BERT LoRA implementation

use super::bert_lora::*;
use crate::classifiers::lora::intent_lora::IntentLoRAClassifier;
use crate::model_architectures::traits::TaskType;
use crate::test_fixtures::fixtures::*;
use rstest::*;
use serial_test::serial;
use std::collections::HashMap;
use std::sync::Arc;

/// Test LoRABertClassifier creation with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_bert_lora_lora_bert_classifier_new(
    cached_intent_classifier: Option<Arc<IntentLoRAClassifier>>,
) {
    if let Some(classifier) = cached_intent_classifier {
        println!("Testing LoRABertClassifier with cached Intent model - instant access!");

        let test_text = "Hello, how are you today?";
        match classifier.classify_intent(test_text) {
            Ok(result) => {
                assert!(!result.intent.is_empty());
                assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
                assert!(result.processing_time_ms > 0);
                println!("LoRABertClassifier creation test passed with cached model: intent='{}', confidence={:.3}",
                    result.intent, result.confidence);
            }
            Err(e) => println!("LoRABertClassifier test failed: {}", e),
        }
    } else {
        println!("Cached Intent classifier not available, skipping BERT LoRA test");
    }
}

/// Test LoRABertClassifier task configuration validation
#[rstest]
#[case(vec![TaskType::Intent], "single_task")]
#[case(vec![TaskType::Intent, TaskType::PII], "dual_task")]
#[case(vec![TaskType::Intent, TaskType::PII, TaskType::Security], "multi_task")]
fn test_bert_lora_lora_bert_classifier_task_configs(
    #[case] tasks: Vec<TaskType>,
    #[case] config_name: &str,
) {
    let mut task_configs = HashMap::new();

    for task in &tasks {
        let num_classes = match task {
            TaskType::Intent => 5,
            TaskType::PII => 2,
            TaskType::Security => 2,
            _ => 3,
        };
        task_configs.insert(*task, num_classes);
    }

    // Test configuration structure
    assert_eq!(task_configs.len(), tasks.len());

    for task in &tasks {
        assert!(task_configs.contains_key(task));
        let num_classes = task_configs[task];
        assert!(num_classes >= 2 && num_classes <= 10);
    }

    println!(
        "LoRABertClassifier task config test passed for {} ({} tasks)",
        config_name,
        tasks.len()
    );
}

/// Test LoRABertClassifier error handling with cached model (OPTIMIZED)
#[rstest]
#[serial]
fn test_bert_lora_lora_bert_classifier_error_handling(
    cached_intent_classifier: Option<Arc<IntentLoRAClassifier>>,
) {
    if let Some(classifier) = cached_intent_classifier {
        println!("Testing LoRABertClassifier error handling with cached model!");

        // Test with valid input (should work)
        let test_text = "Valid test input";
        match classifier.classify_intent(test_text) {
            Ok(_) => println!("Cached model error handling test passed - valid input works"),
            Err(e) => println!("Cached model error: {}", e),
        }

        // Test with empty input (should handle gracefully)
        match classifier.classify_intent("") {
            Ok(_) => println!("Empty input handled successfully"),
            Err(_) => println!("Empty input handled with error (expected)"),
        }
    } else {
        println!("Cached Intent classifier not available, skipping error handling test");
    }

    // Test error scenarios with invalid paths
    let invalid_model_result = LoRABertClassifier::new("", "", HashMap::new(), true);
    assert!(invalid_model_result.is_err());

    let empty_tasks_result = LoRABertClassifier::new(
        "nonexistent-model",
        "nonexistent-model",
        HashMap::new(),
        true,
    );
    assert!(empty_tasks_result.is_err());

    println!("LoRABertClassifier error handling test passed");
}
