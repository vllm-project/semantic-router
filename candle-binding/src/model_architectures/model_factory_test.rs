//! Tests for model factory

use super::config::PathSelectionStrategy;
use super::model_factory::*;
use super::traits::TaskType;
use crate::test_fixtures::fixtures::*;
use crate::test_fixtures::test_utils::get_first_available_model;
use candle_core::Device;
use rstest::*;
use std::collections::HashMap;

/// Test ModelFactory creation and basic operations
#[rstest]
fn test_model_factory_model_factory_creation() {
    let device = Device::Cpu;
    let _factory = ModelFactory::new(device);

    // Test that factory is created successfully
    println!("ModelFactory creation test passed");
}

/// Test ModelFactory configuration with different strategies and real models
#[rstest]
#[case(PathSelectionStrategy::Automatic, "automatic")]
#[case(PathSelectionStrategy::AlwaysLoRA, "always_lora")]
#[case(PathSelectionStrategy::AlwaysTraditional, "always_traditional")]
#[case(PathSelectionStrategy::PerformanceBased, "performance_based")]
fn test_model_factory_model_factory_with_strategies(
    #[case] _strategy: PathSelectionStrategy,
    #[case] strategy_name: &str,
    traditional_model_path: String,
    lora_model_path: String,
) {
    use std::path::Path;
    let device = Device::Cpu;
    let mut factory = ModelFactory::new(device);

    // Test registering models with real model paths if available
    let traditional_path = if Path::new(&traditional_model_path).exists() {
        println!(
            "Using real traditional model for factory test: {}",
            traditional_model_path
        );
        traditional_model_path
    } else {
        println!("Real traditional model not found, using mock path for factory test");
        "nonexistent-model".to_string()
    };

    let traditional_result =
        factory.register_traditional_model("test_traditional", traditional_path, 3, true);
    // Expected to fail due to nonexistent model, but interface should work
    assert!(traditional_result.is_err());

    let mut task_configs = HashMap::new();
    task_configs.insert(TaskType::Intent, 3);

    let lora_path = if Path::new(&lora_model_path).exists() {
        println!(
            "Using real LoRA model for factory test: {}",
            lora_model_path
        );
        lora_model_path.clone()
    } else {
        println!("Real LoRA model not found, using mock path for factory test");
        "nonexistent-model".to_string()
    };

    let lora_result = factory.register_lora_model(
        "test_lora",
        lora_path.clone(),
        lora_path,
        task_configs,
        true,
    );
    // Expected to fail due to nonexistent model, but interface should work
    assert!(lora_result.is_err());

    println!("ModelFactory strategy test passed for {}", strategy_name);
}

/// The Gemma tokenizer must clamp long inputs to max_position_embeddings so a single
/// oversized text neither fails on its own nor takes down a whole batch (issue #3388).
#[rstest]
#[serial_test::serial(gemma_model)]
fn test_model_factory_gemma_tokenizer_truncates_to_max_position_embeddings() {
    let Some(model_path) =
        get_first_available_model(&["mom-embedding-flash", GEMMA_EMBEDDING_300M])
    else {
        println!("Gemma embedding model not found, skipping tokenizer truncation test");
        return;
    };

    let mut factory = ModelFactory::new(test_device());
    factory
        .register_gemma_embedding_model(&model_path)
        .expect("Failed to register Gemma embedding model");

    let tokenizer = factory
        .get_gemma_tokenizer()
        .expect("Gemma tokenizer must be registered");
    let max_len = factory
        .get_gemma_model()
        .expect("Gemma model must be registered")
        .config()
        .max_position_embeddings;

    // Well past 2048 tokens in both Latin and CJK scripts
    let long_latin = "hello world ".repeat(4000);
    let long_cjk = "你好世界".repeat(2000);

    for text in [long_latin.as_str(), long_cjk.as_str()] {
        let encoding = tokenizer.encode(text, true).expect("Tokenization failed");
        assert_eq!(
            encoding.get_ids().len(),
            max_len,
            "encoding must be clamped to max_position_embeddings"
        );
        assert_eq!(encoding.get_attention_mask().len(), max_len);
    }

    let batch = tokenizer
        .encode_batch(vec!["short text", long_latin.as_str()], true)
        .expect("Batch tokenization failed");
    assert!(batch[0].get_ids().len() < max_len);
    assert_eq!(batch[1].get_ids().len(), max_len);
}
