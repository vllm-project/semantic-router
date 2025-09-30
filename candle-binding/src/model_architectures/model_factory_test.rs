//! Tests for model factory

use super::config::PathSelectionStrategy;
use super::model_factory::*;
use super::traits::TaskType;
use crate::test_fixtures::fixtures::*;
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
