//! Integration tests for the current Qwen3 generative API surface.
//!
//! These tests validate public data structures without requiring model weights.

use candle_semantic_router::model_architectures::generative::{
    GuardGenerationResult, MultiAdapterClassificationResult, Qwen3GuardConfig,
};
use candle_semantic_router::model_architectures::prefix_cache::PrefixCacheConfig;

fn sample_classification_result() -> MultiAdapterClassificationResult {
    MultiAdapterClassificationResult {
        adapter_name: "category".to_string(),
        category: "biology".to_string(),
        confidence: 0.85,
        probabilities: vec![0.85, 0.10, 0.03, 0.02],
        all_categories: vec![
            "biology".to_string(),
            "chemistry".to_string(),
            "physics".to_string(),
            "math".to_string(),
        ],
    }
}

#[test]
fn test_multi_adapter_classification_result_api() {
    let result = sample_classification_result();

    assert_eq!(result.adapter_name, "category");
    assert_eq!(result.category, "biology");
    assert_eq!(result.probabilities.len(), result.all_categories.len());

    let sum: f32 = result.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probabilities should sum to 1.0, got {sum}"
    );

    let winning_index = result
        .all_categories
        .iter()
        .position(|label| label == &result.category)
        .expect("winning category should exist");
    assert!((result.confidence - result.probabilities[winning_index]).abs() < 1e-6);
}

#[test]
fn test_guard_generation_result_api() {
    let result = GuardGenerationResult {
        raw_output: "Reasoning: The prompt attempts to bypass safety rules.\nCategory: Jailbreak\nSeverity level: Unsafe".to_string(),
    };

    assert!(result.raw_output.contains("Category: Jailbreak"));
    assert!(result.raw_output.contains("Severity level: Unsafe"));
}

#[test]
fn test_qwen3_guard_config_api() {
    let default_config = Qwen3GuardConfig::default();
    assert_eq!(default_config.temperature, 0.0);
    assert_eq!(default_config.top_p, 0.95);
    assert_eq!(default_config.max_tokens, 512);
    assert!((default_config.repeat_penalty - 1.1).abs() < 1e-6);
    assert_eq!(default_config.repeat_last_n, 64);
    assert!(default_config.prefix_cache.enabled);
    assert!(!default_config.prefix_cache.verbose);

    let custom_config = Qwen3GuardConfig {
        temperature: 0.2,
        top_p: 0.8,
        max_tokens: 128,
        repeat_penalty: 1.0,
        repeat_last_n: 32,
        prefix_cache: PrefixCacheConfig {
            enabled: false,
            verbose: true,
        },
    };

    assert_eq!(custom_config.temperature, 0.2);
    assert_eq!(custom_config.top_p, 0.8);
    assert_eq!(custom_config.max_tokens, 128);
    assert_eq!(custom_config.repeat_last_n, 32);
    assert!(!custom_config.prefix_cache.enabled);
    assert!(custom_config.prefix_cache.verbose);
}

#[test]
fn test_probability_distribution_properties() {
    let result = sample_classification_result();

    for probability in &result.probabilities {
        assert!((0.0..=1.0).contains(probability));
    }

    let mut sorted = result.probabilities.clone();
    sorted.sort_by(|a, b| {
        b.partial_cmp(a)
            .expect("probabilities should be comparable")
    });
    assert_eq!(sorted[0], result.confidence);
}

#[test]
fn test_classification_result_sorting() {
    let result = MultiAdapterClassificationResult {
        adapter_name: "category".to_string(),
        category: "chemistry".to_string(),
        confidence: 0.45,
        probabilities: vec![0.15, 0.45, 0.25, 0.10, 0.05],
        all_categories: vec![
            "biology".to_string(),
            "chemistry".to_string(),
            "physics".to_string(),
            "math".to_string(),
            "other".to_string(),
        ],
    };

    let mut indexed: Vec<(usize, f32)> = result
        .probabilities
        .iter()
        .enumerate()
        .map(|(i, &probability)| (i, probability))
        .collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .expect("probabilities should be comparable")
    });

    assert_eq!(indexed[0].0, 1);
    assert!((indexed[0].1 - 0.45).abs() < 1e-6);
    assert_eq!(indexed[1].0, 2);
    assert!((indexed[1].1 - 0.25).abs() < 1e-6);
    assert_eq!(result.all_categories[indexed[0].0], result.category);
}
