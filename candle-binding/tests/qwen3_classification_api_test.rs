//! Integration test for Qwen3 classification API
//!
//! This tests the API surface without requiring an actual model.

use candle_semantic_router::model_architectures::generative::qwen3_causal::{
    ClassificationResult, GenerationConfig, GenerationResult,
};

#[test]
fn test_classification_result_api() {
    // Test that ClassificationResult can be created and accessed
    let result = ClassificationResult {
        class: 0,
        category: "biology".to_string(),
        confidence: 0.85,
        probabilities: vec![0.85, 0.10, 0.03, 0.02],
    };

    assert_eq!(result.class, 0);
    assert_eq!(result.category, "biology");
    assert!((result.confidence - 0.85).abs() < 1e-6);
    assert_eq!(result.probabilities.len(), 4);

    // Probabilities should sum to ~1.0
    let sum: f32 = result.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probabilities should sum to 1.0, got {}",
        sum
    );

    // Confidence should match max probability
    let max_prob = result
        .probabilities
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    assert!((result.confidence - max_prob).abs() < 1e-6);
}

#[test]
fn test_generation_result_api() {
    // Test GenerationResult structure
    let result = GenerationResult {
        text: "biology".to_string(),
        token_ids: vec![1, 2, 3, 4, 5],
        num_generated: 1,
        tokens_per_second: 100.0,
    };

    assert_eq!(result.text, "biology");
    assert_eq!(result.token_ids.len(), 5);
    assert_eq!(result.num_generated, 1);
    assert!(result.tokens_per_second > 0.0);
}

#[test]
fn test_generation_config_api() {
    // Test default configuration
    let default_config = GenerationConfig::default();
    assert_eq!(default_config.max_new_tokens, 100);
    assert_eq!(default_config.temperature, Some(0.7));
    assert_eq!(default_config.top_p, None);
    assert!((default_config.repeat_penalty - 1.1).abs() < 1e-6);

    // Test custom configuration for classification (greedy)
    let greedy_config = GenerationConfig {
        max_new_tokens: 20,
        temperature: Some(0.0),
        top_p: None,
        repeat_penalty: 1.0,
        repeat_last_n: 64,
        seed: 42,
    };

    assert_eq!(greedy_config.max_new_tokens, 20);
    assert_eq!(greedy_config.temperature, Some(0.0));
    assert!((greedy_config.repeat_penalty - 1.0).abs() < 1e-6);
}

#[test]
fn test_probability_distribution_properties() {
    // Test that a valid probability distribution has correct properties
    let probs = vec![0.7, 0.2, 0.08, 0.02];

    // Sum should be 1.0
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5);

    // All values should be between 0 and 1
    for &p in &probs {
        assert!(p >= 0.0 && p <= 1.0);
    }

    // Should be sorted descending (for this test case)
    for i in 1..probs.len() {
        assert!(probs[i - 1] >= probs[i]);
    }

    // Create result with this distribution
    let result = ClassificationResult {
        class: 0,
        category: "biology".to_string(),
        confidence: probs[0],
        probabilities: probs.clone(),
    };

    assert_eq!(result.confidence, probs[0]);
}

#[test]
fn test_classification_result_sorting() {
    // Test sorting probabilities to find top-k
    let probs = vec![0.15, 0.45, 0.25, 0.10, 0.05];

    let mut indexed: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Top category should be index 1 with probability 0.45
    assert_eq!(indexed[0].0, 1);
    assert!((indexed[0].1 - 0.45).abs() < 1e-6);

    // Second should be index 2 with 0.25
    assert_eq!(indexed[1].0, 2);
    assert!((indexed[1].1 - 0.25).abs() < 1e-6);
}
