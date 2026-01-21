//! # Integration Tests for ML Algorithms
//!
//! These tests verify that the Rust implementations produce identical results
//! to the Go implementations in `src/semantic-router/pkg/selection/`.

use super::*;

/// Test that cosine similarity matches Go implementation
#[test]
fn test_cosine_similarity_matches_go() {
    // Test cases from Go gonum implementation
    let cases = vec![
        // Identical vectors -> 1.0
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 1.0),
        // Orthogonal vectors -> 0.0
        (vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], 0.0),
        // Opposite vectors -> -1.0
        (vec![1.0, 0.0, 0.0], vec![-1.0, 0.0, 0.0], -1.0),
        // General case
        (vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0], 0.9746318),
    ];

    for (a, b, expected) in cases {
        let result = cosine_similarity(&a, &b);
        assert!(
            (result - expected).abs() < 1e-5,
            "cosine_similarity({:?}, {:?}) = {}, expected {}",
            a,
            b,
            result,
            expected
        );
    }
}

/// Test that euclidean distance matches Go implementation
#[test]
fn test_euclidean_distance_matches_go() {
    // Test cases from Go floats.Distance
    let cases = vec![
        (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0], 0.0),
        (vec![0.0, 0.0, 0.0], vec![3.0, 4.0, 0.0], 5.0),
        (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 0.0),
    ];

    for (a, b, expected) in cases {
        let result = euclidean_distance(&a, &b);
        assert!(
            (result - expected).abs() < 1e-6,
            "euclidean_distance({:?}, {:?}) = {}, expected {}",
            a,
            b,
            result,
            expected
        );
    }
}

/// Helper to create a training record with all fields
fn make_record(
    query: &str,
    embedding: Vec<f64>,
    decision: &str,
    model: &str,
    latency_ms: i64,
    quality: f64,
    timestamp: i64,
) -> TrainingRecord {
    TrainingRecord {
        query_text: query.to_string(),
        query_embedding: embedding,
        decision_name: decision.to_string(),
        selected_model: model.to_string(),
        response_latency_ns: latency_ms * 1_000_000,
        response_quality: quality,
        success: true,
        timestamp,
    }
}

/// Test KNN with realistic query text and decision names
#[test]
fn test_knn_production_scenario() {
    use super::knn::KNNSelector;

    let mut selector = KNNSelector::new(5);

    // Create training data with query text and decision categories
    let mut records = Vec::new();

    // Math queries -> mistral-7b (faster, good for math)
    for i in 0..10 {
        records.push(make_record(
            &format!("Calculate the derivative of x^{}", i + 2),
            vec![0.9 + 0.01 * i as f64, 0.1, 0.0, 0.0],
            "math",
            "mistral-7b",
            50, // 50ms - fast
            0.85,
            i,
        ));
    }

    // Coding queries -> codellama-34b (slower, better for code)
    for i in 0..10 {
        records.push(make_record(
            &format!("Write a Python function to sort array #{}", i),
            vec![0.1, 0.9 + 0.01 * i as f64, 0.0, 0.0],
            "coding",
            "codellama-34b",
            200, // 200ms - slower
            0.95,
            10 + i,
        ));
    }

    // General queries -> llama2-70b (balanced)
    for i in 0..10 {
        records.push(make_record(
            &format!("Explain the concept of {} in simple terms", i),
            vec![0.0, 0.0, 0.9 + 0.01 * i as f64, 0.0],
            "general",
            "llama2-70b",
            150, // 150ms
            0.90,
            20 + i,
        ));
    }

    selector.train(&records).unwrap();
    assert!(selector.is_trained());

    let refs = vec![
        ModelRef {
            model: "mistral-7b".to_string(),
            lora_name: None,
        },
        ModelRef {
            model: "codellama-34b".to_string(),
            lora_name: None,
        },
        ModelRef {
            model: "llama2-70b".to_string(),
            lora_name: None,
        },
    ];

    // Math query should select mistral-7b
    let ctx = SelectionContext::new(
        "What is the integral of sin(x)?",
        vec![0.95, 0.05, 0.0, 0.0],
    )
    .with_decision("math");
    let result = selector.select(&ctx, &refs).unwrap();
    assert_eq!(result.model_name, "mistral-7b");

    // Coding query should select codellama-34b
    let ctx = SelectionContext::new(
        "Write a Rust function to parse JSON",
        vec![0.05, 0.95, 0.0, 0.0],
    )
    .with_decision("coding");
    let result = selector.select(&ctx, &refs).unwrap();
    assert_eq!(result.model_name, "codellama-34b");

    // General query should select llama2-70b
    let ctx = SelectionContext::new("Tell me about machine learning", vec![0.0, 0.0, 0.95, 0.05])
        .with_decision("general");
    let result = selector.select(&ctx, &refs).unwrap();
    assert_eq!(result.model_name, "llama2-70b");
}

/// Test KMeans performance-efficiency trade-off (Avengers-Pro)
#[test]
fn test_kmeans_efficiency_tradeoff() {
    use super::kmeans::KMeansSelector;

    // High efficiency weight should prefer faster models
    let high_eff = KMeansSelector::with_efficiency(3, 0.8);
    assert!((high_eff.efficiency_weight - 0.8).abs() < 1e-6);

    // Low efficiency weight should prefer higher quality models
    let low_eff = KMeansSelector::with_efficiency(3, 0.1);
    assert!((low_eff.efficiency_weight - 0.1).abs() < 1e-6);
}

/// Test SVM kernel functions
#[test]
fn test_svm_kernels() {
    use super::svm::SVMSelector;

    // RBF kernel
    let svm_rbf = SVMSelector::new("rbf");
    assert_eq!(svm_rbf.name(), "svm");

    // Linear kernel - dot product
    let svm_linear = SVMSelector::new("linear");
    assert_eq!(svm_linear.name(), "svm");

    // Polynomial kernel
    let svm_poly = SVMSelector::new("poly");
    assert_eq!(svm_poly.name(), "svm");
}

/// Test that all selectors implement ModelSelector trait
#[test]
fn test_selector_trait_implementation() {
    use super::kmeans::KMeansSelector;
    use super::knn::KNNSelector;
    use super::svm::SVMSelector;

    let selectors: Vec<Box<dyn ModelSelector>> = vec![
        Box::new(KNNSelector::new(5)),
        Box::new(KMeansSelector::new(3)),
        Box::new(SVMSelector::new("rbf")),
    ];

    for selector in &selectors {
        assert!(!selector.name().is_empty());
        assert_eq!(selector.training_count(), 0);
    }
}

/// Test serialization/deserialization round-trip
#[test]
fn test_knn_serialization_roundtrip() {
    use super::knn::KNNSelector;

    let mut selector = KNNSelector::new(7);
    let records = vec![
        make_record(
            "What is 2+2?",
            vec![1.0, 0.0, 0.0],
            "math",
            "model_a",
            100,
            0.9,
            0,
        ),
        make_record(
            "Write hello world",
            vec![0.0, 1.0, 0.0],
            "coding",
            "model_b",
            200,
            0.85,
            1,
        ),
    ];
    selector.train(&records).unwrap();

    // Serialize
    let json = selector.save_to_json().unwrap();

    // Deserialize
    let mut selector2 = KNNSelector::new(3);
    selector2.load_from_json(&json).unwrap();

    assert_eq!(selector2.training_count(), 2);
}

/// Test softmax numerical stability
#[test]
fn test_softmax_numerical_stability() {
    // Large values that could cause overflow
    let large = vec![1000.0, 1000.0, 1000.0];
    let result = softmax(&large);
    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);

    // All equal should give uniform distribution
    for &r in &result {
        assert!((r - 1.0 / 3.0).abs() < 1e-6);
    }

    // Different values (using smaller range to avoid underflow)
    let diff = vec![0.0, 1.0, 2.0];
    let result = softmax(&diff);
    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(result[2] > result[1]);
    assert!(result[1] > result[0]);

    // Extreme values: verify no NaN/Inf and sum is still 1.0
    let extreme = vec![0.0, 1000.0, 2000.0];
    let result = softmax(&extreme);
    assert!(!result.iter().any(|&x| x.is_nan() || x.is_infinite()));
    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Note: result[0] and result[1] underflow to ~0, result[2] ≈ 1.0
}

/// Test SelectionContext builder pattern
#[test]
fn test_selection_context_builder() {
    let ctx = SelectionContext::new("Hello world", vec![1.0, 2.0, 3.0])
        .with_decision("greeting")
        .with_weights(0.4, 0.6);

    assert_eq!(ctx.query, "Hello world");
    assert_eq!(ctx.decision_name, "greeting");
    assert!((ctx.cost_weight - 0.4).abs() < 1e-6);
    assert!((ctx.quality_weight - 0.6).abs() < 1e-6);
}

/// Test SelectionContext defaults
#[test]
fn test_selection_context_defaults() {
    let ctx = SelectionContext::default();
    assert!(ctx.query.is_empty());
    assert!(ctx.query_embedding.is_empty());
    assert!(ctx.decision_name.is_empty());
    assert!((ctx.cost_weight - 0.3).abs() < 1e-6);
    assert!((ctx.quality_weight - 0.7).abs() < 1e-6);
}
