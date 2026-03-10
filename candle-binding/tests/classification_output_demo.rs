//! Demonstration of current generative classification results with realistic scenarios.

use candle_semantic_router::model_architectures::generative::MultiAdapterClassificationResult;

fn build_result(
    adapter_name: &str,
    category: &str,
    all_categories: &[&str],
    probabilities: Vec<f32>,
) -> MultiAdapterClassificationResult {
    let category_index = all_categories
        .iter()
        .position(|label| *label == category)
        .expect("category should exist in the label mapping");

    MultiAdapterClassificationResult {
        adapter_name: adapter_name.to_string(),
        category: category.to_string(),
        confidence: probabilities[category_index],
        probabilities,
        all_categories: all_categories
            .iter()
            .map(|label| (*label).to_string())
            .collect(),
    }
}

fn winning_index(result: &MultiAdapterClassificationResult) -> usize {
    result
        .all_categories
        .iter()
        .position(|label| label == &result.category)
        .expect("winning category should exist")
}

fn assert_valid_distribution(result: &MultiAdapterClassificationResult) {
    let sum: f32 = result.probabilities.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "Probabilities must sum to 1.0, got {sum}"
    );
    assert_eq!(result.probabilities.len(), result.all_categories.len());
    assert!((result.confidence - result.probabilities[winning_index(result)]).abs() < 1e-6);
}

#[test]
fn test_classification_output_scenarios() {
    println!("\n=== Classification Output Demo ===\n");

    let categories = [
        "biology",
        "chemistry",
        "physics",
        "math",
        "computer science",
        "other",
    ];

    let bio_result = build_result(
        "category",
        "biology",
        &categories,
        vec![0.92, 0.03, 0.02, 0.01, 0.01, 0.01],
    );
    println!("📌 Scenario 1: \"What is photosynthesis?\"");
    println!(
        "   Adapter: {} | Category: {} | Confidence: {:.1}%",
        bio_result.adapter_name,
        bio_result.category,
        bio_result.confidence * 100.0
    );
    assert_valid_distribution(&bio_result);

    let math_result = build_result(
        "category",
        "math",
        &categories,
        vec![0.02, 0.02, 0.03, 0.88, 0.04, 0.01],
    );
    println!("📌 Scenario 2: \"Calculate the derivative of x^2\"");
    println!(
        "   Category: {} | Confidence: {:.1}%",
        math_result.category,
        math_result.confidence * 100.0
    );
    assert_valid_distribution(&math_result);

    let categories_with_engineering = [
        "biology",
        "chemistry",
        "physics",
        "math",
        "computer science",
        "engineering",
    ];
    let ambiguous_result = build_result(
        "category",
        "physics",
        &categories_with_engineering,
        vec![0.05, 0.08, 0.48, 0.15, 0.10, 0.14],
    );
    println!("📌 Scenario 3: \"What causes motion in objects?\"");
    println!(
        "   Category: {} | Confidence: {:.1}% | Top-2 overlap visible",
        ambiguous_result.category,
        ambiguous_result.confidence * 100.0
    );
    assert_valid_distribution(&ambiguous_result);

    let multi_result = build_result(
        "category",
        "chemistry",
        &categories,
        vec![0.35, 0.52, 0.05, 0.03, 0.03, 0.02],
    );
    println!("📌 Scenario 4: \"How do enzymes catalyze reactions?\"");
    println!(
        "   Category: {} | Confidence: {:.1}% | Biology remains a strong secondary signal",
        multi_result.category,
        multi_result.confidence * 100.0
    );
    assert_valid_distribution(&multi_result);

    let batch_results = vec![
        build_result(
            "category",
            "physics",
            &categories,
            vec![0.02, 0.03, 0.91, 0.02, 0.01, 0.01],
        ),
        build_result(
            "category",
            "physics",
            &categories,
            vec![0.03, 0.04, 0.89, 0.02, 0.01, 0.01],
        ),
        build_result(
            "category",
            "physics",
            &categories,
            vec![0.01, 0.02, 0.93, 0.02, 0.01, 0.01],
        ),
    ];
    let average_confidence: f32 = batch_results
        .iter()
        .map(|result| result.confidence)
        .sum::<f32>()
        / batch_results.len() as f32;
    assert!(average_confidence > 0.9 - 0.02);
    println!(
        "📌 Scenario 5: physics batch average confidence = {:.1}%",
        average_confidence * 100.0
    );
}

#[test]
fn test_entropy_and_uncertainty() {
    fn calculate_entropy(probabilities: &[f32]) -> f32 {
        probabilities
            .iter()
            .filter(|&&probability| probability > 0.0)
            .map(|&probability| -probability * probability.log2())
            .sum()
    }

    let high_certainty = vec![0.95, 0.02, 0.01, 0.01, 0.01];
    let medium_certainty = vec![0.50, 0.30, 0.10, 0.05, 0.05];
    let low_certainty = vec![0.20, 0.20, 0.20, 0.20, 0.20];

    let entropy_high = calculate_entropy(&high_certainty);
    let entropy_medium = calculate_entropy(&medium_certainty);
    let entropy_low = calculate_entropy(&low_certainty);

    assert!(entropy_high < entropy_medium);
    assert!(entropy_medium < entropy_low);
}

#[test]
fn test_realistic_mmlu_pro_distribution() {
    let categories = [
        "biology",
        "business",
        "chemistry",
        "computer science",
        "economics",
        "engineering",
        "health",
        "history",
        "law",
        "math",
        "other",
        "philosophy",
        "physics",
        "psychology",
    ];

    let result = build_result(
        "mmlu-pro",
        "biology",
        &categories,
        vec![
            0.78, 0.02, 0.08, 0.01, 0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00,
        ],
    );

    assert_valid_distribution(&result);
    let top_related_categories: Vec<&str> = result
        .all_categories
        .iter()
        .zip(result.probabilities.iter())
        .filter(|(_, probability)| **probability >= 0.05)
        .map(|(category, _)| category.as_str())
        .collect();

    assert_eq!(result.adapter_name, "mmlu-pro");
    assert_eq!(result.category, "biology");
    assert_eq!(
        top_related_categories,
        vec!["biology", "chemistry", "health"]
    );
}
