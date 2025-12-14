// Accuracy Verification: Sequential vs Continuous Batch
//
// Verifies that continuous batch scheduler produces identical results to sequential processing
// This ensures batching doesn't affect classification accuracy
//
// Usage:
//   cargo run --release --bin verify_batch_accuracy

use candle_core::Device;
use candle_semantic_router::model_architectures::generative::{
    BatchSchedulerConfig, ContinuousBatchScheduler, Qwen3MultiLoRAClassifier,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::Arc;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn load_test_data() -> Vec<TestSample> {
    let data_path = "../bench/data/test_data.json";
    let contents = fs::read_to_string(data_path).expect("Failed to read test_data.json");
    serde_json::from_str(&contents).expect("Failed to parse test_data.json")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     Continuous Batch Accuracy Verification                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    let test_data = load_test_data();
    println!("📊 Loaded {} test samples\n", test_data.len());

    let base_model_path = "../models/Qwen3-0.6B";
    let adapter_path = "../models/qwen3_generative_classifier_r16_fixed";
    let device = Device::cuda_if_available(0)?;

    // =================================================================
    // PHASE 1: Sequential Classification (Baseline)
    // =================================================================
    println!("🔄 Phase 1: Sequential Classification (Baseline)");
    println!("{}", "=".repeat(70));

    let mut sequential_classifier = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    sequential_classifier.load_adapter("category", adapter_path)?;

    let mut sequential_results = Vec::new();
    let mut sequential_correct = 0;

    print!("Processing {} samples sequentially... ", test_data.len());
    for sample in &test_data {
        let result = sequential_classifier.classify_with_adapter(&sample.text, "category")?;

        // Map category name to ID by finding index in all_categories
        let predicted_id = result
            .all_categories
            .iter()
            .position(|cat| cat == &result.category)
            .unwrap_or(0);

        if predicted_id == sample.true_label_id {
            sequential_correct += 1;
        }

        sequential_results.push((
            sample.text.clone(),
            predicted_id,
            result.confidence,
            result.category.clone(),
        ));
    }
    println!("✓");

    let sequential_accuracy = (sequential_correct as f64 / test_data.len() as f64) * 100.0;
    println!("✅ Sequential Results:");
    println!(
        "   Accuracy: {:.1}% ({}/{})\n",
        sequential_accuracy,
        sequential_correct,
        test_data.len()
    );

    // =================================================================
    // PHASE 2: Continuous Batch Classification
    // =================================================================
    println!("🔄 Phase 2: Continuous Batch Classification");
    println!("{}", "=".repeat(70));

    let mut batch_classifier = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    batch_classifier.load_adapter("category", adapter_path)?;

    let batch_config = BatchSchedulerConfig {
        max_batch_size: 4,
        batch_timeout_ms: 5,
        queue_capacity: 1000,
        verbose: false,
    };

    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        batch_classifier,
        batch_config,
    ));

    let mut batch_results = Vec::new();
    let mut batch_correct = 0;

    print!(
        "Processing {} samples via continuous batch... ",
        test_data.len()
    );
    for sample in &test_data {
        let result = scheduler.classify(sample.text.clone(), "category".to_string())?;

        // Map category name to ID by finding index in all_categories
        let predicted_id = result
            .all_categories
            .iter()
            .position(|cat| cat == &result.category)
            .unwrap_or(0);

        if predicted_id == sample.true_label_id {
            batch_correct += 1;
        }

        batch_results.push((
            sample.text.clone(),
            predicted_id,
            result.confidence,
            result.category.clone(),
        ));
    }
    println!("✓");

    let batch_accuracy = (batch_correct as f64 / test_data.len() as f64) * 100.0;
    println!("✅ Continuous Batch Results:");
    println!(
        "   Accuracy: {:.1}% ({}/{})\n",
        batch_accuracy,
        batch_correct,
        test_data.len()
    );

    // =================================================================
    // PHASE 3: Compare Results
    // =================================================================
    println!("🔍 Phase 3: Comparing Predictions");
    println!("{}", "=".repeat(70));

    let mut mismatches = 0;
    let mut confidence_diffs = Vec::new();

    for i in 0..test_data.len() {
        let (text, seq_pred, seq_conf, seq_cat) = &sequential_results[i];
        let (_, batch_pred, batch_conf, batch_cat) = &batch_results[i];

        if seq_pred != batch_pred {
            mismatches += 1;
            println!("❌ Mismatch at sample {}:", i);
            println!("   Text: {}", text);
            println!(
                "   Sequential: class {} \"{}\" (conf: {:.4})",
                seq_pred, seq_cat, seq_conf
            );
            println!(
                "   Batch:      class {} \"{}\" (conf: {:.4})",
                batch_pred, batch_cat, batch_conf
            );
            println!();
        } else {
            let conf_diff = (seq_conf - batch_conf).abs();
            confidence_diffs.push(conf_diff);
        }
    }

    // =================================================================
    // FINAL REPORT
    // =================================================================
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                     VERIFICATION RESULTS                          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!("📊 Accuracy Comparison:");
    println!(
        "   Sequential:      {:.2}% ({}/{})",
        sequential_accuracy,
        sequential_correct,
        test_data.len()
    );
    println!(
        "   Continuous Batch: {:.2}% ({}/{})",
        batch_accuracy,
        batch_correct,
        test_data.len()
    );
    println!(
        "   Accuracy Delta:   {:.2}%",
        (batch_accuracy - sequential_accuracy).abs()
    );
    println!();

    println!("🔍 Prediction Comparison:");
    println!("   Total samples:    {}", test_data.len());
    println!(
        "   Matching:         {} ({:.1}%)",
        test_data.len() - mismatches,
        ((test_data.len() - mismatches) as f64 / test_data.len() as f64) * 100.0
    );
    println!(
        "   Mismatches:       {} ({:.1}%)",
        mismatches,
        (mismatches as f64 / test_data.len() as f64) * 100.0
    );
    println!();

    if !confidence_diffs.is_empty() {
        let avg_conf_diff: f32 =
            confidence_diffs.iter().sum::<f32>() / confidence_diffs.len() as f32;
        let max_conf_diff = confidence_diffs.iter().cloned().fold(0.0f32, f32::max);

        println!("📈 Confidence Score Differences (for matching predictions):");
        println!("   Average:  {:.6}", avg_conf_diff);
        println!("   Maximum:  {:.6}", max_conf_diff);
        println!();
    }

    // Verdict
    if mismatches == 0 {
        println!("✅ ✅ ✅  VERIFICATION PASSED  ✅ ✅ ✅");
        println!("\n🎉 Continuous batch scheduler produces IDENTICAL predictions!");
        println!(
            "   All {} predictions match sequential processing.",
            test_data.len()
        );
        println!(
            "   Classification accuracy is preserved at {:.2}%\n",
            sequential_accuracy
        );
        Ok(())
    } else {
        println!("⚠️  ⚠️  ⚠️   VERIFICATION FAILED  ⚠️  ⚠️  ⚠️");
        println!(
            "\n❌ Found {} prediction mismatches out of {} samples",
            mismatches,
            test_data.len()
        );
        println!("   Continuous batching is affecting classification results!\n");
        Err("Accuracy verification failed".into())
    }
}
