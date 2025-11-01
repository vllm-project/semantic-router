// Pure Batched Inference Benchmark (No Scheduler)
//
// Tests TRUE batched inference at Candle/model level only
// Compares:
// 1. Mutex baseline: single requests (classify_with_adapter)
// 2. Direct batching: manual batching (classify_batch_with_adapter)
//
// No continuous batch scheduler - just raw model-level batching
//
// Usage:
//   cargo run --release --bin pure_batch_inference_bench

use candle_core::Device;
use candle_semantic_router::model_architectures::generative::Qwen3MultiLoRAClassifier;
use serde::{Deserialize, Serialize};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn load_test_data() -> Vec<TestSample> {
    let data_path = "../bench/data/test_data.json";
    let contents = std::fs::read_to_string(data_path).expect("Failed to read test_data.json");
    serde_json::from_str(&contents).expect("Failed to parse test_data.json")
}

static MUTEX_CLASSIFIER: OnceLock<Mutex<Qwen3MultiLoRAClassifier>> = OnceLock::new();

fn initialize_classifier() -> Result<(), Box<dyn std::error::Error>> {
    let base_model_path = "../models/Qwen3-0.6B";
    let adapter_path = "../models/qwen3_generative_classifier_r16_fixed";
    let device = Device::cuda_if_available(0)?;

    println!("🔧 Initializing classifier...");
    println!("  Device: {:?}", device);
    let mut classifier = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    classifier.load_adapter("category", adapter_path)?;

    MUTEX_CLASSIFIER
        .set(Mutex::new(classifier))
        .map_err(|_| "Failed to initialize classifier")?;
    println!("✅ Classifier ready\n");

    Ok(())
}

fn benchmark_sequential(
    test_data: &[TestSample],
    num_iterations: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!(" SEQUENTIAL: Single requests via classify_with_adapter()");
    println!("{}", "=".repeat(70));

    let mut total_time = 0.0;
    let mut latencies = Vec::new();

    for i in 0..num_iterations {
        let sample = &test_data[i % test_data.len()];

        let start = Instant::now();
        {
            let mut classifier = MUTEX_CLASSIFIER.get().unwrap().lock().unwrap();
            let _ = classifier.classify_with_adapter(&sample.text, "category")?;
        }
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        latencies.push(elapsed);
        total_time += elapsed;

        if (i + 1) % 10 == 0 {
            print!("\r  Processed: {}/{} requests", i + 1, num_iterations);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();

    let avg_latency = total_time / num_iterations as f64;
    let qps = 1000.0 / avg_latency;

    println!("\n📊 Sequential Results:");
    println!("  Total requests: {}", num_iterations);
    println!("  Total time: {:.2}s", total_time / 1000.0);
    println!("  Avg latency: {:.2}ms per request", avg_latency);
    println!("  Throughput: {:.1} QPS", qps);

    Ok((avg_latency, qps))
}

fn benchmark_batched(
    test_data: &[TestSample],
    num_iterations: usize,
    batch_size: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!(
        " BATCHED: classify_batch_with_adapter() with batch_size={}",
        batch_size
    );
    println!("{}", "=".repeat(70));

    let num_batches = num_iterations / batch_size;
    let mut batch_times = Vec::new();
    let mut per_request_times = Vec::new();

    for batch_idx in 0..num_batches {
        // Prepare batch of texts
        let mut texts = Vec::new();
        for i in 0..batch_size {
            let idx = (batch_idx * batch_size + i) % test_data.len();
            texts.push(test_data[idx].text.clone());
        }

        // Measure batched inference
        let start = Instant::now();
        {
            let mut classifier = MUTEX_CLASSIFIER.get().unwrap().lock().unwrap();
            let _ = classifier.classify_batch_with_adapter(&texts, "category")?;
        }
        let batch_elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let per_request = batch_elapsed / batch_size as f64;

        batch_times.push(batch_elapsed);
        per_request_times.push(per_request);

        if (batch_idx + 1) % 10 == 0 {
            print!(
                "\r  Processed: {}/{} batches ({} requests)",
                batch_idx + 1,
                num_batches,
                (batch_idx + 1) * batch_size
            );
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }
    }

    println!();

    let total_requests = num_batches * batch_size;
    let total_time: f64 = batch_times.iter().sum();
    let avg_batch_time: f64 = batch_times.iter().sum::<f64>() / batch_times.len() as f64;
    let avg_per_request = avg_batch_time / batch_size as f64;
    let qps = (total_requests as f64) / (total_time / 1000.0);

    println!("\n📊 Batched Results:");
    println!("  Total batches: {}", num_batches);
    println!("  Total requests: {}", total_requests);
    println!("  Total time: {:.2}s", total_time / 1000.0);
    println!(
        "  Avg batch time: {:.2}ms for {} requests",
        avg_batch_time, batch_size
    );
    println!("  Avg per-request: {:.2}ms (within batch)", avg_per_request);
    println!("  Throughput: {:.1} QPS", qps);

    Ok((avg_batch_time, avg_per_request, qps))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║     Pure Batched Inference Benchmark (Model-Level Only)          ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Load test data
    println!("\n📊 Loading test data...");
    let test_data = load_test_data();
    println!("✅ Loaded {} test samples\n", test_data.len());

    // Initialize classifier
    initialize_classifier()?;

    let num_iterations = 200; // Total requests to process

    // Test 1: Sequential baseline
    let (seq_latency, seq_qps) = benchmark_sequential(&test_data, num_iterations)?;

    // Test 2: Batched with batch_size=2
    let (batch2_time, batch2_per_req, batch2_qps) =
        benchmark_batched(&test_data, num_iterations, 2)?;

    // Test 3: Batched with batch_size=4
    let (batch4_time, batch4_per_req, batch4_qps) =
        benchmark_batched(&test_data, num_iterations, 4)?;

    // Test 4: Batched with batch_size=8
    let (batch8_time, batch8_per_req, batch8_qps) =
        benchmark_batched(&test_data, num_iterations, 8)?;

    // Summary comparison
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    COMPARISON SUMMARY                             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<20} {:<15} {:<15} {:<15} {:<15}",
        "Method", "Latency(ms)", "Per-Req(ms)", "QPS", "Speedup"
    );
    println!("{}", "-".repeat(80));

    println!(
        "{:<20} {:<15.2} {:<15.2} {:<15.1} {:<15}",
        "Sequential", seq_latency, seq_latency, seq_qps, "1.00×"
    );

    println!(
        "{:<20} {:<15.2} {:<15.2} {:<15.1} {:<15.2}×",
        "Batched (size=2)",
        batch2_time,
        batch2_per_req,
        batch2_qps,
        batch2_qps / seq_qps
    );

    println!(
        "{:<20} {:<15.2} {:<15.2} {:<15.1} {:<15.2}×",
        "Batched (size=4)",
        batch4_time,
        batch4_per_req,
        batch4_qps,
        batch4_qps / seq_qps
    );

    println!(
        "{:<20} {:<15.2} {:<15.2} {:<15.1} {:<15.2}×",
        "Batched (size=8)",
        batch8_time,
        batch8_per_req,
        batch8_qps,
        batch8_qps / seq_qps
    );

    println!("\n{}", "=".repeat(80));

    println!("\n📈 KEY FINDINGS:\n");

    println!("Batch Size 2:");
    println!(
        "  • Batch time: {:.2}ms (vs {:.2}ms × 2 = {:.2}ms sequential)",
        batch2_time,
        seq_latency,
        seq_latency * 2.0
    );
    println!(
        "  • Per-request speedup: {:.2}× faster",
        seq_latency / batch2_per_req
    );
    println!(
        "  • Throughput: {:.1} → {:.1} QPS ({:+.1}%)\n",
        seq_qps,
        batch2_qps,
        (batch2_qps / seq_qps - 1.0) * 100.0
    );

    println!("Batch Size 4:");
    println!(
        "  • Batch time: {:.2}ms (vs {:.2}ms × 4 = {:.2}ms sequential)",
        batch4_time,
        seq_latency,
        seq_latency * 4.0
    );
    println!(
        "  • Per-request speedup: {:.2}× faster",
        seq_latency / batch4_per_req
    );
    println!(
        "  • Throughput: {:.1} → {:.1} QPS ({:+.1}%)\n",
        seq_qps,
        batch4_qps,
        (batch4_qps / seq_qps - 1.0) * 100.0
    );

    println!("Batch Size 8:");
    println!(
        "  • Batch time: {:.2}ms (vs {:.2}ms × 8 = {:.2}ms sequential)",
        batch8_time,
        seq_latency,
        seq_latency * 8.0
    );
    println!(
        "  • Per-request speedup: {:.2}× faster",
        seq_latency / batch8_per_req
    );
    println!(
        "  • Throughput: {:.1} → {:.1} QPS ({:+.1}%)\n",
        seq_qps,
        batch8_qps,
        (batch8_qps / seq_qps - 1.0) * 100.0
    );

    println!("💡 Conclusion:");
    println!(
        "  • Batching at Candle/model level provides {:.1}× - {:.1}× speedup",
        batch2_qps / seq_qps,
        batch8_qps / seq_qps
    );
    println!(
        "  • Per-request processing time reduced from {:.2}ms to {:.2}ms",
        seq_latency, batch8_per_req
    );
    println!("  • Optimal batch size depends on latency requirements vs throughput needs\n");

    Ok(())
}
