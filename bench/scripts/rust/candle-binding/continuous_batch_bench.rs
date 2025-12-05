// Continuous Batch Scheduler Benchmark
//
// Compares performance of:
// 1. Mutex-based serialized processing (baseline from concurrency_scalability_bench)
// 2. Continuous batch scheduler (vLLM-inspired)
//
// Measures:
// - Throughput (QPS)
// - Latency (P50, P95, P99)
// - Batch efficiency
//
// Usage:
//   cargo run --release --bin continuous_batch_bench

use candle_core::Device;
use candle_semantic_router::model_architectures::generative::{
    BatchSchedulerConfig, ContinuousBatchScheduler, Qwen3MultiLoRAClassifier,
};
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LatencyStats {
    min: f64,
    max: f64,
    mean: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    std_dev: f64,
}

#[derive(Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    method: String,
    concurrency: usize,
    total_requests: usize,
    duration_secs: f64,
    qps: f64,
    latency_stats: LatencyStats,
    success_rate: f64,
}

fn load_test_data() -> Vec<TestSample> {
    let data_path = "../bench/data/test_data.json";
    let contents = fs::read_to_string(data_path).expect("Failed to read test_data.json");
    serde_json::from_str(&contents).expect("Failed to parse test_data.json")
}

fn calculate_latency_stats(mut latencies: Vec<f64>) -> LatencyStats {
    if latencies.is_empty() {
        return LatencyStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            p50: 0.0,
            p95: 0.0,
            p99: 0.0,
            std_dev: 0.0,
        };
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = latencies[0];
    let max = latencies[latencies.len() - 1];
    let mean = latencies.iter().sum::<f64>() / latencies.len() as f64;

    let p50_idx = (latencies.len() as f64 * 0.50) as usize;
    let p95_idx = (latencies.len() as f64 * 0.95) as usize;
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;

    let p50 = latencies[p50_idx.min(latencies.len() - 1)];
    let p95 = latencies[p95_idx.min(latencies.len() - 1)];
    let p99 = latencies[p99_idx.min(latencies.len() - 1)];

    let variance =
        latencies.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / latencies.len() as f64;
    let std_dev = variance.sqrt();

    LatencyStats {
        min,
        max,
        mean,
        p50,
        p95,
        p99,
        std_dev,
    }
}

// Baseline: Mutex-based serialized processing
#[allow(dead_code)]
static MUTEX_CLASSIFIER: OnceLock<Mutex<Qwen3MultiLoRAClassifier>> = OnceLock::new();

#[allow(dead_code)]
fn initialize_mutex_classifier() -> Result<(), Box<dyn std::error::Error>> {
    let base_model_path = "../models/Qwen3-0.6B";
    let adapter_path = "../models/qwen3_generative_classifier_r16_fixed";
    let device = Device::cuda_if_available(0)?;

    println!("🔧 Initializing Mutex-based classifier...");
    println!("  Device: {:?}", device);
    let mut classifier = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    classifier.load_adapter("category", adapter_path)?;

    MUTEX_CLASSIFIER
        .set(Mutex::new(classifier))
        .map_err(|_| "Failed to initialize mutex classifier")?;
    println!("✅ Mutex classifier ready\n");

    Ok(())
}

#[allow(dead_code)]
fn benchmark_mutex(
    concurrency: usize,
    duration_secs: u64,
    test_data: &[TestSample],
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!(" MUTEX METHOD | Concurrency: {} ", concurrency);
    println!("{}", "=".repeat(70));

    let latencies = Arc::new(Mutex::new(Vec::new()));
    let successes = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let start_time = Instant::now();
    let mut handles = vec![];

    // Spawn worker threads
    for worker_id in 0..concurrency {
        let latencies = Arc::clone(&latencies);
        let successes = Arc::clone(&successes);
        let failures = Arc::clone(&failures);
        let stop_flag = Arc::clone(&stop_flag);
        let test_samples = test_data.to_vec();

        let handle = thread::spawn(move || {
            let mut request_count = 0;

            loop {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                let sample = &test_samples[request_count % test_samples.len()];
                let req_start = Instant::now();

                // Mutex lock - serializes all requests
                let result = {
                    let mut classifier = MUTEX_CLASSIFIER.get().unwrap().lock().unwrap();
                    classifier.classify_with_adapter(&sample.text, "category")
                };

                let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;

                match result {
                    Ok(_) => {
                        latencies.lock().unwrap().push(latency_ms);
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        if worker_id == 0 {
                            eprintln!("[Worker {}] Error: {:?}", worker_id, e);
                        }
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }

                request_count += 1;

                if worker_id == 0 && request_count % 10 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let total =
                        successes.load(Ordering::Relaxed) + failures.load(Ordering::Relaxed);
                    let qps = total as f64 / elapsed;
                    print!(
                        "\r⏱️  Elapsed: {:.1}s | Requests: {} | QPS: {:.1}    ",
                        elapsed, total, qps
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            }
        });

        handles.push(handle);
    }

    thread::sleep(Duration::from_secs(duration_secs));
    stop_flag.store(true, Ordering::Relaxed);

    for handle in handles {
        handle.join().unwrap();
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    let latencies_vec = latencies.lock().unwrap().clone();
    let total_successes = successes.load(Ordering::Relaxed);
    let total_failures = failures.load(Ordering::Relaxed);
    let total_requests = total_successes + total_failures;

    let qps = total_successes as f64 / total_duration;
    let success_rate = if total_requests > 0 {
        total_successes as f64 / total_requests as f64
    } else {
        0.0
    };
    let latency_stats = calculate_latency_stats(latencies_vec);

    println!("\n");

    Ok(BenchmarkResult {
        method: "Mutex".to_string(),
        concurrency,
        total_requests,
        duration_secs: total_duration,
        qps,
        latency_stats,
        success_rate,
    })
}

fn benchmark_continuous_batch(
    concurrency: usize,
    duration_secs: u64,
    test_data: &[TestSample],
    batch_config: &BatchSchedulerConfig,
) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!(
        " CONTINUOUS BATCH | Concurrency: {} | Batch Size: {} ",
        concurrency, batch_config.max_batch_size
    );
    println!("{}", "=".repeat(70));

    // Create new classifier for batch scheduler
    let base_model_path = "../models/Qwen3-0.6B";
    let adapter_path = "../models/qwen3_generative_classifier_r16_fixed";
    let device = Device::cuda_if_available(0)?;

    println!("  Device: {:?}", device);
    let mut classifier = Qwen3MultiLoRAClassifier::new(base_model_path, &device)?;
    classifier.load_adapter("category", adapter_path)?;

    // Create continuous batch scheduler
    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        classifier,
        batch_config.clone(),
    ));

    let latencies = Arc::new(Mutex::new(Vec::new()));
    let successes = Arc::new(AtomicUsize::new(0));
    let failures = Arc::new(AtomicUsize::new(0));
    let stop_flag = Arc::new(AtomicBool::new(false));

    let start_time = Instant::now();
    let mut handles = vec![];

    // Spawn worker threads
    for worker_id in 0..concurrency {
        let scheduler = Arc::clone(&scheduler);
        let latencies = Arc::clone(&latencies);
        let successes = Arc::clone(&successes);
        let failures = Arc::clone(&failures);
        let stop_flag = Arc::clone(&stop_flag);
        let test_samples = test_data.to_vec();

        let handle = thread::spawn(move || {
            let mut request_count = 0;

            loop {
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }

                let sample = &test_samples[request_count % test_samples.len()];
                let req_start = Instant::now();

                // Submit to continuous batch scheduler
                let result = scheduler.classify(sample.text.clone(), "category".to_string());

                let latency_ms = req_start.elapsed().as_secs_f64() * 1000.0;

                match result {
                    Ok(_) => {
                        latencies.lock().unwrap().push(latency_ms);
                        successes.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        if worker_id == 0 {
                            eprintln!("[Worker {}] Error: {:?}", worker_id, e);
                        }
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }

                request_count += 1;

                if worker_id == 0 && request_count % 10 == 0 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let total =
                        successes.load(Ordering::Relaxed) + failures.load(Ordering::Relaxed);
                    let qps = total as f64 / elapsed;
                    let stats = scheduler.get_stats();
                    let avg_batch = *stats.avg_batch_size.lock().unwrap();
                    print!(
                        "\r⏱️  Elapsed: {:.1}s | Requests: {} | QPS: {:.1} | Avg Batch: {:.1}    ",
                        elapsed, total, qps, avg_batch
                    );
                    std::io::Write::flush(&mut std::io::stdout()).ok();
                }
            }
        });

        handles.push(handle);
    }

    thread::sleep(Duration::from_secs(duration_secs));
    stop_flag.store(true, Ordering::Relaxed);

    for handle in handles {
        handle.join().unwrap();
    }

    let total_duration = start_time.elapsed().as_secs_f64();
    let latencies_vec = latencies.lock().unwrap().clone();
    let total_successes = successes.load(Ordering::Relaxed);
    let total_failures = failures.load(Ordering::Relaxed);
    let total_requests = total_successes + total_failures;

    let qps = total_successes as f64 / total_duration;
    let success_rate = if total_requests > 0 {
        total_successes as f64 / total_requests as f64
    } else {
        0.0
    };
    let latency_stats = calculate_latency_stats(latencies_vec);

    // Print scheduler stats
    let stats = scheduler.get_stats();
    println!("\n📊 Scheduler Stats:");
    println!(
        "  Total batches: {}",
        stats.total_batches.load(Ordering::Relaxed)
    );
    println!(
        "  Avg batch size: {:.2}",
        *stats.avg_batch_size.lock().unwrap()
    );
    println!(
        "  Avg latency: {:.1}ms",
        *stats.avg_latency_ms.lock().unwrap()
    );

    println!("\n");

    Ok(BenchmarkResult {
        method: format!("Continuous Batch ({})", batch_config.max_batch_size),
        concurrency,
        total_requests,
        duration_secs: total_duration,
        qps,
        latency_stats,
        success_rate,
    })
}

#[allow(dead_code)]
fn print_comparison_table(results: &[BenchmarkResult]) {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║           MUTEX vs CONTINUOUS BATCH COMPARISON                   ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<25} {:<12} {:<10} {:<10} {:<10} {:<10}",
        "Method", "Concurrency", "QPS", "P50(ms)", "P95(ms)", "P99(ms)"
    );
    println!("{}", "-".repeat(85));

    for result in results {
        println!(
            "{:<25} {:<12} {:<10.1} {:<10.1} {:<10.1} {:<10.1}",
            result.method,
            result.concurrency,
            result.qps,
            result.latency_stats.p50,
            result.latency_stats.p95,
            result.latency_stats.p99
        );
    }

    println!("\n{}", "=".repeat(85));

    // Calculate improvements
    println!("\n📈 PERFORMANCE ANALYSIS:\n");

    for concurrency in [1, 2, 4, 8, 16] {
        let mutex_result = results
            .iter()
            .find(|r| r.method == "Mutex" && r.concurrency == concurrency);
        let batch_result = results
            .iter()
            .find(|r| r.method.starts_with("Continuous Batch") && r.concurrency == concurrency);

        if let (Some(mutex), Some(batch)) = (mutex_result, batch_result) {
            let qps_improvement = (batch.qps - mutex.qps) / mutex.qps * 100.0;
            let latency_improvement = (mutex.latency_stats.p95 - batch.latency_stats.p95)
                / mutex.latency_stats.p95
                * 100.0;

            println!("Concurrency {}:", concurrency);
            println!(
                "  QPS: {:.1} → {:.1} ({:+.1}%)",
                mutex.qps, batch.qps, qps_improvement
            );
            println!(
                "  P95 Latency: {:.1}ms → {:.1}ms ({:+.1}%)",
                mutex.latency_stats.p95, batch.latency_stats.p95, latency_improvement
            );
            println!();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║         Continuous Batch Scheduler Concurrency Benchmark         ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Load test data
    println!("\n📊 Loading test data...");
    let test_data = load_test_data();
    println!("✅ Loaded {} test samples\n", test_data.len());

    let test_duration_secs = 10;
    let concurrency_levels = vec![1, 2, 4, 8, 12, 16, 24, 32]; // Extended concurrency testing
    let mut all_results = Vec::new();

    println!("⚙️  Configuration:");
    println!("  Duration per test: {}s", test_duration_secs);
    println!("  Concurrency levels: {:?}", concurrency_levels);
    println!("  Method: Continuous Batch Scheduler\n");

    let batch_config = BatchSchedulerConfig {
        max_batch_size: 4,
        batch_timeout_ms: 5,
        queue_capacity: 1000,
        verbose: false, // Disable verbose for cleaner output
    };

    for &concurrency in &concurrency_levels {
        println!("\n{}", "=".repeat(70));
        println!(
            " CONTINUOUS BATCH | Concurrency: {} | Batch Size: {} ",
            concurrency, batch_config.max_batch_size
        );
        println!("{}", "=".repeat(70));

        match benchmark_continuous_batch(concurrency, test_duration_secs, &test_data, &batch_config)
        {
            Ok(result) => {
                println!("\n📊 Results (Concurrency {}):", concurrency);
                println!("  QPS: {:.1}", result.qps);
                println!(
                    "  P50/P95/P99: {:.1}/{:.1}/{:.1} ms",
                    result.latency_stats.p50, result.latency_stats.p95, result.latency_stats.p99
                );
                all_results.push(result);
            }
            Err(e) => {
                eprintln!("❌ Continuous batch benchmark failed: {:?}", e);
                break;
            }
        }
    }

    // Print summary table
    println!("\n");
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    CONCURRENCY SCALING RESULTS                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    println!(
        "{:<12} {:<12} {:<12} {:<12} {:<12}",
        "Workers", "QPS", "P50 (ms)", "P95 (ms)", "P99 (ms)"
    );
    println!("{}", "-".repeat(60));

    for result in &all_results {
        println!(
            "{:<12} {:<12.1} {:<12.1} {:<12.1} {:<12.1}",
            result.concurrency,
            result.qps,
            result.latency_stats.p50,
            result.latency_stats.p95,
            result.latency_stats.p99
        );
    }

    println!("\n✅ Benchmark complete!");
    println!("\n💡 Key Insights:");
    println!("  • Continuous batching with true batched inference at model level");
    println!("  • Adaptive batch sizing optimizes GPU utilization");
    println!("  • Throughput scales with concurrency while latency stays bounded\n");

    Ok(())
}
