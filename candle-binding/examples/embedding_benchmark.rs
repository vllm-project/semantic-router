/// Standalone Embedding Performance Benchmark
///
/// This is a simpler, faster benchmark for quick performance checks
/// without the overhead of Criterion statistical analysis.
///
/// # Usage
/// ```bash
/// # CPU benchmark
/// cargo run --release --example embedding_benchmark -- --device cpu
///
/// # GPU benchmark
/// cargo run --release --example embedding_benchmark -- --device gpu
///
/// # Both CPU and GPU
/// cargo run --release --example embedding_benchmark
///
/// # With Flash Attention
/// cargo run --release --features flash-attn --example embedding_benchmark
///
/// # Custom model path
/// cargo run --release --example embedding_benchmark -- --model ./models/mom-embedding-pro
///
/// # Quick test (fewer iterations)
/// cargo run --release --example embedding_benchmark -- --quick
/// ```
use candle_core::{Device, Tensor};
use candle_semantic_router::model_architectures::embedding::{
    continuous_batch_scheduler::ContinuousBatchConfig, qwen3_batched::Qwen3EmbeddingModelBatched,
    qwen3_embedding::Qwen3EmbeddingModel,
};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tokenizers::Tokenizer;

// ========================================================================================
// Configuration
// ========================================================================================

const DEFAULT_MODEL_PATH: &str = "./models/mom-embedding-pro";

const TEST_TEXTS: &[&str] = &[
    "What is machine learning?",
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
    "Computer vision allows machines to interpret and understand visual information.",
];

// ========================================================================================
// Helper Functions
// ========================================================================================

fn prepare_batch(
    texts: &[String],
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
) -> Result<(Tensor, Tensor), Box<dyn std::error::Error>> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| format!("Tokenization error: {}", e))?;

    let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
    let mut all_input_ids = Vec::new();
    let mut all_attention_masks = Vec::new();

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();

        // Left padding for Qwen3
        let pad_len = max_len - ids.len();
        let mut padded_ids = vec![0u32; pad_len];
        padded_ids.extend_from_slice(ids);

        let mut padded_mask = vec![0u32; pad_len];
        padded_mask.extend_from_slice(attention_mask);

        all_input_ids.push(padded_ids);
        all_attention_masks.push(padded_mask);
    }

    let input_ids = Tensor::new(all_input_ids, device)?;
    let attention_mask = Tensor::new(all_attention_masks, device)?;

    Ok((input_ids, attention_mask))
}

fn format_duration(millis: u128) -> String {
    if millis < 1000 {
        format!("{} ms", millis)
    } else {
        format!("{:.2} s", millis as f64 / 1000.0)
    }
}

// ========================================================================================
// Benchmark Functions
// ========================================================================================

fn benchmark_latency(
    model: &Qwen3EmbeddingModel,
    input_ids: &Tensor,
    attention_mask: &Tensor,
    iterations: usize,
) -> (u128, u128, u128) {
    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        model
            .embedding_forward(input_ids, attention_mask)
            .expect("Embedding forward failed");
        times.push(start.elapsed().as_millis());
    }

    times.sort();
    let mean = times.iter().sum::<u128>() / times.len() as u128;
    let median = times[times.len() / 2];
    let p95 = times[(times.len() as f64 * 0.95) as usize];

    (mean, median, p95)
}

fn benchmark_throughput(
    model: &Qwen3EmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
    batch_size: usize,
    iterations: usize,
) -> f64 {
    // Prepare batch
    let texts: Vec<String> = (0..batch_size)
        .map(|i| format!("{} [sample {}]", TEST_TEXTS[i % TEST_TEXTS.len()], i))
        .collect();

    let (input_ids, attention_mask) =
        prepare_batch(&texts, tokenizer, device).expect("Failed to prepare batch");

    // Warmup
    for _ in 0..3 {
        let _ = model.embedding_forward(&input_ids, &attention_mask);
    }

    // Measure
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = model.embedding_forward(&input_ids, &attention_mask);
    }
    let elapsed = start.elapsed();

    let total_embeddings = iterations * batch_size;
    total_embeddings as f64 / elapsed.as_secs_f64()
}

/// Benchmark concurrent requests (simulates real-world API server)
fn benchmark_concurrent_requests(
    model: Arc<Qwen3EmbeddingModel>,
    tokenizer: &Tokenizer,
    device: &Device,
    num_threads: usize,
    requests_per_thread: usize,
) -> f64 {
    println!("\n{}", "─".repeat(80));
    println!("🔥 CONCURRENT REQUEST BENCHMARK (Real-World Scenario)");
    println!("{}", "─".repeat(80));
    println!("  Simulating {} concurrent clients", num_threads);
    println!(
        "  Each client sends {} requests sequentially",
        requests_per_thread
    );
    println!("  Total requests: {}", num_threads * requests_per_thread);
    println!();

    // Prepare test inputs for each thread
    let test_texts: Vec<String> = TEST_TEXTS
        .iter()
        .cycle()
        .take(num_threads)
        .enumerate()
        .map(|(i, &text)| format!("{} [client-{}]", text, i))
        .collect();

    // Tokenize all texts upfront
    let mut thread_inputs = Vec::new();
    for text in &test_texts {
        let encodings = tokenizer
            .encode_batch(vec![text.clone()], true)
            .expect("Tokenization failed");

        let ids: Vec<u32> = encodings[0].get_ids().to_vec();
        let mask: Vec<u32> = encodings[0].get_attention_mask().to_vec();

        // Create 2D array for batch_size=1
        let input_ids = Tensor::new(vec![ids], device).expect("Failed to create tensor");
        let attention_mask = Tensor::new(vec![mask], device).expect("Failed to create tensor");

        thread_inputs.push((input_ids, attention_mask));
    }

    // Wrap inputs in Arc for sharing across threads
    let thread_inputs = Arc::new(thread_inputs);

    // Counter for completed requests
    let completed = Arc::new(Mutex::new(0usize));

    println!("  Starting {} threads...", num_threads);
    let start = Instant::now();

    // Spawn worker threads
    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let model = Arc::clone(&model);
            let inputs = Arc::clone(&thread_inputs);
            let completed = Arc::clone(&completed);
            let input_idx = thread_id % thread_inputs.len();

            thread::spawn(move || {
                // Each thread sends requests_per_thread individual requests
                for _ in 0..requests_per_thread {
                    let (input_ids, attention_mask) = &inputs[input_idx];

                    // This is a SINGLE embedding request (batch_size=1)
                    let _embedding = model
                        .embedding_forward(input_ids, attention_mask)
                        .expect("Embedding failed");

                    // Update counter
                    let mut count = completed.lock().unwrap();
                    *count += 1;
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();
    let total_requests = num_threads * requests_per_thread;
    let throughput = total_requests as f64 / elapsed.as_secs_f64();

    println!("  ✅ Completed in {:.2}s", elapsed.as_secs_f64());
    println!("  📈 Throughput: {:.2} emb/s", throughput);

    throughput
}

/// Benchmark concurrent requests with batched model
fn benchmark_concurrent_requests_batched(
    model: Arc<Qwen3EmbeddingModelBatched>,
    tokenizer: &Tokenizer,
    device: &Device,
    num_threads: usize,
    requests_per_thread: usize,
) -> f64 {
    // Prepare test inputs for each thread
    let test_texts: Vec<String> = TEST_TEXTS
        .iter()
        .cycle()
        .take(num_threads)
        .enumerate()
        .map(|(i, &text)| format!("{} [client-{}]", text, i))
        .collect();

    // Tokenize all texts upfront
    let mut thread_inputs = Vec::new();
    for text in &test_texts {
        let encodings = tokenizer
            .encode_batch(vec![text.clone()], true)
            .expect("Tokenization failed");

        let ids: Vec<u32> = encodings[0].get_ids().to_vec();
        let mask: Vec<u32> = encodings[0].get_attention_mask().to_vec();

        // Create 2D array for batch_size=1
        let input_ids = Tensor::new(vec![ids], device).expect("Failed to create tensor");
        let attention_mask = Tensor::new(vec![mask], device).expect("Failed to create tensor");

        thread_inputs.push((input_ids, attention_mask));
    }

    // Wrap inputs in Arc for sharing across threads
    let thread_inputs = Arc::new(thread_inputs);

    // Counter for completed requests
    let completed = Arc::new(Mutex::new(0usize));

    let start = Instant::now();

    // Spawn worker threads
    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let model = Arc::clone(&model);
            let inputs = Arc::clone(&thread_inputs);
            let completed = Arc::clone(&completed);
            let input_idx = thread_id % thread_inputs.len();

            thread::spawn(move || {
                // Each thread sends requests_per_thread individual requests
                for _ in 0..requests_per_thread {
                    let (input_ids, attention_mask) = &inputs[input_idx];

                    // This is a SINGLE embedding request (batch_size=1)
                    // Continuous batching will automatically group concurrent requests
                    let _embedding = model
                        .embedding_forward(input_ids, attention_mask)
                        .expect("Embedding failed");

                    // Update counter
                    let mut count = completed.lock().unwrap();
                    *count += 1;
                }
            })
        })
        .collect();

    // Wait for all threads to complete
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();
    let total_requests = num_threads * requests_per_thread;
    let throughput = total_requests as f64 / elapsed.as_secs_f64();

    println!("  ✅ Completed in {:.2}s", elapsed.as_secs_f64());
    println!("  📈 Throughput: {:.2} emb/s", throughput);

    throughput
}

fn run_benchmarks(
    device_name: &str,
    device: &Device,
    model_path: &str,
    quick: bool,
    skip_concurrent: bool,
) {
    println!("\n{}", "=".repeat(80));
    println!("  {} Performance Benchmark", device_name);
    println!("{}\n", "=".repeat(80));

    // Load model
    println!("📦 Loading model from: {}", model_path);
    let model_start = Instant::now();
    let model = match Qwen3EmbeddingModel::load(model_path, device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model: {:?}", e);
            eprintln!("   Make sure MODEL_PATH points to a valid Qwen3-Embedding model");
            return;
        }
    };
    let model_load_time = model_start.elapsed();
    println!(
        "✅ Model loaded in {}",
        format_duration(model_load_time.as_millis())
    );

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "❌ Failed to load tokenizer from {}: {:?}",
                tokenizer_path, e
            );
            return;
        }
    };

    let iterations = if quick { 10 } else { 50 };

    // 1. Single Embedding Latency
    println!("\n{}", "─".repeat(80));
    println!("1️⃣  Single Embedding Latency");
    println!("{}", "─".repeat(80));

    let text = vec![TEST_TEXTS[0].to_string()];
    let (input_ids, attention_mask) =
        prepare_batch(&text, &tokenizer, device).expect("Failed to prepare batch");

    let (mean, median, p95) = benchmark_latency(&model, &input_ids, &attention_mask, iterations);

    println!("  Mean:   {} ms", mean);
    println!("  Median: {} ms", median);
    println!("  P95:    {} ms", p95);

    // 2. Batch Scaling
    println!("\n{}", "─".repeat(80));
    println!("2️⃣  Batch Scaling Performance");
    println!("{}", "─".repeat(80));
    println!("  Batch Size | Latency (ms) | Throughput (emb/s)");
    println!("  -----------|--------------|-------------------");

    for batch_size in &[1, 4, 8, 16, 32, 64] {
        let texts: Vec<String> = (0..*batch_size)
            .map(|i| TEST_TEXTS[i % TEST_TEXTS.len()].to_string())
            .collect();

        let (input_ids, attention_mask) =
            prepare_batch(&texts, &tokenizer, device).expect("Failed to prepare batch");

        let (mean, _, _) = benchmark_latency(&model, &input_ids, &attention_mask, iterations / 2);
        let throughput = (*batch_size as f64 * 1000.0) / mean as f64;

        println!("  {:10} | {:12} | {:17.2}", batch_size, mean, throughput);
    }

    // 3. Sustained Throughput
    println!("\n{}", "─".repeat(80));
    println!("3️⃣  Sustained Throughput");
    println!("{}", "─".repeat(80));

    let batch_size = 32;
    let throughput_iterations = if quick { 20 } else { 100 };

    println!(
        "  Testing with batch_size={}, iterations={}",
        batch_size, throughput_iterations
    );
    let throughput = benchmark_throughput(
        &model,
        &tokenizer,
        device,
        batch_size,
        throughput_iterations,
    );

    println!("  Throughput: {:.2} embeddings/second", throughput);
    println!("  Time per embedding: {:.2} ms", 1000.0 / throughput);

    // 4. Sequence Length Scaling
    println!("\n{}", "─".repeat(80));
    println!("4️⃣  Sequence Length Scaling");
    println!("{}", "─".repeat(80));
    println!("  Approx Tokens | Latency (ms)");
    println!("  --------------|-------------");

    for &multiplier in &[1, 2, 4, 8] {
        let text = vec![TEST_TEXTS[0].repeat(multiplier)];
        let (input_ids, attention_mask) =
            prepare_batch(&text, &tokenizer, device).expect("Failed to prepare batch");

        let actual_tokens = input_ids.dim(1).unwrap_or(0);
        let (mean, _, _) = benchmark_latency(&model, &input_ids, &attention_mask, iterations / 4);

        println!("  {:13} | {:12}", actual_tokens, mean);
    }

    // 5. Concurrent Requests Comparison (Real-World Scenario)
    if !skip_concurrent {
        let num_threads = if quick { 8 } else { 16 };
        let requests_per_thread = if quick { 5 } else { 10 };

        println!("\n{}", "=".repeat(80));
        println!("🔥 CONCURRENT REQUEST COMPARISON (Real-World API Server Scenario)");
        println!("{}", "=".repeat(80));
        println!(
            "  Simulating {} concurrent clients, each sending {} requests",
            num_threads, requests_per_thread
        );
        println!(
            "  Total: {} individual embedding requests",
            num_threads * requests_per_thread
        );
        println!();

        // Test 1: Baseline (without continuous batching)
        println!("📊 Test 1: BASELINE (Standard Model)");
        println!("  Processing individual requests sequentially...");
        let baseline_throughput = benchmark_concurrent_requests(
            Arc::new(model),
            &tokenizer,
            device,
            num_threads,
            requests_per_thread,
        );

        println!();

        // Test 2: With Continuous Batching
        println!("📊 Test 2: WITH CONTINUOUS BATCHING");
        println!("  Loading model with continuous batching enabled...");

        let max_wait_ms = 5;
        let max_batch = 32;

        let batch_config = ContinuousBatchConfig {
            max_batch_size: max_batch,
            max_wait_time_ms: max_wait_ms,
            min_batch_size: 1,
            enable_dynamic: true,
            max_seq_len_diff: 128,
            verbose: false,
        };

        let batched_model = match Qwen3EmbeddingModel::load(model_path, device) {
            Ok(m) => Qwen3EmbeddingModelBatched::from_model(m, batch_config),
            Err(e) => {
                eprintln!("  ❌ Failed to load batched model: {:?}", e);
                return;
            }
        };

        println!("  Processing with automatic request batching...");
        let batched_throughput = benchmark_concurrent_requests_batched(
            Arc::new(batched_model),
            &tokenizer,
            device,
            num_threads,
            requests_per_thread,
        );

        // Show comparison
        println!();
        println!("{}", "=".repeat(80));
        println!("📈 PERFORMANCE COMPARISON");
        println!("{}", "=".repeat(80));
        println!();
        println!(
            "  Metric                          | Baseline    | Continuous Batch | Improvement"
        );
        println!(
            "  --------------------------------|-------------|------------------|------------"
        );
        println!(
            "  Throughput (emb/s)              | {:11.2} | {:16.2} | {:8.2}x",
            baseline_throughput,
            batched_throughput,
            batched_throughput / baseline_throughput
        );
        println!(
            "  Time per embedding (ms)         | {:11.2} | {:16.2} | {:8.2}x faster",
            1000.0 / baseline_throughput,
            1000.0 / batched_throughput,
            baseline_throughput / batched_throughput
        );
        println!();

        let improvement_pct =
            ((batched_throughput - baseline_throughput) / baseline_throughput) * 100.0;
        if improvement_pct > 50.0 {
            println!(
                "  🚀 HUGE WIN: {:.1}% improvement with continuous batching!",
                improvement_pct
            );
        } else if improvement_pct > 20.0 {
            println!(
                "  ✅ GOOD: {:.1}% improvement with continuous batching",
                improvement_pct
            );
        } else if improvement_pct > 0.0 {
            println!(
                "  ✓ Modest: {:.1}% improvement with continuous batching",
                improvement_pct
            );
        } else {
            println!("  ⚠️  No improvement - baseline already optimal for this workload");
        }

        println!();
        println!("  💡 Note: Continuous batching automatically groups concurrent requests");
        println!(
            "     arriving within {}ms window into efficient batches of up to {}.",
            max_wait_ms, max_batch
        );
    } else {
        println!(
            "\n  ⏭️  Skipping concurrent request benchmark (use without --skip-concurrent to run)"
        );
    }

    println!("\n{}", "=".repeat(80));
}

// ========================================================================================
// Main
// ========================================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut model_path = DEFAULT_MODEL_PATH.to_string();
    let mut device_filter = "both".to_string();
    let mut quick = false;
    let mut skip_concurrent = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    model_path = args[i + 1].clone();
                    i += 2;
                } else {
                    eprintln!("❌ --model requires a path argument");
                    std::process::exit(1);
                }
            }
            "--device" | "-d" => {
                if i + 1 < args.len() {
                    device_filter = args[i + 1].to_lowercase();
                    i += 2;
                } else {
                    eprintln!("❌ --device requires cpu, gpu, or both");
                    std::process::exit(1);
                }
            }
            "--quick" | "-q" => {
                quick = true;
                i += 1;
            }
            "--skip-concurrent" => {
                skip_concurrent = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Embedding Performance Benchmark\n");
                println!("Usage: embedding_benchmark [OPTIONS]\n");
                println!("Options:");
                println!(
                    "  --model, -m <PATH>    Model directory path (default: {})",
                    DEFAULT_MODEL_PATH
                );
                println!("  --device, -d <TYPE>   Device: cpu, gpu, or both (default: both)");
                println!("  --quick, -q           Run fewer iterations for faster results");
                println!("  --skip-concurrent     Skip concurrent request benchmark");
                println!("  --help, -h            Show this help message");
                std::process::exit(0);
            }
            _ => {
                eprintln!("❌ Unknown argument: {}", args[i]);
                eprintln!("   Use --help for usage information");
                std::process::exit(1);
            }
        }
    }

    // Print configuration
    println!("\n🔧 Configuration:");
    println!("  Model: {}", model_path);
    println!("  Device: {}", device_filter);
    println!("  Quick mode: {}", quick);

    #[cfg(feature = "flash-attn")]
    println!("  Flash Attention: ✅ Enabled");

    #[cfg(not(feature = "flash-attn"))]
    println!("  Flash Attention: ❌ Disabled");

    // Run benchmarks
    match device_filter.as_str() {
        "cpu" => {
            run_benchmarks("CPU", &Device::Cpu, &model_path, quick, skip_concurrent);
        }
        "gpu" => match Device::cuda_if_available(0) {
            Ok(device) => {
                run_benchmarks("GPU", &device, &model_path, quick, skip_concurrent);
            }
            Err(_) => {
                eprintln!("❌ GPU not available");
                std::process::exit(1);
            }
        },
        "both" | _ => {
            // CPU
            run_benchmarks("CPU", &Device::Cpu, &model_path, quick, skip_concurrent);

            // GPU (if available)
            if let Ok(device) = Device::cuda_if_available(0) {
                run_benchmarks("GPU", &device, &model_path, quick, skip_concurrent);

                // Comparison
                println!("\n{}", "=".repeat(80));
                println!("  📊 CPU vs GPU Comparison");
                println!("{}\n", "=".repeat(80));
                println!("  See individual benchmark results above for detailed comparison");
            } else {
                println!("\n⚠️  GPU not available, CPU-only results shown");
            }
        }
    }

    println!("\n✅ Benchmark complete!\n");
}
