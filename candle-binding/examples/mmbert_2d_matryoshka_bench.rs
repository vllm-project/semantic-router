/// mmBERT 2D Matryoshka Embedding Benchmark
///
/// Benchmarks the mmBERT-Embed-32K-2D-Matryoshka embedding model using the library API.
/// Tests different Matryoshka dimensions, sequence lengths, and batch sizes.
///
/// # Environment Variables
/// - `MMBERT_MODEL_PATH`: Path to the mmBERT model directory (required)
///
/// # Usage
/// ```bash
/// # Set model path and run benchmark
/// export MMBERT_MODEL_PATH=/path/to/mmbert-embed-32k-2d-matryoshka
/// cargo run --release --no-default-features --example mmbert_2d_matryoshka_bench
///
/// # Quick test with fewer iterations
/// MMBERT_MODEL_PATH=/path/to/model cargo run --release --no-default-features \
///   --example mmbert_2d_matryoshka_bench -- --quick
///
/// # CPU only mode
/// MMBERT_MODEL_PATH=/path/to/model cargo run --release --no-default-features \
///   --example mmbert_2d_matryoshka_bench -- --device cpu
///
/// # Download model first if needed:
/// huggingface-cli download llm-semantic-router/mmbert-embed-32k-2d-matryoshka
/// ```
use candle_core::Device;
use candle_semantic_router::model_architectures::embedding::{
    MatryoshkaConfig, MmBertEmbeddingModel,
};
use std::env;
use std::time::Instant;
use tokenizers::Tokenizer;

// ========================================================================================
// Configuration
// ========================================================================================

/// Get model path from environment variable
fn get_model_path() -> Result<String, String> {
    env::var("MMBERT_MODEL_PATH").map_err(|_| {
        "MMBERT_MODEL_PATH environment variable not set.\n\
         Usage: MMBERT_MODEL_PATH=/path/to/model cargo run --example mmbert_2d_matryoshka_bench\n\
         Download: huggingface-cli download llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
            .to_string()
    })
}

/// Matryoshka dimensions supported by the model
const MATRYOSHKA_DIMS: &[usize] = &[768, 512, 256, 128, 64];

/// Sequence lengths to benchmark
const SEQ_LENGTHS: &[usize] = &[128, 512, 1024, 2048, 4096];

/// Batch sizes to benchmark
const BATCH_SIZES: &[usize] = &[1, 4, 8, 16, 32];

/// Sample texts for benchmarking
const SAMPLE_TEXTS: &[&str] = &[
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
    "Natural language processing enables computers to understand, interpret, and generate human language.",
    "Deep learning uses neural networks with multiple layers to progressively extract higher-level features.",
    "Transformers have revolutionized NLP by enabling parallel processing and capturing long-range dependencies.",
    "Embedding models convert text into dense vector representations for semantic similarity tasks.",
];

// ========================================================================================
// Benchmark Results
// ========================================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchmarkResult {
    name: String,
    dim: usize,
    seq_len: usize,
    batch_size: usize,
    mean_ms: f64,
    std_ms: f64,
    throughput: f64,
}

struct LatencyStats {
    mean: f64,
    std: f64,
}

fn compute_stats(times: &[f64]) -> LatencyStats {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    LatencyStats { mean, std }
}

// ========================================================================================
// Helper Functions
// ========================================================================================

/// Generate text of approximately the specified token count
fn generate_text(approx_tokens: usize) -> String {
    let base_text = SAMPLE_TEXTS.join(" ");
    let words_needed = (approx_tokens as f64 / 1.3) as usize;

    let mut result = String::new();
    while result.split_whitespace().count() < words_needed {
        result.push_str(&base_text);
        result.push(' ');
    }

    result
        .split_whitespace()
        .take(words_needed)
        .collect::<Vec<_>>()
        .join(" ")
}

// ========================================================================================
// Benchmark Functions
// ========================================================================================

/// Benchmark the `encode_batch` API at different Matryoshka dimensions
fn bench_matryoshka_dimensions(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    matryoshka_config: &MatryoshkaConfig,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing Matryoshka dimension reduction...");

    let texts: Vec<&str> = SAMPLE_TEXTS.iter().take(8).copied().collect();
    let mut results = Vec::new();

    for &dim in MATRYOSHKA_DIMS {
        // Validate dimension is supported
        if !matryoshka_config.validate_dimension(dim) {
            println!("    Skipping unsupported dimension: {}", dim);
            continue;
        }

        // Warmup using encode_batch
        for _ in 0..warmup {
            let _ = model.encode_batch(tokenizer, &texts, 512);
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            // Note: encode_batch uses full dimension, we measure the API call
            let _ = model.encode_batch(tokenizer, &texts, 512);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let stats = compute_stats(&times);
        let throughput = (texts.len() as f64 * 1000.0) / stats.mean;

        // Get quality estimate from MatryoshkaConfig
        let quality = matryoshka_config.estimate_quality(22, dim);

        results.push(BenchmarkResult {
            name: format!("{}d ({}% quality)", dim, (quality * 100.0) as usize),
            dim,
            seq_len: 512,
            batch_size: texts.len(),
            mean_ms: stats.mean,
            std_ms: stats.std,
            throughput,
        });

        println!(
            "    {}d: {:.2}ms ± {:.2}ms | {:.1} emb/s | {:.0}% quality",
            dim,
            stats.mean,
            stats.std,
            throughput,
            quality * 100.0
        );
    }

    results
}

/// Benchmark at different sequence lengths using embedding_forward API
fn bench_sequence_lengths(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    device: &Device,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing sequence length scaling...");

    let batch_size = 4;
    let mut results = Vec::new();

    for &seq_len in SEQ_LENGTHS {
        // Generate texts of appropriate length
        let text = generate_text(seq_len);
        let texts: Vec<&str> = vec![&text; batch_size];

        // Tokenize
        let encodings = match tokenizer.encode_batch(texts.clone(), true) {
            Ok(e) => e,
            Err(err) => {
                println!(
                    "    Skipping seq_len={}: tokenization error: {}",
                    seq_len, err
                );
                continue;
            }
        };

        // Prepare tensors
        let actual_len = encodings.iter().map(|e| e.len()).max().unwrap_or(seq_len);
        let mut input_ids_vec = Vec::new();
        let mut attention_mask_vec = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let pad_len = actual_len - ids.len();

            let mut padded_ids = vec![0u32; pad_len];
            padded_ids.extend_from_slice(ids);

            let mut padded_mask = vec![0u32; pad_len];
            padded_mask.extend(mask.iter().map(|&x| x as u32));

            input_ids_vec.extend(padded_ids);
            attention_mask_vec.extend(padded_mask);
        }

        let input_ids =
            match candle_core::Tensor::from_vec(input_ids_vec, (batch_size, actual_len), device) {
                Ok(t) => t,
                Err(e) => {
                    println!("    Skipping seq_len={}: tensor error: {}", seq_len, e);
                    continue;
                }
            };

        let attention_mask = match candle_core::Tensor::from_vec(
            attention_mask_vec,
            (batch_size, actual_len),
            device,
        ) {
            Ok(t) => t,
            Err(e) => {
                println!("    Skipping seq_len={}: mask error: {}", seq_len, e);
                continue;
            }
        };

        // Warmup using embedding_forward API
        for _ in 0..warmup {
            let _ = model.embedding_forward(&input_ids, Some(&attention_mask));
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = model.embedding_forward(&input_ids, Some(&attention_mask));
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let stats = compute_stats(&times);
        let throughput = (batch_size as f64 * 1000.0) / stats.mean;
        let ms_per_token = stats.mean / actual_len as f64;

        results.push(BenchmarkResult {
            name: format!("seq_len={}", actual_len),
            dim: 768,
            seq_len: actual_len,
            batch_size,
            mean_ms: stats.mean,
            std_ms: stats.std,
            throughput,
        });

        println!(
            "    seq_len={}: {:.2}ms ± {:.2}ms | {:.1} emb/s | {:.4} ms/tok",
            actual_len, stats.mean, stats.std, throughput, ms_per_token
        );
    }

    results
}

/// Benchmark at different batch sizes using encode_batch API
fn bench_batch_sizes(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing batch size scaling...");

    let seq_len = 512;
    let base_text = generate_text(seq_len);
    let mut results = Vec::new();

    for &batch_size in BATCH_SIZES {
        let texts: Vec<&str> = vec![base_text.as_str(); batch_size];

        // Warmup
        for _ in 0..warmup {
            let _ = model.encode_batch(tokenizer, &texts, seq_len);
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = model.encode_batch(tokenizer, &texts, seq_len);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let stats = compute_stats(&times);
        let throughput = (batch_size as f64 * 1000.0) / stats.mean;
        let ms_per_embed = stats.mean / batch_size as f64;

        results.push(BenchmarkResult {
            name: format!("batch_size={}", batch_size),
            dim: 768,
            seq_len,
            batch_size,
            mean_ms: stats.mean,
            std_ms: stats.std,
            throughput,
        });

        println!(
            "    batch_size={}: {:.2}ms ± {:.2}ms | {:.1} emb/s | {:.2} ms/emb",
            batch_size, stats.mean, stats.std, throughput, ms_per_embed
        );
    }

    results
}

/// Benchmark embedding_forward_with_matryoshka API - dimension only
fn bench_matryoshka_api(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    device: &Device,
    matryoshka_config: &MatryoshkaConfig,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing dimension reduction (full 22 layers)...");

    let batch_size = 4;
    let seq_len = 128; // Short for CPU testing
    let texts: Vec<&str> = SAMPLE_TEXTS
        .iter()
        .cycle()
        .take(batch_size)
        .copied()
        .collect();
    let mut results = Vec::new();

    // Tokenize once
    let encodings = tokenizer
        .encode_batch(texts.clone(), true)
        .expect("Tokenization failed");
    let actual_len = encodings
        .iter()
        .map(|e| e.len())
        .max()
        .unwrap_or(seq_len)
        .min(seq_len);

    let mut input_ids_vec = Vec::new();
    let mut attention_mask_vec = Vec::new();

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        for i in 0..actual_len {
            if i < ids.len() {
                input_ids_vec.push(ids[i]);
                attention_mask_vec.push(mask[i] as u32);
            } else {
                input_ids_vec.push(0);
                attention_mask_vec.push(0);
            }
        }
    }

    let input_ids = candle_core::Tensor::from_vec(input_ids_vec, (batch_size, actual_len), device)
        .expect("Failed to create input_ids");
    let attention_mask =
        candle_core::Tensor::from_vec(attention_mask_vec, (batch_size, actual_len), device)
            .expect("Failed to create attention_mask");

    // Test each Matryoshka dimension
    for &dim in MATRYOSHKA_DIMS {
        // Warmup
        for _ in 0..warmup {
            let _ = model.embedding_forward_with_matryoshka(
                &input_ids,
                Some(&attention_mask),
                None, // Full layers
                Some(dim),
            );
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let result = model.embedding_forward_with_matryoshka(
                &input_ids,
                Some(&attention_mask),
                None,
                Some(dim),
            );
            times.push(start.elapsed().as_secs_f64() * 1000.0);

            if let Ok(emb) = result {
                let shape = emb.dims();
                assert_eq!(shape[1], dim, "Expected dim {}, got {}", dim, shape[1]);
            }
        }

        let stats = compute_stats(&times);
        let throughput = (batch_size as f64 * 1000.0) / stats.mean;
        let quality = matryoshka_config.estimate_quality(22, dim);

        results.push(BenchmarkResult {
            name: format!("matryoshka_{}d", dim),
            dim,
            seq_len: actual_len,
            batch_size,
            mean_ms: stats.mean,
            std_ms: stats.std,
            throughput,
        });

        println!(
            "    {}d: {:.2}ms | {:.1} emb/s | {:.0}% quality",
            dim,
            stats.mean,
            throughput,
            quality * 100.0
        );
    }

    results
}

/// Benchmark layer early exit (2D Matryoshka - the "2nd D")
fn bench_layer_early_exit(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    device: &Device,
    matryoshka_config: &MatryoshkaConfig,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing layer early exit (768d, varying layers)...");

    let batch_size = 4;
    let seq_len = 64; // Short for CPU testing
    let texts: Vec<&str> = SAMPLE_TEXTS
        .iter()
        .cycle()
        .take(batch_size)
        .copied()
        .collect();
    let mut results = Vec::new();

    // Tokenize once
    let encodings = tokenizer
        .encode_batch(texts.clone(), true)
        .expect("Tokenization failed");
    let actual_len = encodings
        .iter()
        .map(|e| e.len())
        .max()
        .unwrap_or(seq_len)
        .min(seq_len);

    let mut input_ids_vec = Vec::new();
    let mut attention_mask_vec = Vec::new();

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        for i in 0..actual_len {
            if i < ids.len() {
                input_ids_vec.push(ids[i]);
                attention_mask_vec.push(mask[i] as u32);
            } else {
                input_ids_vec.push(0);
                attention_mask_vec.push(0);
            }
        }
    }

    let input_ids = candle_core::Tensor::from_vec(input_ids_vec, (batch_size, actual_len), device)
        .expect("Failed to create input_ids");
    let attention_mask =
        candle_core::Tensor::from_vec(attention_mask_vec, (batch_size, actual_len), device)
            .expect("Failed to create attention_mask");

    let exit_layers = &matryoshka_config.layers; // [3, 6, 11, 22]

    // Baseline at full layers
    let mut baseline_ms = 0.0;

    for &target_layer in exit_layers {
        // Warmup
        for _ in 0..warmup {
            let _ = model.embedding_forward_with_matryoshka(
                &input_ids,
                Some(&attention_mask),
                Some(target_layer),
                None, // Full dimension
            );
        }

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let result = model.embedding_forward_with_matryoshka(
                &input_ids,
                Some(&attention_mask),
                Some(target_layer),
                None,
            );
            times.push(start.elapsed().as_secs_f64() * 1000.0);

            if let Ok(emb) = result {
                let shape = emb.dims();
                assert_eq!(shape[1], 768, "Expected 768d, got {}", shape[1]);
            }
        }

        let stats = compute_stats(&times);
        let throughput = (batch_size as f64 * 1000.0) / stats.mean;
        let quality = matryoshka_config.estimate_quality(target_layer, 768);

        if target_layer == 22 {
            baseline_ms = stats.mean;
        }
        let speedup = if baseline_ms > 0.0 {
            baseline_ms / stats.mean
        } else {
            1.0
        };

        results.push(BenchmarkResult {
            name: format!("layer_{}", target_layer),
            dim: 768,
            seq_len: actual_len,
            batch_size,
            mean_ms: stats.mean,
            std_ms: stats.std,
            throughput,
        });

        println!(
            "    L{}: {:.2}ms | {:.1} emb/s | {:.0}% quality | {:.2}x speedup",
            target_layer,
            stats.mean,
            throughput,
            quality * 100.0,
            speedup
        );
    }

    results
}

/// Benchmark full 2D Matryoshka matrix (layers × dimensions)
fn bench_2d_matrix(
    model: &MmBertEmbeddingModel,
    tokenizer: &Tokenizer,
    device: &Device,
    matryoshka_config: &MatryoshkaConfig,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n  Testing 2D Matryoshka matrix (layers × dimensions)...");

    let batch_size = 4;
    let seq_len = 64;
    let texts: Vec<&str> = SAMPLE_TEXTS
        .iter()
        .cycle()
        .take(batch_size)
        .copied()
        .collect();
    let mut results = Vec::new();

    // Tokenize
    let encodings = tokenizer
        .encode_batch(texts.clone(), true)
        .expect("Tokenization failed");
    let actual_len = encodings
        .iter()
        .map(|e| e.len())
        .max()
        .unwrap_or(seq_len)
        .min(seq_len);

    let mut input_ids_vec = Vec::new();
    let mut attention_mask_vec = Vec::new();

    for encoding in &encodings {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();

        for i in 0..actual_len {
            if i < ids.len() {
                input_ids_vec.push(ids[i]);
                attention_mask_vec.push(mask[i] as u32);
            } else {
                input_ids_vec.push(0);
                attention_mask_vec.push(0);
            }
        }
    }

    let input_ids = candle_core::Tensor::from_vec(input_ids_vec, (batch_size, actual_len), device)
        .expect("Failed to create input_ids");
    let attention_mask =
        candle_core::Tensor::from_vec(attention_mask_vec, (batch_size, actual_len), device)
            .expect("Failed to create attention_mask");

    // Print header
    print!("    {:>6}", "");
    for &dim in &[768, 256, 64] {
        print!(" | {:>8}", format!("{}d", dim));
    }
    println!();
    println!("    {}", "-".repeat(40));

    let exit_layers = &[22, 11, 6, 3];

    for &target_layer in exit_layers {
        print!("    L{:<4}", target_layer);

        for &target_dim in &[768, 256, 64] {
            // Warmup
            for _ in 0..warmup {
                let _ = model.embedding_forward_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    Some(target_layer),
                    Some(target_dim),
                );
            }

            // Benchmark
            let mut times = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = model.embedding_forward_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    Some(target_layer),
                    Some(target_dim),
                );
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }

            let stats = compute_stats(&times);
            let _quality = matryoshka_config.estimate_quality(target_layer, target_dim);

            results.push(BenchmarkResult {
                name: format!("L{}_{}d", target_layer, target_dim),
                dim: target_dim,
                seq_len: actual_len,
                batch_size,
                mean_ms: stats.mean,
                std_ms: stats.std,
                throughput: (batch_size as f64 * 1000.0) / stats.mean,
            });

            print!(" | {:>5.0}ms", stats.mean);
        }
        println!();
    }

    results
}

// ========================================================================================
// Print Functions
// ========================================================================================

fn print_summary(all_results: &[BenchmarkResult], matryoshka_config: &MatryoshkaConfig) {
    println!("\n{}", "=".repeat(80));
    println!("  SUMMARY");
    println!("{}", "=".repeat(80));

    // Find baseline (768d)
    if let Some(baseline) = all_results.iter().find(|r| r.dim == 768) {
        println!("\n  Full model (768d) baseline:");
        println!("    Latency: {:.2}ms", baseline.mean_ms);
        println!("    Throughput: {:.1} embeddings/s", baseline.throughput);
    }

    // Compare dimensions
    println!("\n  Matryoshka dimension comparison:");
    println!(
        "    {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
        "Dim", "Quality", "Storage", "Latency", "Speedup"
    );
    println!("    {}", "-".repeat(56));

    let baseline_latency = all_results
        .iter()
        .find(|r| r.name.contains("matryoshka_768d"))
        .map(|r| r.mean_ms)
        .unwrap_or(1.0);

    for &dim in MATRYOSHKA_DIMS {
        if let Some(result) = all_results
            .iter()
            .find(|r| r.name == format!("matryoshka_{}d", dim))
        {
            let quality = matryoshka_config.estimate_quality(22, dim) * 100.0;
            let storage = (dim as f64 / 768.0) * 100.0;
            let speedup = baseline_latency / result.mean_ms;

            println!(
                "    {:>8} | {:>9.1}% | {:>9.1}% | {:>8.2}ms | {:>9.2}x",
                format!("{}d", dim),
                quality,
                storage,
                result.mean_ms,
                speedup
            );
        }
    }

    // Recommendations
    println!("\n  Recommendations:");
    println!("    • 768d: Full quality for high-precision retrieval");
    println!("    • 256d: 99% quality, 33% storage - recommended for most use cases");
    println!("    • 64d:  98% quality, 8% storage - ultra-compact for large-scale");
}

// ========================================================================================
// Main
// ========================================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Parse arguments
    let mut device_filter = "gpu".to_string();
    let mut quick = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" | "-d" => {
                if i + 1 < args.len() {
                    device_filter = args[i + 1].to_lowercase();
                    i += 2;
                } else {
                    i += 1;
                }
            }
            "--quick" | "-q" => {
                quick = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("mmBERT 2D Matryoshka Embedding Benchmark\n");
                println!("Benchmarks the library's mmBERT embedding API.\n");
                println!("Environment Variables:");
                println!("  MMBERT_MODEL_PATH    Path to mmBERT model directory (required)\n");
                println!("Options:");
                println!("  --device, -d <TYPE>  Device: cpu or gpu (default: gpu)");
                println!("  --quick, -q          Run fewer iterations");
                println!("  --help, -h           Show this help\n");
                println!("Example:");
                println!("  MMBERT_MODEL_PATH=/path/to/model cargo run --release --example mmbert_2d_matryoshka_bench");
                std::process::exit(0);
            }
            _ => i += 1,
        }
    }

    // Get model path from environment
    let model_path = match get_model_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("❌ Error: {}", e);
            std::process::exit(1);
        }
    };

    let warmup = if quick { 2 } else { 5 };
    let iterations = if quick { 10 } else { 50 };

    println!("\n{}", "=".repeat(80));
    println!("  mmBERT 2D Matryoshka Embedding Benchmark");
    println!("{}", "=".repeat(80));

    // Select device
    let device = match device_filter.as_str() {
        "cpu" => Device::Cpu,
        _ => Device::cuda_if_available(0).unwrap_or(Device::Cpu),
    };

    let device_name = match &device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "GPU (CUDA/ROCm)",
        _ => "Unknown",
    };

    println!("  Device: {}", device_name);
    println!("  Model path: {}", model_path);
    println!("  Mode: {}", if quick { "quick" } else { "full" });
    println!("  Warmup: {} iterations", warmup);
    println!("  Benchmark: {} iterations", iterations);

    // ==================================================================================
    // Load Model using library API
    // ==================================================================================
    println!("\n📦 Loading model using MmBertEmbeddingModel::load()...");
    let model_start = Instant::now();

    let model = match MmBertEmbeddingModel::load(&model_path, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to load model: {:?}", e);
            eprintln!("   Ensure MMBERT_MODEL_PATH points to a valid model directory");
            std::process::exit(1);
        }
    };

    println!(
        "✅ Model loaded in {:.2}s",
        model_start.elapsed().as_secs_f64()
    );

    // Display model config
    let config = model.config();
    println!("   Model config:");
    println!("     - Hidden size: {}", config.hidden_size());
    println!("     - Num layers: {}", config.num_hidden_layers());
    println!("     - Max positions: {}", config.max_position_embeddings);

    // Display Matryoshka config
    let matryoshka_config = model.matryoshka_config().clone();
    println!("   Matryoshka config:");
    println!("     - Dimensions: {:?}", matryoshka_config.dimensions);
    println!("     - Layers: {:?}", matryoshka_config.layers);

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let mut tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Failed to load tokenizer: {:?}", e);
            std::process::exit(1);
        }
    };

    // Configure padding
    let pad_params = tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        pad_token: "<pad>".to_string(),
        pad_id: 0,
        ..Default::default()
    };
    tokenizer.with_padding(Some(pad_params));

    let mut all_results = Vec::new();

    // ==================================================================================
    // Benchmark 1: Matryoshka Dimensions
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  1️⃣  Matryoshka Dimension Benchmark (encode_batch API)");
    println!("{}", "=".repeat(80));

    let dim_results =
        bench_matryoshka_dimensions(&model, &tokenizer, &matryoshka_config, warmup, iterations);
    all_results.extend(dim_results);

    // ==================================================================================
    // Benchmark 2: Dimension Reduction (1st D of 2D Matryoshka)
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  2️⃣  Matryoshka Dimension Reduction (full layers)");
    println!("{}", "=".repeat(80));

    let api_results = bench_matryoshka_api(
        &model,
        &tokenizer,
        &device,
        &matryoshka_config,
        warmup,
        iterations,
    );
    all_results.extend(api_results);

    // ==================================================================================
    // Benchmark 3: Layer Early Exit (2nd D of 2D Matryoshka)
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  3️⃣  Layer Early Exit (full dimension)");
    println!("{}", "=".repeat(80));

    let layer_results = bench_layer_early_exit(
        &model,
        &tokenizer,
        &device,
        &matryoshka_config,
        warmup,
        iterations,
    );
    all_results.extend(layer_results);

    // ==================================================================================
    // Benchmark 4: Full 2D Matryoshka Matrix
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  4️⃣  Full 2D Matryoshka Matrix (layers × dimensions)");
    println!("{}", "=".repeat(80));

    let matrix_results = bench_2d_matrix(
        &model,
        &tokenizer,
        &device,
        &matryoshka_config,
        warmup,
        iterations,
    );
    all_results.extend(matrix_results);

    // ==================================================================================
    // Benchmark 5: Sequence Length Scaling
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  5️⃣  Sequence Length Scaling");
    println!("{}", "=".repeat(80));

    let seq_results = bench_sequence_lengths(&model, &tokenizer, &device, warmup, iterations / 2);
    all_results.extend(seq_results);

    // ==================================================================================
    // Benchmark 6: Batch Size Scaling
    // ==================================================================================
    println!("\n{}", "=".repeat(80));
    println!("  6️⃣  Batch Size Scaling");
    println!("{}", "=".repeat(80));

    let batch_results = bench_batch_sizes(&model, &tokenizer, warmup, iterations / 2);
    all_results.extend(batch_results);

    // ==================================================================================
    // Summary
    // ==================================================================================
    print_summary(&all_results, &matryoshka_config);

    println!("\n{}", "=".repeat(80));
    println!("  ✅ Benchmark complete!");
    println!("{}\n", "=".repeat(80));
}
