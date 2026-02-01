/// mmBERT-32K Scalability Benchmark
///
/// Investigates scalability issues with long sequences and batch sizes on CPU.
/// Tests multi-core utilization, continuous batching, and AVX512 performance.
///
/// # Environment Variables
/// - `MMBERT_32K_MODEL_PATH`: Path to mmBERT-32K classifier model (required)
/// - `RAYON_NUM_THREADS`: Number of threads for parallel processing (optional)
///
/// # Usage with AVX512 (RECOMMENDED)
/// ```bash
/// # Run with AVX512 enabled (best performance on Intel/AMD with AVX512)
/// cd candle-binding && \
///   MMBERT_32K_MODEL_PATH=../models/mmbert32k-intent-classifier-merged \
///   RUSTFLAGS="-C target-cpu=native -C target-feature=+avx512f,+avx512bw,+avx512vl" \
///   cargo run --release --no-default-features --example mmbert_32k_scalability_bench
///
/// # Or use the Makefile target:
/// make bench-mmbert32k-scalability
/// ```
///
/// # Usage with Intel MKL (alternative for Intel CPUs)
/// ```bash
/// # Build with MKL support for optimized BLAS operations
/// cd candle-binding && \
///   MMBERT_32K_MODEL_PATH=../models/mmbert32k-intent-classifier-merged \
///   RUSTFLAGS="-C target-cpu=native" \
///   cargo run --release --no-default-features --features mkl --example mmbert_32k_scalability_bench
/// ```
///
/// # Quick test mode
/// ```bash
/// MMBERT_32K_MODEL_PATH=models/mmbert32k-intent-classifier-merged \
///   RUSTFLAGS="-C target-cpu=native" \
///   cargo run --release --no-default-features --example mmbert_32k_scalability_bench -- --quick
/// ```
///
/// # Test specific sequence lengths
/// ```bash
/// MMBERT_32K_MODEL_PATH=models/mmbert32k-intent-classifier-merged \
///   RUSTFLAGS="-C target-cpu=native" \
///   cargo run --release --no-default-features --example mmbert_32k_scalability_bench -- --seq-len 4096
/// ```
///
/// # With specific thread count
/// ```bash
/// RAYON_NUM_THREADS=8 MMBERT_32K_MODEL_PATH=models/mmbert32k-intent-classifier-merged \
///   RUSTFLAGS="-C target-cpu=native" \
///   cargo run --release --no-default-features --example mmbert_32k_scalability_bench
/// ```
///
/// # AVX512 Verification
/// The benchmark will detect and report CPU SIMD capabilities (SSE4, AVX2, AVX512).
/// For optimal performance on modern Intel/AMD CPUs, AVX512 should be enabled.
///
/// # Single-Inference Multi-Core (BLAS Threading)
/// By default, Candle uses rayon for GEMM operations which can parallelize
/// matrix multiplications within a single inference call.
///
/// Control threading with environment variables:
/// ```bash
/// # Set rayon thread pool size (affects GEMM parallelism)
/// RAYON_NUM_THREADS=8 cargo run --example mmbert_32k_scalability_bench
///
/// # For MKL builds, also set:
/// MKL_NUM_THREADS=8 OMP_NUM_THREADS=8 cargo run --features mkl --example mmbert_32k_scalability_bench
/// ```
///
/// Key performance factors:
/// - AVX512: 2-4x speedup for matrix operations vs AVX2
/// - BLAS threading: Parallelizes large matrix ops within single inference
/// - Batch parallelism: Process multiple requests concurrently
/// - MKL: Additional 20-50% speedup for Intel CPUs with optimized BLAS
use candle_core::DType;
use candle_semantic_router::model_architectures::traditional::modernbert::{
    ModernBertVariant, TraditionalModernBertClassifier,
};
use rayon::prelude::*;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// Import gemm for threading threshold configuration
// The gemm crate has DEFAULT_THREADING_THRESHOLD = 48*48*256 = ~590K ops
// which may be too high for small-batch inference parallelism

// ========================================================================================
// Configuration
// ========================================================================================

/// Sequence lengths to benchmark (tokens) - FULL mode
const SEQ_LENGTHS_FULL: &[usize] = &[128, 256, 512, 1024];

/// Sequence lengths for QUICK mode (shorter for faster testing)
const SEQ_LENGTHS_QUICK: &[usize] = &[128, 256, 512, 1024];

/// Batch sizes to benchmark
const BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16, 32, 64];

/// Thread counts to test for multi-core scaling
const THREAD_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32];

/// Number of warmup iterations
const WARMUP_ITERATIONS: usize = 3;

/// Number of benchmark iterations
const BENCH_ITERATIONS: usize = 10;

/// Sample texts for different categories
const SAMPLE_TEXTS: &[&str] = &[
    "What is photosynthesis and how does it work in plants?",
    "Explain the theory of relativity in simple terms.",
    "How do neural networks learn from data?",
    "What are the main principles of economics?",
    "Describe the structure of a cell.",
    "What is the Pythagorean theorem?",
    "How does quantum computing differ from classical computing?",
    "Explain the concept of supply and demand.",
];

// ========================================================================================
// CPU Feature Detection
// ========================================================================================

fn detect_cpu_features() {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         CPU SIMD FEATURE DETECTION                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    #[cfg(target_arch = "x86_64")]
    {
        let has_sse4_1 = is_x86_feature_detected!("sse4.1");
        let has_sse4_2 = is_x86_feature_detected!("sse4.2");
        let has_avx = is_x86_feature_detected!("avx");
        let has_avx2 = is_x86_feature_detected!("avx2");
        let has_avx512f = is_x86_feature_detected!("avx512f");
        let has_avx512bw = is_x86_feature_detected!("avx512bw");
        let has_avx512vl = is_x86_feature_detected!("avx512vl");
        let has_fma = is_x86_feature_detected!("fma");

        println!("  SIMD Features:");
        println!(
            "    SSE4.1:    {} {}",
            if has_sse4_1 { "✅" } else { "❌" },
            if has_sse4_1 { "enabled" } else { "disabled" }
        );
        println!(
            "    SSE4.2:    {} {}",
            if has_sse4_2 { "✅" } else { "❌" },
            if has_sse4_2 { "enabled" } else { "disabled" }
        );
        println!(
            "    AVX:       {} {}",
            if has_avx { "✅" } else { "❌" },
            if has_avx { "enabled" } else { "disabled" }
        );
        println!(
            "    AVX2:      {} {}",
            if has_avx2 { "✅" } else { "❌" },
            if has_avx2 { "enabled" } else { "disabled" }
        );
        println!(
            "    FMA:       {} {}",
            if has_fma { "✅" } else { "❌" },
            if has_fma { "enabled" } else { "disabled" }
        );
        println!(
            "    AVX-512F:  {} {}",
            if has_avx512f { "✅" } else { "❌" },
            if has_avx512f { "enabled" } else { "disabled" }
        );
        println!(
            "    AVX-512BW: {} {}",
            if has_avx512bw { "✅" } else { "❌" },
            if has_avx512bw { "enabled" } else { "disabled" }
        );
        println!(
            "    AVX-512VL: {} {}",
            if has_avx512vl { "✅" } else { "❌" },
            if has_avx512vl { "enabled" } else { "disabled" }
        );

        if has_avx512f && has_avx512bw && has_avx512vl {
            println!("\n  ✅ Full AVX-512 support detected - optimal for large matrix operations");
        } else if has_avx2 {
            println!("\n  ⚠️  AVX2 detected but no AVX-512 - consider using AVX-512 capable CPU");
        } else {
            println!("\n  ❌ Limited SIMD support - performance may be suboptimal");
        }

        // Check if compiled with native CPU optimizations
        println!("\n  Build Configuration:");
        println!("    Tip: For best performance, build with:");
        println!("    RUSTFLAGS=\"-C target-cpu=native\" cargo build --release");
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("  Non-x86_64 architecture detected");
        println!("  SIMD feature detection not available");
    }
}

// ========================================================================================
// Results Structures
// ========================================================================================

#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    seq_len: usize,
    batch_size: usize,
    threads: usize,
    mean_ms: f64,
    std_ms: f64,
    min_ms: f64,
    max_ms: f64,
    throughput_samples_per_sec: f64,
    throughput_tokens_per_sec: f64,
}

#[derive(Debug)]
struct ScalabilityAnalysis {
    seq_len_scaling: Vec<(usize, f64)>, // (seq_len, time_ratio vs baseline)
    batch_scaling: Vec<(usize, f64)>,   // (batch_size, efficiency)
    thread_scaling: Vec<(usize, f64)>,  // (threads, speedup)
}

fn compute_stats(times: &[f64]) -> (f64, f64, f64, f64) {
    let n = times.len() as f64;
    let mean = times.iter().sum::<f64>() / n;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, std, min, max)
}

// ========================================================================================
// Text Generation
// ========================================================================================

/// Generate text of approximately the specified token count
fn generate_text(approx_tokens: usize) -> String {
    let base_text = SAMPLE_TEXTS.join(" ");
    // Rough estimate: 1.3 tokens per word
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

/// Benchmark inference at various sequence lengths (single-threaded baseline)
fn bench_sequence_length_scaling(
    model: Arc<TraditionalModernBertClassifier>,
    warmup: usize,
    iterations: usize,
    quick_mode: bool,
) -> Vec<BenchmarkResult> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SEQUENCE LENGTH SCALING (Single-Thread)                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let seq_lengths = if quick_mode {
        SEQ_LENGTHS_QUICK
    } else {
        SEQ_LENGTHS_FULL
    };
    println!(
        "  Testing {} sequence lengths: {:?}\n",
        seq_lengths.len(),
        seq_lengths
    );

    let mut results = Vec::new();

    for &seq_len in seq_lengths {
        let text = generate_text(seq_len);

        // Single-threaded warmup
        for _ in 0..warmup {
            let _ = model.classify_text(&text);
        }

        // Benchmark single-threaded inference
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = model.classify_text(&text);
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let (mean, std, min, max) = compute_stats(&times);
        let throughput_samples = 1000.0 / mean;
        let throughput_tokens = (seq_len as f64) * throughput_samples;

        println!(
            "  seq_len={:>6}: mean={:>8.2}ms, std={:>6.2}ms, throughput={:>6.1} samples/s, {:>10.0} tok/s",
            seq_len, mean, std, throughput_samples, throughput_tokens
        );

        results.push(BenchmarkResult {
            name: format!("seq_len_{}", seq_len),
            seq_len,
            batch_size: 1,
            threads: 1,
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            throughput_samples_per_sec: throughput_samples,
            throughput_tokens_per_sec: throughput_tokens,
        });

        // Skip very long sequences if they're too slow
        if mean > 30000.0 {
            println!(
                "    ⚠️  Skipping longer sequences (>{:.0}s per inference)",
                mean / 1000.0
            );
            break;
        }
    }

    results
}

/// Benchmark batch processing efficiency
fn bench_batch_scaling(
    model: &TraditionalModernBertClassifier,
    seq_len: usize,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║                    BATCH SIZE SCALING (seq_len={:>5})                        ║",
        seq_len
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let text = generate_text(seq_len);
    let mut results = Vec::new();

    // Get single-item baseline
    for _ in 0..warmup {
        let _ = model.classify_text(&text);
    }
    let mut baseline_times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = model.classify_text(&text);
        baseline_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let (baseline_mean, _, _, _) = compute_stats(&baseline_times);
    println!("  Baseline (batch=1): {:.2}ms per sample\n", baseline_mean);

    for &batch_size in BATCH_SIZES {
        let texts: Vec<&str> = (0..batch_size).map(|_| text.as_str()).collect();

        // Warmup
        for _ in 0..warmup {
            for t in &texts {
                let _ = model.classify_text(t);
            }
        }

        // Benchmark - process batch sequentially (simulating batch processing)
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let start = Instant::now();
            for t in &texts {
                let _ = model.classify_text(t);
            }
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let (mean, std, min, max) = compute_stats(&times);
        let per_sample = mean / batch_size as f64;
        let efficiency = (baseline_mean / per_sample) * 100.0;
        let throughput_samples = (batch_size as f64 * 1000.0) / mean;
        let throughput_tokens = throughput_samples * seq_len as f64;

        println!(
            "  batch={:>3}: total={:>8.2}ms, per_sample={:>6.2}ms, efficiency={:>5.1}%, throughput={:>6.1} samples/s",
            batch_size, mean, per_sample, efficiency, throughput_samples
        );

        results.push(BenchmarkResult {
            name: format!("batch_{}", batch_size),
            seq_len,
            batch_size,
            threads: 1,
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            throughput_samples_per_sec: throughput_samples,
            throughput_tokens_per_sec: throughput_tokens,
        });
    }

    results
}

/// Benchmark multi-core parallel processing with rayon
fn bench_multicore_scaling(
    model: Arc<TraditionalModernBertClassifier>,
    seq_len: usize,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║               MULTI-CORE SCALING (seq_len={:>5}, batch={:>3})                 ║",
        seq_len, batch_size
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let text = generate_text(seq_len);
    let texts: Vec<String> = (0..batch_size).map(|_| text.clone()).collect();
    let mut results = Vec::new();

    // Get single-threaded baseline first
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            for _ in 0..warmup {
                texts.par_iter().for_each(|t| {
                    let _ = model.classify_text(t);
                });
            }
        });

    let baseline_mean = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            let mut times = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                texts.par_iter().for_each(|t| {
                    let _ = model.classify_text(t);
                });
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let (mean, _, _, _) = compute_stats(&times);
            mean
        });

    println!(
        "  Single-thread baseline: {:.2}ms for {} samples\n",
        baseline_mean, batch_size
    );

    let available_threads = rayon::current_num_threads();
    println!("  Available threads: {}\n", available_threads);

    for &num_threads in THREAD_COUNTS {
        if num_threads > available_threads {
            println!(
                "  threads={:>2}: SKIPPED (only {} available)",
                num_threads, available_threads
            );
            continue;
        }

        let model_clone = Arc::clone(&model);
        let texts_clone = texts.clone();

        // Build thread pool with specific number of threads
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // Warmup in the pool
        pool.install(|| {
            for _ in 0..warmup {
                texts_clone.par_iter().for_each(|t| {
                    let _ = model_clone.classify_text(t);
                });
            }
        });

        // Benchmark
        let mut times = Vec::with_capacity(iterations);
        let processed = Arc::new(AtomicUsize::new(0));

        pool.install(|| {
            for _ in 0..iterations {
                processed.store(0, Ordering::SeqCst);
                let start = Instant::now();
                texts_clone.par_iter().for_each(|t| {
                    let _ = model_clone.classify_text(t);
                    processed.fetch_add(1, Ordering::SeqCst);
                });
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        });

        let (mean, std, min, max) = compute_stats(&times);
        let speedup = baseline_mean / mean;
        let efficiency = (speedup / num_threads as f64) * 100.0;
        let throughput = (batch_size as f64 * 1000.0) / mean;

        println!(
            "  threads={:>2}: mean={:>8.2}ms, speedup={:>5.2}x, efficiency={:>5.1}%, throughput={:>6.1} samples/s",
            num_threads, mean, speedup, efficiency, throughput
        );

        results.push(BenchmarkResult {
            name: format!("threads_{}", num_threads),
            seq_len,
            batch_size,
            threads: num_threads,
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            throughput_samples_per_sec: throughput,
            throughput_tokens_per_sec: throughput * seq_len as f64,
        });
    }

    results
}

/// Benchmark continuous request processing (simulating real workload)
fn bench_continuous_throughput(
    model: Arc<TraditionalModernBertClassifier>,
    duration_secs: u64,
) -> f64 {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║                    CONTINUOUS THROUGHPUT TEST ({}s)                          ║",
        duration_secs
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let texts: Vec<String> = SEQ_LENGTHS_QUICK
        .iter()
        .take(5) // Use first 5 sequence lengths for variety
        .map(|&len| generate_text(len))
        .collect();

    let total_processed = Arc::new(AtomicUsize::new(0));
    let total_tokens = Arc::new(AtomicUsize::new(0));
    let deadline = Instant::now() + Duration::from_secs(duration_secs);

    println!(
        "  Running continuous workload for {}s with {} text variants...\n",
        duration_secs,
        texts.len()
    );

    let start = Instant::now();

    // Create a large workload for parallel processing
    let workload: Vec<&String> = texts.iter().cycle().take(10000).collect();

    // Use all available threads
    workload.par_iter().for_each(|text| {
        if Instant::now() >= deadline {
            return;
        }
        let _ = model.classify_text(text);
        total_processed.fetch_add(1, Ordering::Relaxed);
        total_tokens.fetch_add(text.split_whitespace().count(), Ordering::Relaxed);
    });

    let elapsed = start.elapsed().as_secs_f64();
    let processed = total_processed.load(Ordering::Relaxed);
    let tokens = total_tokens.load(Ordering::Relaxed);
    let throughput = processed as f64 / elapsed;
    let token_throughput = tokens as f64 / elapsed;

    println!("  Results:");
    println!("    Total samples processed: {}", processed);
    println!("    Total time: {:.2}s", elapsed);
    println!("    Throughput: {:.1} samples/s", throughput);
    println!("    Token throughput: {:.0} tokens/s", token_throughput);

    throughput
}

/// Benchmark single-inference BLAS threading
/// Tests if changing rayon thread pool affects single-inference latency
fn bench_single_inference_threading(
    model: Arc<TraditionalModernBertClassifier>,
    seq_len: usize,
    warmup: usize,
    iterations: usize,
) -> Vec<BenchmarkResult> {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║              SINGLE-INFERENCE THREADING (seq_len={:>5})                      ║",
        seq_len
    );
    println!("║              Testing BLAS/GEMM multi-core within one inference               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let text = generate_text(seq_len);
    let mut results = Vec::new();

    // Test with different thread pool sizes
    let thread_configs: &[usize] = &[1, 2, 4, 8, 16];
    let available_threads = rayon::current_num_threads();

    println!("  Default rayon threads: {}\n", available_threads);
    println!("  Testing if single inference benefits from BLAS threading:\n");

    for &num_threads in thread_configs {
        if num_threads > available_threads * 2 {
            continue;
        }

        // Build a new thread pool with specific size
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        // Run single inference in this pool context
        let times: Vec<f64> = pool.install(|| {
            // Warmup
            for _ in 0..warmup {
                let _ = model.classify_text(&text);
            }

            // Benchmark
            let mut times = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let start = Instant::now();
                let _ = model.classify_text(&text);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            times
        });

        let (mean, std, min, max) = compute_stats(&times);
        let throughput = 1000.0 / mean;

        // Compare to single-thread baseline
        let baseline = results
            .first()
            .map(|r: &BenchmarkResult| r.mean_ms)
            .unwrap_or(mean);
        let speedup = baseline / mean;

        println!(
            "  threads={:>2}: mean={:>8.2}ms, std={:>5.2}ms, speedup={:>5.2}x, throughput={:>5.1} samples/s",
            num_threads, mean, std, speedup, throughput
        );

        results.push(BenchmarkResult {
            name: format!("single_inf_threads_{}", num_threads),
            seq_len,
            batch_size: 1,
            threads: num_threads,
            mean_ms: mean,
            std_ms: std,
            min_ms: min,
            max_ms: max,
            throughput_samples_per_sec: throughput,
            throughput_tokens_per_sec: throughput * seq_len as f64,
        });
    }

    // Analysis
    if results.len() >= 2 {
        let single = &results[0];
        let best = results
            .iter()
            .min_by(|a, b| a.mean_ms.partial_cmp(&b.mean_ms).unwrap())
            .unwrap();

        println!("\n  Analysis:");
        if best.threads > 1 && single.mean_ms / best.mean_ms > 1.1 {
            println!(
                "    ✅ BLAS threading IS effective: {:.2}x speedup at {} threads",
                single.mean_ms / best.mean_ms,
                best.threads
            );
            println!("       Single inference uses multiple cores for matrix operations");
        } else {
            println!("    ⚠️  BLAS threading shows minimal benefit for this sequence length");
            println!("       Matrix operations may be too small to benefit from parallelism");
            println!("       Consider using batch parallelism instead (multiple requests)");
        }
    }

    results
}

/// Analyze scalability patterns
fn analyze_scalability(
    seq_results: &[BenchmarkResult],
    batch_results: &[BenchmarkResult],
    thread_results: &[BenchmarkResult],
) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         SCALABILITY ANALYSIS                                 ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Sequence length scaling analysis
    if !seq_results.is_empty() {
        let baseline = seq_results[0].mean_ms;
        println!(
            "  Sequence Length Scaling (relative to seq_len={}):",
            seq_results[0].seq_len
        );
        for r in seq_results {
            let ratio = r.mean_ms / baseline;
            let expected = (r.seq_len as f64 / seq_results[0].seq_len as f64).powi(2); // O(n²) for attention
            let efficiency = (expected / ratio) * 100.0;
            println!(
                "    seq_len={:>6}: {:.2}x slower (expected ~{:.1}x for O(n²)), efficiency={:.1}%",
                r.seq_len, ratio, expected, efficiency
            );
        }
        println!();
    }

    // Batch scaling analysis
    if !batch_results.is_empty() {
        println!("  Batch Processing Efficiency:");
        let single_throughput = batch_results
            .iter()
            .find(|r| r.batch_size == 1)
            .map(|r| r.throughput_samples_per_sec)
            .unwrap_or(1.0);

        for r in batch_results {
            let scaling_efficiency =
                r.throughput_samples_per_sec / (single_throughput * r.batch_size as f64) * 100.0;
            println!(
                "    batch={:>3}: throughput={:.1} samples/s, scaling_efficiency={:.1}%",
                r.batch_size, r.throughput_samples_per_sec, scaling_efficiency
            );
        }
        println!();
    }

    // Multi-core scaling analysis
    if !thread_results.is_empty() {
        println!("  Multi-Core Scaling:");
        let single_thread = thread_results
            .iter()
            .find(|r| r.threads == 1)
            .map(|r| r.mean_ms)
            .unwrap_or(1.0);

        for r in thread_results {
            let speedup = single_thread / r.mean_ms;
            let ideal_speedup = r.threads as f64;
            let parallel_efficiency = (speedup / ideal_speedup) * 100.0;
            println!(
                "    threads={:>2}: speedup={:.2}x (ideal={:.1}x), parallel_efficiency={:.1}%",
                r.threads, speedup, ideal_speedup, parallel_efficiency
            );
        }

        // Identify bottlenecks
        if let (Some(single), Some(max)) = (
            thread_results.iter().find(|r| r.threads == 1),
            thread_results.iter().max_by_key(|r| r.threads),
        ) {
            let actual_speedup = single.mean_ms / max.mean_ms;
            let max_threads = max.threads as f64;

            println!("\n  Bottleneck Analysis:");
            if actual_speedup < max_threads * 0.5 {
                println!("    ⚠️  Significant parallel overhead detected");
                println!("       Possible causes:");
                println!("       - Memory bandwidth saturation");
                println!("       - Lock contention in model inference");
                println!("       - Cache coherency overhead");
            } else if actual_speedup < max_threads * 0.8 {
                println!("    ⚠️  Moderate parallel overhead");
                println!(
                    "       Performance scaling at {:.1}% of ideal",
                    (actual_speedup / max_threads) * 100.0
                );
            } else {
                println!(
                    "    ✅ Good parallel scaling: {:.1}% of ideal speedup",
                    (actual_speedup / max_threads) * 100.0
                );
            }
        }
    }
}

// ========================================================================================
// Main
// ========================================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║           mmBERT-32K SCALABILITY BENCHMARK                                   ║");
    println!("║           Investigating long sequence and batch size performance             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let quick_mode = args.contains(&"--quick".to_string());
    let specific_seq_len: Option<usize> = args
        .iter()
        .position(|a| a == "--seq-len")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok());

    // Parse dtype: --dtype f32|f16|bf16
    let dtype_str = args
        .iter()
        .position(|a| a == "--dtype")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.to_lowercase())
        .unwrap_or_else(|| "f32".to_string());

    let dtype = match dtype_str.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" | _ => DType::F32,
    };

    // Configuration
    let warmup = if quick_mode { 1 } else { WARMUP_ITERATIONS };
    let iterations = if quick_mode { 3 } else { BENCH_ITERATIONS };

    println!("  Configuration:");
    println!("    Mode: {}", if quick_mode { "quick" } else { "full" });
    println!("    Warmup iterations: {}", warmup);
    println!("    Benchmark iterations: {}", iterations);
    println!("    Data type: {:?}", dtype);
    if let Some(len) = specific_seq_len {
        println!("    Specific sequence length: {}", len);
    }
    println!("    Tip: Use --dtype bf16 or --dtype f16 for faster inference");

    // ==================================================================================
    // GEMM Threading Configuration
    // ==================================================================================
    // The gemm crate has a default threading threshold of 48*48*256 = 589,824 operations.
    // Matrices smaller than this threshold run single-threaded.
    // For inference workloads (especially attention), we may want to lower this.
    //
    // Key settings:
    // - threading_threshold: min ops for multi-threaded matmul (default ~600K)
    // - RAYON_NUM_THREADS: controls the rayon thread pool size (when parallelism kicks in)
    //
    println!("\n  GEMM Threading Configuration:");
    let default_threshold = gemm::get_threading_threshold();
    println!(
        "    Default threading threshold: {} ops ({:.1}M)",
        default_threshold,
        default_threshold as f64 / 1e6
    );

    // Check if user wants to force multi-threaded GEMM
    let force_mt_gemm = env::var("FORCE_MT_GEMM").is_ok();
    if force_mt_gemm {
        // Set threshold to 0 to always use multi-threading
        gemm::set_threading_threshold(0);
        println!("    ⚡ FORCE_MT_GEMM=1: Threading threshold set to 0 (always parallel)");
    } else {
        // Lower threshold to 8K ops for better parallelism on smaller matrices
        // This helps with attention computations which may be below default threshold
        let new_threshold = 8 * 1024;
        gemm::set_threading_threshold(new_threshold);
        println!(
            "    Threading threshold lowered to: {} ops ({:.1}K)",
            new_threshold,
            new_threshold as f64 / 1024.0
        );
        println!("    Tip: Set FORCE_MT_GEMM=1 to always use multi-threaded GEMM");
    }

    let rayon_threads = rayon::current_num_threads();
    println!("    Rayon thread pool: {} threads", rayon_threads);
    println!("    Tip: Set RAYON_NUM_THREADS=N to control thread pool size");

    // Detect CPU features
    detect_cpu_features();

    // Get model path
    let model_path = env::var("MMBERT_32K_MODEL_PATH").map_err(|_| {
        "MMBERT_32K_MODEL_PATH environment variable not set.\n\
         Usage: MMBERT_32K_MODEL_PATH=models/mmbert32k-intent-classifier-merged cargo run --example mmbert_32k_scalability_bench"
    })?;

    println!("\n  Loading model from: {}", model_path);
    println!("  Using dtype: {:?}", dtype);

    // Load model with specified dtype
    let start = Instant::now();
    let model = TraditionalModernBertClassifier::load_from_directory_with_variant_and_dtype(
        &model_path,
        true, // use_cpu
        ModernBertVariant::Multilingual32K,
        dtype,
    )?;
    let load_time = start.elapsed();

    println!("  Model loaded in {:.2}s", load_time.as_secs_f64());
    println!("  Variant: {:?}", model.variant());
    println!("  Num classes: {}", model.get_num_classes());

    // Calculate approximate model size based on dtype
    let size_multiplier = match dtype {
        DType::F32 => 4.0,
        DType::F16 | DType::BF16 => 2.0,
        _ => 4.0,
    };
    // mmBERT-32K has ~250M parameters
    let approx_size_mb = 250.0 * size_multiplier;
    println!(
        "  Approximate model size: {:.0} MB ({:?})",
        approx_size_mb, dtype
    );

    // Wrap in Arc for sharing across threads
    let model = Arc::new(model);

    // Run benchmarks
    let seq_results = if specific_seq_len.is_some() {
        vec![] // Skip if specific seq_len requested
    } else {
        bench_sequence_length_scaling(Arc::clone(&model), warmup, iterations, quick_mode)
    };

    let test_seq_len = specific_seq_len.unwrap_or(1024);
    let batch_results = bench_batch_scaling(&model, test_seq_len, warmup, iterations);

    let thread_results = bench_multicore_scaling(
        Arc::clone(&model),
        test_seq_len,
        32, // batch size for thread scaling
        warmup,
        iterations,
    );

    // Single-inference threading test (BLAS parallelism)
    let single_inf_results =
        bench_single_inference_threading(Arc::clone(&model), test_seq_len, warmup, iterations);

    // Also test with longer sequence to see if BLAS threading helps more
    if !quick_mode && specific_seq_len.is_none() {
        println!("\n  Testing single-inference threading with longer sequence (4096 tokens)...");
        let _ = bench_single_inference_threading(Arc::clone(&model), 4096, warmup, iterations);
    }

    // Continuous throughput test
    let continuous_throughput = if quick_mode {
        0.0 // Skip in quick mode
    } else {
        bench_continuous_throughput(Arc::clone(&model), 5)
    };

    // Analyze results
    analyze_scalability(&seq_results, &batch_results, &thread_results);

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              SUMMARY                                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    if !seq_results.is_empty() {
        let min_seq = seq_results.iter().min_by_key(|r| r.seq_len).unwrap();
        let max_seq = seq_results.iter().max_by_key(|r| r.seq_len).unwrap();
        println!(
            "  Sequence Length Range: {} - {} tokens",
            min_seq.seq_len, max_seq.seq_len
        );
        println!(
            "  Latency Range: {:.2}ms - {:.2}ms",
            min_seq.mean_ms, max_seq.mean_ms
        );
    }

    if !thread_results.is_empty() {
        let best = thread_results
            .iter()
            .max_by(|a, b| {
                a.throughput_samples_per_sec
                    .partial_cmp(&b.throughput_samples_per_sec)
                    .unwrap()
            })
            .unwrap();
        println!(
            "  Best multi-core config: {} threads, {:.1} samples/s",
            best.threads, best.throughput_samples_per_sec
        );
    }

    if continuous_throughput > 0.0 {
        println!(
            "  Sustained throughput: {:.1} samples/s",
            continuous_throughput
        );
    }

    println!("\n  ✅ Benchmark complete\n");

    Ok(())
}
