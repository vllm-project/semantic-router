//! Comprehensive benchmark: CPU vs GPU across layers, seq lengths, and batch sizes
//!
//! Usage: benchmark_cpu_vs_gpu <model_base_dir>
//! Example: benchmark_cpu_vs_gpu ./mmbert-2d-matryoshka
//!
//! This will benchmark all layer models in the directory.

use std::env;
use std::path::Path;
use std::time::{Duration, Instant};
use ort::session::Session;
use ort::value::Tensor;

struct BenchmarkResult {
    layer: usize,
    seq_len: usize,
    batch_size: usize,
    cpu_latency_ms: f64,
    gpu_latency_ms: f64,
    speedup: f64,
}

fn create_session(model_path: &Path, use_gpu: bool) -> Result<Session, Box<dyn std::error::Error>> {
    if use_gpu {
        #[cfg(feature = "rocm")]
        {
            use ort::execution_providers::ROCmExecutionProvider;
            Ok(Session::builder()?
                .with_execution_providers([ROCmExecutionProvider::default().build()])?
                .commit_from_file(model_path)?)
        }
        #[cfg(not(feature = "rocm"))]
        {
            Err("ROCm feature not enabled".into())
        }
    } else {
        Ok(Session::builder()?.commit_from_file(model_path)?)
    }
}

fn benchmark_inference(
    session: &mut Session,
    batch_size: usize,
    seq_len: usize,
    warmup_runs: usize,
    bench_runs: usize,
) -> Duration {
    // Create input tensors
    let input_ids: Vec<i64> = vec![101; batch_size * seq_len]; // dummy tokens
    let attention_mask: Vec<i64> = vec![1; batch_size * seq_len];

    // Warmup
    for _ in 0..warmup_runs {
        let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids.clone())).unwrap();
        let attention_mask_tensor = Tensor::from_array(([batch_size, seq_len], attention_mask.clone())).unwrap();
        let _ = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ]).unwrap();
    }

    // Benchmark
    let mut total = Duration::ZERO;
    for _ in 0..bench_runs {
        let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids.clone())).unwrap();
        let attention_mask_tensor = Tensor::from_array(([batch_size, seq_len], attention_mask.clone())).unwrap();

        let start = Instant::now();
        let _ = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
        ]).unwrap();
        total += start.elapsed();
    }

    total / bench_runs as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("================================================================================");
    println!("           CPU vs GPU Benchmark - 2D Matryoshka ONNX Models");
    println!("================================================================================\n");

    let args: Vec<String> = env::args().collect();
    let model_base = args.get(1).map(|s| s.as_str()).unwrap_or("./mmbert-2d-matryoshka");

    println!("Model directory: {}\n", model_base);

    ort::init().commit();

    // Configuration
    let layers = vec![6, 11, 16, 22];
    let seq_lengths = vec![32, 64, 128, 256, 512];
    let batch_sizes = vec![1, 4, 8];
    let warmup_runs = 3;
    let bench_runs = 10;

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // Check which layers are available
    let available_layers: Vec<usize> = layers.iter()
        .filter(|&l| Path::new(model_base).join(format!("layer-{}", l)).join("model.onnx").exists())
        .copied()
        .collect();

    if available_layers.is_empty() {
        println!("No layer models found in {}. Expected structure:", model_base);
        println!("  {}/layer-6/model.onnx", model_base);
        println!("  {}/layer-22/model.onnx", model_base);
        return Ok(());
    }

    println!("Available layers: {:?}", available_layers);
    println!("Sequence lengths: {:?}", seq_lengths);
    println!("Batch sizes: {:?}", batch_sizes);
    println!("Warmup runs: {}, Benchmark runs: {}\n", warmup_runs, bench_runs);

    // Run benchmarks
    for &layer in &available_layers {
        let model_path = Path::new(model_base).join(format!("layer-{}", layer)).join("model.onnx");

        println!("--------------------------------------------------------------------------------");
        println!("Layer {}: {}", layer, model_path.display());
        println!("--------------------------------------------------------------------------------");

        // Load CPU session
        print!("  Loading CPU session... ");
        let mut cpu_session = match create_session(&model_path, false) {
            Ok(s) => { println!("OK"); s }
            Err(e) => { println!("FAILED: {}", e); continue; }
        };

        // Load GPU session
        print!("  Loading GPU session... ");
        let mut gpu_session = match create_session(&model_path, true) {
            Ok(s) => { println!("OK"); s }
            Err(e) => { println!("FAILED: {}", e); continue; }
        };

        println!();
        println!("  {:>6} {:>6} {:>12} {:>12} {:>10}", "SeqLen", "Batch", "CPU (ms)", "GPU (ms)", "Speedup");
        println!("  {:->6} {:->6} {:->12} {:->12} {:->10}", "", "", "", "", "");

        for &seq_len in &seq_lengths {
            for &batch_size in &batch_sizes {
                // CPU benchmark
                let cpu_latency = benchmark_inference(
                    &mut cpu_session,
                    batch_size,
                    seq_len,
                    warmup_runs,
                    bench_runs
                );

                // GPU benchmark
                let gpu_latency = benchmark_inference(
                    &mut gpu_session,
                    batch_size,
                    seq_len,
                    warmup_runs,
                    bench_runs
                );

                let cpu_ms = cpu_latency.as_secs_f64() * 1000.0;
                let gpu_ms = gpu_latency.as_secs_f64() * 1000.0;
                let speedup = cpu_ms / gpu_ms;

                println!("  {:>6} {:>6} {:>12.2} {:>12.2} {:>9.2}x",
                    seq_len, batch_size, cpu_ms, gpu_ms, speedup);

                results.push(BenchmarkResult {
                    layer,
                    seq_len,
                    batch_size,
                    cpu_latency_ms: cpu_ms,
                    gpu_latency_ms: gpu_ms,
                    speedup,
                });
            }
        }
        println!();
    }

    // Summary tables
    println!("\n================================================================================");
    println!("                              SUMMARY TABLES");
    println!("================================================================================\n");

    // GPU Latency by Layer and SeqLen (batch=1)
    println!("GPU Latency (ms) - Batch=1:");
    print!("| SeqLen |");
    for l in &available_layers {
        print!(" Layer {:>2} |", l);
    }
    println!();
    print!("|--------|");
    for _ in &available_layers {
        print!("----------|");
    }
    println!();

    for &seq_len in &seq_lengths {
        print!("| {:>6} |", seq_len);
        for &layer in &available_layers {
            if let Some(r) = results.iter().find(|r| r.layer == layer && r.seq_len == seq_len && r.batch_size == 1) {
                print!(" {:>8.2} |", r.gpu_latency_ms);
            } else {
                print!("      N/A |");
            }
        }
        println!();
    }

    // GPU Speedup vs CPU by Layer and SeqLen (batch=1)
    println!("\nGPU Speedup vs CPU - Batch=1:");
    print!("| SeqLen |");
    for l in &available_layers {
        print!(" Layer {:>2} |", l);
    }
    println!();
    print!("|--------|");
    for _ in &available_layers {
        print!("----------|");
    }
    println!();

    for &seq_len in &seq_lengths {
        print!("| {:>6} |", seq_len);
        for &layer in &available_layers {
            if let Some(r) = results.iter().find(|r| r.layer == layer && r.seq_len == seq_len && r.batch_size == 1) {
                print!(" {:>7.2}x |", r.speedup);
            } else {
                print!("      N/A |");
            }
        }
        println!();
    }

    // Throughput comparison (batch=8)
    println!("\nThroughput (samples/sec) - Batch=8:");
    print!("| SeqLen |");
    for l in &available_layers {
        print!(" L{:>2} CPU | L{:>2} GPU |", l, l);
    }
    println!();
    print!("|--------|");
    for _ in &available_layers {
        print!("---------|---------|");
    }
    println!();

    for &seq_len in &seq_lengths {
        print!("| {:>6} |", seq_len);
        for &layer in &available_layers {
            if let Some(r) = results.iter().find(|r| r.layer == layer && r.seq_len == seq_len && r.batch_size == 8) {
                let cpu_throughput = 8000.0 / r.cpu_latency_ms;
                let gpu_throughput = 8000.0 / r.gpu_latency_ms;
                print!(" {:>7.0} | {:>7.0} |", cpu_throughput, gpu_throughput);
            } else {
                print!("     N/A |     N/A |");
            }
        }
        println!();
    }

    // Best configurations
    println!("\n================================================================================");
    println!("                           RECOMMENDATIONS");
    println!("================================================================================\n");

    // Find best GPU speedup
    if let Some(best) = results.iter().max_by(|a, b| a.speedup.partial_cmp(&b.speedup).unwrap()) {
        println!("Best GPU speedup: {:.2}x (Layer {}, SeqLen {}, Batch {})",
            best.speedup, best.layer, best.seq_len, best.batch_size);
    }

    // Find fastest GPU config
    if let Some(fastest) = results.iter().min_by(|a, b| a.gpu_latency_ms.partial_cmp(&b.gpu_latency_ms).unwrap()) {
        println!("Lowest GPU latency: {:.2}ms (Layer {}, SeqLen {}, Batch {})",
            fastest.gpu_latency_ms, fastest.layer, fastest.seq_len, fastest.batch_size);
    }

    // Layer early-exit benefit
    let layer22_results: Vec<_> = results.iter().filter(|r| r.layer == 22).collect();
    let layer6_results: Vec<_> = results.iter().filter(|r| r.layer == 6).collect();

    if !layer22_results.is_empty() && !layer6_results.is_empty() {
        println!("\nLayer Early-Exit Benefit (Layer 6 vs 22):");
        for &seq_len in &[64, 256, 512] {
            for &batch_size in &[1, 8] {
                if let (Some(l22), Some(l6)) = (
                    layer22_results.iter().find(|r| r.seq_len == seq_len && r.batch_size == batch_size),
                    layer6_results.iter().find(|r| r.seq_len == seq_len && r.batch_size == batch_size),
                ) {
                    let layer_speedup = l22.gpu_latency_ms / l6.gpu_latency_ms;
                    println!("  SeqLen={}, Batch={}: {:.2}x faster with Layer 6",
                        seq_len, batch_size, layer_speedup);
                }
            }
        }
    }

    println!("\n✓ Benchmark complete!");
    Ok(())
}
