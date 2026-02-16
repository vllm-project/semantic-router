//! Concurrent Request Benchmark for ModernBERT-base-32k
//!
//! This script measures inference latency under concurrent load to identify
//! optimal chunking thresholds for production deployment.
//!
//! Usage:
//!   cargo run --example benchmark_concurrent --release --features cuda,flash-attn

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_semantic_router::model_architectures::traditional::modernbert::ModernBertVariant;
use candle_transformers::models::modernbert::{Config, ModernBert};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tokenizers::Tokenizer;

/// Latency statistics for concurrent requests
#[derive(Default, Clone)]
struct LatencyStats {
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    success_count: usize,
    error_count: usize,
}

impl LatencyStats {
    fn from_latencies(latencies: &[f64], error_count: usize) -> Self {
        if latencies.is_empty() {
            return Self {
                success_count: 0,
                error_count,
                ..Default::default()
            };
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
        let p50 = percentile(&sorted, 50.0);
        let p95 = percentile(&sorted, 95.0);
        let p99 = percentile(&sorted, 99.0);

        Self {
            mean_ms: mean,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            success_count: latencies.len(),
            error_count,
        }
    }

    fn print(&self, _concurrency: usize) {
        if self.success_count > 0 {
            println!(
                "mean={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms (success={}, errors={})",
                self.mean_ms,
                self.p50_ms,
                self.p95_ms,
                self.p99_ms,
                self.success_count,
                self.error_count
            );
        } else {
            println!("(failed: {} errors)", self.error_count);
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * p / 100.0) as usize;
    sorted[index]
}

/// Check GPU memory available
fn check_gpu_memory() -> Option<(f64, f64)> {
    use std::process::Command;

    if !cfg!(feature = "cuda") {
        return None;
    }

    let output = Command::new("nvidia-smi")
        .args(&[
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let line = stdout.lines().next()?;
    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

    if parts.len() >= 2 {
        let total_mb = parts[0].parse::<f64>().ok()?;
        let free_mb = parts[1].parse::<f64>().ok()?;
        let total_gb = total_mb / 1024.0;
        let free_gb = free_mb / 1024.0;
        return Some((total_gb, free_gb));
    }

    None
}

/// Create text with exact token count
fn create_text_with_exact_tokens(
    tokenizer: &Tokenizer,
    base_text: &str,
    target_tokens: usize,
) -> Result<String> {
    let encoding = tokenizer
        .encode(base_text, true)
        .map_err(|e| anyhow!("Failed to encode base text: {}", e))?;
    let tokens_per_repetition = encoding.get_ids().len();

    if tokens_per_repetition == 0 {
        return Err(anyhow!("Base text produces 0 tokens"));
    }

    let mut repetitions = (target_tokens / tokens_per_repetition).max(1);
    let mut low = 1;
    let mut high = repetitions * 2;

    while low <= high {
        repetitions = (low + high) / 2;
        let test_text = base_text.repeat(repetitions);

        let encoding = tokenizer
            .encode(test_text.as_str(), true)
            .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
        let actual_tokens = encoding.get_ids().len();

        if actual_tokens == target_tokens {
            return Ok(test_text);
        } else if actual_tokens < target_tokens {
            low = repetitions + 1;
        } else {
            high = repetitions - 1;
        }
    }

    repetitions = high;
    let mut test_text = base_text.repeat(repetitions);
    let encoding = tokenizer
        .encode(test_text.as_str(), true)
        .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
    let actual_tokens = encoding.get_ids().len();

    if actual_tokens < target_tokens {
        let padding = " word";
        loop {
            test_text = format!("{}{}", test_text, padding);
            let encoding = tokenizer
                .encode(test_text.as_str(), true)
                .map_err(|e| anyhow!("Failed to encode text: {}", e))?;
            let tokens = encoding.get_ids().len();

            if tokens >= target_tokens {
                return Ok(test_text);
            }
        }
    }

    Ok(test_text)
}

/// Benchmark single forward pass
fn benchmark_single_forward(
    model: &ModernBert,
    tokenizer: &Tokenizer,
    text: &str,
    device: &Device,
) -> Result<f64> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let seq_len = token_ids.len();

    let input_ids = Tensor::new(&token_ids[..], device)?.unsqueeze(0)?;
    let attention_mask = Tensor::new(&attention_mask[..], device)?.unsqueeze(0)?;

    // Estimate memory usage for this request
    // Input tensors: input_ids [1, seq_len] + attention_mask [1, seq_len] = ~seq_len * 8 bytes
    // During forward pass, attention scores can be [batch, num_heads, seq_len, seq_len]
    // For ModernBERT: num_heads=12, so attention scores = 1 * 12 * seq_len * seq_len * 4 bytes
    // This is the main memory consumer for large sequences!
    let _estimated_mb = (seq_len * seq_len * 12 * 4) as f64 / (1024.0 * 1024.0);

    let start = Instant::now();
    let output = model.forward(&input_ids, &attention_mask)?;
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    // Explicitly drop output tensor to free memory immediately
    drop(output);
    drop(input_ids);
    drop(attention_mask);

    Ok(latency_ms)
}

/// Benchmark concurrent requests
fn benchmark_concurrent(
    model: Arc<ModernBert>,
    tokenizer: Arc<Tokenizer>,
    text: Arc<String>,
    device: Device,
    concurrency: usize,
    requests_per_thread: usize,
) -> LatencyStats {
    let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
    let errors = Arc::new(Mutex::new(0usize));

    let handles: Vec<_> = (0..concurrency)
        .map(|_| {
            let model = Arc::clone(&model);
            let tokenizer = Arc::clone(&tokenizer);
            let text = Arc::clone(&text);
            let device = device.clone();
            let latencies = Arc::clone(&latencies);
            let errors = Arc::clone(&errors);

            thread::spawn(move || {
                for _ in 0..requests_per_thread {
                    match benchmark_single_forward(&model, &tokenizer, text.as_str(), &device) {
                        Ok(latency_ms) => {
                            let mut latencies = latencies.lock().unwrap();
                            latencies.push(latency_ms);
                        }
                        Err(e) => {
                            // Only log first error to avoid spam
                            let mut errors = errors.lock().unwrap();
                            if *errors == 0 {
                                eprintln!("First error: {}", e);
                            }
                            *errors += 1;
                        }
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let latencies = latencies.lock().unwrap();
    let error_count = *errors.lock().unwrap();
    let latencies_vec = latencies.clone();
    drop(latencies); // Release lock early

    // Give CUDA time to clean up memory
    if device.is_cuda() {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    LatencyStats::from_latencies(&latencies_vec, error_count)
}

fn main() -> Result<()> {
    // Check if Flash Attention is enabled
    #[cfg(feature = "flash-attn")]
    {
        println!("✅ Flash Attention 2 is ENABLED - will use memory-efficient attention");
    }
    #[cfg(not(feature = "flash-attn"))]
    {
        println!("⚠️  Flash Attention 2 is DISABLED - large contexts (16K+) may fail due to OOM");
        println!("   To enable: cargo run --example benchmark_concurrent --release --features cuda,flash-attn");
    }

    let device = if cfg!(feature = "cuda") {
        Device::cuda_if_available(0)?
    } else {
        Device::Cpu
    };

    let model_id = "llm-semantic-router/modernbert-base-32k";
    let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, "main".to_string());
    let api = Api::new()?;
    let api = api.repo(repo);

    let config_path = api.get("config.json")?;
    let config_str = std::fs::read_to_string(&config_path)?;
    let mut config: Config = serde_json::from_str(&config_str)?;

    // Override max_position_embeddings for Extended32K variant
    // Check if this is modernbert-base-32k by model ID
    let is_32k_model = model_id.contains("modernbert-base-32k") || model_id.contains("32k");

    let variant = ModernBertVariant::detect_from_config(config_path.to_str().unwrap())?;

    // Force Extended32K if model ID indicates 32K support
    let is_extended_32k = matches!(variant, ModernBertVariant::Extended32K) || is_32k_model;

    if is_extended_32k {
        // For modernbert-base-32k, set to 32768 to support full 32K context
        // Note: This creates a large RoPE cache, but it's needed for 32K tests
        if config.max_position_embeddings < 32768 {
            config.max_position_embeddings = 32768;
        }
    }

    let base_weights_path = api.get("model.safetensors")?;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[base_weights_path], DType::F32, &device)
            .map_err(|e| anyhow!("Failed to load weights: {}", e))?
    };
    let model = ModernBert::load(vb, &config)?;

    let tokenizer_path = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("{}", e))?;

    // Test context lengths (from smallest to largest)
    // Note: 16K and 32K require >22GB VRAM even with Flash Attention 2
    let context_lengths = vec![
        1024, 4096,
        8192,
        // 16384,  // Requires >22GB VRAM (RoPE cache + tensors exceed available memory)
        // 32768,  // Requires >22GB VRAM (RoPE cache + tensors exceed available memory)
    ];
    let concurrency_levels = vec![1, 10, 50, 100];
    let requests_per_thread = 10;

    let model = Arc::new(model);
    let tokenizer = Arc::new(tokenizer);
    let base_text = "The quick brown fox jumps over the lazy dog. ";
    let mut results: Vec<(usize, usize, LatencyStats)> = Vec::new();

    // Run tests by concurrency level first (C=1 for all contexts, then C=10, etc.)
    // This helps memory cleanup between tests
    for &concurrency in &concurrency_levels {
        println!("\n{}", "=".repeat(60));
        println!("Testing with Concurrency={}", concurrency);
        println!("{}", "=".repeat(60));

        for &context_length in &context_lengths {
            println!(
                "\nTesting {} tokens with C={}:",
                context_length, concurrency
            );

            // Check GPU memory before large tests
            if device.is_cuda() && (context_length >= 8192 || concurrency >= 50) {
                if let Some((_total_gb, free_gb)) = check_gpu_memory() {
                    if free_gb < 2.0 {
                        println!("  ⚠️  Low GPU memory ({:.2}GB), skipping test", free_gb);
                        results.push((context_length, concurrency, LatencyStats::default()));
                        continue;
                    }
                }
            }

            let text = create_text_with_exact_tokens(&tokenizer, base_text, context_length)?;
            let text = Arc::new(text);

            print!("  Running... ");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();

            let stats = benchmark_concurrent(
                Arc::clone(&model),
                Arc::clone(&tokenizer),
                Arc::clone(&text),
                device.clone(),
                concurrency,
                requests_per_thread,
            );

            stats.print(concurrency);
            results.push((context_length, concurrency, stats));

            // Clean up memory after each test - give CUDA time to free memory
            if device.is_cuda() {
                let sleep_ms = if context_length >= 8192 {
                    2000 // 2 seconds for 8K contexts
                } else if concurrency >= 50 {
                    1000 // 1 second for high concurrency
                } else if concurrency >= 10 {
                    500 // 0.5 seconds for medium concurrency
                } else {
                    300 // 0.3 seconds for low concurrency
                };
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            }
        }
    }

    println!("\nSummary (Mean Latency in ms):");
    println!(
        "{:<12} | {:>8} | {:>8} | {:>8} | {:>8}",
        "Context", "C=1", "C=10", "C=50", "C=100"
    );
    println!("{}", "-".repeat(50));

    for &context_length in &context_lengths {
        print!("{:<12} |", format!("{} tokens", context_length));
        for &concurrency in &concurrency_levels {
            if let Some((_, _, stats)) = results
                .iter()
                .find(|(c, con, _)| *c == context_length && *con == concurrency)
            {
                print!(" {:>8.2}", stats.mean_ms);
            } else {
                print!(" {:>8}", "N/A");
            }
        }
        println!();
    }

    Ok(())
}
