//! Performance Benchmark Script for ModernBERT-base-32k
//!
//! This script provides detailed performance profiling for ModernBERT-base-32k,
//! breaking down latency by component to identify bottlenecks.
//!
//! Usage:
//!   cargo run --example benchmark_performance --release --features cuda

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_semantic_router::model_architectures::traditional::modernbert::ModernBertVariant;
// Use local copy of ModernBERT with Flash Attention support
// Using full path to avoid re-export issues
use candle_semantic_router::model_architectures::traditional::candle_models::{Config, ModernBert};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::time::Instant;
use tokenizers::Tokenizer;

/// Detailed timing breakdown for a single forward pass
#[derive(Default)]
struct TimingBreakdown {
    tokenization_ms: f64,
    tensor_creation_ms: f64,
    forward_pass_ms: f64,
    total_ms: f64,
}

impl TimingBreakdown {
    fn print(&self, sequence_length: usize) {
        println!("\n   Performance Breakdown ({} tokens):", sequence_length);
        println!(
            "      Tokenization:      {:.2}ms ({:.1}%)",
            self.tokenization_ms,
            (self.tokenization_ms / self.total_ms) * 100.0
        );
        println!(
            "      Tensor Creation:    {:.2}ms ({:.1}%)",
            self.tensor_creation_ms,
            (self.tensor_creation_ms / self.total_ms) * 100.0
        );
        println!(
            "      Forward Pass:       {:.2}ms ({:.1}%)",
            self.forward_pass_ms,
            (self.forward_pass_ms / self.total_ms) * 100.0
        );
        println!("      ────────────────────────────────");
        println!("      Total:             {:.2}ms", self.total_ms);
    }
}

/// Check GPU memory available using nvidia-smi
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

/// Benchmark a single sequence length with detailed timing
fn benchmark_sequence(
    model: &ModernBert,
    tokenizer: &Tokenizer,
    device: &Device,
    target_tokens: usize,
    warmup_iterations: usize,
    benchmark_iterations: usize,
) -> Result<TimingBreakdown> {
    let base_text = "This is a test sentence for performance benchmarking. ";

    // Create text with exact token count
    let test_text = create_text_with_exact_tokens(tokenizer, base_text, target_tokens)?;

    // Warmup
    for _ in 0..warmup_iterations {
        let encoding = tokenizer
            .encode(test_text.as_str(), true)
            .map_err(|e| anyhow!("Failed to encode: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let input_ids_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], device)?.unsqueeze(0)?;
        let _output = model.forward(&input_ids_tensor, &attention_mask_tensor)?;
    }

    // Benchmark
    let mut timing = TimingBreakdown::default();
    let mut tokenization_times = Vec::new();
    let mut tensor_creation_times = Vec::new();
    let mut forward_pass_times = Vec::new();

    for _ in 0..benchmark_iterations {
        // Tokenization
        let start = Instant::now();
        let encoding = tokenizer
            .encode(test_text.as_str(), true)
            .map_err(|e| anyhow!("Failed to encode: {}", e))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let tokenization_time = start.elapsed().as_secs_f64() * 1000.0;
        tokenization_times.push(tokenization_time);

        // Tensor creation
        let start = Instant::now();
        let input_ids_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], device)?.unsqueeze(0)?;
        let tensor_creation_time = start.elapsed().as_secs_f64() * 1000.0;
        tensor_creation_times.push(tensor_creation_time);

        // Forward pass
        let start = Instant::now();
        let _output = model.forward(&input_ids_tensor, &attention_mask_tensor)?;
        let forward_pass_time = start.elapsed().as_secs_f64() * 1000.0;
        forward_pass_times.push(forward_pass_time);
    }

    // Calculate averages
    timing.tokenization_ms = tokenization_times.iter().sum::<f64>() / benchmark_iterations as f64;
    timing.tensor_creation_ms =
        tensor_creation_times.iter().sum::<f64>() / benchmark_iterations as f64;
    timing.forward_pass_ms = forward_pass_times.iter().sum::<f64>() / benchmark_iterations as f64;
    timing.total_ms = timing.tokenization_ms + timing.tensor_creation_ms + timing.forward_pass_ms;

    Ok(timing)
}

fn main() -> Result<()> {
    println!("Performance Benchmark for ModernBERT-base-32k");
    println!("{}", "=".repeat(70));

    // Detect device
    let device = if cfg!(feature = "cuda") {
        match Device::new_cuda(0) {
            Ok(d) => {
                println!("Using GPU (CUDA) for benchmarking");
                if let Some((total_gb, free_gb)) = check_gpu_memory() {
                    println!(
                        "   GPU Memory: {:.2}GB free / {:.2}GB total",
                        free_gb, total_gb
                    );
                }
                d
            }
            Err(e) => {
                println!("GPU not available ({}), falling back to CPU", e);
                Device::Cpu
            }
        }
    } else {
        println!("Running in CPU mode (CUDA feature not enabled)");
        Device::Cpu
    };

    // Load model
    println!("\nLoading ModernBERT-base-32k...");
    let base_model_id = "llm-semantic-router/modernbert-base-32k";
    let repo = Repo::with_revision(
        base_model_id.to_string(),
        RepoType::Model,
        "main".to_string(),
    );
    let api = Api::new()?;
    let api = api.repo(repo);

    let base_config_path = api
        .get("config.json")
        .map_err(|e| anyhow!("Failed to download config.json: {}", e))?;
    let base_tokenizer_path = api
        .get("tokenizer.json")
        .map_err(|e| anyhow!("Failed to download tokenizer.json: {}", e))?;
    let base_weights_path = api
        .get("model.safetensors")
        .map_err(|e| anyhow!("Failed to download model.safetensors: {}", e))?;

    // Load config
    let config_str = std::fs::read_to_string(&base_config_path)?;
    let mut config: Config = serde_json::from_str(&config_str)?;

    // Override max_position_embeddings for Extended32K variant
    let optimal_max_len = if let Device::Cuda(_) = device {
        if let Some((_total_gb, free_gb)) = check_gpu_memory() {
            if free_gb >= 15.0 {
                32768
            } else {
                16384
            }
        } else {
            16384
        }
    } else {
        16384
    };

    if config.max_position_embeddings < optimal_max_len {
        config.max_position_embeddings = optimal_max_len;
    }

    // Load model
    let base_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[base_weights_path], DType::F32, &device)
            .map_err(|e| anyhow!("Failed to load weights: {}", e))?
    };
    let model =
        ModernBert::load(base_vb, &config).map_err(|e| anyhow!("Failed to load model: {}", e))?;

    // Load tokenizer
    let mut tokenizer = Tokenizer::from_file(&base_tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    if let Some(pad_token) = tokenizer.get_padding_mut() {
        pad_token.strategy = tokenizers::PaddingStrategy::BatchLongest;
        pad_token.pad_token = ModernBertVariant::Extended32K.pad_token().to_string();
    }

    println!("Model loaded successfully!");

    // Check available GPU memory after model is loaded
    // For CPU mode, test all available lengths
    let mut test_cases = if device.is_cuda() {
        vec![512, 1024, 4096, 8192]
    } else {
        vec![512, 1024, 4096, 8192] // CPU can handle all these
    };
    if let Device::Cuda(_) = device {
        if let Some((_total_gb, free_gb)) = check_gpu_memory() {
            println!(
                "\nCurrent GPU Memory: {:.2}GB free / {:.2}GB total",
                free_gb, _total_gb
            );

            // Determine what we can test based on available memory
            // Conservative estimates: 16K needs ~8GB free, 32K needs ~15GB free
            if free_gb >= 8.0 {
                println!("GPU has sufficient memory for 16K testing");
                test_cases.push(16384);
            } else {
                println!(
                    "Insufficient GPU memory for 16K testing (need ~8GB free, have {:.2}GB)",
                    free_gb
                );
            }

            // Note: 32K would need ~15GB free, which is unlikely on L4 (22GB total)
            // So we skip 32K for now
        }
    }

    let warmup_iterations = 3;
    let benchmark_iterations = 5;

    println!("\n{}", "=".repeat(70));
    println!("PERFORMANCE BENCHMARKING");
    println!("{}", "=".repeat(70));
    println!("Warmup iterations: {}", warmup_iterations);
    println!("Benchmark iterations: {}", benchmark_iterations);

    for target_tokens in test_cases {
        println!("\n{}", "-".repeat(70));
        println!("Testing {} tokens...", target_tokens);

        // Check memory before each test (especially for large sequences)
        if target_tokens >= 16384 {
            if let Device::Cuda(_) = device {
                if let Some((_total_gb, free_gb)) = check_gpu_memory() {
                    if free_gb < 8.0 {
                        println!(
                            "Skipping {} tokens (insufficient GPU memory: {:.2}GB free)",
                            target_tokens, free_gb
                        );
                        continue;
                    }
                }
            }
        }

        match benchmark_sequence(
            &model,
            &tokenizer,
            &device,
            target_tokens,
            warmup_iterations,
            benchmark_iterations,
        ) {
            Ok(timing) => {
                timing.print(target_tokens);
            }
            Err(e) => {
                if e.to_string().contains("out of memory")
                    || e.to_string().contains("OUT_OF_MEMORY")
                {
                    println!("Skipping {} tokens (GPU out of memory)", target_tokens);
                } else {
                    println!("Benchmark failed: {}", e);
                }
            }
        }
    }

    println!("\n{}", "=".repeat(70));
    println!("Benchmarking complete!");
    println!("{}", "=".repeat(70));

    Ok(())
}
