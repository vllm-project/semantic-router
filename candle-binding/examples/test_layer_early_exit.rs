//! 2D Matryoshka Layer Early Exit Test
//!
//! Verifies that layer early exit provides expected speedups.
//!
//! Usage:
//! ```bash
//! cargo run --release --no-default-features --example test_layer_early_exit
//! ```

use candle_core::{DType, Device, Tensor};
use candle_semantic_router::model_architectures::embedding::MmBertEmbeddingModel;
use std::time::Instant;

fn main() {
    println!("\n2D Matryoshka Layer Early Exit Test");
    println!("{}\n", "=".repeat(50));

    let model_path = std::env::var("MMBERT_MODEL_PATH").expect(
        "MMBERT_MODEL_PATH not set. Download: huggingface-cli download llm-semantic-router/mmbert-embed-32k-2d-matryoshka"
    );

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let device_name = match &device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "GPU",
        _ => "Unknown",
    };

    let model = MmBertEmbeddingModel::load(&model_path, &device).expect("Failed to load model");

    println!("Device: {}", device_name);
    println!(
        "Layers: {}, Hidden: {}\n",
        model.num_layers(),
        model.config().hidden_size
    );

    let batch_size = 4;
    let seq_lengths = [128, 256];
    let target_layers = [3, 6, 11, 22];

    for &seq_len in &seq_lengths {
        println!("seq_len={}, batch_size={}", seq_len, batch_size);
        println!("{}", "-".repeat(50));

        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::U32, &device).unwrap();

        // Measure baseline (L22)
        let start = Instant::now();
        let result = model
            .embedding_forward_with_matryoshka(&input_ids, Some(&attention_mask), Some(22), None)
            .unwrap();
        let _: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        let baseline_ms = start.elapsed().as_secs_f64() * 1000.0;

        for &target_layer in &target_layers {
            let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device).unwrap();
            let attention_mask = Tensor::ones((batch_size, seq_len), DType::U32, &device).unwrap();

            let start = Instant::now();
            let result = model
                .embedding_forward_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    Some(target_layer),
                    None,
                )
                .unwrap();
            let _: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let speedup = baseline_ms / elapsed_ms;
            let expected = 22.0 / target_layer as f64;
            let status = if (speedup - expected).abs() / expected < 0.4 {
                "OK"
            } else {
                "FAIL"
            };

            println!(
                "  L{:2}: {:7.1}ms  {:.2}x (expected {:.2}x)  {}",
                target_layer, elapsed_ms, speedup, expected, status
            );
        }
        println!();
    }
}
