// Comprehensive example demonstrating Qwen3 Multi-LoRA Classification in Rust
//
// This example shows:
// 1. Multi-LoRA adapter loading and switching
// 2. Zero-shot classification (no adapter required)
// 3. Benchmark dataset evaluation
// 4. Error handling
//
// Usage:
//   cd ../../candle-binding
//   cargo run --release --example qwen3_example

use candle_core::Device;
use candle_semantic_router::model_architectures::generative::Qwen3MultiLoRAClassifier;
use serde::{Deserialize, Serialize};
use std::fs;
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
struct TestSample {
    text: String,
    true_label: String,
    true_label_id: usize,
}

fn print_header(title: &str) {
    println!("\n{}", "=".repeat(70));
    println!("  {}", title);
    println!("{}", "=".repeat(70));
}

fn demonstrate_zero_shot(
    classifier: &mut Qwen3MultiLoRAClassifier,
) -> Result<(), Box<dyn std::error::Error>> {
    print_header("ZERO-SHOT CLASSIFICATION (No Adapter Required)");

    let test_cases = vec![
        (
            "Sentiment Analysis",
            "This movie was absolutely fantastic! I loved every minute of it.",
            vec![
                "positive".to_string(),
                "negative".to_string(),
                "neutral".to_string(),
            ],
        ),
        (
            "Topic Classification",
            "The stock market rallied today as investors reacted to positive economic data.",
            vec![
                "science".to_string(),
                "politics".to_string(),
                "sports".to_string(),
                "business".to_string(),
            ],
        ),
        (
            "Intent Detection",
            "What time does the store open?",
            vec![
                "question".to_string(),
                "command".to_string(),
                "statement".to_string(),
            ],
        ),
    ];

    let mut correct = 0;
    for (i, (name, text, categories)) in test_cases.iter().enumerate() {
        println!("\n[{}/{}] {}", i + 1, test_cases.len(), name);
        println!("  Text: {}", text);
        println!("  Categories: {:?}", categories);

        match classifier.classify_zero_shot(text, categories.clone()) {
            Ok(result) => {
                println!(
                    "  ✅ Result: {} ({:.2}% confidence)",
                    result.category,
                    result.confidence * 100.0
                );

                // Simple accuracy check
                if (text.contains("fantastic") && result.category == "positive")
                    || (text.contains("stock market") && result.category == "business")
                    || (text.contains("What time") && result.category == "question")
                {
                    correct += 1;
                }
            }
            Err(e) => {
                println!("  ❌ Error: {:?}", e);
            }
        }
    }

    println!(
        "\n  Accuracy: {}/{} ({:.1}%)",
        correct,
        test_cases.len(),
        (correct as f32 / test_cases.len() as f32) * 100.0
    );

    Ok(())
}

fn demonstrate_multi_adapter(
    classifier: &mut Qwen3MultiLoRAClassifier,
) -> Result<(), Box<dyn std::error::Error>> {
    print_header("MULTI-ADAPTER CLASSIFICATION");

    // Load adapter
    let adapter_path = "../../models/qwen3_generative_classifier_r16";
    println!("\n  Loading adapter from: {}", adapter_path);

    classifier.load_adapter("category", adapter_path)?;
    println!("  ✅ Adapter 'category' loaded successfully");

    // Show loaded adapters
    let adapters = classifier.list_adapters();
    println!("\n  Loaded adapters: {:?}", adapters);

    // Test classification
    let test_texts = vec![
        "What is the weather like today?",
        "I want to book a flight to Paris",
        "Tell me a joke about programming",
    ];

    println!("\n  Testing adapter classification:");
    for (i, text) in test_texts.iter().enumerate() {
        println!("\n  [{}] Text: {}", i + 1, text);
        match classifier.classify_with_adapter(text, "category") {
            Ok(result) => {
                println!(
                    "    ✅ Category: {} ({:.2}% confidence)",
                    result.category,
                    result.confidence * 100.0
                );
            }
            Err(e) => {
                println!("    ❌ Error: {:?}", e);
            }
        }
    }

    Ok(())
}

fn run_benchmark_evaluation(
    classifier: &mut Qwen3MultiLoRAClassifier,
) -> Result<(), Box<dyn std::error::Error>> {
    print_header("BENCHMARK DATASET EVALUATION");

    // Load test data
    let data_path = "../../bench/test_data.json";
    let contents = match fs::read_to_string(data_path) {
        Ok(c) => c,
        Err(e) => {
            println!("\n  ⚠️  Could not load benchmark data: {} (skipping)", e);
            return Ok(());
        }
    };

    let samples: Vec<TestSample> = serde_json::from_str(&contents)?;
    println!("\n  Loaded {} test samples", samples.len());

    // Run evaluation
    let mut correct = 0;
    let start_time = Instant::now();

    for (i, sample) in samples.iter().enumerate() {
        match classifier.classify_with_adapter(&sample.text, "category") {
            Ok(result) => {
                if result.category == sample.true_label {
                    correct += 1;
                }
            }
            Err(e) => {
                println!("  [{}] Error: {:?}", i + 1, e);
            }
        }

        if (i + 1) % 10 == 0 {
            println!("  Progress: {}/{} samples processed", i + 1, samples.len());
        }
    }

    let duration = start_time.elapsed();

    // Results
    let accuracy = (correct as f32 / samples.len() as f32) * 100.0;
    let avg_latency = duration.as_millis() / samples.len() as u128;

    println!("\n  📊 Results:");
    println!(
        "    • Accuracy: {}/{} ({:.2}%)",
        correct,
        samples.len(),
        accuracy
    );
    println!("    • Total time: {:?}", duration);
    println!("    • Avg latency: {}ms per sample", avg_latency);

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Qwen3 Multi-LoRA Classification - Comprehensive Example  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    // Check for model path override
    let base_model_path =
        std::env::var("BASE_MODEL_PATH").unwrap_or_else(|_| "../../models/Qwen3-0.6B".to_string());

    // Select device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("\n🖥️  Using device: {:?}", device);

    // Initialize base model
    println!("🔧 Initializing base model: {}", base_model_path);
    let mut classifier = match Qwen3MultiLoRAClassifier::new(&base_model_path, &device) {
        Ok(c) => {
            println!("✅ Base model initialized successfully");
            c
        }
        Err(e) => {
            eprintln!("❌ Failed to initialize base model: {:?}", e);
            eprintln!("\n💡 Make sure the model exists at: {}", base_model_path);
            eprintln!("   You can download it with:");
            eprintln!(
                "   git clone https://huggingface.co/Qwen/Qwen3-0.6B {}",
                base_model_path
            );
            return Err(Box::new(e));
        }
    };

    // Run demonstration scenarios
    demonstrate_zero_shot(&mut classifier)?;
    demonstrate_multi_adapter(&mut classifier)?;
    run_benchmark_evaluation(&mut classifier)?;

    // Summary
    print_header("✅ All examples completed successfully!");
    println!("\nFor more examples, see:");
    println!("  • ../../candle-binding/semantic-router_test.go (unit tests)");
    println!("  • ../../candle-binding/ZERO_SHOT_CLASSIFICATION.md (documentation)");
    println!("  • ../../candle-binding/MULTI_ADAPTER_IMPLEMENTATION.md (architecture)");
    println!();

    Ok(())
}
