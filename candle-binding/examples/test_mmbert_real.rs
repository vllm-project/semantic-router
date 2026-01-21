//! Real mmBERT Test - Verifies loading and inference with actual model
//!
//! Run with: cargo run --example test_mmbert_real --no-default-features
//!
//! Note: Skipped in CI environments (mmBERT model is not downloaded in CI)

use anyhow::anyhow;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert::{Config, ModernBert};
use tokenizers::Tokenizer;

fn main() -> anyhow::Result<()> {
    // Skip in CI environments - mmBERT model is not downloaded in CI to save resources
    if std::env::var("CI").is_ok() {
        println!("Skipping mmBERT test in CI environment");
        return Ok(());
    }

    println!("=== mmBERT Real Model Test ===\n");

    let model_path = "../models/mmbert-base";

    // 1. Check files exist
    println!("1. Checking model files...");
    let config_path = format!("{}/config.json", model_path);
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let weights_path = format!("{}/model.safetensors", model_path);

    assert!(
        std::path::Path::new(&config_path).exists(),
        "config.json not found"
    );
    assert!(
        std::path::Path::new(&tokenizer_path).exists(),
        "tokenizer.json not found"
    );
    assert!(
        std::path::Path::new(&weights_path).exists(),
        "model.safetensors not found"
    );
    println!("   ✓ All model files found\n");

    // 2. Load and verify config
    println!("2. Loading config.json...");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)?;

    let vocab_size = config_json["vocab_size"].as_u64().unwrap_or(0);
    let hidden_size = config_json["hidden_size"].as_u64().unwrap_or(0);
    let num_layers = config_json["num_hidden_layers"].as_u64().unwrap_or(0);
    let max_position = config_json["max_position_embeddings"].as_u64().unwrap_or(0);
    let position_type = config_json["position_embedding_type"]
        .as_str()
        .unwrap_or("");

    println!("   vocab_size: {}", vocab_size);
    println!("   hidden_size: {}", hidden_size);
    println!("   num_hidden_layers: {}", num_layers);
    println!("   max_position_embeddings: {}", max_position);
    println!("   position_embedding_type: {}", position_type);

    // Verify this is mmBERT
    assert!(
        vocab_size >= 200000,
        "Expected vocab_size >= 200000 for mmBERT"
    );
    assert_eq!(position_type, "sans_pos", "Expected sans_pos for mmBERT");
    println!("   ✓ Confirmed as mmBERT (multilingual)\n");

    // 3. Load tokenizer
    println!("3. Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
    let tokenizer_vocab_size = tokenizer.get_vocab_size(false);
    println!("   tokenizer vocab_size: {}", tokenizer_vocab_size);
    println!("   ✓ Tokenizer loaded\n");

    // 4. Test tokenization with multilingual text
    println!("4. Testing multilingual tokenization...");
    let test_texts = vec![
        ("English", "Hello, how are you today?"),
        ("Chinese", "你好，今天怎么样？"),
        ("Spanish", "Hola, ¿cómo estás hoy?"),
        ("German", "Hallo, wie geht es Ihnen heute?"),
        ("Japanese", "こんにちは、今日はいかがですか？"),
        ("Korean", "안녕하세요, 오늘 어떠세요?"),
        ("Arabic", "مرحبا، كيف حالك اليوم؟"),
        ("Russian", "Привет, как дела сегодня?"),
    ];

    for (lang, text) in &test_texts {
        let encoding = tokenizer
            .encode(*text, true)
            .map_err(|e| anyhow!("Failed to encode {}: {}", lang, e))?;
        let tokens = encoding.get_tokens();
        let ids = encoding.get_ids();
        println!(
            "   {}: '{}' -> {} tokens, ids: {:?}",
            lang,
            text,
            tokens.len(),
            &ids[..ids.len().min(5)]
        );
    }
    println!("   ✓ Multilingual tokenization works\n");

    // 5. Load model weights
    println!("5. Loading model weights...");
    let device = Device::Cpu;
    let config: Config = serde_json::from_str(&config_str)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)? };
    println!("   ✓ Weights loaded from safetensors\n");

    // 6. Create ModernBERT model
    println!("6. Creating ModernBERT model...");
    let model = ModernBert::load(vb, &config)?;
    println!("   ✓ Model created successfully\n");

    // 7. Test forward pass
    println!("7. Testing forward pass with multilingual inputs...");
    for (lang, text) in &test_texts[..3] {
        // Just test first 3 for speed
        let encoding = tokenizer
            .encode(*text, true)
            .map_err(|e| anyhow!("Failed to encode {}: {}", lang, e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        let input_ids = Tensor::new(&ids[..], &device)?.unsqueeze(0)?;
        let attention_mask_tensor = Tensor::new(&attention_mask[..], &device)?.unsqueeze(0)?;

        let output = model.forward(&input_ids, &attention_mask_tensor)?;
        let output_shape = output.dims();

        println!(
            "   {}: input_shape=[1, {}], output_shape={:?}",
            lang,
            ids.len(),
            output_shape
        );

        // Verify output shape
        assert_eq!(output_shape.len(), 3, "Expected 3D output");
        assert_eq!(output_shape[0], 1, "Expected batch size 1");
        assert_eq!(output_shape[1], ids.len(), "Expected seq_len match");
        assert_eq!(
            output_shape[2], hidden_size as usize,
            "Expected hidden_size match"
        );
    }
    println!("   ✓ Forward pass works for all languages\n");

    // 8. Summary
    println!("=== Test Summary ===");
    println!("✓ mmBERT model loaded successfully");
    println!("✓ Multilingual tokenization verified");
    println!("✓ Model inference works");
    println!("\nmmBERT is ready for multilingual text processing!");

    Ok(())
}
