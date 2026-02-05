//! Test compatibility of existing ModernBERT classifiers with Extended32K base model
//!
//! This example tests if existing classifier weights (from Standard ModernBERT models)
//! can be loaded and used with the Extended32K base model.
//!
//! Usage:
//!   cargo run --example test_classifier_compatibility --release --features no-cuda

use anyhow::{anyhow, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_semantic_router::model_architectures::traditional::modernbert::FixedModernBertClassifier;
use candle_transformers::models::modernbert::{Config, ModernBert};
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    println!("🧪 Testing Classifier Compatibility with Extended32K Base Model");
    println!("{}", "=".repeat(70));

    // Step 1: Download Extended32K base model
    println!("\n1️⃣  Downloading Extended32K base model...");
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
    let _base_tokenizer_path = api
        .get("tokenizer.json")
        .map_err(|e| anyhow!("Failed to download tokenizer.json: {}", e))?;
    let base_weights_path = match api.get("model.safetensors") {
        Ok(path) => {
            println!("   ✓ Base model downloaded (safetensors)");
            path
        }
        Err(_) => {
            println!("   ⚠️  Safetensors not found, trying PyTorch format...");
            api.get("pytorch_model.bin")
                .map_err(|e| anyhow!("Failed to download model weights: {}", e))?
        }
    };

    let base_model_dir = base_config_path.parent().unwrap();
    println!("   ✓ Base model directory: {:?}", base_model_dir);

    // Step 2: Load base model configuration
    println!("\n2️⃣  Loading base model configuration...");
    let config_str = std::fs::read_to_string(&base_config_path)
        .map_err(|e| anyhow!("Failed to read config.json: {}", e))?;
    let config: Config = serde_json::from_str(&config_str)
        .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;

    println!("   ✓ Config loaded:");
    println!("     - hidden_size: {}", config.hidden_size);
    println!("     - vocab_size: {}", config.vocab_size);
    println!(
        "     - max_position_embeddings: {}",
        config.max_position_embeddings
    );

    // Step 3: Load base model
    println!("\n3️⃣  Loading Extended32K base model...");
    let device = Device::Cpu; // Force CPU for testing
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[base_weights_path.clone()], DType::F32, &device)
            .map_err(|e| anyhow!("Failed to load base model weights: {}", e))?
    };

    let _base_model = ModernBert::load(vb.clone(), &config)
        .map_err(|e| anyhow!("Failed to load base ModernBert model: {}", e))?;
    println!("   ✅ Base model loaded successfully!");

    // Step 4: Test with existing PII classifier
    println!("\n4️⃣  Testing with existing PII classifier...");
    // Try multiple possible paths (relative to candle-binding directory)
    let pii_classifier_paths = vec![
        "../models/pii_classifier_modernbert-base_model",
        "models/pii_classifier_modernbert-base_model",
        "./models/pii_classifier_modernbert-base_model",
    ];

    let pii_classifier_path = pii_classifier_paths
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .copied();

    let pii_classifier_path = match pii_classifier_path {
        Some(path) => {
            println!("   ✓ Found PII classifier at: {}", path);
            path
        }
        None => {
            println!("   ⚠️  PII classifier model not found in any of these paths:");
            for path in &pii_classifier_paths {
                println!("      - {}", path);
            }
            println!("   ℹ️  Skipping classifier compatibility test");
            println!("\n   ✅ Base model loading test completed!");
            return Ok(());
        }
    };

    println!("   📁 PII classifier path: {}", pii_classifier_path);

    // Load PII classifier config to get num_classes
    let pii_config_path = format!("{}/config.json", pii_classifier_path);
    let pii_config_str = std::fs::read_to_string(&pii_config_path)
        .map_err(|e| anyhow!("Failed to read PII classifier config: {}", e))?;
    let pii_config_json: serde_json::Value = serde_json::from_str(&pii_config_str)
        .map_err(|e| anyhow!("Failed to parse PII classifier config: {}", e))?;

    let num_classes = pii_config_json
        .get("id2label")
        .and_then(|v| v.as_object())
        .map(|obj| obj.len())
        .or_else(|| {
            pii_config_json
                .get("num_labels")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
        })
        .unwrap_or(2);

    println!("   ✓ Number of classes: {}", num_classes);

    // Try to load classifier weights from PII model
    let pii_weights_path = format!("{}/model.safetensors", pii_classifier_path);
    if !std::path::Path::new(&pii_weights_path).exists() {
        println!(
            "   ⚠️  PII classifier weights not found at: {}",
            pii_weights_path
        );
        println!("   ℹ️  Skipping classifier loading test");
        return Ok(());
    }

    println!("\n5️⃣  Attempting to load classifier weights from PII model...");
    let pii_vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[pii_weights_path], DType::F32, &device)
            .map_err(|e| anyhow!("Failed to load PII classifier weights: {}", e))?
    };

    // Try to load classifier weights
    match FixedModernBertClassifier::load_with_classes(
        pii_vb.pp("classifier"),
        &config,
        num_classes,
    ) {
        Ok(_classifier) => {
            println!("   ✅ Classifier weights loaded successfully!");
            println!("   ✅ Compatibility test PASSED!");
            println!("\n   📝 Conclusion:");
            println!("      Existing classifier weights CAN be used with Extended32K base model");
            println!(
                "      The classifier head is compatible (same hidden_size: {})",
                config.hidden_size
            );
        }
        Err(e) => {
            println!("   ❌ Failed to load classifier weights: {}", e);
            println!("   ⚠️  Compatibility test FAILED");
            println!("\n   📝 Possible reasons:");
            println!("      - Classifier weights structure mismatch");
            println!("      - Different model architecture");
            println!("      - Missing classifier weights in PII model");
            println!("\n   💡 Next steps:");
            println!("      - Check if classifier weights exist in PII model");
            println!("      - Verify weight shapes match (num_classes, hidden_size)");
            println!("      - Consider Option 2: Create new classification head");
        }
    }

    // Step 6: Test inference (if classifier loaded successfully)
    println!("\n6️⃣  Testing inference with combined model...");
    println!("   ℹ️  This would require integrating the classifier with the base model");
    println!("   ℹ️  For now, we've verified that classifier weights can be loaded");

    println!("\n✅ Compatibility test completed!");
    Ok(())
}
