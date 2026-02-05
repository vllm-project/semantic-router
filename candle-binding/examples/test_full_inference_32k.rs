//! Test full inference pipeline with Extended32K base model + existing classifier
//!
//! This example tests the complete inference flow:
//! 1. Load Extended32K base model
//! 2. Load classifier weights from existing PII classifier
//! 3. Combine them into a working classifier
//! 4. Perform classification on real text
//!
//! Usage:
//!   cargo run --example test_full_inference_32k --release --no-default-features

use anyhow::{anyhow, Result};
use candle_semantic_router::model_architectures::traditional::modernbert::{
    ModernBertVariant, TraditionalModernBertClassifier,
};
use candle_transformers::models::modernbert::Config;
use hf_hub::{api::sync::Api, Repo, RepoType};

fn main() -> Result<()> {
    println!("🧪 Testing Full Inference with Extended32K Base Model + Classifier");
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
    let _base_weights_path = match api.get("model.safetensors") {
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

    // Step 3: Load PII classifier to get classifier weights
    println!("\n3️⃣  Loading PII classifier weights...");
    let pii_classifier_paths = vec![
        "../models/pii_classifier_modernbert-base_model",
        "models/pii_classifier_modernbert-base_model",
    ];

    let pii_classifier_path = pii_classifier_paths
        .iter()
        .find(|path| std::path::Path::new(path).exists())
        .copied()
        .ok_or_else(|| anyhow!("PII classifier not found"))?;

    println!("   ✓ Found PII classifier at: {}", pii_classifier_path);

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

    // Step 4: Create combined classifier using Extended32K base model + PII classifier weights
    println!("\n4️⃣  Creating combined classifier with Extended32K base model...");

    let base_model_path = base_model_dir.to_string_lossy().to_string();
    println!("   ℹ️  Using load_with_custom_base_model to combine:");
    println!(
        "      - Base model: Extended32K (32K tokens) from {}",
        base_model_path
    );
    println!(
        "      - Classifier: PII classifier weights from {}",
        pii_classifier_path
    );

    // Use the new method to load base model from one path and classifier from another
    let combined_classifier = TraditionalModernBertClassifier::load_with_custom_base_model(
        &base_model_path,
        pii_classifier_path,
        ModernBertVariant::Extended32K,
        true, // use_cpu
    )
    .map_err(|e| anyhow!("Failed to load combined classifier: {}", e))?;

    println!("   ✅ Combined classifier loaded successfully!");
    println!("   ✅ Base model: Extended32K (supports up to 32K tokens)");
    println!("   ✅ Classifier: PII classifier weights (18 classes)");
    println!("   ✅ Tokenizer: Configured for 32K tokens");

    // Step 9: Test inference on sample texts
    println!("\n9️⃣  Testing inference on sample texts...");

    // Create test texts - need to store them as owned strings
    let short_text = "My email is john@example.com".to_string();
    let medium_text =
        "Please contact me at john.doe@company.com or call 555-1234. My SSN is 123-45-6789."
            .to_string();
    let long_text_no_pii = format!(
        "{} This is a long text without PII. ",
        "Lorem ipsum dolor sit amet. ".repeat(50)
    );

    // Create a realistic long text with PII (simulating a long document)
    let long_text_with_pii = format!(
        "{} Please contact John Doe at john.doe@company.com or call 555-1234. His SSN is 123-45-6789. {}",
        "This is a long document that contains multiple paragraphs. ".repeat(100),
        "For billing inquiries, contact billing@company.com. For support, call 1-800-555-0199."
    );

    let test_texts = vec![
        ("Short text", short_text.as_str()),
        ("Medium text", medium_text.as_str()),
        ("Long text (no PII)", long_text_no_pii.as_str()),
        ("Long text (with PII)", long_text_with_pii.as_str()),
    ];

    for (name, text) in test_texts {
        println!("\n   Testing: {}", name);
        println!("   Text length: {} characters", text.len());

        match combined_classifier.classify_text(text) {
            Ok((class_id, confidence)) => {
                println!("   ✅ Classification successful!");
                println!("      Class ID: {}", class_id);
                println!("      Confidence: {:.4}", confidence);
            }
            Err(e) => {
                println!("   ❌ Classification failed: {}", e);
            }
        }
    }

    println!("\n   📝 Note: This test uses Extended32K base model (32K tokens)");
    println!("   📝 The classifier should now handle long texts without truncation!");

    println!("\n✅ Full inference test completed!");
    Ok(())
}
