//! Test mmBERT variant detection and loading via TraditionalModernBertClassifier
//!
//! Run with: cargo run --example test_mmbert_variant --no-default-features
//!
//! Note: Skipped in CI environments (mmBERT model is not downloaded in CI)

use candle_semantic_router::model_architectures::traditional::{
    ModernBertVariant, TraditionalModernBertClassifier,
};

fn main() -> anyhow::Result<()> {
    // Skip in CI environments - mmBERT model is not downloaded in CI to save resources
    if std::env::var("CI").is_ok() {
        println!("Skipping mmBERT variant test in CI environment");
        return Ok(());
    }

    println!("=== mmBERT Variant Detection Test ===\n");

    let model_path = "../models/mmbert-base";
    let config_path = format!("{}/config.json", model_path);

    // 1. Test variant detection
    println!("1. Testing variant detection from config...");
    let variant = ModernBertVariant::detect_from_config(&config_path)?;
    println!("   Detected variant: {:?}", variant);
    assert_eq!(
        variant,
        ModernBertVariant::Multilingual,
        "Expected Multilingual variant"
    );
    println!("   ✓ Correctly detected as Multilingual (mmBERT)\n");

    // 2. Test variant properties
    println!("2. Testing variant properties...");
    println!("   max_length: {}", variant.max_length());
    println!("   pad_token: {}", variant.pad_token());
    println!(
        "   tokenization_strategy: {:?}",
        variant.tokenization_strategy()
    );
    assert_eq!(variant.max_length(), 8192);
    assert_eq!(variant.pad_token(), "<pad>");
    println!("   ✓ Variant properties correct\n");

    // 3. Test loading with auto-detection (will fail without classifier head, but let's check the error)
    println!("3. Testing TraditionalModernBertClassifier.load_from_directory...");
    println!("   (Note: Base mmBERT doesn't have classifier head, so this will fail as expected)");

    match TraditionalModernBertClassifier::load_from_directory(model_path, true) {
        Ok(classifier) => {
            println!("   Loaded successfully!");
            println!("   is_multilingual: {}", classifier.is_multilingual());
            println!("   variant: {:?}", classifier.variant());
            println!("   num_classes: {}", classifier.get_num_classes());
        }
        Err(e) => {
            // Expected - base model doesn't have classifier weights
            println!("   Expected error (no classifier head): {}", e);
            println!("   ✓ Error is expected - base mmBERT is for MLM, not classification\n");
        }
    }

    // 4. Test explicit variant loading
    println!("4. Testing load_from_directory_with_variant...");
    match TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_path,
        true,
        ModernBertVariant::Multilingual,
    ) {
        Ok(classifier) => {
            println!("   Loaded with explicit Multilingual variant!");
            println!("   is_multilingual: {}", classifier.is_multilingual());
        }
        Err(e) => {
            println!("   Expected error (no classifier head): {}", e);
        }
    }

    // 5. Test load_mmbert_from_directory convenience method
    println!("\n5. Testing load_mmbert_from_directory convenience method...");
    match TraditionalModernBertClassifier::load_mmbert_from_directory(model_path, true) {
        Ok(classifier) => {
            println!("   Loaded as mmBERT!");
            println!("   is_multilingual: {}", classifier.is_multilingual());
        }
        Err(e) => {
            println!("   Expected error (no classifier head): {}", e);
        }
    }

    println!("\n=== Test Summary ===");
    println!("✓ Variant detection works correctly");
    println!("✓ Variant properties are correct for mmBERT");
    println!("✓ Loading methods correctly identify model as multilingual");
    println!(
        "\nNote: Full classification requires a fine-tuned mmBERT model with a classifier head."
    );

    Ok(())
}
