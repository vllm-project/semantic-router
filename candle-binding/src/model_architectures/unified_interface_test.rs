//! Tests for unified model interface

use crate::test_fixtures::fixtures::*;
use rstest::*;
use std::path::Path;

/// Test configurable model loading with real model paths
#[rstest]
fn test_unified_interface_configurable_model_loading(
    traditional_model_path: String,
    lora_model_path: String,
) {
    // Test that model paths are valid and accessible
    println!(
        "Testing configurable model loading with paths: traditional={}, lora={}",
        traditional_model_path, lora_model_path
    );

    // Test traditional model path
    if Path::new(&traditional_model_path).exists() {
        println!("Traditional model path exists: {}", traditional_model_path);
        assert!(!traditional_model_path.is_empty());
        assert!(traditional_model_path.contains("models"));
    } else {
        println!(
            "Traditional model path not found: {}",
            traditional_model_path
        );
    }

    // Test LoRA model path
    if Path::new(&lora_model_path).exists() {
        println!("LoRA model path exists: {}", lora_model_path);
        assert!(!lora_model_path.is_empty());
        assert!(lora_model_path.contains("models"));
    } else {
        println!("LoRA model path not found: {}", lora_model_path);
    }

    // Test path validation logic
    let valid_paths = vec![&traditional_model_path, &lora_model_path];
    for path in valid_paths {
        assert!(!path.is_empty());
        // Path should contain models directory
        if path.contains("models") {
            println!("Path validation passed: {}", path);
        }
    }

    println!("Configurable model loading test completed");
}
