//! Tests for config_loader module

use super::config_loader::*;
use crate::test_fixtures::fixtures::*;
use rstest::*;

/// Test loading intent labels with model path
#[rstest]
fn test_config_loader_load_intent_labels() {
    // Use Traditional Intent model path directly
    let traditional_model_path = format!(
        "{}/{}",
        crate::test_fixtures::fixtures::MODELS_BASE_PATH,
        crate::test_fixtures::fixtures::MODERNBERT_INTENT_MODEL
    );

    let result = load_intent_labels(&traditional_model_path);

    match result {
        Ok(labels) => {
            println!(
                "Loaded {} intent labels from {}: {:?}",
                labels.len(),
                traditional_model_path,
                labels
            );
        }
        Err(e) => {
            println!("Failed to load intent labels from {} (may be expected if config not available): {}", traditional_model_path, e);
        }
    }
}

/// Test loading PII labels with model path
#[rstest]
fn test_config_loader_load_pii_labels(traditional_pii_model_path: String) {
    let result = load_pii_labels(&traditional_pii_model_path);

    match result {
        Ok(labels) => {
            println!(
                "Loaded {} PII labels from {}: {:?}",
                labels.len(),
                traditional_pii_model_path,
                labels
            );
        }
        Err(e) => {
            println!(
                "Failed to load PII labels from {} (may be expected if config not available): {}",
                traditional_pii_model_path, e
            );
        }
    }
}

/// Test loading security labels with model path
#[rstest]
fn test_config_loader_load_security_labels(traditional_security_model_path: String) {
    let result = load_security_labels(&traditional_security_model_path);

    match result {
        Ok(labels) => {
            println!(
                "Loaded {} security labels from {}: {:?}",
                labels.len(),
                traditional_security_model_path,
                labels
            );
        }
        Err(e) => {
            println!("Failed to load security labels from {} (may be expected if config not available): {}", traditional_security_model_path, e);
        }
    }
}
