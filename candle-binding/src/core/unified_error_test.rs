//! Tests for unified_error module

use super::unified_error::*;
use rstest::*;

/// Test UnifiedError creation and formatting
#[rstest]
#[case("config_load", "Invalid JSON format", Some("file: config.json".to_string()), "Configuration")]
#[case("model_init", "Model not found", None, "Model")]
#[case("tensor_op", "Shape mismatch", Some("input shape: [1, 768]".to_string()), "Processing")]
fn test_unified_error_unified_error_creation_and_formatting(
    #[case] operation: &str,
    #[case] message: &str,
    #[case] context: Option<String>,
    #[case] error_type: &str,
) {
    let error = match error_type {
        "Configuration" => UnifiedError::Configuration {
            operation: operation.to_string(),
            source: ConfigErrorType::InvalidData(message.to_string()),
            context: context.clone(),
        },
        "Model" => UnifiedError::Model {
            model_type: ModelErrorType::Traditional,
            operation: operation.to_string(),
            source: message.to_string(),
            context: context.clone(),
        },
        "Processing" => UnifiedError::Processing {
            operation: operation.to_string(),
            source: message.to_string(),
            input_context: context.clone(),
        },
        _ => panic!("Unknown error type: {}", error_type),
    };

    // Test error formatting
    let error_string = format!("{}", error);
    assert!(!error_string.is_empty(), "Error string should not be empty");
    assert!(
        error_string.contains(operation),
        "Error should contain operation name"
    );
    assert!(
        error_string.contains(message),
        "Error should contain error message"
    );

    if let Some(ref ctx) = context {
        assert!(
            error_string.contains(ctx),
            "Error should contain context if provided"
        );
    }

    println!("Error formatted as: {}", error_string);
}

/// Test error conversion from standard library errors
#[rstest]
fn test_unified_error_error_conversions() {
    // Test conversion from std::io::Error
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let unified_error: UnifiedError = io_error.into();

    match unified_error {
        UnifiedError::IO {
            operation, source, ..
        } => {
            assert_eq!(operation, "I/O operation");
            assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
            println!("IO error conversion test passed");
        }
        _ => panic!("Expected IO error variant"),
    }

    // Test conversion from serde_json::Error
    let json_error = serde_json::from_str::<serde_json::Value>("{invalid json}").unwrap_err();
    let unified_error: UnifiedError = json_error.into();

    match unified_error {
        UnifiedError::Configuration {
            operation, source, ..
        } => {
            assert_eq!(operation, "JSON parsing");
            match source {
                ConfigErrorType::ParseError(_) => println!("JSON error conversion test passed"),
                _ => panic!("Expected ParseError variant"),
            }
        }
        _ => panic!("Expected Configuration error variant"),
    }
}

/// Test error helper functions
#[rstest]
fn test_unified_error_error_helper_functions() {
    // Test config_errors module functions
    let file_not_found_err = config_errors::file_not_found("config.json");
    match file_not_found_err {
        UnifiedError::Configuration {
            source: ConfigErrorType::FileNotFound(path),
            ..
        } => {
            assert_eq!(path, "config.json");
            println!("file_not_found helper test passed");
        }
        _ => panic!("Expected FileNotFound error"),
    }

    let missing_field_err = config_errors::missing_field("num_classes", "config.json");
    match missing_field_err {
        UnifiedError::Configuration {
            source: ConfigErrorType::MissingField(field),
            context,
            ..
        } => {
            assert_eq!(field, "num_classes");
            assert!(context.is_some());
            println!("missing_field helper test passed");
        }
        _ => panic!("Expected MissingField error"),
    }

    let invalid_json_err = config_errors::invalid_json("config.json", "Unexpected token");
    match invalid_json_err {
        UnifiedError::Configuration {
            source: ConfigErrorType::ParseError(_),
            ..
        } => {
            println!("invalid_json helper test passed");
        }
        _ => panic!("Expected ParseError error"),
    }

    // Test model_errors module functions
    let load_failure_err =
        model_errors::load_failure(ModelErrorType::Traditional, "model.bin", "File corrupted");
    match load_failure_err {
        UnifiedError::Model {
            model_type: ModelErrorType::Traditional,
            operation,
            ..
        } => {
            assert_eq!(operation, "model loading");
            println!("load_failure helper test passed");
        }
        _ => panic!("Expected Model error"),
    }

    let inference_failure_err = model_errors::inference_failure(
        ModelErrorType::LoRA,
        "input: [1, 768]",
        "CUDA out of memory",
    );
    match inference_failure_err {
        UnifiedError::Model {
            model_type: ModelErrorType::LoRA,
            operation,
            ..
        } => {
            assert_eq!(operation, "model inference");
            println!("inference_failure helper test passed");
        }
        _ => panic!("Expected Model error"),
    }

    let tokenizer_failure_err = model_errors::tokenizer_failure("Vocabulary file missing");
    match tokenizer_failure_err {
        UnifiedError::Model {
            model_type: ModelErrorType::Tokenizer,
            ..
        } => {
            println!("tokenizer_failure helper test passed");
        }
        _ => panic!("Expected Tokenizer error"),
    }
}
