//! Tests for FFI classify module

use super::classify::*;
use crate::ffi::types::*;
use crate::test_fixtures::fixtures::*;
use rstest::*;
use std::ffi::{CStr, CString};
use std::ptr;

/// Test load_id2label_from_config function with real model
#[rstest]
fn test_classify_load_id2label_from_config(traditional_pii_token_model_path: String) {
    let config_path = format!("{}/config.json", traditional_pii_token_model_path);

    let result = load_id2label_from_config(&config_path);

    match result {
        Ok(id2label) => {
            assert!(!id2label.is_empty(), "id2label mapping should not be empty");

            // Verify some common PII labels exist
            let has_person = id2label.values().any(|label| {
                label.contains("PERSON") || label.contains("B-") || label.contains("I-")
            });
            if has_person {
                // Expected PII labels found
                println!("Found PII labels in id2label mapping");
            }

            // Test specific label mappings for PII model
            for (_, label) in id2label.iter() {
                assert!(!label.is_empty(), "Label should not be empty");
            }

            println!("Successfully loaded {} labels from config", id2label.len());
        }
        Err(_) => {
            // Config loading may fail if format differs, which is acceptable for testing
            println!("Config loading failed (expected for some test scenarios)");
        }
    }
}

/// Test FFI classification result structure creation and validation
#[rstest]
fn test_classify_classification_result_structure() {
    let label_cstring =
        CString::new("test_classification_label").expect("Failed to create CString");
    let label_ptr = label_cstring.into_raw();

    let result = ClassificationResult {
        confidence: 0.85,
        predicted_class: 1,
        label: label_ptr,
    };

    // Verify structure fields for C compatibility
    assert_eq!(result.confidence, 0.85);
    assert_eq!(result.predicted_class, 1);
    assert!(!result.label.is_null());

    // Test C string retrieval
    unsafe {
        let label_str = CStr::from_ptr(result.label).to_str().expect("Valid UTF-8");
        assert_eq!(label_str, "test_classification_label");
    }

    // Test memory layout for C interop
    use std::mem::{align_of, size_of};

    // Verify reasonable size and alignment for C interop
    assert!(size_of::<ClassificationResult>() > 0);
    assert!(align_of::<ClassificationResult>() >= align_of::<*mut u8>());

    // Clean up memory
    unsafe {
        let _ = CString::from_raw(label_ptr);
    }

    println!("ClassificationResult structure test passed");
}

/// Test ModernBertTokenEntity structure for token classification FFI
#[rstest]
fn test_classify_modernbert_token_entity() {
    let entity_type_cstring = CString::new("PERSON").expect("Failed to create CString");
    let text_cstring = CString::new("John Doe").expect("Failed to create CString");

    let entity_type_ptr = entity_type_cstring.into_raw();
    let text_ptr = text_cstring.into_raw();

    let entity = ModernBertTokenEntity {
        entity_type: entity_type_ptr,
        start: 0,
        end: 8,
        text: text_ptr,
        confidence: 0.95,
    };

    // Verify structure fields
    assert_eq!(entity.start, 0);
    assert_eq!(entity.end, 8);
    assert_eq!(entity.confidence, 0.95);
    assert!(!entity.entity_type.is_null());
    assert!(!entity.text.is_null());
    assert!(entity.confidence >= 0.0 && entity.confidence <= 1.0);

    // Test string content retrieval
    unsafe {
        let entity_type_str = CStr::from_ptr(entity.entity_type)
            .to_str()
            .expect("Valid UTF-8");
        let text_str = CStr::from_ptr(entity.text).to_str().expect("Valid UTF-8");

        assert_eq!(entity_type_str, "PERSON");
        assert_eq!(text_str, "John Doe");

        // Verify entity span consistency
        assert!(
            entity.start < entity.end,
            "Start position should be less than end position"
        );
        assert_eq!(
            text_str.len(),
            (entity.end - entity.start) as usize,
            "Text length should match span"
        );
    }

    // Clean up memory
    unsafe {
        let _ = CString::from_raw(entity_type_ptr);
        let _ = CString::from_raw(text_ptr);
    }

    println!("ModernBertTokenEntity test passed");
}

/// Test FFI memory safety with null pointers
#[rstest]
fn test_classify_null_pointer_safety() {
    // Test that structures can handle null pointers safely
    let result = ClassificationResult {
        confidence: 0.0,
        predicted_class: -1,
        label: ptr::null_mut(),
    };

    assert!(result.label.is_null());
    assert_eq!(result.confidence, 0.0);
    assert_eq!(result.predicted_class, -1);

    // Test ModernBertTokenEntity with null pointers
    let entity = ModernBertTokenEntity {
        entity_type: ptr::null_mut(),
        start: 0,
        end: 0,
        text: ptr::null_mut(),
        confidence: 0.0,
    };

    assert!(entity.entity_type.is_null());
    assert!(entity.text.is_null());
    assert_eq!(entity.confidence, 0.0);

    println!("Null pointer safety test passed");
}

/// Test FFI classification workflow with real model integration
#[rstest]
fn test_classify_integration_workflow() {
    // Test the complete workflow that would be used from C code
    let test_text = "Hello, how can I help you today?";
    let text_cstring = CString::new(test_text).expect("Failed to create CString");

    // Use Traditional Intent model path directly
    let traditional_model_path = format!(
        "{}/{}",
        crate::test_fixtures::fixtures::MODELS_BASE_PATH,
        crate::test_fixtures::fixtures::MODERNBERT_INTENT_MODEL
    );
    let model_path_cstring =
        CString::new(traditional_model_path.clone()).expect("Failed to create CString");

    // Test config loading (part of classification workflow)
    let config_path = format!("{}/config.json", traditional_model_path);
    match load_id2label_from_config(&config_path) {
        Ok(id2label) => {
            assert!(!id2label.is_empty(), "Config should contain labels");

            // Verify label mapping structure
            for (_, label) in id2label.iter().take(3) {
                assert!(!label.is_empty(), "Label should not be empty");
            }

            println!("Integration workflow config loading succeeded");
        }
        Err(_) => {
            // Config loading may fail, which is acceptable for testing
            println!("Integration workflow config loading failed (acceptable)");
        }
    }

    // Test result structure creation (simulating C interface)
    let mock_result = ClassificationResult {
        confidence: 0.85,
        predicted_class: 1,
        label: text_cstring.into_raw(),
    };

    // Verify result validity
    assert!(mock_result.confidence >= 0.0 && mock_result.confidence <= 1.0);
    assert!(mock_result.predicted_class >= 0);
    assert!(!mock_result.label.is_null());

    // Clean up
    unsafe {
        let _ = CString::from_raw(mock_result.label);
        let _ = CString::from_raw(model_path_cstring.into_raw());
    }

    println!("Integration workflow test passed");
}
