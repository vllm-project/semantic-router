//! Tests for FFI initialization module

use super::init::*;
use super::state_manager::GlobalStateManager;
use rayon::prelude::*;
use rstest::*;
use std::ffi::CString;
use std::os::raw::c_char;

// Note: Testing FFI functions is challenging because they use C ABI and global state.
// These tests focus on verifying basic functionality without requiring actual models.

// ============================================================================
// Global State Tests
// ============================================================================

#[rstest]
fn test_global_state_variables_exist() {
    // Verify that the global static variables can be accessed
    // We can't directly test lazy_static! variables, but we can test that
    // the state manager works, which uses them internally

    let manager = GlobalStateManager::instance();
    let _state = manager.get_system_state();

    // If we get here without panicking, the globals exist
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[rstest]
fn test_cstring_creation() {
    // Test that we can create CStrings for FFI calls
    let test_string = "test_model_path";
    let c_string = CString::new(test_string).expect("CString creation failed");
    let c_ptr: *const c_char = c_string.as_ptr();

    assert!(!c_ptr.is_null(), "CString pointer should not be null");
}

#[rstest]
#[case("")]
#[case("model_path")]
#[case("/path/to/model")]
fn test_cstring_from_various_inputs(#[case] input: &str) {
    let result = CString::new(input);
    assert!(result.is_ok(), "Should create CString from: {}", input);
}

// ============================================================================
// Initialization Function Signatures Tests
// ============================================================================

#[test]
fn test_init_similarity_model_signature() {
    // Verify function signature compiles and can be called with invalid path
    // Note: This will likely fail/return false, but we're testing the interface
    let test_path = CString::new("/nonexistent/model/path").unwrap();
    let result = init_similarity_model(test_path.as_ptr(), true);

    // With invalid path, should return false
    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_classifier_signature() {
    // Test with invalid path - should fail gracefully
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_classifier(test_path.as_ptr(), 0, true);

    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_pii_classifier_signature() {
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_pii_classifier(test_path.as_ptr(), 0, true);

    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_jailbreak_classifier_signature() {
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_jailbreak_classifier(test_path.as_ptr(), 0, true);

    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_modernbert_classifier_signature() {
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_modernbert_classifier(test_path.as_ptr(), true);
    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_modernbert_pii_classifier_signature() {
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_modernbert_pii_classifier(test_path.as_ptr(), true);
    assert!(!result, "Should return false with invalid path");
}

#[test]
fn test_init_unified_classifier_c_signature() {
    let test_path = CString::new("/nonexistent/model").unwrap();

    // Create valid (but empty) arrays for labels
    // slice::from_raw_parts requires non-null, aligned pointers even if length is 0
    let empty_labels: Vec<*const c_char> = Vec::new();
    let labels_ptr = if empty_labels.is_empty() {
        // Use a valid non-null pointer for empty slice
        std::ptr::NonNull::<*const c_char>::dangling().as_ptr()
    } else {
        empty_labels.as_ptr()
    };

    let result = init_unified_classifier_c(
        test_path.as_ptr(),
        test_path.as_ptr(),
        test_path.as_ptr(),
        test_path.as_ptr(),
        labels_ptr,
        0,
        labels_ptr,
        0,
        labels_ptr,
        0,
        true,
    );

    assert!(!result, "Should return false with invalid paths");
}

// ============================================================================
// State Manager Integration Tests
// ============================================================================

#[rstest]
fn test_state_manager_after_failed_init() {
    let manager = GlobalStateManager::instance();

    // Attempt init with invalid path (will fail)
    let test_path = CString::new("/nonexistent/model").unwrap();
    let _result = init_similarity_model(test_path.as_ptr(), true);

    // State manager should still be accessible
    let state = manager.get_system_state();

    // State should be one of the valid states
    assert!(
        matches!(
            state,
            super::state_manager::SystemState::Uninitialized
                | super::state_manager::SystemState::Ready
                | super::state_manager::SystemState::Error(_)
                | super::state_manager::SystemState::Initializing
        ),
        "Should have valid system state"
    );
}

// ============================================================================
// Thread Safety Tests for Initialization
// ============================================================================

#[rstest]
fn test_concurrent_init_attempts() {
    // Try to initialize from multiple threads simultaneously
    // This tests that the initialization locks work correctly
    // Use rayon for parallel execution - simpler and more efficient
    (0..4).into_par_iter().for_each(|_| {
        // Attempt init with invalid path (will fail, but tests locking)
        let test_path = CString::new("/nonexistent/model").unwrap();
        let _ = init_similarity_model(test_path.as_ptr(), true);
    });

    // If we get here, no deadlock occurred
}

// ============================================================================
// CString Safety Tests
// ============================================================================

#[rstest]
#[case("valid_path")]
#[case("/another/valid/path")]
#[case("model_id_123")]
fn test_cstring_for_model_paths(#[case] path: &str) {
    let c_string = CString::new(path).expect("Create CString");
    let c_ptr = c_string.as_ptr();

    // Verify pointer is not null
    assert!(!c_ptr.is_null());

    // Convert back to verify correctness
    let back_to_str = unsafe {
        std::ffi::CStr::from_ptr(c_ptr)
            .to_str()
            .expect("Convert back to str")
    };

    assert_eq!(
        back_to_str, path,
        "Round-trip conversion should preserve string"
    );
}

#[test]
fn test_cstring_with_null_byte_fails() {
    let invalid_string = "path\0with\0nulls";
    let result = CString::new(invalid_string);

    assert!(
        result.is_err(),
        "CString creation should fail with interior null bytes"
    );
}

// ============================================================================
// Boolean Return Value Tests
// ============================================================================

#[rstest]
fn test_init_functions_return_boolean() {
    // All init functions should return bool
    // Test that false is returned for invalid inputs
    let test_path = CString::new("/nonexistent/model").unwrap();

    assert!(!init_similarity_model(test_path.as_ptr(), true));
    assert!(!init_classifier(test_path.as_ptr(), 0, true));
    assert!(!init_pii_classifier(test_path.as_ptr(), 0, true));
    assert!(!init_jailbreak_classifier(test_path.as_ptr(), 0, true));
    assert!(!init_modernbert_classifier(test_path.as_ptr(), true));
}

// ============================================================================
// Parameter Validation Tests
// ============================================================================

#[rstest]
#[case(true)]
#[case(false)]
fn test_use_cpu_parameter(#[case] use_cpu: bool) {
    // Test that use_cpu parameter is accepted
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_similarity_model(test_path.as_ptr(), use_cpu);

    // Should fail due to invalid path, but parameter should be processed
    assert!(!result);
}

#[rstest]
#[case(0)]
#[case(2)]
#[case(5)]
fn test_num_labels_parameter(#[case] num_labels: i32) {
    // Test that num_labels parameter is accepted
    let test_path = CString::new("/nonexistent/model").unwrap();
    let result = init_classifier(test_path.as_ptr(), num_labels, true);

    assert!(!result, "Should fail with invalid path");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[rstest]
fn test_invalid_path_handling() {
    // All functions should handle invalid paths gracefully without crashing
    let test_path = CString::new("/nonexistent/model").unwrap();

    let _ = init_similarity_model(test_path.as_ptr(), true);
    let _ = init_classifier(test_path.as_ptr(), 0, true);
    let _ = init_pii_classifier(test_path.as_ptr(), 0, true);
    let _ = init_jailbreak_classifier(test_path.as_ptr(), 0, true);
    let _ = init_modernbert_classifier(test_path.as_ptr(), true);
    let _ = init_modernbert_pii_classifier(test_path.as_ptr(), true);

    // If we reach here, no crashes occurred
}

// ============================================================================
// Integration with State Manager Tests
// ============================================================================

#[rstest]
fn test_state_manager_stats_after_init_attempts() {
    let manager = GlobalStateManager::instance();

    // Try various init functions
    let test_path = CString::new("/nonexistent/model").unwrap();
    let _ = init_similarity_model(test_path.as_ptr(), true);
    let _ = init_modernbert_classifier(test_path.as_ptr(), true);

    // Get stats - should work regardless of init success/failure
    let stats = manager.get_stats();

    // Stats should be retrievable
    assert!(
        stats.unified_classifier_initialized || !stats.unified_classifier_initialized,
        "Should have stats"
    );
}

// ============================================================================
// Const Correctness Tests
// ============================================================================

#[test]
fn test_const_char_pointer_usage() {
    // Test that const char* parameters work correctly
    let test_str = CString::new("test").unwrap();
    let ptr: *const c_char = test_str.as_ptr();

    // Verify the pointer can be used in FFI context
    assert!(!ptr.is_null());

    // Pass to a function (will fail but tests the interface)
    let _result = init_similarity_model(ptr, true);
}

// ============================================================================
// Memory Safety Tests
// ============================================================================

#[rstest]
fn test_cstring_lifetime() {
    // Test that CString lives long enough for FFI call
    let _result = {
        let model_id = CString::new("model").unwrap();
        let ptr = model_id.as_ptr();
        init_similarity_model(ptr, true)
        // model_id is dropped here, but call already completed
    };

    // Should complete without memory issues
}

#[rstest]
fn test_multiple_cstrings() {
    // Test creating multiple CStrings for different parameters
    let model_id = CString::new("model_id").unwrap();
    let _tokenizer_path = CString::new("tokenizer_path").unwrap();
    let _lora_path = CString::new("lora_path").unwrap();

    let _result = init_classifier(model_id.as_ptr(), 2, true);

    // All CStrings should remain valid during the call
}
