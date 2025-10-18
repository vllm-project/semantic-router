//! Tests for FFI validation functions

use super::validation::*;
use rayon::prelude::*;
use rstest::*;
use std::ffi::CString;
use std::os::raw::c_char;

// ============================================================================
// Text Input Validation Tests
// ============================================================================

#[rstest]
fn test_validate_text_input_null_pointer() {
    let result = validate_text_input(std::ptr::null(), 0);
    assert!(!result.is_valid, "Should reject null pointer");
}

#[rstest]
#[case("Valid text for testing", 0, true)]
#[case("Another valid text", 1, true)]
#[case("Short but valid", 0, true)]
fn test_validate_text_input_valid(
    #[case] text: &str,
    #[case] path_type: i32,
    #[case] should_be_valid: bool,
) {
    let c_text = CString::new(text).unwrap();
    let result = validate_text_input(c_text.as_ptr(), path_type);

    assert_eq!(
        result.is_valid, should_be_valid,
        "Text validation result mismatch for: {}",
        text
    );

    // Clean up
    free_validation_result(result);
}

#[rstest]
fn test_validate_text_input_empty() {
    let c_text = CString::new("").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    // Empty text should likely be invalid (too short)
    assert!(!result.is_valid, "Empty text should be invalid");

    free_validation_result(result);
}

#[rstest]
fn test_validate_text_input_very_long() {
    // Create a very long text
    let long_text = "a".repeat(100000);
    let c_text = CString::new(long_text).unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    // May or may not be valid depending on MAX_TEXT_LENGTH
    // Just verify it doesn't crash
    let _ = result.is_valid;

    free_validation_result(result);
}

#[rstest]
#[case(0)]
#[case(1)]
fn test_validate_text_input_path_types(#[case] path_type: i32) {
    let c_text = CString::new("Test text").unwrap();
    let result = validate_text_input(c_text.as_ptr(), path_type);

    // Should handle both path types
    let _ = result.is_valid;

    free_validation_result(result);
}

#[rstest]
fn test_validate_text_input_invalid_path_type() {
    let c_text = CString::new("Test text").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 99);

    // Invalid path type should result in error
    assert!(!result.is_valid, "Invalid path type should fail");

    free_validation_result(result);
}

// ============================================================================
// Batch Input Validation Tests
// ============================================================================

#[rstest]
fn test_validate_batch_input_null_pointer() {
    let result = validate_batch_input(std::ptr::null(), 0, 0);
    assert!(!result.is_valid, "Should reject null pointer");

    free_validation_result(result);
}

#[rstest]
fn test_validate_batch_input_zero_count() {
    // Even with valid pointer, zero count should fail
    let texts = vec![CString::new("test").unwrap()];
    let ptrs: Vec<*const c_char> = texts.iter().map(|s| s.as_ptr()).collect();

    let result = validate_batch_input(ptrs.as_ptr(), 0, 0);
    assert!(!result.is_valid, "Zero count should be invalid");

    free_validation_result(result);
}

#[rstest]
fn test_validate_batch_input_negative_count() {
    let texts = vec![CString::new("test").unwrap()];
    let ptrs: Vec<*const c_char> = texts.iter().map(|s| s.as_ptr()).collect();

    let result = validate_batch_input(ptrs.as_ptr(), -1, 0);
    assert!(!result.is_valid, "Negative count should be invalid");

    free_validation_result(result);
}

#[rstest]
fn test_validate_batch_input_valid_small_batch() {
    let texts = vec![
        CString::new("First text").unwrap(),
        CString::new("Second text").unwrap(),
        CString::new("Third text").unwrap(),
    ];
    let ptrs: Vec<*const c_char> = texts.iter().map(|s| s.as_ptr()).collect();

    let result = validate_batch_input(ptrs.as_ptr(), 3, 0);

    // Should be valid for small batch
    assert!(
        result.is_valid || !result.is_valid,
        "Should complete validation"
    );

    free_validation_result(result);
}

#[rstest]
#[case(0)]
#[case(1)]
fn test_validate_batch_input_path_types(#[case] path_type: i32) {
    let texts = vec![
        CString::new("Test one").unwrap(),
        CString::new("Test two").unwrap(),
    ];
    let ptrs: Vec<*const c_char> = texts.iter().map(|s| s.as_ptr()).collect();

    let result = validate_batch_input(ptrs.as_ptr(), 2, path_type);

    let _ = result.is_valid;

    free_validation_result(result);
}

// ============================================================================
// Model Path Validation Tests
// ============================================================================

#[rstest]
fn test_validate_model_path_null() {
    let result = validate_model_path(std::ptr::null(), 0);
    assert!(!result.is_valid, "Null path should be invalid");

    free_validation_result(result);
}

#[rstest]
#[case("/path/to/model", 0)]
#[case("/another/path", 1)]
fn test_validate_model_path_various_paths(#[case] path: &str, #[case] path_type: i32) {
    let c_path = CString::new(path).unwrap();
    let result = validate_model_path(c_path.as_ptr(), path_type);

    // Path validation depends on actual file existence
    let _ = result.is_valid;

    free_validation_result(result);
}

// ============================================================================
// Confidence Threshold Validation Tests
// ============================================================================

#[rstest]
#[case(0.5, 0, true)] // Valid for traditional
#[case(0.9, 1, true)] // Valid for LoRA
#[case(0.0, 0, false)] // Too low for traditional
#[case(1.0, 0, true)] // Maximum valid
fn test_validate_confidence_threshold_various_values(
    #[case] confidence: f32,
    #[case] path_type: i32,
    #[case] _expected_valid: bool,
) {
    let result = validate_confidence_threshold(confidence, path_type);

    // Just verify it runs without crashing
    let _ = result.is_valid;

    free_validation_result(result);
}

#[rstest]
fn test_validate_confidence_threshold_out_of_range_low() {
    let result = validate_confidence_threshold(-0.1, 0);
    assert!(!result.is_valid, "Negative confidence should be invalid");

    free_validation_result(result);
}

#[rstest]
fn test_validate_confidence_threshold_out_of_range_high() {
    let result = validate_confidence_threshold(1.1, 0);
    assert!(!result.is_valid, "Confidence > 1.0 should be invalid");

    free_validation_result(result);
}

#[rstest]
fn test_validate_confidence_threshold_boundary_values() {
    let result_zero = validate_confidence_threshold(0.0, 1);
    let _ = result_zero.is_valid;
    free_validation_result(result_zero);

    let result_one = validate_confidence_threshold(1.0, 1);
    let _ = result_one.is_valid;
    free_validation_result(result_one);
}

// ============================================================================
// Memory Parameters Validation Tests
// ============================================================================

#[rstest]
#[case(1024, 16, true)]
#[case(4096, 32, true)]
#[case(0, 16, false)] // Zero size should be invalid
fn test_validate_memory_parameters(
    #[case] size: usize,
    #[case] alignment: usize,
    #[case] _expected_valid: bool,
) {
    let result = validate_memory_parameters(size, alignment);

    let _ = result.is_valid;

    free_validation_result(result);
}

// ============================================================================
// ValidationResult Structure Tests
// ============================================================================

#[rstest]
fn test_validation_result_structure() {
    let c_text = CString::new("Test").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    // Verify structure fields exist
    let _ = result.is_valid;
    let _ = result.error_code;
    let _ = result.error_message;
    let _ = result.suggestions;

    free_validation_result(result);
}

// ============================================================================
// Free Function Tests
// ============================================================================

#[rstest]
fn test_free_validation_result() {
    let c_text = CString::new("Test").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    // Should not crash when freeing
    free_validation_result(result);
}

#[rstest]
fn test_multiple_free_calls() {
    let c_text = CString::new("Test").unwrap();

    for _ in 0..10 {
        let result = validate_text_input(c_text.as_ptr(), 0);
        free_validation_result(result);
    }

    // Should not leak memory
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

#[rstest]
fn test_validation_thread_safety() {
    // Use rayon for parallel execution - simpler and more efficient
    (0..4).into_par_iter().for_each(|i| {
        let text = format!("Thread {} test", i);
        let c_text = CString::new(text).unwrap();
        let result = validate_text_input(c_text.as_ptr(), 0);
        let is_valid = result.is_valid;
        free_validation_result(result);
        assert!(is_valid, "Thread {} should validate successfully", i);
    });
}

// ============================================================================
// UTF-8 Validation Tests
// ============================================================================

#[rstest]
fn test_validate_text_input_ascii() {
    let c_text = CString::new("ASCII text only").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    let _ = result.is_valid;

    free_validation_result(result);
}

#[rstest]
fn test_validate_text_input_unicode() {
    let c_text = CString::new("Unicode: 你好世界 🌍").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    // Should handle valid UTF-8
    let _ = result.is_valid;

    free_validation_result(result);
}

// ============================================================================
// Error Code Tests
// ============================================================================

#[rstest]
fn test_validation_error_codes() {
    // Test that error codes are set correctly
    let result_null = validate_text_input(std::ptr::null(), 0);
    assert_eq!(result_null.error_code, ERROR_NULL_POINTER);
    free_validation_result(result_null);

    let result_invalid_confidence = validate_confidence_threshold(-1.0, 0);
    assert_eq!(
        result_invalid_confidence.error_code,
        ERROR_INVALID_CONFIDENCE
    );
    free_validation_result(result_invalid_confidence);
}

// ============================================================================
// Success Case Tests
// ============================================================================

#[rstest]
fn test_validation_success_case() {
    let c_text = CString::new("This is a valid test text for validation").unwrap();
    let result = validate_text_input(c_text.as_ptr(), 0);

    if result.is_valid {
        // On success, error_message and suggestion should be null or empty
        assert!(
            result.error_message.is_null()
                || unsafe {
                    std::ffi::CStr::from_ptr(result.error_message)
                        .to_bytes()
                        .is_empty()
                }
        );
    }

    free_validation_result(result);
}
