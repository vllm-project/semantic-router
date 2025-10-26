//! FFI Validation Functions
//!
//! This module provides comprehensive parameter validation for dual-path architecture.
//! Ensures safety and security for both LoRA and Traditional paths.

use std::ffi::{c_char, CStr, CString};

/// Validation result for parameter checking
#[repr(C)]
pub struct ValidationResult {
    /// Validation success (true) or failure (false)
    pub is_valid: bool,
    /// Error code (0 = success, >0 = specific error)
    pub error_code: i32,
    /// Human-readable error message
    pub error_message: *mut c_char,
    /// Suggested fixes or recommendations
    pub suggestions: *mut c_char,
}

/// Error codes for validation failures
pub const VALIDATION_SUCCESS: i32 = 0;
pub const ERROR_NULL_POINTER: i32 = 1;
pub const ERROR_INVALID_STRING: i32 = 2;
pub const ERROR_TEXT_TOO_LONG: i32 = 3;
pub const ERROR_TEXT_TOO_SHORT: i32 = 4;
pub const ERROR_INVALID_BATCH_SIZE: i32 = 5;
pub const ERROR_INVALID_CONFIDENCE: i32 = 6;
pub const ERROR_INVALID_MODEL_PATH: i32 = 7;
pub const ERROR_UNSUPPORTED_ENCODING: i32 = 8;
pub const ERROR_MEMORY_ALLOCATION: i32 = 9;
pub const ERROR_LORA_SPECIFIC: i32 = 100;
pub const ERROR_TRADITIONAL_SPECIFIC: i32 = 200;

/// Maximum text length for processing (characters)
pub const MAX_TEXT_LENGTH: usize = 10000;
/// Minimum text length for meaningful processing
pub const MIN_TEXT_LENGTH: usize = 1;
/// Maximum batch size for processing
pub const MAX_BATCH_SIZE: i32 = 1000;
/// Maximum model path length
pub const MAX_MODEL_PATH_LENGTH: usize = 1000;

/// Validate text input for classification
///
/// # Safety
/// - `text` must be a valid null-terminated C string or null
/// - `path_type` should be 0 (Traditional) or 1 (LoRA)
#[no_mangle]
pub extern "C" fn validate_text_input(text: *const c_char, path_type: i32) -> ValidationResult {
    // Check for null pointer
    if text.is_null() {
        return create_validation_error(
            ERROR_NULL_POINTER,
            "Text input is null",
            "Provide a valid non-null text string",
        );
    }

    // Convert C string to Rust string
    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return create_validation_error(
                    ERROR_INVALID_STRING,
                    "Text contains invalid UTF-8 characters",
                    "Ensure text is valid UTF-8 encoded",
                )
            }
        }
    };

    // Check text length
    if text_str.len() < MIN_TEXT_LENGTH {
        return create_validation_error(
            ERROR_TEXT_TOO_SHORT,
            "Text is too short for meaningful processing",
            &format!("Provide text with at least {} characters", MIN_TEXT_LENGTH),
        );
    }

    if text_str.len() > MAX_TEXT_LENGTH {
        return create_validation_error(
            ERROR_TEXT_TOO_LONG,
            "Text exceeds maximum length limit",
            &format!("Limit text to {} characters or less", MAX_TEXT_LENGTH),
        );
    }

    // Path-specific validation
    match path_type {
        0 => validate_traditional_text(text_str),
        1 => validate_lora_text(text_str),
        _ => create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Invalid path type specified",
            "Use 0 for Traditional path or 1 for LoRA path",
        ),
    }
}

/// Validate batch input for classification
///
/// # Safety
/// - `texts` must be a valid array of null-terminated C strings or null
/// - `texts_count` must match the actual array size
/// - `path_type` should be 0 (Traditional) or 1 (LoRA)
#[no_mangle]
pub extern "C" fn validate_batch_input(
    texts: *const *const c_char,
    texts_count: i32,
    path_type: i32,
) -> ValidationResult {
    // Check for null pointer
    if texts.is_null() {
        return create_validation_error(
            ERROR_NULL_POINTER,
            "Texts array is null",
            "Provide a valid non-null array of text strings",
        );
    }

    // Check batch size
    if texts_count <= 0 {
        return create_validation_error(
            ERROR_INVALID_BATCH_SIZE,
            "Batch size must be positive",
            "Provide at least one text for batch processing",
        );
    }

    if texts_count > MAX_BATCH_SIZE {
        return create_validation_error(
            ERROR_INVALID_BATCH_SIZE,
            "Batch size exceeds maximum limit",
            &format!("Limit batch size to {} items or less", MAX_BATCH_SIZE),
        );
    }

    // Validate each text in the batch
    for i in 0..texts_count {
        let text_ptr = unsafe { *texts.offset(i as isize) };
        let validation_result = validate_text_input(text_ptr, path_type);

        if !validation_result.is_valid {
            // Add batch context to error message
            let enhanced_message = format!("Batch item {}: {}", i, unsafe {
                CStr::from_ptr(validation_result.error_message).to_string_lossy()
            });

            // Free the original error message
            if !validation_result.error_message.is_null() {
                unsafe {
                    let _ = CString::from_raw(validation_result.error_message);
                }
            }

            return create_validation_error(
                validation_result.error_code,
                &enhanced_message,
                "Fix the invalid item in the batch",
            );
        }

        // Free successful validation result
        free_validation_result(validation_result);
    }

    // Path-specific batch validation
    match path_type {
        0 => validate_traditional_batch(texts_count),
        1 => validate_lora_batch(texts_count),
        _ => create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Invalid path type for batch processing",
            "Use 0 for Traditional path or 1 for LoRA path",
        ),
    }
}

/// Validate model path for initialization
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string or null
/// - `path_type` should be 0 (Traditional) or 1 (LoRA)
#[no_mangle]
pub extern "C" fn validate_model_path(
    model_path: *const c_char,
    path_type: i32,
) -> ValidationResult {
    // Check for null pointer
    if model_path.is_null() {
        return create_validation_error(
            ERROR_NULL_POINTER,
            "Model path is null",
            "Provide a valid model directory path",
        );
    }

    // Convert C string to Rust string
    let path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => {
                return create_validation_error(
                    ERROR_INVALID_STRING,
                    "Model path contains invalid UTF-8 characters",
                    "Ensure model path is valid UTF-8 encoded",
                )
            }
        }
    };

    // Check path length
    if path_str.len() > MAX_MODEL_PATH_LENGTH {
        return create_validation_error(
            ERROR_INVALID_MODEL_PATH,
            "Model path exceeds maximum length",
            &format!("Limit model path to {} characters", MAX_MODEL_PATH_LENGTH),
        );
    }

    // Basic path validation (existence check would require filesystem access)
    if path_str.is_empty() {
        return create_validation_error(
            ERROR_INVALID_MODEL_PATH,
            "Model path is empty",
            "Provide a non-empty model directory path",
        );
    }

    // Path-specific validation
    match path_type {
        0 => validate_traditional_model_path(path_str),
        1 => validate_lora_model_path(path_str),
        _ => create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Invalid path type for model validation",
            "Use 0 for Traditional path or 1 for LoRA path",
        ),
    }
}

/// Validate confidence threshold values
///
/// # Safety
/// - `confidence` should be between 0.0 and 1.0
/// - `path_type` should be 0 (Traditional) or 1 (LoRA)
#[no_mangle]
pub extern "C" fn validate_confidence_threshold(
    confidence: f32,
    path_type: i32,
) -> ValidationResult {
    // Check confidence range
    if confidence < 0.0 || confidence > 1.0 {
        return create_validation_error(
            ERROR_INVALID_CONFIDENCE,
            "Confidence threshold must be between 0.0 and 1.0",
            "Use a confidence value in the range [0.0, 1.0]",
        );
    }

    // Path-specific confidence validation
    match path_type {
        0 => {
            // Traditional path: typically 0.5-0.95
            if confidence < 0.5 {
                return create_validation_error(
                    ERROR_TRADITIONAL_SPECIFIC,
                    "Traditional path confidence threshold too low",
                    "Consider using confidence >= 0.5 for Traditional models",
                );
            }
            create_validation_success()
        }
        1 => {
            // LoRA path: typically 0.8-0.99+
            if confidence < 0.8 {
                return create_validation_error(
                    ERROR_LORA_SPECIFIC,
                    "LoRA path confidence threshold too low",
                    "Consider using confidence >= 0.8 for LoRA models",
                );
            }
            create_validation_success()
        }
        _ => create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Invalid path type for confidence validation",
            "Use 0 for Traditional path or 1 for LoRA path",
        ),
    }
}

/// Validate memory allocation parameters
///
/// # Safety
/// - `size` should be a reasonable memory size
/// - `alignment` should be a valid alignment value
#[no_mangle]
pub extern "C" fn validate_memory_parameters(size: usize, alignment: usize) -> ValidationResult {
    // Check for zero size
    if size == 0 {
        return create_validation_error(
            ERROR_MEMORY_ALLOCATION,
            "Memory allocation size cannot be zero",
            "Specify a positive memory size",
        );
    }

    // Check for reasonable size limits (e.g., 1GB max)
    const MAX_MEMORY_SIZE: usize = 1024 * 1024 * 1024; // 1GB
    if size > MAX_MEMORY_SIZE {
        return create_validation_error(
            ERROR_MEMORY_ALLOCATION,
            "Memory allocation size exceeds reasonable limits",
            &format!("Limit memory allocation to {} bytes", MAX_MEMORY_SIZE),
        );
    }

    // Check alignment (must be power of 2)
    if alignment == 0 || (alignment & (alignment - 1)) != 0 {
        return create_validation_error(
            ERROR_MEMORY_ALLOCATION,
            "Memory alignment must be a power of 2",
            "Use alignment values like 1, 2, 4, 8, 16, etc.",
        );
    }

    create_validation_success()
}

/// Free validation result memory
///
/// # Safety
/// - `result` must be a valid ValidationResult
/// - Only call once per result
#[no_mangle]
pub extern "C" fn free_validation_result(result: ValidationResult) {
    if !result.error_message.is_null() {
        unsafe {
            let _ = CString::from_raw(result.error_message);
        }
    }
    if !result.suggestions.is_null() {
        unsafe {
            let _ = CString::from_raw(result.suggestions);
        }
    }
}

// Helper functions for path-specific validation

fn validate_traditional_text(text: &str) -> ValidationResult {
    // Traditional path specific validation
    // Check for potentially problematic characters or patterns
    if text
        .chars()
        .any(|c| c.is_control() && c != '\n' && c != '\r' && c != '\t')
    {
        return create_validation_error(
            ERROR_TRADITIONAL_SPECIFIC,
            "Text contains control characters that may cause issues",
            "Remove or replace control characters in the text",
        );
    }

    create_validation_success()
}

fn validate_lora_text(text: &str) -> ValidationResult {
    // LoRA path specific validation
    // LoRA models may have different requirements or optimizations

    // Check for very short texts that might not benefit from LoRA processing
    if text.len() < 10 {
        return create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Text may be too short for optimal LoRA processing",
            "Consider using Traditional path for very short texts",
        );
    }

    create_validation_success()
}

fn validate_traditional_batch(batch_size: i32) -> ValidationResult {
    // Traditional path batch validation
    // Traditional models may have different batch size limitations
    if batch_size > 100 {
        return create_validation_error(
            ERROR_TRADITIONAL_SPECIFIC,
            "Large batch sizes may cause memory issues with Traditional models",
            "Consider reducing batch size or using LoRA path for large batches",
        );
    }

    create_validation_success()
}

fn validate_lora_batch(batch_size: i32) -> ValidationResult {
    // LoRA path batch validation
    // LoRA models are optimized for parallel processing
    if batch_size == 1 {
        return create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Single item batches don't utilize LoRA parallel processing advantages",
            "Consider using Traditional path for single items or increase batch size",
        );
    }

    create_validation_success()
}

fn validate_traditional_model_path(path: &str) -> ValidationResult {
    // Traditional model path validation
    // Check for expected file patterns
    if !path.contains("traditional") && !path.contains("bert") && !path.contains("modernbert") {
        return create_validation_error(
            ERROR_TRADITIONAL_SPECIFIC,
            "Model path doesn't appear to be a Traditional model",
            "Ensure the path points to a Traditional model directory",
        );
    }

    create_validation_success()
}

fn validate_lora_model_path(path: &str) -> ValidationResult {
    // LoRA model path validation
    // Check for expected LoRA file patterns
    if !path.contains("lora") && !path.contains("adapter") {
        return create_validation_error(
            ERROR_LORA_SPECIFIC,
            "Model path doesn't appear to be a LoRA model",
            "Ensure the path points to a LoRA model directory with adapter files",
        );
    }

    create_validation_success()
}

// Helper functions for creating validation results

fn create_validation_success() -> ValidationResult {
    ValidationResult {
        is_valid: true,
        error_code: VALIDATION_SUCCESS,
        error_message: std::ptr::null_mut(),
        suggestions: std::ptr::null_mut(),
    }
}

fn create_validation_error(error_code: i32, message: &str, suggestion: &str) -> ValidationResult {
    let error_message =
        CString::new(message).unwrap_or_else(|_| CString::new("Unknown error").unwrap());
    let suggestions = CString::new(suggestion)
        .unwrap_or_else(|_| CString::new("No suggestions available").unwrap());

    ValidationResult {
        is_valid: false,
        error_code,
        error_message: error_message.into_raw(),
        suggestions: suggestions.into_raw(),
    }
}
