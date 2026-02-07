//! FFI bindings for Qwen3 Multi-LoRA Generative Classifier and Qwen3Guard
//!
//! Exposes the Qwen3 multi-adapter system and Qwen3Guard to Go via C ABI.
//!
//! This module provides a thread-safe interface for:
//! - Loading a base Qwen3 model with multiple LoRA adapters
//! - Loading Qwen3Guard model for safety classification
//! - Classifying text with different adapters
//! - Detecting jailbreaks and unsafe content
//! - Managing model lifecycle

use crate::ffi::types::ClassificationResult;
use crate::model_architectures::generative::{Qwen3GuardModel, Qwen3MultiLoRAClassifier};
use candle_core::Device;
use serde_json::Value;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Global multi-adapter classifier instance (for LoRA-based classification)
static GLOBAL_QWEN3_MULTI_CLASSIFIER: OnceLock<Mutex<Qwen3MultiLoRAClassifier>> = OnceLock::new();

/// Global Qwen3Guard instance (for safety/jailbreak detection)
static GLOBAL_QWEN3_GUARD: OnceLock<Mutex<Qwen3GuardModel>> = OnceLock::new();

/// Global Qwen3 zero-shot classifier for preference routing
static GLOBAL_QWEN3_PREFERENCE: OnceLock<Mutex<Qwen3MultiLoRAClassifier>> = OnceLock::new();

/// Generative classification result returned to Go
#[repr(C)]
pub struct GenerativeClassificationResult {
    /// Predicted class index
    pub class_id: i32,

    /// Confidence score (probability)
    pub confidence: f32,

    /// Category name (null-terminated C string, must be freed by caller)
    pub category_name: *mut c_char,

    /// Probabilities for all categories (array, must be freed by caller)
    pub probabilities: *mut f32,

    /// Number of categories
    pub num_categories: i32,

    /// Error flag (true if error occurred)
    pub error: bool,

    /// Error message (null-terminated C string, only set if error=true, must be freed by caller)
    pub error_message: *mut c_char,
}

impl Default for GenerativeClassificationResult {
    fn default() -> Self {
        Self {
            class_id: -1,
            confidence: 0.0,
            category_name: ptr::null_mut(),
            probabilities: ptr::null_mut(),
            num_categories: 0,
            error: true,
            error_message: ptr::null_mut(),
        }
    }
}

// ================================================================================================
// QWEN3 MULTI-LORA ADAPTER SYSTEM FFI
// ================================================================================================

/// Free classification result
///
/// # Safety
/// - Must only be called once per result
/// - Result must have been allocated by classify_with_qwen3_adapter
#[no_mangle]
pub extern "C" fn free_generative_classification_result(
    result: *mut GenerativeClassificationResult,
) {
    if result.is_null() {
        return;
    }

    unsafe {
        // Free category name
        if !(*result).category_name.is_null() {
            let _ = CString::from_raw((*result).category_name);
        }

        // Free probabilities array
        if !(*result).probabilities.is_null() {
            let num_cats = (*result).num_categories as usize;
            let _ = Vec::from_raw_parts((*result).probabilities, num_cats, num_cats);
        }

        // Free error message
        if !(*result).error_message.is_null() {
            let _ = CString::from_raw((*result).error_message);
        }
    }
}

/// Free categories array
///
/// # Safety
/// - Must only be called once per array
/// - Array must have been allocated by get_qwen3_loaded_adapters
#[no_mangle]
pub extern "C" fn free_categories(categories: *mut *mut c_char, num_categories: i32) {
    if categories.is_null() || num_categories <= 0 {
        return;
    }

    unsafe {
        for i in 0..num_categories {
            let ptr = *categories.offset(i as isize);
            if !ptr.is_null() {
                let _ = CString::from_raw(ptr);
            }
        }
        let _ = Vec::from_raw_parts(categories, num_categories as usize, num_categories as usize);
    }
}

/// Helper: create error message C string
fn create_error_message(msg: &str) -> *mut c_char {
    match CString::new(msg) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Initialize Qwen3 Multi-LoRA classifier with base model
///
/// # Arguments
/// - `base_model_path`: Path to Qwen3-0.6B base model directory
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `base_model_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_qwen3_multi_lora_classifier(base_model_path: *const c_char) -> i32 {
    if base_model_path.is_null() {
        eprintln!("Error: base_model_path is null");
        return -1;
    }

    let base_model_path_str = unsafe {
        match CStr::from_ptr(base_model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in base_model_path: {}", e);
                return -1;
            }
        }
    };

    // Determine device (try GPU first, fall back to CPU)
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // Check if already initialized
    if GLOBAL_QWEN3_MULTI_CLASSIFIER.get().is_some() {
        println!("✅ Qwen3 Multi-LoRA classifier already initialized, reusing existing instance");
        return 0;
    }

    // Load multi-adapter classifier
    match Qwen3MultiLoRAClassifier::new(base_model_path_str, &device) {
        Ok(classifier) => {
            match GLOBAL_QWEN3_MULTI_CLASSIFIER.set(Mutex::new(classifier)) {
                Ok(_) => {
                    println!(
                        "✅ Qwen3 Multi-LoRA classifier initialized with base model: {}",
                        base_model_path_str
                    );
                    0
                }
                Err(_) => {
                    println!("✅ Qwen3 Multi-LoRA classifier already initialized (race condition), reusing");
                    0
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to load Qwen3 Multi-LoRA classifier: {}", e);
            -1
        }
    }
}

/// Load a LoRA adapter for the multi-adapter system
///
/// # Arguments
/// - `adapter_name`: Name for this adapter (e.g., "category", "jailbreak")
/// - `adapter_path`: Path to LoRA adapter directory (containing adapter_model.safetensors, adapter_config.json, label_mapping.json)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - Both arguments must be valid null-terminated C strings
#[no_mangle]
pub extern "C" fn load_qwen3_lora_adapter(
    adapter_name: *const c_char,
    adapter_path: *const c_char,
) -> i32 {
    if adapter_name.is_null() || adapter_path.is_null() {
        eprintln!("Error: null pointer passed to load_qwen3_lora_adapter");
        return -1;
    }

    let adapter_name_str = unsafe {
        match CStr::from_ptr(adapter_name).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in adapter_name: {}", e);
                return -1;
            }
        }
    };

    let adapter_path_str = unsafe {
        match CStr::from_ptr(adapter_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in adapter_path: {}", e);
                return -1;
            }
        }
    };

    // Get classifier
    let classifier_mutex = match GLOBAL_QWEN3_MULTI_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            return -1;
        }
    };

    // Load adapter
    match classifier_mutex.lock() {
        Ok(mut classifier) => match classifier.load_adapter(adapter_name_str, adapter_path_str) {
            Ok(_) => {
                println!(
                    "✅ Loaded adapter '{}' from: {}",
                    adapter_name_str, adapter_path_str
                );
                0
            }
            Err(e) => {
                eprintln!(
                    "Error: failed to load adapter '{}': {}",
                    adapter_name_str, e
                );
                -1
            }
        },
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            -1
        }
    }
}

/// Classify text using a specific LoRA adapter
///
/// # Arguments
/// - `text`: Input text to classify (null-terminated C string)
/// - `adapter_name`: Name of the adapter to use (e.g., "category", "jailbreak")
/// - `result`: Pointer to GenerativeClassificationResult struct (allocated by caller)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - All arguments must be valid pointers
/// - Caller must free: result.category_name, result.probabilities, result.error_message
#[no_mangle]
pub extern "C" fn classify_with_qwen3_adapter(
    text: *const c_char,
    adapter_name: *const c_char,
    result: *mut GenerativeClassificationResult,
) -> i32 {
    if text.is_null() || adapter_name.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to classify_with_qwen3_adapter");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    let adapter_name_str = unsafe {
        match CStr::from_ptr(adapter_name).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in adapter_name: {}", e);
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    // Get classifier
    let classifier_mutex = match GLOBAL_QWEN3_MULTI_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message("Classifier not initialized");
            }
            return -1;
        }
    };

    // Classify with adapter
    match classifier_mutex.lock() {
        Ok(mut classifier) => {
            match classifier.classify_with_adapter(text_str, adapter_name_str) {
                Ok(multi_result) => {
                    // Convert MultiAdapterClassificationResult to GenerativeClassificationResult
                    let category_name_c = match CString::new(multi_result.category.as_str()) {
                        Ok(s) => s.into_raw(),
                        Err(e) => {
                            eprintln!("Error: failed to create category name C string: {}", e);
                            unsafe {
                                (*result) = GenerativeClassificationResult::default();
                                (*result).error_message = create_error_message(&format!(
                                    "Failed to create C string: {}",
                                    e
                                ));
                            }
                            return -1;
                        }
                    };

                    // Find class_id from category name in all_categories
                    let class_id = multi_result
                        .all_categories
                        .iter()
                        .position(|cat| cat == &multi_result.category)
                        .unwrap_or(0) as i32;

                    // Allocate probabilities array
                    let mut probabilities = multi_result.probabilities;
                    let probs_ptr = probabilities.as_mut_ptr();
                    let num_categories = probabilities.len();
                    std::mem::forget(probabilities); // Prevent Rust from deallocating

                    unsafe {
                        (*result) = GenerativeClassificationResult {
                            class_id,
                            confidence: multi_result.confidence,
                            category_name: category_name_c,
                            probabilities: probs_ptr,
                            num_categories: num_categories as i32,
                            error: false,
                            error_message: ptr::null_mut(),
                        };
                    }

                    0
                }
                Err(e) => {
                    eprintln!(
                        "Error: classification with adapter '{}' failed: {}",
                        adapter_name_str, e
                    );
                    unsafe {
                        (*result) = GenerativeClassificationResult::default();
                        (*result).error_message =
                            create_error_message(&format!("Classification failed: {}", e));
                    }
                    -1
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message =
                    create_error_message(&format!("Failed to acquire lock: {}", e));
            }
            -1
        }
    }
}

/// Get list of loaded adapter names
///
/// # Arguments
/// - `adapters_out`: Output pointer that will be set to point to array of C strings
/// - `num_adapters`: Output parameter for number of adapters
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - Caller must free each string in the adapters array and the array itself
#[no_mangle]
pub extern "C" fn get_qwen3_loaded_adapters(
    adapters_out: *mut *mut *mut c_char,
    num_adapters: *mut i32,
) -> i32 {
    if adapters_out.is_null() || num_adapters.is_null() {
        eprintln!("Error: null pointer passed to get_qwen3_loaded_adapters");
        return -1;
    }

    let classifier_mutex = match GLOBAL_QWEN3_MULTI_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            return -1;
        }
    };

    match classifier_mutex.lock() {
        Ok(classifier) => {
            let adapter_names = classifier.list_adapters();
            let count = adapter_names.len();

            // Allocate array of C strings
            let mut c_strings: Vec<*mut c_char> = Vec::with_capacity(count);
            for name in adapter_names {
                match CString::new(name.as_str()) {
                    Ok(s) => c_strings.push(s.into_raw()),
                    Err(e) => {
                        eprintln!("Error: failed to create adapter name C string: {}", e);
                        // Free already allocated strings
                        for ptr in c_strings {
                            unsafe {
                                let _ = CString::from_raw(ptr);
                            }
                        }
                        return -1;
                    }
                }
            }

            // Transfer ownership to caller
            unsafe {
                *num_adapters = count as i32;
                *adapters_out = c_strings.as_mut_ptr();
            }
            std::mem::forget(c_strings);

            0
        }
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            -1
        }
    }
}

/// Zero-shot classification with base model (no adapter required)
///
/// # Arguments
/// - `text`: Input text to classify (null-terminated C string)
/// - `categories`: Array of category names (null-terminated C strings)
/// - `num_categories`: Number of categories
/// - `result`: Pointer to GenerativeClassificationResult struct (allocated by caller)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - All arguments must be valid pointers
/// - Caller must free: result.category_name, result.probabilities, result.error_message
#[no_mangle]
pub extern "C" fn classify_zero_shot_qwen3(
    text: *const c_char,
    categories: *const *const c_char,
    num_categories: i32,
    result: *mut GenerativeClassificationResult,
) -> i32 {
    if text.is_null() || categories.is_null() || result.is_null() || num_categories <= 0 {
        eprintln!(
            "Error: null pointer or invalid num_categories passed to classify_zero_shot_qwen3"
        );
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    // Convert C string array to Rust Vec<String>
    let mut cats = Vec::new();
    unsafe {
        for i in 0..num_categories {
            let cat_ptr = *categories.offset(i as isize);
            if cat_ptr.is_null() {
                eprintln!("Error: null category at index {}", i);
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message("Null category in array");
                return -1;
            }
            match CStr::from_ptr(cat_ptr).to_str() {
                Ok(s) => cats.push(s.to_string()),
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in category {}: {}", i, e);
                    (*result) = GenerativeClassificationResult::default();
                    (*result).error_message =
                        create_error_message(&format!("Invalid UTF-8 in category {}", i));
                    return -1;
                }
            }
        }
    }

    // Get classifier
    let classifier_mutex = match GLOBAL_QWEN3_MULTI_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message = create_error_message("Classifier not initialized");
            }
            return -1;
        }
    };

    // Classify zero-shot
    match classifier_mutex.lock() {
        Ok(mut classifier) => {
            match classifier.classify_zero_shot(text_str, cats.clone()) {
                Ok(multi_result) => {
                    // Convert MultiAdapterClassificationResult to GenerativeClassificationResult
                    let category_name_c = match CString::new(multi_result.category.as_str()) {
                        Ok(s) => s.into_raw(),
                        Err(e) => {
                            eprintln!("Error: failed to create category name C string: {}", e);
                            unsafe {
                                (*result) = GenerativeClassificationResult::default();
                                (*result).error_message = create_error_message(&format!(
                                    "Failed to create C string: {}",
                                    e
                                ));
                            }
                            return -1;
                        }
                    };

                    // Find class_id from category name in all_categories
                    let class_id = multi_result
                        .all_categories
                        .iter()
                        .position(|cat| cat == &multi_result.category)
                        .unwrap_or(0) as i32;

                    // Allocate probabilities array
                    let mut probabilities = multi_result.probabilities;
                    let probs_ptr = probabilities.as_mut_ptr();
                    let num_cats = probabilities.len();
                    std::mem::forget(probabilities); // Prevent Rust from deallocating

                    unsafe {
                        (*result) = GenerativeClassificationResult {
                            class_id,
                            confidence: multi_result.confidence,
                            category_name: category_name_c,
                            probabilities: probs_ptr,
                            num_categories: num_cats as i32,
                            error: false,
                            error_message: ptr::null_mut(),
                        };
                    }

                    0
                }
                Err(e) => {
                    eprintln!("Error: zero-shot classification failed: {}", e);
                    unsafe {
                        (*result) = GenerativeClassificationResult::default();
                        (*result).error_message =
                            create_error_message(&format!("Classification failed: {}", e));
                    }
                    -1
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            unsafe {
                (*result) = GenerativeClassificationResult::default();
                (*result).error_message =
                    create_error_message(&format!("Failed to acquire lock: {}", e));
            }
            -1
        }
    }
}

// ================================================================================================
// QWEN3 GUARD FFI (Safety/Jailbreak Detection)
// ================================================================================================

/// Guard generation result returned to Go (raw text only)
#[repr(C)]
pub struct GuardResult {
    /// Raw generated output (null-terminated C string)
    pub raw_output: *mut c_char,

    /// Error flag
    pub error: bool,

    /// Error message (null-terminated C string, only set if error=true)
    pub error_message: *mut c_char,
}

impl Default for GuardResult {
    fn default() -> Self {
        Self {
            raw_output: ptr::null_mut(),
            error: true,
            error_message: ptr::null_mut(),
        }
    }
}

/// Free guard result
///
/// # Safety
/// - Must only be called once per result
/// - Result must have been allocated by classify_with_qwen3_guard
#[no_mangle]
pub extern "C" fn free_guard_result(result: *mut GuardResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        // Free raw output
        if !(*result).raw_output.is_null() {
            let _ = CString::from_raw((*result).raw_output);
        }

        // Free error message
        if !(*result).error_message.is_null() {
            let _ = CString::from_raw((*result).error_message);
        }
    }
}

/// Initialize Qwen3Guard model
///
/// # Arguments
/// - `model_path`: Path to Qwen3Guard model directory
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_qwen3_guard(model_path: *const c_char) -> i32 {
    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return -1;
    }

    let model_path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_path: {}", e);
                return -1;
            }
        }
    };

    // Determine device
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    // Check if already initialized
    if GLOBAL_QWEN3_GUARD.get().is_some() {
        println!("✅ Qwen3Guard already initialized, reusing existing instance");
        return 0;
    }

    // Load guard model
    match Qwen3GuardModel::new(model_path_str, &device, None) {
        Ok(guard) => match GLOBAL_QWEN3_GUARD.set(Mutex::new(guard)) {
            Ok(_) => {
                println!("✅ Qwen3Guard initialized: {}", model_path_str);
                0
            }
            Err(_) => {
                println!("✅ Qwen3Guard already initialized (race condition), reusing");
                0
            }
        },
        Err(e) => {
            eprintln!("Error: failed to load Qwen3Guard: {}", e);
            -1
        }
    }
}

/// Classify text with Qwen3Guard
///
/// # Arguments
/// - `text`: Input text to classify (null-terminated C string)
/// - `mode`: Classification mode ("input" for user prompts, "output" for model responses)
/// - `result`: Pointer to GuardResult struct (allocated by caller)
///
/// # Returns
/// - 0 on success
/// - -1 on error
///
/// # Safety
/// - All arguments must be valid pointers
/// - Caller must free: result.raw_output, result.error_message
#[no_mangle]
pub extern "C" fn classify_with_qwen3_guard(
    text: *const c_char,
    mode: *const c_char,
    result: *mut GuardResult,
) -> i32 {
    if text.is_null() || mode.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to classify_with_qwen3_guard");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    let mode_str = unsafe {
        match CStr::from_ptr(mode).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in mode: {}", e);
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message(&format!("Invalid UTF-8: {}", e));
                return -1;
            }
        }
    };

    // Get guard model
    let guard_mutex = match GLOBAL_QWEN3_GUARD.get() {
        Some(g) => g,
        None => {
            eprintln!("Error: Qwen3Guard not initialized");
            unsafe {
                (*result) = GuardResult::default();
                (*result).error_message = create_error_message("Guard not initialized");
            }
            return -1;
        }
    };

    // Generate with guard
    match guard_mutex.lock() {
        Ok(mut guard) => match guard.generate_guard(text_str, mode_str) {
            Ok(guard_result) => {
                // Convert GuardGenerationResult to GuardResult
                let raw_output_c = match CString::new(guard_result.raw_output.as_str()) {
                    Ok(s) => s.into_raw(),
                    Err(e) => {
                        eprintln!("Error: failed to create raw_output C string: {}", e);
                        unsafe {
                            (*result) = GuardResult::default();
                            (*result).error_message =
                                create_error_message(&format!("Failed to create C string: {}", e));
                        }
                        return -1;
                    }
                };

                unsafe {
                    (*result) = GuardResult {
                        raw_output: raw_output_c,
                        error: false,
                        error_message: ptr::null_mut(),
                    };
                }

                0
            }
            Err(e) => {
                eprintln!("Error: guard classification failed: {}", e);
                unsafe {
                    (*result) = GuardResult::default();
                    (*result).error_message =
                        create_error_message(&format!("Classification failed: {}", e));
                }
                -1
            }
        },
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            unsafe {
                (*result) = GuardResult::default();
                (*result).error_message =
                    create_error_message(&format!("Failed to acquire lock: {}", e));
            }
            -1
        }
    }
}

/// Check if Qwen3Guard is initialized
///
/// # Returns
/// - 1 if initialized
/// - 0 if not initialized
#[no_mangle]
pub extern "C" fn is_qwen3_guard_initialized() -> i32 {
    if GLOBAL_QWEN3_GUARD.get().is_some() {
        1
    } else {
        0
    }
}

/// Check if Qwen3 Multi-LoRA classifier is initialized
///
/// # Returns
/// - 1 if initialized
/// - 0 if not initialized
#[no_mangle]
pub extern "C" fn is_qwen3_multi_lora_initialized() -> i32 {
    if GLOBAL_QWEN3_MULTI_CLASSIFIER.get().is_some() {
        1
    } else {
        0
    }
}

// ================================================================================================
// QWEN3 ZERO-SHOT PREFERENCE CLASSIFIER
// ================================================================================================

fn select_device(use_cpu: bool) -> Device {
    if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    }
}

/// Initialize Qwen3 zero-shot classifier for preference routing (no LoRA required)
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_qwen3_preference_classifier(
    model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return false;
    }

    let model_path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_path: {}", e);
                return false;
            }
        }
    };

    // Avoid re-loading if already initialized
    if GLOBAL_QWEN3_PREFERENCE.get().is_some() {
        println!("✅ Qwen3 preference classifier already initialized, reusing instance");
        return true;
    }

    let device = select_device(use_cpu);

    match Qwen3MultiLoRAClassifier::new(model_path_str, &device) {
        Ok(classifier) => match GLOBAL_QWEN3_PREFERENCE.set(Mutex::new(classifier)) {
            Ok(_) => true,
            Err(_) => {
                println!("✅ Qwen3 preference classifier already initialized (race), reusing");
                true
            }
        },
        Err(e) => {
            eprintln!(
                "Error: failed to initialize Qwen3 preference classifier: {}",
                e
            );
            false
        }
    }
}

/// Classify preference using zero-shot Qwen3 base model
///
/// # Safety
/// - `text` and `labels_json` must be valid null-terminated C strings
#[no_mangle]
pub extern "C" fn classify_qwen3_preference(
    text: *const c_char,
    labels_json: *const c_char,
) -> ClassificationResult {
    let default_result = ClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
        label: ptr::null_mut(),
    };

    if text.is_null() || labels_json.is_null() {
        eprintln!("Error: text or labels_json is null");
        return default_result;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                return default_result;
            }
        }
    };

    let labels_str = unsafe {
        match CStr::from_ptr(labels_json).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in labels_json: {}", e);
                return default_result;
            }
        }
    };

    // Parse labels array from JSON
    let labels_value: Value = match serde_json::from_str(labels_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: failed to parse labels_json: {}", e);
            return default_result;
        }
    };

    let labels: Vec<String> = match labels_value {
        Value::Array(arr) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => {
            eprintln!("Error: labels_json is not an array of strings");
            return default_result;
        }
    };

    if labels.is_empty() {
        eprintln!("Error: labels list is empty");
        return default_result;
    }

    let classifier_mutex = match GLOBAL_QWEN3_PREFERENCE.get() {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 preference classifier not initialized");
            return default_result;
        }
    };

    match classifier_mutex.lock() {
        Ok(mut classifier) => {
            match classifier.classify_zero_shot_multi_tokens(text_str, labels.clone()) {
                Ok(result) => {
                    let class_idx = labels
                        .iter()
                        .position(|label| label == &result.category)
                        .map(|idx| idx as i32)
                        .unwrap_or(-1);

                    ClassificationResult {
                        predicted_class: class_idx,
                        confidence: result.confidence,
                        label: ptr::null_mut(),
                    }
                }
                Err(e) => {
                    eprintln!("Error: Qwen3 preference classification failed: {}", e);
                    default_result
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to acquire preference classifier lock: {}", e);
            default_result
        }
    }
}
