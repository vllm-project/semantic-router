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

use crate::model_architectures::generative::{
    MultiAdapterClassificationResult, Qwen3MultiLoRAClassifier,
};
use crate::registry::get_registry;
use candle_core::Device;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Global multi-adapter classifier instance (for LoRA-based classification)

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
/// - `result` must be a valid pointer to a `GenerativeClassificationResult` initialized by
///   this FFI module.
/// - Must only be called once per result; the owned string and probability pointers inside the
///   result must not be freed elsewhere.
#[no_mangle]
pub unsafe extern "C" fn free_generative_classification_result(
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
/// - `categories` must be the pointer returned by `get_qwen3_loaded_adapters`.
/// - `num_categories` must match the element count returned with that pointer.
/// - Must only be called once per array; the element strings must not be freed elsewhere.
#[no_mangle]
pub unsafe extern "C" fn free_categories(categories: *mut *mut c_char, num_categories: i32) {
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

fn cstr_to_string(ptr: *const c_char, arg_name: &str) -> Result<String, String> {
    unsafe {
        CStr::from_ptr(ptr)
            .to_str()
            .map(str::to_owned)
            .map_err(|e| format!("invalid UTF-8 in {}: {}", arg_name, e))
    }
}

fn write_generative_error(result: *mut GenerativeClassificationResult, msg: &str) -> i32 {
    unsafe {
        (*result) = GenerativeClassificationResult::default();
        (*result).error_message = create_error_message(msg);
    }
    -1
}

fn write_generative_success(
    result: *mut GenerativeClassificationResult,
    multi_result: MultiAdapterClassificationResult,
) -> Result<(), String> {
    let category_name_c = CString::new(multi_result.category.as_str())
        .map_err(|e| format!("Failed to create C string: {}", e))?
        .into_raw();
    let class_id = multi_result
        .all_categories
        .iter()
        .position(|cat| cat == &multi_result.category)
        .unwrap_or(0) as i32;
    let mut probabilities = multi_result.probabilities;
    let probs_ptr = probabilities.as_mut_ptr();
    let num_categories = probabilities.len();
    std::mem::forget(probabilities);

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

    Ok(())
}

fn collect_category_strings(
    categories: *const *const c_char,
    num_categories: i32,
) -> Result<Vec<String>, String> {
    let mut result = Vec::new();
    unsafe {
        for i in 0..num_categories {
            let cat_ptr = *categories.offset(i as isize);
            if cat_ptr.is_null() {
                return Err("Null category in array".to_string());
            }
            result.push(cstr_to_string(cat_ptr, &format!("category {}", i))?);
        }
    }
    Ok(result)
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
/// - `base_model_path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn init_qwen3_multi_lora_classifier(base_model_path: *const c_char) -> i32 {
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
    if get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
        .is_some()
    {
        println!("Qwen3 Multi-LoRA classifier already initialized, reusing existing instance");
        return 0;
    }

    // Load multi-adapter classifier
    match Qwen3MultiLoRAClassifier::new(base_model_path_str, &device) {
        Ok(classifier) => {
            match get_registry().register("global_qwen3_multi_classifier", Mutex::new(classifier)) {
                Ok(_) => {
                    println!(
                        "Qwen3 Multi-LoRA classifier initialized with base model: {}",
                        base_model_path_str
                    );
                    0
                }
                Err(_) => {
                    println!(
                        "Qwen3 Multi-LoRA classifier already initialized (race condition), reusing"
                    );
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
/// - `adapter_name` and `adapter_path` must be valid null-terminated C strings.
#[no_mangle]
pub unsafe extern "C" fn load_qwen3_lora_adapter(
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
    let classifier_mutex = match get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
    {
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
                    "Loaded adapter '{}' from: {}",
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
/// - `text` and `adapter_name` must be valid null-terminated C strings.
/// - `result` must be a valid writable pointer for one `GenerativeClassificationResult`.
/// - Caller must later release owned fields with `free_generative_classification_result`.
#[no_mangle]
pub unsafe extern "C" fn classify_with_qwen3_adapter(
    text: *const c_char,
    adapter_name: *const c_char,
    result: *mut GenerativeClassificationResult,
) -> i32 {
    if text.is_null() || adapter_name.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to classify_with_qwen3_adapter");
        return -1;
    }

    let text_str = match cstr_to_string(text, "text") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            return write_generative_error(result, &e);
        }
    };
    let adapter_name_str = match cstr_to_string(adapter_name, "adapter_name") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            return write_generative_error(result, &e);
        }
    };

    let classifier_mutex = match get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
    {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            return write_generative_error(result, "Classifier not initialized");
        }
    };

    match classifier_mutex.lock() {
        Ok(mut classifier) => {
            match classifier.classify_with_adapter(&text_str, &adapter_name_str) {
                Ok(multi_result) => match write_generative_success(result, multi_result) {
                    Ok(()) => 0,
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        write_generative_error(result, &e)
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Error: classification with adapter '{}' failed: {}",
                        adapter_name_str, e
                    );
                    write_generative_error(result, &format!("Classification failed: {}", e))
                }
            }
        }
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            write_generative_error(result, &format!("Failed to acquire lock: {}", e))
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
/// - `adapters_out` and `num_adapters` must be valid writable output pointers.
/// - Caller must release the returned array with `free_categories`.
#[no_mangle]
pub unsafe extern "C" fn get_qwen3_loaded_adapters(
    adapters_out: *mut *mut *mut c_char,
    num_adapters: *mut i32,
) -> i32 {
    if adapters_out.is_null() || num_adapters.is_null() {
        eprintln!("Error: null pointer passed to get_qwen3_loaded_adapters");
        return -1;
    }

    let classifier_mutex = match get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
    {
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
/// - `text` must be a valid null-terminated C string.
/// - `categories` must point to `num_categories` valid null-terminated C string pointers.
/// - `result` must be a valid writable pointer for one `GenerativeClassificationResult`.
/// - Caller must later release owned fields with `free_generative_classification_result`.
#[no_mangle]
pub unsafe extern "C" fn classify_zero_shot_qwen3(
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

    let text_str = match cstr_to_string(text, "text") {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error: {}", e);
            return write_generative_error(result, &e);
        }
    };
    let cats = match collect_category_strings(categories, num_categories) {
        Ok(cats) => cats,
        Err(e) => {
            eprintln!("Error: {}", e);
            return write_generative_error(result, &e);
        }
    };

    let classifier_mutex = match get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
    {
        Some(c) => c,
        None => {
            eprintln!("Error: Qwen3 Multi-LoRA classifier not initialized");
            return write_generative_error(result, "Classifier not initialized");
        }
    };

    match classifier_mutex.lock() {
        Ok(mut classifier) => match classifier.classify_zero_shot(&text_str, cats) {
            Ok(multi_result) => match write_generative_success(result, multi_result) {
                Ok(()) => 0,
                Err(e) => {
                    eprintln!("Error: {}", e);
                    write_generative_error(result, &e)
                }
            },
            Err(e) => {
                eprintln!("Error: zero-shot classification failed: {}", e);
                write_generative_error(result, &format!("Classification failed: {}", e))
            }
        },
        Err(e) => {
            eprintln!("Error: failed to acquire lock: {}", e);
            write_generative_error(result, &format!("Failed to acquire lock: {}", e))
        }
    }
}

/// Check if Qwen3 Multi-LoRA classifier is initialized
///
/// # Returns
/// - 1 if initialized
/// - 0 if not initialized
#[no_mangle]
pub extern "C" fn is_qwen3_multi_lora_initialized() -> i32 {
    if get_registry()
        .get::<Mutex<Qwen3MultiLoRAClassifier>>("global_qwen3_multi_classifier")
        .is_some()
    {
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
