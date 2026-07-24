//! FFI bindings for Qwen3Guard safety classification.

use crate::model_architectures::generative::Qwen3GuardModel;
use crate::registry::get_registry;
use candle_core::Device;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::{Mutex, OnceLock};

/// Global Qwen3Guard instance (for safety/jailbreak detection)

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

fn create_error_message(msg: &str) -> *mut c_char {
    match CString::new(msg) {
        Ok(s) => s.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free guard result
///
/// # Safety
/// - `result` must be a valid pointer to a `GuardResult` initialized by this FFI module.
/// - Must only be called once per result; the owned string pointers inside the result must not
///   be freed elsewhere.
#[no_mangle]
pub unsafe extern "C" fn free_guard_result(result: *mut GuardResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        if !(*result).raw_output.is_null() {
            let _ = CString::from_raw((*result).raw_output);
        }

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
/// - `model_path` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn init_qwen3_guard(model_path: *const c_char) -> i32 {
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

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

    if get_registry()
        .get::<Mutex<Qwen3GuardModel>>("global_qwen3_guard")
        .is_some()
    {
        println!("Qwen3Guard already initialized, reusing existing instance");
        return 0;
    }

    match Qwen3GuardModel::new(model_path_str, &device, None) {
        Ok(guard) => match get_registry().register("global_qwen3_guard", Mutex::new(guard)) {
            Ok(_) => {
                println!("Qwen3Guard initialized: {}", model_path_str);
                0
            }
            Err(_) => {
                println!("Qwen3Guard already initialized (race condition), reusing");
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
/// - `text` and `mode` must be valid null-terminated C strings.
/// - `result` must be a valid writable pointer for one `GuardResult`.
/// - Caller must later release owned fields with `free_guard_result`.
#[no_mangle]
pub unsafe extern "C" fn classify_with_qwen3_guard(
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

    let guard_mutex = match get_registry().get::<Mutex<Qwen3GuardModel>>("global_qwen3_guard") {
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

    match guard_mutex.lock() {
        Ok(mut guard) => match guard.generate_guard(text_str, mode_str) {
            Ok(guard_result) => {
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
    if get_registry()
        .get::<Mutex<Qwen3GuardModel>>("global_qwen3_guard")
        .is_some()
    {
        1
    } else {
        0
    }
}
