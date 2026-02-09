//! FFI exports for MLP (Multi-Layer Perceptron) Selector
//!
//! GPU-accelerated neural network for model selection.
//! Reference: FusionFactory (arXiv:2507.10540) - Query-level fusion via tailored LLM routers
//!
//! Training is done in Python (src/training/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.

use crate::classifiers::mlp_selector::{MLPDType, MLPSelector};
use candle_core::Device;
use libc::{c_char, c_double, c_int, size_t};
use std::ffi::{CStr, CString};
use std::ptr;
use std::slice;

// =============================================================================
// Helper functions
// =============================================================================

unsafe fn c_str_to_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string())
}

fn string_to_c_str(s: String) -> *mut c_char {
    CString::new(s)
        .map(|cs| cs.into_raw())
        .unwrap_or(ptr::null_mut())
}

// =============================================================================
// MLP FFI (Inference Only - GPU Accelerated via Candle)
// Reference: FusionFactory (arXiv:2507.10540)
// =============================================================================

/// Opaque handle to MLP selector
pub struct MLPHandle(MLPSelector);

/// Create a new MLP selector (untrained)
#[no_mangle]
pub extern "C" fn candle_mlp_new() -> *mut MLPHandle {
    Box::into_raw(Box::new(MLPHandle(MLPSelector::new())))
}

/// Create a new MLP selector with device
/// device_type: 0 = CPU, 1 = CUDA (GPU), 2 = Metal (Apple Silicon)
#[no_mangle]
pub extern "C" fn candle_mlp_new_with_device(device_type: c_int) -> *mut MLPHandle {
    let device = match device_type {
        1 => {
            // Try CUDA
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::Cpu
            }
        }
        2 => {
            // Try Metal
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "metal"))]
            {
                Device::Cpu
            }
        }
        _ => Device::Cpu,
    };

    Box::into_raw(Box::new(MLPHandle(MLPSelector::with_device(device))))
}

/// Create a new MLP selector with device and dtype for mixed precision
/// device_type: 0 = CPU, 1 = CUDA (GPU), 2 = Metal (Apple Silicon)
/// dtype: 0 = F32, 1 = F16, 2 = BF16
#[no_mangle]
pub extern "C" fn candle_mlp_new_with_device_and_dtype(
    device_type: c_int,
    dtype: c_int,
) -> *mut MLPHandle {
    let device = match device_type {
        1 => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::Cpu
            }
        }
        2 => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "metal"))]
            {
                Device::Cpu
            }
        }
        _ => Device::Cpu,
    };

    let mlp_dtype = match dtype {
        1 => MLPDType::F16,
        2 => MLPDType::BF16,
        _ => MLPDType::F32,
    };

    Box::into_raw(Box::new(MLPHandle(MLPSelector::with_device_and_dtype(
        device, mlp_dtype,
    ))))
}

/// Free MLP selector
#[no_mangle]
pub extern "C" fn candle_mlp_free(handle: *mut MLPHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Select model using MLP
#[no_mangle]
pub extern "C" fn candle_mlp_select(
    handle: *const MLPHandle,
    query: *const c_double,
    query_len: size_t,
) -> *mut c_char {
    if handle.is_null() || query.is_null() {
        return ptr::null_mut();
    }

    let selector = unsafe { &(*handle).0 };
    let query_slice = unsafe { slice::from_raw_parts(query, query_len) };

    match selector.select(query_slice) {
        Ok(model) => string_to_c_str(model),
        Err(_) => ptr::null_mut(),
    }
}

/// Check if MLP is trained (has loaded model)
#[no_mangle]
pub extern "C" fn candle_mlp_is_trained(handle: *const MLPHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let selector = unsafe { &(*handle).0 };
    selector.is_trained() as c_int
}

/// Save MLP to JSON
#[no_mangle]
pub extern "C" fn candle_mlp_to_json(handle: *const MLPHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let selector = unsafe { &(*handle).0 };
    match selector.to_json() {
        Ok(json) => string_to_c_str(json),
        Err(_) => ptr::null_mut(),
    }
}

/// Load MLP from JSON (primary way to load trained models)
#[no_mangle]
pub extern "C" fn candle_mlp_from_json(json: *const c_char) -> *mut MLPHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match MLPSelector::from_json(&json_str) {
        Ok(selector) => Box::into_raw(Box::new(MLPHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

/// Load MLP from JSON with specific device
/// device_type: 0 = CPU, 1 = CUDA (GPU), 2 = Metal (Apple Silicon)
#[no_mangle]
pub extern "C" fn candle_mlp_from_json_with_device(
    json: *const c_char,
    device_type: c_int,
) -> *mut MLPHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let device = match device_type {
        1 => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::Cpu
            }
        }
        2 => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "metal"))]
            {
                Device::Cpu
            }
        }
        _ => Device::Cpu,
    };

    match MLPSelector::from_json_with_device(&json_str, device) {
        Ok(selector) => Box::into_raw(Box::new(MLPHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

/// Load MLP from JSON with specific device and dtype for mixed precision
/// device_type: 0 = CPU, 1 = CUDA (GPU), 2 = Metal (Apple Silicon)
/// dtype: 0 = F32, 1 = F16, 2 = BF16
#[no_mangle]
pub extern "C" fn candle_mlp_from_json_with_device_and_dtype(
    json: *const c_char,
    device_type: c_int,
    dtype: c_int,
) -> *mut MLPHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    let device = match device_type {
        1 => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::Cpu
            }
        }
        2 => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "metal"))]
            {
                Device::Cpu
            }
        }
        _ => Device::Cpu,
    };

    let mlp_dtype = match dtype {
        1 => MLPDType::F16,
        2 => MLPDType::BF16,
        _ => MLPDType::F32,
    };

    match MLPSelector::from_json_with_dtype(&json_str, device, mlp_dtype) {
        Ok(selector) => Box::into_raw(Box::new(MLPHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a C string allocated by this library
#[no_mangle]
pub extern "C" fn candle_mlp_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_handle_new() {
        let handle = candle_mlp_new();
        assert!(!handle.is_null());
        candle_mlp_free(handle);
    }

    #[test]
    fn test_mlp_is_trained_empty() {
        let handle = candle_mlp_new();
        assert_eq!(candle_mlp_is_trained(handle), 0);
        candle_mlp_free(handle);
    }
}
