//! FFI exports for Go bindings
//!
//! This module provides inference-only functions for KNN, KMeans, and SVM.
//! Training is done in Python (src/training/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.

use crate::{KMeansSelector, KNNSelector, SVMSelector};
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
    CString::new(s).map(|cs| cs.into_raw()).unwrap_or(ptr::null_mut())
}

// =============================================================================
// KNN FFI (Inference Only)
// =============================================================================

/// Opaque handle to KNN selector
pub struct KNNHandle(KNNSelector);

/// Create a new KNN selector
#[no_mangle]
pub extern "C" fn ml_knn_new(k: c_int) -> *mut KNNHandle {
    Box::into_raw(Box::new(KNNHandle(KNNSelector::new(k as usize))))
}

/// Free KNN selector
#[no_mangle]
pub extern "C" fn ml_knn_free(handle: *mut KNNHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Select model using KNN
#[no_mangle]
pub extern "C" fn ml_knn_select(
    handle: *const KNNHandle,
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

/// Check if KNN is trained (has loaded model)
#[no_mangle]
pub extern "C" fn ml_knn_is_trained(handle: *const KNNHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let selector = unsafe { &(*handle).0 };
    selector.is_trained() as c_int
}

/// Save KNN to JSON
#[no_mangle]
pub extern "C" fn ml_knn_to_json(handle: *const KNNHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let selector = unsafe { &(*handle).0 };
    match selector.to_json() {
        Ok(json) => string_to_c_str(json),
        Err(_) => ptr::null_mut(),
    }
}

/// Load KNN from JSON (primary way to load trained models)
#[no_mangle]
pub extern "C" fn ml_knn_from_json(json: *const c_char) -> *mut KNNHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match KNNSelector::from_json(&json_str) {
        Ok(selector) => Box::into_raw(Box::new(KNNHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

// =============================================================================
// KMeans FFI (Inference Only)
// =============================================================================

/// Opaque handle to KMeans selector
pub struct KMeansHandle(KMeansSelector);

/// Create a new KMeans selector
#[no_mangle]
pub extern "C" fn ml_kmeans_new(num_clusters: c_int) -> *mut KMeansHandle {
    Box::into_raw(Box::new(KMeansHandle(KMeansSelector::new(num_clusters as usize))))
}

/// Free KMeans selector
#[no_mangle]
pub extern "C" fn ml_kmeans_free(handle: *mut KMeansHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Select model using KMeans
#[no_mangle]
pub extern "C" fn ml_kmeans_select(
    handle: *const KMeansHandle,
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

/// Check if KMeans is trained (has loaded model)
#[no_mangle]
pub extern "C" fn ml_kmeans_is_trained(handle: *const KMeansHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let selector = unsafe { &(*handle).0 };
    selector.is_trained() as c_int
}

/// Save KMeans to JSON
#[no_mangle]
pub extern "C" fn ml_kmeans_to_json(handle: *const KMeansHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let selector = unsafe { &(*handle).0 };
    match selector.to_json() {
        Ok(json) => string_to_c_str(json),
        Err(_) => ptr::null_mut(),
    }
}

/// Load KMeans from JSON (primary way to load trained models)
#[no_mangle]
pub extern "C" fn ml_kmeans_from_json(json: *const c_char) -> *mut KMeansHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match KMeansSelector::from_json(&json_str) {
        Ok(selector) => Box::into_raw(Box::new(KMeansHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

// =============================================================================
// SVM FFI (Inference Only)
// =============================================================================

/// Opaque handle to SVM selector
pub struct SVMHandle(SVMSelector);

/// Create a new SVM selector with default (RBF) kernel
#[no_mangle]
pub extern "C" fn ml_svm_new() -> *mut SVMHandle {
    Box::into_raw(Box::new(SVMHandle(SVMSelector::new())))
}

/// Create a new SVM selector with specified kernel
/// kernel_type: 0 = linear, 1 = rbf
/// gamma: RBF gamma parameter (ignored for linear, use 0.0 for auto which defaults to 1.0)
#[no_mangle]
pub extern "C" fn ml_svm_new_with_kernel(kernel_type: c_int, gamma: c_double) -> *mut SVMHandle {
    use crate::svm::KernelType;

    let selector = match kernel_type {
        0 => SVMSelector::with_kernel(KernelType::Linear, 1.0), // Explicit Linear
        1 => {
            // RBF with optional gamma (0.0 = auto, defaults to 1.0)
            let gamma_opt = if gamma > 0.0 { Some(gamma) } else { None };
            SVMSelector::with_rbf(gamma_opt)
        }
        _ => SVMSelector::new(), // Default to RBF
    };

    Box::into_raw(Box::new(SVMHandle(selector)))
}

/// Free SVM selector
#[no_mangle]
pub extern "C" fn ml_svm_free(handle: *mut SVMHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Select model using SVM
#[no_mangle]
pub extern "C" fn ml_svm_select(
    handle: *const SVMHandle,
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

/// Check if SVM is trained (has loaded model)
#[no_mangle]
pub extern "C" fn ml_svm_is_trained(handle: *const SVMHandle) -> c_int {
    if handle.is_null() {
        return 0;
    }
    let selector = unsafe { &(*handle).0 };
    selector.is_trained() as c_int
}

/// Save SVM to JSON
#[no_mangle]
pub extern "C" fn ml_svm_to_json(handle: *const SVMHandle) -> *mut c_char {
    if handle.is_null() {
        return ptr::null_mut();
    }
    let selector = unsafe { &(*handle).0 };
    match selector.to_json() {
        Ok(json) => string_to_c_str(json),
        Err(_) => ptr::null_mut(),
    }
}

/// Load SVM from JSON (primary way to load trained models)
#[no_mangle]
pub extern "C" fn ml_svm_from_json(json: *const c_char) -> *mut SVMHandle {
    let json_str = match unsafe { c_str_to_string(json) } {
        Some(s) => s,
        None => return ptr::null_mut(),
    };

    match SVMSelector::from_json(&json_str) {
        Ok(selector) => Box::into_raw(Box::new(SVMHandle(selector))),
        Err(_) => ptr::null_mut(),
    }
}

// =============================================================================
// Memory management
// =============================================================================

/// Free a C string allocated by this library
#[no_mangle]
pub extern "C" fn ml_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}
