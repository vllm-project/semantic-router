//! FFI exports for Go bindings

use crate::{KMeansSelector, KNNSelector, SVMSelector};
use crate::knn::KNNTrainingRecord;
use crate::kmeans::KMeansTrainingRecord;
use crate::svm::SVMTrainingRecord;
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
// KNN FFI
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

/// Train KNN with embeddings and labels
#[no_mangle]
pub extern "C" fn ml_knn_train(
    handle: *mut KNNHandle,
    embeddings: *const c_double,
    embedding_dim: size_t,
    labels: *const *const c_char,
    num_records: size_t,
) -> c_int {
    if handle.is_null() || embeddings.is_null() || labels.is_null() {
        return -1;
    }

    let selector = unsafe { &mut (*handle).0 };
    let emb_slice = unsafe { slice::from_raw_parts(embeddings, num_records * embedding_dim) };
    let labels_slice = unsafe { slice::from_raw_parts(labels, num_records) };

    let mut records = Vec::with_capacity(num_records);
    for i in 0..num_records {
        let start = i * embedding_dim;
        let end = start + embedding_dim;
        let embedding = emb_slice[start..end].to_vec();

        let label = match unsafe { c_str_to_string(labels_slice[i]) } {
            Some(s) => s,
            None => return -2,
        };

        records.push(KNNTrainingRecord {
            embedding,
            model: label,
        });
    }

    match selector.train(records) {
        Ok(()) => 0,
        Err(_) => -3,
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

/// Check if KNN is trained
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

/// Load KNN from JSON
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
// KMeans FFI
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

/// Train KMeans with embeddings, labels, quality, and latency
#[no_mangle]
pub extern "C" fn ml_kmeans_train(
    handle: *mut KMeansHandle,
    embeddings: *const c_double,
    embedding_dim: size_t,
    labels: *const *const c_char,
    qualities: *const c_double,
    latencies: *const i64,
    num_records: size_t,
) -> c_int {
    if handle.is_null() || embeddings.is_null() || labels.is_null() {
        return -1;
    }

    let selector = unsafe { &mut (*handle).0 };
    let emb_slice = unsafe { slice::from_raw_parts(embeddings, num_records * embedding_dim) };
    let labels_slice = unsafe { slice::from_raw_parts(labels, num_records) };
    let qualities_slice = if qualities.is_null() {
        vec![1.0; num_records]
    } else {
        unsafe { slice::from_raw_parts(qualities, num_records).to_vec() }
    };
    let latencies_slice = if latencies.is_null() {
        vec![0i64; num_records]
    } else {
        unsafe { slice::from_raw_parts(latencies, num_records).to_vec() }
    };

    let mut records = Vec::with_capacity(num_records);
    for i in 0..num_records {
        let start = i * embedding_dim;
        let end = start + embedding_dim;
        let embedding = emb_slice[start..end].to_vec();

        let label = match unsafe { c_str_to_string(labels_slice[i]) } {
            Some(s) => s,
            None => return -2,
        };

        records.push(KMeansTrainingRecord {
            embedding,
            model: label,
            quality: qualities_slice[i],
            latency_ns: latencies_slice[i],
        });
    }

    match selector.train(records) {
        Ok(()) => 0,
        Err(_) => -3,
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

/// Check if KMeans is trained
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

/// Load KMeans from JSON
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
// SVM FFI
// =============================================================================

/// Opaque handle to SVM selector
pub struct SVMHandle(SVMSelector);

/// Create a new SVM selector
#[no_mangle]
pub extern "C" fn ml_svm_new() -> *mut SVMHandle {
    Box::into_raw(Box::new(SVMHandle(SVMSelector::new())))
}

/// Free SVM selector
#[no_mangle]
pub extern "C" fn ml_svm_free(handle: *mut SVMHandle) {
    if !handle.is_null() {
        unsafe { drop(Box::from_raw(handle)) };
    }
}

/// Train SVM with embeddings and labels
#[no_mangle]
pub extern "C" fn ml_svm_train(
    handle: *mut SVMHandle,
    embeddings: *const c_double,
    embedding_dim: size_t,
    labels: *const *const c_char,
    num_records: size_t,
) -> c_int {
    if handle.is_null() || embeddings.is_null() || labels.is_null() {
        return -1;
    }

    let selector = unsafe { &mut (*handle).0 };
    let emb_slice = unsafe { slice::from_raw_parts(embeddings, num_records * embedding_dim) };
    let labels_slice = unsafe { slice::from_raw_parts(labels, num_records) };

    let mut records = Vec::with_capacity(num_records);
    for i in 0..num_records {
        let start = i * embedding_dim;
        let end = start + embedding_dim;
        let embedding = emb_slice[start..end].to_vec();

        let label = match unsafe { c_str_to_string(labels_slice[i]) } {
            Some(s) => s,
            None => return -2,
        };

        records.push(SVMTrainingRecord {
            embedding,
            model: label,
        });
    }

    match selector.train(records) {
        Ok(()) => 0,
        Err(_) => -3,
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

/// Check if SVM is trained
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

/// Load SVM from JSON
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
