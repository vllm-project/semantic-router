//! # FFI Functions for ML-based Model Selection Algorithms
//!
//! This module exposes KNN, KMeans, and SVM algorithms to Go via C FFI.
//! Implements the same interface patterns as other FFI modules.

use std::ffi::{c_char, c_double, c_float, c_int, CStr, CString};
use std::ptr;
use std::sync::OnceLock;

use crate::ffi::memory::allocate_c_string;
use crate::model_architectures::ml_algorithms::{
    KMeansSelector, KNNSelector, ModelRef, ModelSelector, SVMSelector, SelectionContext,
    TrainingRecord,
};

// =============================================================================
// Global Selectors (using OnceLock for thread-safe lazy initialization)
// =============================================================================

static KNN_SELECTOR: OnceLock<std::sync::RwLock<KNNSelector>> = OnceLock::new();
static KMEANS_SELECTOR: OnceLock<std::sync::RwLock<KMeansSelector>> = OnceLock::new();
static SVM_SELECTOR: OnceLock<std::sync::RwLock<SVMSelector>> = OnceLock::new();

// =============================================================================
// FFI Result Structures
// =============================================================================

/// C-compatible training record
#[repr(C)]
pub struct CTrainingRecord {
    pub query_embedding: *const c_double,
    pub embedding_len: c_int,
    pub selected_model: *const c_char,
    pub response_latency_ns: i64,
    pub response_quality: c_double,
    pub success: bool,
    pub timestamp: i64,
}

/// C-compatible model reference
#[repr(C)]
pub struct CModelRef {
    pub model: *const c_char,
    pub lora_name: *const c_char, // Can be null
}

/// C-compatible selection result
#[repr(C)]
pub struct CSelectionResult {
    pub model_name: *mut c_char,
    pub model_index: c_int,
    pub score: c_double,
    pub error: bool,
    pub error_message: *mut c_char,
}

impl Default for CSelectionResult {
    fn default() -> Self {
        Self {
            model_name: ptr::null_mut(),
            model_index: -1,
            score: 0.0,
            error: false,
            error_message: ptr::null_mut(),
        }
    }
}

// =============================================================================
// KNN FFI Functions
// =============================================================================

/// Initialize KNN selector with specified K
///
/// # Safety
/// - Call this before using any other KNN functions
#[no_mangle]
pub extern "C" fn init_knn_selector(k: c_int) -> bool {
    let k = if k <= 0 { 3 } else { k as usize };
    let selector = KNNSelector::new(k);

    KNN_SELECTOR.set(std::sync::RwLock::new(selector)).is_ok()
}

/// Check if KNN selector is initialized
#[no_mangle]
pub extern "C" fn is_knn_initialized() -> bool {
    KNN_SELECTOR.get().is_some()
}

/// Load KNN model from JSON file
///
/// # Safety
/// - `json_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn load_knn_model(json_path: *const c_char) -> bool {
    let path = match unsafe { CStr::from_ptr(json_path).to_str() } {
        Ok(s) => s,
        Err(_) => return false,
    };

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read KNN model file: {}", e);
            return false;
        }
    };

    let selector = KNN_SELECTOR.get_or_init(|| std::sync::RwLock::new(KNNSelector::new(3)));

    match selector.write() {
        Ok(mut s) => match s.load_from_json(&data) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("Failed to load KNN model: {}", e);
                false
            }
        },
        Err(_) => false,
    }
}

/// Train KNN selector with training records
///
/// # Safety
/// - `records` must be a valid array of CTrainingRecord
/// - `num_records` must match the actual array size
#[no_mangle]
pub extern "C" fn train_knn(records: *const CTrainingRecord, num_records: c_int) -> bool {
    if records.is_null() || num_records <= 0 {
        return false;
    }

    let selector = match KNN_SELECTOR.get() {
        Some(s) => s,
        None => return false,
    };

    // Convert C records to Rust TrainingRecords
    let rust_records: Vec<TrainingRecord> = (0..num_records as usize)
        .filter_map(|i| {
            let record = unsafe { &*records.add(i) };
            convert_c_training_record(record)
        })
        .collect();

    if rust_records.is_empty() {
        return false;
    }

    match selector.write() {
        Ok(mut s) => s.train(&rust_records).is_ok(),
        Err(_) => false,
    }
}

/// Select best model using KNN
///
/// # Safety
/// - `query_embedding` must be a valid array of doubles
/// - `refs` must be a valid array of CModelRef
#[no_mangle]
pub extern "C" fn select_with_knn(
    query_embedding: *const c_double,
    embedding_len: c_int,
    refs: *const CModelRef,
    num_refs: c_int,
) -> CSelectionResult {
    if query_embedding.is_null() || embedding_len <= 0 || refs.is_null() || num_refs <= 0 {
        return CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Invalid input parameters") },
            ..Default::default()
        };
    }

    let selector = match KNN_SELECTOR.get() {
        Some(s) => s,
        None => {
            return CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("KNN selector not initialized") },
                ..Default::default()
            }
        }
    };

    // Convert embedding
    let embedding: Vec<f64> = (0..embedding_len as usize)
        .map(|i| unsafe { *query_embedding.add(i) })
        .collect();

    // Convert model refs
    let model_refs = convert_c_model_refs(refs, num_refs as usize);

    let ctx = SelectionContext::new("", embedding);

    match selector.read() {
        Ok(s) => match s.select(&ctx, &model_refs) {
            Some(result) => CSelectionResult {
                model_name: unsafe { allocate_c_string(&result.model_name) },
                model_index: result.model_index as c_int,
                score: result.score,
                error: false,
                error_message: ptr::null_mut(),
            },
            None => CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("Selection returned no result") },
                ..Default::default()
            },
        },
        Err(_) => CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Failed to acquire lock") },
            ..Default::default()
        },
    }
}

/// Get KNN training count
#[no_mangle]
pub extern "C" fn get_knn_training_count() -> c_int {
    KNN_SELECTOR
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.training_count() as c_int)
        .unwrap_or(0)
}

// =============================================================================
// KMeans FFI Functions
// =============================================================================

/// Initialize KMeans selector
///
/// # Safety
/// - Call this before using any other KMeans functions
#[no_mangle]
pub extern "C" fn init_kmeans_selector(num_clusters: c_int, efficiency_weight: c_float) -> bool {
    let num_clusters = if num_clusters <= 0 {
        4
    } else {
        num_clusters as usize
    };
    let selector = if efficiency_weight > 0.0 {
        KMeansSelector::with_efficiency(num_clusters, efficiency_weight as f64)
    } else {
        KMeansSelector::new(num_clusters)
    };

    KMEANS_SELECTOR
        .set(std::sync::RwLock::new(selector))
        .is_ok()
}

/// Check if KMeans selector is initialized
#[no_mangle]
pub extern "C" fn is_kmeans_initialized() -> bool {
    KMEANS_SELECTOR.get().is_some()
}

/// Load KMeans model from JSON file
///
/// # Safety
/// - `json_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn load_kmeans_model(json_path: *const c_char) -> bool {
    let path = match unsafe { CStr::from_ptr(json_path).to_str() } {
        Ok(s) => s,
        Err(_) => return false,
    };

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read KMeans model file: {}", e);
            return false;
        }
    };

    let selector = KMEANS_SELECTOR.get_or_init(|| std::sync::RwLock::new(KMeansSelector::new(4)));

    match selector.write() {
        Ok(mut s) => match s.load_from_json(&data) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("Failed to load KMeans model: {}", e);
                false
            }
        },
        Err(_) => false,
    }
}

/// Train KMeans selector with training records
///
/// # Safety
/// - `records` must be a valid array of CTrainingRecord
/// - `num_records` must match the actual array size
#[no_mangle]
pub extern "C" fn train_kmeans(records: *const CTrainingRecord, num_records: c_int) -> bool {
    if records.is_null() || num_records <= 0 {
        return false;
    }

    let selector = match KMEANS_SELECTOR.get() {
        Some(s) => s,
        None => return false,
    };

    let rust_records: Vec<TrainingRecord> = (0..num_records as usize)
        .filter_map(|i| {
            let record = unsafe { &*records.add(i) };
            convert_c_training_record(record)
        })
        .collect();

    if rust_records.is_empty() {
        return false;
    }

    match selector.write() {
        Ok(mut s) => s.train(&rust_records).is_ok(),
        Err(_) => false,
    }
}

/// Select best model using KMeans
///
/// # Safety
/// - `query_embedding` must be a valid array of doubles
/// - `refs` must be a valid array of CModelRef
#[no_mangle]
pub extern "C" fn select_with_kmeans(
    query_embedding: *const c_double,
    embedding_len: c_int,
    refs: *const CModelRef,
    num_refs: c_int,
) -> CSelectionResult {
    if query_embedding.is_null() || embedding_len <= 0 || refs.is_null() || num_refs <= 0 {
        return CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Invalid input parameters") },
            ..Default::default()
        };
    }

    let selector = match KMEANS_SELECTOR.get() {
        Some(s) => s,
        None => {
            return CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("KMeans selector not initialized") },
                ..Default::default()
            }
        }
    };

    let embedding: Vec<f64> = (0..embedding_len as usize)
        .map(|i| unsafe { *query_embedding.add(i) })
        .collect();

    let model_refs = convert_c_model_refs(refs, num_refs as usize);

    let ctx = SelectionContext::new("", embedding);

    match selector.read() {
        Ok(s) => match s.select(&ctx, &model_refs) {
            Some(result) => CSelectionResult {
                model_name: unsafe { allocate_c_string(&result.model_name) },
                model_index: result.model_index as c_int,
                score: result.score,
                error: false,
                error_message: ptr::null_mut(),
            },
            None => CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("Selection returned no result") },
                ..Default::default()
            },
        },
        Err(_) => CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Failed to acquire lock") },
            ..Default::default()
        },
    }
}

/// Get KMeans training count
#[no_mangle]
pub extern "C" fn get_kmeans_training_count() -> c_int {
    KMEANS_SELECTOR
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.training_count() as c_int)
        .unwrap_or(0)
}

// =============================================================================
// SVM FFI Functions
// =============================================================================

/// Initialize SVM selector
///
/// # Safety
/// - `kernel` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_svm_selector(kernel: *const c_char) -> bool {
    let kernel_str = if kernel.is_null() {
        "rbf"
    } else {
        match unsafe { CStr::from_ptr(kernel).to_str() } {
            Ok(s) => s,
            Err(_) => "rbf",
        }
    };

    let selector = SVMSelector::new(kernel_str);

    SVM_SELECTOR.set(std::sync::RwLock::new(selector)).is_ok()
}

/// Check if SVM selector is initialized
#[no_mangle]
pub extern "C" fn is_svm_initialized() -> bool {
    SVM_SELECTOR.get().is_some()
}

/// Load SVM model from JSON file
///
/// # Safety
/// - `json_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn load_svm_model(json_path: *const c_char) -> bool {
    let path = match unsafe { CStr::from_ptr(json_path).to_str() } {
        Ok(s) => s,
        Err(_) => return false,
    };

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to read SVM model file: {}", e);
            return false;
        }
    };

    let selector = SVM_SELECTOR.get_or_init(|| std::sync::RwLock::new(SVMSelector::new("rbf")));

    match selector.write() {
        Ok(mut s) => match s.load_from_json(&data) {
            Ok(_) => true,
            Err(e) => {
                eprintln!("Failed to load SVM model: {}", e);
                false
            }
        },
        Err(_) => false,
    }
}

/// Train SVM selector with training records
///
/// # Safety
/// - `records` must be a valid array of CTrainingRecord
/// - `num_records` must match the actual array size
#[no_mangle]
pub extern "C" fn train_svm(records: *const CTrainingRecord, num_records: c_int) -> bool {
    if records.is_null() || num_records <= 0 {
        return false;
    }

    let selector = match SVM_SELECTOR.get() {
        Some(s) => s,
        None => return false,
    };

    let rust_records: Vec<TrainingRecord> = (0..num_records as usize)
        .filter_map(|i| {
            let record = unsafe { &*records.add(i) };
            convert_c_training_record(record)
        })
        .collect();

    if rust_records.is_empty() {
        return false;
    }

    match selector.write() {
        Ok(mut s) => s.train(&rust_records).is_ok(),
        Err(_) => false,
    }
}

/// Select best model using SVM
///
/// # Safety
/// - `query_embedding` must be a valid array of doubles
/// - `refs` must be a valid array of CModelRef
#[no_mangle]
pub extern "C" fn select_with_svm(
    query_embedding: *const c_double,
    embedding_len: c_int,
    refs: *const CModelRef,
    num_refs: c_int,
) -> CSelectionResult {
    if query_embedding.is_null() || embedding_len <= 0 || refs.is_null() || num_refs <= 0 {
        return CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Invalid input parameters") },
            ..Default::default()
        };
    }

    let selector = match SVM_SELECTOR.get() {
        Some(s) => s,
        None => {
            return CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("SVM selector not initialized") },
                ..Default::default()
            }
        }
    };

    let embedding: Vec<f64> = (0..embedding_len as usize)
        .map(|i| unsafe { *query_embedding.add(i) })
        .collect();

    let model_refs = convert_c_model_refs(refs, num_refs as usize);

    let ctx = SelectionContext::new("", embedding);

    match selector.read() {
        Ok(s) => match s.select(&ctx, &model_refs) {
            Some(result) => CSelectionResult {
                model_name: unsafe { allocate_c_string(&result.model_name) },
                model_index: result.model_index as c_int,
                score: result.score,
                error: false,
                error_message: ptr::null_mut(),
            },
            None => CSelectionResult {
                error: true,
                error_message: unsafe { allocate_c_string("Selection returned no result") },
                ..Default::default()
            },
        },
        Err(_) => CSelectionResult {
            error: true,
            error_message: unsafe { allocate_c_string("Failed to acquire lock") },
            ..Default::default()
        },
    }
}

/// Get SVM training count
#[no_mangle]
pub extern "C" fn get_svm_training_count() -> c_int {
    SVM_SELECTOR
        .get()
        .and_then(|s| s.read().ok())
        .map(|s| s.training_count() as c_int)
        .unwrap_or(0)
}

// =============================================================================
// Memory Management
// =============================================================================

/// Free a CSelectionResult
///
/// # Safety
/// - Result must be a valid CSelectionResult returned from select_with_* functions
#[no_mangle]
pub extern "C" fn free_selection_result(result: CSelectionResult) {
    if !result.model_name.is_null() {
        unsafe {
            let _ = CString::from_raw(result.model_name);
        }
    }
    if !result.error_message.is_null() {
        unsafe {
            let _ = CString::from_raw(result.error_message);
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Convert C training record to Rust TrainingRecord
fn convert_c_training_record(record: &CTrainingRecord) -> Option<TrainingRecord> {
    if record.query_embedding.is_null()
        || record.embedding_len <= 0
        || record.selected_model.is_null()
    {
        return None;
    }

    let embedding: Vec<f64> = (0..record.embedding_len as usize)
        .map(|i| unsafe { *record.query_embedding.add(i) })
        .collect();

    let selected_model = unsafe {
        CStr::from_ptr(record.selected_model)
            .to_str()
            .ok()?
            .to_string()
    };

    Some(TrainingRecord {
        query_text: String::new(), // FFI doesn't pass query text currently
        query_embedding: embedding,
        decision_name: String::new(), // FFI doesn't pass decision name currently
        selected_model,
        response_latency_ns: record.response_latency_ns,
        response_quality: record.response_quality,
        success: record.success,
        timestamp: record.timestamp,
    })
}

/// Convert C model refs to Rust ModelRefs
fn convert_c_model_refs(refs: *const CModelRef, num_refs: usize) -> Vec<ModelRef> {
    (0..num_refs)
        .filter_map(|i| {
            let c_ref = unsafe { &*refs.add(i) };
            if c_ref.model.is_null() {
                return None;
            }

            let model = unsafe { CStr::from_ptr(c_ref.model).to_str().ok()?.to_string() };
            let lora_name = if c_ref.lora_name.is_null() {
                None
            } else {
                unsafe {
                    CStr::from_ptr(c_ref.lora_name)
                        .to_str()
                        .ok()
                        .map(|s| s.to_string())
                }
            };

            Some(ModelRef { model, lora_name })
        })
        .collect()
}
