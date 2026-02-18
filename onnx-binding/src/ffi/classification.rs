//! Classification FFI Module (ONNX Runtime)
//!
//! Provides C/Go FFI for mmBERT classification models:
//! - Sequence classification (Intent, Jailbreak, Feedback, Factcheck)
//! - Token classification (PII detection)

use crate::model_architectures::classification::{
    ClassifierExecutionProvider, MmBertSequenceClassifier, MmBertTokenClassifier,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::{c_char, CStr, CString};
use std::sync::OnceLock;

// ============================================================================
// FFI Types
// ============================================================================

/// Classification result for FFI
#[repr(C)]
pub struct ClassificationResultFFI {
    /// Predicted label (C string, must be freed)
    pub label: *mut c_char,
    /// Class ID
    pub class_id: i32,
    /// Confidence score
    pub confidence: f32,
    /// Number of classes
    pub num_classes: i32,
    /// Probabilities array (must be freed)
    pub probabilities: *mut f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Error flag
    pub error: bool,
}

impl Default for ClassificationResultFFI {
    fn default() -> Self {
        Self {
            label: std::ptr::null_mut(),
            class_id: -1,
            confidence: 0.0,
            num_classes: 0,
            probabilities: std::ptr::null_mut(),
            processing_time_ms: 0.0,
            error: true,
        }
    }
}

/// PII entity for FFI
#[repr(C)]
pub struct PIIEntityFFI {
    /// Entity text (C string)
    pub text: *mut c_char,
    /// Entity type (e.g., "US_SSN")
    pub entity_type: *mut c_char,
    /// Start offset
    pub start: i32,
    /// End offset
    pub end: i32,
    /// Confidence
    pub confidence: f32,
}

/// PII detection result for FFI
#[repr(C)]
pub struct PIIResultFFI {
    /// Array of detected entities
    pub entities: *mut PIIEntityFFI,
    /// Number of entities
    pub num_entities: i32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Error flag
    pub error: bool,
}

impl Default for PIIResultFFI {
    fn default() -> Self {
        Self {
            entities: std::ptr::null_mut(),
            num_entities: 0,
            processing_time_ms: 0.0,
            error: true,
        }
    }
}

// ============================================================================
// Global Model Storage
// ============================================================================

/// Global storage for sequence classifiers
static SEQUENCE_CLASSIFIERS: OnceLock<Mutex<HashMap<String, MmBertSequenceClassifier>>> = OnceLock::new();

/// Global storage for token classifiers (PII)
static TOKEN_CLASSIFIERS: OnceLock<Mutex<HashMap<String, MmBertTokenClassifier>>> = OnceLock::new();

fn get_seq_classifiers() -> &'static Mutex<HashMap<String, MmBertSequenceClassifier>> {
    SEQUENCE_CLASSIFIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_tok_classifiers() -> &'static Mutex<HashMap<String, MmBertTokenClassifier>> {
    TOKEN_CLASSIFIERS.get_or_init(|| Mutex::new(HashMap::new()))
}

// ============================================================================
// Initialization Functions
// ============================================================================

/// Initialize a sequence classifier (Intent, Jailbreak, etc.)
///
/// # Parameters
/// - `name`: Model name identifier (e.g., "intent", "jailbreak")
/// - `model_path`: Path to model directory
/// - `use_gpu`: If true, try to use GPU (ROCm/CUDA)
///
/// # Returns
/// true on success, false on error
#[no_mangle]
pub extern "C" fn init_sequence_classifier(
    name: *const c_char,
    model_path: *const c_char,
    use_gpu: bool,
) -> bool {
    if name.is_null() || model_path.is_null() {
        eprintln!("Error: null pointer in init_sequence_classifier");
        return false;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        }
    };

    let path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        }
    };

    let provider = if use_gpu {
        ClassifierExecutionProvider::Auto
    } else {
        ClassifierExecutionProvider::Cpu
    };

    match MmBertSequenceClassifier::load(&path_str, provider) {
        Ok(model) => {
            println!("INFO: Loaded sequence classifier '{}' from {}", name_str, path_str);
            println!("INFO: {}", model.model_info());

            let mut classifiers = get_seq_classifiers().lock();
            classifiers.insert(name_str, model);
            true
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load sequence classifier '{}': {:?}", name_str, e);
            false
        }
    }
}

/// Initialize a token classifier (PII detection)
///
/// # Parameters
/// - `name`: Model name identifier (e.g., "pii")
/// - `model_path`: Path to model directory
/// - `use_gpu`: If true, try to use GPU
///
/// # Returns
/// true on success, false on error
#[no_mangle]
pub extern "C" fn init_token_classifier(
    name: *const c_char,
    model_path: *const c_char,
    use_gpu: bool,
) -> bool {
    if name.is_null() || model_path.is_null() {
        eprintln!("Error: null pointer in init_token_classifier");
        return false;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        }
    };

    let path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return false,
        }
    };

    let provider = if use_gpu {
        ClassifierExecutionProvider::Auto
    } else {
        ClassifierExecutionProvider::Cpu
    };

    match MmBertTokenClassifier::load(&path_str, provider) {
        Ok(model) => {
            println!("INFO: Loaded token classifier '{}' from {}", name_str, path_str);
            println!("INFO: {}", model.model_info());

            let mut classifiers = get_tok_classifiers().lock();
            classifiers.insert(name_str, model);
            true
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load token classifier '{}': {:?}", name_str, e);
            false
        }
    }
}

/// Check if a classifier is loaded
#[no_mangle]
pub extern "C" fn is_classifier_loaded(name: *const c_char) -> bool {
    if name.is_null() {
        return false;
    }

    let name_str = unsafe {
        match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let seq_classifiers = get_seq_classifiers().lock();
    let tok_classifiers = get_tok_classifiers().lock();

    seq_classifiers.contains_key(name_str) || tok_classifiers.contains_key(name_str)
}

// ============================================================================
// Classification Functions
// ============================================================================

/// Classify a single text
///
/// # Parameters
/// - `classifier_name`: Name of the classifier to use
/// - `text`: Input text
/// - `result`: Output result pointer
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn classify_text(
    classifier_name: *const c_char,
    text: *const c_char,
    result: *mut ClassificationResultFFI,
) -> i32 {
    if classifier_name.is_null() || text.is_null() || result.is_null() {
        return -1;
    }

    let name_str = unsafe {
        match CStr::from_ptr(classifier_name).to_str() {
            Ok(s) => s,
            Err(_) => {
                *result = ClassificationResultFFI::default();
                return -1;
            }
        }
    };

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                *result = ClassificationResultFFI::default();
                return -1;
            }
        }
    };

    let start_time = std::time::Instant::now();

    let mut classifiers = get_seq_classifiers().lock();
    let classifier = match classifiers.get_mut(name_str) {
        Some(c) => c,
        None => {
            eprintln!("Error: classifier '{}' not found", name_str);
            unsafe {
                *result = ClassificationResultFFI::default();
            }
            return -1;
        }
    };

    match classifier.classify(text_str) {
        Ok(classification) => {
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            let label_cstr = CString::new(classification.label).unwrap();
            let num_classes = classification.probabilities.len() as i32;
            let probs = Box::into_raw(classification.probabilities.into_boxed_slice()) as *mut f32;

            unsafe {
                *result = ClassificationResultFFI {
                    label: label_cstr.into_raw(),
                    class_id: classification.class_id,
                    confidence: classification.confidence,
                    num_classes,
                    probabilities: probs,
                    processing_time_ms,
                    error: false,
                };
            }

            0
        }
        Err(e) => {
            eprintln!("Error: classification failed: {:?}", e);
            unsafe {
                *result = ClassificationResultFFI::default();
            }
            -1
        }
    }
}

/// Detect PII entities in text
///
/// # Parameters
/// - `classifier_name`: Name of the PII classifier
/// - `text`: Input text
/// - `result`: Output result pointer
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn detect_pii(
    classifier_name: *const c_char,
    text: *const c_char,
    result: *mut PIIResultFFI,
) -> i32 {
    if classifier_name.is_null() || text.is_null() || result.is_null() {
        return -1;
    }

    let name_str = unsafe {
        match CStr::from_ptr(classifier_name).to_str() {
            Ok(s) => s,
            Err(_) => {
                *result = PIIResultFFI::default();
                return -1;
            }
        }
    };

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                *result = PIIResultFFI::default();
                return -1;
            }
        }
    };

    let start_time = std::time::Instant::now();

    let mut classifiers = get_tok_classifiers().lock();
    let classifier = match classifiers.get_mut(name_str) {
        Some(c) => c,
        None => {
            eprintln!("Error: PII classifier '{}' not found", name_str);
            unsafe {
                *result = PIIResultFFI::default();
            }
            return -1;
        }
    };

    match classifier.detect_entities(text_str) {
        Ok(detection) => {
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            let entities: Vec<PIIEntityFFI> = detection
                .entities
                .into_iter()
                .map(|e| {
                    let text_cstr = CString::new(e.text).unwrap();
                    let type_cstr = CString::new(e.entity_type).unwrap();
                    PIIEntityFFI {
                        text: text_cstr.into_raw(),
                        entity_type: type_cstr.into_raw(),
                        start: e.start as i32,
                        end: e.end as i32,
                        confidence: e.confidence,
                    }
                })
                .collect();

            let num_entities = entities.len() as i32;
            let entities_ptr = if entities.is_empty() {
                std::ptr::null_mut()
            } else {
                Box::into_raw(entities.into_boxed_slice()) as *mut PIIEntityFFI
            };

            unsafe {
                *result = PIIResultFFI {
                    entities: entities_ptr,
                    num_entities,
                    processing_time_ms,
                    error: false,
                };
            }

            0
        }
        Err(e) => {
            eprintln!("Error: PII detection failed: {:?}", e);
            unsafe {
                *result = PIIResultFFI::default();
            }
            -1
        }
    }
}

// ============================================================================
// Memory Management
// ============================================================================

/// Free classification result
#[no_mangle]
pub extern "C" fn free_classification_result(result: *mut ClassificationResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let r = &mut *result;

        if !r.label.is_null() {
            let _ = CString::from_raw(r.label);
            r.label = std::ptr::null_mut();
        }

        if !r.probabilities.is_null() && r.num_classes > 0 {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(
                r.probabilities,
                r.num_classes as usize,
            ));
            r.probabilities = std::ptr::null_mut();
        }
    }
}

/// Free PII result
#[no_mangle]
pub extern "C" fn free_pii_result(result: *mut PIIResultFFI) {
    if result.is_null() {
        return;
    }

    unsafe {
        let r = &mut *result;

        if !r.entities.is_null() && r.num_entities > 0 {
            let entities = std::slice::from_raw_parts_mut(r.entities, r.num_entities as usize);

            for entity in entities.iter_mut() {
                if !entity.text.is_null() {
                    let _ = CString::from_raw(entity.text);
                }
                if !entity.entity_type.is_null() {
                    let _ = CString::from_raw(entity.entity_type);
                }
            }

            let _ = Box::from_raw(entities);
            r.entities = std::ptr::null_mut();
        }
    }
}

// ============================================================================
// Batch Classification
// ============================================================================

/// Classify multiple texts in batch
///
/// # Parameters
/// - `classifier_name`: Name of the classifier
/// - `texts`: Array of texts
/// - `num_texts`: Number of texts
/// - `results`: Output array (must be pre-allocated)
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn classify_batch(
    classifier_name: *const c_char,
    texts: *const *const c_char,
    num_texts: i32,
    results: *mut ClassificationResultFFI,
) -> i32 {
    if classifier_name.is_null() || texts.is_null() || results.is_null() || num_texts <= 0 {
        return -1;
    }

    let name_str = unsafe {
        match CStr::from_ptr(classifier_name).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };

    // Parse texts
    let mut text_strs = Vec::with_capacity(num_texts as usize);
    for i in 0..num_texts {
        let text_ptr = unsafe { *texts.offset(i as isize) };
        if text_ptr.is_null() {
            return -1;
        }
        let text_str = unsafe {
            match CStr::from_ptr(text_ptr).to_str() {
                Ok(s) => s,
                Err(_) => return -1,
            }
        };
        text_strs.push(text_str);
    }

    let start_time = std::time::Instant::now();

    let mut classifiers = get_seq_classifiers().lock();
    let classifier = match classifiers.get_mut(name_str) {
        Some(c) => c,
        None => {
            eprintln!("Error: classifier '{}' not found", name_str);
            return -1;
        }
    };

    match classifier.classify_batch(&text_strs) {
        Ok(classifications) => {
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
            let per_text_time = processing_time_ms / num_texts as f32;

            for (i, classification) in classifications.into_iter().enumerate() {
                let label_cstr = CString::new(classification.label).unwrap();
                let num_classes = classification.probabilities.len() as i32;
                let probs = Box::into_raw(classification.probabilities.into_boxed_slice()) as *mut f32;

                unsafe {
                    *results.offset(i as isize) = ClassificationResultFFI {
                        label: label_cstr.into_raw(),
                        class_id: classification.class_id,
                        confidence: classification.confidence,
                        num_classes,
                        probabilities: probs,
                        processing_time_ms: per_text_time,
                        error: false,
                    };
                }
            }

            0
        }
        Err(e) => {
            eprintln!("Error: batch classification failed: {:?}", e);
            for i in 0..num_texts as usize {
                unsafe {
                    *results.offset(i as isize) = ClassificationResultFFI::default();
                }
            }
            -1
        }
    }
}
