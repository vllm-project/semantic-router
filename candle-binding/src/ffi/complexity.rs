//! FFI for the complexity classifier.
//!
//! A fine-tuned ModernBERT sequence classifier that predicts prompt reasoning
//! difficulty. The predicted class index is mapped to a difficulty label
//! (conventionally easy/medium/hard) by the Go-side complexity mapping. This
//! mirrors the fact-check classifier; it is kept in its own module so the FFI
//! surface stays self-contained.

use std::ffi::{c_char, CStr};
use std::sync::{Arc, OnceLock};

use crate::ffi::types::ModernBertClassificationResult;
use crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier;

/// Trained complexity classifier (ModernBERT-based sequence classifier).
pub static COMPLEXITY_CLASSIFIER: OnceLock<Arc<TraditionalModernBertClassifier>> = OnceLock::new();

/// Initialize the ModernBERT complexity classifier.
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn init_complexity_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    // Idempotent: a second call after a successful init is a no-op success.
    if COMPLEXITY_CLASSIFIER.get().is_some() {
        println!("Complexity classifier already initialized");
        return true;
    }

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    println!("🔧 Initializing complexity classifier: {}", model_id);

    match TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => match COMPLEXITY_CLASSIFIER.set(Arc::new(model)) {
            Ok(_) => {
                println!("Complexity classifier initialized successfully");
                true
            }
            Err(_) => {
                // Already initialized by another thread; that's fine.
                println!("Complexity classifier already initialized (race condition)");
                true
            }
        },
        Err(e) => {
            eprintln!("Failed to initialize complexity classifier: {}", e);
            false
        }
    }
}

/// Classify text for prompt reasoning complexity.
///
/// The returned `predicted_class` index is mapped to a difficulty label
/// (conventionally easy/medium/hard) by the Go-side complexity mapping.
///
/// # Safety
/// - `text` must be a valid null-terminated C string.
#[no_mangle]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn classify_complexity_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
    };

    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Failed to convert text from C string");
                return default_result;
            }
        }
    };

    if let Some(classifier) = COMPLEXITY_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Complexity classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("Complexity classifier not initialized - call init_complexity_classifier first");
        default_result
    }
}
