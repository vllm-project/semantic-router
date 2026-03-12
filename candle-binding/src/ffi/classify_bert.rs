//! BERT/LoRA text classification FFI functions.
//!
//! Provides classify_candle_bert_text, classify_bert_text, and classify_batch_with_lora.

use crate::ffi::init::{LORA_INTENT_CLASSIFIER, PARALLEL_LORA_ENGINE};
use crate::ffi::memory::{
    allocate_c_string, allocate_lora_intent_array, allocate_lora_pii_array,
    allocate_lora_security_array,
};
use crate::ffi::types::{ClassificationResult, LoRABatchResult};
use crate::model_architectures::traditional::bert::TRADITIONAL_BERT_CLASSIFIER;
use std::ffi::{c_char, CStr};

/// Classify text using Candle BERT
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_candle_bert_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
        label: std::ptr::null_mut(),
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    // Try LoRA intent classifier first (preferred for higher accuracy)
    if let Some(classifier) = LORA_INTENT_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_with_index(text) {
            Ok((class_idx, confidence, ref intent)) => {
                // Allocate C string for intent label
                let label_ptr = unsafe { allocate_c_string(intent) };

                return ClassificationResult {
                    predicted_class: class_idx as i32,
                    confidence,
                    label: label_ptr,
                };
            }
            Err(e) => {
                eprintln!(
                    "LoRA intent classifier error: {}, falling back to Traditional BERT",
                    e
                );
                // Don't return - fall through to Traditional BERT classifier
            }
        }
    }

    // Fallback to Traditional BERT classifier
    if let Some(classifier) = TRADITIONAL_BERT_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => {
                // Allocate C string for class label
                let label_ptr = unsafe { allocate_c_string(&format!("class_{}", class_id)) };

                ClassificationResult {
                    predicted_class: class_id as i32,
                    confidence,
                    label: label_ptr,
                }
            }
            Err(e) => {
                println!("Candle BERT text classification failed: {}", e);
                ClassificationResult {
                    predicted_class: -1,
                    confidence: 0.0,
                    label: std::ptr::null_mut(),
                }
            }
        }
    } else {
        println!("No classifier initialized - call init_candle_bert_classifier first");
        ClassificationResult {
            predicted_class: -1,
            confidence: 0.0,
            label: std::ptr::null_mut(),
        }
    }
}

/// Classify text using BERT
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_bert_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
        label: std::ptr::null_mut(),
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };
    if let Some(classifier) = TRADITIONAL_BERT_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => {
                // Allocate C string for class label
                let label_ptr = unsafe { allocate_c_string(&format!("class_{}", class_id)) };

                ClassificationResult {
                    predicted_class: class_id as i32,
                    confidence,
                    label: label_ptr,
                }
            }
            Err(e) => {
                println!("BERT text classification failed: {}", e);
                ClassificationResult {
                    predicted_class: -1,
                    confidence: 0.0,
                    label: std::ptr::null_mut(),
                }
            }
        }
    } else {
        println!("TraditionalBertClassifier not initialized - call init_bert_classifier first");
        ClassificationResult {
            predicted_class: -1,
            confidence: 0.0,
            label: std::ptr::null_mut(),
        }
    }
}

/// Classify batch with LoRA (high-performance parallel path)
///
/// # Safety
/// - `texts` must be a valid array of null-terminated C strings
/// - `texts_count` must match the actual array size
#[no_mangle]
pub unsafe extern "C" fn classify_batch_with_lora(
    texts: *const *const c_char,
    texts_count: usize,
) -> LoRABatchResult {
    let default_result = LoRABatchResult {
        intent_results: std::ptr::null_mut(),
        pii_results: std::ptr::null_mut(),
        security_results: std::ptr::null_mut(),
        batch_size: 0,
        avg_confidence: 0.0,
    };
    if texts_count == 0 {
        return default_result;
    }
    // Convert C strings to Rust strings
    let mut text_vec = Vec::new();
    for i in 0..texts_count {
        let text_ptr = unsafe { *texts.add(i) };
        let text = unsafe {
            match CStr::from_ptr(text_ptr).to_str() {
                Ok(s) => s,
                Err(_) => return default_result,
            }
        };
        text_vec.push(text);
    }

    let start_time = std::time::Instant::now();

    // Get Arc from OnceLock (zero lock overhead!)
    // OnceLock.get() is just an atomic load - no mutex, no contention
    let engine = match PARALLEL_LORA_ENGINE.get() {
        Some(e) => e.clone(), // Cheap Arc clone for concurrent access
        None => {
            eprintln!("PARALLEL_LORA_ENGINE not initialized");
            return default_result;
        }
    };

    // Now perform inference without holding the lock (allows concurrent requests)
    let text_refs: Vec<&str> = text_vec.iter().map(|s| s.as_ref()).collect();
    match engine.parallel_classify(&text_refs) {
        Ok(parallel_result) => {
            let _processing_time_ms = start_time.elapsed().as_millis() as f32;

            // Allocate C arrays for LoRA results
            let intent_results_ptr =
                unsafe { allocate_lora_intent_array(&parallel_result.intent_results) };
            let pii_results_ptr = unsafe { allocate_lora_pii_array(&parallel_result.pii_results) };
            let security_results_ptr =
                unsafe { allocate_lora_security_array(&parallel_result.security_results) };

            LoRABatchResult {
                intent_results: intent_results_ptr,
                pii_results: pii_results_ptr,
                security_results: security_results_ptr,
                batch_size: texts_count as i32,
                avg_confidence: {
                    let mut total_confidence = 0.0f32;
                    let mut count = 0;

                    // Sum intent confidences
                    for intent in &parallel_result.intent_results {
                        total_confidence += intent.confidence;
                        count += 1;
                    }

                    // Sum PII confidences
                    for pii in &parallel_result.pii_results {
                        total_confidence += pii.confidence;
                        count += 1;
                    }

                    // Sum security confidences
                    for security in &parallel_result.security_results {
                        total_confidence += security.confidence;
                        count += 1;
                    }

                    if count > 0 {
                        total_confidence / count as f32
                    } else {
                        0.0
                    }
                },
            }
        }
        Err(e) => {
            println!("LoRA parallel classification failed: {}", e);
            LoRABatchResult {
                intent_results: std::ptr::null_mut(),
                pii_results: std::ptr::null_mut(),
                security_results: std::ptr::null_mut(),
                batch_size: 0,
                avg_confidence: 0.0,
            }
        }
    }
}
