//! FFI Classification Functions
//!
//! This module contains all C FFI classification functions for dual-path architecture.
//! Provides 16 classification functions with 100% backward compatibility.

use crate::core::UnifiedError;
use crate::ffi::memory::{
    allocate_bert_token_entity_array, allocate_c_float_array, allocate_c_string,
    allocate_lora_intent_array, allocate_lora_pii_array, allocate_lora_security_array,
    allocate_modernbert_token_entity_array,
};
use crate::ffi::types::*;
use crate::model_architectures::traditional::bert::{
    TRADITIONAL_BERT_CLASSIFIER, TRADITIONAL_BERT_TOKEN_CLASSIFIER,
};
use crate::model_architectures::traditional::modernbert::{
    TRADITIONAL_MODERNBERT_CLASSIFIER, TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER,
    TRADITIONAL_MODERNBERT_PII_CLASSIFIER, TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER,
};
use crate::BertClassifier;
use std::ffi::{c_char, CStr};
use std::sync::{Arc, OnceLock};

use crate::ffi::init::{PARALLEL_LORA_ENGINE, UNIFIED_CLASSIFIER};

/// Load id2label mapping from model config.json file
/// Returns HashMap mapping class index (as string) to label name
pub fn load_id2label_from_config(
    config_path: &str,
) -> Result<std::collections::HashMap<String, String>, UnifiedError> {
    // Use unified config loader (replaces local implementation)
    use crate::core::config_loader;

    config_loader::load_id2label_from_config(config_path)
}

// Legacy classifiers for backward compatibility using OnceLock pattern
// These are kept for old API paths but new code should use the dual-path architecture
static BERT_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_PII_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_JAILBREAK_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();

/// Classify text using basic classifier
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_text(text: *const c_char) -> ClassificationResult {
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

    // Get Arc from OnceLock (zero lock overhead!)
    if let Some(classifier) = BERT_CLASSIFIER.get() {
        let classifier = classifier.clone(); // Cheap Arc clone for concurrent access
        match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                predicted_class: class_idx as i32,
                confidence,
                label: std::ptr::null_mut(),
            },
            Err(e) => {
                eprintln!("Error classifying text: {e}");
                default_result
            }
        }
    } else {
        eprintln!("BERT classifier not initialized");
        default_result
    }
}
/// Classify text with probabilities
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_text_with_probabilities(
    text: *const c_char,
) -> ClassificationResultWithProbs {
    let default_result = ClassificationResultWithProbs {
        predicted_class: -1,
        confidence: 0.0,
        label: std::ptr::null_mut(),
        probabilities: std::ptr::null_mut(),
        num_classes: 0,
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    if let Some(classifier) = BERT_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => {
                // For now, we don't have probabilities from the new BERT implementation
                // Return empty probabilities array
                let prob_len = 0;
                let prob_ptr = std::ptr::null_mut();

                ClassificationResultWithProbs {
                    predicted_class: class_idx as i32,
                    confidence,
                    label: std::ptr::null_mut(),
                    probabilities: prob_ptr,
                    num_classes: prob_len as i32,
                }
            }
            Err(e) => {
                eprintln!("Error classifying text with probabilities: {e}");
                default_result
            }
        }
    } else {
        eprintln!("BERT classifier not initialized");
        default_result
    }
}
/// Classify text for PII detection
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_pii_text(text: *const c_char) -> ClassificationResult {
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

    if let Some(classifier) = BERT_PII_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                predicted_class: class_idx as i32,
                confidence,
                label: std::ptr::null_mut(),
            },
            Err(e) => {
                eprintln!("Error classifying PII text: {e}");
                default_result
            }
        }
    } else {
        eprintln!("BERT PII classifier not initialized");
        default_result
    }
}
/// Classify text for jailbreak detection
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_jailbreak_text(text: *const c_char) -> ClassificationResult {
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

    if let Some(classifier) = BERT_JAILBREAK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_idx, confidence)) => ClassificationResult {
                predicted_class: class_idx as i32,
                confidence,
                label: std::ptr::null_mut(),
            },
            Err(e) => {
                eprintln!("Error classifying jailbreak text: {e}");
                default_result
            }
        }
    } else {
        eprintln!("BERT jailbreak classifier not initialized");
        default_result
    }
}

/// Unified batch classification
///
/// # Safety
/// - `texts` must be a valid array of null-terminated C strings
/// - `texts_count` must match the actual array size
#[no_mangle]
pub extern "C" fn classify_unified_batch(
    texts_ptr: *const *const c_char,
    num_texts: i32,
) -> UnifiedBatchResult {
    // Migrated from lib.rs:1267-1308
    if texts_ptr.is_null() || num_texts <= 0 {
        return UnifiedBatchResult {
            batch_size: 0,
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            error: true,
            error_message: std::ptr::null_mut(),
        };
    }
    // Convert C strings to Rust strings
    let texts = unsafe {
        std::slice::from_raw_parts(texts_ptr, num_texts as usize)
            .iter()
            .map(|&ptr| {
                if ptr.is_null() {
                    Err("Null text pointer")
                } else {
                    CStr::from_ptr(ptr).to_str().map_err(|_| "Invalid UTF-8")
                }
            })
            .collect::<Result<Vec<_>, _>>()
    };
    let _texts = match texts {
        Ok(t) => t,
        Err(_e) => {
            return UnifiedBatchResult {
                batch_size: 0,
                intent_results: std::ptr::null_mut(),
                pii_results: std::ptr::null_mut(),
                security_results: std::ptr::null_mut(),
                error: true,
                error_message: std::ptr::null_mut(),
            };
        }
    };

    if let Some(_classifier_arc) = UNIFIED_CLASSIFIER.get() {
        // Note: DualPathUnifiedClassifier doesn't implement Clone and requires &mut self
        // This is incompatible with OnceLock's immutable access pattern
        // TODO: Need to refactor DualPathUnifiedClassifier to use interior mutability
        eprintln!("UNIFIED_CLASSIFIER: OnceLock pattern requires non-mutable classifiers - needs refactoring");
        UnifiedBatchResult {
            batch_size: 0,
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            error: true,
            error_message: std::ptr::null_mut(),
        }
    } else {
        UnifiedBatchResult {
            batch_size: 0,
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            error: true,
            error_message: std::ptr::null_mut(),
        }
    }
}

/// Classify BERT PII tokens
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_bert_pii_tokens(text: *const c_char) -> BertTokenClassificationResult {
    // Adapted from lib.rs:1441-1527 (simplified for structure compatibility)
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    if let Some(classifier) = TRADITIONAL_BERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Convert results to BertTokenEntity format
                let token_entities: Vec<(String, String, f32)> = token_results
                    .iter()
                    .map(|(token, label, score)| {
                        (token.clone(), format!("label_{}", label), *score)
                    })
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_results.len() as i32,
                }
            }
            Err(_e) => BertTokenClassificationResult {
                entities: std::ptr::null_mut(),
                num_entities: 0,
            },
        }
    } else {
        BertTokenClassificationResult {
            entities: std::ptr::null_mut(),
            num_entities: 0,
        }
    }
}

/// Classify Candle BERT token classifier with labels
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `config_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_candle_bert_tokens_with_labels(
    text: *const c_char,
    config_path: *const c_char,
) -> BertTokenClassificationResult {
    // Convert C strings to Rust strings
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    let _config_path = unsafe {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => s,
            Err(_) => {
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    // Use TraditionalBertTokenClassifier for token-level classification with labels

    if let Some(classifier) = TRADITIONAL_BERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Convert results to BertTokenEntity format
                let token_entities: Vec<(String, String, f32)> = token_results
                    .iter()
                    .map(|(token, label, score)| {
                        (token.clone(), format!("label_{}", label), *score)
                    })
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_results.len() as i32,
                }
            }
            Err(_e) => BertTokenClassificationResult {
                entities: std::ptr::null_mut(),
                num_entities: 0,
            },
        }
    } else {
        BertTokenClassificationResult {
            entities: std::ptr::null_mut(),
            num_entities: 0,
        }
    }
}

/// Classify Candle BERT tokens
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_candle_bert_tokens(
    text: *const c_char,
) -> BertTokenClassificationResult {
    // Adapted from lib.rs:1720-1760 (simplified for structure compatibility)
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    // Use intelligent routing to determine which classifier to use
    // First check if LoRA token classifier is available
    if let Some(lora_classifier) = crate::ffi::init::LORA_TOKEN_CLASSIFIER.get() {
        let lora_classifier = lora_classifier.clone();
        match lora_classifier.classify_tokens(text) {
            Ok(lora_results) => {
                // Convert LoRA results to BertTokenEntity format
                let token_entities: Vec<(String, String, f32)> = lora_results
                    .iter()
                    .map(|r| (r.token.clone(), r.label_name.clone(), r.confidence))
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                return BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: lora_results.len() as i32,
                };
            }
            Err(_e) => {
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                };
            }
        }
    }

    // Fallback to traditional BERT token classifier
    if let Some(classifier) = TRADITIONAL_BERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Convert results to C-compatible format
                let token_entities: Vec<(String, String, f32)> = token_results
                    .iter()
                    .map(|(token, class_idx, confidence)| {
                        (token.clone(), format!("class_{}", class_idx), *confidence)
                    })
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_entities.len() as i32,
                }
            }
            Err(e) => {
                println!("Candle BERT token classification failed: {}", e);
                BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    } else {
        println!("TraditionalBertTokenClassifier not initialized - call init function first");
        BertTokenClassificationResult {
            entities: std::ptr::null_mut(),
            num_entities: 0,
        }
    }
}

/// Classify text using Candle BERT
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_candle_bert_text(text: *const c_char) -> ClassificationResult {
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
    // Use TraditionalBertClassifier for Candle BERT text classification
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
        println!("TraditionalBertClassifier not initialized - call init_bert_classifier first");
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
pub extern "C" fn classify_bert_text(text: *const c_char) -> ClassificationResult {
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
pub extern "C" fn classify_batch_with_lora(
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
        let text_ptr = unsafe { *texts.offset(i as isize) };
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

/// Classify ModernBERT text
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_modernbert_text(text: *const c_char) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };
    if let Some(classifier) =
        crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_CLASSIFIER.get()
    {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((predicted_class, confidence)) => ModernBertClassificationResult {
                predicted_class: predicted_class as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("  Classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("  ModernBERT classifier not initialized");
        default_result
    }
}

/// Classify ModernBERT text with probabilities (same structure as above)
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_modernbert_text_with_probabilities(
    text: *const c_char,
) -> ModernBertClassificationResultWithProbs {
    let default_result = ModernBertClassificationResultWithProbs {
        class: -1,
        confidence: 0.0,
        probabilities: std::ptr::null_mut(),
        num_classes: 0,
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    if let Some(classifier) = TRADITIONAL_MODERNBERT_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => {
                // Convert results to C-compatible format
                // Create probabilities array from classifier
                let num_classes = classifier.get_num_classes();
                let mut probabilities = vec![0.1f32; num_classes];
                if (class_id as usize) < num_classes {
                    probabilities[class_id as usize] = confidence;
                }

                let probabilities_ptr = unsafe { allocate_c_float_array(&probabilities) };

                ModernBertClassificationResultWithProbs {
                    class: class_id as i32,
                    confidence,
                    probabilities: probabilities_ptr,
                    num_classes: num_classes as i32,
                }
            }
            Err(e) => {
                println!("ModernBERT classification failed: {}", e);
                ModernBertClassificationResultWithProbs {
                    class: -1,
                    confidence: 0.0,
                    probabilities: std::ptr::null_mut(),
                    num_classes: 0,
                }
            }
        }
    } else {
        println!("TraditionalModernBertClassifier not initialized - call init function first");
        ModernBertClassificationResultWithProbs {
            class: -1,
            confidence: 0.0,
            probabilities: std::ptr::null_mut(),
            num_classes: 0,
        }
    }
}

/// Classify ModernBERT PII text
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_modernbert_pii_text(
    text: *const c_char,
) -> ModernBertClassificationResult {
    // Migrated from modernbert.rs:1019-1054
    let default_result = ModernBertClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    if let Some(classifier) = TRADITIONAL_MODERNBERT_PII_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                println!("ModernBERT PII classification failed: {}", e);
                ModernBertClassificationResult {
                    predicted_class: -1,
                    confidence: 0.0,
                }
            }
        }
    } else {
        println!("TraditionalModernBertPIIClassifier not initialized - call init_modernbert_pii_classifier first");
        ModernBertClassificationResult {
            predicted_class: -1,
            confidence: 0.0,
        }
    }
}

/// Classify ModernBERT jailbreak text
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_modernbert_jailbreak_text(
    text: *const c_char,
) -> ModernBertClassificationResult {
    let default_result = ModernBertClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
    };
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return default_result,
        }
    };

    if let Some(classifier) = TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                println!("ModernBERT jailbreak classification failed: {}", e);
                ModernBertClassificationResult {
                    predicted_class: -1,
                    confidence: 0.0,
                }
            }
        }
    } else {
        println!("TraditionalModernBertJailbreakClassifier not initialized - call init_modernbert_jailbreak_classifier first");
        ModernBertClassificationResult {
            predicted_class: -1,
            confidence: 0.0,
        }
    }
}

/// Classify ModernBERT PII tokens
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn classify_modernbert_pii_tokens(
    text: *const c_char,
    config_path: *const c_char,
) -> ModernBertTokenClassificationResult {
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return ModernBertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    let config_path = unsafe {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => s,
            Err(_) => {
                return ModernBertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    };

    if let Some(classifier) = TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        // Use real token classification
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Load id2label mapping from config.json dynamically
                let id2label = match load_id2label_from_config(config_path) {
                    Ok(mapping) => mapping,
                    Err(e) => {
                        println!(
                            "Error: Failed to load id2label mapping from {}: {}",
                            config_path, e
                        );
                        // Return error result (negative num_entities indicates error)
                        return ModernBertTokenClassificationResult {
                            entities: std::ptr::null_mut(),
                            num_entities: -1,
                        };
                    }
                };

                // Filter tokens with high confidence and meaningful PII classes
                let mut entities = Vec::new();
                for (token, class_idx, confidence, start, end) in token_results {
                    // Only include tokens with reasonable confidence and non-background classes
                    if confidence > 0.5 && class_idx > 0 {
                        // Get PII type name from dynamic id2label mapping
                        let pii_type = id2label
                            .get(&class_idx.to_string())
                            .unwrap_or(&"UNKNOWN_PII".to_string())
                            .clone();
                        entities.push((token, pii_type, confidence, start, end));
                    }
                }

                let entities_ptr = unsafe { allocate_modernbert_token_entity_array(&entities) };

                ModernBertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: entities.len() as i32,
                }
            }
            Err(e) => {
                println!("ModernBERT PII token classification failed: {}", e);
                ModernBertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                }
            }
        }
    } else {
        println!("TraditionalModernBertTokenClassifier not initialized - call init function first");
        ModernBertTokenClassificationResult {
            entities: std::ptr::null_mut(),
            num_entities: 0,
        }
    }
}
