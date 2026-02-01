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
use crate::ffi::types::BertTokenEntity;
use crate::ffi::types::*;
use crate::model_architectures::traditional::bert::{
    TRADITIONAL_BERT_CLASSIFIER, TRADITIONAL_BERT_TOKEN_CLASSIFIER,
};
use crate::model_architectures::traditional::modernbert::{
    TRADITIONAL_MODERNBERT_CLASSIFIER, TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER,
    TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER, TRADITIONAL_MODERNBERT_PII_CLASSIFIER,
    TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER,
};
use crate::BertClassifier;
use std::ffi::CString;
use std::ffi::{c_char, CStr};
use std::sync::{Arc, OnceLock};

use crate::ffi::init::{
    FEEDBACK_DETECTOR_CLASSIFIER, LORA_INTENT_CLASSIFIER, LORA_JAILBREAK_CLASSIFIER,
    PARALLEL_LORA_ENGINE, UNIFIED_CLASSIFIER,
};
// Import DeBERTa classifier for jailbreak detection
use super::init::DEBERTA_JAILBREAK_CLASSIFIER;

// Classification constants for consistent category detection
/// PII detection positive class identifier (numeric)
const PII_POSITIVE_CLASS: usize = 1;
/// PII detection positive class identifier (string)
const PII_POSITIVE_CLASS_STR: &str = "1";

/// Security threat detection positive class identifier (numeric)
const SECURITY_THREAT_CLASS: usize = 1;
/// Security threat detection positive class identifier (string)
const SECURITY_THREAT_CLASS_STR: &str = "1";

/// Keywords used to identify security threats in category names
const SECURITY_THREAT_KEYWORDS: &[&str] = &["jailbreak", "unsafe", "threat"];

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
/// Classify text for jailbreak detection with LoRA auto-detection
///
/// Tries LoRA jailbreak classifier first, falls back to Traditional BERT
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

    // Try LoRA jailbreak classifier first (preferred for higher accuracy)
    if let Some(classifier) = LORA_JAILBREAK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_with_index(text) {
            Ok((class_idx, confidence, ref label)) => {
                // Allocate C string for label
                let label_ptr = unsafe { allocate_c_string(label) };

                return ClassificationResult {
                    predicted_class: class_idx as i32,
                    confidence,
                    label: label_ptr,
                };
            }
            Err(e) => {
                eprintln!(
                    "LoRA jailbreak classifier error: {}, falling back to Traditional BERT",
                    e
                );
                // Don't return - fall through to Traditional BERT classifier
            }
        }
    }

    // Fallback to Traditional BERT classifier
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
        eprintln!("No jailbreak classifier initialized - call init_jailbreak_classifier first");
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

    // Get classifier from OnceLock (now with interior mutability support)
    let classifier = match UNIFIED_CLASSIFIER.get() {
        Some(c) => c,
        None => {
            return UnifiedBatchResult {
                batch_size: 0,
                intent_results: std::ptr::null_mut(),
                pii_results: std::ptr::null_mut(),
                security_results: std::ptr::null_mut(),
                error: true,
                error_message: unsafe { allocate_c_string("Unified classifier not initialized") },
            };
        }
    };

    // Define tasks for unified classification (Intent, PII, Security)
    use crate::model_architectures::TaskType;
    let tasks = vec![TaskType::Intent, TaskType::PII, TaskType::Security];

    // Call classify_intelligent (now works with interior mutability)
    let text_refs: Vec<&str> = _texts.iter().map(|s| s.as_ref()).collect();
    match classifier.classify_intelligent(&text_refs, &tasks) {
        Ok(result) => {
            // Convert UnifiedClassificationResult to UnifiedBatchResult
            // Note: UnifiedClassificationResult provides aggregated results for the batch,
            // so we replicate the same result for each text in the batch
            // SAFETY: The batch_size passed to convert_unified_result_to_batch matches the number of texts (_texts.len()),
            // and memory allocation for the result is properly handled, satisfying the function's safety requirements.
            unsafe { convert_unified_result_to_batch(&result, _texts.len()) }
        }
        Err(e) => UnifiedBatchResult {
            batch_size: 0,
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            error: true,
            error_message: unsafe { allocate_c_string(&format!("Classification failed: {}", e)) },
        },
    }
}

/// Convert UnifiedClassificationResult to UnifiedBatchResult for FFI
///
/// # Safety
/// - Allocates C memory that must be freed by the caller
/// - batch_size must match the number of texts in the original request
unsafe fn convert_unified_result_to_batch(
    result: &crate::classifiers::unified::UnifiedClassificationResult,
    batch_size: usize,
) -> UnifiedBatchResult {
    use crate::ffi::types::{IntentResult, PIIResult, SecurityResult};
    use crate::model_architectures::TaskType;

    // Extract task results from the HashMap
    let intent_task_result = result.task_results.get(&TaskType::Intent);
    let pii_task_result = result.task_results.get(&TaskType::PII);
    let security_task_result = result.task_results.get(&TaskType::Security);

    // Allocate Intent results array
    let intent_results = if let Some(intent) = intent_task_result {
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            // Note: Replicating the same result for all texts since UnifiedClassificationResult
            // provides aggregated results, not per-text results
            // Use category_name from UnifiedTaskResult (loaded from model config)
            results.push(IntentResult {
                category: allocate_c_string(&intent.category_name),
                confidence: intent.confidence,
                probabilities: std::ptr::null_mut(), // No detailed probabilities available
                num_probabilities: 0,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut IntentResult
    } else {
        // If no Intent result, allocate default values
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            results.push(IntentResult {
                category: allocate_c_string("unknown"),
                confidence: 0.0,
                probabilities: std::ptr::null_mut(),
                num_probabilities: 0,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut IntentResult
    };

    // Allocate PII results array
    let pii_results = if let Some(pii) = pii_task_result {
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            // Use category_name to determine if PII is detected
            // Common PII labels: "no_pii", "has_pii" or "0", "1" etc.
            let has_pii = pii.category_name.to_lowercase().contains("pii")
                || pii.category_name == PII_POSITIVE_CLASS_STR
                || pii.predicted_class == PII_POSITIVE_CLASS;
            results.push(PIIResult {
                has_pii,
                pii_types: std::ptr::null_mut(), // No detailed PII types available
                num_pii_types: 0,
                confidence: pii.confidence,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut PIIResult
    } else {
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            results.push(PIIResult {
                has_pii: false,
                pii_types: std::ptr::null_mut(),
                num_pii_types: 0,
                confidence: 0.0,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut PIIResult
    };

    // Allocate Security results array
    let security_results = if let Some(security) = security_task_result {
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            // Use category_name to determine if jailbreak is detected
            // Common labels: "safe", "jailbreak", "unsafe" or "0", "1" etc.
            let category_lower = security.category_name.to_lowercase();
            let is_jailbreak = SECURITY_THREAT_KEYWORDS
                .iter()
                .any(|&keyword| category_lower.contains(keyword))
                || security.category_name == SECURITY_THREAT_CLASS_STR
                || security.predicted_class == SECURITY_THREAT_CLASS;

            // Use category_name as threat_type if jailbreak detected, otherwise "none"
            let threat_type = if is_jailbreak {
                &security.category_name
            } else {
                "none"
            };

            results.push(SecurityResult {
                is_jailbreak,
                threat_type: allocate_c_string(threat_type),
                confidence: security.confidence,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut SecurityResult
    } else {
        let mut results = Vec::with_capacity(batch_size);
        for _i in 0..batch_size {
            results.push(SecurityResult {
                is_jailbreak: false,
                threat_type: allocate_c_string("none"),
                confidence: 0.0,
            });
        }
        let boxed = results.into_boxed_slice();
        Box::into_raw(boxed) as *mut SecurityResult
    };

    UnifiedBatchResult {
        batch_size: batch_size as i32,
        intent_results,
        pii_results,
        security_results,
        error: false,
        error_message: std::ptr::null_mut(),
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

    // Intelligent routing: Check LoRA token classifier first, then fall back to traditional

    // Try LoRA token classifier first
    if let Some(classifier) = crate::ffi::init::LORA_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Filter out "O" (Outside) labels - only return actual entities
                let token_entities: Vec<(String, String, f32)> = token_results
                    .iter()
                    .filter(|result| result.label_name != "O" && result.label_id != 0)
                    .map(|result| {
                        (
                            result.token.clone(),
                            result.label_name.clone(),
                            result.confidence,
                        )
                    })
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                return BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_entities.len() as i32,
                };
            }
            Err(_e) => {
                // Fall through to traditional classifier
            }
        }
    }

    // Fall back to traditional BERT token classifier
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
                // Filter out "O" (Outside) labels - only return actual entities
                // Convert LoRA results to BertTokenEntity format
                let token_entities: Vec<(String, String, f32)> = lora_results
                    .iter()
                    .filter(|r| r.label_name != "O" && r.label_id != 0)
                    .map(|r| (r.token.clone(), r.label_name.clone(), r.confidence))
                    .collect();

                let entities_ptr = unsafe { allocate_bert_token_entity_array(&token_entities) };

                return BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_entities.len() as i32,
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

                return BertTokenClassificationResult {
                    entities: entities_ptr,
                    num_entities: token_entities.len() as i32,
                };
            }
            Err(e) => {
                println!("Candle BERT token classification failed: {}", e);
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                };
            }
        }
    }

    // Fallback to ModernBERT token classifier (for PII detection with ModernBERT models)
    if let Some(classifier) = TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_tokens(text) {
            Ok(token_results) => {
                // Filter non-background classes; Go layer applies confidence threshold
                // Keep real positions (start, end) for accurate entity extraction
                let token_entities: Vec<(String, String, f32, usize, usize)> = token_results
                    .iter()
                    .filter(|(_, class_idx, _, _, _)| *class_idx > 0)
                    .map(|(token, class_idx, confidence, start, end)| {
                        (
                            token.clone(),
                            format!("class_{}", class_idx),
                            *confidence,
                            *start,
                            *end,
                        )
                    })
                    .collect();

                let entities_ptr =
                    unsafe { allocate_modernbert_token_entity_array(&token_entities) };

                return BertTokenClassificationResult {
                    entities: entities_ptr as *mut BertTokenEntity,
                    num_entities: token_entities.len() as i32,
                };
            }
            Err(e) => {
                println!("ModernBERT token classification failed: {}", e);
                return BertTokenClassificationResult {
                    entities: std::ptr::null_mut(),
                    num_entities: 0,
                };
            }
        }
    }

    // No classifier available
    println!("No token classifier initialized (Traditional BERT, ModernBERT, or LoRA) - call init function first");
    BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
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

/// Classify text for jailbreak/prompt injection detection using DeBERTa v3
///
/// This function uses the ProtectAI DeBERTa v3 Base Prompt Injection model
/// to detect jailbreak attempts and prompt injection attacks with high accuracy.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `ClassificationResult` with:
/// - `predicted_class`: 0 for SAFE, 1 for INJECTION, -1 for error
/// - `confidence`: confidence score (0.0-1.0)
/// - `label`: null pointer (not used)
///
/// # Example
/// ```c
/// ClassificationResult result = classify_deberta_jailbreak_text("Ignore all previous instructions");
/// if (result.predicted_class == 1) {
///     printf("Injection detected with %.2f%% confidence\n", result.confidence * 100.0);
/// }
/// ```
#[no_mangle]
pub extern "C" fn classify_deberta_jailbreak_text(text: *const c_char) -> ClassificationResult {
    let default_result = ClassificationResult {
        predicted_class: -1,
        confidence: 0.0,
        label: std::ptr::null_mut(),
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

    if let Some(classifier) = DEBERTA_JAILBREAK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((label, confidence)) => {
                // Convert string label to class index
                // The model returns "SAFE" (0) or "INJECTION" (1)
                let predicted_class = if label == "INJECTION" { 1 } else { 0 };

                ClassificationResult {
                    predicted_class,
                    confidence,
                    label: std::ptr::null_mut(),
                }
            }
            Err(e) => {
                eprintln!("DeBERTa v3 jailbreak classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("DeBERTa v3 jailbreak classifier not initialized - call init_deberta_jailbreak_classifier first");
        default_result
    }
}

/// Classify text for fact-checking needs using halugate-sentinel model
///
/// This function uses the halugate-sentinel ModernBERT model to determine
/// whether a prompt requires external fact verification.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: 0 for NO_FACT_CHECK_NEEDED, 1 for FACT_CHECK_NEEDED, -1 for error
/// - `confidence`: confidence score (0.0-1.0)
///
/// # Example
/// ```c
/// ModernBertClassificationResult result = classify_fact_check_text("When was the Eiffel Tower built?");
/// if (result.predicted_class == 1) {
///     printf("Fact-checking needed with %.2f%% confidence\n", result.confidence * 100.0);
/// }
/// ```
#[no_mangle]
pub extern "C" fn classify_fact_check_text(text: *const c_char) -> ModernBertClassificationResult {
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

    if let Some(classifier) = TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Fact-check classification failed: {}", e);
                ModernBertClassificationResult {
                    predicted_class: -1,
                    confidence: 0.0,
                }
            }
        }
    } else {
        eprintln!("Fact-check classifier not initialized - call init_fact_check_classifier first");
        ModernBertClassificationResult {
            predicted_class: -1,
            confidence: 0.0,
        }
    }
}

/// Classify text for user feedback detection
///
/// This function uses the feedback-detector ModernBERT model to determine
/// user satisfaction from follow-up messages.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: 0=NEED_CLARIFICATION, 1=SAT, 2=WANT_DIFFERENT, 3=WRONG_ANSWER, -1=error
/// - `confidence`: confidence score (0.0-1.0)
///
/// # Model Label Mapping (from config.json)
/// - 0: NEED_CLARIFICATION - User needs more explanation
/// - 1: SAT - User is satisfied
/// - 2: WANT_DIFFERENT - User wants alternative options
/// - 3: WRONG_ANSWER - System provided incorrect information
///
/// # Example
/// ```c
/// ModernBertClassificationResult result = classify_feedback_text("Thanks!");
/// if (result.predicted_class == 1) {
///     printf("User is satisfied with %.2f%% confidence\n", result.confidence * 100.0);
/// }
/// ```
#[no_mangle]
pub extern "C" fn classify_feedback_text(text: *const c_char) -> ModernBertClassificationResult {
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

    if let Some(classifier) = FEEDBACK_DETECTOR_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("Feedback detection failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("Feedback detector not initialized - call init_feedback_detector first");
        default_result
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

/// Detect hallucinations in an LLM answer given context
///
/// This is a token-level classifier that determines if each token in the answer
/// is SUPPORTED (grounded in context) or HALLUCINATED (not grounded).
///
/// # Arguments
/// - `context`: Tool results or RAG context that should ground the answer
/// - `question`: The original user question
/// - `answer`: The LLM-generated answer to verify
/// - `threshold`: Confidence threshold for hallucination detection (0.0-1.0)
///                Only tokens with confidence >= threshold are considered hallucinated
///
/// # Safety
/// - `context` must be a valid null-terminated C string (tool results/RAG context)
/// - `question` must be a valid null-terminated C string (user's question)
/// - `answer` must be a valid null-terminated C string (LLM's response)
#[no_mangle]
pub extern "C" fn detect_hallucinations(
    context: *const c_char,
    question: *const c_char,
    answer: *const c_char,
    threshold: f32,
) -> HallucinationDetectionResult {
    // Parse input strings
    let context = unsafe {
        match CStr::from_ptr(context).to_str() {
            Ok(s) => s,
            Err(_) => {
                return HallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid context string"),
                    ..Default::default()
                }
            }
        }
    };

    let question = unsafe {
        match CStr::from_ptr(question).to_str() {
            Ok(s) => s,
            Err(_) => {
                return HallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid question string"),
                    ..Default::default()
                }
            }
        }
    };

    let answer = unsafe {
        match CStr::from_ptr(answer).to_str() {
            Ok(s) => s,
            Err(_) => {
                return HallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid answer string"),
                    ..Default::default()
                }
            }
        }
    };

    // Check if model is initialized
    let classifier = match crate::ffi::init::HALLUCINATION_CLASSIFIER.get() {
        Some(c) => c.clone(),
        None => {
            return HallucinationDetectionResult {
                error: true,
                error_message: unsafe {
                    allocate_c_string("Hallucination detection model not initialized")
                },
                ..Default::default()
            }
        }
    };

    // Hallucination detector expects context and answer as separate segments
    // Format: context [SEP] answer (the tokenizer will add [CLS] and final [SEP])
    // We need to include question in context if provided
    let full_context = if question.is_empty() {
        context.to_string()
    } else {
        format!("{} Question: {}", context, question)
    };

    // Combine context and answer with separator
    // ModernBERT tokenizer uses [SEP] token (id 50282) to separate segments
    let formatted_input = format!("{} [SEP] {}", full_context, answer);

    // Find where answer starts (after [SEP])
    let answer_char_start = full_context.len() + " [SEP] ".len();

    // Classify tokens
    match classifier.classify_tokens(&formatted_input) {
        Ok(token_results) => {
            // Only process tokens that are part of the answer (after context + separator)
            let answer_start_pos = answer_char_start;

            // Collect hallucinated spans
            // Label 1 = HALLUCINATED, Label 0 = SUPPORTED
            let mut hallucinated_spans: Vec<(String, i32, i32, f32, String)> = Vec::new();
            let mut current_span_start: Option<i32> = None;
            let mut current_span_end: Option<i32> = None;
            let mut current_span_confidence: f32 = 0.0;
            let mut hallucination_token_count = 0;
            let mut total_answer_tokens = 0;
            let mut max_hallucination_confidence: f32 = 0.0;

            for (_token, class_idx, confidence, start, _end) in &token_results {
                // Only process tokens in the answer section
                let token_start = *start as usize;
                if token_start < answer_start_pos {
                    continue;
                }

                total_answer_tokens += 1;

                // Use provided threshold (default to 0.5 if invalid)
                let effective_threshold = if threshold > 0.0 && threshold <= 1.0 {
                    threshold
                } else {
                    0.5
                };

                // Check if token is hallucinated (class 1) AND confidence >= threshold
                if *class_idx == 1 && *confidence >= effective_threshold {
                    hallucination_token_count += 1;
                    if *confidence > max_hallucination_confidence {
                        max_hallucination_confidence = *confidence;
                    }

                    // Start or continue a hallucinated span
                    // Use character offsets to track span boundaries in the original answer
                    let token_offset_in_answer = (*start as i32) - answer_start_pos as i32;
                    let token_end_in_answer = (*_end as i32) - answer_start_pos as i32;

                    if current_span_start.is_none() {
                        current_span_start = Some(token_offset_in_answer);
                        current_span_end = Some(token_end_in_answer);
                        current_span_confidence = *confidence;
                    } else {
                        // Extend the span end
                        current_span_end = Some(token_end_in_answer);
                    }

                    // Update confidence to max of span
                    if *confidence > current_span_confidence {
                        current_span_confidence = *confidence;
                    }
                } else {
                    // End current span if there was one
                    if let (Some(start_pos), Some(end_pos)) = (current_span_start, current_span_end)
                    {
                        // Extract span text from original answer using offsets
                        let span_text = if start_pos >= 0
                            && end_pos > start_pos
                            && (end_pos as usize) <= answer.len()
                        {
                            answer[start_pos as usize..end_pos as usize].to_string()
                        } else {
                            // Fallback: empty string if offsets are invalid
                            String::new()
                        };

                        if !span_text.is_empty() {
                            hallucinated_spans.push((
                                span_text,
                                start_pos,
                                end_pos,
                                current_span_confidence,
                                "HALLUCINATED".to_string(),
                            ));
                        }
                        current_span_start = None;
                        current_span_end = None;
                        current_span_confidence = 0.0;
                    }
                }
            }

            // Don't forget the last span
            if let (Some(start_pos), Some(end_pos)) = (current_span_start, current_span_end) {
                // Extract span text from original answer using offsets
                let span_text = if start_pos >= 0
                    && end_pos > start_pos
                    && (end_pos as usize) <= answer.len()
                {
                    answer[start_pos as usize..end_pos as usize].to_string()
                } else {
                    String::new()
                };

                if !span_text.is_empty() {
                    hallucinated_spans.push((
                        span_text,
                        start_pos,
                        end_pos,
                        current_span_confidence,
                        "HALLUCINATED".to_string(),
                    ));
                }
            }

            // Calculate overall confidence
            let has_hallucination = !hallucinated_spans.is_empty();
            let overall_confidence = if has_hallucination {
                max_hallucination_confidence
            } else if total_answer_tokens > 0 {
                // If no hallucination, confidence is based on how sure we are
                1.0 - (hallucination_token_count as f32 / total_answer_tokens as f32)
            } else {
                1.0
            };

            // Allocate spans array
            let num_spans = hallucinated_spans.len() as i32;
            let spans_ptr = if num_spans > 0 {
                unsafe { allocate_hallucination_span_array(&hallucinated_spans) }
            } else {
                std::ptr::null_mut()
            };

            HallucinationDetectionResult {
                has_hallucination,
                confidence: overall_confidence,
                spans: spans_ptr,
                num_spans,
                error: false,
                error_message: std::ptr::null_mut(),
            }
        }
        Err(e) => HallucinationDetectionResult {
            error: true,
            error_message: unsafe { allocate_c_string(&format!("Classification failed: {}", e)) },
            ..Default::default()
        },
    }
}

/// Allocate hallucination span array for FFI
/// # Safety
/// Caller must free the memory using free_hallucination_detection_result
unsafe fn allocate_hallucination_span_array(
    spans: &[(String, i32, i32, f32, String)],
) -> *mut HallucinationSpan {
    if spans.is_empty() {
        return std::ptr::null_mut();
    }

    let layout = std::alloc::Layout::array::<HallucinationSpan>(spans.len()).unwrap();
    let ptr = std::alloc::alloc(layout) as *mut HallucinationSpan;

    for (i, (text, start, end, confidence, label)) in spans.iter().enumerate() {
        let span = HallucinationSpan {
            text: allocate_c_string(text),
            start: *start,
            end: *end,
            confidence: *confidence,
            label: allocate_c_string(label),
        };
        std::ptr::write(ptr.add(i), span);
    }

    ptr
}

/// Classify NLI (Natural Language Inference) for a premise-hypothesis pair
///
/// This function determines the logical relationship between a premise (context)
/// and a hypothesis (claim/span). Used for post-processing hallucination detection.
///
/// # Returns
/// - Entailment (0): The premise supports the hypothesis
/// - Neutral (1): The premise neither supports nor contradicts
/// - Contradiction (2): The premise contradicts the hypothesis
///
/// # Safety
/// - `premise` must be a valid null-terminated C string (e.g., context/tool results)
/// - `hypothesis` must be a valid null-terminated C string (e.g., the claim to verify)
#[no_mangle]
pub extern "C" fn classify_nli(premise: *const c_char, hypothesis: *const c_char) -> NLIResult {
    // Parse input strings
    let premise = unsafe {
        match CStr::from_ptr(premise).to_str() {
            Ok(s) => s,
            Err(_) => {
                return NLIResult {
                    error: true,
                    error_message: allocate_c_string("Invalid premise string"),
                    ..Default::default()
                }
            }
        }
    };

    let hypothesis = unsafe {
        match CStr::from_ptr(hypothesis).to_str() {
            Ok(s) => s,
            Err(_) => {
                return NLIResult {
                    error: true,
                    error_message: allocate_c_string("Invalid hypothesis string"),
                    ..Default::default()
                }
            }
        }
    };

    // Check if NLI model is initialized
    let classifier = match crate::ffi::init::NLI_CLASSIFIER.get() {
        Some(c) => c.clone(),
        None => {
            return NLIResult {
                error: true,
                error_message: unsafe { allocate_c_string("NLI model not initialized") },
                ..Default::default()
            }
        }
    };

    // Format input for NLI: premise [SEP] hypothesis
    // ModernBERT NLI models use [SEP] token (id 50282) to separate segments
    let nli_input = format!("{} [SEP] {}", premise, hypothesis);

    // Classify
    match classifier.classify_text(&nli_input) {
        Ok((class_idx, confidence)) => {
            // Map class index to NLI label
            // Standard NLI ordering: 0=entailment, 1=neutral, 2=contradiction
            let label = match class_idx {
                0 => NLILabel::Entailment,
                1 => NLILabel::Neutral,
                2 => NLILabel::Contradiction,
                _ => NLILabel::Error,
            };

            // Get probabilities if available (approximate from confidence)
            // In a full implementation, we'd get all probabilities from the classifier
            let (entailment_prob, neutral_prob, contradiction_prob) = match class_idx {
                0 => (
                    confidence,
                    (1.0 - confidence) / 2.0,
                    (1.0 - confidence) / 2.0,
                ),
                1 => (
                    (1.0 - confidence) / 2.0,
                    confidence,
                    (1.0 - confidence) / 2.0,
                ),
                2 => (
                    (1.0 - confidence) / 2.0,
                    (1.0 - confidence) / 2.0,
                    confidence,
                ),
                _ => (0.0, 0.0, 0.0),
            };

            NLIResult {
                label,
                confidence,
                entailment_prob,
                neutral_prob,
                contradiction_prob,
                error: false,
                error_message: std::ptr::null_mut(),
            }
        }
        Err(e) => NLIResult {
            error: true,
            error_message: unsafe {
                allocate_c_string(&format!("NLI classification failed: {}", e))
            },
            ..Default::default()
        },
    }
}

/// Detect hallucinations with NLI explanations (enhanced pipeline)
///
/// This function combines token-level hallucination detection with NLI-based
/// verification to provide detailed explanations for each hallucinated span.
///
/// Pipeline:
/// 1. Run hallucination detector to find hallucinated spans
/// 2. For each span, run NLI (context vs. span) to classify as:
///    - CONTRADICTION: Direct conflict with context (severity 4)
///    - NEUTRAL: Not supported by context / fabrication (severity 2)
///    - ENTAILMENT: Supported (false positive, remove from results)
///
/// # Arguments
/// - `threshold`: Confidence threshold for hallucination detection (0.0-1.0)
///
/// # Safety
/// - `context` must be a valid null-terminated C string
/// - `question` must be a valid null-terminated C string
/// - `answer` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn detect_hallucinations_with_nli(
    context: *const c_char,
    question: *const c_char,
    answer: *const c_char,
    threshold: f32,
) -> EnhancedHallucinationDetectionResult {
    // Parse input strings
    let context_str = unsafe {
        match CStr::from_ptr(context).to_str() {
            Ok(s) => s,
            Err(_) => {
                return EnhancedHallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid context string"),
                    ..Default::default()
                }
            }
        }
    };

    let question_str = unsafe {
        match CStr::from_ptr(question).to_str() {
            Ok(s) => s,
            Err(_) => {
                return EnhancedHallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid question string"),
                    ..Default::default()
                }
            }
        }
    };

    let _answer_str = unsafe {
        match CStr::from_ptr(answer).to_str() {
            Ok(s) => s,
            Err(_) => {
                return EnhancedHallucinationDetectionResult {
                    error: true,
                    error_message: allocate_c_string("Invalid answer string"),
                    ..Default::default()
                }
            }
        }
    };

    // Step 1: Run hallucination detection with threshold
    let hallucination_result = detect_hallucinations(context, question, answer, threshold);

    if hallucination_result.error {
        return EnhancedHallucinationDetectionResult {
            error: true,
            error_message: hallucination_result.error_message,
            ..Default::default()
        };
    }

    // If no hallucinations detected, return early
    if !hallucination_result.has_hallucination || hallucination_result.num_spans == 0 {
        // Free the hallucination detection result
        crate::ffi::memory::free_hallucination_detection_result(hallucination_result);
        return EnhancedHallucinationDetectionResult {
            has_hallucination: false,
            confidence: 1.0,
            spans: std::ptr::null_mut(),
            num_spans: 0,
            error: false,
            error_message: std::ptr::null_mut(),
        };
    }

    // Check if NLI model is available
    let nli_available = crate::ffi::init::NLI_CLASSIFIER.get().is_some();

    // Step 2: Process each span with NLI
    let mut enhanced_spans: Vec<EnhancedHallucinationSpan> = Vec::new();

    unsafe {
        let spans_slice = std::slice::from_raw_parts(
            hallucination_result.spans,
            hallucination_result.num_spans as usize,
        );

        for span in spans_slice {
            let span_text = if !span.text.is_null() {
                CStr::from_ptr(span.text).to_str().unwrap_or("")
            } else {
                ""
            };

            // Skip empty spans
            if span_text.is_empty() {
                continue;
            }

            // Run NLI on this span (context = premise, span = hypothesis)
            let (nli_label, nli_confidence, severity, explanation) = if nli_available {
                let nli_input_premise = format!("{} {}", context_str, question_str);
                let premise_cstr = std::ffi::CString::new(nli_input_premise).unwrap();
                let hypothesis_cstr = std::ffi::CString::new(span_text).unwrap();

                let nli_result = classify_nli(premise_cstr.as_ptr(), hypothesis_cstr.as_ptr());

                if nli_result.error {
                    // If NLI fails, use hallucination detector only
                    (
                        NLILabel::Neutral,
                        0.0,
                        2,
                        "NLI classification failed, based on hallucination detector only"
                            .to_string(),
                    )
                } else {
                    let (sev, expl) = match nli_result.label {
                        NLILabel::Contradiction => {
                            (4, format!("CONTRADICTION: This claim directly conflicts with the provided context (confidence: {:.1}%)", nli_result.confidence * 100.0))
                        }
                        NLILabel::Neutral => {
                            (2, format!("FABRICATION: This claim is not supported by the provided context (confidence: {:.1}%)", nli_result.confidence * 100.0))
                        }
                        NLILabel::Entailment => {
                            // NLI says it's supported, but hallucination detector flagged it
                            // This is a potential false positive from the detector
                            // Include with low severity as it may be a subtle issue
                            (1, format!("UNCERTAIN: Hallucination detector flagged this but NLI suggests it may be supported (confidence: {:.1}%)", nli_result.confidence * 100.0))
                        }
                        NLILabel::Error => {
                            (2, "Unable to determine relationship with context".to_string())
                        }
                    };
                    (nli_result.label, nli_result.confidence, sev, expl)
                }
            } else {
                // No NLI model, use hallucination detector confidence only
                let sev = if span.confidence > 0.8 { 3 } else { 2 };
                (
                    NLILabel::Neutral,
                    0.0,
                    sev,
                    format!(
                        "Unsupported claim detected (confidence: {:.1}%)",
                        span.confidence * 100.0
                    ),
                )
            };

            enhanced_spans.push(EnhancedHallucinationSpan {
                text: allocate_c_string(span_text),
                start: span.start,
                end: span.end,
                hallucination_confidence: span.confidence,
                nli_label,
                nli_confidence,
                severity,
                explanation: allocate_c_string(&explanation),
            });
        }

        // Free the original hallucination detection result
        crate::ffi::memory::free_hallucination_detection_result(hallucination_result);
    }

    // Allocate enhanced spans array
    let num_spans = enhanced_spans.len() as i32;
    let has_hallucination = num_spans > 0;

    // Calculate overall confidence (max of individual confidences)
    let overall_confidence = enhanced_spans
        .iter()
        .map(|s| s.hallucination_confidence.max(s.nli_confidence))
        .fold(0.0f32, |acc, c| acc.max(c));

    let spans_ptr = if num_spans > 0 {
        let layout =
            std::alloc::Layout::array::<EnhancedHallucinationSpan>(num_spans as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut EnhancedHallucinationSpan };

        for (i, span) in enhanced_spans.into_iter().enumerate() {
            unsafe {
                std::ptr::write(ptr.add(i), span);
            }
        }
        ptr
    } else {
        std::ptr::null_mut()
    };

    EnhancedHallucinationDetectionResult {
        has_hallucination,
        confidence: overall_confidence,
        spans: spans_ptr,
        num_spans,
        error: false,
        error_message: std::ptr::null_mut(),
    }
}

// ============================================================================
// mmBERT-32K Classification Functions (32K context, YaRN RoPE scaling)
// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
// ============================================================================

use crate::ffi::init::{
    MMBERT_32K_FACTCHECK_CLASSIFIER, MMBERT_32K_FEEDBACK_CLASSIFIER, MMBERT_32K_INTENT_CLASSIFIER,
    MMBERT_32K_JAILBREAK_CLASSIFIER, MMBERT_32K_PII_CLASSIFIER,
};

/// Classify text using mmBERT-32K intent classifier
///
/// Classifies text into MMLU-Pro academic categories for intelligent request routing.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: category index (0-13), -1 for error
/// - `confidence`: confidence score (0.0-1.0)
#[no_mangle]
pub extern "C" fn classify_mmbert_32k_intent(
    text: *const c_char,
) -> ModernBertClassificationResult {
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

    if let Some(classifier) = MMBERT_32K_INTENT_CLASSIFIER.get() {
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("mmBERT-32K intent classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K intent classifier not initialized");
        default_result
    }
}

/// Classify text using mmBERT-32K fact-check classifier
///
/// Determines if text needs fact-checking.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: 0=NO_FACT_CHECK_NEEDED, 1=FACT_CHECK_NEEDED, -1=error
/// - `confidence`: confidence score (0.0-1.0)
#[no_mangle]
pub extern "C" fn classify_mmbert_32k_factcheck(
    text: *const c_char,
) -> ModernBertClassificationResult {
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

    if let Some(classifier) = MMBERT_32K_FACTCHECK_CLASSIFIER.get() {
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("mmBERT-32K fact-check classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K fact-check classifier not initialized");
        default_result
    }
}

/// Classify text using mmBERT-32K jailbreak detector
///
/// Detects prompt injection/jailbreak attempts.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: 0=benign, 1=jailbreak, -1=error
/// - `confidence`: confidence score (0.0-1.0)
#[no_mangle]
pub extern "C" fn classify_mmbert_32k_jailbreak(
    text: *const c_char,
) -> ModernBertClassificationResult {
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

    if let Some(classifier) = MMBERT_32K_JAILBREAK_CLASSIFIER.get() {
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("mmBERT-32K jailbreak classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K jailbreak classifier not initialized");
        default_result
    }
}

/// Classify text using mmBERT-32K feedback detector
///
/// Detects user satisfaction from follow-up messages.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
///
/// # Returns
/// `ModernBertClassificationResult` with:
/// - `predicted_class`: 0=SAT, 1=NEED_CLARIFICATION, 2=WRONG_ANSWER, 3=WANT_DIFFERENT, -1=error
/// - `confidence`: confidence score (0.0-1.0)
#[no_mangle]
pub extern "C" fn classify_mmbert_32k_feedback(
    text: *const c_char,
) -> ModernBertClassificationResult {
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

    if let Some(classifier) = MMBERT_32K_FEEDBACK_CLASSIFIER.get() {
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("mmBERT-32K feedback classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K feedback classifier not initialized");
        default_result
    }
}

/// Classify tokens for PII detection using mmBERT-32K
///
/// Detects 17 types of PII entities using BIO tagging.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `model_config_path` must be a valid null-terminated C string or null
///
/// # Returns
/// `ModernBertTokenClassificationResult` with detected PII entities
#[no_mangle]
pub extern "C" fn classify_mmbert_32k_pii_tokens(
    text: *const c_char,
    model_config_path: *const c_char,
) -> ModernBertTokenClassificationResult {
    let default_result = ModernBertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
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

    let config_path = if model_config_path.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(model_config_path).to_str() {
                Ok(s) => Some(s),
                Err(_) => None,
            }
        }
    };

    if let Some(classifier) = MMBERT_32K_PII_CLASSIFIER.get() {
        match classifier.classify_tokens(text) {
            Ok(entities) => {
                let num_entities = entities.len() as i32;
                if num_entities == 0 {
                    return default_result;
                }

                // Allocate memory for entities
                let layout =
                    std::alloc::Layout::array::<ModernBertTokenEntity>(num_entities as usize)
                        .unwrap();
                let ptr = unsafe { std::alloc::alloc(layout) as *mut ModernBertTokenEntity };

                // Load label mapping if config path provided
                let label_map: Option<std::collections::HashMap<usize, String>> = config_path
                    .and_then(|path| {
                        let label_path = std::path::Path::new(path)
                            .parent()
                            .map(|p| p.join("label_mapping.json"))
                            .unwrap_or_else(|| std::path::PathBuf::from("label_mapping.json"));
                        std::fs::read_to_string(&label_path)
                            .ok()
                            .and_then(|s| serde_json::from_str(&s).ok())
                    });

                for (i, (_label, class_id, confidence, start, end)) in
                    entities.into_iter().enumerate()
                {
                    let entity_type = label_map
                        .as_ref()
                        .and_then(|m| m.get(&class_id))
                        .cloned()
                        .unwrap_or_else(|| format!("LABEL_{}", class_id));

                    let entity_text = if start < text.len() && end <= text.len() {
                        &text[start..end]
                    } else {
                        ""
                    };

                    unsafe {
                        std::ptr::write(
                            ptr.add(i),
                            ModernBertTokenEntity {
                                entity_type: CString::new(entity_type).unwrap().into_raw(),
                                start: start as i32,
                                end: end as i32,
                                text: CString::new(entity_text).unwrap().into_raw(),
                                confidence,
                            },
                        );
                    }
                }

                ModernBertTokenClassificationResult {
                    entities: ptr,
                    num_entities,
                }
            }
            Err(e) => {
                eprintln!("mmBERT-32K PII classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K PII classifier not initialized");
        default_result
    }
}
