//! FFI Classification Functions
//!
//! This module contains core C FFI classification functions for dual-path architecture.
//! Legacy BERT, batch, and BERT token classifiers. See classify_bert, classify_modernbert,
//! classify_hallucination, classify_mmbert for additional classifiers.

use crate::core::UnifiedError;
use crate::ffi::memory::{
    allocate_bert_token_entity_array, allocate_c_string, allocate_modernbert_token_entity_array,
};
use crate::ffi::types::BertTokenEntity;
use crate::ffi::types::*;
use crate::model_architectures::traditional::bert::TRADITIONAL_BERT_TOKEN_CLASSIFIER;
use crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER;
use crate::BertClassifier;
use std::ffi::{c_char, CStr};
use std::sync::{Arc, OnceLock};

use crate::ffi::init::{LORA_JAILBREAK_CLASSIFIER, UNIFIED_CLASSIFIER};

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
    use crate::core::config_loader;

    config_loader::load_id2label_from_config(config_path)
}

// Legacy classifiers for backward compatibility using OnceLock pattern
static BERT_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_PII_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_JAILBREAK_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();

/// Classify text using basic classifier
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_text(text: *const c_char) -> ClassificationResult {
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

    if let Some(classifier) = BERT_CLASSIFIER.get() {
        let classifier = classifier.clone();
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
pub unsafe extern "C" fn classify_text_with_probabilities(
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
            Ok((class_idx, confidence)) => ClassificationResultWithProbs {
                predicted_class: class_idx as i32,
                confidence,
                label: std::ptr::null_mut(),
                probabilities: std::ptr::null_mut(),
                num_classes: 0,
            },
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
pub unsafe extern "C" fn classify_pii_text(text: *const c_char) -> ClassificationResult {
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
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_jailbreak_text(text: *const c_char) -> ClassificationResult {
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

    if let Some(classifier) = LORA_JAILBREAK_CLASSIFIER.get() {
        let classifier = classifier.clone();
        match classifier.classify_with_index(text) {
            Ok((class_idx, confidence, ref label)) => {
                return ClassificationResult {
                    predicted_class: class_idx as i32,
                    confidence,
                    label: unsafe { allocate_c_string(label) },
                };
            }
            Err(e) => {
                eprintln!(
                    "LoRA jailbreak classifier error: {}, falling back to Traditional BERT",
                    e
                );
            }
        }
    }

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
pub unsafe extern "C" fn classify_unified_batch(
    texts_ptr: *const *const c_char,
    num_texts: i32,
) -> UnifiedBatchResult {
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

    use crate::model_architectures::TaskType;
    let tasks = vec![TaskType::Intent, TaskType::PII, TaskType::Security];

    let text_refs: Vec<&str> = _texts.iter().map(|s| s.as_ref()).collect();
    match classifier.classify_intelligent(&text_refs, &tasks) {
        Ok(result) => unsafe { convert_unified_result_to_batch(&result, _texts.len()) },
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

/// Allocate Intent results array for batch
unsafe fn allocate_intent_results(
    intent: Option<&crate::classifiers::unified::UnifiedTaskResult>,
    batch_size: usize,
) -> *mut crate::ffi::types::IntentResult {
    use crate::ffi::types::IntentResult;

    let (category, confidence) = intent
        .map(|i| (i.category_name.as_str(), i.confidence))
        .unwrap_or(("unknown", 0.0));

    let mut results = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        results.push(IntentResult {
            category: allocate_c_string(category),
            confidence,
            probabilities: std::ptr::null_mut(),
            num_probabilities: 0,
        });
    }
    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut IntentResult
}

/// Allocate PII results array for batch
unsafe fn allocate_pii_results(
    pii: Option<&crate::classifiers::unified::UnifiedTaskResult>,
    batch_size: usize,
) -> *mut crate::ffi::types::PIIResult {
    use crate::ffi::types::PIIResult;

    let mut results = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let (has_pii, confidence) = match pii {
            Some(p) => {
                let has_pii = p.category_name.to_lowercase().contains("pii")
                    || p.category_name == PII_POSITIVE_CLASS_STR
                    || p.predicted_class == PII_POSITIVE_CLASS;
                (has_pii, p.confidence)
            }
            None => (false, 0.0),
        };
        results.push(PIIResult {
            has_pii,
            pii_types: std::ptr::null_mut(),
            num_pii_types: 0,
            confidence,
        });
    }
    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut PIIResult
}

/// Allocate Security results array for batch
unsafe fn allocate_security_results(
    security: Option<&crate::classifiers::unified::UnifiedTaskResult>,
    batch_size: usize,
) -> *mut crate::ffi::types::SecurityResult {
    use crate::ffi::types::SecurityResult;

    let mut results = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        let (is_jailbreak, threat_type, confidence) = match security {
            Some(s) => {
                let category_lower = s.category_name.to_lowercase();
                let is_jailbreak = SECURITY_THREAT_KEYWORDS
                    .iter()
                    .any(|&kw| category_lower.contains(kw))
                    || s.category_name == SECURITY_THREAT_CLASS_STR
                    || s.predicted_class == SECURITY_THREAT_CLASS;
                let threat_type = if is_jailbreak {
                    s.category_name.as_str()
                } else {
                    "none"
                };
                (is_jailbreak, threat_type, s.confidence)
            }
            None => (false, "none", 0.0),
        };
        results.push(SecurityResult {
            is_jailbreak,
            threat_type: allocate_c_string(threat_type),
            confidence,
        });
    }
    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut SecurityResult
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
    use crate::model_architectures::TaskType;

    let intent_results =
        allocate_intent_results(result.task_results.get(&TaskType::Intent), batch_size);
    let pii_results = allocate_pii_results(result.task_results.get(&TaskType::PII), batch_size);
    let security_results =
        allocate_security_results(result.task_results.get(&TaskType::Security), batch_size);

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
pub unsafe extern "C" fn classify_bert_pii_tokens(text: *const c_char) -> BertTokenClassificationResult {
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
                let token_entities: Vec<(String, String, f32)> = token_results
                    .iter()
                    .map(|(token, class_idx, score)| {
                        (token.clone(), format!("label_{}", class_idx), *score)
                    })
                    .collect();

                BertTokenClassificationResult {
                    entities: unsafe { allocate_bert_token_entity_array(&token_entities) },
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
/// - `text` and `config_path` must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn classify_candle_bert_tokens_with_labels(
    text: *const c_char,
    config_path: *const c_char,
) -> BertTokenClassificationResult {
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

    if let Some(classifier) = crate::ffi::init::LORA_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        if let Ok(token_results) = classifier.classify_tokens(text) {
            let token_entities: Vec<(String, String, f32)> = token_results
                .iter()
                .filter(|r| r.label_name != "O" && r.label_id != 0)
                .map(|r| (r.token.clone(), r.label_name.clone(), r.confidence))
                .collect();

            return BertTokenClassificationResult {
                entities: unsafe { allocate_bert_token_entity_array(&token_entities) },
                num_entities: token_entities.len() as i32,
            };
        }
    }

    if let Some(classifier) = TRADITIONAL_BERT_TOKEN_CLASSIFIER.get() {
        let classifier = classifier.clone();
        if let Ok(token_results) = classifier.classify_tokens(text) {
            let token_entities: Vec<(String, String, f32)> = token_results
                .iter()
                .map(|(token, class_idx, score)| {
                    (token.clone(), format!("label_{}", class_idx), *score)
                })
                .collect();

            return BertTokenClassificationResult {
                entities: unsafe { allocate_bert_token_entity_array(&token_entities) },
                num_entities: token_results.len() as i32,
            };
        }
    }

    BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
    }
}

/// Try LoRA token classifier for classify_candle_bert_tokens.
/// Returns None if LoRA classifier not initialized; Some(result) otherwise.
fn try_lora_bert_tokens(text: &str) -> Option<BertTokenClassificationResult> {
    let classifier = crate::ffi::init::LORA_TOKEN_CLASSIFIER.get()?;
    match classifier.clone().classify_tokens(text) {
        Ok(lora_results) => {
            let token_entities: Vec<(String, String, f32)> = lora_results
                .iter()
                .filter(|r| r.label_name != "O" && r.label_id != 0)
                .map(|r| (r.token.clone(), r.label_name.clone(), r.confidence))
                .collect();

            Some(BertTokenClassificationResult {
                entities: unsafe { allocate_bert_token_entity_array(&token_entities) },
                num_entities: token_entities.len() as i32,
            })
        }
        Err(_e) => Some(BertTokenClassificationResult {
            entities: std::ptr::null_mut(),
            num_entities: 0,
        }),
    }
}

/// Try traditional BERT token classifier for classify_candle_bert_tokens.
/// Returns None if classifier not initialized; Some(result) otherwise.
fn try_traditional_bert_tokens(text: &str) -> Option<BertTokenClassificationResult> {
    let classifier = TRADITIONAL_BERT_TOKEN_CLASSIFIER.get()?;
    match classifier.clone().classify_tokens(text) {
        Ok(token_results) => {
            let token_entities: Vec<(String, String, f32)> = token_results
                .iter()
                .map(|(token, class_idx, confidence)| {
                    (token.clone(), format!("class_{}", class_idx), *confidence)
                })
                .collect();

            Some(BertTokenClassificationResult {
                entities: unsafe { allocate_bert_token_entity_array(&token_entities) },
                num_entities: token_results.len() as i32,
            })
        }
        Err(e) => {
            println!("Candle BERT token classification failed: {}", e);
            Some(BertTokenClassificationResult {
                entities: std::ptr::null_mut(),
                num_entities: 0,
            })
        }
    }
}

/// Try ModernBERT token classifier for classify_candle_bert_tokens.
/// Returns None if classifier not initialized; Some(result) otherwise.
fn try_modernbert_tokens(text: &str) -> Option<BertTokenClassificationResult> {
    let classifier = TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER.get()?;
    match classifier.clone().classify_tokens(text) {
        Ok(token_results) => {
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

            Some(BertTokenClassificationResult {
                entities: unsafe {
                    allocate_modernbert_token_entity_array(&token_entities) as *mut BertTokenEntity
                },
                num_entities: token_entities.len() as i32,
            })
        }
        Err(e) => {
            println!("ModernBERT token classification failed: {}", e);
            Some(BertTokenClassificationResult {
                entities: std::ptr::null_mut(),
                num_entities: 0,
            })
        }
    }
}

/// Classify Candle BERT tokens
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_candle_bert_tokens(
    text: *const c_char,
) -> BertTokenClassificationResult {
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

    if let Some(result) = try_lora_bert_tokens(text) {
        return result;
    }
    if let Some(result) = try_traditional_bert_tokens(text) {
        return result;
    }
    if let Some(result) = try_modernbert_tokens(text) {
        return result;
    }

    println!("No token classifier initialized (Traditional BERT, ModernBERT, or LoRA) - call init function first");
    BertTokenClassificationResult {
        entities: std::ptr::null_mut(),
        num_entities: 0,
    }
}
