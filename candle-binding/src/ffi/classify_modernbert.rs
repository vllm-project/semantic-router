//! ModernBERT, DeBERTa, fact-check, feedback, and token classification FFI functions.

use crate::ffi::classify::load_id2label_from_config;
use crate::ffi::init::{DEBERTA_JAILBREAK_CLASSIFIER, FEEDBACK_DETECTOR_CLASSIFIER};
use crate::ffi::memory::{allocate_c_float_array, allocate_modernbert_token_entity_array};
use crate::ffi::types::{
    ClassificationResult, ModernBertClassificationResult, ModernBertClassificationResultWithProbs,
    ModernBertTokenClassificationResult,
};
use crate::model_architectures::traditional::modernbert::{
    TRADITIONAL_MODERNBERT_CLASSIFIER, TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER,
    TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER, TRADITIONAL_MODERNBERT_PII_CLASSIFIER,
    TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER,
};
use std::ffi::{c_char, CStr};

/// Classify ModernBERT text
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_modernbert_text(text: *const c_char) -> ModernBertClassificationResult {
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
    if let Some(classifier) = TRADITIONAL_MODERNBERT_CLASSIFIER.get() {
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
pub unsafe extern "C" fn classify_modernbert_text_with_probabilities(
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
                if class_id < num_classes {
                    probabilities[class_id] = confidence;
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
pub unsafe extern "C" fn classify_modernbert_pii_text(
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
pub unsafe extern "C" fn classify_modernbert_jailbreak_text(
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
#[no_mangle]
pub unsafe extern "C" fn classify_deberta_jailbreak_text(text: *const c_char) -> ClassificationResult {
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
#[no_mangle]
pub unsafe extern "C" fn classify_fact_check_text(text: *const c_char) -> ModernBertClassificationResult {
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
#[no_mangle]
pub unsafe extern "C" fn classify_feedback_text(text: *const c_char) -> ModernBertClassificationResult {
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
pub unsafe extern "C" fn classify_modernbert_pii_tokens(
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
