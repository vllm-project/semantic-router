//! mmBERT-32K classification FFI functions (32K context, YaRN RoPE scaling).
//!
//! Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn

use crate::ffi::init_mmbert::{
    MMBERT_32K_FACTCHECK_CLASSIFIER, MMBERT_32K_FEEDBACK_CLASSIFIER, MMBERT_32K_INTENT_CLASSIFIER,
    MMBERT_32K_JAILBREAK_CLASSIFIER, MMBERT_32K_MODALITY_CLASSIFIER, MMBERT_32K_PII_CLASSIFIER,
};
use crate::ffi::types::{
    ModernBertClassificationResult, ModernBertTokenClassificationResult, ModernBertTokenEntity,
};
use std::ffi::{c_char, CStr, CString};

/// Classify text using mmBERT-32K intent classifier
///
/// Classifies text into MMLU-Pro academic categories for intelligent request routing.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_intent(
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
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_factcheck(
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
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_jailbreak(
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
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_feedback(
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
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_pii_tokens(
    text: *const c_char,
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

                // Always use LABEL_{class_id} format — Go side translates via PIIMapping (pii_type_mapping.json)
                for (i, (_token, class_id, confidence, start, end)) in
                    entities.into_iter().enumerate()
                {
                    let entity_type = format!("LABEL_{}", class_id);

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

/// Classify text using mmBERT-32K modality routing classifier
///
/// Determines the appropriate response modality for a user prompt:
/// - AR (0): Text-only response via autoregressive LLM
/// - DIFFUSION (1): Image generation via diffusion model
/// - BOTH (2): Hybrid response requiring both text and image
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn classify_mmbert_32k_modality(
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

    if let Some(classifier) = MMBERT_32K_MODALITY_CLASSIFIER.get() {
        match classifier.classify_text(text) {
            Ok((class_id, confidence)) => ModernBertClassificationResult {
                predicted_class: class_id as i32,
                confidence,
            },
            Err(e) => {
                eprintln!("mmBERT-32K modality classification failed: {}", e);
                default_result
            }
        }
    } else {
        eprintln!("mmBERT-32K modality classifier not initialized");
        default_result
    }
}
