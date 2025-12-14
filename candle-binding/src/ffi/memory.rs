//! FFI Memory Management Functions
//!
//! This module contains all C FFI memory management functions for dual-path architecture.
//! Provides 9 memory management functions with 100% backward compatibility.

use crate::ffi::types::*;
use std::ffi::{c_char, CString};

/// Free tokenization result
///
/// # Safety
/// - `result` must be a valid TokenizationResult structure
#[no_mangle]
pub extern "C" fn free_tokenization_result(result: TokenizationResult) {
    // Free the token_ids array
    unsafe {
        if !result.token_ids.is_null() && result.token_count > 0 {
            let _token_ids_vec = Vec::from_raw_parts(
                result.token_ids,
                result.token_count as usize,
                result.token_count as usize,
            );
        }

        // Free the tokens string array
        if !result.tokens.is_null() && result.token_count > 0 {
            let tokens_slice =
                std::slice::from_raw_parts_mut(result.tokens, result.token_count as usize);
            for token_ptr in tokens_slice {
                if !token_ptr.is_null() {
                    let _ = CString::from_raw(*token_ptr);
                }
            }
            let _tokens_vec = Vec::from_raw_parts(
                result.tokens,
                result.token_count as usize,
                result.token_count as usize,
            );
        }
    }
}

/// Free C string
///
/// # Safety
/// - `s` must be a valid pointer allocated by this library
#[no_mangle]
pub extern "C" fn free_cstring(s: *mut c_char) {
    // Migrated from lib.rs:746-752
    unsafe {
        if !s.is_null() {
            let _ = CString::from_raw(s);
        }
    }
}

/// Free embedding data
///
/// # Safety
/// - `data` must be a valid pointer allocated by this library
/// - `length` must match the original allocation size
#[no_mangle]
pub extern "C" fn free_embedding(data: *mut f32, length: i32) {
    // Migrated from lib.rs:756-763
    if !data.is_null() && length > 0 {
        unsafe {
            // Reconstruct the vector so that Rust can properly deallocate it
            let _vec = Vec::from_raw_parts(data, length as usize, length as usize);
            // The vector will be dropped and the memory freed when _vec goes out of scope
        }
    }
}

/// Free probabilities array
///
/// # Safety
/// - `probabilities` must be a valid pointer allocated by this library
/// - `num_classes` must match the original allocation size
#[no_mangle]
pub extern "C" fn free_probabilities(probabilities: *mut f32, num_classes: i32) {
    // Migrated from lib.rs:966-978
    if !probabilities.is_null() && num_classes > 0 {
        unsafe {
            let _: Box<[f32]> = Box::from_raw(std::slice::from_raw_parts_mut(
                probabilities,
                num_classes as usize,
            ));
        }
    }
}

/// Free unified batch result
///
/// # Safety
/// - `result` must be a valid UnifiedBatchResult structure
#[no_mangle]
pub extern "C" fn free_unified_batch_result(result: UnifiedBatchResult) {
    // Adapted from lib.rs:1309-1360 (simplified for current structure)
    if result.batch_size <= 0 {
        return;
    }

    let batch_size = result.batch_size as usize;

    // Free intent results
    if !result.intent_results.is_null() {
        unsafe {
            let intent_slice = std::slice::from_raw_parts_mut(result.intent_results, batch_size);
            for intent in intent_slice {
                if !intent.category.is_null() {
                    let _ = CString::from_raw(intent.category);
                }
            }
            let _ = Vec::from_raw_parts(result.intent_results, batch_size, batch_size);
        }
    }

    // Free PII results
    if !result.pii_results.is_null() {
        unsafe {
            let pii_slice = std::slice::from_raw_parts_mut(result.pii_results, batch_size);
            for pii in pii_slice {
                // Free PII types array if present
                if !pii.pii_types.is_null() && pii.num_pii_types > 0 {
                    let types_slice =
                        std::slice::from_raw_parts_mut(pii.pii_types, pii.num_pii_types as usize);
                    for type_ptr in types_slice {
                        if !type_ptr.is_null() {
                            let _ = CString::from_raw(*type_ptr);
                        }
                    }
                    let _ = Vec::from_raw_parts(
                        pii.pii_types,
                        pii.num_pii_types as usize,
                        pii.num_pii_types as usize,
                    );
                }
            }
            let _ = Vec::from_raw_parts(result.pii_results, batch_size, batch_size);
        }
    }

    // Free security results
    if !result.security_results.is_null() {
        unsafe {
            let security_slice =
                std::slice::from_raw_parts_mut(result.security_results, batch_size);
            for security in security_slice {
                if !security.threat_type.is_null() {
                    let _ = CString::from_raw(security.threat_type);
                }
            }
            let _ = Vec::from_raw_parts(result.security_results, batch_size, batch_size);
        }
    }
}

/// Free BERT token classification result
///
/// # Safety
/// - `result` must be a valid BertTokenClassificationResult structure
#[no_mangle]
pub extern "C" fn free_bert_token_classification_result(result: BertTokenClassificationResult) {
    if result.num_entities > 0 && !result.entities.is_null() {
        unsafe {
            // Free BertTokenEntity array
            let entities_slice =
                std::slice::from_raw_parts_mut(result.entities, result.num_entities as usize);
            for entity in entities_slice {
                // Free entity_type string
                if !entity.entity_type.is_null() {
                    let _ = CString::from_raw(entity.entity_type);
                }
                // Free text string
                if !entity.text.is_null() {
                    let _ = CString::from_raw(entity.text);
                }
            }
            // Free the entities array itself
            let _ = Vec::from_raw_parts(
                result.entities,
                result.num_entities as usize,
                result.num_entities as usize,
            );
        }
    }
}

/// Free LoRA batch result
///
/// # Safety
/// - `result` must be a valid LoRABatchResult structure
#[no_mangle]
pub extern "C" fn free_lora_batch_result(result: LoRABatchResult) {
    // Migrated from lib.rs:2072-2170
    if result.batch_size <= 0 {
        return;
    }

    // Free intent results
    if !result.intent_results.is_null() {
        let intent_slice = unsafe {
            std::slice::from_raw_parts_mut(result.intent_results, result.batch_size as usize)
        };
        for intent in intent_slice {
            if !intent.category.is_null() {
                unsafe {
                    let _ = CString::from_raw(intent.category);
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.intent_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }

    // Free PII results
    if !result.pii_results.is_null() {
        let pii_slice = unsafe {
            std::slice::from_raw_parts_mut(result.pii_results, result.batch_size as usize)
        };
        for pii in pii_slice {
            if !pii.pii_types.is_null() && pii.num_pii_types > 0 {
                let pii_types_slice = unsafe {
                    std::slice::from_raw_parts_mut(pii.pii_types, pii.num_pii_types as usize)
                };
                for pii_type in pii_types_slice {
                    if !pii_type.is_null() {
                        unsafe {
                            let _ = CString::from_raw(*pii_type);
                        }
                    }
                }
                unsafe {
                    let _ = Vec::from_raw_parts(
                        pii.pii_types,
                        pii.num_pii_types as usize,
                        pii.num_pii_types as usize,
                    );
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.pii_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }

    // Free security results
    if !result.security_results.is_null() {
        let security_slice = unsafe {
            std::slice::from_raw_parts_mut(result.security_results, result.batch_size as usize)
        };
        for security in security_slice {
            if !security.threat_type.is_null() {
                unsafe {
                    let _ = CString::from_raw(security.threat_type);
                }
            }
        }
        unsafe {
            let _ = Vec::from_raw_parts(
                result.security_results,
                result.batch_size as usize,
                result.batch_size as usize,
            );
        }
    }
}

/// Free ModernBERT probabilities array
///
/// # Safety
/// - `probabilities` must be a valid pointer allocated by this library
/// - `num_classes` must match the original allocation size
#[no_mangle]
pub extern "C" fn free_modernbert_probabilities(probabilities: *mut f32, num_classes: i32) {
    // Migrated from modernbert.rs:1006-1015
    if !probabilities.is_null() && num_classes > 0 {
        unsafe {
            let _: Box<[f32]> = Box::from_raw(std::slice::from_raw_parts_mut(
                probabilities,
                num_classes as usize,
            ));
        }
    }
}

/// Free ModernBERT token result
///
/// # Safety
/// - `result` must be a valid ModernBertTokenClassificationResult structure
#[no_mangle]
pub extern "C" fn free_modernbert_token_result(result: ModernBertTokenClassificationResult) {
    // Free the entities array
    if result.num_entities > 0 {
        unsafe {
            if !result.entities.is_null() {
                // Convert back to Vec and let it drop
                let entities_slice =
                    std::slice::from_raw_parts_mut(result.entities, result.num_entities as usize);

                // Free each entity's strings
                for entity in entities_slice {
                    if !entity.entity_type.is_null() {
                        let _ = CString::from_raw(entity.entity_type);
                    }
                    if !entity.text.is_null() {
                        let _ = CString::from_raw(entity.text);
                    }
                }

                // Free the entities array itself
                let _ = Vec::from_raw_parts(
                    result.entities,
                    result.num_entities as usize,
                    result.num_entities as usize,
                );
            }
        }
    }
}

// ========== Helper functions for common memory allocation patterns ==========

/// Allocate and populate C string from Rust string
///
/// # Safety
/// - Returns a pointer that must be freed with free_cstring
pub unsafe fn allocate_c_string(s: &str) -> *mut c_char {
    match CString::new(s) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Allocate and populate C string array from Rust string vector
///
/// # Safety
/// - Returns a pointer that must be freed with free_c_string_array
pub unsafe fn allocate_c_string_array(strings: &[String]) -> *mut *mut c_char {
    if strings.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_strings: Vec<*mut c_char> = Vec::with_capacity(strings.len());
    for s in strings {
        c_strings.push(allocate_c_string(s));
    }

    let ptr = c_strings.as_mut_ptr();
    std::mem::forget(c_strings);
    ptr
}

/// Allocate and populate C int array from Rust usize vector
///
/// # Safety
/// - Returns a pointer that must be freed with free_int_array
pub unsafe fn allocate_c_int_array(values: &[usize]) -> *mut i32 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_ints: Vec<i32> = Vec::with_capacity(values.len());
    for &v in values {
        c_ints.push(v as i32);
    }

    let ptr = c_ints.as_mut_ptr();
    std::mem::forget(c_ints);
    ptr
}

/// Allocate and populate C float array from Rust f32 vector
///
/// # Safety
/// - Returns a pointer that must be freed with free_float_array
pub unsafe fn allocate_c_float_array(values: &[f32]) -> *mut f32 {
    if values.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_floats: Vec<f32> = Vec::with_capacity(values.len());
    c_floats.extend_from_slice(values);

    let ptr = c_floats.as_mut_ptr();
    std::mem::forget(c_floats);
    ptr
}

/// Free C string array
///
/// # Safety
/// - `array` must be allocated by allocate_c_string_array
/// - `length` must match the original array size
#[no_mangle]
pub extern "C" fn free_c_string_array(array: *mut *mut c_char, length: i32) {
    if !array.is_null() && length > 0 {
        unsafe {
            let strings_slice = std::slice::from_raw_parts_mut(array, length as usize);
            for string_ptr in strings_slice {
                if !string_ptr.is_null() {
                    let _ = CString::from_raw(*string_ptr);
                }
            }
            let _ = Vec::from_raw_parts(array, length as usize, length as usize);
        }
    }
}

/// Free C int array
///
/// # Safety
/// - `array` must be allocated by allocate_c_int_array
/// - `length` must match the original array size
#[no_mangle]
pub extern "C" fn free_c_int_array(array: *mut i32, length: i32) {
    if !array.is_null() && length > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(array, length as usize, length as usize);
        }
    }
}

/// Free C float array
///
/// # Safety
/// - `array` must be allocated by allocate_c_float_array
/// - `length` must match the original array size
#[no_mangle]
pub extern "C" fn free_c_float_array(array: *mut f32, length: i32) {
    if !array.is_null() && length > 0 {
        unsafe {
            let _ = Vec::from_raw_parts(array, length as usize, length as usize);
        }
    }
}

/// Free hallucination detection result
///
/// # Safety
/// - `result` must be obtained from detect_hallucinations
#[no_mangle]
pub extern "C" fn free_hallucination_detection_result(result: HallucinationDetectionResult) {
    unsafe {
        // Free error message if present
        if !result.error_message.is_null() {
            let _ = CString::from_raw(result.error_message);
        }

        // Free spans array
        if result.num_spans > 0 && !result.spans.is_null() {
            let spans_slice =
                std::slice::from_raw_parts_mut(result.spans, result.num_spans as usize);

            // Free each span's strings
            for span in spans_slice {
                if !span.text.is_null() {
                    let _ = CString::from_raw(span.text);
                }
                if !span.label.is_null() {
                    let _ = CString::from_raw(span.label);
                }
            }

            // Free the spans array itself using the layout-aware deallocation
            let layout =
                std::alloc::Layout::array::<HallucinationSpan>(result.num_spans as usize).unwrap();
            std::alloc::dealloc(result.spans as *mut u8, layout);
        }
    }
}

/// Free NLI result
///
/// # Safety
/// - `result` must be obtained from classify_nli
#[no_mangle]
pub extern "C" fn free_nli_result(result: NLIResult) {
    unsafe {
        // Free error message if present
        if !result.error_message.is_null() {
            let _ = CString::from_raw(result.error_message);
        }
    }
}

/// Free enhanced hallucination detection result
///
/// # Safety
/// - `result` must be obtained from detect_hallucinations_with_nli
#[no_mangle]
pub extern "C" fn free_enhanced_hallucination_detection_result(
    result: EnhancedHallucinationDetectionResult,
) {
    unsafe {
        // Free error message if present
        if !result.error_message.is_null() {
            let _ = CString::from_raw(result.error_message);
        }

        // Free spans array
        if result.num_spans > 0 && !result.spans.is_null() {
            let spans_slice =
                std::slice::from_raw_parts_mut(result.spans, result.num_spans as usize);

            // Free each span's strings
            for span in spans_slice {
                if !span.text.is_null() {
                    let _ = CString::from_raw(span.text);
                }
                if !span.explanation.is_null() {
                    let _ = CString::from_raw(span.explanation);
                }
            }

            // Free the spans array itself using the layout-aware deallocation
            let layout =
                std::alloc::Layout::array::<EnhancedHallucinationSpan>(result.num_spans as usize)
                    .unwrap();
            std::alloc::dealloc(result.spans as *mut u8, layout);
        }
    }
}

/// Convert IntentResult to LoRAIntentResult and allocate
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn convert_intent_to_lora_intent(
    intent: &crate::classifiers::lora::intent_lora::IntentResult,
) -> crate::ffi::types::LoRAIntentResult {
    // Create probabilities array
    let _probabilities = vec![intent.confidence, 1.0 - intent.confidence];

    crate::ffi::types::LoRAIntentResult {
        category: allocate_c_string(&intent.intent),
        confidence: intent.confidence,
    }
}

/// Convert PIIResult to LoRAPIIResult and allocate
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn convert_pii_to_lora_pii(
    pii: &crate::classifiers::lora::pii_lora::PIIResult,
) -> crate::ffi::types::LoRAPIIResult {
    crate::ffi::types::LoRAPIIResult {
        has_pii: pii.has_pii,
        pii_types: allocate_c_string_array(&pii.pii_types),
        num_pii_types: pii.pii_types.len() as i32,
        confidence: pii.confidence,
    }
}

/// Convert SecurityResult to LoRASecurityResult and allocate
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn convert_security_to_lora_security(
    security: &crate::classifiers::lora::security_lora::SecurityResult,
) -> crate::ffi::types::LoRASecurityResult {
    let threat_type = if security.threat_types.is_empty() {
        "none".to_string()
    } else {
        security.threat_types[0].clone()
    };

    crate::ffi::types::LoRASecurityResult {
        is_jailbreak: security.is_threat,
        threat_type: allocate_c_string(&threat_type),
        confidence: security.confidence,
    }
}

/// Allocate C array of LoRAIntentResult
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_lora_intent_array(
    results: &[crate::classifiers::lora::intent_lora::IntentResult],
) -> *mut crate::ffi::types::LoRAIntentResult {
    if results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_results = Vec::with_capacity(results.len());
    for result in results {
        c_results.push(convert_intent_to_lora_intent(result));
    }

    let boxed = c_results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::LoRAIntentResult
}

/// Allocate C array of LoRAPIIResult
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_lora_pii_array(
    results: &[crate::classifiers::lora::pii_lora::PIIResult],
) -> *mut crate::ffi::types::LoRAPIIResult {
    if results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_results = Vec::with_capacity(results.len());
    for result in results {
        c_results.push(convert_pii_to_lora_pii(result));
    }

    let boxed = c_results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::LoRAPIIResult
}

/// Allocate C array of LoRASecurityResult
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_lora_security_array(
    results: &[crate::classifiers::lora::security_lora::SecurityResult],
) -> *mut crate::ffi::types::LoRASecurityResult {
    if results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut c_results = Vec::with_capacity(results.len());
    for result in results {
        c_results.push(convert_security_to_lora_security(result));
    }

    let boxed = c_results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::LoRASecurityResult
}

/// Allocate C array of BertTokenEntity
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_bert_token_entity_array(
    token_results: &[(String, String, f32)],
) -> *mut crate::ffi::types::BertTokenEntity {
    if token_results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut entities = Vec::with_capacity(token_results.len());
    for (i, (token, label, confidence)) in token_results.iter().enumerate() {
        entities.push(crate::ffi::types::BertTokenEntity {
            entity_type: allocate_c_string(label),
            start: i as i32 * token.len() as i32, // Simplified position calculation
            end: (i + 1) as i32 * token.len() as i32,
            text: allocate_c_string(token),
            confidence: *confidence,
        });
    }

    let boxed = entities.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::BertTokenEntity
}

/// Allocate C array of ModernBertTokenEntity
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_modernbert_token_entity_array(
    token_results: &[(String, String, f32, usize, usize)],
) -> *mut crate::ffi::types::ModernBertTokenEntity {
    if token_results.is_empty() {
        return std::ptr::null_mut();
    }

    let mut entities = Vec::with_capacity(token_results.len());
    for (token, label, score, start, end) in token_results.iter() {
        entities.push(crate::ffi::types::ModernBertTokenEntity {
            entity_type: allocate_c_string(label),
            start: *start as i32, // Real start position
            end: *end as i32,     // Real end position
            text: allocate_c_string(token),
            confidence: *score,
        });
    }

    let boxed = entities.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::ModernBertTokenEntity
}

/// Allocate C array of IntentResult (traditional)
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_intent_result_array(count: usize) -> *mut crate::ffi::types::IntentResult {
    if count == 0 {
        return std::ptr::null_mut();
    }

    let mut results = Vec::with_capacity(count);
    for i in 0..count {
        let probabilities = vec![0.8f32, 0.2f32]; // Default probabilities
        results.push(crate::ffi::types::IntentResult {
            category: allocate_c_string(&format!("intent_{}", i)),
            confidence: 0.8 + (i as f32 * 0.01),
            probabilities: allocate_c_float_array(&probabilities),
            num_probabilities: probabilities.len() as i32,
        });
    }

    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::IntentResult
}

/// Allocate C array of PIIResult (traditional)
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_pii_result_array(count: usize) -> *mut crate::ffi::types::PIIResult {
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate empty PII results - real results are populated by LoRA classifiers
    let mut results = Vec::with_capacity(count);
    for _i in 0..count {
        results.push(crate::ffi::types::PIIResult {
            has_pii: false,
            pii_types: std::ptr::null_mut(),
            confidence: 0.0,
            num_pii_types: 0,
        });
    }

    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::PIIResult
}

/// Allocate C array of SecurityResult (traditional)
///
/// # Safety
/// - Returns a pointer that must be freed appropriately
pub unsafe fn allocate_security_result_array(
    count: usize,
) -> *mut crate::ffi::types::SecurityResult {
    if count == 0 {
        return std::ptr::null_mut();
    }

    // Allocate empty security results - real results are populated by LoRA classifiers
    let mut results = Vec::with_capacity(count);
    for _i in 0..count {
        results.push(crate::ffi::types::SecurityResult {
            is_jailbreak: false,
            threat_type: allocate_c_string("none"),
            confidence: 0.0,
        });
    }

    let boxed = results.into_boxed_slice();
    Box::into_raw(boxed) as *mut crate::ffi::types::SecurityResult
}
