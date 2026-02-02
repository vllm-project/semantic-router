//! Unified Classifier FFI Module (stub implementations)
//!
//! These are stub implementations for compatibility with candle-binding.
//! The unified classifier (LoRA-based) is not yet implemented in onnx-binding.

use std::ffi::{c_char, CString};
use std::ptr;

// ============================================================================
// FFI Types for Unified Classifier
// ============================================================================

/// Intent classification result
#[repr(C)]
pub struct CIntentResult {
    pub category: *mut c_char,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_probabilities: i32,
}

/// PII detection result
#[repr(C)]
pub struct CPIIResult {
    pub has_pii: bool,
    pub pii_types: *mut *mut c_char,
    pub num_pii_types: i32,
    pub confidence: f32,
}

/// Security/jailbreak result
#[repr(C)]
pub struct CSecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: *mut c_char,
    pub confidence: f32,
}

/// Combined classification result
#[repr(C)]
pub struct CUnifiedResult {
    pub intent: CIntentResult,
    pub pii: CPIIResult,
    pub security: CSecurityResult,
    pub processing_time_ms: f32,
    pub error: bool,
    pub error_message: *mut c_char,
}

/// Batch classification result
#[repr(C)]
pub struct CUnifiedBatchResult {
    pub results: *mut CUnifiedResult,
    pub num_results: i32,
    pub total_processing_time_ms: f32,
    pub error: bool,
    pub error_message: *mut c_char,
}

/// LoRA batch result
#[repr(C)]
pub struct CLoRABatchResult {
    pub results: *mut CUnifiedResult,
    pub num_results: i32,
    pub total_processing_time_ms: f32,
    pub error: bool,
    pub error_message: *mut c_char,
}

// ============================================================================
// Stub Implementations
// ============================================================================

fn create_error_message(msg: &str) -> *mut c_char {
    CString::new(msg).unwrap().into_raw()
}

fn create_default_intent() -> CIntentResult {
    CIntentResult {
        category: ptr::null_mut(),
        confidence: 0.0,
        probabilities: ptr::null_mut(),
        num_probabilities: 0,
    }
}

fn create_default_pii() -> CPIIResult {
    CPIIResult {
        has_pii: false,
        pii_types: ptr::null_mut(),
        num_pii_types: 0,
        confidence: 0.0,
    }
}

fn create_default_security() -> CSecurityResult {
    CSecurityResult {
        is_jailbreak: false,
        threat_type: ptr::null_mut(),
        confidence: 0.0,
    }
}

/// Initialize unified classifier (stub - not implemented)
#[no_mangle]
pub extern "C" fn init_unified_classifier_c(
    _intent_model_path: *const c_char,
    _pii_model_path: *const c_char,
    _security_model_path: *const c_char,
    _architecture: *const c_char,
    _use_cpu: bool,
) -> bool {
    eprintln!("WARNING: init_unified_classifier_c is not implemented in onnx-binding");
    false
}

/// Initialize LoRA unified classifier (stub - not implemented)
#[no_mangle]
pub extern "C" fn init_lora_unified_classifier(
    _intent_model_path: *const c_char,
    _pii_model_path: *const c_char,
    _security_model_path: *const c_char,
    _architecture: *const c_char,
    _use_cpu: bool,
) -> bool {
    eprintln!("WARNING: init_lora_unified_classifier is not implemented in onnx-binding");
    false
}

/// Classify batch with unified classifier (stub)
#[no_mangle]
pub extern "C" fn classify_unified_batch(
    _texts: *const *const c_char,
    _num_texts: i32,
) -> CUnifiedBatchResult {
    CUnifiedBatchResult {
        results: ptr::null_mut(),
        num_results: 0,
        total_processing_time_ms: 0.0,
        error: true,
        error_message: create_error_message("unified classifier not implemented in onnx-binding"),
    }
}

/// Classify batch with LoRA (stub)
#[no_mangle]
pub extern "C" fn classify_batch_with_lora(
    _texts: *const *const c_char,
    _num_texts: i32,
) -> CLoRABatchResult {
    CLoRABatchResult {
        results: ptr::null_mut(),
        num_results: 0,
        total_processing_time_ms: 0.0,
        error: true,
        error_message: create_error_message("LoRA classifier not implemented in onnx-binding"),
    }
}

/// Free unified batch result
#[no_mangle]
pub extern "C" fn free_unified_batch_result(result: *mut CUnifiedBatchResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let r = &mut *result;

        if !r.error_message.is_null() {
            let _ = CString::from_raw(r.error_message);
            r.error_message = ptr::null_mut();
        }

        if !r.results.is_null() && r.num_results > 0 {
            let results = std::slice::from_raw_parts_mut(r.results, r.num_results as usize);
            for res in results.iter_mut() {
                free_unified_result_inner(res);
            }
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.results, r.num_results as usize));
            r.results = ptr::null_mut();
        }
    }
}

/// Free LoRA batch result
#[no_mangle]
pub extern "C" fn free_lora_batch_result(result: *mut CLoRABatchResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let r = &mut *result;

        if !r.error_message.is_null() {
            let _ = CString::from_raw(r.error_message);
            r.error_message = ptr::null_mut();
        }

        if !r.results.is_null() && r.num_results > 0 {
            let results = std::slice::from_raw_parts_mut(r.results, r.num_results as usize);
            for res in results.iter_mut() {
                free_unified_result_inner(res);
            }
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(r.results, r.num_results as usize));
            r.results = ptr::null_mut();
        }
    }
}

/// Helper to free a unified result's contents
unsafe fn free_unified_result_inner(result: &mut CUnifiedResult) {
    // Free intent
    if !result.intent.category.is_null() {
        let _ = CString::from_raw(result.intent.category);
        result.intent.category = ptr::null_mut();
    }
    if !result.intent.probabilities.is_null() && result.intent.num_probabilities > 0 {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(
            result.intent.probabilities,
            result.intent.num_probabilities as usize,
        ));
        result.intent.probabilities = ptr::null_mut();
    }

    // Free PII
    if !result.pii.pii_types.is_null() && result.pii.num_pii_types > 0 {
        let types = std::slice::from_raw_parts_mut(result.pii.pii_types, result.pii.num_pii_types as usize);
        for t in types.iter_mut() {
            if !t.is_null() {
                let _ = CString::from_raw(*t);
            }
        }
        let _ = Box::from_raw(types);
        result.pii.pii_types = ptr::null_mut();
    }

    // Free security
    if !result.security.threat_type.is_null() {
        let _ = CString::from_raw(result.security.threat_type);
        result.security.threat_type = ptr::null_mut();
    }

    // Free error message
    if !result.error_message.is_null() {
        let _ = CString::from_raw(result.error_message);
        result.error_message = ptr::null_mut();
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_create_error_message() {
        let msg = create_error_message("test error");
        assert!(!msg.is_null());

        // Clean up
        unsafe {
            let _ = CString::from_raw(msg);
        }
    }

    #[test]
    fn test_create_default_intent() {
        let intent = create_default_intent();
        assert!(intent.category.is_null());
        assert_eq!(intent.confidence, 0.0);
        assert!(intent.probabilities.is_null());
        assert_eq!(intent.num_probabilities, 0);
    }

    #[test]
    fn test_create_default_pii() {
        let pii = create_default_pii();
        assert!(!pii.has_pii);
        assert!(pii.pii_types.is_null());
        assert_eq!(pii.num_pii_types, 0);
        assert_eq!(pii.confidence, 0.0);
    }

    #[test]
    fn test_create_default_security() {
        let security = create_default_security();
        assert!(!security.is_jailbreak);
        assert!(security.threat_type.is_null());
        assert_eq!(security.confidence, 0.0);
    }

    #[test]
    fn test_init_unified_classifier_returns_false() {
        // Should return false since models don't exist
        let result = init_unified_classifier_c(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            true,
        );
        assert!(!result);
    }

    #[test]
    fn test_init_lora_unified_classifier_returns_false() {
        let result = init_lora_unified_classifier(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            ptr::null(),
            true,
        );
        assert!(!result);
    }

    #[test]
    fn test_classify_unified_batch_returns_error() {
        let result = classify_unified_batch(ptr::null(), 0);
        assert!(result.error);
        assert!(result.results.is_null());
        assert_eq!(result.num_results, 0);

        // Clean up
        if !result.error_message.is_null() {
            unsafe {
                let _ = CString::from_raw(result.error_message);
            }
        }
    }

    #[test]
    fn test_classify_batch_with_lora_returns_error() {
        let result = classify_batch_with_lora(ptr::null(), 0);
        assert!(result.error);
        assert!(result.results.is_null());
        assert_eq!(result.num_results, 0);

        // Clean up
        if !result.error_message.is_null() {
            unsafe {
                let _ = CString::from_raw(result.error_message);
            }
        }
    }

    #[test]
    fn test_free_unified_batch_result_null_safe() {
        // Should not panic when called with null
        free_unified_batch_result(ptr::null_mut());
    }

    #[test]
    fn test_free_lora_batch_result_null_safe() {
        // Should not panic when called with null
        free_lora_batch_result(ptr::null_mut());
    }

    #[test]
    fn test_struct_sizes() {
        // Ensure FFI structs have non-zero size
        assert!(std::mem::size_of::<CIntentResult>() > 0);
        assert!(std::mem::size_of::<CPIIResult>() > 0);
        assert!(std::mem::size_of::<CSecurityResult>() > 0);
        assert!(std::mem::size_of::<CUnifiedResult>() > 0);
        assert!(std::mem::size_of::<CUnifiedBatchResult>() > 0);
        assert!(std::mem::size_of::<CLoRABatchResult>() > 0);
    }
}
