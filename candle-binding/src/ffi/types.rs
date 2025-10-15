//! FFI Type Definitions

use std::ffi::c_char;

/// Basic classification result structure
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub confidence: f32,
    pub predicted_class: i32,
    pub label: *mut c_char,
}

/// Classification result with probabilities
#[repr(C)]
#[derive(Debug)]
pub struct ClassificationResultWithProbs {
    pub confidence: f32,
    pub predicted_class: i32,
    pub label: *mut c_char,
    pub probabilities: *mut f32,
    pub num_classes: i32,
}

/// Embedding result structure (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct EmbeddingResult {
    pub data: *mut f32,
    pub length: i32,
    pub error: bool,
}

/// Tokenization result structure (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct TokenizationResult {
    pub token_ids: *mut i32,
    pub token_count: i32,
    pub tokens: *mut *mut c_char,
    pub error: bool,
}

/// Similarity result for single comparison
#[repr(C)]
#[derive(Debug)]
pub struct SimilarityResult {
    pub index: i32,
    pub similarity: f32,
    pub text: *mut c_char,
}

/// Multiple similarity results
#[repr(C)]
#[derive(Debug)]
pub struct SimilarityResults {
    pub results: *mut SimilarityResult,
    pub length: i32,
    pub success: bool,
}

/// ModernBERT classification result
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ModernBertClassificationResult {
    pub predicted_class: i32,
    pub confidence: f32,
}

/// ModernBERT classification result with probabilities
#[repr(C)]
#[derive(Debug)]
pub struct ModernBertClassificationResultWithProbs {
    pub class: i32,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_classes: i32,
}

/// ModernBERT token entity (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct ModernBertTokenEntity {
    pub entity_type: *mut c_char,
    pub start: i32,
    pub end: i32,
    pub text: *mut c_char,
    pub confidence: f32,
}

/// ModernBERT token classification result (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct ModernBertTokenClassificationResult {
    pub entities: *mut ModernBertTokenEntity,
    pub num_entities: i32,
}

/// Legacy ModernBERT token classification result (for backward compatibility)
#[repr(C)]
#[derive(Debug)]
pub struct LegacyModernBertTokenClassificationResult {
    pub tokens: *mut *mut c_char,
    pub labels: *mut *mut c_char,
    pub scores: *mut f32,
    pub num_tokens: i32,
    pub success: bool,
}

/// BERT token entity structure
#[repr(C)]
#[derive(Debug)]
pub struct BertTokenEntity {
    pub entity_type: *mut c_char,
    pub start: i32,
    pub end: i32,
    pub text: *mut c_char,
    pub confidence: f32,
}

/// BERT token classification result (must match Go's C struct definition)
#[repr(C)]
#[derive(Debug)]
pub struct BertTokenClassificationResult {
    pub entities: *mut BertTokenEntity,
    pub num_entities: i32,
}

/// Candle BERT token result
#[repr(C)]
#[derive(Debug)]
pub struct CandleBertTokenResult {
    pub tokens: *mut *mut c_char,
    pub labels: *mut *mut c_char,
    pub label_ids: *mut i32,
    pub scores: *mut f32,
    pub num_tokens: i32,
    pub success: bool,
}

/// Batch classification result
#[repr(C)]
#[derive(Debug)]
pub struct BatchClassificationResult {
    pub results: *mut ClassificationResult,
    pub length: i32,
    pub success: bool,
}

/// Unified batch processing result (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct UnifiedBatchResult {
    pub intent_results: *mut IntentResult,
    pub pii_results: *mut PIIResult,
    pub security_results: *mut SecurityResult,
    pub batch_size: i32,
    pub error: bool,
    pub error_message: *mut c_char,
}

/// Intent classification result (matches Go CIntentResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct IntentResult {
    pub category: *mut c_char,
    pub confidence: f32,
    pub probabilities: *mut f32,
    pub num_probabilities: i32,
}

/// PII detection result (matches Go CPIIResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PIIResult {
    pub has_pii: bool,
    pub pii_types: *mut *mut c_char,
    pub num_pii_types: i32,
    pub confidence: f32,
}

/// Security/Jailbreak detection result (matches Go CSecurityResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct SecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: *mut c_char,
    pub confidence: f32,
}

/// Enhanced classification result with metadata
#[repr(C)]
#[derive(Debug)]
pub struct EnhancedClassificationResult {
    pub confidence: f32,
    pub predicted_class: i32,
    pub processing_time_ms: f32,
    pub model_version: *mut c_char,
}

/// Multi-language classification result
#[repr(C)]
#[derive(Debug)]
pub struct MultiLangResult {
    pub confidence: f32,
    pub predicted_class: i32,
    pub detected_language: *mut c_char,
    pub language_confidence: f32,
}

/// Performance metrics structure
#[repr(C)]
#[derive(Debug)]
pub struct PerformanceMetrics {
    pub inference_time_ms: f32,
    pub memory_usage_mb: f32,
    pub throughput_qps: f32,
    pub model_load_time_ms: f32,
}

/// LoRA batch processing result (matches Go C struct)
#[repr(C)]
#[derive(Debug)]
pub struct LoRABatchResult {
    pub intent_results: *mut LoRAIntentResult,
    pub pii_results: *mut LoRAPIIResult,
    pub security_results: *mut LoRASecurityResult,
    pub batch_size: i32,
    pub avg_confidence: f32,
}

/// LoRA intent classification result (matches Go LoRAIntentResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LoRAIntentResult {
    pub category: *mut c_char,
    pub confidence: f32,
}

/// LoRA PII detection result (matches Go LoRAPIIResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LoRAPIIResult {
    pub has_pii: bool,
    pub pii_types: *mut *mut c_char,
    pub num_pii_types: i32,
    pub confidence: f32,
}

/// LoRA security/jailbreak detection result (matches Go LoRASecurityResult)
#[repr(C)]
#[derive(Debug, Clone)]
pub struct LoRASecurityResult {
    pub is_jailbreak: bool,
    pub threat_type: *mut c_char,
    pub confidence: f32,
}

impl Default for ClassificationResult {
    fn default() -> Self {
        Self {
            confidence: 0.0,
            predicted_class: -1,
            label: std::ptr::null_mut(),
        }
    }
}

impl Default for EmbeddingResult {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            length: 0,
            error: true,
        }
    }
}

impl Default for TokenizationResult {
    fn default() -> Self {
        Self {
            token_ids: std::ptr::null_mut(),
            token_count: 0,
            tokens: std::ptr::null_mut(),
            error: true,
        }
    }
}

impl Default for LoRABatchResult {
    fn default() -> Self {
        Self {
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            batch_size: 0,
            avg_confidence: 0.0,
        }
    }
}

impl Default for UnifiedBatchResult {
    fn default() -> Self {
        Self {
            intent_results: std::ptr::null_mut(),
            pii_results: std::ptr::null_mut(),
            security_results: std::ptr::null_mut(),
            batch_size: 0,
            error: false,
            error_message: std::ptr::null_mut(),
        }
    }
}

/// Validate that a C structure pointer is not null and properly aligned
pub unsafe fn validate_c_struct_ptr<T>(ptr: *const T) -> bool {
    !ptr.is_null() && (ptr as usize) % std::mem::align_of::<T>() == 0
}

/// Get the size of any C structure for ABI compatibility checking
pub fn get_struct_size<T>() -> usize {
    std::mem::size_of::<T>()
}

/// Get the alignment of any C structure for ABI compatibility checking
pub fn get_struct_align<T>() -> usize {
    std::mem::align_of::<T>()
}
