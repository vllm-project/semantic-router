//! FFI Initialization Functions
//!
//! This module contains core C FFI initialization functions for dual-path architecture.
//! Additional init functions are in init_mmbert.rs and init_classifiers.rs.

use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::core::similarity::BertSimilarity;
use crate::BertClassifier;

// Global state using OnceLock for zero-cost reads after initialization
// OnceLock<Arc<T>> pattern provides:
// - Zero lock overhead on reads (atomic load only)
// - Concurrent access via Arc cloning
// - Thread-safe initialization guarantee
// - No dependency on lazy_static
pub static BERT_SIMILARITY: OnceLock<Arc<BertSimilarity>> = OnceLock::new();
static BERT_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_PII_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
static BERT_JAILBREAK_CLASSIFIER: OnceLock<Arc<BertClassifier>> = OnceLock::new();
// Feedback detector classifier (exported for use in classify.rs)
pub static FEEDBACK_DETECTOR_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
// DeBERTa v3 jailbreak/prompt injection classifier (exported for use in classify.rs)
pub static DEBERTA_JAILBREAK_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::deberta_v3::DebertaV3Classifier>,
> = OnceLock::new();
// Unified classifier for dual-path architecture (exported for use in classify.rs)
pub static UNIFIED_CLASSIFIER: OnceLock<
    Arc<crate::classifiers::unified::DualPathUnifiedClassifier>,
> = OnceLock::new();
// Parallel LoRA engine for high-performance classification (primary path for LoRA models)
// Already wrapped in Arc for cheap cloning and concurrent access
pub static PARALLEL_LORA_ENGINE: OnceLock<
    Arc<crate::classifiers::lora::parallel_engine::ParallelLoRAEngine>,
> = OnceLock::new();
// LoRA token classifier for token-level classification
pub static LORA_TOKEN_CLASSIFIER: OnceLock<
    Arc<crate::classifiers::lora::token_lora::LoRATokenClassifier>,
> = OnceLock::new();
// LoRA intent classifier for sequence classification
pub static LORA_INTENT_CLASSIFIER: OnceLock<
    Arc<crate::classifiers::lora::intent_lora::IntentLoRAClassifier>,
> = OnceLock::new();
// Hallucination detector (ModernBERT token classifier for RAG verification)
pub static HALLUCINATION_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier>,
> = OnceLock::new();
// ModernBERT NLI classifier for hallucination explanation (NLI post-processing)
// Model: tasksource/ModernBERT-base-nli
pub static NLI_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
// LoRA jailbreak classifier for security threat detection
pub static LORA_JAILBREAK_CLASSIFIER: OnceLock<
    Arc<crate::classifiers::lora::security_lora::SecurityLoRAClassifier>,
> = OnceLock::new();

/// Model type detection for intelligent routing
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ModelType {
    LoRA,
    Traditional,
}

/// Detect model type based on actual model weights and structure
///
/// This function implements intelligent routing by checking:
/// 1. Actual LoRA weights in model.safetensors (unmerged LoRA)
/// 2. lora_config.json existence (merged LoRA models)
/// 3. Model path naming patterns (contains "lora")
/// 4. Fallback to traditional model
pub(crate) fn detect_model_type(model_path: &str) -> ModelType {
    let path = Path::new(model_path);

    // Check 1: Look for actual LoRA weights in model file (unmerged LoRA)
    let weights_path = path.join("model.safetensors");
    if weights_path.exists() {
        if let Ok(has_lora_weights) = check_for_lora_weights(&weights_path) {
            if has_lora_weights {
                return ModelType::LoRA;
            }
        }
    }

    // Check 2: Look for lora_config.json (merged LoRA models)
    // Merged LoRA models should still route to LoRA path for high-performance implementation
    let lora_config_path = path.join("lora_config.json");
    if lora_config_path.exists() {
        return ModelType::LoRA;
    }

    // Default to traditional model
    ModelType::Traditional
}

/// Load labels from model config.json file
pub(crate) fn load_labels_from_model_config(
    model_path: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Use unified config loader (replaces local implementation)
    use crate::core::config_loader;

    match config_loader::load_labels_from_model_config(model_path) {
        Ok(result) => Ok(result),
        Err(unified_err) => Err(Box::new(unified_err)),
    }
}

/// Check if model file contains actual LoRA weights
fn check_for_lora_weights(weights_path: &Path) -> Result<bool, Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Read;

    // Configuration for LoRA weight detection
    const BUFFER_SIZE: usize = 8192; // 8KB should be sufficient for safetensors headers
    const LORA_WEIGHT_PATTERNS: &[&str] = &[
        "lora_A",
        "lora_B",
        "lora_up",
        "lora_down",
        "adapter",
        "delta_weight",
        "scaling",
    ];

    // Read a portion of the safetensors file to check for LoRA weight names
    let mut file = File::open(weights_path)?;
    let mut buffer = vec![0u8; BUFFER_SIZE];
    file.read_exact(&mut buffer)?;

    // Convert to string and check for LoRA weight patterns
    let content = String::from_utf8_lossy(&buffer);

    // Check for any LoRA weight pattern
    for pattern in LORA_WEIGHT_PATTERNS {
        if content.contains(pattern) {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Initialize similarity model
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
#[no_mangle]
pub unsafe extern "C" fn init_similarity_model(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match BertSimilarity::new(model_id, use_cpu) {
        Ok(model) => {
            // Set using OnceLock - returns false if already initialized (safe to re-call)
            BERT_SIMILARITY.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT: {e}");
            false
        }
    }
}

/// Check if BERT similarity model is initialized
///
/// This function checks the Rust-side OnceLock state to determine if the model
/// has been initialized. This is the source of truth for initialization status.
///
/// # Returns
/// `true` if BERT_SIMILARITY OnceLock contains an initialized model, `false` otherwise
///
/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn is_similarity_model_initialized() -> bool {
    BERT_SIMILARITY.get().is_some()
}

/// Initialize traditional BERT classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
#[no_mangle]
pub unsafe extern "C" fn init_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2, got {num_classes}");
        return false;
    }

    match BertClassifier::new(model_id, num_classes as usize, use_cpu) {
        Ok(classifier) => BERT_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
        Err(e) => {
            eprintln!("Failed to initialize BERT classifier: {e}");
            false
        }
    }
}

/// Initialize PII classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_pii_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Ensure num_classes is valid
    if num_classes < 2 {
        eprintln!("Number of classes must be at least 2, got {num_classes}");
        return false;
    }

    match BertClassifier::new(model_id, num_classes as usize, use_cpu) {
        Ok(classifier) => BERT_PII_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
        Err(e) => {
            eprintln!("Failed to initialize BERT PII classifier: {e}");
            false
        }
    }
}

/// Initialize jailbreak classifier with LoRA auto-detection
///
/// Intelligent model type detection (same pattern as intent classifier):
/// 1. Checks for lora_config.json → Routes to LoRA jailbreak classifier
/// 2. Falls back to Traditional BERT if LoRA config not found
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_jailbreak_classifier(
    model_id: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Intelligent model type detection (same as intent classifier)
    let model_type = detect_model_type(model_path);

    match model_type {
        ModelType::LoRA => {
            // Check if already initialized
            if LORA_JAILBREAK_CLASSIFIER.get().is_some() {
                return true; // Already initialized, return success
            }

            // Route to LoRA jailbreak classifier (SecurityLoRAClassifier)
            match crate::classifiers::lora::security_lora::SecurityLoRAClassifier::new(
                model_path, use_cpu,
            ) {
                Ok(classifier) => LORA_JAILBREAK_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
                Err(e) => {
                    eprintln!(
                        "  ERROR: Failed to initialize LoRA jailbreak classifier: {}",
                        e
                    );
                    false
                }
            }
        }
        ModelType::Traditional => {
            eprintln!("🔍 Detected Traditional BERT model for jailbreak classification");

            // Ensure num_classes is valid
            if num_classes < 2 {
                eprintln!("Number of classes must be at least 2, got {num_classes}");
                return false;
            }

            // Initialize Traditional BERT jailbreak classifier
            match BertClassifier::new(model_path, num_classes as usize, use_cpu) {
                Ok(classifier) => BERT_JAILBREAK_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
                Err(e) => {
                    eprintln!("Failed to initialize BERT jailbreak classifier: {e}");
                    false
                }
            }
        }
    }
}

/// Initialize ModernBERT classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_modernbert_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Try to initialize the actual ModernBERT model using traditional architecture
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => {
            crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_CLASSIFIER
                .set(Arc::new(model))
                .is_ok()
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT classifier: {}", e);
            false
        }
    }
}

/// Initialize ModernBERT PII classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_modernbert_pii_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Try to initialize the actual ModernBERT PII model
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => {
            crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_PII_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT PII classifier: {}", e);
            false
        }
    }
}

/// Initialize ModernBERT PII token classifier
///
/// # Safety
/// - All pointer parameters must be valid null-terminated C strings
#[no_mangle]
pub unsafe extern "C" fn init_modernbert_pii_token_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    // Migrated from modernbert.rs:868-890
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Create the token classifier
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier::new(model_id, use_cpu) {
        Ok(classifier) => {
            crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER.set(Arc::new(classifier)).is_ok()
        }
        Err(e) => {
            println!("  ERROR: Failed to initialize ModernBERT PII token classifier: {}", e);
            false
        }
    }
}

/// Initialize ModernBERT jailbreak classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_modernbert_jailbreak_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Try to initialize the actual ModernBERT jailbreak model
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => {
            crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT jailbreak classifier: {}", e);
            false
        }
    }
}
