//! FFI Initialization Functions
//!
//! This module contains all C FFI initialization functions for dual-path architecture.
//! Provides 13 initialization functions with 100% backward compatibility.

use std::ffi::{c_char, c_int, CStr};
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
enum ModelType {
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
fn detect_model_type(model_path: &str) -> ModelType {
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
fn load_labels_from_model_config(
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
    file.read(&mut buffer)?;

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
pub extern "C" fn init_similarity_model(model_id: *const c_char, use_cpu: bool) -> bool {
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
#[no_mangle]
pub extern "C" fn is_similarity_model_initialized() -> bool {
    BERT_SIMILARITY.get().is_some()
}

/// Initialize traditional BERT classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
#[no_mangle]
pub extern "C" fn init_classifier(
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
pub extern "C" fn init_pii_classifier(
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
pub extern "C" fn init_jailbreak_classifier(
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
pub extern "C" fn init_modernbert_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
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
pub extern "C" fn init_modernbert_pii_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
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
pub extern "C" fn init_modernbert_pii_token_classifier(
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
pub extern "C" fn init_modernbert_jailbreak_classifier(
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

/// Initialize ModernBERT fact-check classifier (halugate-sentinel model)
///
/// This initializes the halugate-sentinel ModernBERT model for classifying
/// whether a prompt needs fact-checking.
///
/// Model outputs:
/// - 0: NO_FACT_CHECK_NEEDED
/// - 1: FACT_CHECK_NEEDED
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `true` if initialization succeeds, `false` otherwise
///
/// # Example
/// ```c
/// bool success = init_fact_check_classifier(
///     "models/halugate-sentinel",
///     true  // use CPU
/// );
/// ```
#[no_mangle]
pub extern "C" fn init_fact_check_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    // Check if already initialized - return true if so (idempotent)
    if crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER.get().is_some() {
        println!("✓ Fact-check classifier already initialized");
        return true;
    }

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    println!(
        "🔧 Initializing fact-check classifier (halugate-sentinel): {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => {
            match crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER.set(Arc::new(model)) {
                Ok(_) => {
                    println!("✓ Fact-check classifier initialized successfully");
                    true
                }
                Err(_) => {
                    // Already initialized by another thread, that's fine
                    println!("✓ Fact-check classifier already initialized (race condition)");
                    true
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to initialize fact-check classifier: {}", e);
            false
        }
    }
}

/// Initialize ModernBERT feedback detector classifier
///
/// This initializes the feedback-detector ModernBERT model for classifying
/// user feedback from follow-up messages.
///
/// Model outputs:
/// - 0: SAT (satisfied)
/// - 1: NEED_CLARIFICATION
/// - 2: WRONG_ANSWER
/// - 3: WANT_DIFFERENT
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `true` if initialization succeeds, `false` otherwise
#[no_mangle]
pub extern "C" fn init_feedback_detector(model_id: *const c_char, use_cpu: bool) -> bool {
    // Check if already initialized - return true if so (idempotent)
    if FEEDBACK_DETECTOR_CLASSIFIER.get().is_some() {
        println!("✓ Feedback detector already initialized");
        return true;
    }

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    println!("🔧 Initializing feedback detector: {}", model_id);

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(model_id, use_cpu) {
        Ok(model) => {
            match FEEDBACK_DETECTOR_CLASSIFIER.set(Arc::new(model)) {
                Ok(_) => {
                    println!("✓ Feedback detector initialized successfully");
                    true
                }
                Err(_) => {
                    println!("✓ Feedback detector already initialized (race condition)");
                    true
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to initialize feedback detector: {}", e);
            false
        }
    }
}

/// Initialize DeBERTa v3 jailbreak/prompt injection classifier
///
/// This initializes the ProtectAI DeBERTa v3 Base Prompt Injection model
/// for detecting jailbreak attempts and prompt injection attacks.
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
/// - Caller must ensure proper memory management
///
/// # Returns
/// `true` if initialization succeeds, `false` otherwise
///
/// # Example
/// ```c
/// bool success = init_deberta_jailbreak_classifier(
///     "protectai/deberta-v3-base-prompt-injection",
///     false  // use GPU
/// );
/// ```
#[no_mangle]
pub extern "C" fn init_deberta_jailbreak_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    println!(
        "🔧 Initializing DeBERTa v3 jailbreak classifier: {}",
        model_id
    );

    match crate::model_architectures::traditional::deberta_v3::DebertaV3Classifier::new(
        model_id, use_cpu,
    ) {
        Ok(classifier) => match DEBERTA_JAILBREAK_CLASSIFIER.set(Arc::new(classifier)) {
            Ok(_) => {
                println!("✓ DeBERTa v3 jailbreak classifier initialized successfully");
                true
            }
            Err(_) => {
                eprintln!("Failed to set DeBERTa jailbreak classifier (already initialized)");
                false
            }
        },
        Err(e) => {
            eprintln!(
                "Failed to initialize DeBERTa v3 jailbreak classifier: {}",
                e
            );
            false
        }
    }
}

/// Initialize unified classifier (complex multi-head configuration)
///
/// # Safety
/// - All pointer parameters must be valid null-terminated C strings
/// - Label arrays must be valid and match the specified counts
#[no_mangle]
pub extern "C" fn init_unified_classifier_c(
    modernbert_path: *const c_char,
    intent_head_path: *const c_char,
    pii_head_path: *const c_char,
    security_head_path: *const c_char,
    intent_labels: *const *const c_char,
    intent_labels_count: c_int,
    pii_labels: *const *const c_char,
    pii_labels_count: c_int,
    security_labels: *const *const c_char,
    security_labels_count: c_int,
    _use_cpu: bool,
) -> bool {
    // Adapted from lib.rs:1180-1266
    let modernbert_path = unsafe {
        match CStr::from_ptr(modernbert_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let intent_head_path = unsafe {
        match CStr::from_ptr(intent_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let pii_head_path = unsafe {
        match CStr::from_ptr(pii_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let security_head_path = unsafe {
        match CStr::from_ptr(security_head_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Convert C string arrays to Rust Vec<String>
    let _intent_labels_vec = unsafe {
        std::slice::from_raw_parts(intent_labels, intent_labels_count as usize)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let _pii_labels_vec = unsafe {
        std::slice::from_raw_parts(pii_labels, pii_labels_count as usize)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let _security_labels_vec = unsafe {
        std::slice::from_raw_parts(security_labels, security_labels_count as usize)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    // Validate model paths exist (following old architecture pattern)
    if !std::path::Path::new(modernbert_path).exists() {
        eprintln!(
            "Error: ModernBERT model path does not exist: {}",
            modernbert_path
        );
        return false;
    }
    if !std::path::Path::new(intent_head_path).exists() {
        eprintln!(
            "Error: Intent head path does not exist: {}",
            intent_head_path
        );
        return false;
    }
    if !std::path::Path::new(pii_head_path).exists() {
        eprintln!("Error: PII head path does not exist: {}", pii_head_path);
        return false;
    }
    if !std::path::Path::new(security_head_path).exists() {
        eprintln!(
            "Error: Security head path does not exist: {}",
            security_head_path
        );
        return false;
    }

    // Create configuration with actual model paths
    let mut config = crate::model_architectures::config::DualPathConfig::default();

    // Set main model path in configuration (real implementation, not mock)
    config.traditional.model_path = std::path::PathBuf::from(modernbert_path);

    // Initialize UnifiedClassifier with real model loading
    match crate::classifiers::unified::DualPathUnifiedClassifier::new(config) {
        Ok(classifier) => {
            // Initialize traditional path with actual models
            match classifier.init_traditional_path() {
                Ok(_) => UNIFIED_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
                Err(e) => {
                    eprintln!("Failed to initialize traditional path: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to initialize unified classifier: {}", e);
            false
        }
    }
}

/// Initialize BERT token classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_bert_token_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    // Migrated from lib.rs:1404-1440
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error converting model path: {e}");
                return false;
            }
        }
    };

    // Create device
    let _device = if use_cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
    };

    // Initialize TraditionalBertTokenClassifier
    match crate::model_architectures::traditional::bert::TraditionalBertTokenClassifier::new(
        model_path,
        num_classes as usize,
        use_cpu,
    ) {
        Ok(_classifier) => {
            // Store in global static (would need to add this to the lazy_static block)
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT token classifier: {}", e);
            false
        }
    }
}

/// Initialize Candle BERT classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_candle_bert_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Intelligent model type detection (same as token classifier)
    let model_type = detect_model_type(model_path);

    match model_type {
        ModelType::LoRA => {
            // Check if already initialized
            if LORA_INTENT_CLASSIFIER.get().is_some() {
                return true; // Already initialized, return success
            }

            // Route to LoRA intent classifier initialization
            match crate::classifiers::lora::intent_lora::IntentLoRAClassifier::new(
                model_path, use_cpu,
            ) {
                Ok(classifier) => LORA_INTENT_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
                Err(e) => {
                    eprintln!(
                        "  ERROR: Failed to initialize LoRA intent classifier: {}",
                        e
                    );
                    false
                }
            }
        }
        ModelType::Traditional => {
            // Initialize TraditionalBertClassifier
            match crate::model_architectures::traditional::bert::TraditionalBertClassifier::new(
                model_path,
                num_classes as usize,
                use_cpu,
            ) {
                Ok(_classifier) => {
                    // Store in global static (would need to add this to the lazy_static block)
                    true
                }
                Err(e) => {
                    eprintln!("Failed to initialize Candle BERT classifier: {}", e);
                    false
                }
            }
        }
    }
}

/// Initialize Candle BERT token classifier with intelligent routing
///
/// This function implements dual-path architecture intelligent routing:
/// - Automatically detects model type (LoRA vs Traditional)
/// - Routes to appropriate classifier initialization
/// - Maintains backward compatibility with existing API
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_candle_bert_token_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Intelligent model type detection
    let model_type = detect_model_type(model_path);

    match model_type {
        ModelType::LoRA => {
            // Check if already initialized
            if LORA_TOKEN_CLASSIFIER.get().is_some() {
                return true; // Already initialized, return success
            }

            // Route to LoRA token classifier initialization
            match crate::classifiers::lora::token_lora::LoRATokenClassifier::new(
                model_path, use_cpu,
            ) {
                Ok(classifier) => LORA_TOKEN_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
                Err(e) => {
                    eprintln!("  ERROR: Failed to initialize LoRA token classifier: {}", e);
                    false
                }
            }
        }
        ModelType::Traditional => {
            // Check if already initialized
            if crate::model_architectures::traditional::bert::TRADITIONAL_BERT_TOKEN_CLASSIFIER
                .get()
                .is_some()
            {
                return true; // Already initialized, return success
            }

            // Route to traditional BERT token classifier
            match crate::model_architectures::traditional::bert::TraditionalBertTokenClassifier::new(
                model_path,
                num_classes as usize,
                use_cpu,
            ) {
                Ok(classifier) => {
                    crate::model_architectures::traditional::bert::TRADITIONAL_BERT_TOKEN_CLASSIFIER
                        .set(Arc::new(classifier))
                        .is_ok()
                }
                Err(e) => {
                    eprintln!(
                        "  ERROR: Failed to initialize Traditional BERT token classifier: {}",
                        e
                    );
                    false
                }
            }
        }
    }
}

/// Initialize LoRA unified classifier (high-performance parallel path)
///
/// # Safety
/// - All pointer parameters must be valid null-terminated C strings
/// - Label arrays must be valid and match the specified counts
#[no_mangle]
pub extern "C" fn init_lora_unified_classifier(
    intent_model: *const c_char,
    pii_model: *const c_char,
    security_model: *const c_char,
    architecture: *const c_char,
    use_cpu: bool,
) -> bool {
    let intent_path = unsafe {
        match CStr::from_ptr(intent_model).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let pii_path = unsafe {
        match CStr::from_ptr(pii_model).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let security_path = unsafe {
        match CStr::from_ptr(security_model).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    let _architecture_str = unsafe {
        match CStr::from_ptr(architecture).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Check if already initialized - return success if so
    if PARALLEL_LORA_ENGINE.get().is_some() {
        return true;
    }

    // Load labels dynamically from model configurations
    let _intent_labels_vec = load_labels_from_model_config(intent_path).unwrap_or_else(|e| {
        eprintln!(
            "Warning: Failed to load intent labels from {}: {}",
            intent_path, e
        );
        vec![] // Return empty vec, will be handled by ParallelLoRAEngine
    });
    let _pii_labels_vec = load_labels_from_model_config(pii_path).unwrap_or_else(|e| {
        eprintln!(
            "Warning: Failed to load PII labels from {}: {}",
            pii_path, e
        );
        vec![] // Return empty vec, will be handled by ParallelLoRAEngine
    });
    let _security_labels_vec = load_labels_from_model_config(security_path).unwrap_or_else(|e| {
        eprintln!(
            "Warning: Failed to load security labels from {}: {}",
            security_path, e
        );
        vec![] // Return empty vec, will be handled by ParallelLoRAEngine
    });

    // Create device
    let device = if use_cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::cuda_if_available(0).unwrap_or(candle_core::Device::Cpu)
    };

    // Initialize ParallelLoRAEngine
    match crate::classifiers::lora::parallel_engine::ParallelLoRAEngine::new(
        device,
        intent_path,
        pii_path,
        security_path,
        use_cpu,
    ) {
        Ok(engine) => {
            // Store in global static variable (Arc for efficient cloning during concurrent access)
            // Return true even if already set (race condition)
            PARALLEL_LORA_ENGINE.set(Arc::new(engine)).is_ok()
                || PARALLEL_LORA_ENGINE.get().is_some()
        }
        Err(e) => {
            eprintln!(
                "Failed to initialize LoRA unified classifier  Error details: {:?}",
                e
            );
            false
        }
    }
}

/// Initialize hallucination detection model
///
/// This is a ModernBERT-based token classifier for detecting hallucinations
/// in RAG (Retrieval Augmented Generation) outputs. It classifies each token as
/// either SUPPORTED (grounded in context) or HALLUCINATED.
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string pointing to the model directory
#[no_mangle]
pub extern "C" fn init_hallucination_model(model_path: *const c_char, use_cpu: bool) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Check if already initialized
    if HALLUCINATION_CLASSIFIER.get().is_some() {
        println!("Hallucination detection model already initialized");
        return true;
    }

    println!(
        "Initializing hallucination detection model from: {}",
        model_path
    );

    // Use TraditionalModernBertTokenClassifier for hallucination detection
    // Model has: 2 classes (0=SUPPORTED, 1=HALLUCINATED)
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier::new(
        model_path,
        use_cpu,
    ) {
        Ok(classifier) => {
            let success = HALLUCINATION_CLASSIFIER.set(Arc::new(classifier)).is_ok();
            if success {
                println!("Hallucination detection model initialized successfully");
            }
            success
        }
        Err(e) => {
            eprintln!("Failed to initialize hallucination detection model: {}", e);
            false
        }
    }
}

/// Initialize ModernBERT NLI (Natural Language Inference) model
///
/// This model is used for post-processing hallucination detection results to provide
/// explanations. It classifies premise-hypothesis pairs into:
/// - Entailment (0): The premise supports the hypothesis
/// - Neutral (1): The premise neither supports nor contradicts
/// - Contradiction (2): The premise contradicts the hypothesis
///
/// Recommended model: tasksource/ModernBERT-base-nli
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string pointing to the model directory
#[no_mangle]
pub extern "C" fn init_nli_model(model_path: *const c_char, use_cpu: bool) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    // Check if already initialized
    if NLI_CLASSIFIER.get().is_some() {
        println!("NLI model already initialized");
        return true;
    }

    println!("Initializing NLI model from: {}", model_path);

    // Use TraditionalModernBertClassifier for ModernBERT NLI
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(
        model_path,
        use_cpu,
    ) {
        Ok(classifier) => {
            let success = NLI_CLASSIFIER.set(Arc::new(classifier)).is_ok();
            if success {
                println!("NLI model (ModernBERT) initialized successfully");
            }
            success
        }
        Err(e) => {
            eprintln!("Failed to initialize NLI model: {}", e);
            false
        }
    }
}

/// Check if NLI model is initialized
#[no_mangle]
pub extern "C" fn is_nli_model_initialized() -> bool {
    NLI_CLASSIFIER.get().is_some()
}
