//! FFI Initialization Functions for Specialized Classifiers
//!
//! Contains initialization functions for fact-check, feedback, DeBERTa, unified,
//! traditional BERT, LoRA, hallucination, and NLI classifiers.

use std::ffi::{c_char, c_int, CStr};
use std::sync::Arc;

use super::init::{
    detect_model_type, load_labels_from_model_config, ModelType, DEBERTA_JAILBREAK_CLASSIFIER,
    FEEDBACK_DETECTOR_CLASSIFIER, HALLUCINATION_CLASSIFIER, LORA_INTENT_CLASSIFIER,
    LORA_TOKEN_CLASSIFIER, NLI_CLASSIFIER, PARALLEL_LORA_ENGINE, UNIFIED_CLASSIFIER,
};

/// Parse a C string pointer into a Rust `&str`, returning `None` on null/invalid UTF-8.
///
/// # Safety
/// Caller must ensure `ptr` is a valid, null-terminated C string.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    CStr::from_ptr(ptr).to_str().ok()
}

/// Validate that a filesystem path exists, printing an error if not.
fn validate_path_exists(path: &str, label: &str) -> bool {
    if std::path::Path::new(path).exists() {
        return true;
    }
    eprintln!("Error: {} path does not exist: {}", label, path);
    false
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
pub unsafe extern "C" fn init_fact_check_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    // Check if already initialized - return true if so (idempotent)
    if crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_FACT_CHECK_CLASSIFIER.get().is_some() {
        println!("Fact-check classifier already initialized");
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
                    println!("Fact-check classifier initialized successfully");
                    true
                }
                Err(_) => {
                    // Already initialized by another thread, that's fine
                    println!("Fact-check classifier already initialized (race condition)");
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
pub unsafe extern "C" fn init_feedback_detector(model_id: *const c_char, use_cpu: bool) -> bool {
    // Check if already initialized - return true if so (idempotent)
    if FEEDBACK_DETECTOR_CLASSIFIER.get().is_some() {
        println!("Feedback detector already initialized");
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
                    println!("Feedback detector initialized successfully");
                    true
                }
                Err(_) => {
                    println!("Feedback detector already initialized (race condition)");
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
pub unsafe extern "C" fn init_deberta_jailbreak_classifier(
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
                println!("DeBERTa v3 jailbreak classifier initialized successfully");
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
pub unsafe extern "C" fn init_unified_classifier_c(
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
    let modernbert_path = unsafe {
        match cstr_to_str(modernbert_path) {
            Some(s) => s,
            None => return false,
        }
    };
    let intent_head_path = unsafe {
        match cstr_to_str(intent_head_path) {
            Some(s) => s,
            None => return false,
        }
    };
    let pii_head_path = unsafe {
        match cstr_to_str(pii_head_path) {
            Some(s) => s,
            None => return false,
        }
    };
    let security_head_path = unsafe {
        match cstr_to_str(security_head_path) {
            Some(s) => s,
            None => return false,
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

    if !validate_path_exists(modernbert_path, "ModernBERT model")
        || !validate_path_exists(intent_head_path, "Intent head")
        || !validate_path_exists(pii_head_path, "PII head")
        || !validate_path_exists(security_head_path, "Security head")
    {
        return false;
    }

    let mut config = crate::model_architectures::config::DualPathConfig::default();
    config.traditional.model_path = std::path::PathBuf::from(modernbert_path);

    match crate::classifiers::unified::DualPathUnifiedClassifier::new(config) {
        Ok(classifier) => match classifier.init_traditional_path() {
            Ok(_) => UNIFIED_CLASSIFIER.set(Arc::new(classifier)).is_ok(),
            Err(e) => {
                eprintln!("Failed to initialize traditional path: {}", e);
                false
            }
        },
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
pub unsafe extern "C" fn init_bert_token_classifier(
    model_path: *const c_char,
    num_classes: i32,
    use_cpu: bool,
) -> bool {
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error converting model path: {e}");
                return false;
            }
        }
    };

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
        Ok(_classifier) => true,
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
pub unsafe extern "C" fn init_candle_bert_classifier(
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
                Ok(classifier) => {
                    crate::model_architectures::traditional::bert::TRADITIONAL_BERT_CLASSIFIER
                        .set(Arc::new(classifier))
                        .is_ok()
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
pub unsafe extern "C" fn init_candle_bert_token_classifier(
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
#[no_mangle]
pub unsafe extern "C" fn init_lora_unified_classifier(
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
pub unsafe extern "C" fn init_hallucination_model(model_path: *const c_char, use_cpu: bool) -> bool {
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
pub unsafe extern "C" fn init_nli_model(model_path: *const c_char, use_cpu: bool) -> bool {
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
///
/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn is_nli_model_initialized() -> bool {
    NLI_CLASSIFIER.get().is_some()
}
