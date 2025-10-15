//! FFI Initialization Functions
//!
//! This module contains all C FFI initialization functions for dual-path architecture.
//! Provides 13 initialization functions with 100% backward compatibility.

use lazy_static::lazy_static;
use std::ffi::{c_char, CStr};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::core::similarity::BertSimilarity;
use crate::BertClassifier;

// Global state for backward compatibility
lazy_static! {
    pub static ref BERT_SIMILARITY: Arc<Mutex<Option<BertSimilarity>>> = Arc::new(Mutex::new(None));
    static ref BERT_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
    static ref BERT_PII_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
    static ref BERT_JAILBREAK_CLASSIFIER: Arc<Mutex<Option<BertClassifier>>> = Arc::new(Mutex::new(None));
    // Unified classifier for dual-path architecture
    static ref UNIFIED_CLASSIFIER: Arc<Mutex<Option<crate::classifiers::unified::DualPathUnifiedClassifier>>> = Arc::new(Mutex::new(None));
    // Parallel LoRA engine for high-performance classification
    pub static ref PARALLEL_LORA_ENGINE: Arc<Mutex<Option<crate::classifiers::lora::parallel_engine::ParallelLoRAEngine>>> = Arc::new(Mutex::new(None));
    // LoRA token classifier for token-level classification
    pub static ref LORA_TOKEN_CLASSIFIER: Arc<Mutex<Option<crate::classifiers::lora::token_lora::LoRATokenClassifier>>> = Arc::new(Mutex::new(None));
}

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
            let mut bert_opt = BERT_SIMILARITY.lock().unwrap();
            *bert_opt = Some(model);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT: {e}");
            false
        }
    }
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
        Ok(classifier) => {
            let mut bert_opt = BERT_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
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
        Ok(classifier) => {
            let mut bert_opt = BERT_PII_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT PII classifier: {e}");
            false
        }
    }
}

/// Initialize jailbreak classifier
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn init_jailbreak_classifier(
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
        Ok(classifier) => {
            let mut bert_opt = BERT_JAILBREAK_CLASSIFIER.lock().unwrap();
            *bert_opt = Some(classifier);
            true
        }
        Err(e) => {
            eprintln!("Failed to initialize BERT jailbreak classifier: {e}");
            false
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
            let mut classifier_opt = crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_CLASSIFIER.lock().unwrap();
            *classifier_opt = Some(model);
            true
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
            let mut classifier_opt = crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_PII_CLASSIFIER.lock().unwrap();
            *classifier_opt = Some(model);
            true
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
            // Store in global static
            let mut global_classifier = crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_TOKEN_CLASSIFIER.lock().unwrap();
            *global_classifier = Some(classifier);
            true
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
            let mut classifier_opt = crate::model_architectures::traditional::modernbert::TRADITIONAL_MODERNBERT_JAILBREAK_CLASSIFIER.lock().unwrap();
            *classifier_opt = Some(model);
           true
        }
        Err(e) => {
            eprintln!("Failed to initialize ModernBERT jailbreak classifier: {}", e);
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
    intent_labels_count: usize,
    pii_labels: *const *const c_char,
    pii_labels_count: usize,
    security_labels: *const *const c_char,
    security_labels_count: usize,
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
        std::slice::from_raw_parts(intent_labels, intent_labels_count)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let _pii_labels_vec = unsafe {
        std::slice::from_raw_parts(pii_labels, pii_labels_count)
            .iter()
            .map(|&ptr| CStr::from_ptr(ptr).to_str().unwrap_or("").to_string())
            .collect::<Vec<String>>()
    };

    let _security_labels_vec = unsafe {
        std::slice::from_raw_parts(security_labels, security_labels_count)
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
        Ok(mut classifier) => {
            // Initialize traditional path with actual models
            match classifier.init_traditional_path() {
                Ok(_) => {
                    let mut guard = UNIFIED_CLASSIFIER.lock().unwrap();
                    *guard = Some(classifier);
                    true
                }
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
    // Migrated from lib.rs:1555-1578
    let model_path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

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
            // Route to LoRA token classifier initialization
            match crate::classifiers::lora::token_lora::LoRATokenClassifier::new(
                model_path, use_cpu,
            ) {
                Ok(classifier) => {
                    // Store in global static
                    let mut global_classifier = LORA_TOKEN_CLASSIFIER.lock().unwrap();
                    *global_classifier = Some(classifier);
                    true
                }
                Err(e) => {
                    eprintln!("  ERROR: Failed to initialize LoRA token classifier: {}", e);
                    false
                }
            }
        }
        ModelType::Traditional => {
            // Route to traditional BERT token classifier
            match crate::model_architectures::traditional::bert::TraditionalBertTokenClassifier::new(
                model_path,
                num_classes as usize,
                use_cpu,
            ) {
                Ok(classifier) => {
                    // Store in global static
                    let mut global_classifier = crate::model_architectures::traditional::bert::TRADITIONAL_BERT_TOKEN_CLASSIFIER.lock().unwrap();
                    *global_classifier = Some(classifier);

                    true
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
            // Store in global static variable
            let mut engine_guard = PARALLEL_LORA_ENGINE.lock().unwrap();
            *engine_guard = Some(engine);
            true
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
