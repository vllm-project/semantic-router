//! FFI Initialization Functions for mmBERT (Multilingual ModernBERT)
//!
//! Contains initialization functions for mmBERT (8K context) and mmBERT-32K
//! (YaRN RoPE scaling, 32K context) classifiers.

use std::ffi::{c_char, CStr};
use std::sync::{Arc, OnceLock};

// ============================================================================
// mmBERT (Multilingual ModernBERT) Initialization Functions
// ============================================================================

pub static MMBERT_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static MMBERT_TOKEN_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier>,
> = OnceLock::new();

// Global statics for mmBERT-32K classifiers (32K context with YaRN RoPE scaling)
pub static MMBERT_32K_INTENT_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static MMBERT_32K_FACTCHECK_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static MMBERT_32K_JAILBREAK_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static MMBERT_32K_FEEDBACK_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();
pub static MMBERT_32K_PII_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier>,
> = OnceLock::new();
pub static MMBERT_32K_MODALITY_CLASSIFIER: OnceLock<
    Arc<crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier>,
> = OnceLock::new();

/// Initialize mmBERT classifier (multilingual ModernBERT)
///
/// mmBERT is a multilingual encoder supporting 1800+ languages with:
/// - 256k vocabulary
/// - 8192 max sequence length
/// - RoPE positional embeddings
///
/// Reference: https://huggingface.co/jhu-clsp/mmBERT-base
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
///
/// # Returns
/// - true if initialization succeeded
/// - false if initialization failed
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "🌐 Initializing mmBERT (multilingual) classifier from: {}",
        model_id
    );

    // Explicitly load as Multilingual variant
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual,
    ) {
        Ok(model) => {
            let is_multilingual = model.is_multilingual();
            eprintln!("   mmBERT loaded (is_multilingual: {})", is_multilingual);
            MMBERT_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT classifier: {}", e);
            false
        }
    }
}

/// Initialize mmBERT classifier with auto-detection
///
/// This function auto-detects whether a model is mmBERT (multilingual) or standard ModernBERT
/// based on the model's config.json (vocab_size >= 200000 and position_embedding_type == "sans_pos").
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
///
/// # Returns
/// - true if initialization succeeded
/// - false if initialization failed
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_classifier_auto(model_id: *const c_char, use_cpu: bool) -> bool {
    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!("🔍 Auto-detecting ModernBERT variant from: {}", model_id);

    // Load with auto-detection (will detect mmBERT from config.json)
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory(
        model_id,
        use_cpu,
    ) {
        Ok(model) => {
            let variant = model.variant();
            let is_multilingual = model.is_multilingual();
            eprintln!("   Detected variant: {:?} (multilingual: {})", variant, is_multilingual);
            MMBERT_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize classifier: {}", e);
            false
        }
    }
}

/// Initialize mmBERT token classifier (multilingual)
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
///
/// # Returns
/// - true if initialization succeeded
/// - false if initialization failed
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_token_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "🌐 Initializing mmBERT (multilingual) token classifier from: {}",
        model_id
    );

    // Explicitly load as Multilingual variant
    match crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier::new_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual,
    ) {
        Ok(classifier) => {
            let is_multilingual = classifier.is_multilingual();
            eprintln!("   mmBERT token classifier loaded (is_multilingual: {})", is_multilingual);
            MMBERT_TOKEN_CLASSIFIER.set(Arc::new(classifier)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT token classifier: {}", e);
            false
        }
    }
}

/// Check if a model is mmBERT (multilingual) based on config.json
///
/// Returns true if the model has vocab_size >= 200000 and uses sans_pos position embeddings.
///
/// # Safety
/// - `config_path` must be a valid null-terminated C string pointing to config.json
#[no_mangle]
pub unsafe extern "C" fn is_mmbert_model(config_path: *const c_char) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let config_path = unsafe {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertVariant::detect_from_config(config_path) {
        Ok(variant) => {
            // Both Multilingual and Multilingual32K are mmBERT variants
            variant == ModernBertVariant::Multilingual
                || variant == ModernBertVariant::Multilingual32K
        }
        Err(_) => false,
    }
}

// ============================================================================
// mmBERT-32K (YaRN RoPE scaling) FFI functions
// These support 32K context length with multilingual capabilities
// Reference: https://huggingface.co/llm-semantic-router/mmbert-32k-yarn
// ============================================================================

/// Initialize mmBERT-32K intent classifier
///
/// Model classifies text into MMLU-Pro academic categories for request routing.
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_intent_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "🎯 Initializing mmBERT-32K intent classifier from: {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(model) => {
            eprintln!("   mmBERT-32K intent classifier loaded (32K context, YaRN RoPE)");
            MMBERT_32K_INTENT_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K intent classifier: {}", e);
            false
        }
    }
}

/// Initialize mmBERT-32K fact-check classifier
///
/// Model classifies if text needs fact-checking.
/// Outputs: 0=NO_FACT_CHECK_NEEDED, 1=FACT_CHECK_NEEDED
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_factcheck_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "Initializing mmBERT-32K fact-check classifier from: {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(model) => {
            eprintln!("   mmBERT-32K fact-check classifier loaded");
            MMBERT_32K_FACTCHECK_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K fact-check classifier: {}", e);
            false
        }
    }
}

/// Initialize mmBERT-32K jailbreak detector
///
/// Model detects prompt injection/jailbreak attempts.
/// Outputs: 0=benign, 1=jailbreak
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_jailbreak_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "Initializing mmBERT-32K jailbreak detector from: {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(model) => {
            eprintln!("   mmBERT-32K jailbreak detector loaded");
            MMBERT_32K_JAILBREAK_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K jailbreak detector: {}", e);
            false
        }
    }
}

/// Initialize mmBERT-32K feedback detector
///
/// Model detects user satisfaction from follow-up messages.
/// Outputs: 0=SAT, 1=NEED_CLARIFICATION, 2=WRONG_ANSWER, 3=WANT_DIFFERENT
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_feedback_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "📊 Initializing mmBERT-32K feedback detector from: {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(model) => {
            eprintln!("   mmBERT-32K feedback detector loaded");
            MMBERT_32K_FEEDBACK_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K feedback detector: {}", e);
            false
        }
    }
}

/// Initialize mmBERT-32K PII detector (token classification)
///
/// Model detects 17 types of PII entities using BIO tagging.
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_pii_classifier(model_id: *const c_char, use_cpu: bool) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!("Initializing mmBERT-32K PII detector from: {}", model_id);

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertTokenClassifier::new_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(classifier) => {
            eprintln!("   mmBERT-32K PII detector loaded");
            MMBERT_32K_PII_CLASSIFIER.set(Arc::new(classifier)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K PII detector: {}", e);
            false
        }
    }
}

/// Initialize mmBERT-32K modality routing classifier
///
/// Classifies user prompt intent into response modality:
/// - AR (0): Text-only response via autoregressive LLM
/// - DIFFUSION (1): Image generation via diffusion model
/// - BOTH (2): Hybrid response requiring both text and image
///
/// Reference: https://huggingface.co/llm-semantic-router/mmbert32k-modality-router-merged
///
/// # Safety
/// - `model_id` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_32k_modality_classifier(
    model_id: *const c_char,
    use_cpu: bool,
) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let model_id = unsafe {
        match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    eprintln!(
        "🎯 Initializing mmBERT-32K modality routing classifier from: {}",
        model_id
    );

    match crate::model_architectures::traditional::modernbert::TraditionalModernBertClassifier::load_from_directory_with_variant(
        model_id,
        use_cpu,
        ModernBertVariant::Multilingual32K,
    ) {
        Ok(model) => {
            eprintln!("   mmBERT-32K modality router loaded (AR/DIFFUSION/BOTH, 32K context)");
            MMBERT_32K_MODALITY_CLASSIFIER.set(Arc::new(model)).is_ok()
        }
        Err(e) => {
            eprintln!("   ✗ Failed to initialize mmBERT-32K modality router: {}", e);
            false
        }
    }
}

/// Check if a model is mmBERT-32K (YaRN scaled) based on config.json
///
/// Returns true if the model has max_position_embeddings >= 16384 or rope_theta >= 100000
///
/// # Safety
/// - `config_path` must be a valid null-terminated C string pointing to config.json
#[no_mangle]
pub unsafe extern "C" fn is_mmbert_32k_model(config_path: *const c_char) -> bool {
    use crate::model_architectures::traditional::modernbert::ModernBertVariant;

    let config_path = unsafe {
        match CStr::from_ptr(config_path).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        }
    };

    match ModernBertVariant::detect_from_config(config_path) {
        Ok(variant) => variant == ModernBertVariant::Multilingual32K,
        Err(_) => false,
    }
}
