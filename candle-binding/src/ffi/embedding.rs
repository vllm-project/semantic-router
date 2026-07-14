//! Embedding Generation FFI Module
//!
//! This module provides Foreign Function Interface (FFI) functions for
//! intelligent embedding generation with automatic model selection.

use crate::classifiers::unified::{DualPathUnifiedClassifier, EmbeddingRequirements};
use crate::ffi::types::{
    BatchSimilarityResult, EmbeddingResult, EmbeddingSimilarityResult, SimilarityMatch,
};
use crate::model_architectures::embedding::tokenizer_contract::{
    encode_embedding_batch_checked, encode_embedding_batch_for_routing, encode_embedding_checked,
    prepare_embedding_tokenizer, EmbeddingTokenError,
};
use crate::model_architectures::ModelType;
use std::ffi::{c_char, CStr};

use super::image_input::{decode_resize_to_chw_f32, MAX_MULTIMODAL_IMAGE_ENCODED_BYTES};

//Import embedding models and model factory
use crate::model_architectures::config::{DualPathConfig, EmbeddingConfig};
use crate::model_architectures::model_factory::ModelFactory;
use std::sync::{Mutex, OnceLock};

// ============================================================================
// Refactoring: Shared embedding generation logic
// ============================================================================

/// Padding direction for tokenized sequences
#[derive(Clone, Copy, Debug)]
enum PaddingSide {
    /// Left padding (Qwen3)
    Left,
    /// Right padding (Gemma)
    Right,
}

#[derive(Clone, Copy, Debug)]
struct BatchEmbeddingSpec<'a> {
    target_dim: Option<usize>,
    max_tokens: usize,
    model_name: &'a str,
    pad_token_id: u32,
    pad_side: PaddingSide,
}

const QWEN3_EMBEDDING_CONTEXT: usize = 32_768;
const MMBERT_EMBEDDING_CONTEXT: usize = 32_768;
const GEMMA_EMBEDDING_CONTEXT: usize = 2_048;
const MULTIMODAL_TEXT_CONTEXT: usize = 512;

type GeneratedEmbedding = (Vec<f32>, usize);

const EMBEDDING_INPUT_TOO_LONG_STATUS: i32 = -3;

#[derive(Debug)]
enum EmbeddingGenerationError {
    Token(EmbeddingTokenError),
    Internal(String),
}

impl EmbeddingGenerationError {
    fn status(&self) -> i32 {
        match self {
            Self::Token(error) if error.is_input_too_long() => EMBEDDING_INPUT_TOO_LONG_STATUS,
            _ => -1,
        }
    }
}

impl std::fmt::Display for EmbeddingGenerationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token(error) => error.fmt(formatter),
            Self::Internal(detail) => formatter.write_str(detail),
        }
    }
}

impl From<EmbeddingTokenError> for EmbeddingGenerationError {
    fn from(error: EmbeddingTokenError) -> Self {
        Self::Token(error)
    }
}

impl From<String> for EmbeddingGenerationError {
    fn from(detail: String) -> Self {
        Self::Internal(detail)
    }
}

type EmbeddingGenerationResult<T> = Result<T, EmbeddingGenerationError>;

/// Global singleton for ModelFactory
pub(crate) static GLOBAL_MODEL_FACTORY: OnceLock<ModelFactory> = OnceLock::new();

/// Serializes every embedding-model publication. Loading can be expensive, but
/// more importantly a concurrent multimodal/unified startup must never race to
/// publish incompatible contents into a process-global `OnceLock`.
static EMBEDDING_INIT_LOCK: Mutex<()> = Mutex::new(());

use crate::model_architectures::embedding::MultiModalEmbeddingModel;
use tokenizers::Tokenizer as MmTokenizer;

/// Standalone multimodal model storage — allows initialization independent of the
/// main ModelFactory (which uses OnceLock and can only be set once).
static STANDALONE_MULTIMODAL: OnceLock<(MultiModalEmbeddingModel, MmTokenizer, String)> =
    OnceLock::new();

pub(crate) fn truncate_embedding_to_dimension(
    embedding: Vec<f32>,
    target_dim: Option<usize>,
) -> Vec<f32> {
    let Some(dim) = target_dim else {
        return embedding;
    };

    let actual_dim = if dim > embedding.len() {
        eprintln!(
            "WARNING: Requested dimension {} exceeds model dimension {}, using full dimension",
            dim,
            embedding.len()
        );
        embedding.len()
    } else {
        dim
    };

    if actual_dim >= embedding.len() {
        return embedding;
    }

    let mut truncated = embedding[..actual_dim].to_vec();
    let norm = truncated
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if norm > 0.0 {
        for value in &mut truncated {
            *value /= norm;
        }
    }
    truncated
}

/// Get a reference to the multimodal model + tokenizer, checking standalone first
/// then falling back to the factory.
fn get_multimodal_refs() -> Option<(&'static MultiModalEmbeddingModel, &'static MmTokenizer)> {
    if let Some((model, tokenizer, _)) = STANDALONE_MULTIMODAL.get() {
        return Some((model, tokenizer));
    }
    if let Some(factory) = GLOBAL_MODEL_FACTORY.get() {
        if let (Some(model), Some(tokenizer)) = (
            factory.get_multimodal_model(),
            factory.get_multimodal_tokenizer(),
        ) {
            return Some((model, tokenizer));
        }
    }
    None
}

/// Generic internal helper for single text embedding generation
///
/// This function extracts common logic for both Qwen3 and Gemma models.
/// Model-specific logic (tokenizer retrieval and forward pass) is handled via closures.
///
/// # Parameters
/// - `text`: Input text to encode
/// - `target_dim`: Optional target dimension for Matryoshka truncation
/// - `get_tokenizer`: Closure to retrieve the model-specific tokenizer
/// - `forward_fn`: Closure to execute model forward pass (receives input_ids, attention_mask, returns embedding tensor)
fn generate_embedding_internal<'a, F, G>(
    text: &str,
    target_dim: Option<usize>,
    max_tokens: usize,
    model_name: &str,
    get_tokenizer: G,
    forward_fn: F,
) -> EmbeddingGenerationResult<GeneratedEmbedding>
where
    F: Fn(Vec<u32>, Vec<u32>) -> Result<candle_core::Tensor, String>,
    G: Fn() -> Option<&'a tokenizers::Tokenizer>,
{
    // Get tokenizer
    let tokenizer = get_tokenizer().ok_or_else(|| "Tokenizer not available".to_string())?;

    // Tokenize single text
    let encoding = encode_embedding_checked(tokenizer, text, max_tokens, model_name)?;
    let sequence_length = encoding.get_ids().len();

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

    // Forward pass - returns [1, hidden_dim]
    let embedding_tensor = forward_fn(token_ids, attention_mask)?;

    // Squeeze batch dimension: [1, hidden_dim] -> [hidden_dim]
    let embedding_1d = embedding_tensor
        .squeeze(0)
        .map_err(|e| format!("Failed to squeeze batch dimension: {:?}", e))?;

    // Convert to Vec<f32>
    let embedding_vec = embedding_1d
        .to_vec1::<f32>()
        .map_err(|e| format!("Failed to convert embedding to vec: {:?}", e))?;

    Ok((
        truncate_embedding_to_dimension(embedding_vec, target_dim),
        sequence_length,
    ))
}

/// Generic internal helper for batch embedding generation
///
/// This function extracts common logic for both Qwen3 and Gemma models.
/// Model-specific logic (tokenizer retrieval and forward pass) is handled via closures.
fn generate_embeddings_batch_internal<'a, F, G>(
    texts: &[&str],
    spec: BatchEmbeddingSpec<'_>,
    get_tokenizer: G,
    forward_fn: F,
) -> EmbeddingGenerationResult<Vec<Vec<f32>>>
where
    F: Fn(Vec<u32>, Vec<u32>, usize, usize) -> Result<candle_core::Tensor, String>,
    G: Fn() -> Option<&'a tokenizers::Tokenizer>,
{
    if texts.is_empty() {
        return Err("Empty text list".to_string().into());
    }

    // Get tokenizer
    let tokenizer = get_tokenizer().ok_or_else(|| "Tokenizer not available".to_string())?;

    // Batch tokenize all texts
    let encodings =
        encode_embedding_batch_checked(tokenizer, texts, spec.max_tokens, spec.model_name)?;

    // Find max sequence length for padding
    let max_len = encodings
        .iter()
        .map(|enc| enc.get_ids().len())
        .max()
        .unwrap_or(0);

    // Prepare batch tensors
    let mut batch_token_ids = Vec::new();
    let mut batch_attention_mask = Vec::new();

    for encoding in &encodings {
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();

        // Pad to max_len based on padding side
        let pad_len = max_len - token_ids.len();
        let (padded_ids, padded_mask) = match spec.pad_side {
            PaddingSide::Left => {
                // Left padding
                let mut padded_ids = vec![spec.pad_token_id; pad_len];
                padded_ids.extend(token_ids);

                let mut padded_mask = vec![0u32; pad_len];
                padded_mask.extend(attention_mask);

                (padded_ids, padded_mask)
            }
            PaddingSide::Right => {
                // Right padding
                let mut padded_ids = token_ids.clone();
                padded_ids.extend(vec![spec.pad_token_id; pad_len]);

                let mut padded_mask = attention_mask.clone();
                padded_mask.extend(vec![0u32; pad_len]);

                (padded_ids, padded_mask)
            }
        };

        batch_token_ids.push(padded_ids);
        batch_attention_mask.push(padded_mask);
    }

    let batch_size = texts.len();
    let flat_ids: Vec<u32> = batch_token_ids.into_iter().flatten().collect();
    let flat_mask: Vec<u32> = batch_attention_mask.into_iter().flatten().collect();

    // Forward_fn is responsible for:
    // 1. Getting the model and its device
    // 2. Creating tensors on the correct device with shape (batch_size, max_len)
    // 3. Calling model.embedding_forward with the correct signature
    let embeddings = forward_fn(flat_ids, flat_mask, batch_size, max_len)?;

    let embeddings_data = embeddings
        .to_vec2::<f32>()
        .map_err(|e| format!("Failed to convert embeddings to vec: {:?}", e))?;

    let target_dim = if let Some(dim) = spec.target_dim {
        let embedding_dim = embeddings_data.first().map_or(0, Vec::len);
        if dim > embedding_dim {
            eprintln!(
                "WARNING: Requested dimension {} exceeds model dimension {}, using full dimension",
                dim, embedding_dim
            );
            Some(embedding_dim)
        } else {
            Some(dim)
        }
    } else {
        None
    };

    Ok(embeddings_data
        .into_iter()
        .map(|emb| truncate_embedding_to_dimension(emb, target_dim))
        .collect())
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct EmbeddingFactoryRequest {
    qwen3_path: Option<String>,
    gemma_path: Option<String>,
    mmbert_path: Option<String>,
}

impl EmbeddingFactoryRequest {
    fn is_empty(&self) -> bool {
        self.qwen3_path.is_none() && self.gemma_path.is_none() && self.mmbert_path.is_none()
    }

    fn validate_factory(&self, factory: &ModelFactory) -> Result<(), String> {
        validate_requested_model(
            "Qwen3",
            self.qwen3_path.as_deref(),
            factory.get_qwen3_model().is_some(),
            factory.get_qwen3_model_path(),
        )?;
        validate_requested_model(
            "Gemma",
            self.gemma_path.as_deref(),
            factory.get_gemma_model().is_some(),
            factory.get_gemma_model_path(),
        )?;
        validate_requested_model(
            "mmBERT",
            self.mmbert_path.as_deref(),
            factory.get_mmbert_model().is_some(),
            factory.get_mmbert_model_path(),
        )
    }
}

fn validate_requested_model(
    model: &str,
    requested_path: Option<&str>,
    loaded: bool,
    loaded_path: Option<&str>,
) -> Result<(), String> {
    let Some(requested_path) = requested_path else {
        return Ok(());
    };
    if !loaded {
        return Err(format!("requested {model} model is not loaded"));
    }
    if loaded_path != Some(requested_path) {
        return Err(format!(
            "requested {model} model path does not match the process-global model"
        ));
    }
    Ok(())
}

fn initialize_once_with_validation<T>(
    slot: &OnceLock<T>,
    build: impl FnOnce() -> Result<T, String>,
    validate: impl Fn(&T) -> Result<(), String>,
) -> Result<(), String> {
    if let Some(existing) = slot.get() {
        return validate(existing);
    }

    let candidate = build()?;
    validate(&candidate)?;
    match slot.set(candidate) {
        Ok(()) => Ok(()),
        Err(_) => slot
            .get()
            .ok_or_else(|| "global model publication lost without a winner".to_string())
            .and_then(validate),
    }
}

fn parse_optional_model_path(
    model_path: *const c_char,
    field: &str,
) -> Result<Option<String>, String> {
    if model_path.is_null() {
        return Ok(None);
    }
    let path = unsafe { CStr::from_ptr(model_path) }
        .to_str()
        .map_err(|error| format!("invalid UTF-8 in {field}: {error}"))?;
    if path.is_empty() {
        return Ok(None);
    }
    Ok(Some(path.to_string()))
}

fn build_embedding_factory(
    request: &EmbeddingFactoryRequest,
    use_cpu: bool,
) -> Result<ModelFactory, String> {
    use candle_core::Device;

    if request.is_empty() {
        return Err("at least one embedding model path must be provided".to_string());
    }
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };
    let mut factory = ModelFactory::new(device);

    if let Some(path) = request.qwen3_path.as_deref() {
        factory
            .register_qwen3_embedding_model(path)
            .map_err(|error| format!("failed to register Qwen3 model: {error:?}"))?;
    }
    if let Some(path) = request.gemma_path.as_deref() {
        factory
            .register_gemma_embedding_model(path)
            .map_err(|error| format!("failed to register Gemma model: {error:?}"))?;
    }
    if let Some(path) = request.mmbert_path.as_deref() {
        factory
            .register_mmbert_embedding_model(path)
            .map_err(|error| format!("failed to register mmBERT model: {error:?}"))?;
    }
    request.validate_factory(&factory)?;
    Ok(factory)
}

fn initialize_embedding_factory(request: EmbeddingFactoryRequest, use_cpu: bool) -> bool {
    let _guard = EMBEDDING_INIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let result = initialize_once_with_validation(
        &GLOBAL_MODEL_FACTORY,
        || build_embedding_factory(&request, use_cpu),
        |factory| request.validate_factory(factory),
    );
    if let Err(error) = result {
        eprintln!("ERROR: embedding model initialization failed: {error}");
        return false;
    }
    true
}

/// Initialize mmBERT embedding model with 2D Matryoshka support.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_mmbert_embedding_model(model_path: *const c_char, use_cpu: bool) -> bool {
    let mmbert_path = match parse_optional_model_path(model_path, "mmBERT model path") {
        Ok(Some(path)) => Some(path),
        Ok(None) => {
            eprintln!("ERROR: mmBERT model path is required");
            return false;
        }
        Err(error) => {
            eprintln!("ERROR: {error}");
            return false;
        }
    };
    initialize_embedding_factory(
        EmbeddingFactoryRequest {
            mmbert_path,
            ..EmbeddingFactoryRequest::default()
        },
        use_cpu,
    )
}

/// Initialize embedding models with given paths (including mmBERT).
/// Existing process-global state is successful only when it contains every
/// requested model at the same path.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_embedding_models_with_mmbert(
    qwen3_model_path: *const c_char,
    gemma_model_path: *const c_char,
    mmbert_model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    let request = match (
        parse_optional_model_path(qwen3_model_path, "Qwen3 model path"),
        parse_optional_model_path(gemma_model_path, "Gemma model path"),
        parse_optional_model_path(mmbert_model_path, "mmBERT model path"),
    ) {
        (Ok(qwen3_path), Ok(gemma_path), Ok(mmbert_path)) => EmbeddingFactoryRequest {
            qwen3_path,
            gemma_path,
            mmbert_path,
        },
        (qwen3, gemma, mmbert) => {
            let error = qwen3
                .err()
                .or_else(|| gemma.err())
                .or_else(|| mmbert.err())
                .unwrap_or_else(|| "invalid embedding model path".to_string());
            eprintln!("ERROR: {error}");
            return false;
        }
    };
    initialize_embedding_factory(request, use_cpu)
}

/// Initialize Qwen3 and/or Gemma embedding models.
/// Existing process-global state is successful only when it contains every
/// requested model at the same path.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_embedding_models(
    qwen3_model_path: *const c_char,
    gemma_model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    let request = match (
        parse_optional_model_path(qwen3_model_path, "Qwen3 model path"),
        parse_optional_model_path(gemma_model_path, "Gemma model path"),
    ) {
        (Ok(qwen3_path), Ok(gemma_path)) => EmbeddingFactoryRequest {
            qwen3_path,
            gemma_path,
            mmbert_path: None,
        },
        (qwen3, gemma) => {
            let error = qwen3
                .err()
                .or_else(|| gemma.err())
                .unwrap_or_else(|| "invalid embedding model path".to_string());
            eprintln!("ERROR: {error}");
            return false;
        }
    };
    initialize_embedding_factory(request, use_cpu)
}

/// Helper function to create a temporary classifier for routing decisions
///
/// This is used when no global classifier is available. It creates a minimal
/// DualPathUnifiedClassifier with default configuration.
fn create_temp_classifier() -> Result<DualPathUnifiedClassifier, String> {
    use crate::model_architectures::config::{GlobalConfig, LoRAConfig, TraditionalConfig};

    DualPathUnifiedClassifier::new(DualPathConfig {
        traditional: TraditionalConfig::default(),
        lora: LoRAConfig::default(),
        embedding: EmbeddingConfig::default(),
        global: GlobalConfig::default(),
    })
    .map_err(|e| format!("Failed to create classifier: {:?}", e))
}

/// Helper function to create an error result
fn create_error_result() -> EmbeddingResult {
    EmbeddingResult {
        data: std::ptr::null_mut(),
        length: 0,
        error: true,
        model_type: -1,
        sequence_length: 0,
        processing_time_ms: 0.0,
    }
}

/// Internal helper to generate embedding for Qwen3
/// Generate embeddings for multiple texts in a single batch (Qwen3)
/// Returns a 2D vector: [num_texts, embedding_dim]
fn generate_qwen3_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<Vec<Vec<f32>>> {
    use candle_core::Tensor;

    // Qwen3-specific configuration
    const QWEN3_PAD_TOKEN_ID: u32 = 151643;
    let pad_side = PaddingSide::Left;

    // Use the generic internal function
    generate_embeddings_batch_internal(
        texts,
        BatchEmbeddingSpec {
            target_dim,
            max_tokens: QWEN3_EMBEDDING_CONTEXT,
            model_name: "Qwen3",
            pad_token_id: QWEN3_PAD_TOKEN_ID,
            pad_side,
        },
        || factory.get_qwen3_tokenizer(),
        |flat_ids, flat_mask, batch_size, max_len| {
            // Get model
            let model = factory
                .get_qwen3_model()
                .ok_or_else(|| "Qwen3 model not available".to_string())?;

            // Create tensors on the correct device
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

            // Forward pass - returns [batch_size, hidden_dim]
            model
                .embedding_forward(&input_ids, &attention_mask)
                .map_err(|e| format!("Model forward failed: {:?}", e))
        },
    )
}

fn generate_qwen3_embedding(
    factory: &ModelFactory,
    text: &str,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<GeneratedEmbedding> {
    use candle_core::Tensor;

    // Use the generic internal function
    generate_embedding_internal(
        text,
        target_dim,
        QWEN3_EMBEDDING_CONTEXT,
        "Qwen3",
        || factory.get_qwen3_tokenizer(),
        |token_ids, attention_mask| {
            // Get model
            let model = factory
                .get_qwen3_model()
                .ok_or_else(|| "Qwen3 model not available".to_string())?;

            // Create tensors on the correct device
            let device = model.device();
            let input_ids = Tensor::new(token_ids.as_slice(), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze input_ids: {:?}", e))?;

            let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze attention_mask: {:?}", e))?;

            // Forward pass - returns [1, hidden_dim]
            model
                .embedding_forward(&input_ids, &attention_mask_tensor)
                .map_err(|e| format!("Forward pass failed: {:?}", e))
        },
    )
}

/// Internal helper to generate embedding for Gemma
/// Generate embeddings for multiple texts in a single batch (Gemma)
/// Returns a 2D vector: [num_texts, embedding_dim]
fn generate_gemma_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<Vec<Vec<f32>>> {
    use candle_core::Tensor;

    // Gemma-specific configuration
    const GEMMA_PAD_TOKEN_ID: u32 = 0;
    let pad_side = PaddingSide::Right;

    // Use the generic internal function
    generate_embeddings_batch_internal(
        texts,
        BatchEmbeddingSpec {
            target_dim,
            max_tokens: GEMMA_EMBEDDING_CONTEXT,
            model_name: "Gemma",
            pad_token_id: GEMMA_PAD_TOKEN_ID,
            pad_side,
        },
        || factory.get_gemma_tokenizer(),
        |flat_ids, flat_mask, batch_size, max_len| {
            // Get model
            let model = factory
                .get_gemma_model()
                .ok_or_else(|| "Gemma model not available".to_string())?;

            // Create tensors on the correct device
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

            // Forward pass - returns [batch_size, hidden_dim]
            // Note: Gemma requires Some(&attention_mask)
            model
                .embedding_forward(&input_ids, Some(&attention_mask))
                .map_err(|e| format!("Model forward failed: {:?}", e))
        },
    )
}

fn generate_gemma_embedding(
    factory: &ModelFactory,
    text: &str,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<GeneratedEmbedding> {
    use candle_core::Tensor;

    // Use the generic internal function
    generate_embedding_internal(
        text,
        target_dim,
        GEMMA_EMBEDDING_CONTEXT,
        "Gemma",
        || factory.get_gemma_tokenizer(),
        |token_ids, attention_mask| {
            // Get model
            let model = factory
                .get_gemma_model()
                .ok_or_else(|| "Gemma model not available".to_string())?;

            // Create tensors on the correct device
            let device = model.device();
            let input_ids = Tensor::new(token_ids.as_slice(), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze input_ids: {:?}", e))?;

            let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze attention_mask: {:?}", e))?;

            // Forward pass - returns [1, hidden_dim]
            // Note: Gemma requires Some(&attention_mask_tensor)
            model
                .embedding_forward(&input_ids, Some(&attention_mask_tensor))
                .map_err(|e| format!("Forward pass failed: {:?}", e))
        },
    )
}

/// Internal helper to generate embedding for mmBERT with 2D Matryoshka
fn generate_mmbert_embedding(
    factory: &ModelFactory,
    text: &str,
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<GeneratedEmbedding> {
    use candle_core::Tensor;

    let model = factory
        .get_mmbert_model()
        .ok_or_else(|| "mmBERT model not available".to_string())?;

    let tokenizer = factory
        .get_mmbert_tokenizer()
        .ok_or_else(|| "mmBERT tokenizer not available".to_string())?;

    // Tokenize
    let encoding = encode_embedding_checked(tokenizer, text, MMBERT_EMBEDDING_CONTEXT, "mmBERT")?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let seq_len = token_ids.len();

    // Create tensors
    let device = model.device();
    let input_ids = Tensor::from_vec(token_ids, (1, seq_len), device)
        .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
    let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, seq_len), device)
        .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

    // Forward pass with 2D Matryoshka (layer early exit + dimension truncation)
    let embedding = model
        .embedding_forward_with_matryoshka(
            &input_ids,
            Some(&attention_mask_tensor),
            target_layer,
            target_dim,
        )
        .map_err(|e| format!("mmBERT forward failed: {:?}", e))?;

    // Convert to Vec<f32>
    let embedding = embedding
        .squeeze(0)
        .map_err(|e| format!("Failed to squeeze: {:?}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("Failed to convert to vec: {:?}", e))?;
    Ok((embedding, seq_len))
}

/// Generate embeddings for multiple texts in a single batch (mmBERT)
fn generate_mmbert_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<Vec<Vec<f32>>> {
    use candle_core::Tensor;

    generate_embeddings_batch_internal(
        texts,
        BatchEmbeddingSpec {
            target_dim: None,
            max_tokens: MMBERT_EMBEDDING_CONTEXT,
            model_name: "mmBERT",
            pad_token_id: 0,
            pad_side: PaddingSide::Right,
        },
        || factory.get_mmbert_tokenizer(),
        |flat_ids, flat_mask, batch_size, max_len| {
            let model = factory
                .get_mmbert_model()
                .ok_or_else(|| "mmBERT model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), device)
                .map_err(|error| format!("Failed to create mmBERT input tensor: {error:?}"))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), device)
                .map_err(|error| format!("Failed to create mmBERT mask tensor: {error:?}"))?;
            model
                .embedding_forward_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    target_layer,
                    target_dim,
                )
                .map_err(|error| format!("mmBERT batch forward failed: {error:?}"))
        },
    )
}

/// Internal helper to generate text embedding via the multi-modal model
fn generate_multimodal_text_embedding(
    text: &str,
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<GeneratedEmbedding> {
    use candle_core::Tensor;

    let (model, tokenizer) =
        get_multimodal_refs().ok_or_else(|| "Multi-modal model not available".to_string())?;

    let encoding =
        encode_embedding_checked(tokenizer, text, MULTIMODAL_TEXT_CONTEXT, "multimodal")?;

    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let seq_len = token_ids.len();

    let device = model.device();
    let input_ids = Tensor::from_vec(token_ids, (1, seq_len), device)
        .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
    let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, seq_len), device)
        .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

    let embedding = model
        .encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask_tensor),
            target_layer,
            target_dim,
        )
        .map_err(|e| format!("Multi-modal text encoding failed: {:?}", e))?;

    let embedding = embedding
        .squeeze(0)
        .map_err(|e| format!("Failed to squeeze: {:?}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("Failed to convert to vec: {:?}", e))?;
    Ok((embedding, seq_len))
}

/// Generate text embeddings for multiple texts in a single batch (multi-modal)
fn generate_multimodal_text_embeddings_batch(
    texts: &[&str],
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> EmbeddingGenerationResult<Vec<Vec<f32>>> {
    use candle_core::Tensor;

    generate_embeddings_batch_internal(
        texts,
        BatchEmbeddingSpec {
            target_dim: None,
            max_tokens: MULTIMODAL_TEXT_CONTEXT,
            model_name: "multimodal",
            pad_token_id: 0,
            pad_side: PaddingSide::Right,
        },
        || get_multimodal_refs().map(|(_, tokenizer)| tokenizer),
        |flat_ids, flat_mask, batch_size, max_len| {
            let (model, _) = get_multimodal_refs()
                .ok_or_else(|| "Multi-modal model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), device)
                .map_err(|error| format!("Failed to create multimodal input tensor: {error:?}"))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), device)
                .map_err(|error| format!("Failed to create multimodal mask tensor: {error:?}"))?;
            model
                .encode_text_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    target_layer,
                    target_dim,
                )
                .map_err(|error| format!("Multi-modal batch forward failed: {error:?}"))
        },
    )
}

/// Get embedding with automatic model selection (smart routing)
///
/// This function automatically selects the best embedding model based on:
/// - Sequence length
/// - Quality priority (0.0 to 1.0)
/// - Latency priority (0.0 to 1.0)
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `result` must be a valid pointer to EmbeddingResult
///
/// # Returns
/// 0 on success, -3 when the selected tokenizer context is exceeded, -1 on internal error
#[no_mangle]
pub extern "C" fn get_embedding_smart(
    text: *const c_char,
    quality_priority: f32,
    latency_priority: f32,
    result: *mut EmbeddingResult,
) -> i32 {
    // Simply forward to get_embedding_with_dim with target_dim = 0 (auto)
    get_embedding_with_dim(text, quality_priority, latency_priority, 0, result)
}

/// Get embedding with automatic model selection and target dimension
///
/// This function is similar to `get_embedding_smart` but also supports Matryoshka representation
/// by allowing the caller to specify a target dimension (768, 512, 256, or 128).
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `result` must be a valid pointer to EmbeddingResult
///
/// # Returns
/// 0 on success, -3 when the selected tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn get_embedding_with_dim(
    text: *const c_char,
    quality_priority: f32,
    latency_priority: f32,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_with_dim");
        return -1;
    }

    let text_str = unsafe {
        match std::ffi::CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    let target_dimension = (target_dim > 0).then_some(target_dim as usize);
    let model_type = match select_auto_embedding_model(
        GLOBAL_MODEL_FACTORY.get(),
        &[text_str],
        target_dimension,
        quality_priority,
        latency_priority,
    ) {
        Ok(model_type) => model_type,
        Err(error) => {
            eprintln!("Error: failed to select embedding model: {error}");
            unsafe {
                (*result) = create_error_result();
            }
            return error.status();
        }
    };

    let model_type_str = match model_type {
        ModelType::Qwen3Embedding => "qwen3",
        ModelType::GemmaEmbedding => "gemma",
        ModelType::MmBertEmbedding => "mmbert",
        ModelType::MultiModalEmbedding => "multimodal",
        _ => unreachable!("auto embedding selection returned an unsupported model"),
    };

    // Call get_embedding_2d_matryoshka which handles all model types
    let model_type_cstr = std::ffi::CString::new(model_type_str).unwrap();
    get_embedding_2d_matryoshka(text, model_type_cstr.as_ptr(), 0, target_dim, result)
}

/// Get embedding with manually specified model type (no automatic routing)
///
/// This function bypasses the automatic routing logic and directly uses the specified model.
/// Useful when the caller explicitly wants to use a specific embedding model.
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `model_type_str`: "qwen3", "gemma", or "mmbert"
/// - `target_dim`: Target dimension (768, 512, 256, or 128, 0 for default)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -3 when the selected tokenizer context is exceeded, -1 on internal error
#[no_mangle]
pub extern "C" fn get_embedding_with_model_type(
    text: *const c_char,
    model_type_str: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    // Forward to 2D Matryoshka function with target_layer=0 (full layers)
    get_embedding_2d_matryoshka(text, model_type_str, 0, target_dim, result)
}

/// Get embedding with 2D Matryoshka support (layer early exit + dimension truncation)
///
/// This function supports the full 2D Matryoshka API for mmBERT models:
/// - Layer early exit: Use fewer layers (3, 6, 11, or 22) for faster inference
/// - Dimension truncation: Use smaller dimensions (64, 128, 256, 512, 768)
///
/// For qwen3 and gemma models, only dimension truncation is supported (target_layer is ignored).
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `model_type_str`: "qwen3", "gemma", or "mmbert"
/// - `target_layer`: Target layer for early exit (0 for full model, only mmbert supports this)
/// - `target_dim`: Target dimension (0 for default)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -3 when the selected tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn get_embedding_2d_matryoshka(
    text: *const c_char,
    model_type_str: *const c_char,
    target_layer: i32,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || model_type_str.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_2d_matryoshka");
        return -1;
    }

    let text_str = unsafe {
        match std::ffi::CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    let model_type_str = unsafe {
        match std::ffi::CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    // Parse model type
    let model_type = match model_type_str {
        "qwen3" => ModelType::Qwen3Embedding,
        "gemma" => ModelType::GemmaEmbedding,
        "mmbert" => ModelType::MmBertEmbedding,
        "multimodal" => ModelType::MultiModalEmbedding,
        _ => {
            eprintln!(
                "Error: invalid model type '{}' (must be 'qwen3', 'gemma', 'mmbert', or 'multimodal')",
                model_type_str
            );
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };

    let factory = GLOBAL_MODEL_FACTORY.get();

    let start_time = std::time::Instant::now();

    // Generate embedding based on model type
    let embedding_result = match model_type {
        ModelType::Qwen3Embedding => require_unified_factory(factory)
            .and_then(|factory| generate_qwen3_embedding(factory, text_str, target_dimension)),
        ModelType::GemmaEmbedding => require_unified_factory(factory)
            .and_then(|factory| generate_gemma_embedding(factory, text_str, target_dimension)),
        ModelType::MmBertEmbedding => require_unified_factory(factory).and_then(|factory| {
            generate_mmbert_embedding(factory, text_str, layer, target_dimension)
        }),
        ModelType::MultiModalEmbedding => {
            generate_multimodal_text_embedding(text_str, layer, target_dimension)
        }
        _ => {
            eprintln!("Error: unsupported model type: {:?}", model_type);
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    match embedding_result {
        Ok((embedding_vec, sequence_length)) => {
            let length = embedding_vec.len() as i32;
            let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            let model_type_id = embedding_model_id(model_type).unwrap_or(-1);

            unsafe {
                (*result) = EmbeddingResult {
                    data,
                    length,
                    error: false,
                    model_type: model_type_id,
                    sequence_length: sequence_length as i32,
                    processing_time_ms,
                };
            }

            0
        }
        Err(e) => {
            eprintln!("Error: embedding generation failed: {}", e);
            unsafe {
                (*result) = create_error_result();
            }
            e.status()
        }
    }
}

fn embedding_model_id(model_type: ModelType) -> Option<i32> {
    match model_type {
        ModelType::Qwen3Embedding => Some(0),
        ModelType::GemmaEmbedding => Some(1),
        ModelType::MmBertEmbedding => Some(2),
        ModelType::MultiModalEmbedding => Some(3),
        _ => None,
    }
}

fn require_unified_factory(
    factory: Option<&ModelFactory>,
) -> EmbeddingGenerationResult<&ModelFactory> {
    factory.ok_or_else(|| "ModelFactory not initialized".to_string().into())
}

fn preferred_auto_embedding_model(sequence_length: usize) -> ModelType {
    if (513..=GEMMA_EMBEDDING_CONTEXT).contains(&sequence_length) {
        ModelType::GemmaEmbedding
    } else {
        ModelType::Qwen3Embedding
    }
}

fn embedding_model_context(model_type: ModelType) -> Option<usize> {
    match model_type {
        ModelType::Qwen3Embedding => Some(QWEN3_EMBEDDING_CONTEXT),
        ModelType::GemmaEmbedding => Some(GEMMA_EMBEDDING_CONTEXT),
        ModelType::MmBertEmbedding => Some(MMBERT_EMBEDDING_CONTEXT),
        ModelType::MultiModalEmbedding => Some(MULTIMODAL_TEXT_CONTEXT),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AutoEmbeddingCandidate {
    model_type: ModelType,
    sequence_length: usize,
}

fn routing_sequence_length(
    tokenizer: &MmTokenizer,
    texts: &[&str],
    model: &str,
) -> Result<usize, EmbeddingTokenError> {
    if texts.is_empty() {
        return Err(EmbeddingTokenError::EmptyEncoding {
            model: format!("{model} routing batch"),
        });
    }
    encode_embedding_batch_for_routing(tokenizer, texts, model)?
        .iter()
        .map(|encoding| encoding.get_ids().len())
        .max()
        .ok_or_else(|| EmbeddingTokenError::EmptyEncoding {
            model: format!("{model} routing batch"),
        })
}

fn push_factory_auto_candidate(
    candidates: &mut Vec<AutoEmbeddingCandidate>,
    model_type: ModelType,
    model_name: &str,
    model_loaded: bool,
    tokenizer: Option<&MmTokenizer>,
    texts: &[&str],
) -> EmbeddingGenerationResult<()> {
    if model_loaded != tokenizer.is_some() {
        return Err(format!(
            "{model_name} model/tokenizer initialization invariant is inconsistent"
        )
        .into());
    }
    if let Some(tokenizer) = tokenizer {
        candidates.push(AutoEmbeddingCandidate {
            model_type,
            sequence_length: routing_sequence_length(tokenizer, texts, model_name)?,
        });
    }
    Ok(())
}

fn auto_embedding_candidates(
    factory: Option<&ModelFactory>,
    texts: &[&str],
) -> EmbeddingGenerationResult<Vec<AutoEmbeddingCandidate>> {
    let mut candidates = Vec::with_capacity(4);
    if let Some(factory) = factory {
        push_factory_auto_candidate(
            &mut candidates,
            ModelType::Qwen3Embedding,
            "Qwen3",
            factory.get_qwen3_model().is_some(),
            factory.get_qwen3_tokenizer(),
            texts,
        )?;
        push_factory_auto_candidate(
            &mut candidates,
            ModelType::GemmaEmbedding,
            "Gemma",
            factory.get_gemma_model().is_some(),
            factory.get_gemma_tokenizer(),
            texts,
        )?;
        push_factory_auto_candidate(
            &mut candidates,
            ModelType::MmBertEmbedding,
            "mmBERT",
            factory.get_mmbert_model().is_some(),
            factory.get_mmbert_tokenizer(),
            texts,
        )?;
    }
    if let Some((_, tokenizer)) = get_multimodal_refs() {
        candidates.push(AutoEmbeddingCandidate {
            model_type: ModelType::MultiModalEmbedding,
            sequence_length: routing_sequence_length(tokenizer, texts, "multimodal")?,
        });
    }
    if candidates.is_empty() {
        return Err("no embedding model is initialized".to_string().into());
    }
    Ok(candidates)
}

fn resolve_auto_embedding_model(
    preferred: ModelType,
    candidates: &[AutoEmbeddingCandidate],
) -> Option<ModelType> {
    let preference_order = match preferred {
        ModelType::Qwen3Embedding => [
            ModelType::Qwen3Embedding,
            ModelType::MmBertEmbedding,
            ModelType::GemmaEmbedding,
            ModelType::MultiModalEmbedding,
        ],
        ModelType::GemmaEmbedding => [
            ModelType::GemmaEmbedding,
            ModelType::MmBertEmbedding,
            ModelType::Qwen3Embedding,
            ModelType::MultiModalEmbedding,
        ],
        ModelType::MmBertEmbedding => [
            ModelType::MmBertEmbedding,
            ModelType::Qwen3Embedding,
            ModelType::GemmaEmbedding,
            ModelType::MultiModalEmbedding,
        ],
        ModelType::MultiModalEmbedding => [
            ModelType::MultiModalEmbedding,
            ModelType::MmBertEmbedding,
            ModelType::Qwen3Embedding,
            ModelType::GemmaEmbedding,
        ],
        _ => return None,
    };

    preference_order.into_iter().find(|model| {
        candidates.iter().any(|candidate| {
            candidate.model_type == *model
                && embedding_model_context(*model)
                    .is_some_and(|context| candidate.sequence_length <= context)
        })
    })
}

fn no_compatible_auto_model_error(
    candidates: &[AutoEmbeddingCandidate],
) -> EmbeddingGenerationError {
    let candidate = candidates
        .iter()
        .filter_map(|candidate| {
            embedding_model_context(candidate.model_type).map(|context| (candidate, context))
        })
        .max_by_key(|(_, context)| *context);
    match candidate {
        Some((candidate, maximum)) => EmbeddingTokenError::InputTooLong {
            model: format!("auto-routed {:?}", candidate.model_type),
            count: candidate.sequence_length,
            maximum,
        }
        .into(),
        None => "no compatible embedding model is available"
            .to_string()
            .into(),
    }
}

fn select_auto_embedding_model(
    factory: Option<&ModelFactory>,
    texts: &[&str],
    target_dimension: Option<usize>,
    quality_priority: f32,
    latency_priority: f32,
) -> EmbeddingGenerationResult<ModelType> {
    let candidates = auto_embedding_candidates(factory, texts)?;
    // The classifier accepts a single length. Use the conservative exact maximum
    // across available tokenizers, while candidate eligibility below remains
    // model-specific. Capping only avoids asking the preference classifier about
    // lengths outside its own domain; it does not weaken context enforcement.
    let sequence_length = candidates
        .iter()
        .map(|candidate| candidate.sequence_length)
        .max()
        .unwrap_or(0)
        .min(QWEN3_EMBEDDING_CONTEXT);
    let requirements = EmbeddingRequirements {
        sequence_length,
        quality_priority,
        latency_priority,
        target_dimension,
    };
    let classifier = create_temp_classifier().map_err(EmbeddingGenerationError::from)?;
    let preferred = classifier
        .select_embedding_model(&requirements)
        .unwrap_or_else(|error| {
            eprintln!(
                "WARNING: pair embedding model selection failed ({error:?}); checking safe fallbacks"
            );
            preferred_auto_embedding_model(sequence_length)
        });

    resolve_auto_embedding_model(preferred, &candidates)
        .ok_or_else(|| no_compatible_auto_model_error(&candidates))
}

fn generate_similarity_embeddings_batch(
    factory: Option<&ModelFactory>,
    texts: &[&str],
    model_type: ModelType,
    target_layer: Option<usize>,
    target_dimension: Option<usize>,
) -> EmbeddingGenerationResult<(Vec<Vec<f32>>, i32)> {
    let embeddings = match model_type {
        ModelType::Qwen3Embedding => generate_qwen3_embeddings_batch(
            require_unified_factory(factory)?,
            texts,
            target_dimension,
        )?,
        ModelType::GemmaEmbedding => generate_gemma_embeddings_batch(
            require_unified_factory(factory)?,
            texts,
            target_dimension,
        )?,
        ModelType::MmBertEmbedding => generate_mmbert_embeddings_batch(
            require_unified_factory(factory)?,
            texts,
            target_layer,
            target_dimension,
        )?,
        ModelType::MultiModalEmbedding => {
            generate_multimodal_text_embeddings_batch(texts, target_layer, target_dimension)?
        }
        _ => return Err(format!("unsupported similarity embedding model: {model_type:?}").into()),
    };
    let model_id = embedding_model_id(model_type)
        .ok_or_else(|| format!("missing metadata id for embedding model: {model_type:?}"))?;
    Ok((embeddings, model_id))
}

fn generate_similarity_embeddings(
    factory: Option<&ModelFactory>,
    text1: &str,
    text2: &str,
    model_type: ModelType,
    target_layer: Option<usize>,
    target_dimension: Option<usize>,
) -> EmbeddingGenerationResult<(Vec<f32>, Vec<f32>, i32)> {
    let (embedding1, embedding2) = match model_type {
        ModelType::Qwen3Embedding => {
            let factory = require_unified_factory(factory)?;
            (
                generate_qwen3_embedding(factory, text1, target_dimension)?.0,
                generate_qwen3_embedding(factory, text2, target_dimension)?.0,
            )
        }
        ModelType::GemmaEmbedding => {
            let factory = require_unified_factory(factory)?;
            (
                generate_gemma_embedding(factory, text1, target_dimension)?.0,
                generate_gemma_embedding(factory, text2, target_dimension)?.0,
            )
        }
        ModelType::MmBertEmbedding => {
            let factory = require_unified_factory(factory)?;
            (
                generate_mmbert_embedding(factory, text1, target_layer, target_dimension)?.0,
                generate_mmbert_embedding(factory, text2, target_layer, target_dimension)?.0,
            )
        }
        ModelType::MultiModalEmbedding => (
            generate_multimodal_text_embedding(text1, target_layer, target_dimension)?.0,
            generate_multimodal_text_embedding(text2, target_layer, target_dimension)?.0,
        ),
        _ => return Err(format!("unsupported similarity embedding model: {model_type:?}").into()),
    };
    let model_id = embedding_model_id(model_type)
        .ok_or_else(|| format!("missing metadata id for embedding model: {model_type:?}"))?;
    Ok((embedding1, embedding2, model_id))
}

fn validate_similarity_embedding_batch_cardinality(
    embeddings_batch: &[Vec<f32>],
    expected_rows: usize,
) -> Result<(), String> {
    if embeddings_batch.len() != expected_rows {
        return Err(format!(
            "expected {expected_rows} similarity embedding rows, got {}",
            embeddings_batch.len()
        ));
    }
    Ok(())
}

/// Calculate cosine similarity between two texts using embeddings
///
/// This function:
/// 1. Generates embeddings for both texts using the specified model (or auto-routing)
/// 2. Calculates cosine similarity between the two embeddings
/// 3. Returns the similarity score along with metadata
///
/// # Parameters
/// - `text1`: First text (C string)
/// - `text2`: Second text (C string)
/// - `model_type_str`: "auto", "qwen3", or "gemma"
/// - `target_dim`: Target dimension (0 for default, or 768/512/256/128)
/// - `result`: Output pointer for similarity result
///
/// # Returns
/// 0 on success, -3 when either tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn calculate_embedding_similarity(
    text1: *const c_char,
    text2: *const c_char,
    model_type_str: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingSimilarityResult,
) -> i32 {
    calculate_embedding_similarity_with_options(
        text1,
        text2,
        model_type_str,
        0,
        target_dim,
        0.5,
        0.5,
        result,
    )
}

/// Calculate pair similarity while honoring the complete public routing
/// contract. The legacy C entry point above delegates here with defaults.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn calculate_embedding_similarity_with_options(
    text1: *const c_char,
    text2: *const c_char,
    model_type_str: *const c_char,
    target_layer: i32,
    target_dim: i32,
    quality_priority: f32,
    latency_priority: f32,
    result: *mut EmbeddingSimilarityResult,
) -> i32 {
    if text1.is_null() || text2.is_null() || model_type_str.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to calculate_embedding_similarity");
        return -1;
    }

    let start_time = std::time::Instant::now();

    // Parse text1
    let text1_str = unsafe {
        match CStr::from_ptr(text1).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text1: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return -1;
            }
        }
    };
    // Parse text2
    let text2_str = unsafe {
        match CStr::from_ptr(text2).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text2: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return -1;
            }
        }
    };

    // Parse model type
    let model_type_str = unsafe {
        match CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return -1;
            }
        }
    };

    // Validate model type
    if model_type_str != "auto"
        && model_type_str != "qwen3"
        && model_type_str != "gemma"
        && model_type_str != "mmbert"
    {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', 'gemma', or 'mmbert')",
            model_type_str
        );
        unsafe {
            (*result) = EmbeddingSimilarityResult::default();
        }
        return -1;
    }
    if !(0.0..=1.0).contains(&quality_priority)
        || !(0.0..=1.0).contains(&latency_priority)
        || (target_layer > 0 && model_type_str != "mmbert")
    {
        eprintln!("Error: invalid similarity routing options");
        unsafe {
            (*result) = EmbeddingSimilarityResult::default();
        }
        return -1;
    }

    let target_layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };
    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let factory = GLOBAL_MODEL_FACTORY.get();

    // Resolve one model for the pair, then reuse that exact embedding space for
    // both operands. Independently auto-routing each operand can produce a
    // numerically plausible but semantically invalid cross-model cosine score.
    let model_type = if model_type_str == "auto" {
        let pair = [text1_str, text2_str];
        match select_auto_embedding_model(
            factory,
            &pair,
            target_dimension,
            quality_priority,
            latency_priority,
        ) {
            Ok(model) => model,
            Err(error) => {
                eprintln!("Error: failed to select pair embedding model: {error}");
                unsafe {
                    (*result) = EmbeddingSimilarityResult::default();
                }
                return error.status();
            }
        }
    } else {
        match model_type_str {
            "qwen3" => ModelType::Qwen3Embedding,
            "gemma" => ModelType::GemmaEmbedding,
            "mmbert" => ModelType::MmBertEmbedding,
            _ => unreachable!("model type was validated above"),
        }
    };

    let (emb1_vec, emb2_vec, model_type_id) = match generate_similarity_embeddings(
        factory,
        text1_str,
        text2_str,
        model_type,
        target_layer,
        target_dimension,
    ) {
        Ok(embeddings) => embeddings,
        Err(error) => {
            eprintln!("Error: failed to generate similarity embeddings: {error}");
            unsafe {
                (*result) = EmbeddingSimilarityResult::default();
            }
            return error.status();
        }
    };

    // Ensure both embeddings have the same dimension
    if emb1_vec.len() != emb2_vec.len() {
        eprintln!(
            "Error: embeddings have different dimensions ({} vs {})",
            emb1_vec.len(),
            emb2_vec.len()
        );
        unsafe {
            (*result) = EmbeddingSimilarityResult::default();
        }
        return -1;
    }

    // Calculate cosine similarity: (A · B) / (||A|| * ||B||)
    let dot_product: f32 = emb1_vec
        .iter()
        .zip(emb2_vec.iter())
        .map(|(a, b)| a * b)
        .sum();
    let norm1: f32 = emb1_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2_vec.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = if norm1 > 0.0 && norm2 > 0.0 {
        dot_product / (norm1 * norm2)
    } else {
        0.0
    };

    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = EmbeddingSimilarityResult {
            similarity,
            model_type: model_type_id,
            processing_time_ms,
            error: false,
        };
    }

    0
}

/// Calculate batch similarity: find top-k most similar candidates for a query
///
/// This function uses TRUE BATCH PROCESSING for optimal performance:
/// 1. Batch tokenizes all texts (query + candidates) together
/// 2. Single forward pass to generate all embeddings
/// 3. Calculates cosine similarity between query and each candidate
/// 4. Returns top-k most similar candidates, sorted by similarity (descending)
///
/// Performance improvement: ~N times faster than loop-based approach (N = num_candidates)
///
/// # Parameters
/// - `query`: Query text (C string)
/// - `candidates`: Array of candidate texts (C string array)
/// - `num_candidates`: Number of candidates
/// - `top_k`: Maximum number of matches to return (0 = return all)
/// - `model_type_str`: "auto", "qwen3", or "gemma"
/// - `target_dim`: Target dimension (0 for default, or 768/512/256/128)
/// - `result`: Output pointer for batch similarity result
///
/// # Returns
/// 0 on success, -3 when any tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn calculate_similarity_batch(
    query: *const c_char,
    candidates: *const *const c_char,
    num_candidates: i32,
    top_k: i32,
    model_type_str: *const c_char,
    target_dim: i32,
    result: *mut BatchSimilarityResult,
) -> i32 {
    calculate_similarity_batch_with_options(
        query,
        candidates,
        num_candidates,
        top_k,
        model_type_str,
        0,
        target_dim,
        0.5,
        0.5,
        result,
    )
}

/// Calculate batch similarity while honoring the complete public routing
/// contract. The legacy C entry point above delegates here with defaults.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn calculate_similarity_batch_with_options(
    query: *const c_char,
    candidates: *const *const c_char,
    num_candidates: i32,
    top_k: i32,
    model_type_str: *const c_char,
    target_layer: i32,
    target_dim: i32,
    quality_priority: f32,
    latency_priority: f32,
    result: *mut BatchSimilarityResult,
) -> i32 {
    if query.is_null() || candidates.is_null() || model_type_str.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to calculate_similarity_batch");
        return -1;
    }

    if num_candidates <= 0 {
        eprintln!("Error: num_candidates must be positive");
        unsafe {
            (*result) = BatchSimilarityResult::default();
        }
        return -1;
    }

    let start_time = std::time::Instant::now();

    // Parse query text
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in query: {}", e);
                (*result) = BatchSimilarityResult::default();
                return -1;
            }
        }
    };

    // Parse model type
    let model_type_str = unsafe {
        match CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = BatchSimilarityResult::default();
                return -1;
            }
        }
    };

    // Validate model type
    if model_type_str != "auto"
        && model_type_str != "qwen3"
        && model_type_str != "gemma"
        && model_type_str != "mmbert"
    {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', 'gemma', or 'mmbert')",
            model_type_str
        );
        unsafe {
            (*result) = BatchSimilarityResult::default();
        }
        return -1;
    }
    if !(0.0..=1.0).contains(&quality_priority)
        || !(0.0..=1.0).contains(&latency_priority)
        || (target_layer > 0 && model_type_str != "mmbert")
    {
        eprintln!("Error: invalid batch similarity routing options");
        unsafe {
            (*result) = BatchSimilarityResult::default();
        }
        return -1;
    }

    // Parse candidate texts
    let mut candidate_texts = Vec::with_capacity(num_candidates as usize);
    for i in 0..num_candidates {
        let candidate_ptr = unsafe { *candidates.offset(i as isize) };
        if candidate_ptr.is_null() {
            eprintln!("Error: null candidate at index {}", i);
            unsafe {
                (*result) = BatchSimilarityResult::default();
            }
            return -1;
        }

        let candidate_str = unsafe {
            match CStr::from_ptr(candidate_ptr).to_str() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in candidate {}: {}", i, e);
                    (*result) = BatchSimilarityResult::default();
                    return -1;
                }
            }
        };
        candidate_texts.push(candidate_str);
    }

    let factory = GLOBAL_MODEL_FACTORY.get();

    // Prepare all texts for batch processing: [query, candidate1, candidate2, ...]
    let mut all_texts: Vec<&str> = Vec::with_capacity(1 + num_candidates as usize);
    all_texts.push(query_str);
    all_texts.extend(candidate_texts.iter().copied());

    let target_layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };
    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let model_type = if model_type_str == "auto" {
        match select_auto_embedding_model(
            factory,
            &all_texts,
            target_dimension,
            quality_priority,
            latency_priority,
        ) {
            Ok(model) => model,
            Err(error) => {
                eprintln!("Error: failed to select batch embedding model: {error}");
                unsafe {
                    (*result) = BatchSimilarityResult::default();
                }
                return error.status();
            }
        }
    } else {
        match model_type_str {
            "qwen3" => ModelType::Qwen3Embedding,
            "gemma" => ModelType::GemmaEmbedding,
            "mmbert" => ModelType::MmBertEmbedding,
            _ => unreachable!("model type was validated above"),
        }
    };

    let (embeddings_batch, model_type_id) = match generate_similarity_embeddings_batch(
        factory,
        &all_texts,
        model_type,
        target_layer,
        target_dimension,
    ) {
        Ok(generated) => generated,
        Err(error) => {
            eprintln!("Error: batch embedding generation failed: {error}");
            unsafe {
                (*result) = BatchSimilarityResult::default();
            }
            return error.status();
        }
    };

    // Require one output row for the query and one for every candidate before
    // indexing or returning candidate identities across the C ABI boundary.
    let expected_rows = 1 + num_candidates as usize;
    if let Err(error) =
        validate_similarity_embedding_batch_cardinality(&embeddings_batch, expected_rows)
    {
        eprintln!("Error: batch similarity contract mismatch: {error}");
        unsafe {
            (*result) = BatchSimilarityResult::default();
        }
        return -1;
    }

    let query_embedding = &embeddings_batch[0];

    // Calculate similarities with all candidates
    let mut similarities = Vec::with_capacity(num_candidates as usize);

    for (idx, candidate_embedding) in embeddings_batch[1..].iter().enumerate() {
        // Ensure dimensions match
        if query_embedding.len() != candidate_embedding.len() {
            eprintln!(
                "Error: dimension mismatch at candidate {} ({} vs {})",
                idx,
                query_embedding.len(),
                candidate_embedding.len()
            );
            unsafe {
                (*result) = BatchSimilarityResult::default();
            }
            return -1;
        }

        // Calculate cosine similarity
        let dot_product: f32 = query_embedding
            .iter()
            .zip(candidate_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_query: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_candidate: f32 = candidate_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        let similarity = if norm_query > 0.0 && norm_candidate > 0.0 {
            dot_product / (norm_query * norm_candidate)
        } else {
            0.0
        };

        similarities.push((idx, similarity));
    }

    // Sort by similarity (descending)
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top-k
    let k = if top_k <= 0 || top_k > num_candidates {
        num_candidates as usize
    } else {
        top_k as usize
    };
    let top_matches: Vec<SimilarityMatch> = similarities
        .iter()
        .take(k)
        .map(|(idx, sim)| SimilarityMatch {
            index: *idx as i32,
            similarity: *sim,
        })
        .collect();

    let num_matches = top_matches.len() as i32;
    let matches_ptr = Box::into_raw(top_matches.into_boxed_slice()) as *mut SimilarityMatch;

    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = BatchSimilarityResult {
            matches: matches_ptr,
            num_matches,
            model_type: model_type_id,
            processing_time_ms,
            error: false,
        };
    }

    0
}

/// Free batch similarity result
///
/// This function should be called to release memory allocated for batch similarity matching.
///
/// # Parameters
/// - `result`: Pointer to the BatchSimilarityResult to free
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_batch_similarity_result(result: *mut BatchSimilarityResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let batch_result = &mut *result;

        // Free the matches array if it's not null. The array was allocated with
        // `into_boxed_slice()`, so it must be reclaimed as a `Box<[SimilarityMatch]>`
        // over the full length. Reconstructing a `Box<SimilarityMatch>` from the
        // element pointer frees with the layout of a single element and is undefined
        // behavior whenever num_matches > 1.
        if !batch_result.matches.is_null() && batch_result.num_matches > 0 {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                batch_result.matches,
                batch_result.num_matches as usize,
            ));
        }

        // Reset the result
        batch_result.matches = std::ptr::null_mut();
        batch_result.num_matches = 0;
    }
}

/// Get information about loaded embedding models
///
/// This function returns metadata about all available embedding models,
/// including their loading status, capabilities, and configuration.
///
/// # Parameters
/// - `result`: Output pointer for models information result
///
/// # Returns
/// 0 on success, -1 on error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn get_embedding_models_info(
    result: *mut crate::ffi::types::EmbeddingModelsInfoResult,
) -> i32 {
    use crate::ffi::types::{EmbeddingModelInfo, EmbeddingModelsInfoResult};
    use std::ffi::CString;

    if result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_models_info");
        return -1;
    }

    // Get global model factory
    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe {
                (*result) = EmbeddingModelsInfoResult::default();
            }
            return -1;
        }
    };

    // Check which models are loaded
    let qwen3_loaded = factory.get_qwen3_model().is_some();
    let gemma_loaded = factory.get_gemma_model().is_some();

    // Get model paths from factory
    let qwen3_path = factory.get_qwen3_model_path();
    let gemma_path = factory.get_gemma_model_path();

    // Create model info array
    let mut models_vec = Vec::new();

    // Qwen3 model info
    {
        let model_name = CString::new("qwen3").unwrap();
        let model_path = if let Some(path) = qwen3_path {
            CString::new(path).unwrap()
        } else {
            CString::new("").unwrap()
        };

        models_vec.push(EmbeddingModelInfo {
            model_name: model_name.into_raw(),
            is_loaded: qwen3_loaded,
            max_sequence_length: if qwen3_loaded {
                QWEN3_EMBEDDING_CONTEXT as i32
            } else {
                0
            },
            default_dimension: if qwen3_loaded { 1024 } else { 0 },
            model_path: model_path.into_raw(),
        });
    }

    // Gemma model info
    {
        let model_name = CString::new("gemma").unwrap();
        let model_path = if let Some(path) = gemma_path {
            CString::new(path).unwrap()
        } else {
            CString::new("").unwrap()
        };

        models_vec.push(EmbeddingModelInfo {
            model_name: model_name.into_raw(),
            is_loaded: gemma_loaded,
            max_sequence_length: if gemma_loaded {
                GEMMA_EMBEDDING_CONTEXT as i32
            } else {
                0
            },
            default_dimension: if gemma_loaded { 768 } else { 0 },
            model_path: model_path.into_raw(),
        });
    }

    let num_models = models_vec.len() as i32;
    let models_ptr = Box::into_raw(models_vec.into_boxed_slice()) as *mut EmbeddingModelInfo;

    unsafe {
        (*result) = EmbeddingModelsInfoResult {
            models: models_ptr,
            num_models,
            error: false,
        };
    }

    0
}

/// Free embedding models info result
///
/// This function should be called to release memory allocated for models information.
///
/// # Parameters
/// - `result`: Pointer to the EmbeddingModelsInfoResult to free
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_embedding_models_info(
    result: *mut crate::ffi::types::EmbeddingModelsInfoResult,
) {
    use std::ffi::CString;

    if result.is_null() {
        return;
    }

    unsafe {
        let info_result = &mut *result;

        // Free each model info
        if !info_result.models.is_null() && info_result.num_models > 0 {
            let models_slice =
                std::slice::from_raw_parts_mut(info_result.models, info_result.num_models as usize);

            for model_info in models_slice.iter_mut() {
                // Free model_name string
                if !model_info.model_name.is_null() {
                    let _ = CString::from_raw(model_info.model_name);
                }
                // Free model_path string
                if !model_info.model_path.is_null() {
                    let _ = CString::from_raw(model_info.model_path);
                }
            }

            // Free the models array as the boxed slice it was allocated as. The
            // element-pointer form would free with single-element layout while
            // num_models is always 2, so reclaim the full-length slice.
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                info_result.models,
                info_result.num_models as usize,
            ));
        }

        // Reset the result
        info_result.models = std::ptr::null_mut();
        info_result.num_models = 0;
    }
}

// ============================================================================
// Continuous Batching FFI Functions
// ============================================================================

use crate::model_architectures::embedding::continuous_batch_scheduler::ContinuousBatchConfig;
use crate::model_architectures::embedding::qwen3_batched::Qwen3EmbeddingModelBatched;
use crate::model_architectures::embedding::qwen3_embedding::Qwen3EmbeddingModel;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Batched model with tokenizer and device info
struct BatchedModelContext {
    model: Arc<Qwen3EmbeddingModelBatched>,
    tokenizer: Arc<Mutex<Tokenizer>>, // Tokenizer needs mutex (not thread-safe)
    device: candle_core::Device,
}

/// Global singleton for batched model
static GLOBAL_BATCHED_MODEL: OnceLock<BatchedModelContext> = OnceLock::new();

/// Initialize Qwen3 embedding model with continuous batching
///
/// This function loads the Qwen3 model and wraps it with continuous batching scheduler.
///
/// # Parameters
/// - `qwen3_model_path`: Path to Qwen3 model directory (C string)
/// - `max_batch_size`: Maximum batch size for continuous batching
/// - `max_wait_ms`: Maximum wait time in milliseconds before processing a batch
/// - `use_cpu`: Whether to use CPU instead of GPU
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed or already initialized
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_embedding_models_batched(
    qwen3_model_path: *const c_char,
    max_batch_size: i32,
    max_wait_ms: u64,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    // Check if already initialized
    if GLOBAL_BATCHED_MODEL.get().is_some() {
        eprintln!("Warning: batched embedding model already initialized");
        return true;
    }

    // Parse model path
    let model_path = if qwen3_model_path.is_null() {
        eprintln!("Error: qwen3_model_path is null");
        return false;
    } else {
        unsafe {
            match CStr::from_ptr(qwen3_model_path).to_str() {
                Ok(s) if !s.is_empty() => s.to_string(),
                _ => {
                    eprintln!("Error: invalid qwen3_model_path");
                    return false;
                }
            }
        }
    };

    // Determine device
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    // Load tokenizer
    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = match Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("failed to load tokenizer: {error:?}"))
        .and_then(|tokenizer| {
            prepare_embedding_tokenizer(tokenizer, "Qwen3 batched")
                .map_err(|error| error.to_string())
        }) {
        Ok(tokenizer) => tokenizer,
        Err(e) => {
            eprintln!(
                "ERROR: Failed to load tokenizer from {}: {:?}",
                tokenizer_path, e
            );
            return false;
        }
    };

    // Load base model
    let base_model = match Qwen3EmbeddingModel::load(&model_path, &device) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("ERROR: Failed to load Qwen3 model: {:?}", e);
            return false;
        }
    };

    // Create continuous batch config
    let batch_config = ContinuousBatchConfig {
        max_batch_size: max_batch_size as usize,
        max_wait_time_ms: max_wait_ms,
        min_batch_size: 1,
        enable_dynamic: true,
        max_seq_len_diff: 512,
        verbose: false,
    };

    // Wrap with continuous batching
    println!(
        "Initializing continuous batching (max_batch={}, max_wait={}ms)",
        max_batch_size, max_wait_ms
    );
    let batched_model = Qwen3EmbeddingModelBatched::from_model(base_model, batch_config);

    // Create context with tokenizer (wrap in Arc for concurrent access)
    let context = BatchedModelContext {
        model: Arc::new(batched_model),
        tokenizer: Arc::new(Mutex::new(tokenizer)),
        device: device.clone(),
    };

    // Store in global singleton (no outer Mutex needed - Arc handles concurrency)
    if GLOBAL_BATCHED_MODEL.set(context).is_err() {
        eprintln!("ERROR: Failed to set global batched model");
        return false;
    }

    println!("INFO: Batched embedding model initialized successfully");
    true
}

/// Get embedding using continuous batching model
///
/// # Safety
/// - `text` must be a valid null-terminated C string
/// - `model_type` must be a valid null-terminated C string (currently only "qwen3" supported)
/// - `result` must be a valid pointer to EmbeddingResult
///
/// # Returns
/// 0 on success, -3 when the tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn get_embedding_batched(
    text: *const c_char,
    model_type: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || model_type.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_batched");
        return -1;
    }

    // Parse text
    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    // Parse model type (currently only qwen3 supported)
    let model_type_str = unsafe {
        match CStr::from_ptr(model_type).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    if model_type_str != "qwen3" {
        eprintln!(
            "Error: unsupported model type '{}' for batched embeddings (only 'qwen3' supported)",
            model_type_str
        );
        unsafe {
            (*result) = create_error_result();
        }
        return -1;
    }

    // Get batched model context (OnceLock - no lock needed!)
    let batched_context = match GLOBAL_BATCHED_MODEL.get() {
        Some(ctx) => ctx,
        None => {
            eprintln!("Error: batched embedding model not initialized. Call init_embedding_models_batched first.");
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    // Clone Arc references for concurrent access
    let tokenizer = Arc::clone(&batched_context.tokenizer);
    let model = Arc::clone(&batched_context.model);

    // Tokenize (brief lock on tokenizer only)
    let (ids_raw, mask_raw, seq_len) = {
        let tokenizer_guard = match tokenizer.lock() {
            Ok(guard) => guard,
            Err(e) => {
                eprintln!("Error: failed to lock tokenizer: {}", e);
                unsafe {
                    (*result) = create_error_result();
                }
                return -1;
            }
        };

        let encoding = match encode_embedding_checked(
            &tokenizer_guard,
            text_str,
            QWEN3_EMBEDDING_CONTEXT,
            "Qwen3 batched",
        ) {
            Ok(encoding) => encoding,
            Err(e) => {
                eprintln!("Error: tokenization failed: {}", e);
                unsafe {
                    (*result) = create_error_result();
                }
                return EmbeddingGenerationError::from(e).status();
            }
        };

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = encoding.len();

        // Tokenizer lock released here - can now process concurrently!
        (ids, mask, seq_len)
    };

    // Generate embedding using continuous batching
    // NO LOCK HELD - multiple requests execute concurrently!
    // Model is Arc-wrapped and internally thread-safe via channels
    let embedding_vec = match model.embedding_forward_from_raw(ids_raw, mask_raw) {
        Ok(vec) => vec,
        Err(e) => {
            eprintln!("Error: embedding generation failed: {:?}", e);
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };
    let final_embedding = truncate_embedding_to_dimension(embedding_vec, target_dimension);

    let length = final_embedding.len() as i32;
    let data = Box::into_raw(final_embedding.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = EmbeddingResult {
            data,
            length,
            error: false,
            model_type: 0, // Qwen3
            sequence_length: seq_len as i32,
            processing_time_ms,
        };
    }

    0
}

// ============================================================================
// Multi-Modal Embedding FFI Functions
// ============================================================================

/// Initialize multi-modal embedding model (text + image + audio)
///
/// Model: llm-semantic-router/multi-modal-embed-small
/// - Text: MiniLM-L6-v2 (22M params, 384-dim)
/// - Image: SigLIP-base-patch16-512 (86M params, 768→384 projection)
/// - Audio: Whisper-tiny encoder (8M params, 384-dim)
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_multimodal_embedding_model(
    model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    let path = match parse_optional_model_path(model_path, "multimodal model path") {
        Ok(Some(path)) => path,
        Ok(None) => {
            eprintln!("ERROR: multimodal model path is required");
            return false;
        }
        Err(error) => {
            eprintln!("ERROR: {error}");
            return false;
        }
    };

    let _guard = EMBEDDING_INIT_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // A legacy process may already hold multimodal state in the unified
    // factory. Treat it as idempotent only when it is the exact requested path.
    if let Some(factory) = GLOBAL_MODEL_FACTORY.get() {
        if factory.get_multimodal_model().is_some() {
            return validate_requested_model(
                "multimodal",
                Some(&path),
                true,
                factory.get_multimodal_model_path(),
            )
            .map(|()| true)
            .unwrap_or_else(|error| {
                eprintln!("ERROR: {error}");
                false
            });
        }
    }

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    // Multimodal state is always published independently. It therefore cannot
    // win the unified factory's OnceLock and make a concurrent unified
    // initializer falsely report success without its requested models.
    let result = initialize_once_with_validation(
        &STANDALONE_MULTIMODAL,
        || {
            let model = MultiModalEmbeddingModel::load(&path, &device)
                .map_err(|error| format!("failed to load multimodal model: {error:?}"))?;
            let tokenizer_path = format!("{path}/tokenizer.json");
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|error| {
                    format!("failed to load multimodal tokenizer from {tokenizer_path}: {error:?}")
                })
                .and_then(|tokenizer| {
                    prepare_embedding_tokenizer(tokenizer, "multimodal")
                        .map_err(|error| error.to_string())
                })?;
            Ok((model, tokenizer, path.clone()))
        },
        |(_, _, loaded_path)| {
            if loaded_path == &path {
                Ok(())
            } else {
                Err(
                    "requested multimodal model path does not match the process-global model"
                        .to_string(),
                )
            }
        },
    );
    match result {
        Ok(()) => true,
        Err(error) => {
            eprintln!("ERROR: multimodal model initialization failed: {error}");
            false
        }
    }
}

/// Encode text using the multi-modal embedding model
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `target_dim`: Target dimension (0 for default 384, or 32/64/128/256)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -3 when the tokenizer context is exceeded, -1 on internal error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_text(
    text: *const c_char,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_text");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = MultiModalEmbeddingResult::default();
                return -1;
            }
        }
    };

    let (model, tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Tokenize without tokenizer-configured truncation, then enforce context.
    let encoding = match encode_embedding_checked(
        tokenizer,
        text_str,
        MULTIMODAL_TEXT_CONTEXT,
        "multimodal",
    ) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: tokenization failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return EmbeddingGenerationError::from(e).status();
        }
    };

    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
    let seq_len = ids.len();

    let device = model.device();
    let input_ids = match candle_core::Tensor::from_vec(ids, (1, seq_len), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: tensor creation failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };
    let attention_mask = match candle_core::Tensor::from_vec(mask, (1, seq_len), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: tensor creation failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let embedding = match model.encode_text_with_matryoshka(
        &input_ids,
        Some(&attention_mask),
        None,
        target_dimension,
    ) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: text encoding failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 0, // text
            processing_time_ms,
        };
    }

    0
}

/// Encode image bytes: decode + resize (Catmull-Rom cubic, support-weighted) +
/// forward.
///
/// Preferred entry point for image embedding from raw JPEG/PNG bytes. All
/// preprocessing happens in Rust via `decode_resize_to_chw_f32`, which
/// approximates PIL's `Image.BICUBIC` + `antialias=True` behavior used by
/// `SiglipProcessor`. Validated end-to-end against the PyTorch reference in
/// `docs/probe-2026-05-25-image-drift-isolation/`: cosine >= 0.999 across a
/// 20-image corpus; the prior Go-side 4-tap bilinear path averaged cosine
/// 0.99 on the same fixtures.
///
/// See `decode_resize_to_chw_f32` for the resize-filter discussion, known
/// limitations (RGBA alpha discard, no EXIF auto-apply), and rationale.
///
/// # Parameters
/// - `bytes_ptr`: Pointer to raw JPEG/PNG bytes
/// - `bytes_len`: Number of bytes
/// - `target_dim`: Target embedding dimension (0 for default 384)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// - `0` on success
/// - `-2` when the raw image input is empty, oversized, or cannot be decoded
/// - `-1` for internal failures (invalid output pointer, unavailable model,
///   tensor construction, model forward pass, or result conversion)
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_image_from_bytes(
    bytes_ptr: *const u8,
    bytes_len: usize,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if result.is_null() {
        eprintln!("Error: null result pointer in multimodal_encode_image_from_bytes");
        return -1;
    }
    unsafe {
        (*result) = MultiModalEmbeddingResult::default();
    }

    if bytes_ptr.is_null() || bytes_len == 0 {
        eprintln!("Error: null/empty input to multimodal_encode_image_from_bytes");
        return -2;
    }
    if bytes_len > MAX_MULTIMODAL_IMAGE_ENCODED_BYTES {
        eprintln!("Error: encoded image exceeds {MAX_MULTIMODAL_IMAGE_ENCODED_BYTES} bytes");
        return -2;
    }

    let start_time = std::time::Instant::now();
    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };

    const TARGET_W: u32 = 512;
    const TARGET_H: u32 = 512;
    let pixels = match decode_resize_to_chw_f32(bytes, TARGET_W, TARGET_H) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return -2;
        }
    };

    // Decode before looking up the model so malformed client input is always
    // classified as invalid input, even when the service has not initialized
    // the multimodal model yet.
    let (model, _tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            return -1;
        }
    };

    let h = TARGET_H as usize;
    let w = TARGET_W as usize;
    let device = model.device();
    let pixel_tensor = match candle_core::Tensor::from_slice(&pixels, (1, 3, h, w), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: pixel tensor creation failed: {:?}", e);
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding = match model.encode_image_with_dim(&pixel_tensor, target_dimension) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: image encoding failed: {:?}", e);
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 1, // image
            processing_time_ms,
        };
    }

    0
}

/// Encode image pixel data using the multi-modal embedding model.
///
/// # Parameters
/// - `pixel_data`: Raw pixel data as float array (RGB, normalized [0, 1]), shape [3 * H * W]
/// - `height`: Image height (must be 512 for SigLIP-base-patch16-512)
/// - `width`: Image width (must be 512)
/// - `target_dim`: Target dimension (0 for default 384)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_image(
    pixel_data: *const f32,
    height: i32,
    width: i32,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if pixel_data.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_image");
        return -1;
    }

    unsafe {
        *result = MultiModalEmbeddingResult::default();
    }
    if height <= 0 || width <= 0 {
        eprintln!(
            "Error: image dimensions must be positive (got {}x{})",
            height, width
        );
        return -1;
    }
    let h = height as usize;
    let w = width as usize;
    const MAX_IMAGE_SIDE: usize = 8192;
    const MAX_IMAGE_PIXELS: usize = 16_777_216;
    if h > MAX_IMAGE_SIDE || w > MAX_IMAGE_SIDE || h.saturating_mul(w) > MAX_IMAGE_PIXELS {
        eprintln!("Error: image dimensions exceed the native input budget");
        return -1;
    }
    let pixel_count = match 3usize.checked_mul(h).and_then(|value| value.checked_mul(w)) {
        Some(count) if count <= (isize::MAX as usize) / std::mem::size_of::<f32>() => count,
        _ => {
            eprintln!("Error: image tensor dimensions overflow addressable memory");
            return -1;
        }
    };

    let (model, _tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    let pixels = unsafe { std::slice::from_raw_parts(pixel_data, pixel_count) };
    let device = model.device();

    let pixel_tensor = match candle_core::Tensor::from_slice(pixels, (1, 3, h, w), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: pixel tensor creation failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding = match model.encode_image_with_dim(&pixel_tensor, target_dimension) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: image encoding failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 1, // image
            processing_time_ms,
        };
    }

    0
}

/// Encode audio mel-spectrogram using the multi-modal embedding model
///
/// # Parameters
/// - `mel_data`: Mel spectrogram float array, shape [n_mels * time_frames]
/// - `n_mels`: Number of mel bins (typically 80)
/// - `time_frames`: Number of time frames
/// - `target_dim`: Target dimension (0 for default 384)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_audio(
    mel_data: *const f32,
    n_mels: i32,
    time_frames: i32,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if mel_data.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_audio");
        return -1;
    }

    unsafe {
        *result = MultiModalEmbeddingResult::default();
    }
    if n_mels <= 0 || time_frames <= 0 {
        eprintln!(
            "Error: n_mels and time_frames must be positive (got {}x{})",
            n_mels, time_frames
        );
        return -1;
    }
    let mels = n_mels as usize;
    let frames = time_frames as usize;
    let total = match mels.checked_mul(frames) {
        Some(total) if total <= (isize::MAX as usize) / std::mem::size_of::<f32>() => total,
        _ => {
            eprintln!(
                "Error: mel spectrogram dimensions overflow addressable memory ({}x{})",
                n_mels, time_frames
            );
            return -1;
        }
    };

    let (model, _tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    let mel_slice = unsafe { std::slice::from_raw_parts(mel_data, total) };
    let device = model.device();

    let mel_tensor = match candle_core::Tensor::from_slice(mel_slice, (1, mels, frames), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: mel tensor creation failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding = match model.encode_audio_with_dim(&mel_tensor, target_dimension) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: audio encoding failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            unsafe {
                (*result) = MultiModalEmbeddingResult::default();
            }
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 2, // audio
            processing_time_ms,
        };
    }

    0
}

/// Free multi-modal embedding result data
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_multimodal_embedding(data: *mut f32, length: i32) {
    if data.is_null() || length <= 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, length as usize));
    }
}

/// Shutdown the continuous batching scheduler
///
/// This function should be called when you're done using the batched model
/// to properly clean up resources and stop the background scheduler thread.
#[no_mangle]
pub extern "C" fn shutdown_embedding_batched() {
    // The scheduler thread will automatically stop when the model is dropped
    // This is handled by the Drop implementation of Qwen3EmbeddingModelBatched
    println!("INFO: Shutting down batched embedding model");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::image_input::{
        test_support::{make_png_with_declared_dimensions, make_test_png},
        MAX_MULTIMODAL_IMAGE_DIMENSION, MAX_MULTIMODAL_IMAGE_ENCODED_BYTES,
        MAX_MULTIMODAL_IMAGE_PIXELS,
    };
    use crate::model_architectures::embedding::tokenizer_contract::validate_embedding_token_count;
    use std::sync::{Arc, Barrier};
    use tokenizers::models::wordpiece::WordPiece;

    fn subword_tokenizer() -> MmTokenizer {
        let vocabulary = [
            ("[UNK]".to_string(), 0),
            ("a".to_string(), 1),
            ("##a".to_string(), 2),
        ];
        let model = WordPiece::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".to_string())
            .max_input_chars_per_word(QWEN3_EMBEDDING_CONTEXT + 1)
            .build()
            .expect("build subword tokenizer");
        prepare_embedding_tokenizer(MmTokenizer::new(model), "test-subword")
            .expect("prepare subword tokenizer")
    }

    #[test]
    fn embedding_token_context_boundaries_are_fail_closed() {
        for (model, limit) in [
            ("Qwen3", QWEN3_EMBEDDING_CONTEXT),
            ("mmBERT", MMBERT_EMBEDDING_CONTEXT),
            ("Gemma", GEMMA_EMBEDDING_CONTEXT),
            ("multimodal", MULTIMODAL_TEXT_CONTEXT),
        ] {
            assert!(validate_embedding_token_count(limit, limit, model).is_ok());
            let error = validate_embedding_token_count(limit + 1, limit, model).unwrap_err();
            assert!(error.is_input_too_long());
            assert_eq!(
                EmbeddingGenerationError::from(error).status(),
                EMBEDDING_INPUT_TOO_LONG_STATUS
            );
        }
        let empty = validate_embedding_token_count(0, QWEN3_EMBEDDING_CONTEXT, "Qwen3")
            .expect_err("zero tokens must fail");
        assert!(!empty.is_input_too_long());
        assert_eq!(EmbeddingGenerationError::from(empty).status(), -1);
    }

    #[test]
    fn auto_routing_uses_true_subword_length_without_whitespace() {
        let tokenizer = subword_tokenizer();
        let long_text = "a".repeat(GEMMA_EMBEDDING_CONTEXT + 1);
        assert_eq!(long_text.split_whitespace().count(), 1);

        let exact = routing_sequence_length(&tokenizer, &["a", &long_text], "test-subword")
            .expect("measure true routing length");

        assert_eq!(exact, GEMMA_EMBEDDING_CONTEXT + 1);
        assert_eq!(
            resolve_auto_embedding_model(
                ModelType::GemmaEmbedding,
                &[
                    AutoEmbeddingCandidate {
                        model_type: ModelType::GemmaEmbedding,
                        sequence_length: exact,
                    },
                    AutoEmbeddingCandidate {
                        model_type: ModelType::Qwen3Embedding,
                        sequence_length: exact,
                    },
                ],
            ),
            Some(ModelType::Qwen3Embedding)
        );
    }

    #[test]
    fn auto_routing_fails_with_typed_length_error_when_every_candidate_is_too_long() {
        let candidates = [
            AutoEmbeddingCandidate {
                model_type: ModelType::GemmaEmbedding,
                sequence_length: GEMMA_EMBEDDING_CONTEXT + 1,
            },
            AutoEmbeddingCandidate {
                model_type: ModelType::MultiModalEmbedding,
                sequence_length: MULTIMODAL_TEXT_CONTEXT + 1,
            },
        ];

        assert_eq!(
            no_compatible_auto_model_error(&candidates).status(),
            EMBEDDING_INPUT_TOO_LONG_STATUS
        );
    }

    #[test]
    fn once_publication_is_retryable_after_build_failure() {
        let slot = OnceLock::new();
        let failed = initialize_once_with_validation(
            &slot,
            || Err::<usize, _>("injected model load failure".to_string()),
            |_| Ok(()),
        );
        assert!(failed.is_err());
        assert!(slot.get().is_none());

        initialize_once_with_validation(
            &slot,
            || Ok(7),
            |value| {
                (*value == 7)
                    .then_some(())
                    .ok_or_else(|| "wrong model identity".to_string())
            },
        )
        .expect("retry succeeds");
        assert_eq!(slot.get(), Some(&7));
    }

    #[test]
    fn existing_once_publication_is_not_false_success_for_another_request() {
        let slot = OnceLock::from("models/first".to_string());
        let result = initialize_once_with_validation(
            &slot,
            || Ok("models/second".to_string()),
            |loaded_path| {
                (loaded_path == "models/second")
                    .then_some(())
                    .ok_or_else(|| "requested model path does not match".to_string())
            },
        );

        assert!(result.is_err());
        assert_eq!(slot.get().map(String::as_str), Some("models/first"));
    }

    #[test]
    fn concurrent_once_publication_never_reports_an_unvalidated_winner() {
        let slot = OnceLock::new();
        let barrier = Arc::new(Barrier::new(8));
        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _ in 0..8 {
                let barrier = Arc::clone(&barrier);
                let slot = &slot;
                handles.push(scope.spawn(move || {
                    barrier.wait();
                    initialize_once_with_validation(
                        slot,
                        || Ok(42),
                        |value| {
                            (*value == 42)
                                .then_some(())
                                .ok_or_else(|| "unexpected published value".to_string())
                        },
                    )
                }));
            }
            for handle in handles {
                handle
                    .join()
                    .expect("initializer thread")
                    .expect("valid winner");
            }
        });
        assert_eq!(slot.get(), Some(&42));
    }

    #[test]
    fn similarity_batch_requires_query_and_every_candidate_row() {
        let complete = vec![vec![1.0], vec![0.5], vec![0.25]];
        assert!(validate_similarity_embedding_batch_cardinality(&complete, 3).is_ok());
        assert!(validate_similarity_embedding_batch_cardinality(&complete[..2], 3).is_err());

        let mut extra = complete;
        extra.push(vec![0.0]);
        assert!(validate_similarity_embedding_batch_cardinality(&extra, 3).is_err());
    }

    #[test]
    fn batch_similarity_rejects_null_model_type_before_c_string_parse() {
        let query = std::ffi::CString::new("query").unwrap();
        let candidate = std::ffi::CString::new("candidate").unwrap();
        let candidates = [candidate.as_ptr()];
        let mut result = crate::ffi::types::BatchSimilarityResult::default();

        let status = calculate_similarity_batch(
            query.as_ptr(),
            candidates.as_ptr(),
            1,
            1,
            std::ptr::null(),
            0,
            &mut result,
        );

        assert_eq!(status, -1);
        assert!(result.error);
    }

    #[test]
    fn multimodal_image_bytes_reports_invalid_input_before_model_lookup() {
        let garbage = b"not a real image file";
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status =
            multimodal_encode_image_from_bytes(garbage.as_ptr(), garbage.len(), 0, &mut result);

        assert_eq!(status, -2, "undecodable bytes must be invalid input");
        assert!(result.error);
        assert!(result.data.is_null());
        assert_eq!(result.length, 0);
    }

    #[test]
    fn multimodal_image_bytes_rejects_truncated_png_before_model_lookup() {
        let mut truncated = make_test_png(2, 2, [1, 2, 3]);
        truncated.truncate(truncated.len() - 8);
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status =
            multimodal_encode_image_from_bytes(truncated.as_ptr(), truncated.len(), 0, &mut result);

        assert_eq!(status, -2, "truncated PNG must be invalid input");
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_bytes_rejects_gif_and_webp_before_model_lookup() {
        let inputs: [(&str, &[u8]); 2] = [
            ("GIF", b"GIF89a\x01\x00\x01\x00"),
            ("WebP", b"RIFF\x04\x00\x00\x00WEBP"),
        ];

        for (format, bytes) in inputs {
            let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();
            let status =
                multimodal_encode_image_from_bytes(bytes.as_ptr(), bytes.len(), 0, &mut result);

            assert_eq!(status, -2, "{format} must be rejected as invalid input");
            assert!(result.error);
            assert!(result.data.is_null());
        }
    }

    #[test]
    fn multimodal_image_bytes_rejects_oversized_dimensions_before_model_lookup() {
        let oversized = make_png_with_declared_dimensions(MAX_MULTIMODAL_IMAGE_DIMENSION + 1, 1);
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status =
            multimodal_encode_image_from_bytes(oversized.as_ptr(), oversized.len(), 0, &mut result);

        assert_eq!(status, -2, "oversized dimensions must be invalid input");
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_bytes_rejects_oversized_pixel_area_before_model_lookup() {
        let side = (MAX_MULTIMODAL_IMAGE_PIXELS as f64).sqrt() as u32 + 1;
        assert!(side <= MAX_MULTIMODAL_IMAGE_DIMENSION);
        let oversized = make_png_with_declared_dimensions(side, side);
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status =
            multimodal_encode_image_from_bytes(oversized.as_ptr(), oversized.len(), 0, &mut result);

        assert_eq!(status, -2, "oversized pixel area must be invalid input");
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_bytes_rejects_oversized_encoded_input_before_reading_it() {
        let one_byte = 0_u8;
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status = multimodal_encode_image_from_bytes(
            &one_byte,
            MAX_MULTIMODAL_IMAGE_ENCODED_BYTES + 1,
            0,
            &mut result,
        );

        assert_eq!(status, -2, "oversized encoded bytes must be invalid input");
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_bytes_reports_unavailable_model_as_internal() {
        assert!(
            get_multimodal_refs().is_none(),
            "this contract test requires no multimodal model fixture"
        );
        let valid = make_test_png(1, 1, [1, 2, 3]);
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let status =
            multimodal_encode_image_from_bytes(valid.as_ptr(), valid.len(), 0, &mut result);

        assert_eq!(status, -1, "an unavailable model is an internal failure");
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_bytes_reports_empty_input_separately_from_internal_errors() {
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let empty_status = multimodal_encode_image_from_bytes(std::ptr::null(), 0, 0, &mut result);
        let null_result_status =
            multimodal_encode_image_from_bytes(std::ptr::null(), 0, 0, std::ptr::null_mut());

        assert_eq!(empty_status, -2, "empty image bytes are invalid input");
        assert_eq!(null_result_status, -1, "a null output pointer is internal");
    }

    #[test]
    fn multimodal_audio_rejects_invalid_dimensions_before_slice_creation() {
        let one_value = 0_f32;
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let negative = multimodal_encode_audio(&one_value, -1, -1, 0, &mut result);
        assert_eq!(negative, -1);
        assert!(result.error);
        assert!(result.data.is_null());

        let overflowing = multimodal_encode_audio(&one_value, i32::MAX, i32::MAX, 0, &mut result);
        assert_eq!(overflowing, -1);
        assert!(result.error);
        assert!(result.data.is_null());
    }

    #[test]
    fn multimodal_image_rejects_invalid_dimensions_before_slice_creation() {
        let one_value = 0_f32;
        let mut result = crate::ffi::types::MultiModalEmbeddingResult::default();

        let negative = multimodal_encode_image(&one_value, -1, -1, 0, &mut result);
        assert_eq!(negative, -1);
        assert!(result.error);
        assert!(result.data.is_null());

        let oversized = multimodal_encode_image(&one_value, 8193, 1, 0, &mut result);
        assert_eq!(oversized, -1);
        assert!(result.error);
        assert!(result.data.is_null());
    }
}
