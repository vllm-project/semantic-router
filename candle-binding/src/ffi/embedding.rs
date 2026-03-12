//! Embedding Generation FFI Module
//!
//! This module provides Foreign Function Interface (FFI) functions for
//! intelligent embedding generation with automatic model selection.

use crate::model_architectures::model_factory::ModelFactory;
use std::ffi::{c_char, CStr};
use std::sync::OnceLock;

// ============================================================================
// Refactoring: Shared embedding generation logic
// ============================================================================

/// Padding direction for tokenized sequences
#[derive(Clone, Copy, Debug)]
pub(crate) enum PaddingSide {
    /// Left padding (Qwen3)
    Left,
    /// Right padding (Gemma)
    Right,
}

/// Global singleton for ModelFactory
pub(crate) static GLOBAL_MODEL_FACTORY: OnceLock<ModelFactory> = OnceLock::new();

use crate::model_architectures::embedding::MultiModalEmbeddingModel;
use tokenizers::Tokenizer as MmTokenizer;

/// Standalone multimodal model storage — allows initialization independent of the
/// main ModelFactory (which uses OnceLock and can only be set once).
pub(crate) static STANDALONE_MULTIMODAL: OnceLock<(MultiModalEmbeddingModel, MmTokenizer, String)> =
    OnceLock::new();

/// Get a reference to the multimodal model + tokenizer, checking standalone first
/// then falling back to the factory.
pub(crate) fn get_multimodal_refs(
) -> Option<(&'static MultiModalEmbeddingModel, &'static MmTokenizer)> {
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
pub(crate) fn generate_embedding_internal<'a, F, G>(
    text: &str,
    target_dim: Option<usize>,
    get_tokenizer: G,
    forward_fn: F,
) -> Result<Vec<f32>, String>
where
    F: Fn(Vec<u32>, Vec<u32>) -> Result<candle_core::Tensor, String>,
    G: Fn() -> Option<&'a tokenizers::Tokenizer>,
{
    // Get tokenizer
    let tokenizer = get_tokenizer().ok_or_else(|| "Tokenizer not available".to_string())?;

    // Tokenize single text
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization failed: {:?}", e))?;

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

    // Apply Matryoshka truncation if requested
    let result = if let Some(dim) = target_dim {
        // Gracefully degrade to model's max dimension if requested dimension is too large
        let actual_dim = if dim > embedding_vec.len() {
            eprintln!(
                "WARNING: Requested dimension {} exceeds model dimension {}, using full dimension",
                dim,
                embedding_vec.len()
            );
            embedding_vec.len()
        } else {
            dim
        };
        embedding_vec[..actual_dim].to_vec()
    } else {
        embedding_vec
    };

    Ok(result)
}

/// Pad a single encoding to max_len
fn pad_encoding(
    token_ids: &[u32],
    attention_mask: &[u32],
    max_len: usize,
    pad_token_id: u32,
    pad_side: PaddingSide,
) -> (Vec<u32>, Vec<u32>) {
    let pad_len = max_len - token_ids.len();
    match pad_side {
        PaddingSide::Left => {
            let mut padded_ids = vec![pad_token_id; pad_len];
            padded_ids.extend(token_ids);
            let mut padded_mask = vec![0u32; pad_len];
            padded_mask.extend(attention_mask);
            (padded_ids, padded_mask)
        }
        PaddingSide::Right => {
            let mut padded_ids = token_ids.to_vec();
            padded_ids.extend(vec![pad_token_id; pad_len]);
            let mut padded_mask = attention_mask.to_vec();
            padded_mask.extend(vec![0u32; pad_len]);
            (padded_ids, padded_mask)
        }
    }
}

/// Apply Matryoshka truncation to batch embeddings
fn truncate_embeddings_batch(
    data: Vec<Vec<f32>>,
    target_dim: Option<usize>,
    embedding_dim: usize,
) -> Vec<Vec<f32>> {
    match target_dim {
        Some(dim) => {
            let actual_dim = if dim > embedding_dim {
                eprintln!(
                    "WARNING: Requested dimension {} exceeds model dimension {}, using full dimension",
                    dim, embedding_dim
                );
                embedding_dim
            } else {
                dim
            };
            data.into_iter()
                .map(|emb| emb[..actual_dim].to_vec())
                .collect()
        }
        None => data,
    }
}

/// Generic internal helper for batch embedding generation
pub(crate) fn generate_embeddings_batch_internal<'a, F, G>(
    texts: &[&str],
    target_dim: Option<usize>,
    pad_token_id: u32,
    pad_side: PaddingSide,
    get_tokenizer: G,
    forward_fn: F,
) -> Result<Vec<Vec<f32>>, String>
where
    F: Fn(Vec<u32>, Vec<u32>, usize, usize) -> Result<candle_core::Tensor, String>,
    G: Fn() -> Option<&'a tokenizers::Tokenizer>,
{
    if texts.is_empty() {
        return Err("Empty text list".to_string());
    }

    let tokenizer = get_tokenizer().ok_or_else(|| "Tokenizer not available".to_string())?;
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| format!("Batch tokenization failed: {:?}", e))?;

    let max_len = encodings
        .iter()
        .map(|enc| enc.get_ids().len())
        .max()
        .unwrap_or(0);

    let mut batch_token_ids = Vec::new();
    let mut batch_attention_mask = Vec::new();
    for encoding in &encodings {
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let (padded_ids, padded_mask) =
            pad_encoding(&token_ids, &attention_mask, max_len, pad_token_id, pad_side);
        batch_token_ids.push(padded_ids);
        batch_attention_mask.push(padded_mask);
    }

    let batch_size = texts.len();
    let flat_ids: Vec<u32> = batch_token_ids.into_iter().flatten().collect();
    let flat_mask: Vec<u32> = batch_attention_mask.into_iter().flatten().collect();

    let embeddings = forward_fn(flat_ids, flat_mask, batch_size, max_len)?;
    let embedding_dim = embeddings
        .dim(1)
        .map_err(|e| format!("Failed to get embedding dimension: {:?}", e))?;
    let embeddings_data = embeddings
        .to_vec2::<f32>()
        .map_err(|e| format!("Failed to convert embeddings to vec: {:?}", e))?;

    Ok(truncate_embeddings_batch(
        embeddings_data,
        target_dim,
        embedding_dim,
    ))
}

/// Initialize mmBERT embedding model with 2D Matryoshka support
///
/// This model supports:
/// - 32K context length
/// - Multilingual (1800+ languages via Glot500)
/// - 2D Matryoshka: dimension reduction (768→64) AND layer early exit (22→3 layers)
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed
#[no_mangle]
pub unsafe extern "C" fn init_mmbert_embedding_model(model_path: *const c_char, use_cpu: bool) -> bool {
    use candle_core::Device;

    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return false;
    }

    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) if !s.is_empty() => s.to_string(),
            _ => {
                eprintln!("Error: invalid model_path");
                return false;
            }
        }
    };

    // Check if already initialized
    if let Some(factory) = GLOBAL_MODEL_FACTORY.get() {
        if factory.get_mmbert_model().is_some() {
            eprintln!("WARNING: mmBERT model already initialized");
            return true;
        }
    }

    // Determine device
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    // Create or get factory
    let factory = if GLOBAL_MODEL_FACTORY.get().is_some() {
        // Factory exists but mmbert not loaded - we can't modify OnceLock
        eprintln!("Error: ModelFactory already initialized without mmBERT. Initialize mmBERT first or use init_embedding_models_with_mmbert.");
        return false;
    } else {
        let mut factory = ModelFactory::new(device);
        match factory.register_mmbert_embedding_model(&path) {
            Ok(_) => {
                println!("INFO: mmBERT embedding model registered successfully");
            }
            Err(e) => {
                eprintln!("ERROR: Failed to register mmBERT model: {:?}", e);
                return false;
            }
        }
        factory
    };

    match GLOBAL_MODEL_FACTORY.set(factory) {
        Ok(_) => true,
        Err(_) => {
            eprintln!("Error: Failed to set global model factory");
            false
        }
    }
}

/// Parse optional C string path to Option<String>
fn parse_path_ptr(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(ptr).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                _ => None,
            }
        }
    }
}

/// Validate at least one path exists on disk
fn validate_paths_exist(paths: &[Option<String>]) -> bool {
    let has_valid = paths.iter().any(|p| {
        p.as_ref()
            .is_some_and(|s| std::path::Path::new(s).exists())
    });
    if !has_valid {
        for p in paths.iter().flatten() {
            eprintln!("Error: model path does not exist: {}", p);
        }
    }
    has_valid
}

/// Initialize embedding models with given paths (including mmBERT)
///
/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn init_embedding_models_with_mmbert(
    qwen3_model_path: *const c_char,
    gemma_model_path: *const c_char,
    mmbert_model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    let qwen3_path = parse_path_ptr(qwen3_model_path);
    let gemma_path = parse_path_ptr(gemma_model_path);
    let mmbert_path = parse_path_ptr(mmbert_model_path);

    if qwen3_path.is_none() && gemma_path.is_none() && mmbert_path.is_none() {
        eprintln!("Error: at least one model path must be provided");
        return false;
    }

    if !validate_paths_exist(&[qwen3_path.clone(), gemma_path.clone(), mmbert_path.clone()]) {
        return false;
    }

    if GLOBAL_MODEL_FACTORY.get().is_some() {
        eprintln!("WARNING: ModelFactory already initialized");
        return true;
    }

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    let mut factory = ModelFactory::new(device);

    if let Some(path) = qwen3_path {
        if let Err(e) = factory.register_qwen3_embedding_model(&path) {
            eprintln!("ERROR: Failed to register Qwen3 model: {:?}", e);
            return false;
        }
    }
    if let Some(path) = gemma_path {
        if let Err(e) = factory.register_gemma_embedding_model(&path) {
            eprintln!("WARNING: Failed to register Gemma model: {:?}", e);
        }
    }
    if let Some(path) = mmbert_path {
        if let Err(e) = factory.register_mmbert_embedding_model(&path) {
            eprintln!("ERROR: Failed to register mmBERT model: {:?}", e);
            return false;
        }
        println!("INFO: mmBERT embedding model registered with 2D Matryoshka support");
    }

    match GLOBAL_MODEL_FACTORY.set(factory) {
        Ok(_) => true,
        Err(_) => true,
    }
}

/// Initialize embedding models with given paths
///
/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn init_embedding_models(
    qwen3_model_path: *const c_char,
    gemma_model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    let qwen3_path = parse_path_ptr(qwen3_model_path);
    let gemma_path = parse_path_ptr(gemma_model_path);

    if qwen3_path.is_none() && gemma_path.is_none() {
        eprintln!("Error: at least one embedding model path must be provided");
        return false;
    }

    if !validate_paths_exist(&[qwen3_path.clone(), gemma_path.clone()]) {
        return false;
    }

    if GLOBAL_MODEL_FACTORY.get().is_some() {
        eprintln!("WARNING: ModelFactory already initialized");
        return true;
    }

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    let mut factory = ModelFactory::new(device);

    if let Some(path) = qwen3_path {
        if let Err(e) = factory.register_qwen3_embedding_model(&path) {
            eprintln!("ERROR: Failed to register Qwen3 model: {:?}", e);
            return false;
        }
    }
    if let Some(path) = gemma_path {
        if let Err(e) = factory.register_gemma_embedding_model(&path) {
            eprintln!("WARNING: Failed to register Gemma model: {:?}", e);
            eprintln!("WARNING: Continuing with Qwen3 only. This is expected if Gemma model is not downloaded (e.g., missing HF_TOKEN for gated models)");
        } else {
            println!(
                "INFO: Gemma embedding model registered successfully from {}",
                path
            );
        }
    }

    match GLOBAL_MODEL_FACTORY.set(factory) {
        Ok(_) => true,
        Err(_) => true,
    }
}
