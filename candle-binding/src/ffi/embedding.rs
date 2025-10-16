//! Embedding Generation FFI Module
//!
//! This module provides Foreign Function Interface (FFI) functions for
//! intelligent embedding generation with automatic model selection.

use crate::classifiers::unified::{DualPathUnifiedClassifier, EmbeddingRequirements};
use crate::ffi::types::{
    BatchSimilarityResult, EmbeddingResult, EmbeddingSimilarityResult, SimilarityMatch,
};
use crate::model_architectures::ModelType;
use std::ffi::{c_char, CStr};

//Import embedding models and model factory
use crate::model_architectures::config::{DualPathConfig, EmbeddingConfig};
use crate::model_architectures::model_factory::ModelFactory;
use std::sync::OnceLock;

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

/// Global singleton for ModelFactory
static GLOBAL_MODEL_FACTORY: OnceLock<ModelFactory> = OnceLock::new();

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
        if dim > embedding_vec.len() {
            return Err(format!(
                "Target dimension {} exceeds model dimension {}",
                dim,
                embedding_vec.len()
            ));
        }
        embedding_vec[..dim].to_vec()
    } else {
        embedding_vec
    };

    Ok(result)
}

/// Generic internal helper for batch embedding generation
///
/// This function extracts common logic for both Qwen3 and Gemma models.
/// Model-specific logic (tokenizer retrieval and forward pass) is handled via closures.
fn generate_embeddings_batch_internal<'a, F, G>(
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

    // Get tokenizer
    let tokenizer = get_tokenizer().ok_or_else(|| "Tokenizer not available".to_string())?;

    // Batch tokenize all texts
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| format!("Batch tokenization failed: {:?}", e))?;

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
        let (padded_ids, padded_mask) = match pad_side {
            PaddingSide::Left => {
                // Left padding
                let mut padded_ids = vec![pad_token_id; pad_len];
                padded_ids.extend(token_ids);

                let mut padded_mask = vec![0u32; pad_len];
                padded_mask.extend(attention_mask);

                (padded_ids, padded_mask)
            }
            PaddingSide::Right => {
                // Right padding
                let mut padded_ids = token_ids.clone();
                padded_ids.extend(vec![pad_token_id; pad_len]);

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

    // Extract embeddings for each text
    let embedding_dim = embeddings
        .dim(1)
        .map_err(|e| format!("Failed to get embedding dimension: {:?}", e))?;

    let embeddings_data = embeddings
        .to_vec2::<f32>()
        .map_err(|e| format!("Failed to convert embeddings to vec: {:?}", e))?;

    // Apply Matryoshka truncation if requested
    let result_embeddings = if let Some(dim) = target_dim {
        if dim > embedding_dim {
            return Err(format!(
                "Target dimension {} exceeds model dimension {}",
                dim, embedding_dim
            ));
        }
        embeddings_data
            .into_iter()
            .map(|emb| emb[..dim].to_vec())
            .collect()
    } else {
        embeddings_data
    };

    Ok(result_embeddings)
}

/// Initialize embedding models with given paths
///
/// # Safety
/// - `qwen3_model_path` and `gemma_model_path` must be valid null-terminated C strings or null
/// - Must be called before any embedding generation functions
/// - Can only be called once (subsequent calls will be ignored)
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed or already initialized
#[no_mangle]
pub extern "C" fn init_embedding_models(
    qwen3_model_path: *const c_char,
    gemma_model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    // Parse model paths
    let qwen3_path = if qwen3_model_path.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(qwen3_model_path).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                _ => None,
            }
        }
    };

    let gemma_path = if gemma_model_path.is_null() {
        None
    } else {
        unsafe {
            match CStr::from_ptr(gemma_model_path).to_str() {
                Ok(s) if !s.is_empty() => Some(s.to_string()),
                _ => None,
            }
        }
    };

    // Check if at least one model path is provided
    if qwen3_path.is_none() && gemma_path.is_none() {
        eprintln!("Error: at least one embedding model path must be provided");
        return false;
    }

    // Determine device
    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    // Create ModelFactory
    let mut factory = ModelFactory::new(device);

    // Register Qwen3 model if path provided
    if let Some(path) = qwen3_path {
        match factory.register_qwen3_embedding_model(&path) {
            Ok(_) => println!(
                "INFO: Qwen3 embedding model registered successfully from {}",
                path
            ),
            Err(e) => {
                eprintln!("ERROR: Failed to register Qwen3 model: {:?}", e);
                return false;
            }
        }
    }

    // Register Gemma model if path provided
    if let Some(path) = gemma_path {
        match factory.register_gemma_embedding_model(&path) {
            Ok(_) => println!(
                "INFO: Gemma embedding model registered successfully from {}",
                path
            ),
            Err(e) => {
                eprintln!("ERROR: Failed to register Gemma model: {:?}", e);
                return false;
            }
        }
    }

    // Try to initialize the global factory
    match GLOBAL_MODEL_FACTORY.set(factory) {
        Ok(_) => {
            println!("INFO: ModelFactory initialized successfully");
            true
        }
        Err(_) => {
            eprintln!("WARNING: ModelFactory already initialized");
            false
        }
    }
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
) -> Result<Vec<Vec<f32>>, String> {
    use candle_core::Tensor;

    // Qwen3-specific configuration
    const QWEN3_PAD_TOKEN_ID: u32 = 151643;
    let pad_side = PaddingSide::Left;

    // Use the generic internal function
    generate_embeddings_batch_internal(
        texts,
        target_dim,
        QWEN3_PAD_TOKEN_ID,
        pad_side,
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
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    // Use the generic internal function
    generate_embedding_internal(
        text,
        target_dim,
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
) -> Result<Vec<Vec<f32>>, String> {
    use candle_core::Tensor;

    // Gemma-specific configuration
    const GEMMA_PAD_TOKEN_ID: u32 = 0;
    let pad_side = PaddingSide::Right;

    // Use the generic internal function
    generate_embeddings_batch_internal(
        texts,
        target_dim,
        GEMMA_PAD_TOKEN_ID,
        pad_side,
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
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    // Use the generic internal function
    generate_embedding_internal(
        text,
        target_dim,
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
/// 0 on success, -1 on error
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
/// 0 on success, -1 on error
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

    // Create requirements for routing
    let requirements = EmbeddingRequirements {
        sequence_length: text_str.split_whitespace().count(),
        quality_priority,
        latency_priority,
        target_dimension: if target_dim > 0 {
            Some(target_dim as usize)
        } else {
            None
        },
    };

    // Create temporary classifier for routing
    let classifier = match create_temp_classifier() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to create classifier: {}", e);
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    // Select model based on requirements
    let model_type = match classifier.select_embedding_model(&requirements) {
        Ok(mt) => mt,
        Err(e) => {
            eprintln!("Error: model selection failed: {:?}", e);
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    // Convert ModelType to string for get_embedding_with_model_type
    let model_type_str = match model_type {
        ModelType::Qwen3Embedding => "qwen3",
        ModelType::GemmaEmbedding => "gemma",
        _ => {
            eprintln!("Error: unsupported model type: {:?}", model_type);
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    // Call get_embedding_with_model_type
    let model_type_cstr = std::ffi::CString::new(model_type_str).unwrap();
    get_embedding_with_model_type(text, model_type_cstr.as_ptr(), target_dim, result)
}

/// Get embedding with manually specified model type (no automatic routing)
///
/// This function bypasses the automatic routing logic and directly uses the specified model.
/// Useful when the caller explicitly wants to use a specific embedding model.
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `model_type_str`: "qwen3" or "gemma"
/// - `target_dim`: Target dimension (768, 512, 256, or 128, 0 for default)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embedding_with_model_type(
    text: *const c_char,
    model_type_str: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || model_type_str.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_with_model_type");
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
        _ => {
            eprintln!(
                "Error: invalid model type '{}' (must be 'qwen3' or 'gemma')",
                model_type_str
            );
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    let requirements = EmbeddingRequirements {
        sequence_length: text_str.split_whitespace().count(),
        quality_priority: 0.5,
        latency_priority: 0.5,
        target_dimension: if target_dim > 0 {
            Some(target_dim as usize)
        } else {
            None
        },
    };

    // Get model factory
    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("Error: ModelFactory not initialized");
            unsafe {
                (*result) = create_error_result();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    // Generate embedding based on model type
    let embedding_result = match model_type {
        ModelType::Qwen3Embedding => {
            generate_qwen3_embedding(factory, text_str, requirements.target_dimension)
        }
        ModelType::GemmaEmbedding => {
            generate_gemma_embedding(factory, text_str, requirements.target_dimension)
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
        Ok(embedding_vec) => {
            let length = embedding_vec.len() as i32;
            let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            // Map ModelType enum to FFI integer values
            let model_type_id = match model_type {
                ModelType::Qwen3Embedding => 0,
                ModelType::GemmaEmbedding => 1,
                _ => -1,
            };

            unsafe {
                (*result) = EmbeddingResult {
                    data,
                    length,
                    error: false,
                    model_type: model_type_id,
                    sequence_length: requirements.sequence_length as i32,
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
            -1
        }
    }
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
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn calculate_embedding_similarity(
    text1: *const c_char,
    text2: *const c_char,
    model_type_str: *const c_char,
    target_dim: i32,
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
    if model_type_str != "auto" && model_type_str != "qwen3" && model_type_str != "gemma" {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', or 'gemma')",
            model_type_str
        );
        unsafe {
            (*result) = EmbeddingSimilarityResult::default();
        }
        return -1;
    }

    // Get target dimension
    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Get model factory
    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe {
                (*result) = EmbeddingSimilarityResult::default();
            }
            return -1;
        }
    };

    // Generate embeddings directly based on model_type
    let (emb1_vec, emb2_vec, model_type_id) = if model_type_str == "auto" {
        // Auto mode: use routing for each text independently

        let mut emb_result1 = EmbeddingResult::default();
        let status1 = get_embedding_with_dim(
            text1,
            0.5, // default quality priority
            0.5, // default latency priority
            target_dim,
            &mut emb_result1 as *mut EmbeddingResult,
        );

        if status1 != 0 || emb_result1.error {
            eprintln!("Error generating embedding for text1");
            unsafe {
                (*result) = EmbeddingSimilarityResult::default();
            }
            return -1;
        }

        let mut emb_result2 = EmbeddingResult::default();
        let status2 = get_embedding_with_dim(
            text2,
            0.5,
            0.5,
            target_dim,
            &mut emb_result2 as *mut EmbeddingResult,
        );

        if status2 != 0 || emb_result2.error {
            eprintln!("Error generating embedding for text2");
            if !emb_result1.data.is_null() {
                crate::ffi::memory::free_embedding(emb_result1.data, emb_result1.length);
            }
            unsafe {
                (*result) = EmbeddingSimilarityResult::default();
            }
            return -1;
        }

        // Convert to Vec
        let emb1 = unsafe {
            std::slice::from_raw_parts(emb_result1.data, emb_result1.length as usize).to_vec()
        };
        let emb2 = unsafe {
            std::slice::from_raw_parts(emb_result2.data, emb_result2.length as usize).to_vec()
        };

        let model_id = emb_result1.model_type;

        // Free the raw data
        crate::ffi::memory::free_embedding(emb_result1.data, emb_result1.length);
        crate::ffi::memory::free_embedding(emb_result2.data, emb_result2.length);

        (emb1, emb2, model_id)
    } else {
        // Manual mode: directly use specified model

        let (emb1, emb2, model_id) = if model_type_str == "qwen3" {
            let emb1 = generate_qwen3_embedding(factory, text1_str, target_dimension)
                .map_err(|e| {
                    eprintln!("Error generating Qwen3 embedding for text1: {}", e);
                    e
                })
                .ok();
            let emb2 = generate_qwen3_embedding(factory, text2_str, target_dimension)
                .map_err(|e| {
                    eprintln!("Error generating Qwen3 embedding for text2: {}", e);
                    e
                })
                .ok();
            (emb1, emb2, 0)
        } else {
            // "gemma"
            let emb1 = generate_gemma_embedding(factory, text1_str, target_dimension)
                .map_err(|e| {
                    eprintln!("Error generating Gemma embedding for text1: {}", e);
                    e
                })
                .ok();
            let emb2 = generate_gemma_embedding(factory, text2_str, target_dimension)
                .map_err(|e| {
                    eprintln!("Error generating Gemma embedding for text2: {}", e);
                    e
                })
                .ok();
            (emb1, emb2, 1)
        };

        match (emb1, emb2) {
            (Some(e1), Some(e2)) => (e1, e2, model_id),
            _ => {
                eprintln!("Error: failed to generate embeddings");
                unsafe {
                    (*result) = EmbeddingSimilarityResult::default();
                }
                return -1;
            }
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
/// 0 on success, -1 on error
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
    if query.is_null() || candidates.is_null() || result.is_null() {
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
    if model_type_str != "auto" && model_type_str != "qwen3" && model_type_str != "gemma" {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', or 'gemma')",
            model_type_str
        );
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

    // Get global model factory
    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe {
                (*result) = BatchSimilarityResult::default();
            }
            return -1;
        }
    };

    // Determine which model to use
    let (use_qwen3, model_type_id) = if model_type_str == "qwen3" {
        (true, 0)
    } else if model_type_str == "gemma" {
        (false, 1)
    } else {
        // "auto": use simple heuristic (can be improved with routing logic)
        let avg_len = (query_str.len() + candidate_texts.iter().map(|s| s.len()).sum::<usize>())
            / (1 + candidate_texts.len());
        if avg_len > 512 {
            (true, 0) // Qwen3 for longer texts
        } else {
            (false, 1) // Gemma for shorter texts
        }
    };

    // Prepare all texts for batch processing: [query, candidate1, candidate2, ...]
    let mut all_texts: Vec<&str> = Vec::with_capacity(1 + num_candidates as usize);
    all_texts.push(query_str);
    all_texts.extend(candidate_texts.iter().copied());

    // Target dimension
    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Batch generate embeddings using the appropriate model
    let embeddings_batch = if use_qwen3 {
        match generate_qwen3_embeddings_batch(factory, &all_texts, target_dimension) {
            Ok(embs) => embs,
            Err(e) => {
                eprintln!("Error: Qwen3 batch embedding generation failed: {}", e);
                unsafe {
                    (*result) = BatchSimilarityResult::default();
                }
                return -1;
            }
        }
    } else {
        match generate_gemma_embeddings_batch(factory, &all_texts, target_dimension) {
            Ok(embs) => embs,
            Err(e) => {
                eprintln!("Error: Gemma batch embedding generation failed: {}", e);
                unsafe {
                    (*result) = BatchSimilarityResult::default();
                }
                return -1;
            }
        }
    };

    // Extract query embedding (first one)
    if embeddings_batch.is_empty() {
        eprintln!("Error: empty embeddings batch");
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
#[no_mangle]
pub extern "C" fn free_batch_similarity_result(result: *mut BatchSimilarityResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let batch_result = &mut *result;

        // Free the matches array if it's not null
        if !batch_result.matches.is_null() && batch_result.num_matches > 0 {
            let matches_slice = std::slice::from_raw_parts_mut(
                batch_result.matches,
                batch_result.num_matches as usize,
            );
            let _ = Box::from_raw(matches_slice.as_mut_ptr());
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
            max_sequence_length: if qwen3_loaded { 32768 } else { 0 },
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
            max_sequence_length: if gemma_loaded { 8192 } else { 0 },
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

            for i in 0..models_slice.len() {
                let model_info = &mut models_slice[i];
                // Free model_name string
                if !model_info.model_name.is_null() {
                    let _ = CString::from_raw(model_info.model_name);
                }
                // Free model_path string
                if !model_info.model_path.is_null() {
                    let _ = CString::from_raw(model_info.model_path);
                }
            }

            // Free the models array
            let _ = Box::from_raw(models_slice.as_mut_ptr());
        }

        // Reset the result
        info_result.models = std::ptr::null_mut();
        info_result.num_models = 0;
    }
}
