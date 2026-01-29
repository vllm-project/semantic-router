//! Embedding Generation FFI Module (ONNX Runtime)
//!
//! This module provides Foreign Function Interface (FFI) functions for
//! mmBERT embedding generation with 2D Matryoshka support using ONNX Runtime.

use crate::ffi::types::{
    BatchSimilarityResult, EmbeddingModelInfo, EmbeddingModelsInfoResult, EmbeddingResult,
    EmbeddingSimilarityResult, MatryoshkaInfo, SimilarityMatch,
};
use crate::model_architectures::embedding::mmbert_embedding::MmBertEmbeddingModel;
use parking_lot::Mutex;
use std::ffi::{c_char, CStr, CString};
use std::sync::OnceLock;

/// Global singleton for MmBertEmbeddingModel (wrapped in Mutex for mutable access)
static GLOBAL_MMBERT_MODEL: OnceLock<Mutex<MmBertEmbeddingModel>> = OnceLock::new();

// ============================================================================
// Initialization Functions
// ============================================================================

/// Initialize mmBERT embedding model with 2D Matryoshka support
///
/// This model supports:
/// - 32K context length
/// - Multilingual (1800+ languages via Glot500)
/// - 2D Matryoshka: dimension reduction (768→64) AND layer early exit (22→3 layers)
/// - AMD GPU via ROCm, NVIDIA GPU via CUDA, or CPU
///
/// # Safety
/// - `model_path` must be a valid null-terminated C string
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed
#[no_mangle]
pub extern "C" fn init_mmbert_embedding_model(model_path: *const c_char, use_cpu: bool) -> bool {
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
    if GLOBAL_MMBERT_MODEL.get().is_some() {
        eprintln!("WARNING: mmBERT model already initialized");
        return true;
    }

    // Load model
    match MmBertEmbeddingModel::load(&path, use_cpu) {
        Ok(model) => {
            println!("INFO: mmBERT embedding model loaded from {}", path);
            println!("INFO: {}", model.model_info());

            match GLOBAL_MMBERT_MODEL.set(Mutex::new(model)) {
                Ok(_) => true,
                Err(_) => {
                    eprintln!("Error: Failed to set global model (race condition?)");
                    false
                }
            }
        }
        Err(e) => {
            eprintln!("ERROR: Failed to load mmBERT model: {:?}", e);
            false
        }
    }
}

/// Check if mmBERT model is initialized
#[no_mangle]
pub extern "C" fn is_mmbert_model_initialized() -> bool {
    GLOBAL_MMBERT_MODEL.get().is_some()
}

// ============================================================================
// Embedding Generation Functions
// ============================================================================

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

/// Get embedding with 2D Matryoshka support (layer early exit + dimension truncation)
///
/// This function supports the full 2D Matryoshka API:
/// - Layer early exit: Use fewer layers (3, 6, 11, or 22) for faster inference
/// - Dimension truncation: Use smaller dimensions (64, 128, 256, 512, 768)
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `target_layer`: Target layer for early exit (0 for full model)
/// - `target_dim`: Target dimension (0 for default 768)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embedding_2d_matryoshka(
    text: *const c_char,
    target_layer: i32,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_2d_matryoshka");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                *result = create_error_result();
                return -1;
            }
        }
    };

    // Get model
    let model_lock = match GLOBAL_MMBERT_MODEL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: mmBERT model not initialized");
            unsafe {
                *result = create_error_result();
            }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    // Convert parameters
    let layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };

    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Generate embedding
    let mut model = model_lock.lock();
    match model.encode_single(text_str, layer, dim) {
        Ok(embedding) => {
            let length = embedding.len() as i32;
            let embedding_vec: Vec<f32> = embedding.to_vec();
            let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

            unsafe {
                *result = EmbeddingResult {
                    data,
                    length,
                    error: false,
                    model_type: 0, // mmbert
                    sequence_length: text_str.split_whitespace().count() as i32,
                    processing_time_ms,
                };
            }

            0
        }
        Err(e) => {
            eprintln!("Error: embedding generation failed: {:?}", e);
            unsafe {
                *result = create_error_result();
            }
            -1
        }
    }
}

/// Get embedding (shortcut for full model, full dimension)
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embedding(text: *const c_char, result: *mut EmbeddingResult) -> i32 {
    get_embedding_2d_matryoshka(text, 0, 0, result)
}

/// Get embedding with target dimension only (no layer early exit)
///
/// # Parameters
/// - `text`: Input text (C string)
/// - `target_dim`: Target dimension (768, 512, 256, 128, or 64; 0 for default)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embedding_with_dim(
    text: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    get_embedding_2d_matryoshka(text, 0, target_dim, result)
}

// ============================================================================
// Batch Embedding Functions
// ============================================================================

/// Generate embeddings for multiple texts in batch
///
/// # Parameters
/// - `texts`: Array of input texts (C string array)
/// - `num_texts`: Number of texts
/// - `target_layer`: Target layer for early exit (0 for full model)
/// - `target_dim`: Target dimension (0 for default 768)
/// - `results`: Output array for embedding results (must be pre-allocated with num_texts elements)
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embeddings_batch(
    texts: *const *const c_char,
    num_texts: i32,
    target_layer: i32,
    target_dim: i32,
    results: *mut EmbeddingResult,
) -> i32 {
    if texts.is_null() || results.is_null() || num_texts <= 0 {
        eprintln!("Error: invalid parameters to get_embeddings_batch");
        return -1;
    }

    // Parse texts
    let mut text_strs = Vec::with_capacity(num_texts as usize);
    for i in 0..num_texts {
        let text_ptr = unsafe { *texts.offset(i as isize) };
        if text_ptr.is_null() {
            eprintln!("Error: null text at index {}", i);
            return -1;
        }

        let text_str = unsafe {
            match CStr::from_ptr(text_ptr).to_str() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in text {}: {}", i, e);
                    return -1;
                }
            }
        };
        text_strs.push(text_str);
    }

    // Get model
    let model_lock = match GLOBAL_MMBERT_MODEL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: mmBERT model not initialized");
            return -1;
        }
    };

    let start_time = std::time::Instant::now();

    // Convert parameters
    let layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };

    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Generate embeddings
    let mut model = model_lock.lock();
    match model.encode(&text_strs, layer, dim) {
        Ok(embeddings) => {
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
            let per_text_time = processing_time_ms / num_texts as f32;

            for i in 0..num_texts as usize {
                let embedding = embeddings.row(i).to_vec();
                let length = embedding.len() as i32;
                let data = Box::into_raw(embedding.into_boxed_slice()) as *mut f32;

                unsafe {
                    *results.offset(i as isize) = EmbeddingResult {
                        data,
                        length,
                        error: false,
                        model_type: 0, // mmbert
                        sequence_length: text_strs[i].split_whitespace().count() as i32,
                        processing_time_ms: per_text_time,
                    };
                }
            }

            0
        }
        Err(e) => {
            eprintln!("Error: batch embedding generation failed: {:?}", e);
            // Set error for all results
            for i in 0..num_texts as usize {
                unsafe {
                    *results.offset(i as isize) = create_error_result();
                }
            }
            -1
        }
    }
}

// ============================================================================
// Similarity Functions
// ============================================================================

/// Calculate cosine similarity between two texts
///
/// # Parameters
/// - `text1`: First text (C string)
/// - `text2`: Second text (C string)
/// - `target_layer`: Target layer for early exit (0 for full model)
/// - `target_dim`: Target dimension (0 for default 768)
/// - `result`: Output pointer for similarity result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn calculate_embedding_similarity(
    text1: *const c_char,
    text2: *const c_char,
    target_layer: i32,
    target_dim: i32,
    result: *mut EmbeddingSimilarityResult,
) -> i32 {
    if text1.is_null() || text2.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to calculate_embedding_similarity");
        return -1;
    }

    let start_time = std::time::Instant::now();

    // Generate embeddings
    let mut emb1_result = EmbeddingResult::default();
    let mut emb2_result = EmbeddingResult::default();

    let status1 = get_embedding_2d_matryoshka(text1, target_layer, target_dim, &mut emb1_result);
    if status1 != 0 || emb1_result.error {
        unsafe {
            *result = EmbeddingSimilarityResult::default();
        }
        return -1;
    }

    let status2 = get_embedding_2d_matryoshka(text2, target_layer, target_dim, &mut emb2_result);
    if status2 != 0 || emb2_result.error {
        // Free first embedding
        crate::ffi::memory::free_embedding(emb1_result.data, emb1_result.length);
        unsafe {
            *result = EmbeddingSimilarityResult::default();
        }
        return -1;
    }

    // Calculate cosine similarity
    let emb1 = unsafe {
        std::slice::from_raw_parts(emb1_result.data, emb1_result.length as usize)
    };
    let emb2 = unsafe {
        std::slice::from_raw_parts(emb2_result.data, emb2_result.length as usize)
    };

    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = if norm1 > 0.0 && norm2 > 0.0 {
        dot_product / (norm1 * norm2)
    } else {
        0.0
    };

    // Free embeddings
    crate::ffi::memory::free_embedding(emb1_result.data, emb1_result.length);
    crate::ffi::memory::free_embedding(emb2_result.data, emb2_result.length);

    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        *result = EmbeddingSimilarityResult {
            similarity,
            model_type: 0, // mmbert
            processing_time_ms,
            error: false,
        };
    }

    0
}

/// Calculate batch similarity: find top-k most similar candidates for a query
///
/// # Parameters
/// - `query`: Query text (C string)
/// - `candidates`: Array of candidate texts (C string array)
/// - `num_candidates`: Number of candidates
/// - `top_k`: Maximum number of matches to return (0 = return all)
/// - `target_layer`: Target layer for early exit (0 for full model)
/// - `target_dim`: Target dimension (0 for default 768)
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
    target_layer: i32,
    target_dim: i32,
    result: *mut BatchSimilarityResult,
) -> i32 {
    if query.is_null() || candidates.is_null() || result.is_null() || num_candidates <= 0 {
        eprintln!("Error: invalid parameters to calculate_similarity_batch");
        return -1;
    }

    let start_time = std::time::Instant::now();

    // Parse query
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in query: {}", e);
                *result = BatchSimilarityResult::default();
                return -1;
            }
        }
    };

    // Parse candidates
    let mut candidate_strs = Vec::with_capacity(num_candidates as usize);
    for i in 0..num_candidates {
        let candidate_ptr = unsafe { *candidates.offset(i as isize) };
        if candidate_ptr.is_null() {
            eprintln!("Error: null candidate at index {}", i);
            unsafe {
                *result = BatchSimilarityResult::default();
            }
            return -1;
        }

        let candidate_str = unsafe {
            match CStr::from_ptr(candidate_ptr).to_str() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in candidate {}: {}", i, e);
                    *result = BatchSimilarityResult::default();
                    return -1;
                }
            }
        };
        candidate_strs.push(candidate_str);
    }

    // Get model
    let model_lock = match GLOBAL_MMBERT_MODEL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: mmBERT model not initialized");
            unsafe {
                *result = BatchSimilarityResult::default();
            }
            return -1;
        }
    };

    // Convert parameters
    let layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };

    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    // Build all texts (query + candidates)
    let mut all_texts: Vec<&str> = Vec::with_capacity(1 + num_candidates as usize);
    all_texts.push(query_str);
    all_texts.extend(candidate_strs.iter().copied());

    // Generate embeddings in batch
    let mut model = model_lock.lock();
    let embeddings = match model.encode(&all_texts, layer, dim) {
        Ok(embs) => embs,
        Err(e) => {
            eprintln!("Error: batch embedding generation failed: {:?}", e);
            unsafe {
                *result = BatchSimilarityResult::default();
            }
            return -1;
        }
    };

    // Extract query embedding (first one)
    let query_embedding = embeddings.row(0);

    // Calculate similarities with all candidates
    let mut similarities = Vec::with_capacity(num_candidates as usize);
    for i in 0..num_candidates as usize {
        let candidate_embedding = embeddings.row(i + 1);

        // Cosine similarity
        let dot_product: f32 = query_embedding
            .iter()
            .zip(candidate_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_query: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_candidate: f32 = candidate_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();

        let similarity = if norm_query > 0.0 && norm_candidate > 0.0 {
            dot_product / (norm_query * norm_candidate)
        } else {
            0.0
        };

        similarities.push((i, similarity));
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
        *result = BatchSimilarityResult {
            matches: matches_ptr,
            num_matches,
            model_type: 0, // mmbert
            processing_time_ms,
            error: false,
        };
    }

    0
}

// ============================================================================
// Information Functions
// ============================================================================

/// Get information about the loaded mmBERT model
///
/// # Parameters
/// - `result`: Output pointer for model info result
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_embedding_models_info(result: *mut EmbeddingModelsInfoResult) -> i32 {
    if result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_models_info");
        return -1;
    }

    let model_opt = GLOBAL_MMBERT_MODEL.get();

    let mut models_vec = Vec::new();

    // mmBERT model info
    let model_name = CString::new("mmbert").unwrap();
    let (is_loaded, max_seq, default_dim, model_path, supports_exit, layers_str) = if let Some(model_lock) = model_opt {
        let model = model_lock.lock();
        let config = model.config();
        let layers = model
            .available_exit_layers()
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let path = CString::new(model.model_info()).unwrap();
        let layers_cstr = CString::new(layers).unwrap();
        (
            true,
            config.max_position_embeddings as i32,
            config.hidden_size as i32,
            path.into_raw(),
            model.supports_layer_exit(),
            layers_cstr.into_raw(),
        )
    } else {
        let empty = CString::new("").unwrap();
        (
            false,
            0,
            0,
            empty.clone().into_raw(),
            false,
            empty.into_raw(),
        )
    };

    models_vec.push(EmbeddingModelInfo {
        model_name: model_name.into_raw(),
        is_loaded,
        max_sequence_length: max_seq,
        default_dimension: default_dim,
        model_path,
        supports_layer_exit: supports_exit,
        available_layers: layers_str,
    });

    let num_models = models_vec.len() as i32;
    let models_ptr = Box::into_raw(models_vec.into_boxed_slice()) as *mut EmbeddingModelInfo;

    unsafe {
        *result = EmbeddingModelsInfoResult {
            models: models_ptr,
            num_models,
            error: false,
        };
    }

    0
}

/// Get Matryoshka configuration info
///
/// # Parameters
/// - `result`: Output pointer for MatryoshkaInfo
///
/// # Returns
/// 0 on success, -1 on error
#[no_mangle]
pub extern "C" fn get_matryoshka_info(result: *mut MatryoshkaInfo) -> i32 {
    if result.is_null() {
        return -1;
    }

    let config = crate::model_architectures::embedding::mmbert_embedding::MatryoshkaConfig::default();

    let dimensions = config
        .dimensions
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let layers = config
        .layers
        .iter()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(",");

    let dim_cstr = CString::new(dimensions).unwrap();
    let layers_cstr = CString::new(layers).unwrap();

    unsafe {
        *result = MatryoshkaInfo {
            dimensions: dim_cstr.into_raw(),
            layers: layers_cstr.into_raw(),
            supports_2d: true,
        };
    }

    0
}

/// Free MatryoshkaInfo
#[no_mangle]
pub extern "C" fn free_matryoshka_info(info: *mut MatryoshkaInfo) {
    if info.is_null() {
        return;
    }

    unsafe {
        let info_ref = &mut *info;
        if !info_ref.dimensions.is_null() {
            let _ = CString::from_raw(info_ref.dimensions);
        }
        if !info_ref.layers.is_null() {
            let _ = CString::from_raw(info_ref.layers);
        }
    }
}
