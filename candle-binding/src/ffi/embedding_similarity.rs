//! Embedding similarity calculations and models info FFI.
//!
//! Cosine similarity, batch similarity, and embedding models metadata.

use crate::ffi::embedding::GLOBAL_MODEL_FACTORY;
use crate::ffi::embedding_routing::{
    generate_gemma_embedding, generate_gemma_embeddings_batch, generate_qwen3_embedding,
    generate_qwen3_embeddings_batch, get_embedding_with_dim,
};
use crate::ffi::types::{
    BatchSimilarityResult, EmbeddingModelInfo, EmbeddingModelsInfoResult, EmbeddingResult,
    EmbeddingSimilarityResult, SimilarityMatch,
};
use std::ffi::{c_char, CStr, CString};

/// Parse C string inputs for similarity calculation
fn parse_similarity_inputs(
    text1: *const c_char,
    text2: *const c_char,
    model_type_str: *const c_char,
    result: *mut EmbeddingSimilarityResult,
) -> Result<(String, String, String), i32> {
    let text1_str = unsafe {
        match CStr::from_ptr(text1).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text1: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return Err(-1);
            }
        }
    };
    let text2_str = unsafe {
        match CStr::from_ptr(text2).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text2: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return Err(-1);
            }
        }
    };
    let model_type_str = unsafe {
        match CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = EmbeddingSimilarityResult::default();
                return Err(-1);
            }
        }
    };
    Ok((text1_str, text2_str, model_type_str))
}

/// Compute cosine similarity between two embedding vectors
fn cosine_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm1 > 0.0 && norm2 > 0.0 {
        dot_product / (norm1 * norm2)
    } else {
        0.0
    }
}

/// Get embeddings in auto mode (routing) for both texts
fn get_auto_mode_embeddings(
    text1: *const c_char,
    text2: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingSimilarityResult,
) -> Result<(Vec<f32>, Vec<f32>, i32), i32> {
    let mut emb_result1 = EmbeddingResult::default();
    let status1 = unsafe {
        get_embedding_with_dim(
            text1,
            0.5,
            0.5,
            target_dim,
            &mut emb_result1 as *mut EmbeddingResult,
        )
    };
    if status1 != 0 || emb_result1.error {
        eprintln!("Error generating embedding for text1");
        if !emb_result1.data.is_null() {
            unsafe { crate::ffi::memory::free_embedding(emb_result1.data, emb_result1.length) };
        }
        unsafe { (*result) = EmbeddingSimilarityResult::default() }
        return Err(-1);
    }

    let mut emb_result2 = EmbeddingResult::default();
    let status2 = unsafe {
        get_embedding_with_dim(
            text2,
            0.5,
            0.5,
            target_dim,
            &mut emb_result2 as *mut EmbeddingResult,
        )
    };
    if status2 != 0 || emb_result2.error {
        eprintln!("Error generating embedding for text2");
        if !emb_result1.data.is_null() {
            unsafe { crate::ffi::memory::free_embedding(emb_result1.data, emb_result1.length) };
        }
        if !emb_result2.data.is_null() {
            unsafe { crate::ffi::memory::free_embedding(emb_result2.data, emb_result2.length) };
        }
        unsafe { (*result) = EmbeddingSimilarityResult::default() }
        return Err(-1);
    }

    let emb1 = unsafe {
        std::slice::from_raw_parts(emb_result1.data, emb_result1.length as usize).to_vec()
    };
    let emb2 = unsafe {
        std::slice::from_raw_parts(emb_result2.data, emb_result2.length as usize).to_vec()
    };
    let model_id = emb_result1.model_type;

    unsafe { crate::ffi::memory::free_embedding(emb_result1.data, emb_result1.length) };
    unsafe { crate::ffi::memory::free_embedding(emb_result2.data, emb_result2.length) };

    Ok((emb1, emb2, model_id))
}

/// Get embeddings in manual mode (qwen3 or gemma)
fn get_manual_mode_embeddings(
    factory: &crate::model_architectures::model_factory::ModelFactory,
    text1_str: &str,
    text2_str: &str,
    model_type_str: &str,
    target_dimension: Option<usize>,
    result: *mut EmbeddingSimilarityResult,
) -> Result<(Vec<f32>, Vec<f32>, i32), i32> {
    let (emb1, emb2, model_id) = if model_type_str == "qwen3" {
        let emb1 = generate_qwen3_embedding(factory, text1_str, target_dimension)
            .map_err(|e| eprintln!("Error generating Qwen3 embedding for text1: {}", e))
            .ok();
        let emb2 = generate_qwen3_embedding(factory, text2_str, target_dimension)
            .map_err(|e| eprintln!("Error generating Qwen3 embedding for text2: {}", e))
            .ok();
        (emb1, emb2, 0)
    } else {
        let emb1 = generate_gemma_embedding(factory, text1_str, target_dimension)
            .map_err(|e| eprintln!("Error generating Gemma embedding for text1: {}", e))
            .ok();
        let emb2 = generate_gemma_embedding(factory, text2_str, target_dimension)
            .map_err(|e| eprintln!("Error generating Gemma embedding for text2: {}", e))
            .ok();
        (emb1, emb2, 1)
    };

    match (emb1, emb2) {
        (Some(e1), Some(e2)) => Ok((e1, e2, model_id)),
        _ => {
            eprintln!("Error: failed to generate embeddings");
            unsafe { (*result) = EmbeddingSimilarityResult::default() }
            Err(-1)
        }
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn calculate_embedding_similarity(
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

    let (text1_str, text2_str, model_type_str) =
        match parse_similarity_inputs(text1, text2, model_type_str, result) {
            Ok(t) => t,
            Err(e) => return e,
        };

    if model_type_str != "auto" && model_type_str != "qwen3" && model_type_str != "gemma" {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', or 'gemma')",
            model_type_str
        );
        unsafe { (*result) = EmbeddingSimilarityResult::default() }
        return -1;
    }

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe { (*result) = EmbeddingSimilarityResult::default() }
            return -1;
        }
    };

    let (emb1_vec, emb2_vec, model_type_id) = if model_type_str == "auto" {
        match get_auto_mode_embeddings(text1, text2, target_dim, result) {
            Ok(t) => t,
            Err(e) => return e,
        }
    } else {
        match get_manual_mode_embeddings(
            factory,
            &text1_str,
            &text2_str,
            &model_type_str,
            target_dimension,
            result,
        ) {
            Ok(t) => t,
            Err(e) => return e,
        }
    };

    if emb1_vec.len() != emb2_vec.len() {
        eprintln!(
            "Error: embeddings have different dimensions ({} vs {})",
            emb1_vec.len(),
            emb2_vec.len()
        );
        unsafe { (*result) = EmbeddingSimilarityResult::default() }
        return -1;
    }

    let similarity = cosine_similarity(&emb1_vec, &emb2_vec);
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

/// Parse batch similarity inputs
fn parse_batch_inputs(
    query: *const c_char,
    candidates: *const *const c_char,
    num_candidates: i32,
    model_type_str: *const c_char,
    result: *mut BatchSimilarityResult,
) -> Result<(String, Vec<String>, String), i32> {
    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in query: {}", e);
                (*result) = BatchSimilarityResult::default();
                return Err(-1);
            }
        }
    };

    let model_type_str = unsafe {
        match CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s.to_string(),
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = BatchSimilarityResult::default();
                return Err(-1);
            }
        }
    };

    let mut candidate_texts = Vec::with_capacity(num_candidates as usize);
    for i in 0..num_candidates {
        let candidate_ptr = unsafe { *candidates.offset(i as isize) };
        if candidate_ptr.is_null() {
            eprintln!("Error: null candidate at index {}", i);
            unsafe { (*result) = BatchSimilarityResult::default() }
            return Err(-1);
        }
        let candidate_str = unsafe {
            match CStr::from_ptr(candidate_ptr).to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in candidate {}: {}", i, e);
                    (*result) = BatchSimilarityResult::default();
                    return Err(-1);
                }
            }
        };
        candidate_texts.push(candidate_str);
    }

    Ok((query_str, candidate_texts, model_type_str))
}

/// Compute similarities between query and each candidate
fn compute_candidate_similarities(
    query_embedding: &[f32],
    embeddings_batch: &[Vec<f32>],
) -> Result<Vec<(usize, f32)>, ()> {
    let mut similarities = Vec::with_capacity(embeddings_batch.len() - 1);
    for (idx, candidate_embedding) in embeddings_batch[1..].iter().enumerate() {
        if query_embedding.len() != candidate_embedding.len() {
            eprintln!(
                "Error: dimension mismatch at candidate {} ({} vs {})",
                idx,
                query_embedding.len(),
                candidate_embedding.len()
            );
            return Err(());
        }
        let sim = cosine_similarity(query_embedding, candidate_embedding);
        similarities.push((idx, sim));
    }
    Ok(similarities)
}

fn resolve_batch_model(
    model_type_str: &str,
    query_str: &str,
    candidate_texts: &[String],
) -> Result<(bool, i32), ()> {
    if model_type_str == "qwen3" {
        Ok((true, 0))
    } else if model_type_str == "gemma" {
        Ok((false, 1))
    } else if model_type_str == "auto" {
        let avg_len = (query_str.len() + candidate_texts.iter().map(|s| s.len()).sum::<usize>())
            / (1 + candidate_texts.len());
        Ok(if avg_len > 512 {
            (true, 0)
        } else {
            (false, 1)
        })
    } else {
        eprintln!(
            "Error: invalid model type '{}' (must be 'auto', 'qwen3', or 'gemma')",
            model_type_str
        );
        Err(())
    }
}

fn generate_batch_embeddings(
    use_qwen3: bool,
    factory: &crate::model_architectures::model_factory::ModelFactory,
    all_texts: &[&str],
    target_dimension: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    if use_qwen3 {
        generate_qwen3_embeddings_batch(factory, all_texts, target_dimension)
    } else {
        generate_gemma_embeddings_batch(factory, all_texts, target_dimension)
    }
}

fn write_batch_result(
    result: *mut BatchSimilarityResult,
    mut similarities: Vec<(usize, f32)>,
    top_k: i32,
    num_candidates: i32,
    model_type_id: i32,
    processing_time_ms: f32,
) {
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

    unsafe {
        (*result) = BatchSimilarityResult {
            matches: matches_ptr,
            num_matches,
            model_type: model_type_id,
            processing_time_ms,
            error: false,
        };
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn calculate_similarity_batch(
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
        unsafe { (*result) = BatchSimilarityResult::default() }
        return -1;
    }

    let start_time = std::time::Instant::now();

    let (query_str, candidate_texts, mtype) =
        match parse_batch_inputs(query, candidates, num_candidates, model_type_str, result) {
            Ok(t) => t,
            Err(e) => return e,
        };

    let (use_qwen3, model_type_id) = match resolve_batch_model(&mtype, &query_str, &candidate_texts) {
        Ok(v) => v,
        Err(()) => {
            unsafe { (*result) = BatchSimilarityResult::default() }
            return -1;
        }
    };

    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe { (*result) = BatchSimilarityResult::default() }
            return -1;
        }
    };

    let all_texts: Vec<&str> = std::iter::once(query_str.as_str())
        .chain(candidate_texts.iter().map(|s| s.as_str()))
        .collect();
    let target_dimension = if target_dim > 0 { Some(target_dim as usize) } else { None };

    let embeddings_batch = match generate_batch_embeddings(use_qwen3, factory, &all_texts, target_dimension) {
        Ok(embs) if !embs.is_empty() => embs,
        Ok(_) => {
            eprintln!("Error: empty embeddings batch");
            unsafe { (*result) = BatchSimilarityResult::default() }
            return -1;
        }
        Err(e) => {
            eprintln!("Error: batch embedding generation failed: {}", e);
            unsafe { (*result) = BatchSimilarityResult::default() }
            return -1;
        }
    };

    let similarities = match compute_candidate_similarities(&embeddings_batch[0], &embeddings_batch) {
        Ok(s) => s,
        Err(()) => {
            unsafe { (*result) = BatchSimilarityResult::default() }
            return -1;
        }
    };

    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
    write_batch_result(result, similarities, top_k, num_candidates, model_type_id, processing_time_ms);
    0
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn free_batch_similarity_result(result: *mut BatchSimilarityResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        let batch_result = &mut *result;
        if !batch_result.matches.is_null() && batch_result.num_matches > 0 {
            let matches_slice = std::slice::from_raw_parts_mut(
                batch_result.matches,
                batch_result.num_matches as usize,
            );
            let _ = Box::from_raw(matches_slice.as_mut_ptr());
        }
        batch_result.matches = std::ptr::null_mut();
        batch_result.num_matches = 0;
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_models_info(result: *mut EmbeddingModelsInfoResult) -> i32 {
    if result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_models_info");
        return -1;
    }

    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("ERROR: ModelFactory not initialized");
            unsafe { (*result) = EmbeddingModelsInfoResult::default() }
            return -1;
        }
    };

    let qwen3_loaded = factory.get_qwen3_model().is_some();
    let gemma_loaded = factory.get_gemma_model().is_some();
    let qwen3_path = factory.get_qwen3_model_path();
    let gemma_path = factory.get_gemma_model_path();

    let mut models_vec = Vec::new();

    let model_name = CString::new("qwen3").unwrap();
    let model_path = CString::new(qwen3_path.unwrap_or("").to_string()).unwrap();
    models_vec.push(EmbeddingModelInfo {
        model_name: model_name.into_raw(),
        is_loaded: qwen3_loaded,
        max_sequence_length: if qwen3_loaded { 32768 } else { 0 },
        default_dimension: if qwen3_loaded { 1024 } else { 0 },
        model_path: model_path.into_raw(),
    });

    let model_name = CString::new("gemma").unwrap();
    let model_path = CString::new(gemma_path.unwrap_or("").to_string()).unwrap();
    models_vec.push(EmbeddingModelInfo {
        model_name: model_name.into_raw(),
        is_loaded: gemma_loaded,
        max_sequence_length: if gemma_loaded { 8192 } else { 0 },
        default_dimension: if gemma_loaded { 768 } else { 0 },
        model_path: model_path.into_raw(),
    });

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

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn free_embedding_models_info(result: *mut EmbeddingModelsInfoResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        let info_result = &mut *result;
        if !info_result.models.is_null() && info_result.num_models > 0 {
            let models_slice =
                std::slice::from_raw_parts_mut(info_result.models, info_result.num_models as usize);
            for model_info in models_slice.iter_mut() {
                if !model_info.model_name.is_null() {
                    let _ = CString::from_raw(model_info.model_name);
                }
                if !model_info.model_path.is_null() {
                    let _ = CString::from_raw(model_info.model_path);
                }
            }
            let _ = Box::from_raw(models_slice.as_mut_ptr());
        }
        info_result.models = std::ptr::null_mut();
        info_result.num_models = 0;
    }
}
