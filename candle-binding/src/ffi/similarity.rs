//! FFI Similarity Functions

use crate::ffi::init::BERT_SIMILARITY;
use crate::ffi::types::*;
use std::ffi::{c_char, CStr};

/// Get text embedding
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn get_text_embedding(text: *const c_char, max_length: i32) -> EmbeddingResult {
    // Migrated from lib.rs:555-629
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return EmbeddingResult {
                    data: std::ptr::null_mut(),
                    length: 0,
                    error: true,
                    model_type: -1,
                    sequence_length: 0,
                    processing_time_ms: 0.0,
                }
            }
        }
    };

    let bert = match BERT_SIMILARITY.get() {
        Some(b) => b.clone(),
        None => {
            eprintln!("BERT model not initialized");
            return EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true,
                model_type: -1,
                sequence_length: 0,
                processing_time_ms: 0.0,
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.get_embedding(text, max_length_opt) {
        Ok(embedding) => {
            match embedding.flatten_all() {
                Ok(flat_embedding) => {
                    match flat_embedding.to_vec1::<f32>() {
                        Ok(vec) => {
                            let length = vec.len() as i32;
                            // Allocate memory that will be freed by Go
                            let data = vec.as_ptr() as *mut f32;
                            std::mem::forget(vec); // Don't drop the vector - Go will own the memory now
                            EmbeddingResult {
                                data,
                                length,
                                error: false,
                                model_type: -1, // BERT model (not Qwen3/Gemma)
                                sequence_length: 0,
                                processing_time_ms: 0.0,
                            }
                        }
                        Err(_) => EmbeddingResult {
                            data: std::ptr::null_mut(),
                            length: 0,
                            error: true,
                            model_type: -1,
                            sequence_length: 0,
                            processing_time_ms: 0.0,
                        },
                    }
                }
                Err(_) => EmbeddingResult {
                    data: std::ptr::null_mut(),
                    length: 0,
                    error: true,
                    model_type: -1,
                    sequence_length: 0,
                    processing_time_ms: 0.0,
                },
            }
        }
        Err(e) => {
            eprintln!("Error getting embedding: {e}");
            EmbeddingResult {
                data: std::ptr::null_mut(),
                length: 0,
                error: true,
                model_type: -1,
                sequence_length: 0,
                processing_time_ms: 0.0,
            }
        }
    }
}

/// Calculate similarity between two texts
///
/// # Safety
/// - `text1` and `text2` must be valid null-terminated C strings
#[no_mangle]
pub extern "C" fn calculate_similarity(
    text1: *const c_char,
    text2: *const c_char,
    max_length: i32,
) -> f32 {
    // Migrated from lib.rs:630-673
    let text1 = unsafe {
        match CStr::from_ptr(text1).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };

    let text2 = unsafe {
        match CStr::from_ptr(text2).to_str() {
            Ok(s) => s,
            Err(_) => return -1.0,
        }
    };

    let bert = match BERT_SIMILARITY.get() {
        Some(b) => b.clone(),
        None => {
            eprintln!("BERT model not initialized");
            return -1.0;
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.calculate_similarity(text1, text2, max_length_opt) {
        Ok(similarity) => similarity,
        Err(e) => {
            eprintln!("Error calculating similarity: {e}");
            -1.0
        }
    }
}

/// Find most similar text from a list
///
/// # Safety
/// - `query_text` must be a valid null-terminated C string
/// - `texts` must be a valid array of null-terminated C strings
/// - `texts_count` must match the actual array size
#[no_mangle]
pub extern "C" fn find_most_similar(
    query: *const c_char,
    candidates_ptr: *const *const c_char,
    num_candidates: i32,
    max_length: i32,
) -> SimilarityResult {
    // Migrated from lib.rs:674-745
    let query = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(_) => {
                return SimilarityResult {
                    index: -1,
                    similarity: -1.0,
                    text: std::ptr::null_mut(),
                }
            }
        }
    };

    // Convert the array of C strings to Rust strings
    let candidates: Vec<&str> = unsafe {
        let mut result = Vec::with_capacity(num_candidates as usize);
        let candidates_slice = std::slice::from_raw_parts(candidates_ptr, num_candidates as usize);

        for &cstr in candidates_slice {
            match CStr::from_ptr(cstr).to_str() {
                Ok(s) => result.push(s),
                Err(_) => {
                    return SimilarityResult {
                        index: -1,
                        similarity: -1.0,
                        text: std::ptr::null_mut(),
                    }
                }
            }
        }

        result
    };

    let bert = match BERT_SIMILARITY.get() {
        Some(b) => b.clone(),
        None => {
            eprintln!("BERT model not initialized");
            return SimilarityResult {
                index: -1,
                similarity: -1.0,
                text: std::ptr::null_mut(),
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    match bert.find_most_similar(query, &candidates, max_length_opt) {
        Ok((idx, score)) => {
            // Allocate C string for the most similar text
            let most_similar_text = if idx < candidates.len() {
                unsafe { crate::ffi::memory::allocate_c_string(&candidates[idx]) }
            } else {
                std::ptr::null_mut()
            };

            SimilarityResult {
                index: idx as i32,
                similarity: score,
                text: most_similar_text,
            }
        }
        Err(e) => {
            eprintln!("Error finding most similar: {e}");
            SimilarityResult {
                index: -1,
                similarity: -1.0,
                text: std::ptr::null_mut(),
            }
        }
    }
}
