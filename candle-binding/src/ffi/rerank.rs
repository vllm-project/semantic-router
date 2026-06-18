//! FFI bindings for the cross-encoder reranker.
//!
//! Mirrors the batch-similarity FFI: an init function that loads the model into
//! a global `OnceLock`, a `rerank_documents` call that returns a heap-allocated
//! array of `(index, score)` matches, and a `free_rerank_result` to release it.

use std::ffi::{c_char, CStr};
use std::sync::Arc;

use crate::ffi::types::{RerankMatch, RerankResult};
use crate::model_architectures::traditional::{BertCrossEncoder, CROSS_ENCODER};

/// Initialize the global cross-encoder reranker from a model path (local dir or
/// HuggingFace Hub id). Safe to call once; subsequent calls are no-ops that
/// return success if a model is already loaded.
///
/// Returns 0 on success, -1 on error.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_cross_encoder(model_path: *const c_char, use_cpu: bool) -> i32 {
    if model_path.is_null() {
        eprintln!("Error: null model_path passed to init_cross_encoder");
        return -1;
    }

    if CROSS_ENCODER.get().is_some() {
        return 0; // already initialized
    }

    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_path: {}", e);
                return -1;
            }
        }
    };

    match BertCrossEncoder::new(path, use_cpu) {
        Ok(model) => {
            let _ = CROSS_ENCODER.set(Arc::new(model));
            0
        }
        Err(e) => {
            eprintln!("Error: failed to initialize cross-encoder: {}", e);
            -1
        }
    }
}

/// Returns true if the cross-encoder reranker has been initialized.
#[no_mangle]
pub extern "C" fn is_cross_encoder_initialized() -> bool {
    CROSS_ENCODER.get().is_some()
}

/// Rerank `documents` against `query` using the global cross-encoder.
///
/// Returns 0 on success, -1 on error. On success `result.matches` points to a
/// heap-allocated array of `num_matches` entries (sorted by score descending)
/// that must be freed with `free_rerank_result`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn rerank_documents(
    query: *const c_char,
    documents: *const *const c_char,
    num_documents: i32,
    top_n: i32,
    result: *mut RerankResult,
) -> i32 {
    if query.is_null() || documents.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to rerank_documents");
        return -1;
    }

    if num_documents <= 0 {
        eprintln!("Error: num_documents must be positive");
        unsafe { (*result) = RerankResult::default() };
        return -1;
    }

    let model = match CROSS_ENCODER.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: cross-encoder not initialized");
            unsafe { (*result) = RerankResult::default() };
            return -1;
        }
    };

    let query_str = unsafe {
        match CStr::from_ptr(query).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in query: {}", e);
                unsafe_default(result);
                return -1;
            }
        }
    };

    let mut docs: Vec<String> = Vec::with_capacity(num_documents as usize);
    let doc_ptrs = unsafe { std::slice::from_raw_parts(documents, num_documents as usize) };
    for (i, &doc_ptr) in doc_ptrs.iter().enumerate() {
        if doc_ptr.is_null() {
            eprintln!("Error: null document at index {}", i);
            unsafe_default(result);
            return -1;
        }
        let s = unsafe {
            match CStr::from_ptr(doc_ptr).to_str() {
                Ok(s) => s.to_string(),
                Err(e) => {
                    eprintln!("Error: invalid UTF-8 in document {}: {}", i, e);
                    unsafe_default(result);
                    return -1;
                }
            }
        };
        docs.push(s);
    }

    let start = std::time::Instant::now();
    let doc_refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();

    let ranked = match model.rerank(query_str, &doc_refs, top_n) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: cross-encoder rerank failed: {}", e);
            unsafe_default(result);
            return -1;
        }
    };

    let matches: Vec<RerankMatch> = ranked
        .iter()
        .map(|(idx, score)| RerankMatch {
            index: *idx as i32,
            score: *score,
        })
        .collect();

    let num_matches = matches.len() as i32;
    let matches_ptr = Box::into_raw(matches.into_boxed_slice()) as *mut RerankMatch;
    let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = RerankResult {
            matches: matches_ptr,
            num_matches,
            processing_time_ms,
            error: false,
        };
    }
    0
}

/// Free a `RerankResult` previously returned by `rerank_documents`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_rerank_result(result: *mut RerankResult) {
    if result.is_null() {
        return;
    }
    unsafe {
        let r = &mut *result;
        if !r.matches.is_null() && r.num_matches > 0 {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                r.matches,
                r.num_matches as usize,
            ));
            r.matches = std::ptr::null_mut();
            r.num_matches = 0;
        }
    }
}

#[inline]
fn unsafe_default(result: *mut RerankResult) {
    unsafe { (*result) = RerankResult::default() };
}
