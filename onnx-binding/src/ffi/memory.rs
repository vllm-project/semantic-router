//! Memory management functions for FFI

use std::ffi::{c_char, CString};

/// Free an embedding array allocated by Rust
///
/// # Safety
/// - `data` must be a valid pointer allocated by Rust
/// - `length` must match the original allocation length
#[no_mangle]
pub extern "C" fn free_embedding(data: *mut f32, length: i32) {
    if data.is_null() || length <= 0 {
        return;
    }

    unsafe {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(data, length as usize));
    }
}

/// Free a C string allocated by Rust
///
/// # Safety
/// - `s` must be a valid CString pointer allocated by Rust
#[no_mangle]
pub extern "C" fn free_cstring(s: *mut c_char) {
    if s.is_null() {
        return;
    }

    unsafe {
        let _ = CString::from_raw(s);
    }
}

/// Free batch similarity result
///
/// # Safety
/// - `result` must be a valid pointer to BatchSimilarityResult
#[no_mangle]
pub extern "C" fn free_batch_similarity_result(result: *mut super::types::BatchSimilarityResult) {
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

/// Free embedding models info result
///
/// # Safety
/// - `result` must be a valid pointer to EmbeddingModelsInfoResult
#[no_mangle]
pub extern "C" fn free_embedding_models_info(result: *mut super::types::EmbeddingModelsInfoResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let info_result = &mut *result;

        if !info_result.models.is_null() && info_result.num_models > 0 {
            let models_slice = std::slice::from_raw_parts_mut(
                info_result.models,
                info_result.num_models as usize,
            );

            for model_info in models_slice.iter_mut() {
                if !model_info.model_name.is_null() {
                    let _ = CString::from_raw(model_info.model_name);
                }
                if !model_info.model_path.is_null() {
                    let _ = CString::from_raw(model_info.model_path);
                }
                if !model_info.available_layers.is_null() {
                    let _ = CString::from_raw(model_info.available_layers);
                }
            }

            let _ = Box::from_raw(models_slice.as_mut_ptr());
        }

        info_result.models = std::ptr::null_mut();
        info_result.num_models = 0;
    }
}
