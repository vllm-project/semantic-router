//! Embedding dimension contract FFI.

use crate::ffi::embedding::loaded_embedding_dimension_contract;
use crate::ffi::types::EmbeddingDimensionContractResult;
use std::ffi::{c_char, CStr, CString};

fn append_supported_dimension(dimensions: &mut Vec<i32>, dimension: usize) {
    if dimension > 0 && dimension <= i32::MAX as usize {
        let dimension = dimension as i32;
        if !dimensions.contains(&dimension) {
            dimensions.push(dimension);
        }
    }
}

pub(crate) fn write_embedding_dimension_contract(
    result: *mut EmbeddingDimensionContractResult,
    model_name: &str,
    native_dimension: usize,
    declared_dimensions: &[usize],
) -> i32 {
    if result.is_null()
        || native_dimension == 0
        || native_dimension > i32::MAX as usize
        || declared_dimensions
            .iter()
            .any(|dimension| *dimension == 0 || *dimension > i32::MAX as usize)
    {
        if !result.is_null() {
            unsafe {
                *result = EmbeddingDimensionContractResult::default();
            }
        }
        return -1;
    }

    let mut supported_dimensions = Vec::with_capacity(declared_dimensions.len() + 1);
    append_supported_dimension(&mut supported_dimensions, native_dimension);
    for dimension in declared_dimensions {
        append_supported_dimension(&mut supported_dimensions, *dimension);
    }

    let model_name = match CString::new(model_name) {
        Ok(value) => value.into_raw(),
        Err(_) => {
            unsafe {
                *result = EmbeddingDimensionContractResult::default();
            }
            return -1;
        }
    };
    let num_supported_dimensions = supported_dimensions.len() as i32;
    let supported_dimensions = Box::into_raw(supported_dimensions.into_boxed_slice()) as *mut i32;

    unsafe {
        *result = EmbeddingDimensionContractResult {
            model_name,
            native_dimension: native_dimension as i32,
            supported_dimensions,
            num_supported_dimensions,
            error: false,
        };
    }

    0
}

fn normalize_model_type(model_type: &CStr) -> Option<&'static str> {
    let value = model_type.to_str().ok()?;
    match value.trim().to_ascii_lowercase().as_str() {
        "" | "bert" | "mmbert" => Some("mmbert"),
        other => {
            eprintln!("ERROR: unsupported ONNX embedding model type: {other}");
            None
        }
    }
}

/// Return the dimension contract of the loaded mmBERT model.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn get_embedding_dimension_contract(
    model_type: *const c_char,
    result: *mut EmbeddingDimensionContractResult,
) -> i32 {
    if model_type.is_null() || result.is_null() {
        return -1;
    }

    let model_type = unsafe { CStr::from_ptr(model_type) };
    let model_type = match normalize_model_type(model_type) {
        Some(value) => value,
        None => {
            unsafe { *result = EmbeddingDimensionContractResult::default() };
            return -1;
        }
    };
    let (native_dimension, supported_dimensions) = match loaded_embedding_dimension_contract() {
        Some(contract) => contract,
        None => {
            eprintln!("ERROR: mmBERT embedding model is not loaded");
            unsafe { *result = EmbeddingDimensionContractResult::default() };
            return -1;
        }
    };

    write_embedding_dimension_contract(result, model_type, native_dimension, &supported_dimensions)
}

/// Free a dimension contract returned by `get_embedding_dimension_contract`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_embedding_dimension_contract(result: *mut EmbeddingDimensionContractResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let contract = &mut *result;
        if !contract.model_name.is_null() {
            let _ = CString::from_raw(contract.model_name);
        }
        if !contract.supported_dimensions.is_null() && contract.num_supported_dimensions > 0 {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                contract.supported_dimensions,
                contract.num_supported_dimensions as usize,
            ));
        }
        *contract = EmbeddingDimensionContractResult::default();
    }
}
