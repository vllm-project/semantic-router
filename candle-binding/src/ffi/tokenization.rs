//! FFI Tokenization Functions

use crate::ffi::init::BERT_SIMILARITY;
use crate::ffi::types::*;
use std::ffi::{c_char, CStr};

/// Tokenize text
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub extern "C" fn tokenize_text(text: *const c_char, max_length: i32) -> TokenizationResult {
    // Adapted from lib.rs:410-483 to match types.rs TokenizationResult structure
    let text = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => {
                return TokenizationResult {
                    token_ids: std::ptr::null_mut(),
                    token_count: 0,
                    tokens: std::ptr::null_mut(),
                    error: true,
                }
            }
        }
    };

    let bert = match BERT_SIMILARITY.get() {
        Some(b) => b.clone(),
        None => {
            eprintln!("BERT model not initialized");
            return TokenizationResult {
                token_ids: std::ptr::null_mut(),
                token_count: 0,
                tokens: std::ptr::null_mut(),
                error: true,
            };
        }
    };

    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };

    // Call the actual tokenization method
    match bert.tokenize_text(text, max_length_opt) {
        Ok((token_ids, token_strings)) => {
            let token_count = token_ids.len() as i32;

            // Convert Vec<i32> to C-compatible array
            let mut token_ids_vec = token_ids.into_boxed_slice();
            let token_ids_ptr = token_ids_vec.as_mut_ptr();
            std::mem::forget(token_ids_vec); // Prevent deallocation

            // Convert Vec<String> to C-compatible char** array
            let mut c_strings: Vec<*mut c_char> = token_strings
                .into_iter()
                .map(|s| match std::ffi::CString::new(s) {
                    Ok(cs) => cs.into_raw(),
                    Err(_) => std::ptr::null_mut(),
                })
                .collect();

            let tokens_ptr = if c_strings.is_empty() {
                std::ptr::null_mut()
            } else {
                let ptr = c_strings.as_mut_ptr();
                std::mem::forget(c_strings); // Prevent deallocation
                ptr
            };

            TokenizationResult {
                token_ids: token_ids_ptr,
                token_count,
                tokens: tokens_ptr,
                error: false,
            }
        }
        Err(e) => {
            eprintln!("Error tokenizing text: {}", e);
            TokenizationResult {
                token_ids: std::ptr::null_mut(),
                token_count: 0,
                tokens: std::ptr::null_mut(),
                error: true,
            }
        }
    }
}
