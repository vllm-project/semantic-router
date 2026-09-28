//! FFI for splitting an input into the windows an embedding model can read.
//!
//! `get_text_embedding` truncates at the model's window, so a caller that embeds
//! a long input once sees only its opening. These ranges let a caller embed every
//! part of it instead.

use crate::ffi::init::BERT_SIMILARITY;
use std::ffi::{c_char, CStr};

/// Byte ranges of an input that each fit the model's window, as start and end
/// pairs laid out back to back in `offsets`.
#[repr(C)]
#[derive(Debug)]
pub struct TextWindowsResult {
    pub offsets: *mut i32,
    pub window_count: i32,
    pub error: bool,
}

fn failed_windows() -> TextWindowsResult {
    TextWindowsResult {
        offsets: std::ptr::null_mut(),
        window_count: 0,
        error: true,
    }
}

/// Byte ranges of `text` that each fit the model's embedding window.
///
/// Free the result with `free_text_windows`.
///
/// # Safety
/// - `text` must be a valid null-terminated C string
#[no_mangle]
pub unsafe extern "C" fn get_text_windows(
    text: *const c_char,
    max_length: i32,
) -> TextWindowsResult {
    let text = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(text) => text,
        Err(_) => return failed_windows(),
    };
    let bert = match BERT_SIMILARITY.get() {
        Some(bert) => bert.clone(),
        None => {
            eprintln!("BERT model not initialized");
            return failed_windows();
        }
    };
    let max_length_opt = if max_length <= 0 {
        None
    } else {
        Some(max_length as usize)
    };
    let ranges = match bert.window_byte_ranges(text, max_length_opt) {
        Ok(ranges) => ranges,
        Err(e) => {
            eprintln!("Failed to window text: {e}");
            return failed_windows();
        }
    };

    let mut flat: Vec<i32> = Vec::with_capacity(ranges.len() * 2);
    for (start, end) in &ranges {
        flat.push(*start as i32);
        flat.push(*end as i32);
    }
    flat.shrink_to_fit();
    let window_count = ranges.len() as i32;
    let offsets = flat.as_mut_ptr();
    std::mem::forget(flat);
    TextWindowsResult {
        offsets,
        window_count,
        error: false,
    }
}

/// Free the ranges returned by `get_text_windows`.
///
/// # Safety
/// - `result` must come from `get_text_windows` and must not be freed twice
#[no_mangle]
pub unsafe extern "C" fn free_text_windows(result: TextWindowsResult) {
    if result.offsets.is_null() || result.window_count <= 0 {
        return;
    }
    let length = result.window_count as usize * 2;
    let _ = unsafe { Vec::from_raw_parts(result.offsets, length, length) };
}
