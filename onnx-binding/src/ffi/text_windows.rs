//! FFI for splitting input into byte ranges that fit the ONNX embedding model.

use crate::ffi::embedding::GLOBAL_MMBERT_MODEL;
use std::ffi::{c_char, CStr};

/// Byte ranges laid out as consecutive start/end pairs in `offsets`.
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

fn window_ranges(offsets: &[(usize, usize)], budget: usize, stride: usize) -> Vec<(usize, usize)> {
    if offsets.is_empty() || budget == 0 {
        return Vec::new();
    }

    let stride = stride.clamp(1, budget);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < offsets.len() {
        let end = (start + budget).min(offsets.len());
        ranges.push((offsets[start].0, offsets[end - 1].1));
        if end == offsets.len() {
            break;
        }
        start += stride;
    }
    ranges
}

/// Return overlapping byte ranges that each fit the loaded mmBERT model.
///
/// `max_length <= 0` selects the model's configured context length. Two token
/// positions are reserved for the special tokens added during embedding.
///
/// # Safety
/// `text` must point to a valid, null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn get_text_windows(
    text: *const c_char,
    max_length: i32,
) -> TextWindowsResult {
    if text.is_null() {
        return failed_windows();
    }
    let text = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(text) => text,
        Err(_) => return failed_windows(),
    };
    let Some(model_lock) = GLOBAL_MMBERT_MODEL.get() else {
        eprintln!("mmBERT model not initialized");
        return failed_windows();
    };
    let model = model_lock.lock();
    let window = if max_length <= 0 {
        model.config().max_position_embeddings
    } else {
        max_length as usize
    };
    let mut tokenizer = model.tokenizer().clone();
    if let Err(error) = tokenizer.with_truncation(None) {
        eprintln!("Failed to disable tokenizer truncation for windowing: {error}");
        return failed_windows();
    }
    let encoding = match tokenizer.encode(text, false) {
        Ok(encoding) => encoding,
        Err(error) => {
            eprintln!("Failed to tokenize text for windowing: {error}");
            return failed_windows();
        }
    };
    let offsets: Vec<(usize, usize)> = encoding
        .get_offsets()
        .iter()
        .copied()
        .filter(|(start, end)| end > start)
        .collect();
    let budget = window.saturating_sub(2).max(1);
    let ranges = window_ranges(&offsets, budget, budget.div_ceil(2));

    let mut flat = Vec::with_capacity(ranges.len() * 2);
    for (start, end) in &ranges {
        let (Ok(start), Ok(end)) = (i32::try_from(*start), i32::try_from(*end)) else {
            return failed_windows();
        };
        flat.push(start);
        flat.push(end);
    }
    let Ok(window_count) = i32::try_from(ranges.len()) else {
        return failed_windows();
    };
    let offsets = Box::into_raw(flat.into_boxed_slice()).cast::<i32>();
    TextWindowsResult {
        offsets,
        window_count,
        error: false,
    }
}

/// Free ranges returned by `get_text_windows`.
///
/// # Safety
/// `result` must originate from `get_text_windows` and must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn free_text_windows(result: TextWindowsResult) {
    if result.offsets.is_null() || result.window_count <= 0 {
        return;
    }
    let length = result.window_count as usize * 2;
    let slice = std::ptr::slice_from_raw_parts_mut(result.offsets, length);
    let _ = unsafe { Box::from_raw(slice) };
}

#[cfg(test)]
mod tests {
    use super::window_ranges;

    #[test]
    fn ranges_cover_all_tokens() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(
            window_ranges(&offsets, 2, 2),
            vec![(0, 5), (6, 11), (12, 14)]
        );
        assert_eq!(window_ranges(&offsets, 8, 4), vec![(0, 14)]);
    }

    #[test]
    fn ranges_overlap_when_stride_is_shorter_than_budget() {
        let offsets = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)];
        assert_eq!(window_ranges(&offsets, 4, 2), vec![(0, 11), (6, 14)]);
    }

    #[test]
    fn ranges_handle_empty_and_zero_budget() {
        assert!(window_ranges(&[], 4, 2).is_empty());
        assert!(window_ranges(&[(0, 2)], 0, 1).is_empty());
        assert_eq!(window_ranges(&[(0, 2)], 1, 0), vec![(0, 2)]);
    }
}
