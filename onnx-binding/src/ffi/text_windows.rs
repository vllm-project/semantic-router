//! RAG windowing through the same loaded mmBERT model used by `get_embedding`.

use super::embedding::GLOBAL_MMBERT_MODEL;
use std::ffi::{c_char, CStr};

#[repr(C)]
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

/// Return overlapping byte ranges that fit the loaded embedding model.
///
/// # Safety
/// `text` must be null or a valid null-terminated C string. Free a successful
/// result exactly once with `free_text_windows`.
#[no_mangle]
pub unsafe extern "C" fn get_text_windows(
    text: *const c_char,
    max_length: i32,
) -> TextWindowsResult {
    if text.is_null() {
        return failed_windows();
    }
    let text = match unsafe { CStr::from_ptr(text) }.to_str() {
        Ok(text) if text.len() <= i32::MAX as usize => text,
        _ => return failed_windows(),
    };
    let Some(model_lock) = GLOBAL_MMBERT_MODEL.get() else {
        return failed_windows();
    };
    let model = model_lock.lock();
    let limit = (max_length > 0).then_some(max_length as usize);
    match crate::core::text_windows::text_windows(
        model.tokenizer(),
        text,
        model.config().max_position_embeddings,
        limit,
    ) {
        Ok(ranges) => pack_windows(ranges),
        Err(error) => {
            eprintln!("Failed to window ONNX embedding text: {error}");
            failed_windows()
        }
    }
}

fn pack_windows(ranges: Vec<(usize, usize)>) -> TextWindowsResult {
    let Ok(window_count) = i32::try_from(ranges.len()) else {
        return failed_windows();
    };
    let offsets: Result<Vec<i32>, _> = ranges
        .into_iter()
        .flat_map(|(start, end)| [start, end])
        .map(i32::try_from)
        .collect();
    let Ok(offsets) = offsets else {
        return failed_windows();
    };
    TextWindowsResult {
        offsets: if offsets.is_empty() {
            std::ptr::null_mut()
        } else {
            Box::into_raw(offsets.into_boxed_slice()) as *mut i32
        },
        window_count,
        error: false,
    }
}

/// Release a window array using its original boxed-slice allocation layout.
///
/// # Safety
/// `result` must come from `get_text_windows` and must not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn free_text_windows(result: TextWindowsResult) {
    if !result.offsets.is_null() && result.window_count > 0 {
        let slice =
            std::ptr::slice_from_raw_parts_mut(result.offsets, result.window_count as usize * 2);
        drop(unsafe { Box::from_raw(slice) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_windows_reject_invalid_input_and_unloaded_model() {
        for text in [
            std::ptr::null(),
            c"query".as_ptr(),
            c"".as_ptr(),
            c"\xff".as_ptr(),
        ] {
            let result = unsafe { get_text_windows(text, 0) };
            assert!(result.error);
            assert!(result.offsets.is_null());
            assert_eq!(result.window_count, 0);
            unsafe { free_text_windows(result) };
        }
    }

    #[test]
    fn text_windows_round_trip_empty_and_multiple_ranges() {
        for ranges in [vec![], vec![(0, 4), (2, 8), (6, 12)]] {
            let expected: Vec<_> = ranges
                .iter()
                .flat_map(|&(start, end)| [start as i32, end as i32])
                .collect();
            let result = pack_windows(ranges);
            assert!(!result.error);
            assert_eq!(result.window_count as usize * 2, expected.len());
            if expected.is_empty() {
                assert!(result.offsets.is_null());
            } else {
                assert_eq!(
                    unsafe { std::slice::from_raw_parts(result.offsets, expected.len()) },
                    expected
                );
            }
            unsafe { free_text_windows(result) };
        }
    }

    #[test]
    fn text_windows_reject_offset_overflow() {
        let result = pack_windows(vec![(0, i32::MAX as usize + 1)]);
        assert!(result.error);
        assert!(result.offsets.is_null());
    }
}
