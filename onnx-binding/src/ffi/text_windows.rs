//! Tokenizer-backed byte windows for the Candle-compatible RAG embedding API.

use super::embedding::GLOBAL_MMBERT_MODEL;
use std::ffi::{c_char, CStr};
use tokenizers::Tokenizer;

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

fn text_windows(
    tokenizer: &Tokenizer,
    text: &str,
    window: usize,
) -> Result<Vec<(usize, usize)>, String> {
    let mut tokenizer = tokenizer.clone();
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    let encoding = tokenizer.encode(text, false).map_err(|e| e.to_string())?;
    let offsets: Vec<_> = encoding
        .get_offsets()
        .iter()
        .copied()
        .filter(|(start, end)| end > start)
        .collect();
    // mmBERT adds a start and end token when each window is embedded.
    let budget = window.saturating_sub(2).max(1);
    let stride = budget.div_ceil(2);
    let mut windows = Vec::new();
    let mut start = 0;
    while start < offsets.len() {
        let end = (start + budget).min(offsets.len());
        windows.push((offsets[start].0, offsets[end - 1].1));
        if end == offsets.len() {
            break;
        }
        start += stride;
    }
    Ok(windows)
}

/// Return overlapping byte ranges that fit the loaded embedding model's window.
/// A positive `max_length` may further reduce the model's context limit.
///
/// # Safety
/// `text` must point to a valid null-terminated C string, or be null.
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
    let model_window = model.config().max_position_embeddings;
    let window = if max_length > 0 {
        model_window.min(max_length as usize)
    } else {
        model_window
    };
    let ranges = match text_windows(model.tokenizer(), text, window) {
        Ok(ranges) => ranges,
        Err(_) => return failed_windows(),
    };
    if ranges.is_empty() {
        return TextWindowsResult {
            offsets: std::ptr::null_mut(),
            window_count: 0,
            error: false,
        };
    }
    let Ok(window_count) = i32::try_from(ranges.len()) else {
        return failed_windows();
    };
    let flat = ranges
        .into_iter()
        .flat_map(|(start, end)| [start as i32, end as i32])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    TextWindowsResult {
        offsets: Box::into_raw(flat) as *mut i32,
        window_count,
        error: false,
    }
}

/// Release the byte ranges returned by `get_text_windows` exactly once.
///
/// # Safety
/// `result` must be an unfreed result from `get_text_windows`.
#[no_mangle]
pub unsafe extern "C" fn free_text_windows(result: TextWindowsResult) {
    if !result.offsets.is_null() && result.window_count > 0 {
        let offsets =
            std::ptr::slice_from_raw_parts_mut(result.offsets, result.window_count as usize * 2);
        drop(unsafe { Box::from_raw(offsets) });
    }
}

#[cfg(test)]
mod tests {
    use super::{free_text_windows, get_text_windows, text_windows};
    use std::ffi::CString;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::WhitespaceSplit;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{Tokenizer, TruncationParams};

    fn tokenizer() -> Tokenizer {
        let model = WordLevel::builder()
            .vocab(
                [
                    "[UNK]", "[CLS]", "[SEP]", "one", "two", "三", "四", "five", "six", "七",
                ]
                .into_iter()
                .enumerate()
                .map(|(id, token)| (token.to_string(), id as u32))
                .collect(),
            )
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("[CLS] $A [SEP]")
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        tokenizer
    }

    #[test]
    fn windows_cover_tail_overlap_and_preserve_utf8_boundaries() {
        let mut tokenizer = tokenizer();
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 4,
                ..Default::default()
            }))
            .unwrap();
        let text = "one two 三 四 five six 七";
        let windows = text_windows(&tokenizer, text, 6).unwrap();
        let slices: Vec<_> = windows
            .iter()
            .map(|&(start, end)| &text[start..end])
            .collect();
        assert_eq!(slices, ["one two 三 四", "三 四 five six", "five six 七"]);
        tokenizer.with_truncation(None).unwrap();
        for slice in slices {
            assert!(tokenizer.encode(slice, true).unwrap().len() <= 6);
        }
    }

    #[test]
    fn short_and_empty_inputs_need_no_extra_windows() {
        let tokenizer = tokenizer();
        assert_eq!(text_windows(&tokenizer, "one 三", 6).unwrap(), [(0, 7)]);
        assert!(text_windows(&tokenizer, "", 6).unwrap().is_empty());
        assert!(text_windows(&tokenizer, "   ", 6).unwrap().is_empty());
    }

    #[test]
    fn tiny_requested_windows_keep_the_candle_minimum_one_content_token() {
        let tokenizer = tokenizer();
        for window in [1, 2] {
            assert_eq!(
                text_windows(&tokenizer, "one 三", window).unwrap(),
                [(0, 3), (4, 7)]
            );
        }
    }

    #[test]
    fn ffi_reports_invalid_input_and_uninitialized_model() {
        for text in [std::ptr::null(), c"text".as_ptr()] {
            let result = unsafe { get_text_windows(text, 0) };
            assert!(result.error);
            assert!(result.offsets.is_null());
            unsafe { free_text_windows(result) };
        }
        let invalid = CString::new(vec![0xff]).unwrap();
        let result = unsafe { get_text_windows(invalid.as_ptr(), 0) };
        assert!(result.error);
        unsafe { free_text_windows(result) };
    }
}
