//! Multi-modal Embedding FFI Module (ONNX Runtime)
//!
//! Provides FFI functions for multi-modal embedding (text, image, audio)
//! matching the candle-binding multimodal API.

use crate::ffi::types::MultiModalEmbeddingResult;
use crate::model_architectures::embedding::multimodal_embedding::MultiModalEmbeddingModel;
use std::ffi::{c_char, CStr};
use std::sync::OnceLock;

static GLOBAL_MULTIMODAL: OnceLock<MultiModalEmbeddingModel> = OnceLock::new();

/// Initialize multi-modal embedding model.
///
/// `model_path` must point to a directory containing
/// `text_encoder.onnx`, `image_encoder.onnx`, `audio_encoder.onnx`,
/// `tokenizer.json`, and optionally `config.json`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn init_multimodal_embedding_model(
    model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    if model_path.is_null() {
        eprintln!("Error: model_path is null");
        return false;
    }
    let path = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) if !s.is_empty() => s.to_string(),
            _ => {
                eprintln!("Error: invalid model_path");
                return false;
            }
        }
    };
    if GLOBAL_MULTIMODAL.get().is_some() {
        eprintln!("WARNING: multi-modal model already initialized");
        return true;
    }
    match MultiModalEmbeddingModel::load(&path, use_cpu) {
        Ok(model) => match GLOBAL_MULTIMODAL.set(model) {
            Ok(()) => true,
            Err(_) => {
                eprintln!("WARNING: multi-modal model already initialized by another thread");
                true
            }
        },
        Err(e) => {
            eprintln!("ERROR: Failed to load multi-modal model: {:?}", e);
            false
        }
    }
}

/// Encode text into a multi-modal embedding.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_text(
    text: *const c_char,
    target_dim: i32,
    result: *mut MultiModalEmbeddingResult,
) -> i32 {
    if text.is_null() || result.is_null() {
        return -1;
    }
    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(_) => return -1,
        }
    };
    let res = unsafe { &mut *result };
    *res = MultiModalEmbeddingResult::default();

    let model = match GLOBAL_MULTIMODAL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: multi-modal model not initialized");
            return -1;
        }
    };

    let start = std::time::Instant::now();
    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    match model.encode_text(text_str, dim) {
        Ok(emb) => {
            let len = emb.len();
            let mut data = emb.to_vec().into_boxed_slice();
            res.data = data.as_mut_ptr();
            res.length = len as i32;
            res.error = false;
            res.modality = 0; // text
            res.processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
            std::mem::forget(data);
            0
        }
        Err(e) => {
            eprintln!("Error encoding text: {:?}", e);
            -1
        }
    }
}

/// Encode pre-processed image pixels into a multi-modal embedding.
///
/// `pixel_data` is a [3*height*width] float32 array in [0,1], CHW layout.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_image(
    pixel_data: *const f32,
    height: i32,
    width: i32,
    target_dim: i32,
    result: *mut MultiModalEmbeddingResult,
) -> i32 {
    if pixel_data.is_null() || result.is_null() {
        return -1;
    }
    let res = unsafe { &mut *result };
    *res = MultiModalEmbeddingResult::default();

    if height <= 0 || width <= 0 {
        eprintln!(
            "Error: invalid image dimensions: height={}, width={}",
            height, width
        );
        res.error = true;
        return -1;
    }
    let h = height as usize;
    let w = width as usize;
    let len = match 3usize.checked_mul(h).and_then(|v| v.checked_mul(w)) {
        Some(l) => l,
        None => {
            eprintln!(
                "Error: image size overflow for height={}, width={}",
                height, width
            );
            res.error = true;
            return -1;
        }
    };
    let pixels = unsafe { std::slice::from_raw_parts(pixel_data, len) };

    let model = match GLOBAL_MULTIMODAL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: multi-modal model not initialized");
            return -1;
        }
    };

    let start = std::time::Instant::now();
    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    match model.encode_image(pixels, h, w, dim) {
        Ok(emb) => {
            let elen = emb.len();
            let mut data = emb.to_vec().into_boxed_slice();
            res.data = data.as_mut_ptr();
            res.length = elen as i32;
            res.error = false;
            res.modality = 1; // image
            res.processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
            std::mem::forget(data);
            0
        }
        Err(e) => {
            eprintln!("Error encoding image: {:?}", e);
            -1
        }
    }
}

/// Decode JPEG/PNG image bytes, resize to `(target_w, target_h)`, and convert
/// to CHW float32 pixels in `[0, 1]`. Returns `Err(String)` with a human-readable
/// cause on decode failure or dimension overflow.
///
/// This is the same decode+resize implementation as candle-binding's
/// `decode_resize_to_chw_f32` (candle-binding/src/ffi/embedding.rs): same
/// `image` crate version, same `FilterType::CatmullRom` cubic filter, same
/// CHW packing. Kept identical on purpose so the two bindings produce the
/// same pixels for the same input image — see #2166.
fn decode_resize_to_chw_f32(
    bytes: &[u8],
    target_w: u32,
    target_h: u32,
) -> Result<Vec<f32>, String> {
    let w = target_w as usize;
    let h = target_h as usize;
    let n_pixels = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(3))
        .ok_or_else(|| {
            format!(
                "target dimensions overflow usize: {}x{}x3",
                target_w, target_h
            )
        })?;

    let img = image::load_from_memory(bytes)
        .map_err(|e| format!("image decode failed: {:?}", e))?
        .to_rgb8();
    let resized = image::imageops::resize(
        &img,
        target_w,
        target_h,
        image::imageops::FilterType::CatmullRom,
    );

    let raw = resized.as_raw();
    debug_assert_eq!(
        raw.len(),
        n_pixels,
        "resize produced unexpected pixel count"
    );
    let mut pixels = vec![0f32; n_pixels];
    let plane = h * w;
    let inv = 1.0f32 / 255.0;
    for i in 0..plane {
        let base = i * 3;
        pixels[i] = raw[base] as f32 * inv;
        pixels[plane + i] = raw[base + 1] as f32 * inv;
        pixels[2 * plane + i] = raw[base + 2] as f32 * inv;
    }
    Ok(pixels)
}

/// Encode image bytes: decode + resize (Catmull-Rom cubic, matching
/// candle-binding) + forward.
///
/// Preferred entry point for image embedding from raw JPEG/PNG bytes — keeps
/// preprocessing in Rust so both bindings share the same pixel pipeline
/// instead of each doing their own decode/resize.
///
/// # Parameters
/// - `bytes_ptr`: Pointer to raw JPEG/PNG bytes
/// - `bytes_len`: Number of bytes
/// - `target_dim`: Target embedding dimension (0 for default)
/// - `result`: Output pointer for embedding result
///
/// # Returns
/// 0 on success, -1 on error
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_image_from_bytes(
    bytes_ptr: *const u8,
    bytes_len: usize,
    target_dim: i32,
    result: *mut MultiModalEmbeddingResult,
) -> i32 {
    if bytes_ptr.is_null() || result.is_null() || bytes_len == 0 {
        eprintln!("Error: null/empty input to multimodal_encode_image_from_bytes");
        return -1;
    }
    let res = unsafe { &mut *result };
    *res = MultiModalEmbeddingResult::default();

    let model = match GLOBAL_MULTIMODAL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: multi-modal model not initialized");
            return -1;
        }
    };

    let start = std::time::Instant::now();
    let bytes = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };

    const TARGET_W: u32 = 512;
    const TARGET_H: u32 = 512;
    let pixels = match decode_resize_to_chw_f32(bytes, TARGET_W, TARGET_H) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Error: {}", e);
            return -1;
        }
    };

    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    match model.encode_image(&pixels, TARGET_H as usize, TARGET_W as usize, dim) {
        Ok(emb) => {
            let elen = emb.len();
            let mut data = emb.to_vec().into_boxed_slice();
            res.data = data.as_mut_ptr();
            res.length = elen as i32;
            res.error = false;
            res.modality = 1; // image
            res.processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
            std::mem::forget(data);
            0
        }
        Err(e) => {
            eprintln!("Error encoding image: {:?}", e);
            -1
        }
    }
}

#[cfg(test)]
mod decode_resize_tests {
    use super::decode_resize_to_chw_f32;

    #[test]
    fn rejects_invalid_bytes() {
        let garbage = b"not a real image file";
        let result = decode_resize_to_chw_f32(garbage, 8, 8);
        assert!(result.is_err(), "decoder should reject garbage bytes");
        assert!(
            result.unwrap_err().contains("image decode failed"),
            "error should mention decode failure"
        );
    }

    #[test]
    fn overflow_guard_on_huge_dims() {
        // 2x2 white PNG, minimal valid image so decode succeeds and the
        // overflow guard on the *target* dimensions is what's exercised.
        let bytes: [u8; 79] = [
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x08, 0x02, 0x00, 0x00,
            0x00, 0xfd, 0xd4, 0x9a, 0x73, 0x00, 0x00, 0x00, 0x16, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9c, 0x63, 0xfc, 0xff, 0xff, 0x3f, 0x03, 0x03, 0x03, 0x13, 0x03, 0x03, 0x03, 0x03,
            0x03, 0x03, 0x00, 0x24, 0x06, 0x03, 0x01, 0xfc, 0x35, 0xde, 0x9b, 0x00, 0x00, 0x00,
            0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        let result = decode_resize_to_chw_f32(&bytes, u32::MAX, u32::MAX);
        assert!(
            result.is_err(),
            "u32::MAX x u32::MAX dimensions should be rejected"
        );
        assert!(
            result.unwrap_err().contains("overflow"),
            "error should mention overflow"
        );
    }
}

/// Golden-image parity test for #2166: onnx-binding's `decode_resize_to_chw_f32`
/// (this file) must produce the same pixels as candle-binding's
/// `decode_resize_to_chw_f32` (candle-binding/src/ffi/embedding.rs) for the
/// same input image, since both bindings are supposed to share one
/// preprocessing contract.
///
/// `test_data/image_resize_parity_reference.json` holds the reference CHW
/// pixel arrays, generated by running candle-binding's actual resize (same
/// `image` crate version, same `FilterType::CatmullRom`) on the three real
/// fixtures already checked into
/// `e2e/testcases/testdata/image-fixtures/`, downscaled to 32x32 to keep the
/// fixture small — the resize algorithm doesn't change behavior based on
/// target size, so 32x32 exercises the same code path as the real 512x512
/// production size.
///
/// Declared tolerance: max per-channel absolute difference < 1e-4 (pixels are
/// in [0, 1], so this is tighter than 1/2550 of the full range). Not exact
/// equality, since floating-point rounding can differ by an ULP or two across
/// compilers/CPUs even for identical source code; 1e-4 is loose enough to
/// absorb that while still catching a real algorithmic regression (the old
/// nearest-neighbor bug produced differences up to 0.66 on these fixtures).
#[cfg(test)]
mod golden_image_parity {
    use super::decode_resize_to_chw_f32;
    use serde::Deserialize;

    const TOLERANCE: f32 = 1e-4;

    #[derive(Deserialize)]
    struct ReferenceCase {
        name: String,
        target_w: u32,
        target_h: u32,
        chw_pixels: Vec<f32>,
    }

    fn reference_cases() -> Vec<ReferenceCase> {
        let json = include_str!("../../test_data/image_resize_parity_reference.json");
        serde_json::from_str(json).expect("parse image_resize_parity_reference.json")
    }

    fn assert_matches_reference(name: &str, bytes: &[u8]) {
        let cases = reference_cases();
        let case = cases
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("no reference case named {name}"));

        let pixels = decode_resize_to_chw_f32(bytes, case.target_w, case.target_h)
            .unwrap_or_else(|e| panic!("resize failed for {name}: {e}"));

        assert_eq!(
            pixels.len(),
            case.chw_pixels.len(),
            "{name}: pixel count mismatch"
        );
        let max_diff = pixels
            .iter()
            .zip(case.chw_pixels.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff < TOLERANCE,
            "{name}: max_diff={max_diff} exceeds declared tolerance {TOLERANCE}"
        );
    }

    #[test]
    fn code_screenshot_matches_reference() {
        assert_matches_reference(
            "code_screenshot",
            include_bytes!("../../../e2e/testcases/testdata/image-fixtures/code_screenshot.jpg"),
        );
    }

    #[test]
    fn conference_room_matches_reference() {
        assert_matches_reference(
            "conference_room",
            include_bytes!("../../../e2e/testcases/testdata/image-fixtures/conference_room.jpg"),
        );
    }

    #[test]
    fn passport_sample_matches_reference() {
        assert_matches_reference(
            "passport_sample",
            include_bytes!("../../../e2e/testcases/testdata/image-fixtures/passport_sample.jpg"),
        );
    }
}

/// Encode mel spectrogram into a multi-modal embedding.
///
/// `mel_data` is a [n_mels*time_frames] float32 array in row-major order.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn multimodal_encode_audio(
    mel_data: *const f32,
    n_mels: i32,
    time_frames: i32,
    target_dim: i32,
    result: *mut MultiModalEmbeddingResult,
) -> i32 {
    if mel_data.is_null() || result.is_null() {
        return -1;
    }
    let res = unsafe { &mut *result };
    *res = MultiModalEmbeddingResult::default();

    if n_mels <= 0 || time_frames <= 0 {
        eprintln!(
            "Error: n_mels and time_frames must be > 0 (got n_mels={}, time_frames={})",
            n_mels, time_frames
        );
        res.error = true;
        return -1;
    }
    let nm = n_mels as usize;
    let tf = time_frames as usize;
    let len = match nm.checked_mul(tf) {
        Some(l) => l,
        None => {
            eprintln!(
                "Error: overflow computing mel spectrogram length (n_mels={}, time_frames={})",
                n_mels, time_frames
            );
            res.error = true;
            return -1;
        }
    };
    let mel = unsafe { std::slice::from_raw_parts(mel_data, len) };

    let model = match GLOBAL_MULTIMODAL.get() {
        Some(m) => m,
        None => {
            eprintln!("Error: multi-modal model not initialized");
            return -1;
        }
    };

    let start = std::time::Instant::now();
    let dim = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    match model.encode_audio(mel, nm, tf, dim) {
        Ok(emb) => {
            let elen = emb.len();
            let mut data = emb.to_vec().into_boxed_slice();
            res.data = data.as_mut_ptr();
            res.length = elen as i32;
            res.error = false;
            res.modality = 2; // audio
            res.processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;
            std::mem::forget(data);
            0
        }
        Err(e) => {
            eprintln!("Error encoding audio: {:?}", e);
            -1
        }
    }
}

/// Free a multi-modal embedding result's data buffer.
#[no_mangle]
pub extern "C" fn free_multimodal_embedding(data: *mut f32, length: i32) {
    if !data.is_null() && length > 0 {
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, length as usize));
        }
    }
}
