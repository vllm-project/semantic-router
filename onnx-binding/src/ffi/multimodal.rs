//! Multi-modal Embedding FFI Module (ONNX Runtime)
//!
//! Provides FFI functions for multi-modal embedding (text, image, audio)
//! matching the candle-binding multimodal API.

use crate::ffi::types::MultiModalEmbeddingResult;
use crate::model_architectures::embedding::multimodal_embedding::MultiModalEmbeddingModel;
use std::ffi::{c_char, CStr};
use std::sync::{Mutex, OnceLock};

use super::embedding_error_status;
use super::init_once::{initialize_once_with_identity, InitializedModel, ModelInitIdentity};

static GLOBAL_MULTIMODAL: OnceLock<InitializedModel<MultiModalEmbeddingModel>> = OnceLock::new();
static MULTIMODAL_INIT_LOCK: Mutex<()> = Mutex::new(());

/// Initialize multi-modal embedding model.
///
/// `model_path` must point to a directory containing
/// `text_encoder.onnx`, `image_encoder.onnx`, `audio_encoder.onnx`,
/// `tokenizer.json`, and optionally `config.json`.
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
    let identity = ModelInitIdentity {
        model_path: path.clone(),
        use_cpu,
    };
    let result =
        initialize_once_with_identity(&GLOBAL_MULTIMODAL, &MULTIMODAL_INIT_LOCK, identity, || {
            MultiModalEmbeddingModel::load(&path, use_cpu)
                .map_err(|error| format!("failed to load multi-modal model: {error:?}"))
        });
    if let Err(error) = result {
        eprintln!("ERROR: multi-modal model initialization failed: {error}");
        return false;
    }
    true
}

/// Encode text into a multi-modal embedding.
///
/// Returns 0 on success, -3 when the tokenizer context is exceeded, and -1
/// for invalid FFI arguments, unavailable models, or internal failures.
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
        Some(m) => &m.value,
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
            embedding_error_status(&e)
        }
    }
}

/// Encode pre-processed image pixels into a multi-modal embedding.
///
/// `pixel_data` is a [3*height*width] float32 array in [0,1], CHW layout.
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
    const MAX_IMAGE_SIDE: usize = 8192;
    const MAX_IMAGE_PIXELS: usize = 16_777_216;
    if h > MAX_IMAGE_SIDE || w > MAX_IMAGE_SIDE || h.saturating_mul(w) > MAX_IMAGE_PIXELS {
        eprintln!("Error: image dimensions exceed the native input budget");
        res.error = true;
        return -1;
    }
    let len = match 3usize.checked_mul(h).and_then(|v| v.checked_mul(w)) {
        Some(l) if l <= (isize::MAX as usize) / std::mem::size_of::<f32>() => l,
        _ => {
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
        Some(m) => &m.value,
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

/// Encode mel spectrogram into a multi-modal embedding.
///
/// `mel_data` is a [n_mels*time_frames] float32 array in row-major order.
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
        Some(l) if l <= (isize::MAX as usize) / std::mem::size_of::<f32>() => l,
        _ => {
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
        Some(m) => &m.value,
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
            let _ =
                Box::from_raw(std::slice::from_raw_parts_mut(data, length as usize) as *mut [f32]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_tensor_ffi_rejects_unsafe_dimensions_before_slice_creation() {
        let one_value = 0_f32;
        let mut result = MultiModalEmbeddingResult::default();

        assert_eq!(
            multimodal_encode_image(&one_value, -1, -1, 0, &mut result),
            -1
        );
        assert!(result.error);
        assert_eq!(
            multimodal_encode_image(&one_value, 8193, 1, 0, &mut result),
            -1
        );
        assert!(result.error);

        assert_eq!(
            multimodal_encode_audio(&one_value, -1, -1, 0, &mut result),
            -1
        );
        assert!(result.error);
        assert_eq!(
            multimodal_encode_audio(&one_value, i32::MAX, i32::MAX, 0, &mut result),
            -1
        );
        assert!(result.error);
    }
}
