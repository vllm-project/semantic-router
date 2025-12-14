//! FFI bindings for vision transformer image embeddings

use crate::ffi::types::EmbeddingResult;
use crate::model_architectures::vision::{
    clip_encoder::{get_vision_encoder, init_vision_encoder},
    image_utils::preprocess_image,
};
use crate::model_architectures::vision::VisionEncoder;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use candle_core::{Device, IndexOp};

/// Initialize vision encoder (call once at startup)
///
/// # Arguments
/// - `model_id`: HuggingFace model ID (e.g., "openai/clip-vit-base-patch32") or local path
/// - `device_type`: Device type string ("cpu" or "cuda")
///
/// # Returns
/// - `true` if initialization succeeded
/// - `false` if initialization failed
#[no_mangle]
pub extern "C" fn init_vision_encoder_ffi(
    model_id: *const c_char,
    device_type: *const c_char,
) -> bool {
    unsafe {
        let model_id_str = match CStr::from_ptr(model_id).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };
        
        let device_type_str = match CStr::from_ptr(device_type).to_str() {
            Ok(s) => s,
            Err(_) => return false,
        };
        
        let device = if device_type_str == "cuda" {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        } else {
            Device::Cpu
        };
        
        eprintln!("init_vision_encoder_ffi: Starting initialization: model_id={}, device={:?}", model_id_str, device);
        match init_vision_encoder(model_id_str, device) {
            Ok(_) => {
                eprintln!("init_vision_encoder_ffi: Initialization succeeded");
                true
            }
            Err(e) => {
                eprintln!("ERROR: init_vision_encoder_ffi failed: {}", e);
                eprintln!("ERROR: Error details: {:?}", e);
                false
            }
        }
    }
}

/// Extract image embedding from image data
///
/// # Arguments
/// - `image_data`: Pointer to raw image bytes
/// - `data_len`: Length of image data in bytes
/// - `mime_type`: MIME type string (e.g., "image/jpeg", "image/png")
///
/// # Returns
/// - Pointer to EmbeddingResult (must be freed with free_image_embedding_result)
#[no_mangle]
pub extern "C" fn get_image_embedding(
    image_data: *const u8,
    data_len: usize,
    mime_type: *const c_char,
) -> *mut EmbeddingResult {
    unsafe {
        // Get encoder
        let encoder = match get_vision_encoder() {
            Some(e) => e,
            None => {
                // Try to auto-initialize with default model
                eprintln!("Vision encoder not initialized, attempting auto-initialization...");
                let default_model = CString::new("openai/clip-vit-base-patch32").unwrap();
                if !init_vision_encoder_ffi(default_model.as_ptr(), CString::new("cpu").unwrap().as_ptr()) {
                    eprintln!("ERROR: Vision encoder auto-initialization failed");
                    return EmbeddingResult::error("Vision encoder not initialized and auto-initialization failed");
                }
                eprintln!("Vision encoder auto-initialization succeeded");
                match get_vision_encoder() {
                    Some(e) => e,
                    None => {
                        eprintln!("ERROR: Failed to get vision encoder after initialization");
                        return EmbeddingResult::error("Failed to get vision encoder after initialization");
                    }
                }
            }
        };
        
        // Convert C strings to Rust
        let mime_type_str = match CStr::from_ptr(mime_type).to_str() {
            Ok(s) => s,
            Err(_) => return EmbeddingResult::error("Invalid MIME type string"),
        };
        
        // Get image data
        let image_bytes = std::slice::from_raw_parts(image_data, data_len);
        
        // Preprocess image
        let device = Device::Cpu; // TODO: Get from encoder
        eprintln!("Preprocessing image (size: {} bytes, mime: {})", data_len, mime_type_str);
        let image_tensor = match preprocess_image(image_bytes, mime_type_str, &device) {
            Ok(t) => {
                eprintln!("Image preprocessing succeeded");
                t
            }
            Err(e) => {
                eprintln!("ERROR: Image preprocessing failed: {}", e);
                return EmbeddingResult::error(&format!("Image preprocessing failed: {}", e));
            }
        };
        
        // Get embedding
        eprintln!("Locking encoder for inference...");
        let encoder_guard = match encoder.lock() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("ERROR: Failed to lock encoder: {:?}", e);
                return EmbeddingResult::error("Failed to lock encoder");
            }
        };
        
        eprintln!("Running CLIP vision transformer inference...");
        let embedding_tensor = match encoder_guard.encode(&image_tensor) {
            Ok(e) => {
                eprintln!("CLIP encoding succeeded");
                e
            }
            Err(e) => {
                eprintln!("ERROR: CLIP encoding failed: {}", e);
                return EmbeddingResult::error(&format!("Encoding failed: {}", e));
            }
        };
        
        // Convert to Vec<f32>
        // The embedding tensor is [batch, embedding_dim] = [1, 512]
        // We need to extract just the embedding vector [512]
        eprintln!("Converting tensor to Vec<f32>...");
        eprintln!("Embedding tensor shape: {:?}", embedding_tensor.shape());
        
        let embedding_vec = match embedding_tensor.rank() {
            2 => {
                // Remove batch dimension: [1, 512] -> [512]
                let batch_size = match embedding_tensor.dim(0) {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("ERROR: Failed to get batch dimension: {}", e);
                        return EmbeddingResult::error(&format!("Failed to get batch dimension: {}", e));
                    }
                };
                if batch_size != 1 {
                    eprintln!("ERROR: Expected batch size 1, got {}", batch_size);
                    return EmbeddingResult::error(&format!("Unexpected batch size: {}", batch_size));
                }
                // Get the first (and only) row: [1, 512] -> [512]
                let embedding_1d = match embedding_tensor.i(0) {
                    Ok(t) => t,
                    Err(e) => {
                        eprintln!("ERROR: Failed to index tensor: {}", e);
                        return EmbeddingResult::error(&format!("Failed to index tensor: {}", e));
                    }
                };
                match embedding_1d.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("ERROR: Failed to convert 1D tensor to vec: {}", e);
                        return EmbeddingResult::error(&format!("Failed to convert 1D tensor to vec: {}", e));
                    }
                }
            }
            1 => {
                // Already 1D
                match embedding_tensor.to_vec1::<f32>() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("ERROR: Failed to convert 1D tensor to vec: {}", e);
                        return EmbeddingResult::error(&format!("Failed to convert 1D tensor to vec: {}", e));
                    }
                }
            }
            rank => {
                eprintln!("ERROR: Unexpected tensor rank: {}", rank);
                return EmbeddingResult::error(&format!("Unexpected tensor rank: {}, expected 1 or 2", rank));
            }
        };
        
        eprintln!("Tensor conversion succeeded (dim: {})", embedding_vec.len());
        
        // Create result
        EmbeddingResult::success(embedding_vec)
    }
}

/// Free embedding result memory
///
/// # Arguments
/// - `result`: Pointer to EmbeddingResult to free
#[no_mangle]
pub extern "C" fn free_image_embedding_result(result: *mut EmbeddingResult) {
    unsafe {
        if !result.is_null() {
            let _ = Box::from_raw(result);
        }
    }
}


