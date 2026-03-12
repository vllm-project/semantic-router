//! Batched embedding and multi-modal embedding FFI.
//!
//! Continuous batching, multi-modal (text/image/audio) encoding.

use crate::ffi::embedding::{get_multimodal_refs, GLOBAL_MODEL_FACTORY, STANDALONE_MULTIMODAL};
use crate::ffi::embedding_routing::create_error_result;
use crate::ffi::types::EmbeddingResult;
use crate::model_architectures::embedding::continuous_batch_scheduler::ContinuousBatchConfig;
use crate::model_architectures::embedding::qwen3_batched::Qwen3EmbeddingModelBatched;
use crate::model_architectures::embedding::qwen3_embedding::Qwen3EmbeddingModel;
use crate::model_architectures::embedding::MultiModalEmbeddingModel;
use crate::model_architectures::model_factory::ModelFactory;
use std::ffi::{c_char, CStr};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Batched model with tokenizer and device info
struct BatchedModelContext {
    model: Arc<Qwen3EmbeddingModelBatched>,
    tokenizer: Arc<Mutex<Tokenizer>>,
    device: candle_core::Device,
}

/// Global singleton for batched model
static GLOBAL_BATCHED_MODEL: OnceLock<BatchedModelContext> = OnceLock::new();

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn init_embedding_models_batched(
    qwen3_model_path: *const c_char,
    max_batch_size: i32,
    max_wait_ms: u64,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

    if GLOBAL_BATCHED_MODEL.get().is_some() {
        eprintln!("Warning: batched embedding model already initialized");
        return true;
    }

    let model_path = if qwen3_model_path.is_null() {
        eprintln!("Error: qwen3_model_path is null");
        return false;
    } else {
        unsafe {
            match CStr::from_ptr(qwen3_model_path).to_str() {
                Ok(s) if !s.is_empty() => s.to_string(),
                _ => {
                    eprintln!("Error: invalid qwen3_model_path");
                    return false;
                }
            }
        }
    };

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    let tokenizer_path = format!("{}/tokenizer.json", model_path);
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "ERROR: Failed to load tokenizer from {}: {:?}",
                tokenizer_path, e
            );
            return false;
        }
    };

    let base_model = match Qwen3EmbeddingModel::load(&model_path, &device) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("ERROR: Failed to load Qwen3 model: {:?}", e);
            return false;
        }
    };

    let batch_config = ContinuousBatchConfig {
        max_batch_size: max_batch_size as usize,
        max_wait_time_ms: max_wait_ms,
        min_batch_size: 1,
        enable_dynamic: true,
        max_seq_len_diff: 512,
        verbose: false,
    };

    println!(
        "Initializing continuous batching (max_batch={}, max_wait={}ms)",
        max_batch_size, max_wait_ms
    );
    let batched_model = Qwen3EmbeddingModelBatched::from_model(base_model, batch_config);

    let context = BatchedModelContext {
        model: Arc::new(batched_model),
        tokenizer: Arc::new(Mutex::new(tokenizer)),
        device: device.clone(),
    };

    if GLOBAL_BATCHED_MODEL.set(context).is_err() {
        eprintln!("ERROR: Failed to set global batched model");
        return false;
    }

    println!("INFO: Batched embedding model initialized successfully");
    true
}

/// Tokenize text and get (ids, mask, seq_len) for batched model
fn tokenize_for_batched(
    tokenizer: &Arc<Mutex<Tokenizer>>,
    text_str: &str,
    result: *mut EmbeddingResult,
) -> Result<(Vec<u32>, Vec<u32>, usize), i32> {
    let tokenizer_guard = match tokenizer.lock() {
        Ok(guard) => guard,
        Err(e) => {
            eprintln!("Error: failed to lock tokenizer: {}", e);
            unsafe { (*result) = create_error_result() }
            return Err(-1);
        }
    };

    let encodings = match tokenizer_guard.encode_batch(vec![text_str.to_string()], true) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Error: tokenization failed: {}", e);
            unsafe { (*result) = create_error_result() }
            return Err(-1);
        }
    };

    let ids: Vec<u32> = encodings[0].get_ids().to_vec();
    let mask: Vec<u32> = encodings[0].get_attention_mask().to_vec();
    let seq_len = encodings[0].len();

    Ok((ids, mask, seq_len))
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_batched(
    text: *const c_char,
    model_type: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || model_type.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_batched");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    let model_type_str = unsafe {
        match CStr::from_ptr(model_type).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    if model_type_str != "qwen3" {
        eprintln!(
            "Error: unsupported model type '{}' for batched embeddings (only 'qwen3' supported)",
            model_type_str
        );
        unsafe { (*result) = create_error_result() }
        return -1;
    }

    let batched_context = match GLOBAL_BATCHED_MODEL.get() {
        Some(ctx) => ctx,
        None => {
            eprintln!("Error: batched embedding model not initialized. Call init_embedding_models_batched first.");
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();
    let tokenizer = Arc::clone(&batched_context.tokenizer);
    let model = Arc::clone(&batched_context.model);

    let (ids_raw, mask_raw, seq_len) = match tokenize_for_batched(&tokenizer, text_str, result) {
        Ok(t) => t,
        Err(e) => return e,
    };

    let embedding_vec = match model.embedding_forward_from_raw(ids_raw, mask_raw) {
        Ok(vec) => vec,
        Err(e) => {
            eprintln!("Error: embedding generation failed: {:?}", e);
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let final_embedding = if target_dim > 0 && (target_dim as usize) < embedding_vec.len() {
        embedding_vec[..(target_dim as usize)].to_vec()
    } else {
        embedding_vec
    };

    let length = final_embedding.len() as i32;
    let data = Box::into_raw(final_embedding.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = EmbeddingResult {
            data,
            length,
            error: false,
            model_type: 0,
            sequence_length: seq_len as i32,
            processing_time_ms,
        };
    }

    0
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn init_multimodal_embedding_model(
    model_path: *const c_char,
    use_cpu: bool,
) -> bool {
    use candle_core::Device;

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

    if get_multimodal_refs().is_some() {
        eprintln!("WARNING: Multi-modal model already initialized");
        return true;
    }

    let device = if use_cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0).unwrap_or(Device::Cpu)
    };

    if GLOBAL_MODEL_FACTORY.get().is_none() {
        let mut factory = ModelFactory::new(device.clone());
        match factory.register_multimodal_embedding_model(&path) {
            Ok(_) => println!("INFO: Multi-modal embedding model registered in ModelFactory"),
            Err(e) => {
                eprintln!("ERROR: Failed to register multi-modal model: {:?}", e);
                return false;
            }
        }
        return match GLOBAL_MODEL_FACTORY.set(factory) {
            Ok(_) => true,
            Err(_) => {
                eprintln!("Error: Failed to set global model factory");
                false
            }
        };
    }

    println!(
        "INFO: ModelFactory already initialized, loading multimodal model into standalone storage"
    );
    let model = match MultiModalEmbeddingModel::load(&path, &device) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to load multi-modal model: {:?}", e);
            return false;
        }
    };
    let tokenizer_path = format!("{}/tokenizer.json", path);
    let tokenizer = match Tokenizer::from_file(&tokenizer_path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!(
                "ERROR: Failed to load multi-modal tokenizer from {}: {:?}",
                tokenizer_path, e
            );
            return false;
        }
    };
    match STANDALONE_MULTIMODAL.set((model, tokenizer, path)) {
        Ok(_) => {
            println!("INFO: Multi-modal model registered in standalone storage");
            true
        }
        Err(_) => {
            eprintln!("Error: Standalone multimodal storage already set");
            false
        }
    }
}

/// Tokenize and encode text with multimodal model, return embedding vec
fn encode_text_multimodal_internal(
    model: &MultiModalEmbeddingModel,
    tokenizer: &tokenizers::Tokenizer,
    text_str: &str,
    target_dimension: Option<usize>,
) -> Result<Vec<f32>, String> {
    let encoding = tokenizer
        .encode(text_str, true)
        .map_err(|e| format!("Tokenization failed: {:?}", e))?;

    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let mask: Vec<u32> = encoding
        .get_attention_mask().to_vec();
    let seq_len = ids.len();

    let device = model.device();
    let input_ids = candle_core::Tensor::from_vec(ids, (1, seq_len), device)
        .map_err(|e| format!("Tensor creation failed: {:?}", e))?;
    let attention_mask = candle_core::Tensor::from_vec(mask, (1, seq_len), device)
        .map_err(|e| format!("Tensor creation failed: {:?}", e))?;

    let embedding = model
        .encode_text_with_matryoshka(&input_ids, Some(&attention_mask), None, target_dimension)
        .map_err(|e| format!("Text encoding failed: {:?}", e))?;

    embedding
        .squeeze(0)
        .map_err(|e| format!("Squeeze failed: {:?}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("Conversion failed: {:?}", e))
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn multimodal_encode_text(
    text: *const c_char,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_text");
        return -1;
    }

    let text_str = unsafe {
        match CStr::from_ptr(text).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in text: {}", e);
                (*result) = MultiModalEmbeddingResult::default();
                return -1;
            }
        }
    };

    let (model, tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();
    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding_vec =
        match encode_text_multimodal_internal(model, tokenizer, text_str, target_dimension) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Error: {}", e);
                unsafe { (*result) = MultiModalEmbeddingResult::default() }
                return -1;
            }
        };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 0,
            processing_time_ms,
        };
    }

    0
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn multimodal_encode_image(
    pixel_data: *const f32,
    height: i32,
    width: i32,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if pixel_data.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_image");
        return -1;
    }

    let (model, _tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();
    let h = height as usize;
    let w = width as usize;
    let pixel_count = 3 * h * w;
    let pixels = unsafe { std::slice::from_raw_parts(pixel_data, pixel_count) };
    let device = model.device();

    let pixel_tensor = match candle_core::Tensor::from_slice(pixels, (1, 3, h, w), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: pixel tensor creation failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding = match model.encode_image_with_dim(&pixel_tensor, target_dimension) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: image encoding failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 1,
            processing_time_ms,
        };
    }

    0
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn multimodal_encode_audio(
    mel_data: *const f32,
    n_mels: i32,
    time_frames: i32,
    target_dim: i32,
    result: *mut crate::ffi::types::MultiModalEmbeddingResult,
) -> i32 {
    use crate::ffi::types::MultiModalEmbeddingResult;

    if mel_data.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to multimodal_encode_audio");
        return -1;
    }

    let (model, _tokenizer) = match get_multimodal_refs() {
        Some(refs) => refs,
        None => {
            eprintln!("Error: Multi-modal model not loaded");
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let start_time = std::time::Instant::now();
    let mels = n_mels as usize;
    let frames = time_frames as usize;
    let total = mels * frames;
    let mel_slice = unsafe { std::slice::from_raw_parts(mel_data, total) };
    let device = model.device();

    let mel_tensor = match candle_core::Tensor::from_slice(mel_slice, (1, mels, frames), device) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Error: mel tensor creation failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };

    let embedding = match model.encode_audio_with_dim(&mel_tensor, target_dimension) {
        Ok(emb) => emb,
        Err(e) => {
            eprintln!("Error: audio encoding failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let embedding_vec = match embedding.squeeze(0).and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error: embedding conversion failed: {:?}", e);
            unsafe { (*result) = MultiModalEmbeddingResult::default() }
            return -1;
        }
    };

    let length = embedding_vec.len() as i32;
    let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
    let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;

    unsafe {
        (*result) = MultiModalEmbeddingResult {
            data,
            length,
            error: false,
            modality: 2,
            processing_time_ms,
        };
    }

    0
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn free_multimodal_embedding(data: *mut f32, length: i32) {
    if data.is_null() || length <= 0 {
        return;
    }
    unsafe {
        let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(data, length as usize));
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn shutdown_embedding_batched() {
    println!("INFO: Shutting down batched embedding model");
}
