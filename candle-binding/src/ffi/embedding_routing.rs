//! Embedding routing and model-type embedding FFI.
//!
//! Smart routing, per-model embedding generation, and 2D Matryoshka support.

use crate::ffi::embedding::PaddingSide;
use crate::ffi::embedding::{generate_embedding_internal, generate_embeddings_batch_internal};
use crate::ffi::embedding::{get_multimodal_refs, GLOBAL_MODEL_FACTORY};
use crate::ffi::types::EmbeddingResult;
use crate::model_architectures::model_factory::ModelFactory;
use crate::model_architectures::ModelType;
use std::ffi::{c_char, CStr, CString};

use crate::classifiers::unified::{DualPathUnifiedClassifier, EmbeddingRequirements};
use crate::model_architectures::config::DualPathConfig;

/// Helper function to create a temporary classifier for routing decisions
fn create_temp_classifier() -> Result<DualPathUnifiedClassifier, String> {
    DualPathUnifiedClassifier::new(DualPathConfig::default())
        .map_err(|e| format!("Failed to create classifier: {:?}", e))
}

/// Helper function to create an error result
pub(crate) fn create_error_result() -> EmbeddingResult {
    EmbeddingResult {
        data: std::ptr::null_mut(),
        length: 0,
        error: true,
        model_type: -1,
        sequence_length: 0,
        processing_time_ms: 0.0,
    }
}

/// Resolve model type to string with fallback based on factory availability
fn resolve_model_type_str(
    model_type: ModelType,
    factory: Option<&'static ModelFactory>,
) -> Result<&'static str, ()> {
    match model_type {
        ModelType::Qwen3Embedding => {
            if factory.is_some_and(|f| f.get_qwen3_model().is_some()) {
                Ok("qwen3")
            } else if factory.is_some_and(|f| f.get_mmbert_model().is_some()) {
                eprintln!("INFO: Qwen3 not available, falling back to mmbert");
                Ok("mmbert")
            } else if factory.is_some_and(|f| f.get_gemma_model().is_some()) {
                eprintln!("INFO: Qwen3 not available, falling back to gemma");
                Ok("gemma")
            } else {
                eprintln!(
                    "Error: Qwen3Embedding selected but not available and no fallback available"
                );
                Err(())
            }
        }
        ModelType::GemmaEmbedding => {
            if factory.is_some_and(|f| f.get_gemma_model().is_some()) {
                Ok("gemma")
            } else if factory.is_some_and(|f| f.get_mmbert_model().is_some()) {
                eprintln!("INFO: Gemma not available, falling back to mmbert");
                Ok("mmbert")
            } else if factory.is_some_and(|f| f.get_qwen3_model().is_some()) {
                eprintln!("INFO: Gemma not available, falling back to qwen3");
                Ok("qwen3")
            } else {
                eprintln!(
                    "Error: GemmaEmbedding selected but not available and no fallback available"
                );
                Err(())
            }
        }
        ModelType::MmBertEmbedding => {
            if factory.is_some_and(|f| f.get_mmbert_model().is_some()) {
                Ok("mmbert")
            } else {
                eprintln!("Error: MmBertEmbedding selected but not available");
                Err(())
            }
        }
        ModelType::MultiModalEmbedding => {
            if get_multimodal_refs().is_some() {
                Ok("multimodal")
            } else {
                eprintln!("Error: MultiModalEmbedding selected but not available");
                Err(())
            }
        }
        _ => {
            eprintln!("Error: unsupported model type: {:?}", model_type);
            Err(())
        }
    }
}

/// Internal helper to generate embedding for Qwen3
pub(crate) fn generate_qwen3_embedding(
    factory: &ModelFactory,
    text: &str,
    target_dim: Option<usize>,
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    generate_embedding_internal(
        text,
        target_dim,
        || factory.get_qwen3_tokenizer(),
        |token_ids, attention_mask| {
            let model = factory
                .get_qwen3_model()
                .ok_or_else(|| "Qwen3 model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::new(token_ids.as_slice(), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze input_ids: {:?}", e))?;
            let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze attention_mask: {:?}", e))?;
            model
                .embedding_forward(&input_ids, &attention_mask_tensor)
                .map_err(|e| format!("Forward pass failed: {:?}", e))
        },
    )
}

/// Generate embeddings for multiple texts in a single batch (Qwen3)
pub(crate) fn generate_qwen3_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_dim: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    use candle_core::Tensor;

    const QWEN3_PAD_TOKEN_ID: u32 = 151643;
    generate_embeddings_batch_internal(
        texts,
        target_dim,
        QWEN3_PAD_TOKEN_ID,
        PaddingSide::Left,
        || factory.get_qwen3_tokenizer(),
        |flat_ids, flat_mask, batch_size, max_len| {
            let model = factory
                .get_qwen3_model()
                .ok_or_else(|| "Qwen3 model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;
            model
                .embedding_forward(&input_ids, &attention_mask)
                .map_err(|e| format!("Model forward failed: {:?}", e))
        },
    )
}

/// Internal helper to generate embedding for Gemma
pub(crate) fn generate_gemma_embedding(
    factory: &ModelFactory,
    text: &str,
    target_dim: Option<usize>,
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    generate_embedding_internal(
        text,
        target_dim,
        || factory.get_gemma_tokenizer(),
        |token_ids, attention_mask| {
            let model = factory
                .get_gemma_model()
                .ok_or_else(|| "Gemma model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::new(token_ids.as_slice(), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze input_ids: {:?}", e))?;
            let attention_mask_tensor = Tensor::new(attention_mask.as_slice(), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?
                .unsqueeze(0)
                .map_err(|e| format!("Failed to unsqueeze attention_mask: {:?}", e))?;
            model
                .embedding_forward(&input_ids, Some(&attention_mask_tensor))
                .map_err(|e| format!("Forward pass failed: {:?}", e))
        },
    )
}

/// Generate embeddings for multiple texts in a single batch (Gemma)
pub(crate) fn generate_gemma_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_dim: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    use candle_core::Tensor;

    const GEMMA_PAD_TOKEN_ID: u32 = 0;
    generate_embeddings_batch_internal(
        texts,
        target_dim,
        GEMMA_PAD_TOKEN_ID,
        PaddingSide::Right,
        || factory.get_gemma_tokenizer(),
        |flat_ids, flat_mask, batch_size, max_len| {
            let model = factory
                .get_gemma_model()
                .ok_or_else(|| "Gemma model not available".to_string())?;
            let device = model.device();
            let input_ids = Tensor::from_vec(flat_ids, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
            let attention_mask = Tensor::from_vec(flat_mask, (batch_size, max_len), &device)
                .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;
            model
                .embedding_forward(&input_ids, Some(&attention_mask))
                .map_err(|e| format!("Model forward failed: {:?}", e))
        },
    )
}

/// Internal helper to generate embedding for mmBERT with 2D Matryoshka
pub(crate) fn generate_mmbert_embedding(
    factory: &ModelFactory,
    text: &str,
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    let model = factory
        .get_mmbert_model()
        .ok_or_else(|| "mmBERT model not available".to_string())?;
    let tokenizer = factory
        .get_mmbert_tokenizer()
        .ok_or_else(|| "mmBERT tokenizer not available".to_string())?;

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization failed: {:?}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding
        .get_attention_mask().to_vec();
    let seq_len = token_ids.len();

    let device = model.device();
    let input_ids = Tensor::from_vec(token_ids, (1, seq_len), device)
        .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
    let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, seq_len), device)
        .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

    let embedding = model
        .embedding_forward_with_matryoshka(
            &input_ids,
            Some(&attention_mask_tensor),
            target_layer,
            target_dim,
        )
        .map_err(|e| format!("mmBERT forward failed: {:?}", e))?;

    embedding
        .squeeze(0)
        .map_err(|e| format!("Failed to squeeze: {:?}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("Failed to convert to vec: {:?}", e))
}

/// Generate embeddings for multiple texts in a single batch (mmBERT)
pub(crate) fn generate_mmbert_embeddings_batch(
    factory: &ModelFactory,
    texts: &[&str],
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    let model = factory
        .get_mmbert_model()
        .ok_or_else(|| "mmBERT model not available".to_string())?;
    let tokenizer = factory
        .get_mmbert_tokenizer()
        .ok_or_else(|| "mmBERT tokenizer not available".to_string())?;

    let embeddings = model
        .encode_batch_with_matryoshka(tokenizer, texts, 8192, target_layer, target_dim)
        .map_err(|e| format!("mmBERT batch encoding failed: {:?}", e))?;

    embeddings
        .to_vec2::<f32>()
        .map_err(|e| format!("Failed to convert embeddings: {:?}", e))
}

/// Internal helper to generate text embedding via the multi-modal model
pub(crate) fn generate_multimodal_text_embedding(
    _factory: &ModelFactory,
    text: &str,
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> Result<Vec<f32>, String> {
    use candle_core::Tensor;

    let (model, tokenizer) =
        get_multimodal_refs().ok_or_else(|| "Multi-modal model not available".to_string())?;

    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("Tokenization failed: {:?}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let attention_mask: Vec<u32> = encoding
        .get_attention_mask().to_vec();
    let seq_len = token_ids.len();

    let device = model.device();
    let input_ids = Tensor::from_vec(token_ids, (1, seq_len), device)
        .map_err(|e| format!("Failed to create input_ids tensor: {:?}", e))?;
    let attention_mask_tensor = Tensor::from_vec(attention_mask, (1, seq_len), device)
        .map_err(|e| format!("Failed to create attention_mask tensor: {:?}", e))?;

    let embedding = model
        .encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask_tensor),
            target_layer,
            target_dim,
        )
        .map_err(|e| format!("Multi-modal text encoding failed: {:?}", e))?;

    embedding
        .squeeze(0)
        .map_err(|e| format!("Failed to squeeze: {:?}", e))?
        .to_vec1::<f32>()
        .map_err(|e| format!("Failed to convert to vec: {:?}", e))
}

/// Generate text embeddings for multiple texts in a single batch (multi-modal)
pub(crate) fn generate_multimodal_text_embeddings_batch(
    _factory: &ModelFactory,
    texts: &[&str],
    target_layer: Option<usize>,
    target_dim: Option<usize>,
) -> Result<Vec<Vec<f32>>, String> {
    let (model, tokenizer) =
        get_multimodal_refs().ok_or_else(|| "Multi-modal model not available".to_string())?;

    let embeddings = model
        .encode_text_batch_with_matryoshka(tokenizer, texts, 512, target_layer, target_dim)
        .map_err(|e| format!("Multi-modal batch encoding failed: {:?}", e))?;

    embeddings
        .to_vec2::<f32>()
        .map_err(|e| format!("Failed to convert embeddings: {:?}", e))
}

/// Parse model type string to ModelType enum
fn parse_model_type(model_type_str: &str) -> Result<ModelType, ()> {
    match model_type_str {
        "qwen3" => Ok(ModelType::Qwen3Embedding),
        "gemma" => Ok(ModelType::GemmaEmbedding),
        "mmbert" => Ok(ModelType::MmBertEmbedding),
        "multimodal" => Ok(ModelType::MultiModalEmbedding),
        _ => {
            eprintln!(
                "Error: invalid model type '{}' (must be 'qwen3', 'gemma', 'mmbert', or 'multimodal')",
                model_type_str
            );
            Err(())
        }
    }
}

/// Generate embedding by model type and write to result
fn generate_and_write_embedding(
    factory: &ModelFactory,
    text_str: &str,
    model_type: ModelType,
    target_layer: Option<usize>,
    target_dim: Option<usize>,
    result: *mut EmbeddingResult,
) -> i32 {
    let start_time = std::time::Instant::now();

    let embedding_result = match model_type {
        ModelType::Qwen3Embedding => generate_qwen3_embedding(factory, text_str, target_dim),
        ModelType::GemmaEmbedding => generate_gemma_embedding(factory, text_str, target_dim),
        ModelType::MmBertEmbedding => {
            generate_mmbert_embedding(factory, text_str, target_layer, target_dim)
        }
        ModelType::MultiModalEmbedding => {
            generate_multimodal_text_embedding(factory, text_str, target_layer, target_dim)
        }
        _ => {
            eprintln!("Error: unsupported model type: {:?}", model_type);
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    match embedding_result {
        Ok(embedding_vec) => {
            let length = embedding_vec.len() as i32;
            let data = Box::into_raw(embedding_vec.into_boxed_slice()) as *mut f32;
            let processing_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
            let model_type_id = match model_type {
                ModelType::Qwen3Embedding => 0,
                ModelType::GemmaEmbedding => 1,
                ModelType::MmBertEmbedding => 2,
                ModelType::MultiModalEmbedding => 3,
                _ => -1,
            };
            unsafe {
                (*result) = EmbeddingResult {
                    data,
                    length,
                    error: false,
                    model_type: model_type_id,
                    sequence_length: text_str.split_whitespace().count() as i32,
                    processing_time_ms,
                };
            }
            0
        }
        Err(e) => {
            eprintln!("Error: embedding generation failed: {}", e);
            unsafe { (*result) = create_error_result() }
            -1
        }
    }
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_smart(
    text: *const c_char,
    quality_priority: f32,
    latency_priority: f32,
    result: *mut EmbeddingResult,
) -> i32 {
    get_embedding_with_dim(text, quality_priority, latency_priority, 0, result)
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_with_dim(
    text: *const c_char,
    quality_priority: f32,
    latency_priority: f32,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_with_dim");
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

    let requirements = EmbeddingRequirements {
        sequence_length: text_str.split_whitespace().count(),
        quality_priority,
        latency_priority,
        target_dimension: if target_dim > 0 {
            Some(target_dim as usize)
        } else {
            None
        },
    };

    let classifier = match create_temp_classifier() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Error: failed to create classifier: {}", e);
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let model_type = match classifier.select_embedding_model(&requirements) {
        Ok(mt) => mt,
        Err(e) => {
            eprintln!("Error: model selection failed: {:?}", e);
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let factory = GLOBAL_MODEL_FACTORY.get();
    let model_type_str = match resolve_model_type_str(model_type, factory) {
        Ok(s) => s,
        Err(()) => {
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let model_type_cstr = CString::new(model_type_str).unwrap();
    get_embedding_2d_matryoshka(text, model_type_cstr.as_ptr(), 0, target_dim, result)
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_with_model_type(
    text: *const c_char,
    model_type_str: *const c_char,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    get_embedding_2d_matryoshka(text, model_type_str, 0, target_dim, result)
}

/// # Safety
/// Caller must ensure all pointer arguments are valid, non-null, and point to valid C strings where applicable.
#[no_mangle]
pub unsafe extern "C" fn get_embedding_2d_matryoshka(
    text: *const c_char,
    model_type_str: *const c_char,
    target_layer: i32,
    target_dim: i32,
    result: *mut EmbeddingResult,
) -> i32 {
    if text.is_null() || model_type_str.is_null() || result.is_null() {
        eprintln!("Error: null pointer passed to get_embedding_2d_matryoshka");
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
        match CStr::from_ptr(model_type_str).to_str() {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error: invalid UTF-8 in model_type: {}", e);
                (*result) = create_error_result();
                return -1;
            }
        }
    };

    let model_type = match parse_model_type(model_type_str) {
        Ok(mt) => mt,
        Err(()) => {
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    let target_dimension = if target_dim > 0 {
        Some(target_dim as usize)
    } else {
        None
    };
    let layer = if target_layer > 0 {
        Some(target_layer as usize)
    } else {
        None
    };

    let factory = match GLOBAL_MODEL_FACTORY.get() {
        Some(f) => f,
        None => {
            eprintln!("Error: ModelFactory not initialized");
            unsafe { (*result) = create_error_result() }
            return -1;
        }
    };

    generate_and_write_embedding(
        factory,
        text_str,
        model_type,
        layer,
        target_dimension,
        result,
    )
}
