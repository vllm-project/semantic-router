//! Static embedding capability discovery for the Candle binding.
//!
//! The query in this module deliberately does not inspect loaded model state.
//! It reports facts owned by the compiled native binding and is safe to call
//! before any model factory has been initialized.

use std::{slice, str};

pub const EMBEDDING_CAPABILITIES_VERSION_V1: u32 = 1;

pub const CAPABILITY_STATUS_OK: i32 = 0;
pub const CAPABILITY_STATUS_UNSUPPORTED_MODEL: i32 = 1;
pub const CAPABILITY_STATUS_INVALID_INPUT: i32 = 2;

pub const BACKEND_CANDLE: u32 = 1;

pub const MODEL_TYPE_QWEN3: u32 = 1;
pub const MODEL_TYPE_GEMMA: u32 = 2;
pub const MODEL_TYPE_MMBERT: u32 = 3;
pub const MODEL_TYPE_MULTIMODAL: u32 = 4;

pub const MODALITY_TEXT: u32 = 1 << 0;
pub const MODALITY_IMAGE: u32 = 1 << 1;
pub const MODALITY_AUDIO: u32 = 1 << 2;

pub const DEVICE_CPU: u32 = 1 << 0;
pub const DEVICE_CUDA: u32 = 1 << 1;
pub const DEVICE_METAL: u32 = 1 << 3;

/// Version 1 of the stable C representation returned to Go.
///
/// `supported_dimensions` is borrowed native memory. The caller must copy it
/// before returning from the FFI call and must never free it. Version 1 returns
/// a null pointer and zero length because the current inference entrypoints do
/// not impose a finite dimension allowlist.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingCapabilitiesV1 {
    pub version: u32,
    pub struct_size: u32,
    pub backend: u32,
    pub model_type: u32,
    pub supports_batching: u8,
    pub reserved: [u8; 3],
    pub modalities: u32,
    pub devices: u32,
    pub supported_dimensions: *const u32,
    pub num_supported_dimensions: usize,
}

impl Default for EmbeddingCapabilitiesV1 {
    fn default() -> Self {
        Self {
            version: EMBEDDING_CAPABILITIES_VERSION_V1,
            struct_size: std::mem::size_of::<Self>() as u32,
            backend: BACKEND_CANDLE,
            model_type: 0,
            supports_batching: 0,
            reserved: [0; 3],
            modalities: 0,
            devices: supported_devices(),
            supported_dimensions: std::ptr::null(),
            num_supported_dimensions: 0,
        }
    }
}

const fn supported_devices() -> u32 {
    DEVICE_CPU
        | if cfg!(feature = "cuda") {
            DEVICE_CUDA
        } else {
            0
        }
        | if cfg!(feature = "metal") {
            DEVICE_METAL
        } else {
            0
        }
}

fn descriptor_for(model_type: &str) -> Option<(u32, bool, u32)> {
    match model_type {
        "qwen3" => Some((MODEL_TYPE_QWEN3, true, MODALITY_TEXT)),
        "gemma" => Some((MODEL_TYPE_GEMMA, false, MODALITY_TEXT)),
        "mmbert" => Some((MODEL_TYPE_MMBERT, false, MODALITY_TEXT)),
        "multimodal" => Some((
            MODEL_TYPE_MULTIMODAL,
            false,
            MODALITY_TEXT | MODALITY_IMAGE | MODALITY_AUDIO,
        )),
        _ => None,
    }
}

/// Return versioned static capabilities for a request-facing embedding model type.
///
/// The input is pointer-and-length rather than a C string so embedded NUL bytes
/// cannot be truncated into a different, valid model type.
///
/// # Safety
///
/// - `model_type` must reference `model_type_len` readable bytes when the length
///   is non-zero.
/// - `result` must reference writable memory for `EmbeddingCapabilitiesV1`.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn candle_embedding_capabilities_v1(
    model_type: *const u8,
    model_type_len: usize,
    result: *mut EmbeddingCapabilitiesV1,
) -> i32 {
    if result.is_null() {
        return CAPABILITY_STATUS_INVALID_INPUT;
    }

    unsafe {
        *result = EmbeddingCapabilitiesV1::default();
    }

    if model_type.is_null() && model_type_len > 0 {
        return CAPABILITY_STATUS_INVALID_INPUT;
    }

    let bytes = if model_type_len == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(model_type, model_type_len) }
    };
    let raw = match str::from_utf8(bytes) {
        Ok(value) => value,
        Err(_) => return CAPABILITY_STATUS_INVALID_INPUT,
    };
    let canonical = raw.trim().to_ascii_lowercase();
    let Some((model_type, supports_batching, modalities)) = descriptor_for(&canonical) else {
        return CAPABILITY_STATUS_UNSUPPORTED_MODEL;
    };

    unsafe {
        (*result).model_type = model_type;
        (*result).supports_batching = u8::from(supports_batching);
        (*result).modalities = modalities;
    }
    CAPABILITY_STATUS_OK
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(value: &[u8]) -> (i32, EmbeddingCapabilitiesV1) {
        let mut result = EmbeddingCapabilitiesV1::default();
        let status = candle_embedding_capabilities_v1(value.as_ptr(), value.len(), &mut result);
        (status, result)
    }

    #[test]
    fn normalizes_known_model_types() {
        let (status, result) = query(b"  Qwen3  ");
        assert_eq!(status, CAPABILITY_STATUS_OK);
        assert_eq!(result.version, EMBEDDING_CAPABILITIES_VERSION_V1);
        assert_eq!(result.backend, BACKEND_CANDLE);
        assert_eq!(result.model_type, MODEL_TYPE_QWEN3);
        assert_eq!(result.supports_batching, 1);
        assert_eq!(result.modalities, MODALITY_TEXT);
        assert!(result.supported_dimensions.is_null());
        assert_eq!(result.num_supported_dimensions, 0);
        assert_ne!(result.devices & DEVICE_CPU, 0);
    }

    #[test]
    fn reports_multimodal_modalities() {
        let (status, result) = query(b"multimodal");
        assert_eq!(status, CAPABILITY_STATUS_OK);
        assert_eq!(result.model_type, MODEL_TYPE_MULTIMODAL);
        assert_eq!(
            result.modalities,
            MODALITY_TEXT | MODALITY_IMAGE | MODALITY_AUDIO
        );
        assert_eq!(result.supports_batching, 0);
    }

    #[test]
    fn distinguishes_unsupported_and_invalid_input() {
        let (status, _) = query(b"unknown");
        assert_eq!(status, CAPABILITY_STATUS_UNSUPPORTED_MODEL);

        let (status, _) = query(b"qwen3\0ignored");
        assert_eq!(status, CAPABILITY_STATUS_UNSUPPORTED_MODEL);

        let (status, _) = query(&[0xff]);
        assert_eq!(status, CAPABILITY_STATUS_INVALID_INPUT);

        let status = candle_embedding_capabilities_v1(b"qwen3".as_ptr(), 5, std::ptr::null_mut());
        assert_eq!(status, CAPABILITY_STATUS_INVALID_INPUT);
    }
}
