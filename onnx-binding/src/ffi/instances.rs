//! Private wire ABI for the typed Go instance package. Each task has a distinct
//! entry point; JSON is only its transport representation, not a generic task API.

use crate::{core::instance_options::InstanceOptions, instances, UnifiedError, UnifiedResult};
use serde::Serialize;
use std::{
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
};

#[repr(C)]
pub struct InstanceResult {
    pub handle: u64,
    pub payload: *mut c_char,
    pub error: *mut c_char,
    pub error_kind: *mut c_char,
}

fn string(value: String) -> *mut c_char {
    CString::new(value.replace('\0', "\u{fffd}"))
        .expect("NUL removed")
        .into_raw()
}

fn result<T: Serialize>(call: impl FnOnce() -> UnifiedResult<T>) -> InstanceResult {
    let output = catch_unwind(AssertUnwindSafe(call)).unwrap_or_else(|_| {
        Err(UnifiedError::Inference {
            operation: "native_call".into(),
            source: "native provider panicked".into(),
        })
    });
    match output {
        Ok(value) => match serde_json::to_string(&value) {
            Ok(payload) => InstanceResult {
                handle: 0,
                payload: string(payload),
                error: ptr::null_mut(),
                error_kind: ptr::null_mut(),
            },
            Err(error) => failure("invalid_output", &error.to_string()),
        },
        Err(error) => {
            let kind = match &error {
                UnifiedError::Config { field, .. } if field == "handle" => "closed",
                UnifiedError::Config { field, .. } if field == "provider" => "capability",
                UnifiedError::Config { .. } | UnifiedError::InvalidJson { .. } => "configuration",
                UnifiedError::Validation { field, .. }
                    if field == "input_tokens" || field == "audio_frames" =>
                {
                    "input_limit"
                }
                UnifiedError::Validation { .. } => "invalid_input",
                UnifiedError::FileNotFound { .. } | UnifiedError::ModelLoad { .. } => "load",
                UnifiedError::Inference { operation, .. }
                    if ["distribution", "token_spans", "embedding"]
                        .contains(&operation.as_str()) =>
                {
                    "invalid_output"
                }
                _ => "inference",
            };
            failure(kind, &error.to_string())
        }
    }
}

fn failure(kind: &str, message: &str) -> InstanceResult {
    InstanceResult {
        handle: 0,
        payload: ptr::null_mut(),
        error: string(message.into()),
        error_kind: string(kind.into()),
    }
}

unsafe fn text<'a>(value: *const c_char) -> UnifiedResult<&'a str> {
    if value.is_null() {
        return Err(crate::core::unified_error::errors::config_error(
            "input",
            "null string",
        ));
    }
    CStr::from_ptr(value)
        .to_str()
        .map_err(|e| crate::core::unified_error::errors::config_error("input", &e.to_string()))
}

unsafe fn load(
    options: *const c_char,
    load: fn(InstanceOptions) -> UnifiedResult<u64>,
) -> InstanceResult {
    let mut handle = 0;
    let mut output = result(|| {
        let options: InstanceOptions = serde_json::from_str(text(options)?).map_err(|e| {
            crate::core::unified_error::errors::config_error("options", &e.to_string())
        })?;
        handle = load(options)?;
        Ok(())
    });
    output.handle = handle;
    output
}

macro_rules! loader {
    ($name:ident, $load:path) => {
        /// # Safety
        /// `options` must point to a live NUL-terminated UTF-8 options object.
        #[no_mangle]
        pub unsafe extern "C" fn $name(options: *const c_char) -> InstanceResult {
            load(options, $load)
        }
    };
}
loader!(ort_instance_load_sequence, instances::load_sequence);
loader!(ort_instance_load_token, instances::load_token);
loader!(ort_instance_load_embedding, instances::load_embedding);
loader!(ort_instance_load_multimodal, instances::load_multimodal);

#[no_mangle]
pub extern "C" fn ort_instance_clone(handle: u64) -> InstanceResult {
    let mut cloned = 0;
    let mut output = result(|| {
        cloned = instances::clone_handle(handle)?;
        Ok(())
    });
    output.handle = cloned;
    output
}

#[no_mangle]
pub extern "C" fn ort_instance_close(handle: u64) {
    instances::close(handle);
}

#[no_mangle]
pub extern "C" fn ort_instance_info(handle: u64) -> InstanceResult {
    result(|| instances::info(handle))
}

#[no_mangle]
pub extern "C" fn ort_instance_finish_profiling(handle: u64) -> InstanceResult {
    result(|| instances::finish_profiling(handle))
}

/// # Safety
/// `input` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_classify(
    handle: u64,
    input: *const c_char,
) -> InstanceResult {
    result(|| instances::classify(handle, text(input)?))
}

/// # Safety
/// `input` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_detect_tokens(
    handle: u64,
    input: *const c_char,
) -> InstanceResult {
    result(|| instances::detect_tokens(handle, text(input)?))
}

/// # Safety
/// `input` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_text_windows(
    handle: u64,
    input: *const c_char,
    max_tokens: usize,
) -> InstanceResult {
    result(|| instances::text_windows(handle, text(input)?, max_tokens))
}

/// # Safety
/// `input` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_encode_text(
    handle: u64,
    input: *const c_char,
    layer: usize,
    dimension: usize,
) -> InstanceResult {
    result(|| {
        instances::encode_text(
            handle,
            text(input)?,
            (layer != 0).then_some(layer),
            (dimension != 0).then_some(dimension),
        )
    })
}

unsafe fn slice<'a, T>(data: *const T, length: usize) -> UnifiedResult<&'a [T]> {
    if data.is_null() || length == 0 || length > (isize::MAX as usize) / std::mem::size_of::<T>() {
        return Err(crate::core::unified_error::errors::config_error(
            "input",
            "null, empty or oversized input buffer",
        ));
    }
    Ok(std::slice::from_raw_parts(data, length))
}

/// # Safety
/// `pixels` must point to `length` readable f32 values for the call duration.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_encode_image(
    handle: u64,
    pixels: *const f32,
    length: usize,
    height: usize,
    width: usize,
    dimension: usize,
) -> InstanceResult {
    result(|| {
        if height.checked_mul(width).and_then(|n| n.checked_mul(3)) != Some(length) {
            return Err(crate::core::unified_error::errors::config_error(
                "pixels",
                "invalid CHW shape",
            ));
        }
        instances::encode_image(
            handle,
            slice(pixels, length)?,
            height,
            width,
            (dimension != 0).then_some(dimension),
        )
    })
}

/// # Safety
/// `bytes` must point to `length` readable bytes for the call duration.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_encode_image_bytes(
    handle: u64,
    bytes: *const u8,
    length: usize,
    dimension: usize,
) -> InstanceResult {
    result(|| {
        instances::encode_image_bytes(
            handle,
            slice(bytes, length)?,
            (dimension != 0).then_some(dimension),
        )
    })
}

/// # Safety
/// `mel` must point to `length` readable f32 values for the call duration.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_encode_audio(
    handle: u64,
    mel: *const f32,
    length: usize,
    n_mels: usize,
    frames: usize,
    dimension: usize,
) -> InstanceResult {
    result(|| {
        instances::encode_audio(
            handle,
            slice(mel, length)?,
            n_mels,
            frames,
            (dimension != 0).then_some(dimension),
        )
    })
}

/// # Safety
/// The result must have been returned by this ABI and not previously freed.
#[no_mangle]
pub unsafe extern "C" fn ort_instance_result_free(output: InstanceResult) {
    for value in [output.payload, output.error, output.error_kind] {
        if !value.is_null() {
            drop(CString::from_raw(value));
        }
    }
}
