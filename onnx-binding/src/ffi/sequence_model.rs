//! Independently owned sequence heads for configurable Safety/Hazard models.
//! The caller serializes predict/close; no global classifier slot is replaced.

use crate::model_architectures::classification::{
    ClassifierExecutionProvider, MmBertSequenceClassifier,
};
use serde::Deserialize;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Options {
    model_path: String,
    use_cpu: bool,
    max_sequence_length: usize,
    labels: Vec<String>,
    multi_label: bool,
}

struct SequenceModel {
    model: MmBertSequenceClassifier,
    tokenizer: tokenizers::Tokenizer,
    limit: usize,
    multi_label: bool,
}

fn load(options: Options) -> Result<SequenceModel, String> {
    let config_path = std::path::Path::new(&options.model_path).join("config.json");
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(config_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    validate_contract(&config, &options)?;
    let mut tokenizer = tokenizers::Tokenizer::from_file(
        std::path::Path::new(&options.model_path).join("tokenizer.json"),
    )
    .map_err(|e| e.to_string())?;
    tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
    tokenizer.with_padding(None);
    let provider = if options.use_cpu {
        ClassifierExecutionProvider::Cpu
    } else {
        ClassifierExecutionProvider::Auto
    };
    let model = MmBertSequenceClassifier::load_with_max_sequence_length(
        &options.model_path,
        provider,
        options.max_sequence_length,
    )
    .map_err(|e| e.to_string())?;
    Ok(SequenceModel {
        model,
        tokenizer,
        limit: options.max_sequence_length,
        multi_label: options.multi_label,
    })
}

fn validate_contract(config: &serde_json::Value, options: &Options) -> Result<(), String> {
    if options.max_sequence_length == 0 {
        return Err("max_sequence_length must be positive".into());
    }
    let labels = config["id2label"]
        .as_object()
        .ok_or("model config requires id2label")?;
    if labels.len() != options.labels.len() || labels.is_empty() {
        return Err("model label count differs from configured labels".into());
    }
    for (index, expected) in options.labels.iter().enumerate() {
        if labels.get(&index.to_string()).and_then(|v| v.as_str()) != Some(expected.as_str()) {
            return Err(format!(
                "model label at index {index} differs from configured labels"
            ));
        }
    }
    let task = config["problem_type"]
        .as_str()
        .unwrap_or("single_label_classification");
    let expected = if options.multi_label {
        "multi_label_classification"
    } else {
        "single_label_classification"
    };
    if task != expected {
        return Err(format!(
            "model problem_type {task:?} differs from required {expected:?}"
        ));
    }
    Ok(())
}

fn string_result(value: serde_json::Value) -> *mut c_char {
    CString::new(value.to_string())
        .expect("JSON escapes interior NUL")
        .into_raw()
}

/// # Safety
/// options must be a valid NUL-terminated JSON string; error must be writable.
/// The returned handle is uniquely owned and must be closed exactly once.
#[no_mangle]
pub unsafe extern "C" fn onnx_sequence_model_open(
    options: *const c_char,
    error: *mut *mut c_char,
) -> *mut c_void {
    if error.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        *error = std::ptr::null_mut();
    }
    let loaded = catch_unwind(AssertUnwindSafe(|| {
        if options.is_null() {
            return Err("model options are required".to_string());
        }
        let text = unsafe { CStr::from_ptr(options) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let options = serde_json::from_str(text).map_err(|e| e.to_string())?;
        load(options)
    }))
    .unwrap_or_else(|_| Err("sequence model loading panicked".into()));
    match loaded {
        Ok(model) => Box::into_raw(Box::new(model)).cast(),
        Err(message) => {
            unsafe {
                *error = string_result(serde_json::json!({"error": message}));
            }
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// handle must be a live handle from open; text must be a NUL-terminated string.
/// Caller must serialize this operation with predict and close on the handle.
#[no_mangle]
pub unsafe extern "C" fn onnx_sequence_model_predict(
    handle: *mut c_void,
    text: *const c_char,
) -> *mut c_char {
    let outcome = catch_unwind(AssertUnwindSafe(|| -> Result<Vec<f32>, String> {
        if handle.is_null() || text.is_null() {
            return Err("live model and text are required".into());
        }
        let model = unsafe { &mut *handle.cast::<SequenceModel>() };
        let text = unsafe { CStr::from_ptr(text) }
            .to_str()
            .map_err(|e| e.to_string())?;
        let encoding = model
            .tokenizer
            .encode(text, true)
            .map_err(|e| e.to_string())?;
        if encoding.len() > model.limit {
            return Err(format!(
                "input has {} tokens, exceeds {} token model budget",
                encoding.len(),
                model.limit
            ));
        }
        let mut results = model
            .model
            .classify_batch_with_activation(&[text], model.multi_label)
            .map_err(|e| e.to_string())?;
        results
            .pop()
            .map(|result| result.probabilities)
            .ok_or_else(|| "model returned no result".into())
    }))
    .unwrap_or_else(|_| Err("sequence inference panicked".into()));
    match outcome {
        Ok(scores) => string_result(serde_json::json!({"scores": scores})),
        Err(message) => string_result(serde_json::json!({"error": message})),
    }
}

/// # Safety
/// handle must be null or uniquely owned and live, with no in-flight prediction.
#[no_mangle]
pub unsafe extern "C" fn onnx_sequence_model_close(handle: *mut c_void) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle.cast::<SequenceModel>()) });
    }
}

/// # Safety
/// value must be null or an unfreed string returned by this module.
#[no_mangle]
pub unsafe extern "C" fn onnx_sequence_model_free(value: *mut c_char) {
    if !value.is_null() {
        drop(unsafe { CString::from_raw(value) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sequence_contract_rejects_wrong_label_order_and_activation() {
        let cfg = serde_json::json!({"id2label":{"0":"safe","1":"unsafe"},"problem_type":"single_label_classification"});
        let mut options = Options {
            model_path: String::new(),
            use_cpu: true,
            max_sequence_length: 32768,
            labels: vec!["safe".into(), "unsafe".into()],
            multi_label: false,
        };
        assert!(validate_contract(&cfg, &options).is_ok());
        options.labels.reverse();
        assert!(validate_contract(&cfg, &options).is_err());
        options.labels.reverse();
        options.multi_label = true;
        assert!(validate_contract(&cfg, &options).is_err());
    }
    #[test]
    fn sequence_ffi_returns_owned_errors_for_invalid_arguments() {
        let mut error = std::ptr::null_mut();
        unsafe {
            assert!(onnx_sequence_model_open(std::ptr::null(), &mut error).is_null());
            assert!(!error.is_null());
            onnx_sequence_model_free(error);
            let result = onnx_sequence_model_predict(std::ptr::null_mut(), std::ptr::null());
            assert!(CStr::from_ptr(result).to_str().unwrap().contains("error"));
            onnx_sequence_model_free(result);
            onnx_sequence_model_close(std::ptr::null_mut());
        }
    }
}
