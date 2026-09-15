//! Namespaced C ABI. JSON is only the FFI serialization format; task inputs and
//! outputs stay separate and typed on both sides of this private boundary.
use super::*;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

unsafe fn string<'a>(ptr: *const c_char) -> Result<&'a str> {
    ensure!(!ptr.is_null(), "configuration: null string argument");
    Ok(CStr::from_ptr(ptr).to_str()?)
}

fn error_with_context(error: anyhow::Error, fallback: &str) -> anyhow::Error {
    let message = error.to_string();
    if [
        "configuration:",
        "capability:",
        "input_limit:",
        "result_invalid:",
        "execution:",
        "closed:",
        "load:",
    ]
    .iter()
    .any(|prefix| message.starts_with(prefix))
    {
        error
    } else {
        anyhow!("{fallback}: {message}")
    }
}

fn reply<T: Serialize>(operation: impl FnOnce() -> Result<T>) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(operation));
    let body = match result {
        Ok(Ok(value)) => serde_json::json!({"value": value}),
        Ok(Err(error)) => {
            serde_json::json!({"error": error_with_context(error, "execution").to_string()})
        }
        Err(_) => serde_json::json!({"error": "execution: native inference panicked"}),
    };
    CString::new(body.to_string())
        .expect("JSON contains no NUL")
        .into_raw()
}

macro_rules! loader {
    ($name:ident, $task:literal) => {
        /// Load an owned native task instance.
        /// # Safety
        /// `options` must point to a live NUL-terminated UTF-8 JSON string.
        /// Free the returned string with `candle_instance_free_string`.
        #[no_mangle]
        pub unsafe extern "C" fn $name(options: *const c_char) -> *mut c_char {
            reply(|| {
                let operation = || insert(load(serde_json::from_str(string(options)?)?, $task)?);
                operation().map_err(|error| error_with_context(error, "load"))
            })
        }
    };
}
loader!(candle_instance_load_backbone, "backbone");
loader!(candle_instance_load_sequence, "sequence");
loader!(candle_instance_load_token, "token");
loader!(candle_instance_load_nli, "nli");
loader!(candle_instance_load_hallucination, "hallucination");
loader!(candle_instance_load_embedding, "embedding");

/// Clone a binding reference. The returned handle closes independently.
#[no_mangle]
pub extern "C" fn candle_instance_clone(handle: u64) -> *mut c_char {
    reply(|| insert(get(handle)?))
}

/// Release a binding reference; active calls keep their own strong reference.
#[no_mangle]
pub extern "C" fn candle_instance_close(handle: u64) -> *mut c_char {
    reply(|| close(handle))
}

/// Return immutable effective task/provider information.
#[no_mangle]
pub extern "C" fn candle_instance_info(handle: u64) -> *mut c_char {
    reply(|| Ok(get(handle)?.info.clone()))
}

/// Bind a sequence or token head to an explicitly selected backbone.
/// # Safety
/// `path` and `task` must be live NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_bind_head(
    handle: u64,
    path: *const c_char,
    task: *const c_char,
) -> *mut c_char {
    reply(|| {
        insert(bind_head(
            get(handle)?.as_ref(),
            string(path)?,
            string(task)?,
        )?)
    })
}

/// Classify text with a sequence handle.
/// # Safety
/// `text` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_sequence(handle: u64, text: *const c_char) -> *mut c_char {
    reply(|| get(handle)?.sequence(string(text)?))
}

/// Classify UTF-8 token spans with a token handle.
/// # Safety
/// `text` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_tokens(handle: u64, text: *const c_char) -> *mut c_char {
    reply(|| get(handle)?.tokens(string(text)?))
}

/// Classify a premise and hypothesis while retaining the hypothesis window.
/// # Safety
/// String arguments must point to live NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_nli(
    handle: u64,
    premise: *const c_char,
    hypothesis: *const c_char,
) -> *mut c_char {
    reply(|| get(handle)?.nli(string(premise)?, string(hypothesis)?))
}

/// Detect answer-relative hallucinated spans with the maintained input template.
/// # Safety
/// String arguments must point to live NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_hallucination(
    handle: u64,
    context: *const c_char,
    question: *const c_char,
    answer: *const c_char,
    threshold: f32,
) -> *mut c_char {
    reply(|| {
        get(handle)?.hallucination(
            string(context)?,
            string(question)?,
            string(answer)?,
            threshold,
        )
    })
}

/// Embed text with optional Matryoshka dimension and layer.
/// # Safety
/// `text` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_embedding(
    handle: u64,
    text: *const c_char,
    dimension: usize,
    layer: usize,
) -> *mut c_char {
    reply(|| get(handle)?.embedding(string(text)?, dimension, layer))
}

/// Free a string returned by this module, including error responses.
/// # Safety
/// `ptr` is null or was returned by this module and has not yet been freed.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Embed encoded PNG/JPEG image bytes with owned multimodal weights.
/// # Safety
/// `bytes` must point to `length` live bytes for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_image(
    handle: u64,
    bytes: *const u8,
    length: usize,
    dimension: usize,
) -> *mut c_char {
    reply(|| {
        ensure!(
            !bytes.is_null() && length > 0 && length <= isize::MAX as usize,
            "configuration: invalid image buffer"
        );
        get(handle)?.image(std::slice::from_raw_parts(bytes, length), dimension)
    })
}

/// Embed an explicitly shaped mel spectrogram with owned multimodal weights.
/// # Safety
/// `samples` must point to `length` live f32 values for this call.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_audio(
    handle: u64,
    samples: *const f32,
    length: usize,
    mel_bins: usize,
    frames: usize,
    dimension: usize,
) -> *mut c_char {
    reply(|| {
        ensure!(
            !samples.is_null()
                && length > 0
                && length <= isize::MAX as usize / std::mem::size_of::<f32>(),
            "configuration: invalid spectrogram buffer"
        );
        get(handle)?.audio(
            std::slice::from_raw_parts(samples, length),
            mel_bins,
            frames,
            dimension,
        )
    })
}

loader!(candle_instance_load_guard, "guard");
loader!(candle_instance_load_generative, "generative");

/// Evaluate prompt/response safety using the existing generation template.
/// # Safety
/// String arguments must point to live NUL-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_guard(
    handle: u64,
    text: *const c_char,
    mode: *const c_char,
) -> *mut c_char {
    reply(|| get(handle)?.guard(string(text)?, string(mode)?))
}

/// Score labels with a prepared adapter or explicit zero-shot candidates.
/// # Safety
/// Strings are live NUL-terminated UTF-8; categories is a JSON string array.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_generative(
    handle: u64,
    text: *const c_char,
    adapter: *const c_char,
    categories: *const c_char,
    multi_token: bool,
) -> *mut c_char {
    reply(|| {
        let adapter = string(adapter)?;
        get(handle)?.generative(
            string(text)?,
            (!adapter.is_empty()).then_some(adapter),
            serde_json::from_str(string(categories)?)?,
            multi_token,
        )
    })
}

/// Compute original UTF-8 byte windows using this embedding instance's tokenizer.
/// # Safety
/// `text` must point to a live NUL-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn candle_instance_text_windows(
    handle: u64,
    text: *const c_char,
    max_tokens: usize,
) -> *mut c_char {
    reply(|| get(handle)?.text_windows(string(text)?, max_tokens))
}
