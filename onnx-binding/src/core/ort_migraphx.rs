//! AMD ONNX Runtime execution-provider helpers.

use ort::ortsys;
use ort::session::builder::SessionBuilder;
use ort::session::Session;
use ort::AsPointer;
use std::ffi::CString;
use std::path::Path;

/// Result of an AMD provider-selection attempt.
pub struct AmdSession {
    pub session: Session,
    pub provider: &'static str,
    pub fallback_reason: Option<String>,
}

/// Register MIGraphX through ONNX Runtime's generic provider-options API.
///
/// The typed `OrtMIGraphXProviderOptions` layout changed across ORT releases.
/// The generic string-key API matches ONNX Runtime's Python provider path and
/// avoids ABI mismatches when dynamically loading AMD ORT 1.23.x.
pub fn append_migraphx_execution_provider(
    builder: &mut SessionBuilder,
    device_id: i32,
) -> Result<(), ort::Error> {
    let provider = CString::new("MIGraphXExecutionProvider").expect("static provider name");
    let device_key = CString::new("device_id").expect("static option key");
    let device_value = CString::new(device_id.to_string()).expect("device id string");

    let keys = [device_key.as_ptr()];
    let values = [device_value.as_ptr()];

    ort::ortsys![unsafe SessionOptionsAppendExecutionProvider(
        builder.ptr_mut(),
        provider.as_ptr(),
        keys.as_ptr(),
        values.as_ptr(),
        keys.len(),
    )?];

    Ok(())
}

/// Return the configured CK FlashAttention custom-op library, if any.
pub fn ck_flash_attention_library() -> Option<String> {
    std::env::var("ORT_CK_FLASH_ATTN_LIB")
        .ok()
        .filter(|value| !value.is_empty())
}

/// Return the CK FlashAttention library only for FA-optimized ONNX artifacts.
pub fn ck_flash_attention_library_for_model(onnx_path: &Path) -> Option<String> {
    if is_ck_flash_attention_model(onnx_path) {
        ck_flash_attention_library()
    } else {
        None
    }
}

/// Detect the shipped CK FlashAttention artifact naming convention.
pub fn is_ck_flash_attention_model(onnx_path: &Path) -> bool {
    onnx_path
        .file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            let name = name.to_ascii_lowercase();
            name.contains("_fa") || name.contains("flash")
        })
        .unwrap_or(false)
}

/// Create an AMD session using the Semantic Router AMD provider policy.
///
/// Portable ONNX artifacts use MIGraphX first, then ROCm. CK FlashAttention
/// artifacts are an explicit exception: the custom op is registered only on
/// the ROCm path, so those sessions skip MIGraphX and go straight to ROCm.
pub fn create_amd_session(onnx_path: &Path, ck_fa_lib: Option<&str>) -> Result<AmdSession, String> {
    let mut fallback_reasons = Vec::new();

    if ck_fa_lib.is_none() {
        #[cfg(feature = "migraphx")]
        {
            println!("INFO: Attempting MIGraphX execution provider...");
            match create_migraphx_session(onnx_path) {
                Ok(session) => {
                    println!("INFO: Using MIGraphX execution provider (AMD GPU) - verified");
                    return Ok(AmdSession {
                        session,
                        provider: "migraphx",
                        fallback_reason: None,
                    });
                }
                Err(error) => {
                    let reason = format!("MIGraphX EP failed: {error}");
                    println!("WARN: {reason}");
                    fallback_reasons.push(reason);
                }
            }
        }
        #[cfg(not(feature = "migraphx"))]
        {
            fallback_reasons.push("MIGraphX feature is not enabled".to_string());
        }
    } else {
        let reason = "CK FlashAttention custom op requires ROCm EP; skipping MIGraphX".to_string();
        println!("INFO: {reason}");
        fallback_reasons.push(reason);
    }

    #[cfg(feature = "rocm")]
    {
        println!("INFO: Attempting ROCm execution provider...");
        match create_rocm_session(onnx_path, ck_fa_lib) {
            Ok(session) => {
                println!("INFO: Using ROCm execution provider (AMD GPU) - verified");
                return Ok(AmdSession {
                    session,
                    provider: "rocm",
                    fallback_reason: if fallback_reasons.is_empty() {
                        None
                    } else {
                        Some(fallback_reasons.join("; "))
                    },
                });
            }
            Err(error) => {
                let reason = format!("ROCm EP failed: {error}");
                println!("WARN: {reason}");
                fallback_reasons.push(reason);
            }
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        fallback_reasons.push("ROCm feature is not enabled".to_string());
    }

    if fallback_reasons.is_empty() {
        Err(format!(
            "no AMD execution provider was available for {}",
            onnx_path.display()
        ))
    } else {
        Err(fallback_reasons.join("; "))
    }
}

#[cfg(feature = "migraphx")]
fn create_migraphx_session(onnx_path: &Path) -> Result<Session, ort::Error> {
    let mut builder = Session::builder()?;
    append_migraphx_execution_provider(&mut builder, 0)?;
    builder.commit_from_file(onnx_path)
}

#[cfg(feature = "rocm")]
fn create_rocm_session(onnx_path: &Path, ck_fa_lib: Option<&str>) -> Result<Session, ort::Error> {
    use crate::core::gpu_memory;
    use ort::execution_providers::{ArenaExtendStrategy, ROCmExecutionProvider};

    if let Some(lib) = ck_fa_lib {
        println!("INFO: CK FlashAttention custom op library: {lib}");
    }

    let mem_limit = gpu_memory::get_gpu_mem_limit();
    let builder =
        Session::builder()?.with_execution_providers([ROCmExecutionProvider::default()
            .with_mem_limit(mem_limit)
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .build()
            .error_on_failure()])?;
    let builder = if let Some(lib) = ck_fa_lib {
        builder.with_operator_library(lib)?
    } else {
        builder
    };
    builder.commit_from_file(onnx_path)
}
