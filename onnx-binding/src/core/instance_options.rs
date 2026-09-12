//! Explicit execution configuration for owned ONNX Runtime sessions.
//!
//! Legacy loaders retain their historical provider policy. Instance loaders use
//! this policy exclusively: an unavailable GPU provider fails preparation.

use crate::core::unified_error::{errors, UnifiedResult};
use ort::{execution_providers::CPUExecutionProvider, session::Session};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Provider {
    #[default]
    Cpu,
    Migraphx,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    /// Preserve the graph's own types; this does not assert all weights are FP32.
    #[default]
    Native,
    /// Explicitly enable MIGraphX FP16 conversion.
    Fp16,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Overflow {
    #[default]
    Reject,
    TruncateRight,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct InstanceOptions {
    pub model_path: String,
    pub model_file: Option<String>,
    pub provider: Provider,
    pub device_id: i32,
    pub precision: Precision,
    pub max_input_tokens: Option<usize>,
    pub overflow: Overflow,
    pub intra_threads: Option<usize>,
    pub profile_prefix: Option<String>,
    #[serde(skip)]
    pub evidence: Arc<Mutex<Vec<SessionEvidence>>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEvidence {
    pub runtime_build: String,
    pub graph: String,
    pub provider: &'static str,
    pub device_id: i32,
    pub precision: Precision,
    pub cpu_fallback_disabled: bool,
    pub profile_prefix: Option<String>,
}

// ROCm/onnxruntime@2716b9b93a reads these after explicit provider options.
// A model-cache override can load a compiled program whose cache key omits
// precision. Reject these process-wide inputs instead of changing the caller's
// environment or advertising execution facts the instance cannot guarantee.
const MIGRAPHX_EXECUTION_OVERRIDES: [&str; 5] = [
    "ORT_MIGRAPHX_FP16_ENABLE",
    "ORT_MIGRAPHX_BF16_ENABLE",
    "ORT_MIGRAPHX_FP8_ENABLE",
    "ORT_MIGRAPHX_INT8_ENABLE",
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
];

impl InstanceOptions {
    pub fn validate(&self) -> UnifiedResult<()> {
        if self.model_path.is_empty() || self.device_id < 0 {
            return Err(errors::config_error(
                "instance",
                "model_path must be nonempty and device_id nonnegative",
            ));
        }
        if self.provider == Provider::Cpu
            && (self.device_id != 0 || self.precision != Precision::Native)
        {
            return Err(errors::config_error(
                "provider",
                "CPU requires device_id=0 and native graph precision",
            ));
        }
        if self.max_input_tokens == Some(0) || self.intra_threads == Some(0) {
            return Err(errors::config_error(
                "budget",
                "token and thread budgets must be positive",
            ));
        }
        if self.profile_prefix.as_deref() == Some("") {
            return Err(errors::config_error(
                "profile_prefix",
                "must be nonempty when supplied",
            ));
        }
        if self.provider == Provider::Migraphx {
            for name in MIGRAPHX_EXECUTION_OVERRIDES {
                if std::env::var_os(name).is_some_and(|value| !value.is_empty()) {
                    return Err(errors::config_error(
                        "provider",
                        &format!(
                            "owned MIGraphX execution forbids nonempty {name}; configure precision per instance and leave compiled-model caching disabled"
                        ),
                    ));
                }
            }
        }
        #[cfg(not(feature = "migraphx"))]
        if self.provider == Provider::Migraphx {
            return Err(errors::config_error(
                "provider",
                "MIGraphX support was not compiled; CPU fallback is forbidden",
            ));
        }
        Ok(())
    }

    pub fn effective_limit(&self, task_limit: usize) -> UnifiedResult<usize> {
        let limit = self.max_input_tokens.unwrap_or(task_limit);
        if limit == 0 || limit > task_limit {
            return Err(errors::validation(
                "max_input_tokens",
                &format!("1..={task_limit}"),
                &limit.to_string(),
            ));
        }
        Ok(limit)
    }

    pub fn configure_tokenizer(
        &self,
        tokenizer: &mut Tokenizer,
        task_limit: usize,
    ) -> UnifiedResult<()> {
        let max_length = self.effective_limit(task_limit)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        Ok(())
    }

    pub fn select_graph(&self, candidates: Vec<PathBuf>) -> UnifiedResult<PathBuf> {
        if let Some(ref file) = self.model_file {
            let file = Path::new(file);
            let path = if file.is_absolute() {
                file.to_path_buf()
            } else {
                Path::new(&self.model_path).join(file)
            };
            if !path.is_file() {
                return Err(errors::file_not_found(&path.display().to_string()));
            }
            return Ok(path);
        }
        candidates
            .into_iter()
            .find(|p| p.is_file())
            .ok_or_else(|| errors::model_load(&self.model_path, "no supported ONNX graph found"))
    }

    pub fn create_session(&self, path: &Path) -> UnifiedResult<Session> {
        self.validate()?;
        let ort_error = |e: ort::Error| errors::ort_error(&e.to_string());
        let mut builder = Session::builder()
            .map_err(ort_error)?
            .with_no_environment_execution_providers()
            .map_err(ort_error)?;
        if let Some(threads) = self.intra_threads {
            builder = builder.with_intra_threads(threads).map_err(ort_error)?;
        }
        let provider_name = match self.provider {
            Provider::Cpu => {
                builder = builder
                    .with_execution_providers([CPUExecutionProvider::default()
                        .build()
                        .error_on_failure()])
                    .map_err(ort_error)?;
                "CPUExecutionProvider"
            }
            Provider::Migraphx => {
                #[cfg(feature = "migraphx")]
                {
                    builder = builder
                        .with_config_entry("session.disable_cpu_ep_fallback", "1")
                        .map_err(ort_error)?;
                    append_migraphx(
                        &mut builder,
                        self.device_id,
                        self.precision == Precision::Fp16,
                    )?;
                }
                #[cfg(not(feature = "migraphx"))]
                return Err(errors::config_error("provider", "MIGraphX is unavailable"));
                #[cfg(feature = "migraphx")]
                "MIGraphXExecutionProvider"
            }
        };
        static PROFILE_SEQUENCE: AtomicU64 = AtomicU64::new(1);
        let profile_prefix = self.profile_prefix.as_ref().map(|prefix| {
            format!(
                "{prefix}-{}",
                PROFILE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            )
        });
        if let Some(ref prefix) = profile_prefix {
            builder = builder.with_profiling(prefix).map_err(ort_error)?;
        }
        let session = builder
            .commit_from_file(path)
            .map_err(|e| errors::model_load(&path.display().to_string(), &e.to_string()))?;
        self.evidence.lock().push(SessionEvidence {
            runtime_build: runtime_build_info(),
            graph: path.display().to_string(),
            provider: provider_name,
            device_id: self.device_id,
            precision: self.precision,
            cpu_fallback_disabled: self.provider == Provider::Migraphx,
            profile_prefix,
        });
        Ok(session)
    }
}

// ROCm's ORT 1.22.1 build changes the frozen upstream provider struct without
// changing ORT_API_VERSION. Never pass ort-sys's upstream layout to this build.
// Source: ROCm/onnxruntime@2716b9b93a, onnxruntime_c_api.h.
#[cfg(feature = "migraphx")]
#[repr(C)]
struct Rocm7MigraphxOptions {
    device_id: i32,
    fp16: i32,
    bf16: i32,
    fp8: i32,
    int8: i32,
    native_calibration: i32,
    calibration_table: *const std::ffi::c_char,
    cache_dir: *const std::ffi::c_char,
    exhaustive_tune: bool,
    memory_limit: usize,
    arena_extend_strategy: i32,
}

#[cfg(feature = "migraphx")]
fn append_migraphx(
    builder: &mut ort::session::builder::SessionBuilder,
    device_id: i32,
    fp16: bool,
) -> UnifiedResult<()> {
    use ort::AsPointer;
    use std::ffi::CString;
    let build_info = runtime_build_info();
    if build_info.contains("git-commit-id=2716b9b93a,") {
        let options = Rocm7MigraphxOptions {
            device_id,
            fp16: fp16.into(),
            bf16: 0,
            fp8: 0,
            int8: 0,
            native_calibration: 0,
            calibration_table: std::ptr::null(),
            cache_dir: std::ptr::null(),
            exhaustive_tune: false,
            memory_limit: usize::MAX,
            arena_extend_strategy: 0,
        };
        // SAFETY: the checked build commit's public C header defines exactly
        // this repr(C) layout. ORT copies options synchronously during append.
        let status = unsafe {
            (ort::api().SessionOptionsAppendExecutionProvider_MIGraphX)(
                builder.ptr_mut(),
                (&options as *const Rocm7MigraphxOptions).cast(),
            )
        };
        return unsafe { ort::error::status_to_result(status) }
            .map_err(|e| errors::ort_error(&e.to_string()));
    }
    let keys = [c"device_id", c"migraphx_fp16_enable"];
    let values = [
        CString::new(device_id.to_string()).unwrap(),
        CString::new(if fp16 { "1" } else { "0" }).unwrap(),
    ];
    let key_ptrs = keys.map(|s| s.as_ptr());
    let value_ptrs = values.each_ref().map(|s| s.as_ptr());
    // Other versions must support the named API. Do not guess a legacy layout
    // from the API version, which is identical across incompatible builds.
    let status = unsafe {
        (ort::api().SessionOptionsAppendExecutionProvider)(
            builder.ptr_mut(),
            c"MIGraphX".as_ptr(),
            key_ptrs.as_ptr(),
            value_ptrs.as_ptr(),
            keys.len(),
        )
    };
    unsafe { ort::error::status_to_result(status) }.map_err(|e| {
        errors::config_error(
            "provider",
            &format!("MIGraphX runtime has no supported options ABI: {e}"),
        )
    })
}

fn runtime_build_info() -> String {
    let pointer = unsafe { (ort::api().GetBuildInfoString)() };
    if pointer.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_precision_and_device_are_explicit() {
        let mut options = InstanceOptions {
            model_path: "unused".into(),
            ..Default::default()
        };
        assert!(options.validate().is_ok());
        options.device_id = 1;
        assert!(options.validate().is_err());
        options.device_id = 0;
        options.precision = Precision::Fp16;
        assert!(options.validate().is_err());
    }

    #[test]
    fn task_limit_cannot_be_raised_by_a_deployment_budget() {
        let options = InstanceOptions {
            max_input_tokens: Some(513),
            ..Default::default()
        };
        assert!(options.effective_limit(512).is_err());
    }

    #[test]
    fn migraphx_environment_overrides_cannot_change_owned_execution() {
        const CASE: &str = "CORE_ORT_EXECUTION_ENV_TEST_CASE";
        const TEST: &str = "core::instance_options::tests::migraphx_environment_overrides_cannot_change_owned_execution";
        if let Ok(name) = std::env::var(CASE) {
            let original = std::env::var_os(&name).expect("child environment value");
            let options = InstanceOptions {
                model_path: "unused".into(),
                provider: Provider::Migraphx,
                ..Default::default()
            };
            let result = options.validate();
            if original.is_empty() {
                // A CPU-only build may still reject missing MIGraphX support.
                assert!(!result.is_err_and(|error| error.to_string().contains(&name)));
            } else {
                let error = result.unwrap_err().to_string();
                assert!(error.contains(&name) && error.contains("forbids nonempty"));
            }
            assert_eq!(std::env::var_os(&name).as_ref(), Some(&original));
            assert!(InstanceOptions {
                provider: Provider::Cpu,
                ..options
            }
            .validate()
            .is_ok());
            return;
        }

        // Serialize isolated child processes: no environment mutation races
        // with other tests, and the parent caller's values stay untouched.
        for name in MIGRAPHX_EXECUTION_OVERRIDES {
            let original = std::env::var_os(name);
            for value in ["", "0", "1", "invalid-value"] {
                let mut child = std::process::Command::new(std::env::current_exe().unwrap());
                child.args(["--exact", TEST, "--test-threads=1"]);
                for variable in MIGRAPHX_EXECUTION_OVERRIDES {
                    child.env_remove(variable);
                }
                let output = child.env(CASE, name).env(name, value).output().unwrap();
                assert!(
                    output.status.success(),
                    "{name} case {value:?}: {} {}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            assert_eq!(std::env::var_os(name), original);
        }
    }

    #[cfg(not(feature = "migraphx"))]
    #[test]
    fn gpu_request_never_becomes_cpu_in_a_cpu_build() {
        let options = InstanceOptions {
            model_path: "unused".into(),
            provider: Provider::Migraphx,
            ..Default::default()
        };
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("CPU fallback is forbidden"));
    }
}
