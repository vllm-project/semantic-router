//! Explicit execution configuration for owned ONNX Runtime sessions.
//!
//! Legacy loaders retain their historical provider policy. Instance loaders use
//! this policy exclusively: an unavailable GPU provider fails preparation.

use crate::core::{
    artifact_identity::{ArtifactDigest, ArtifactSnapshot},
    compilation_cache::{CompilationCacheLease, CompilationIdentity, SharedCacheEvidence},
    execution_contract::{
        session_input_schema, validate_contract, validate_session, ExecutionInput,
    },
    onnx_artifacts::capture_onnx,
    unified_error::{errors, UnifiedResult},
};
use ort::{execution_providers::CPUExecutionProvider, session::Session};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::OsString,
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
    Rocm,
}

/// An installed, trusted custom-op implementation; never a caller-supplied path.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CustomOpsProfile {
    #[default]
    None,
    CkFlashAttention,
}

const CK_FLASH_ATTENTION_LIBRARY: &str = "/usr/local/lib/libort_ck_flash_attn.so.1";

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
    pub custom_ops_profile: CustomOpsProfile,
    /// Permit ORT's CPU EP explicitly. Profiling is required; this does not
    /// promise that the CPU nodes are limited to shape/control operations.
    pub allow_cpu_fallback: bool,
    pub max_input_tokens: Option<usize>,
    /// Whole-document budget used only by typed sequence/token window tasks.
    /// Omission preserves the physical input budget.
    pub document_max_input_tokens: Option<usize>,
    /// Physical execution window; cannot raise the single-forward input budget.
    pub execution_max_input_tokens: Option<usize>,
    /// Optional owned MIGraphX compiled-program storage root.
    pub compilation_cache_dir: Option<PathBuf>,
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
    pub custom_ops_profile: CustomOpsProfile,
    pub custom_ops_library: Option<String>,
    pub custom_ops_sha256: Option<String>,
    pub profile_prefix: Option<String>,
    pub artifacts: Vec<ArtifactDigest>,
    pub execution_max_input_tokens: Option<usize>,
    pub execution_inputs: Vec<ExecutionInput>,
    /// Actual loaded session declarations; -1 denotes a dynamic dimension.
    pub input_schema: Vec<ExecutionInput>,
    pub compilation_cache: Option<SharedCacheEvidence>,
    /// Effective provider compiler controls captured once before preparation.
    pub compiler_flags: BTreeMap<String, String>,
}

pub struct PreparedSession {
    pub session: Session,
    pub cache_lease: Option<CompilationCacheLease>,
    pub artifacts: Vec<ArtifactSnapshot>,
}

// ROCm/onnxruntime@2716b9b93a reads these after explicit provider options.
// A model-cache override can load a compiled program whose cache key omits
// precision. Reject these process-wide inputs instead of changing the caller's
// environment or advertising execution facts the instance cannot guarantee.
const MIGRAPHX_EXECUTION_OVERRIDES: [&str; 6] = [
    "ORT_MIGRAPHX_FP16_ENABLE",
    "ORT_MIGRAPHX_BF16_ENABLE",
    "ORT_MIGRAPHX_FP8_ENABLE",
    "ORT_MIGRAPHX_INT8_ENABLE",
    "ORT_MIGRAPHX_MODEL_CACHE_PATH",
    "ORT_MIGRAPHX_EXHAUSTIVE_TUNE_OPS",
];

impl InstanceOptions {
    pub fn validate(&self) -> UnifiedResult<()> {
        self.validate_configuration()?;
        self.compiler_flags(std::env::vars_os())?;
        self.validate_provider_available()
    }

    fn validate_configuration(&self) -> UnifiedResult<()> {
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
        if self.max_input_tokens == Some(0)
            || self.document_max_input_tokens == Some(0)
            || self.execution_max_input_tokens == Some(0)
            || self.intra_threads == Some(0)
        {
            return Err(errors::config_error(
                "budget",
                "token and thread budgets must be positive",
            ));
        }
        if self.provider == Provider::Rocm && self.precision != Precision::Native {
            return Err(errors::config_error(
                "precision",
                "ROCm preserves the selected graph's native types; fp16 conversion is MIGraphX-only",
            ));
        }
        if self.custom_ops_profile != CustomOpsProfile::None && self.provider != Provider::Rocm {
            return Err(errors::config_error(
                "custom_ops_profile",
                "ck_flash_attention requires the ROCm execution provider",
            ));
        }
        if self.allow_cpu_fallback
            && (self.provider != Provider::Rocm || self.profile_prefix.is_none())
        {
            return Err(errors::config_error(
                "allow_cpu_fallback",
                "explicit CPU fallback requires ROCm and an ORT profile prefix; node placement must be audited",
            ));
        }
        if self.profile_prefix.as_deref() == Some("") {
            return Err(errors::config_error(
                "profile_prefix",
                "must be nonempty when supplied",
            ));
        }
        if let Some(root) = &self.compilation_cache_dir {
            if self.provider != Provider::Migraphx || root.as_os_str().is_empty() {
                return Err(errors::config_error(
                    "compilation_cache_dir",
                    "compiled-program caching requires MIGraphX and a nonempty directory",
                ));
            }
        }
        Ok(())
    }

    fn validate_provider_available(&self) -> UnifiedResult<()> {
        #[cfg(not(feature = "migraphx"))]
        if self.provider == Provider::Migraphx {
            return Err(errors::config_error(
                "provider",
                "MIGraphX support was not compiled; CPU fallback is forbidden",
            ));
        }
        #[cfg(not(feature = "rocm"))]
        if self.provider == Provider::Rocm {
            return Err(errors::config_error(
                "provider",
                "ROCm support was not compiled; CPU fallback is forbidden",
            ));
        }
        Ok(())
    }

    /// Process compiler controls are deployment inputs. Capture them once per
    /// session; cache selection and representation evidence share these bytes.
    /// Callers must keep the process environment stable during preparation.
    pub(crate) fn compiler_flags(
        &self,
        environment: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> UnifiedResult<BTreeMap<String, String>> {
        if self.provider == Provider::Rocm {
            return Ok([
                ("arena_extend_strategy".into(), "kSameAsRequested".into()),
                ("memory_limit".into(), "usize_max".into()),
            ]
            .into());
        }
        if self.provider != Provider::Migraphx {
            return Ok(BTreeMap::new());
        }
        let mut flags: BTreeMap<String, String> = [
            ("bf16", "false"),
            ("fp8", "false"),
            ("int8", "false"),
            ("exhaustive_tune", "false"),
            ("memory_limit", "usize_max"),
            ("arena_extend_strategy", "0"),
            ("cpu_fallback", "disabled"),
            ("graph_optimization", "ort_default"),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), value.into()))
        .collect();
        for (name, value) in environment {
            let Some(name) = name.to_str() else { continue };
            if !name.starts_with("MIGRAPHX_") && !name.starts_with("ORT_MIGRAPHX_") {
                continue;
            }
            if MIGRAPHX_EXECUTION_OVERRIDES.contains(&name) && !value.is_empty() {
                return Err(errors::config_error(
                    "provider",
                    &format!("owned MIGraphX execution forbids nonempty {name}; configure precision per instance and use only typed compiled-program caching"),
                ));
            }
            let value = value.to_str().ok_or_else(|| {
                errors::config_error("provider", "compiler environment must be UTF-8")
            })?;
            flags.insert(format!("environment:{name}"), value.into());
        }
        flags.insert(
            "intra_threads".into(),
            self.intra_threads
                .map_or("default".into(), |value| value.to_string()),
        );
        Ok(flags)
    }

    /// Include this same file in the owning session's artifact snapshot before
    /// loading and verify it afterwards, alongside every graph/external tensor.
    pub fn custom_ops_library_path(&self) -> UnifiedResult<Option<PathBuf>> {
        self.validate()?;
        Ok(self.resolved_custom_ops_library())
    }

    fn resolved_custom_ops_library(&self) -> Option<PathBuf> {
        match self.custom_ops_profile {
            CustomOpsProfile::None => None,
            CustomOpsProfile::CkFlashAttention => Some(PathBuf::from(CK_FLASH_ATTENTION_LIBRARY)),
        }
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

    pub fn document_limit(
        &self,
        physical_limit: usize,
        supports_windows: bool,
    ) -> UnifiedResult<usize> {
        if self.document_max_input_tokens.is_some() && !supports_windows {
            return Err(errors::config_error(
                "document_max_input_tokens",
                "requires a typed sequence or token window task",
            ));
        }
        let limit = self.document_max_input_tokens.unwrap_or(physical_limit);
        if limit == 0 || limit < physical_limit {
            return Err(errors::validation(
                "document_max_input_tokens",
                &format!("at least {physical_limit}"),
                &limit.to_string(),
            ));
        }
        Ok(limit)
    }

    pub fn execution_limit(&self, task_limit: usize) -> UnifiedResult<usize> {
        let physical_limit = self.effective_limit(task_limit)?;
        let limit = self.execution_max_input_tokens.unwrap_or(physical_limit);
        if limit == 0 || limit > physical_limit {
            return Err(errors::validation(
                "execution_max_input_tokens",
                &format!("1..={physical_limit}"),
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
        if self.compilation_cache_dir.is_some() {
            return Err(errors::config_error(
                "compilation_cache",
                "this architecture must provide resolved inputs and retain the cache lease",
            ));
        }
        Ok(self.create_session_impl(path, &[], false)?.session)
    }

    /// Prepare a dynamic, uncached architecture while retaining artifact evidence.
    pub fn prepare_session(&self, path: &Path) -> UnifiedResult<PreparedSession> {
        if self.compilation_cache_dir.is_some() {
            return Err(errors::config_error(
                "compilation_cache",
                "cached preparation requires resolved execution inputs",
            ));
        }
        self.create_session_impl(path, &[], false)
    }

    pub fn create_session_with_contract(
        &self,
        path: &Path,
        inputs: &[ExecutionInput],
    ) -> UnifiedResult<PreparedSession> {
        validate_contract(inputs)?;
        let mut inputs = inputs.to_vec();
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        self.create_session_impl(path, &inputs, false)
    }

    /// Prepare an explicitly fixed input contract on any provider. The owner
    /// must pad/reject requests to these exact shapes. Resolving symbolic sizes
    /// before partitioning lets ORT fold shape-only work without CPU fallback.
    pub fn create_session_with_fixed_contract(
        &self,
        path: &Path,
        inputs: &[ExecutionInput],
    ) -> UnifiedResult<PreparedSession> {
        validate_contract(inputs)?;
        let mut inputs = inputs.to_vec();
        inputs.sort_by(|a, b| a.name.cmp(&b.name));
        self.create_session_impl(path, &inputs, true)
    }

    fn create_session_impl(
        &self,
        path: &Path,
        inputs: &[ExecutionInput],
        fixed_shapes: bool,
    ) -> UnifiedResult<PreparedSession> {
        self.validate_configuration()?;
        let mut compiler_flags = self.compiler_flags(std::env::vars_os())?;
        if self.provider == Provider::Migraphx || fixed_shapes {
            compiler_flags.insert(
                "input_dimensions".into(),
                if inputs.is_empty() {
                    "graph_declared"
                } else {
                    "resolved_before_partitioning"
                }
                .into(),
            );
        }
        self.validate_provider_available()?;
        let custom_ops_library = self.resolved_custom_ops_library();
        let mut artifacts = capture_onnx(path)
            .map_err(|error| errors::model_load(&path.display().to_string(), &error.to_string()))?;
        if let Some(library) = &custom_ops_library {
            artifacts.push(
                ArtifactSnapshot::capture(library, "custom-ops").map_err(|error| {
                    errors::model_load(&library.display().to_string(), &error.to_string())
                })?,
            );
        }
        let (cache_lease, runtime_artifacts) = if let Some(root) = &self.compilation_cache_dir {
            validate_contract(inputs)?;
            #[cfg(feature = "migraphx")]
            {
                // Load this provider through its public factory before identifying
                // its actual libraries. This creates no model session or inference.
                let mut probe =
                    Session::builder().map_err(|error| errors::ort_error(&error.to_string()))?;
                append_migraphx(
                    &mut probe,
                    self.device_id,
                    self.precision == Precision::Fp16,
                    None,
                )?;
                drop(probe);
            }
            let gpu = crate::core::migraphx_identity::gpu_identity(self.device_id)
                .map_err(|error| errors::config_error("compilation_cache", &error.to_string()))?;
            let runtime = crate::core::migraphx_identity::runtime_artifacts()
                .map_err(|error| errors::config_error("compilation_cache", &error.to_string()))?;
            let runtime_digests = runtime
                .iter()
                .map(|item| item.digest.clone())
                .collect::<Vec<_>>();
            let identity = CompilationIdentity {
                version: 1,
                artifacts: artifacts.iter().map(|item| item.digest.clone()).collect(),
                inputs: inputs.to_vec(),
                precision: match self.precision {
                    Precision::Native => "native",
                    Precision::Fp16 => "fp16",
                }
                .into(),
                runtime_build: runtime_build_info(),
                runtime_artifacts: runtime_digests.clone(),
                gpu,
                compiler_flags: compiler_flags.clone(),
            };
            let mut snapshots = artifacts.clone();
            snapshots.extend(runtime);
            (
                Some(CompilationCacheLease::prepare(root, identity, snapshots)?),
                Some(runtime_digests),
            )
        } else {
            (None, None)
        };
        let ort_error = |e: ort::Error| errors::ort_error(&e.to_string());
        let mut builder = Session::builder()
            .map_err(ort_error)?
            .with_no_environment_execution_providers()
            .map_err(ort_error)?;
        if (self.provider == Provider::Migraphx || fixed_shapes) && !inputs.is_empty() {
            let declared = crate::core::onnx_artifacts::input_schema(path)
                .map_err(|error| errors::config_error("execution_inputs", &error.to_string()))?;
            let overrides = crate::core::execution_contract::dimension_overrides(&declared, inputs)
                .map_err(|error| errors::config_error("execution_inputs", &error.to_string()))?;
            for (symbol, size) in overrides {
                builder = builder
                    .with_dimension_override(symbol, size)
                    .map_err(ort_error)?;
            }
        }
        if let Some(threads) = self.intra_threads {
            builder = builder.with_intra_threads(threads).map_err(ort_error)?;
        }
        let custom_ops_sha256 = artifacts
            .iter()
            .find(|item| item.digest.role == "custom-ops")
            .map(|item| item.digest.sha256.clone());
        if let Some(library) = &custom_ops_library {
            builder = builder.with_operator_library(library).map_err(ort_error)?;
        }
        let provider_name = match self.provider {
            Provider::Cpu => {
                builder = builder
                    .with_execution_providers([CPUExecutionProvider::default()
                        // Reuse CPU work buffers across operators and Loop iterations.
                        // The crate default disables this arena, repeatedly faulting
                        // and zeroing large attention buffers on long inputs.
                        .with_arena_allocator(true)
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
                        cache_lease.as_ref().map(|lease| lease.directory.as_path()),
                    )?;
                }
                #[cfg(not(feature = "migraphx"))]
                return Err(errors::config_error("provider", "MIGraphX is unavailable"));
                #[cfg(feature = "migraphx")]
                "MIGraphXExecutionProvider"
            }
            Provider::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    builder = builder
                        .with_config_entry(
                            "session.disable_cpu_ep_fallback",
                            if self.allow_cpu_fallback { "0" } else { "1" },
                        )
                        .map_err(ort_error)?
                        .with_execution_providers([
                            ort::execution_providers::ROCmExecutionProvider::default()
                                .with_device_id(self.device_id)
                                // Loop subgraphs retain their working arenas.
                                // Geometric growth can exhaust VRAM despite a
                                // bounded attention working set; reserve the
                                // requested size without changing execution.
                                .with_arena_extend_strategy(
                                    ort::execution_providers::ArenaExtendStrategy::SameAsRequested,
                                )
                                .build()
                                .error_on_failure(),
                        ])
                        .map_err(ort_error)?;
                }
                #[cfg(not(feature = "rocm"))]
                return Err(errors::config_error("provider", "ROCm is unavailable"));
                #[cfg(feature = "rocm")]
                "ROCMExecutionProvider"
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
        if !inputs.is_empty() {
            validate_session(&session.inputs, inputs)?;
            if self.provider == Provider::Migraphx {
                for input in &session.inputs {
                    let ort::value::ValueType::Tensor { shape, .. } = &input.input_type else {
                        return Err(errors::config_error(
                            "execution_inputs",
                            "expected tensor input",
                        ));
                    };
                    if shape.iter().any(|size| *size <= 0) {
                        return Err(errors::config_error(
                            "execution_inputs",
                            "MIGraphX session did not resolve every execution dimension",
                        ));
                    }
                }
            }
        }
        for artifact in &artifacts {
            artifact.verify().map_err(|error| {
                errors::model_load(&path.display().to_string(), &error.to_string())
            })?;
        }
        if let Some(expected) = runtime_artifacts {
            let current = crate::core::migraphx_identity::runtime_artifacts()
                .map_err(|error| errors::config_error("compilation_cache", &error.to_string()))?;
            if current
                .iter()
                .map(|item| item.digest.clone())
                .collect::<Vec<_>>()
                != expected
            {
                return Err(errors::config_error(
                    "compilation_cache",
                    &format!("runtime libraries changed during session preparation: before={expected:?}; after={:?}", current.iter().map(|item| &item.digest).collect::<Vec<_>>()),
                ));
            }
        }
        self.evidence.lock().push(SessionEvidence {
            runtime_build: runtime_build_info(),
            graph: path.display().to_string(),
            provider: provider_name,
            device_id: self.device_id,
            precision: self.precision,
            cpu_fallback_disabled: self.provider == Provider::Migraphx
                || (self.provider == Provider::Rocm && !self.allow_cpu_fallback),
            custom_ops_profile: self.custom_ops_profile,
            custom_ops_library: custom_ops_library.map(|path| path.display().to_string()),
            custom_ops_sha256,
            profile_prefix,
            artifacts: artifacts.iter().map(|item| item.digest.clone()).collect(),
            execution_max_input_tokens: self.execution_max_input_tokens,
            execution_inputs: if self.provider == Provider::Migraphx || fixed_shapes {
                inputs.to_vec()
            } else {
                Vec::new()
            },
            input_schema: session_input_schema(&session.inputs),
            compilation_cache: cache_lease.as_ref().map(|lease| lease.evidence.clone()),
            compiler_flags,
        });
        Ok(PreparedSession {
            session,
            cache_lease,
            artifacts,
        })
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
    cache_dir: Option<&Path>,
) -> UnifiedResult<()> {
    use ort::AsPointer;
    use std::ffi::CString;
    let build_info = runtime_build_info();
    let cache_name = cache_dir
        .map(|path| {
            CString::new(path.as_os_str().as_encoded_bytes())
                .map_err(|_| errors::config_error("compilation_cache_dir", "path contains NUL"))
        })
        .transpose()?;
    if build_info.contains("git-commit-id=2716b9b93a,") {
        let options = Rocm7MigraphxOptions {
            device_id,
            fp16: fp16.into(),
            bf16: 0,
            fp8: 0,
            int8: 0,
            native_calibration: 0,
            calibration_table: std::ptr::null(),
            cache_dir: cache_name
                .as_ref()
                .map_or(std::ptr::null(), |name| name.as_ptr()),
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
    if cache_dir.is_some() {
        return Err(errors::config_error(
            "compilation_cache",
            "this runtime has no verified per-instance compiled-program cache ABI",
        ));
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
    fn rocm_allocation_policy_participates_in_execution_evidence() {
        let options = InstanceOptions {
            provider: Provider::Rocm,
            ..Default::default()
        };
        let flags = options.compiler_flags([]).unwrap();
        assert_eq!(flags["arena_extend_strategy"], "kSameAsRequested");
        assert_eq!(flags["memory_limit"], "usize_max");
        assert!(InstanceOptions::default()
            .compiler_flags([])
            .unwrap()
            .is_empty());
    }

    #[test]
    fn removed_classifier_session_bank_option_is_rejected() {
        let error = serde_json::from_str::<InstanceOptions>(
            r#"{"model_path":"unused","short_sequence_tokens":512}"#,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("unknown field `short_sequence_tokens`"));
    }

    #[test]
    fn compiler_identity_is_frozen_without_requiring_a_cache() {
        let options = InstanceOptions {
            provider: Provider::Migraphx,
            intra_threads: Some(8),
            ..Default::default()
        };
        assert!(options.compilation_cache_dir.is_none());
        let baseline = options.compiler_flags([]).unwrap();
        let mut environment = vec![(
            OsString::from("MIGRAPHX_SET_GEMM_PROVIDER"),
            OsString::from("rocblas"),
        )];
        let frozen = options.compiler_flags(environment.clone()).unwrap();
        assert_ne!(baseline, frozen);
        environment[0].1 = "other".into();
        assert_eq!(frozen["environment:MIGRAPHX_SET_GEMM_PROVIDER"], "rocblas");
        // The same descriptor is copied into the cache, not recaptured from env.
        let cached = InstanceOptions {
            compilation_cache_dir: Some("elsewhere".into()),
            ..options.clone()
        };
        assert_eq!(baseline, cached.compiler_flags([]).unwrap());
        assert_eq!(
            baseline,
            options
                .compiler_flags([("UNRELATED_SETTING".into(), "changed".into())])
                .unwrap()
        );
        assert!(InstanceOptions {
            provider: Provider::Cpu,
            ..options
        }
        .compiler_flags(environment)
        .unwrap()
        .is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_compiler_control_is_rejected_but_unrelated_values_are_ignored() {
        use std::os::unix::ffi::OsStringExt;
        let options = InstanceOptions {
            provider: Provider::Migraphx,
            ..Default::default()
        };
        let invalid = OsString::from_vec(vec![0xff]);
        assert!(options
            .compiler_flags([("MIGRAPHX_SET_GEMM_PROVIDER".into(), invalid.clone())])
            .is_err());
        assert_eq!(
            options.compiler_flags([]).unwrap(),
            options
                .compiler_flags([("UNRELATED_SETTING".into(), invalid)])
                .unwrap()
        );
    }

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
    fn document_budget_does_not_raise_physical_execution_capacity() {
        let mut options = InstanceOptions {
            model_path: "unused".into(),
            max_input_tokens: Some(32768),
            document_max_input_tokens: Some(262144),
            execution_max_input_tokens: Some(32768),
            ..Default::default()
        };
        assert_eq!(options.document_limit(32768, true).unwrap(), 262144);
        assert_eq!(options.effective_limit(32768).unwrap(), 32768);
        assert_eq!(options.execution_limit(32768).unwrap(), 32768);
        assert!(options.document_limit(32768, false).is_err());
        options.execution_max_input_tokens = Some(32769);
        assert!(options.execution_limit(32768).is_err());
        options.execution_max_input_tokens = None;
        options.document_max_input_tokens = Some(32767);
        assert!(options.document_limit(32768, true).is_err());
        options.document_max_input_tokens = Some(0);
        assert!(options.validate_configuration().is_err());
        options.document_max_input_tokens = None;
        assert_eq!(options.document_limit(32768, true).unwrap(), 32768);
        assert_eq!(options.document_limit(32768, false).unwrap(), 32768);
    }

    #[test]
    fn execution_capacity_and_cache_are_explicit() {
        let mut options = InstanceOptions {
            model_path: "unused".into(),
            max_input_tokens: Some(32768),
            execution_max_input_tokens: Some(2048),
            ..Default::default()
        };
        assert_eq!(options.effective_limit(32768).unwrap(), 32768);
        assert_eq!(options.execution_limit(32768).unwrap(), 2048);
        options.execution_max_input_tokens = Some(32769);
        assert!(options.execution_limit(32768).is_err());
        options.execution_max_input_tokens = Some(0);
        assert!(options.validate().is_err());
        options.execution_max_input_tokens = None;
        options.compilation_cache_dir = Some("cache".into());
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires MIGraphX"));
        assert!(options
            .create_session(Path::new("unused"))
            .unwrap_err()
            .to_string()
            .contains("resolved inputs"));
    }

    #[test]
    fn custom_ops_profiles_are_named_and_provider_specific() {
        assert!(serde_json::from_str::<InstanceOptions>(
            r#"{"model_path":"unused","custom_ops_profile":"/tmp/untrusted.so"}"#,
        )
        .is_err());
        for provider in [Provider::Cpu, Provider::Migraphx] {
            let options = InstanceOptions {
                model_path: "unused".into(),
                provider,
                custom_ops_profile: CustomOpsProfile::CkFlashAttention,
                ..Default::default()
            };
            assert!(options
                .validate()
                .unwrap_err()
                .to_string()
                .contains("requires the ROCm"));
        }
    }

    #[test]
    fn rocm_conversion_and_cpu_fallback_cannot_be_implicit() {
        let mut options = InstanceOptions {
            model_path: "unused".into(),
            provider: Provider::Rocm,
            precision: Precision::Fp16,
            ..Default::default()
        };
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("native types"));
        options.precision = Precision::Native;
        options.allow_cpu_fallback = true;
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("profile prefix"));
        options.profile_prefix = Some(String::new());
        assert!(options.validate().is_err());
        options.profile_prefix = Some("placement-proof".into());
        options.provider = Provider::Cpu;
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires ROCm"));
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn unavailable_rocm_never_falls_back_to_cpu() {
        let options = InstanceOptions {
            model_path: "unused".into(),
            provider: Provider::Rocm,
            ..Default::default()
        };
        assert!(options
            .validate()
            .unwrap_err()
            .to_string()
            .contains("CPU fallback is forbidden"));
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn rocm_profile_resolves_only_the_installed_library() {
        let options = InstanceOptions {
            model_path: "unused".into(),
            provider: Provider::Rocm,
            custom_ops_profile: CustomOpsProfile::CkFlashAttention,
            ..Default::default()
        };
        assert_eq!(
            options.custom_ops_library_path().unwrap(),
            Some(PathBuf::from(CK_FLASH_ATTENTION_LIBRARY))
        );
        assert!(!options.allow_cpu_fallback);
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
