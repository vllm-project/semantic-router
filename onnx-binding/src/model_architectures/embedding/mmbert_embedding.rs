//! mmBERT Embedding Model Implementation using ONNX Runtime (32K Context, 2D Matryoshka)
//!
//! This module implements the mmBERT-Embed-32K-2D-Matryoshka model using ONNX Runtime,
//! enabling AMD GPU (ROCm), NVIDIA GPU (CUDA), and CPU inference.
//!
//! ## Model Highlights
//! - **Parameters**: 307M
//! - **Context Length**: 32,768 tokens
//! - **Languages**: 1800+ (via Glot500)
//! - **Embedding Dim**: 768 (supports 64-768 via Matryoshka)
//! - **Architecture**: ModernBERT encoder with YaRN scaling
//!
//! ## 2D Matryoshka Support
//! This model supports two dimensions of flexibility:
//! 1. **Dimension Reduction** (Matryoshka): Truncate embeddings to smaller dimensions
//! 2. **Layer Reduction** (Adaptive): Use intermediate layer outputs for faster inference
//!
//! ## ONNX Runtime Benefits
//! - **AMD GPU Support**: Via ROCm execution provider
//! - **Cross-platform**: Works on Linux, Windows, macOS
//! - **Optimized inference**: Graph optimizations, operator fusion

use super::runtime_identity::{
    json_digest, tokenizer_digest, ArtifactDigest, ArtifactSnapshot, RuntimeIdentity,
};
use crate::core::instance_options::{
    CustomOpsProfile, InstanceOptions, Overflow, Provider, SessionEvidence,
};
use crate::core::onnx_artifacts::capture_onnx;
use crate::core::unified_error::{errors, UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::pooling::{
    l2_normalize, mean_pool_3d, truncate_dimension,
};
use crate::model_architectures::modernbert_inputs;
use half::f16;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

// ============================================================================
// Configuration
// ============================================================================

/// mmBERT Embedding model configuration
#[derive(Debug, Clone, serde::Serialize)]
pub struct MmBertEmbeddingConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    pub pad_token_id: u32,
}

impl Default for MmBertEmbeddingConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 1152,
            max_position_embeddings: 32768,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
        }
    }
}

impl MmBertEmbeddingConfig {
    /// Load configuration from a pretrained model directory
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let model_dir = model_path.as_ref();
        let config_candidates = [
            model_dir.join("config.json"),
            model_dir.join("onnx").join("config.json"),
        ];
        let config_path = config_candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .ok_or_else(|| {
                errors::file_not_found(&format!(
                    "config.json not found under {} (checked root and onnx/)",
                    model_dir.display()
                ))
            })?;

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| errors::file_not_found(&config_path.display().to_string()))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            errors::invalid_json(&config_path.display().to_string(), &e.to_string())
        })?;

        Ok(Self {
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(256000) as usize,
            hidden_size: config_json["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            intermediate_size: config_json["intermediate_size"].as_u64().unwrap_or(1152) as usize,
            max_position_embeddings: config_json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            layer_norm_eps: config_json["layer_norm_eps"].as_f64().unwrap_or(1e-5),
            pad_token_id: config_json["pad_token_id"].as_u64().unwrap_or(0) as u32,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
}

// ============================================================================
// Matryoshka Configuration
// ============================================================================

/// 2D Matryoshka dimensions configuration
#[derive(Debug, Clone)]
pub struct MatryoshkaConfig {
    pub dimensions: Vec<usize>,
    pub layers: Vec<usize>,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            dimensions: vec![768, 512, 256, 128, 64],
            layers: vec![3, 6, 11, 22],
        }
    }
}

impl MatryoshkaConfig {
    /// Build a config for a specific model directory, reading the early-exit
    /// layer list from the model's own `onnx/model_config.json`
    /// (`available_layers`) so the layers are a single source of truth rather
    /// than a hardcoded list that can drift from the shipped model.
    ///
    /// Falls back to the built-in default layers when the manifest is absent
    /// or does not declare `available_layers`.
    pub fn from_model_dir<P: AsRef<Path>>(model_path: P) -> Self {
        let model_dir = model_path.as_ref();
        let manifest_candidates = [
            model_dir.join("onnx").join("model_config.json"),
            model_dir.join("model_config.json"),
        ];

        let layers = manifest_candidates
            .iter()
            .find(|p| p.exists())
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|json| {
                json.get("available_layers")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|l| l.as_u64().map(|n| n as usize))
                            .collect::<Vec<usize>>()
                    })
            })
            .filter(|layers| !layers.is_empty());

        match layers {
            Some(layers) => Self {
                layers,
                ..Self::default()
            },
            None => Self::default(),
        }
    }

    pub fn validate_dimension(&self, dim: usize) -> bool {
        self.dimensions.contains(&dim)
    }

    pub fn validate_layer(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Estimate quality factor for a given layer/dimension combination
    /// Returns a value between 0 and 1, where 1 is best quality
    pub fn estimate_quality(&self, layer: usize, dim: usize) -> f32 {
        let layer_factor = match layer {
            22 => 1.0,
            11 => 0.67,
            6 => 0.56,
            3 => 0.55,
            _ => (layer as f32 / 22.0).max(0.5),
        };

        let dim_factor = match dim {
            768 => 1.0,
            512 => 0.995,
            256 => 0.99,
            128 => 0.985,
            64 => 0.98,
            _ => (dim as f32 / 768.0).max(0.9),
        };

        layer_factor * dim_factor
    }

    /// Estimate speedup factor for early layer exit
    pub fn estimate_speedup(&self, layer: usize) -> f32 {
        22.0 / layer as f32
    }
}

// ============================================================================
// Execution Provider Selection
// ============================================================================

/// Available execution providers for ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionProvider {
    /// CPU execution (always available)
    Cpu,
    /// AMD GPU via ROCm
    Rocm,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Intel acceleration via OpenVINO
    OpenVino,
    /// Windows GPU via DirectML
    DirectMl,
}

impl ExecutionProvider {
    /// Get the best available execution provider
    pub fn best_available() -> Self {
        #[cfg(feature = "rocm")]
        {
            return ExecutionProvider::Rocm;
        }

        #[cfg(feature = "cuda")]
        {
            return ExecutionProvider::Cuda;
        }

        #[cfg(feature = "directml")]
        {
            return ExecutionProvider::DirectMl;
        }

        #[cfg(feature = "openvino")]
        {
            return ExecutionProvider::OpenVino;
        }

        #[allow(unreachable_code)]
        ExecutionProvider::Cpu
    }
}

// ============================================================================
// mmBERT Embedding Model (ONNX Runtime)
// ============================================================================

struct LoadedSession {
    session: Session,
    cache_lease: Option<crate::core::compilation_cache::CompilationCacheLease>,
    artifacts: Vec<ArtifactDigest>,
    runtime: String,
}

/// mmBERT Embedding Model using ONNX Runtime
///
/// This model supports:
/// - AMD GPU via ROCm
/// - NVIDIA GPU via CUDA
/// - 2D Matryoshka (layer early exit + dimension truncation)
/// - 32K context length
/// - Multilingual (1800+ languages)
pub struct MmBertEmbeddingModel {
    /// ONNX Runtime session
    session: LoadedSession,
    identity: RuntimeIdentity,
    /// Tokenizer
    tokenizer: Arc<Tokenizer>,
    /// Model configuration
    config: MmBertEmbeddingConfig,
    /// Model-declared Matryoshka metadata retained for capability discovery.
    matryoshka_config: MatryoshkaConfig,
    /// Model path
    model_path: String,
    /// Layer represented by the primary graph; that graph is loaded only once.
    primary_layer: usize,
    /// Additional loaded exit graphs, indexed by their actual layer.
    layer_sessions: BTreeMap<usize, LoadedSession>,
    /// MIGraphX compiles a program for each shape. Fix its tensor shape to an
    /// explicit deployment budget without padding the tokenizer's real usage.
    execution_sequence_length: Option<usize>,
    /// Only an explicit owned truncate policy permits tokenizer overflow.
    reject_overflow: bool,
}

impl MmBertEmbeddingModel {
    /// Load the model from a directory containing ONNX model and tokenizer
    ///
    /// # Arguments
    /// * `model_path` - Path to directory containing model.onnx and tokenizer.json
    /// * `use_cpu` - If true, force CPU execution; otherwise use best available provider
    ///
    /// # Returns
    /// * `UnifiedResult<Self>` - The loaded model or an error
    pub fn load<P: AsRef<Path>>(model_path: P, use_cpu: bool) -> UnifiedResult<Self> {
        Self::load_impl(model_path, use_cpu, None)
    }

    pub fn load_with_options(options: &InstanceOptions) -> UnifiedResult<Self> {
        options.validate()?;
        Self::load_impl(
            &options.model_path,
            options.provider == Provider::Cpu,
            Some(options),
        )
    }

    fn load_impl<P: AsRef<Path>>(
        model_path: P,
        use_cpu: bool,
        options: Option<&InstanceOptions>,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();
        let model_dir = model_path.as_ref();

        // Load configuration
        let config = MmBertEmbeddingConfig::from_pretrained(&model_path)?;
        let execution_sequence_length = match options {
            Some(options) if options.provider == Provider::Migraphx => {
                if options.max_input_tokens.is_none() {
                    return Err(errors::config_error(
                        "max_input_tokens",
                        "owned MIGraphX embeddings require an explicit positive input token budget for their fixed execution shape",
                    ));
                }
                Some(options.execution_limit(config.max_position_embeddings)?)
            }
            _ => None,
        };

        // Resolve the early-exit layer list from the model's own manifest
        // (single source of truth) so it never drifts from what is shipped.
        let matryoshka_config = MatryoshkaConfig::from_model_dir(&model_path);

        // Load tokenizer
        let tokenizer_candidates = [
            model_dir.join("tokenizer.json"),
            model_dir.join("onnx").join("tokenizer.json"),
        ];
        let tokenizer_path = tokenizer_candidates
            .iter()
            .find(|p| p.exists())
            .cloned()
            .ok_or_else(|| {
                errors::file_not_found(&format!(
                    "tokenizer.json not found under {} (checked root and onnx/)",
                    model_dir.display()
                ))
            })?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        if let Some(options) = options {
            options.configure_tokenizer(&mut tokenizer, config.max_position_embeddings)?;
        } else {
            // Legacy artifact tokenizers may retain a short training limit.
            tokenizer
                .with_truncation(None)
                .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        }
        tokenizer.with_padding(None);

        // Owned selection follows the explicit execution profile, never legacy
        // environment variables. An explicitly named graph remains authoritative.
        let onnx_candidates = if options.is_some_and(|o| o.model_file.is_some()) {
            vec![]
        } else if let Some(options) = options {
            Self::owned_primary_candidates(&model_path, config.num_hidden_layers, options)?
        } else {
            Self::find_onnx_models(&model_path, config.num_hidden_layers, use_cpu)?
        };
        let onnx_candidates = match options {
            Some(options) => vec![options.select_graph(onnx_candidates)?],
            None => onnx_candidates,
        };

        // Create ONNX Runtime session with fallback across candidates.
        // We intentionally prefer GPU-optimized model variants first.
        let mut selected_session: Option<LoadedSession> = None;
        let mut selected_path: Option<std::path::PathBuf> = None;
        let mut last_error: Option<String> = None;
        for onnx_path in onnx_candidates {
            let loaded =
                Self::load_session(&onnx_path, use_cpu, options, config.max_position_embeddings);
            match loaded {
                Ok(session) => {
                    selected_path = Some(onnx_path);
                    selected_session = Some(session);
                    break;
                }
                Err(e) => {
                    let reason = format!("{:?}", e);
                    println!(
                        "WARN: Failed to initialize mmBERT session from {}: {}",
                        onnx_path.display(),
                        reason
                    );
                    last_error = Some(format!("{}: {}", onnx_path.display(), reason));
                }
            }
        }
        let session = match selected_session {
            Some(session) => session,
            None => {
                let detail =
                    last_error.unwrap_or_else(|| "no ONNX candidate was loadable".to_string());
                return Err(errors::model_load(&model_path_str, &detail));
            }
        };
        let selected_path = selected_path.expect("a loaded session has a selected graph");
        println!(
            "INFO: Selected mmBERT ONNX file: {}",
            selected_path.display()
        );

        // Check for layer-specific ONNX files (for early exit support)
        let (primary_layer, layer_sessions) = Self::load_layer_sessions(
            &model_path,
            use_cpu,
            &matryoshka_config.layers,
            options,
            &selected_path,
            config.num_hidden_layers,
            config.max_position_embeddings,
        )?;

        let identity = RuntimeIdentity {
            version: 1,
            model_type: "mmbert",
            runtime: session.runtime.clone(),
            effective_config_sha256: json_digest(&config)
                .map_err(|e| errors::model_load(&model_path_str, &e.to_string()))?,
            tokenizer_sha256: tokenizer_digest(&tokenizer)
                .map_err(|e| errors::model_load(&model_path_str, &e.to_string()))?,
            artifacts: session.artifacts.clone(),
            layer: config.num_hidden_layers,
            dimension: config.hidden_size,
            max_sequence_length: config.max_position_embeddings,
            pooling_contract: "onnx-raw-hidden:attention-mask-mean-f32:truncate-before-l2:v1",
        };
        Ok(Self {
            session,
            identity,
            tokenizer: Arc::new(tokenizer),
            config,
            matryoshka_config,
            model_path: model_path_str,
            primary_layer,
            layer_sessions,
            execution_sequence_length,
            reject_overflow: options.is_none_or(|o| o.overflow == Overflow::Reject),
        })
    }

    const GRAPH_VARIANTS: [&'static str; 3] = ["model", "model_fa", "model_fa_fp16"];

    fn parse_graph_layer(value: &str) -> UnifiedResult<usize> {
        if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) {
            if let Ok(layer) = value.parse::<usize>() {
                if layer > 0 {
                    return Ok(layer);
                }
            }
        }
        Err(errors::config_error(
            "primary_layer",
            "selected graph has an invalid layer name",
        ))
    }

    /// Recognize maintained filenames without merging graph precision variants.
    /// Unknown custom names retain exact-name companion lookup.
    fn graph_filename(
        name: &std::ffi::OsStr,
    ) -> UnifiedResult<Option<(&'static str, Option<usize>)>> {
        let Some(stem) = name.to_str().and_then(|name| name.strip_suffix(".onnx")) else {
            return Ok(None);
        };
        for variant in Self::GRAPH_VARIANTS {
            if stem == variant {
                return Ok(Some((variant, None)));
            }
            if let Some(layer) = stem.strip_prefix(&format!("{variant}_layer_")) {
                return Ok(Some((variant, Some(Self::parse_graph_layer(layer)?))));
            }
        }
        Ok(None)
    }

    fn owned_primary_candidates<P: AsRef<Path>>(
        model_path: P,
        full_layer: usize,
        options: &InstanceOptions,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let ck = options.provider == Provider::Rocm
            && options.custom_ops_profile == CustomOpsProfile::CkFlashAttention;
        let names: &[&str] = if ck {
            &["model_fa_fp16.onnx", "model_fa.onnx"]
        } else {
            &[
                "model.onnx",
                "encoder.onnx",
                "mmbert.onnx",
                "model_optimized.onnx",
                "model_sdpa_fp16.onnx",
            ]
        };
        let root = model_path.as_ref();
        let directories = [
            root.to_path_buf(),
            root.join("onnx"),
            root.join(format!("onnx/layer-{full_layer}")),
        ];
        let mut paths: Vec<_> = directories
            .iter()
            .flat_map(|directory| names.iter().map(|name| directory.join(name)))
            .filter(|path| path.is_file())
            .collect();
        for name in names {
            if let Some((variant, _)) = Self::graph_filename(std::ffi::OsStr::new(name))? {
                let flat = format!("{variant}_layer_{full_layer}.onnx");
                paths.extend(
                    [root.join(&flat), root.join("onnx").join(flat)]
                        .into_iter()
                        .filter(|path| path.is_file()),
                );
            }
        }
        if paths.is_empty() {
            return Err(errors::model_load(
                &root.display().to_string(),
                "no full-layer graph matches the owned execution profile; specify model_file for a custom graph",
            ));
        }
        Ok(paths)
    }

    fn owned_layer_candidates(
        root: &Path,
        layer: usize,
        primary_path: &Path,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let name = primary_path.file_name().ok_or_else(|| {
            errors::config_error("model_file", "the selected graph has no filename")
        })?;
        let variant = Self::graph_filename(name)?.map(|(variant, _)| variant);
        let nested_name = variant
            .map(|variant| std::ffi::OsString::from(format!("{variant}.onnx")))
            .unwrap_or_else(|| name.to_owned());
        let mut paths = vec![root.join(format!("onnx/layer-{layer}")).join(nested_name)];
        // Nested graphs retain priority. Flat companions must use the exact
        // selected variant, including whole-encoder vs attention-only precision.
        if let Some(variant) = variant {
            let flat = format!("{variant}_layer_{layer}.onnx");
            paths.extend([root.join(&flat), root.join("onnx").join(flat)]);
        }
        let flat_alternatives = Self::GRAPH_VARIANTS.into_iter().flat_map(|variant| {
            let flat = format!("{variant}_layer_{layer}.onnx");
            [root.join(&flat), root.join("onnx").join(flat)]
        });
        if !paths.iter().any(|path| path.is_file())
            && Self::layer_candidates(root, layer, false, true)
                .into_iter()
                .chain(flat_alternatives)
                .any(|path| path.is_file())
        {
            return Err(errors::config_error(
                "layer_graph",
                &format!(
                    "layer {layer} has graphs but none match selected primary {}",
                    name.to_string_lossy()
                ),
            ));
        }
        Ok(paths)
    }

    /// Find ONNX model candidates in priority order.
    ///
    /// Searches: model_path/, model_path/onnx/, and HuggingFace-style
    /// model_path/onnx/layer-{N}/ subdirectories (highest layer first as primary).
    fn find_onnx_models<P: AsRef<Path>>(
        model_path: P,
        full_layer: usize,
        use_cpu: bool,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let dir = model_path.as_ref();
        let onnx_subdir = dir.join("onnx");

        let has_fa = std::env::var("ORT_CK_FLASH_ATTN_LIB")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some();
        // See the note in mmbert_classifier.rs: an FP16 graph is the slowest
        // candidate on the CPU provider, so it goes last there rather than first.
        let candidates: &[&str] = if use_cpu {
            &[
                "model.onnx",
                "encoder.onnx",
                "mmbert.onnx",
                "model_optimized.onnx",
                "model_sdpa_fp16.onnx",
            ]
        } else if has_fa {
            &[
                "model_fa_fp16.onnx",
                "model_fa.onnx",
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "encoder.onnx",
                "mmbert.onnx",
                "model_optimized.onnx",
            ]
        } else {
            &[
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "encoder.onnx",
                "mmbert.onnx",
                "model_optimized.onnx",
            ]
        };

        let mut search_dirs: Vec<std::path::PathBuf> = vec![dir.to_path_buf(), onnx_subdir.clone()];

        // The primary session must implement the complete encoder. An absent
        // full graph must not silently substitute a shallower embedding space.
        let full_layer_dir = onnx_subdir.join(format!("layer-{full_layer}"));
        if full_layer_dir.is_dir() {
            search_dirs.push(full_layer_dir);
        }

        let mut results: Vec<std::path::PathBuf> = Vec::new();
        for base_dir in &search_dirs {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            for candidate in candidates {
                let path = base_dir.join(candidate);
                if path.exists() && !results.iter().any(|p| p == &path) {
                    results.push(path);
                }
            }
        }

        // Also include any other .onnx files as a final fallback set.
        for base_dir in &[dir.to_path_buf(), onnx_subdir] {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if path.extension().is_some_and(|ext| ext == "onnx")
                        && (!name.starts_with("model_layer_")
                            || name == format!("model_layer_{full_layer}.onnx"))
                        && (!name.contains("_fa") || !use_cpu && has_fa)
                        && !results.iter().any(|p| p == &path)
                    {
                        results.push(path);
                    }
                }
            }
        }

        if results.is_empty() {
            return Err(errors::file_not_found(&format!(
                "No ONNX model found in {} (checked root, onnx/, and onnx/layer-*/)",
                dir.display()
            )));
        }
        Ok(results)
    }

    /// Bind graph bytes and execution semantics to the session that loaded them.
    fn load_session(
        path: &Path,
        use_cpu: bool,
        options: Option<&InstanceOptions>,
        task_limit: usize,
    ) -> UnifiedResult<LoadedSession> {
        let Some(options) = options else {
            return Self::create_session(path, use_cpu);
        };
        let error =
            |e: anyhow::Error| errors::model_load(&path.display().to_string(), &e.to_string());
        let prepared = modernbert_inputs::prepare_session(
            options,
            path,
            options.execution_limit(task_limit)?,
        )?;
        let session = prepared.session;
        let snapshots = prepared.artifacts;
        modernbert_inputs::validate(&session.inputs)?;
        let evidence = options.evidence.lock();
        let actual = evidence.last().ok_or_else(|| {
            errors::model_load(
                &path.display().to_string(),
                "owned session did not record execution evidence",
            )
        })?;
        let runtime = Self::owned_runtime_identity(actual, options.intra_threads).map_err(error)?;
        drop(evidence);
        for snapshot in &snapshots {
            snapshot.verify().map_err(error)?;
        }
        Ok(LoadedSession {
            session,
            cache_lease: prepared.cache_lease,
            runtime: format!("onnx-mmbert-owned-v1:{runtime}"),
            artifacts: snapshots.into_iter().map(|s| s.digest).collect(),
        })
    }

    fn owned_runtime_identity(
        actual: &SessionEvidence,
        intra_threads: Option<usize>,
    ) -> anyhow::Result<String> {
        json_digest(&serde_json::json!({
            "contract": "onnx-mmbert-owned-v1",
            "runtime_build": actual.runtime_build,
            "compiler_flags": actual.compiler_flags,
            "provider": actual.provider,
            "device_id": actual.device_id,
            "precision": actual.precision,
            "cpu_fallback_disabled": actual.cpu_fallback_disabled,
            "custom_ops_profile": actual.custom_ops_profile,
            "custom_ops_sha256": actual.custom_ops_sha256,
            "intra_threads": intra_threads,
            "execution_inputs": actual.execution_inputs,
            "execution_max_input_tokens": actual.execution_max_input_tokens,
        }))
    }

    /// Create an ONNX Runtime session with the legacy execution-provider policy.
    fn create_session<P: AsRef<Path>>(onnx_path: P, use_cpu: bool) -> UnifiedResult<LoadedSession> {
        let path = onnx_path.as_ref();
        let error =
            |e: anyhow::Error| errors::model_load(&path.display().to_string(), &e.to_string());
        let mut snapshots = capture_onnx(path).map_err(error)?;
        // This library is registered only on the ROCm path; include it only if
        // that provider actually succeeds, not merely because it was requested.
        let custom = if !use_cpu && cfg!(any(feature = "rocm", feature = "migraphx")) {
            std::env::var("ORT_CK_FLASH_ATTN_LIB")
                .ok()
                .filter(|p| !p.is_empty())
                .map(|p| ArtifactSnapshot::capture(Path::new(&p), "custom-operators"))
        } else {
            None
        };
        let (session, runtime) = Self::create_session_inner(path, use_cpu)?;
        modernbert_inputs::validate(&session.inputs)?;
        if runtime == "onnx-mmbert-rocm-v1" {
            if let Some(custom) = custom {
                snapshots.push(custom.map_err(error)?);
            }
        }
        for snapshot in &snapshots {
            snapshot.verify().map_err(error)?;
        }
        Ok(LoadedSession {
            session,
            cache_lease: None,
            runtime: runtime.to_string(),
            artifacts: snapshots.into_iter().map(|s| s.digest).collect(),
        })
    }

    fn create_session_inner<P: AsRef<Path>>(
        onnx_path: P,
        use_cpu: bool,
    ) -> UnifiedResult<(Session, &'static str)> {
        let onnx_path_str = onnx_path.as_ref().display().to_string();

        // Build session with execution providers
        let session = if use_cpu {
            Session::builder()
                .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                .commit_from_file(onnx_path.as_ref())
                .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))?
        } else {
            #[cfg(any(feature = "rocm", feature = "migraphx"))]
            {
                use crate::core::gpu_memory;
                use ort::execution_providers::{ArenaExtendStrategy, ROCmExecutionProvider};
                let mem_limit = gpu_memory::get_gpu_mem_limit();
                let ck_fa_lib = std::env::var("ORT_CK_FLASH_ATTN_LIB")
                    .ok()
                    .filter(|s| !s.is_empty());
                if let Some(ref lib) = ck_fa_lib {
                    println!("INFO: CK Flash Attention custom op library: {}", lib);
                }
                let maybe_register_custom_ops = |builder: ort::session::builder::SessionBuilder| -> Result<
                    ort::session::builder::SessionBuilder,
                    ort::Error,
                > {
                    if let Some(ref lib) = ck_fa_lib {
                        builder.with_operator_library(lib)
                    } else {
                        Ok(builder)
                    }
                };
                println!("INFO: Attempting ROCm execution provider...");
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([ROCmExecutionProvider::default()
                        .with_mem_limit(mem_limit)
                        .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
                        .build()
                        .error_on_failure()])
                    .and_then(maybe_register_custom_ops)
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!("INFO: Using ROCm execution provider (AMD GPU) — verified");
                        return Ok((session, "onnx-mmbert-rocm-v1"));
                    }
                    Err(e) => println!("WARN: ROCm EP failed: {}", e),
                }
            }

            #[cfg(feature = "migraphx")]
            {
                use ort::execution_providers::MIGraphXExecutionProvider;
                println!("INFO: Attempting MIGraphX execution provider...");
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([MIGraphXExecutionProvider::default()
                        .with_fp16(true)
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!("INFO: Using MIGraphX execution provider (AMD GPU) — verified");
                        return Ok((session, "onnx-mmbert-migraphx-fp16-v1"));
                    }
                    Err(e) => println!("WARN: MIGraphX EP failed: {}", e),
                }
            }

            #[cfg(feature = "cuda")]
            {
                use crate::core::gpu_memory;
                use ort::execution_providers::{
                    ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                };
                // Bound the per-session CUDA arena and request-sized arena
                // growth, mirroring the classifier path. The 2D-Matryoshka
                // embedding model opens one primary session plus one session
                // per early-exit layer declared in the model manifest, so
                // without a memory
                // limit each unbounded BFC arena tries to grab a large block
                // up front and later sessions OOM (CUBLAS_STATUS_ALLOC_FAILED)
                // on a shared/busy GPU, silently falling back to CPU.
                let mem_limit = gpu_memory::get_gpu_mem_limit();
                match Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([CUDAExecutionProvider::default()
                        .with_memory_limit(mem_limit)
                        .with_arena_extend_strategy(CudaArenaStrategy::SameAsRequested)
                        .build()
                        .error_on_failure()])
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!("INFO: Using CUDA execution provider (NVIDIA GPU) — verified");
                        return Ok((session, "onnx-mmbert-cuda-v1"));
                    }
                    Err(e) => println!("WARN: CUDA EP failed: {}", e),
                }
            }

            // Fallback to CPU
            println!("INFO: Using CPU execution provider");
            Session::builder()
                .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                .commit_from_file(onnx_path.as_ref())
                .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))?
        };

        Ok((session, "onnx-mmbert-cpu-v1"))
    }

    /// Resolve the selected graph's layer once from the artifact contract.
    fn primary_graph_layer(
        model_dir: &Path,
        selected_path: &Path,
        canonical_path: &Path,
        layers: &[usize],
        full_depth: usize,
    ) -> UnifiedResult<usize> {
        let mut selected_layer = None;
        let canonical_root = std::fs::canonicalize(model_dir)
            .map_err(|_| errors::file_not_found(&model_dir.display().to_string()))?;
        // Layer membership belongs to the artifact layout and manifest, not to
        // one preferred graph filename. Check symlink targets as well so an
        // alias cannot advertise a different layer from the graph it selects.
        for (path, root) in [
            (selected_path, model_dir),
            (canonical_path, canonical_root.as_path()),
        ] {
            let Ok(relative) = path.strip_prefix(root) else {
                continue;
            };
            let parent = relative.parent().unwrap_or(Path::new(""));
            let directory_layer = if parent.parent() == Some(Path::new("onnx")) {
                parent
                    .file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.strip_prefix("layer-"))
            } else {
                None
            };
            let filename_layer = if parent == Path::new("")
                || parent == Path::new("onnx")
                || directory_layer.is_some()
            {
                Self::graph_filename(relative.file_name().unwrap_or_default())?
                    .and_then(|(_, layer)| layer)
            } else {
                None
            };
            let directory_layer = directory_layer.map(Self::parse_graph_layer).transpose()?;
            for layer in [directory_layer, filename_layer].into_iter().flatten() {
                if !layers.contains(&layer) {
                    return Err(errors::config_error(
                        "primary_layer",
                        "selected graph's layer is absent from the artifact's available_layers",
                    ));
                }
                if selected_layer.is_some_and(|previous| previous != layer) {
                    return Err(errors::config_error(
                        "primary_layer",
                        "selected graph has conflicting layer declarations",
                    ));
                }
                selected_layer = Some(layer);
            }
        }
        Ok(selected_layer.unwrap_or(full_depth))
    }

    /// Load layer-specific ONNX sessions for early exit support.
    ///
    /// Searches the legacy flat model_layer_{N}.onnx layouts and
    /// HuggingFace-style onnx/layer-{N}/ graph directories.
    fn load_layer_sessions<P: AsRef<Path>>(
        model_path: P,
        use_cpu: bool,
        layers: &[usize],
        options: Option<&InstanceOptions>,
        primary_path: &Path,
        full_depth: usize,
        task_limit: usize,
    ) -> UnifiedResult<(usize, BTreeMap<usize, LoadedSession>)> {
        let mut sessions = BTreeMap::new();
        let canonical_primary = std::fs::canonicalize(primary_path)
            .map_err(|_| errors::file_not_found(&primary_path.display().to_string()))?;
        let model_dir = model_path.as_ref();
        let primary_layer = Self::primary_graph_layer(
            model_dir,
            primary_path,
            &canonical_primary,
            layers,
            full_depth,
        )?;
        let legacy_fa = options.is_none()
            && !use_cpu
            && cfg!(any(feature = "rocm", feature = "migraphx"))
            && std::env::var("ORT_CK_FLASH_ATTN_LIB")
                .ok()
                .is_some_and(|s| !s.is_empty());
        for &layer in layers {
            // The selected graph is the sole session for its declared layer.
            if layer == primary_layer {
                continue;
            }
            let candidates = if options.is_some() {
                Self::owned_layer_candidates(model_dir, layer, primary_path)?
            } else {
                Self::layer_candidates(model_dir, layer, use_cpu, legacy_fa)
            };
            for path in candidates {
                if !path.is_file() {
                    continue;
                }
                let canonical_path = std::fs::canonicalize(&path)
                    .map_err(|_| errors::file_not_found(&path.display().to_string()))?;
                if canonical_path == canonical_primary {
                    return Err(errors::config_error(
                        "primary_layer",
                        "another declared layer aliases the selected primary graph",
                    ));
                }
                if options.is_some()
                    && Self::primary_graph_layer(model_dir, &path, &canonical_path, layers, layer)?
                        != layer
                {
                    return Err(errors::config_error(
                        "layer_graph",
                        "companion graph has a conflicting layer declaration",
                    ));
                }
                let loaded = Self::load_session(&path, use_cpu, options, task_limit);
                match loaded {
                    Ok(session) => {
                        sessions.insert(layer, session);
                        break;
                    }
                    Err(error) => {
                        if options.is_some() {
                            return Err(error);
                        }
                        println!(
                            "WARN: Layer-{layer} candidate {} failed: {error:?}",
                            path.display()
                        );
                    }
                }
            }
        }

        Ok((primary_layer, sessions))
    }

    fn layer_candidates(
        dir: &Path,
        layer: usize,
        use_cpu: bool,
        has_fa: bool,
    ) -> Vec<std::path::PathBuf> {
        let filename = format!("model_layer_{layer}.onnx");
        let nested = dir.join("onnx").join(format!("layer-{layer}"));
        let mut paths = vec![dir.join(&filename), dir.join("onnx").join(filename)];
        if !use_cpu && has_fa {
            paths.push(nested.join("model_fa_fp16.onnx"));
            paths.push(nested.join("model_fa.onnx"));
        }
        if !use_cpu {
            paths.push(nested.join("model_sdpa_fp16.onnx"));
        }
        paths.push(nested.join("model.onnx"));
        paths
    }

    /// Get the model configuration
    pub fn config(&self) -> &MmBertEmbeddingConfig {
        &self.config
    }

    /// Declared widths used by this model's Matryoshka implementation.
    pub fn matryoshka_dimensions(&self) -> Vec<usize> {
        self.matryoshka_config.dimensions.clone()
    }

    /// Get the tokenizer
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.config.num_hidden_layers
    }

    /// Check if layer early exit is supported
    pub fn supports_layer_exit(&self) -> bool {
        self.primary_layer != self.config.num_hidden_layers || !self.layer_sessions.is_empty()
    }

    /// Get available early exit layers
    pub fn available_exit_layers(&self) -> Vec<usize> {
        let mut layers: Vec<_> = self.layer_sessions.keys().copied().collect();
        layers.push(self.primary_layer);
        layers.sort_unstable();
        layers.dedup();
        layers
    }

    /// Generate embeddings with 2D Matryoshka support
    ///
    /// # Arguments
    /// * `texts` - Input texts to embed
    /// * `target_layer` - Target layer for early exit (None = full model)
    /// * `target_dim` - Target dimension for truncation (None = full dimension)
    ///
    /// # Returns
    /// * `UnifiedResult<Array2<f32>>` - [batch_size, target_dim] embeddings
    pub fn encode(
        &mut self,
        texts: &[&str],
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Array2<f32>> {
        if texts.is_empty() {
            return Err(UnifiedError::Validation {
                field: "texts".to_string(),
                expected: "non-empty".to_string(),
                actual: "empty".to_string(),
            });
        }

        if let Some(layer) = target_layer {
            if !self.available_exit_layers().contains(&layer) {
                return Err(errors::inference_error(
                    "target_layer",
                    &format!("layer {layer} is not loaded"),
                ));
            }
        }
        if let Some(dim) = target_dim {
            if dim == 0 || dim > self.config.hidden_size {
                return Err(errors::inference_error(
                    "target_dim",
                    &format!("unsupported dimension {dim}"),
                ));
            }
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        if encodings.iter().any(|e| {
            e.len() > self.config.max_position_embeddings
                || (self.reject_overflow && !e.get_overflowing().is_empty())
        }) {
            return Err(errors::tokenization_error(
                "input exceeds the embedding model context window",
            ));
        }
        // Find max sequence length
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let max_len = max_len.min(self.config.max_position_embeddings);
        let max_len = self.execution_sequence_length.unwrap_or(max_len);

        // Prepare input tensors
        let batch_size = texts.len();
        let mut input_ids = vec![self.config.pad_token_id as i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let seq_len = encoding.len().min(max_len);
            for j in 0..seq_len {
                input_ids[i * max_len + j] = encoding.get_ids()[j] as i64;
                attention_mask[i * max_len + j] = encoding.get_attention_mask()[j] as i64;
            }
        }

        // Create ndarray tensors
        let input_ids_array = Array2::from_shape_vec((batch_size, max_len), input_ids.clone())
            .map_err(|e| errors::inference_error("create_input_ids", &e.to_string()))?;

        let attention_mask_array =
            Array2::from_shape_vec((batch_size, max_len), attention_mask.clone())
                .map_err(|e| errors::inference_error("create_attention_mask", &e.to_string()))?;

        // Run inference - inline session selection to avoid borrow checker issues
        let embeddings =
            self.run_inference_with_layer(target_layer, &input_ids_array, &attention_mask_array)?;

        // Apply dimension truncation if requested
        let embeddings = if let Some(dim) = target_dim {
            if dim < embeddings.shape()[1] {
                truncate_dimension(&embeddings, dim)
            } else {
                embeddings
            }
        } else {
            embeddings
        };

        // L2 normalize
        let normalized = l2_normalize(&embeddings);

        Ok(normalized)
    }

    /// Describe the same loaded graph selected by an actual embedding call.
    pub(crate) fn runtime_descriptor(
        &self,
        layer: usize,
        dimension: usize,
    ) -> anyhow::Result<RuntimeIdentity> {
        let effective_layer = if layer == 0 {
            self.primary_layer
        } else {
            layer
        };
        let effective_dim = if dimension == 0 {
            self.config.hidden_size
        } else {
            dimension
        };
        anyhow::ensure!(
            self.available_exit_layers().contains(&effective_layer),
            "mmbert layer is not loaded"
        );
        anyhow::ensure!(
            effective_dim > 0 && effective_dim <= self.config.hidden_size,
            "unsupported mmbert dimension"
        );
        let session = if effective_layer == self.primary_layer {
            &self.session
        } else {
            self.layer_sessions
                .get(&effective_layer)
                .ok_or_else(|| anyhow::anyhow!("mmbert layer is not loaded"))?
        };
        let mut result = self.identity.for_exit(effective_layer, effective_dim);
        result.artifacts = session.artifacts.clone();
        result.runtime = session.runtime.clone();
        Ok(result)
    }

    /// Run inference on the ONNX model with optional layer selection
    fn run_inference_with_layer(
        &mut self,
        target_layer: Option<usize>,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> UnifiedResult<Array2<f32>> {
        let loaded = if target_layer.is_none_or(|layer| layer == self.primary_layer) {
            &mut self.session
        } else {
            self.layer_sessions
                .get_mut(&target_layer.unwrap())
                .ok_or_else(|| {
                    errors::config_error("target_layer", "requested layer has no loaded session")
                })?
        };
        let session = &mut loaded.session;
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        let input_ids_flat: Vec<i64> = input_ids.iter().copied().collect();
        let attention_mask_flat: Vec<i64> = attention_mask.iter().copied().collect();
        let outputs = modernbert_inputs::run_cached(
            session,
            &mut loaded.cache_lease,
            input_ids_flat,
            attention_mask_flat,
            batch_size,
            seq_len,
        )?;

        // Extract output
        // ONNX models can have different output formats:
        // 1. Direct pooled output [batch, hidden_dim]
        // 2. Sequence output [batch, seq_len, hidden_dim] - needs pooling
        let output_names = [
            "last_hidden_state",
            "sentence_embedding",
            "pooler_output",
            "embeddings",
        ];

        // Extract tensor data as f32, with f16 fallback for FA FP16 models.
        macro_rules! try_extract_f32 {
            ($val:expr) => {{
                let mut result: Option<(Vec<usize>, Vec<f32>)> = None;
                if let Ok((shape, data)) = $val.try_extract_tensor::<f32>() {
                    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                    result = Some((dims, data.to_vec()));
                } else if let Ok((shape, data)) = $val.try_extract_tensor::<f16>() {
                    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                    let f32_data: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                    result = Some((dims, f32_data));
                }
                result
            }};
        }

        let process_output =
            |dims: &[usize], flat: Vec<f32>, mask: &Array2<i64>| -> UnifiedResult<Array2<f32>> {
                if dims.len() == 2 {
                    Array2::from_shape_vec((dims[0], dims[1]), flat)
                        .map_err(|e| errors::inference_error("reshape_output", &e.to_string()))
                } else if dims.len() == 3 {
                    let (b, s, h) = (dims[0], dims[1], dims[2]);
                    let hidden = Array3::from_shape_vec((b, s, h), flat).map_err(|e| {
                        errors::inference_error("reshape_hidden_states", &e.to_string())
                    })?;
                    let mask_f32: Array2<f32> = mask.mapv(|x| x as f32);
                    Ok(mean_pool_3d(&hidden, &mask_f32))
                } else {
                    Err(errors::inference_error(
                        "extract_output",
                        &format!("Unexpected tensor rank: {}", dims.len()),
                    ))
                }
            };

        for name in &output_names {
            if let Some(output_value) = outputs.get(*name) {
                if let Some((dims, flat)) = try_extract_f32!(output_value) {
                    return process_output(&dims, flat, attention_mask);
                }
            }
        }

        // Try first output if named outputs not found
        if let Some((_, output_value)) = outputs.iter().next() {
            if let Some((dims, flat)) = try_extract_f32!(output_value) {
                return process_output(&dims, flat, attention_mask);
            }
        }

        Err(errors::inference_error(
            "extract_output",
            "Failed to extract output tensor (tried f32 and f16)",
        ))
    }

    /// Encode a single text
    pub fn encode_single(
        &mut self,
        text: &str,
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Array1<f32>> {
        let embeddings = self.encode(&[text], target_layer, target_dim)?;
        Ok(embeddings.row(0).to_owned())
    }

    pub fn finish_profiling(&mut self) -> UnifiedResult<Vec<String>> {
        let mut paths = vec![self
            .session
            .session
            .end_profiling()
            .map_err(|e| errors::ort_error(&e.to_string()))?];
        for session in self.layer_sessions.values_mut() {
            paths.push(
                session
                    .session
                    .end_profiling()
                    .map_err(|e| errors::ort_error(&e.to_string()))?,
            );
        }
        Ok(paths)
    }

    /// Get model information for debugging
    pub fn model_info(&self) -> String {
        format!(
            "MmBertEmbeddingModel(path={}, hidden_size={}, layers={}, layer_exit={}, available_exits={:?})",
            self.model_path,
            self.config.hidden_size,
            self.config.num_hidden_layers,
            self.supports_layer_exit(),
            self.available_exit_layers()
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncached_execution_identity_includes_frozen_compiler_controls() {
        let options = InstanceOptions {
            provider: Provider::Migraphx,
            ..Default::default()
        };
        let mut evidence = SessionEvidence {
            runtime_build: "fixed-runtime".into(),
            graph: "model.onnx".into(),
            provider: "MIGraphXExecutionProvider",
            device_id: 0,
            precision: crate::core::instance_options::Precision::Native,
            cpu_fallback_disabled: true,
            custom_ops_profile: CustomOpsProfile::None,
            custom_ops_library: None,
            custom_ops_sha256: None,
            profile_prefix: None,
            artifacts: vec![],
            execution_max_input_tokens: Some(32768),
            execution_inputs: vec![],
            compilation_cache: None,
            compiler_flags: options.compiler_flags([]).unwrap(),
        };
        let baseline = MmBertEmbeddingModel::owned_runtime_identity(&evidence, None).unwrap();
        evidence.compiler_flags = options
            .compiler_flags([("MIGRAPHX_SET_GEMM_PROVIDER".into(), "rocblas".into())])
            .unwrap();
        let rocblas = MmBertEmbeddingModel::owned_runtime_identity(&evidence, None).unwrap();
        assert_ne!(baseline, rocblas);
        evidence.graph = "relocated/model.onnx".into();
        evidence.profile_prefix = Some("observation-only".into());
        assert_eq!(
            rocblas,
            MmBertEmbeddingModel::owned_runtime_identity(&evidence, None).unwrap()
        );
        evidence.compiler_flags = options
            .compiler_flags([("UNRELATED_SETTING".into(), "1".into())])
            .unwrap();
        assert_eq!(
            baseline,
            MmBertEmbeddingModel::owned_runtime_identity(&evidence, None).unwrap()
        );
    }

    #[test]
    fn owned_primary_selection_uses_typed_profile() {
        let root = tempfile::tempdir().unwrap();
        let full = root.path().join("onnx/layer-22");
        std::fs::create_dir_all(&full).unwrap();
        for name in ["model.onnx", "model_fa_fp16.onnx", "model_fa.onnx"] {
            std::fs::write(full.join(name), []).unwrap();
        }
        for provider in [Provider::Cpu, Provider::Migraphx, Provider::Rocm] {
            let options = InstanceOptions {
                provider,
                ..Default::default()
            };
            let paths =
                MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap();
            assert_eq!(paths, vec![full.join("model.onnx")]);
        }
        let options = InstanceOptions {
            provider: Provider::Rocm,
            custom_ops_profile: CustomOpsProfile::CkFlashAttention,
            ..Default::default()
        };
        let paths =
            MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap();
        assert_eq!(
            paths,
            vec![full.join("model_fa_fp16.onnx"), full.join("model_fa.onnx")]
        );
    }

    #[test]
    fn owned_ck_selection_rejects_portable_and_early_substitutes() {
        let root = tempfile::tempdir().unwrap();
        let early = root.path().join("onnx/layer-6");
        std::fs::create_dir_all(&early).unwrap();
        std::fs::write(early.join("model_fa_fp16.onnx"), []).unwrap();
        std::fs::write(root.path().join("model.onnx"), []).unwrap();
        std::fs::write(root.path().join("custom.onnx"), []).unwrap();
        let options = InstanceOptions {
            provider: Provider::Rocm,
            custom_ops_profile: CustomOpsProfile::CkFlashAttention,
            ..Default::default()
        };
        assert!(MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).is_err());
        let explicit = InstanceOptions {
            model_path: root.path().to_string_lossy().into_owned(),
            model_file: Some("custom.onnx".into()),
            ..options
        };
        assert_eq!(
            explicit.select_graph(vec![]).unwrap(),
            root.path().join("custom.onnx")
        );
    }

    #[test]
    fn owned_companion_selection_preserves_selected_graph_variant() {
        let root = tempfile::tempdir().unwrap();
        let early = root.path().join("onnx/layer-6");
        std::fs::create_dir_all(&early).unwrap();
        std::fs::write(early.join("model.onnx"), []).unwrap();
        std::fs::write(early.join("model_fa.onnx"), []).unwrap();
        let primary = root.path().join("onnx/layer-22/model_fa_fp16.onnx");
        // Registering the same CK library does not make attention-only and
        // whole-encoder half graphs interchangeable.
        assert!(MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &primary).is_err());
        std::fs::write(early.join("model_fa_fp16.onnx"), []).unwrap();
        assert_eq!(
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &primary).unwrap(),
            vec![
                early.join("model_fa_fp16.onnx"),
                root.path().join("model_fa_fp16_layer_6.onnx"),
                root.path().join("onnx/model_fa_fp16_layer_6.onnx"),
            ]
        );
        let portable = root.path().join("onnx/layer-22/model.onnx");
        let paths =
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &portable).unwrap();
        assert_eq!(paths[0], early.join("model.onnx"));
        assert!(paths
            .iter()
            .all(|path| !path.to_string_lossy().contains("_fa")));
        // An unavailable optional exit is not advertised as a loaded session.
        assert!(
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 11, &primary)
                .unwrap()
                .iter()
                .all(|path| !path.exists())
        );
    }

    #[test]
    fn owned_portable_selection_preserves_flat_full_and_exit_layout() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("model_layer_22.onnx");
        let early = root.path().join("model_layer_6.onnx");
        std::fs::write(&early, []).unwrap();
        let options = InstanceOptions::default();
        assert!(MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).is_err());
        std::fs::write(&primary, []).unwrap();
        assert_eq!(
            MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap(),
            vec![primary.clone()]
        );
        assert!(
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &primary)
                .unwrap()
                .contains(&early)
        );
        assert!(
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &primary)
                .unwrap()
                .iter()
                .all(|path| !path.ends_with("model_layer_22.onnx"))
        );
    }

    #[test]
    fn owned_flat_ck_primaries_preserve_profile_and_canonical_priority() {
        let root = tempfile::tempdir().unwrap();
        let onnx = root.path().join("onnx");
        std::fs::create_dir(&onnx).unwrap();
        for name in [
            "model_layer_22.onnx",
            "model_fa_layer_22.onnx",
            "model_fa_fp16_layer_22.onnx",
            "model_fa_layer_6.onnx",
        ] {
            std::fs::write(onnx.join(name), []).unwrap();
        }
        for provider in [Provider::Cpu, Provider::Migraphx, Provider::Rocm] {
            let options = InstanceOptions {
                provider,
                ..Default::default()
            };
            assert_eq!(
                MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap(),
                vec![onnx.join("model_layer_22.onnx")]
            );
        }
        let options = InstanceOptions {
            provider: Provider::Rocm,
            custom_ops_profile: CustomOpsProfile::CkFlashAttention,
            ..Default::default()
        };
        let expected = vec![
            onnx.join("model_fa_fp16_layer_22.onnx"),
            onnx.join("model_fa_layer_22.onnx"),
        ];
        assert_eq!(
            MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap(),
            expected
        );
        let canonical = onnx.join("model_fa.onnx");
        std::fs::write(&canonical, []).unwrap();
        assert_eq!(
            MmBertEmbeddingModel::owned_primary_candidates(root.path(), 22, &options).unwrap(),
            [vec![canonical], expected].concat()
        );
    }

    #[test]
    fn owned_flat_companions_keep_exact_variant_and_nested_priority() {
        let root = tempfile::tempdir().unwrap();
        let early = root.path().join("onnx/layer-6");
        std::fs::create_dir_all(&early).unwrap();
        for variant in ["model", "model_fa", "model_fa_fp16"] {
            let nested = early.join(format!("{variant}.onnx"));
            let flat = root.path().join(format!("onnx/{variant}_layer_6.onnx"));
            std::fs::write(&nested, []).unwrap();
            std::fs::write(&flat, []).unwrap();
            for primary_name in [
                format!("{variant}.onnx"),
                format!("{variant}_layer_22.onnx"),
            ] {
                let primary = root.path().join("onnx").join(primary_name);
                let paths =
                    MmBertEmbeddingModel::owned_layer_candidates(root.path(), 6, &primary).unwrap();
                assert_eq!(
                    paths,
                    vec![
                        nested.clone(),
                        root.path().join(format!("{variant}_layer_6.onnx")),
                        flat.clone(),
                    ]
                );
            }
        }
        // Known alternatives must produce an explicit mismatch, including both
        // CK precision directions and portable graphs beside CK-only exits.
        for (selected, available) in [
            ("model_fa", "model_fa_fp16"),
            ("model_fa_fp16", "model_fa"),
            ("model", "model_fa"),
            ("model_fa", "model"),
        ] {
            let flat = root.path().join(format!("onnx/{available}_layer_11.onnx"));
            std::fs::write(&flat, []).unwrap();
            let primary = root.path().join(format!("{selected}.onnx"));
            assert!(
                MmBertEmbeddingModel::owned_layer_candidates(root.path(), 11, &primary).is_err()
            );
            std::fs::remove_file(flat).unwrap();
        }
        let custom = root.path().join("custom.onnx");
        assert_eq!(
            MmBertEmbeddingModel::owned_layer_candidates(root.path(), 11, &custom).unwrap(),
            vec![root.path().join("onnx/layer-11/custom.onnx")]
        );
    }

    #[test]
    fn selected_flat_graph_layers_require_exact_names_and_manifest_membership() {
        let root = tempfile::tempdir().unwrap();
        let resolve = |name: &str| {
            let selected = root.path().join(name);
            MmBertEmbeddingModel::primary_graph_layer(
                root.path(),
                &selected,
                &selected,
                &[3, 6, 11, 22],
                22,
            )
        };
        for variant in ["model", "model_fa", "model_fa_fp16"] {
            assert_eq!(resolve(&format!("{variant}.onnx")).unwrap(), 22);
            assert_eq!(resolve(&format!("onnx/{variant}_layer_6.onnx")).unwrap(), 6);
            for suffix in [
                "",
                "0",
                "+6",
                "6_fa",
                "6_dim_768",
                "7",
                "999999999999999999999",
            ] {
                let name = format!("{variant}_layer_{suffix}.onnx");
                assert!(resolve(&name).is_err(), "{name}");
                if suffix != "7" {
                    assert!(MmBertEmbeddingModel::owned_layer_candidates(
                        root.path(),
                        3,
                        &root.path().join(name)
                    )
                    .is_err());
                }
            }
        }
        assert!(resolve("onnx/layer-11/model_fa_layer_6.onnx").is_err());
        assert_eq!(resolve("onnx/layer-6/model_fa_layer_6.onnx").unwrap(), 6);
        assert_eq!(resolve("onnx/layer-6/custom.onnx").unwrap(), 6);
    }

    #[cfg(unix)]
    #[test]
    fn selected_flat_ck_symlinks_cannot_contradict_layer_declarations() {
        let root = tempfile::tempdir().unwrap();
        let onnx = root.path().join("onnx");
        std::fs::create_dir(&onnx).unwrap();
        let target = onnx.join("model_fa_layer_6.onnx");
        std::fs::write(&target, []).unwrap();
        for (name, layer) in [("model_fa.onnx", Some(6)), ("model_fa_layer_11.onnx", None)] {
            let selected = root.path().join(name);
            std::os::unix::fs::symlink(&target, &selected).unwrap();
            let result = MmBertEmbeddingModel::primary_graph_layer(
                root.path(),
                &selected,
                &std::fs::canonicalize(&selected).unwrap(),
                &[3, 6, 11, 22],
                22,
            );
            if let Some(layer) = layer {
                assert_eq!(result.unwrap(), layer);
            } else {
                assert!(result.is_err());
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn owned_companion_layer_alias_is_rejected_before_session_load() {
        let root = tempfile::tempdir().unwrap();
        let primary = root.path().join("model_fa.onnx");
        let target = root.path().join("model_fa_layer_6.onnx");
        // Empty files ensure this check runs before attempting to read an ONNX
        // graph; a session-load error would not prove layer alias rejection.
        std::fs::write(&primary, []).unwrap();
        std::fs::write(&target, []).unwrap();
        std::os::unix::fs::symlink(&target, root.path().join("model_fa_layer_3.onnx")).unwrap();
        let error = MmBertEmbeddingModel::load_layer_sessions(
            root.path(),
            true,
            &[3, 6, 22],
            Some(&InstanceOptions::default()),
            &primary,
            22,
            32768,
        )
        .err()
        .expect("a companion cannot declare different selected and canonical layers");
        assert!(error.to_string().contains("conflicting layer declarations"));
    }

    #[test]
    fn cpu_layer_candidates_exclude_custom_gpu_graphs() {
        let root = Path::new("models/example");
        let cpu = MmBertEmbeddingModel::layer_candidates(root, 6, true, true);
        assert!(cpu
            .iter()
            .all(|path| !path.to_string_lossy().contains("_fa")));
        assert!(cpu
            .iter()
            .all(|path| !path.to_string_lossy().contains("fp16")));
        let gpu = MmBertEmbeddingModel::layer_candidates(root, 6, false, true);
        let fa = gpu
            .iter()
            .position(|p| p.ends_with("model_fa_fp16.onnx"))
            .unwrap();
        let portable = gpu.iter().position(|p| p.ends_with("model.onnx")).unwrap();
        assert!(fa < portable);
    }

    #[test]
    fn primary_graph_never_substitutes_an_early_exit() {
        let root = tempfile::tempdir().unwrap();
        let early = root.path().join("onnx/layer-6");
        std::fs::create_dir_all(&early).unwrap();
        std::fs::write(early.join("model.onnx"), []).unwrap();
        std::fs::write(root.path().join("model_layer_6.onnx"), []).unwrap();
        assert!(MmBertEmbeddingModel::find_onnx_models(root.path(), 22, true).is_err());
        let full = root.path().join("onnx/layer-22");
        std::fs::create_dir_all(&full).unwrap();
        std::fs::write(full.join("model.onnx"), []).unwrap();
        let candidates = MmBertEmbeddingModel::find_onnx_models(root.path(), 22, true).unwrap();
        assert_eq!(candidates, vec![full.join("model.onnx")]);
    }

    #[test]
    fn test_matryoshka_config_defaults() {
        let config = MatryoshkaConfig::default();
        assert_eq!(config.dimensions, vec![768, 512, 256, 128, 64]);
        assert_eq!(config.layers, vec![3, 6, 11, 22]);
    }

    #[test]
    fn test_matryoshka_from_model_dir_reads_manifest() {
        // SSoT: early-exit layers must come from the model's own
        // onnx/model_config.json (available_layers), NOT a hardcoded list.
        // The official mmbert-embed-32k-2d-matryoshka ships [6, 11, 16, 22].
        let dir = tempfile::tempdir().unwrap();
        let onnx_dir = dir.path().join("onnx");
        std::fs::create_dir_all(&onnx_dir).unwrap();
        std::fs::write(
            onnx_dir.join("model_config.json"),
            r#"{"total_layers": 22, "available_layers": [6, 11, 16, 22]}"#,
        )
        .unwrap();

        let config = MatryoshkaConfig::from_model_dir(dir.path());

        assert_eq!(config.layers, vec![6, 11, 16, 22]);
    }

    #[test]
    fn test_matryoshka_from_model_dir_fallback_without_manifest() {
        // No model_config.json present -> fall back to the built-in default
        // rather than erroring, preserving backward compat.
        let dir = tempfile::tempdir().unwrap();
        let config = MatryoshkaConfig::from_model_dir(dir.path());
        assert_eq!(config.layers, MatryoshkaConfig::default().layers);
    }

    #[test]
    fn test_matryoshka_validation() {
        let config = MatryoshkaConfig::default();
        assert!(config.validate_dimension(768));
        assert!(config.validate_dimension(64));
        assert!(!config.validate_dimension(100));
        assert!(config.validate_layer(22));
        assert!(config.validate_layer(6));
        assert!(!config.validate_layer(10));
    }

    #[test]
    fn test_quality_estimation() {
        let config = MatryoshkaConfig::default();
        assert!((config.estimate_quality(22, 768) - 1.0).abs() < 0.001);
        assert!((config.estimate_quality(22, 64) - 0.98).abs() < 0.001);
        assert!((config.estimate_quality(6, 768) - 0.56).abs() < 0.001);
    }

    #[test]
    fn test_speedup_estimation() {
        let config = MatryoshkaConfig::default();
        assert!((config.estimate_speedup(22) - 1.0).abs() < 0.001);
        assert!((config.estimate_speedup(11) - 2.0).abs() < 0.001);
        assert!((config.estimate_speedup(6) - 3.67).abs() < 0.1);
    }

    #[test]
    fn test_execution_provider_best() {
        // Should return CPU when no GPU features are enabled
        let provider = ExecutionProvider::best_available();
        // At minimum, CPU should always work
        assert!(
            provider == ExecutionProvider::Cpu
                || provider == ExecutionProvider::Rocm
                || provider == ExecutionProvider::Cuda
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = MmBertEmbeddingConfig::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.max_position_embeddings, 32768);
    }
}
