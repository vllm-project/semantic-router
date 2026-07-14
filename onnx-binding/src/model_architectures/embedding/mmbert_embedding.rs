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

use crate::core::unified_error::{errors, UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::pooling::{
    l2_normalize, mean_pool_3d, truncate_dimension,
};
use crate::model_architectures::embedding::tokenizer_contract::{
    encode_embedding_batch_checked, prepare_embedding_tokenizer,
};
use half::f16;
use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

// ============================================================================
// Configuration
// ============================================================================

/// mmBERT Embedding model configuration
#[derive(Debug, Clone)]
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

/// Validate the runtime output against the input batch before any row access or
/// sequence pooling. A malformed or mismatched ONNX artifact must be reported
/// as an inference error, never become a panic across the C ABI boundary.
fn validate_embedding_output_shape(
    dims: &[usize],
    expected_input_shape: (usize, usize),
    expected_hidden: usize,
) -> UnifiedResult<()> {
    let (expected_batch, expected_sequence) = expected_input_shape;
    let valid = match dims {
        [batch, hidden] => *batch == expected_batch && *hidden == expected_hidden,
        [batch, sequence, hidden] => {
            *batch == expected_batch && *sequence == expected_sequence && *hidden == expected_hidden
        }
        _ => false,
    };
    if valid {
        return Ok(());
    }

    Err(UnifiedError::Validation {
        field: "embedding_output_shape".to_string(),
        expected: format!(
            "[batch={expected_batch}, hidden={expected_hidden}] or [batch={expected_batch}, sequence={expected_sequence}, hidden={expected_hidden}]"
        ),
        actual: format!("{dims:?}"),
    })
}

fn process_embedding_output(
    dims: &[usize],
    flat: Vec<f32>,
    attention_mask: &Array2<i64>,
    expected_hidden: usize,
) -> UnifiedResult<Array2<f32>> {
    validate_embedding_output_shape(dims, attention_mask.dim(), expected_hidden)?;
    match dims {
        [batch, hidden] => Array2::from_shape_vec((*batch, *hidden), flat)
            .map_err(|e| errors::inference_error("reshape_output", &e.to_string())),
        [batch, sequence, hidden] => {
            let hidden_states = Array3::from_shape_vec((*batch, *sequence, *hidden), flat)
                .map_err(|e| errors::inference_error("reshape_hidden_states", &e.to_string()))?;
            let mask_f32: Array2<f32> = attention_mask.mapv(|x| x as f32);
            Ok(mean_pool_3d(&hidden_states, &mask_f32))
        }
        _ => Err(errors::inference_error(
            "extract_output",
            "embedding output shape validation drifted before processing",
        )),
    }
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
    session: Session,
    /// Tokenizer
    tokenizer: Arc<Tokenizer>,
    /// Model configuration
    config: MmBertEmbeddingConfig,
    /// Matryoshka configuration
    matryoshka_config: MatryoshkaConfig,
    /// Model path
    model_path: String,
    /// Full layer represented by the primary session. Candidate discovery only
    /// returns artifacts whose layout proves this layer contract.
    primary_layer: usize,
    /// Whether the model supports layer early exit (requires multiple ONNX files)
    supports_layer_exit: bool,
    /// Layer-specific sessions (for early exit support)
    layer_sessions: Vec<Option<Session>>,
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
        let model_path_str = model_path.as_ref().display().to_string();
        let model_dir = model_path.as_ref();

        // Load configuration
        let config = MmBertEmbeddingConfig::from_pretrained(&model_path)?;

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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        let tokenizer = prepare_embedding_tokenizer(tokenizer, "mmBERT")?;

        // Find ONNX model candidates (priority order)
        let primary_layer = config.num_hidden_layers;
        let onnx_candidates = Self::find_onnx_models(&model_path, primary_layer)?;

        // Create ONNX Runtime session with fallback across candidates.
        // We intentionally prefer GPU-optimized model variants first.
        let mut selected_session: Option<Session> = None;
        let mut selected_path: Option<std::path::PathBuf> = None;
        let mut last_error: Option<String> = None;
        for onnx_path in onnx_candidates {
            match Self::create_session(&onnx_path, use_cpu) {
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
        if let Some(path) = selected_path {
            println!("INFO: Selected mmBERT ONNX file: {}", path.display());
        }

        // Check for layer-specific ONNX files (for early exit support)
        let (supports_layer_exit, layer_sessions) = Self::load_layer_sessions(
            &model_path,
            use_cpu,
            &matryoshka_config.layers,
            primary_layer,
        );

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            matryoshka_config,
            model_path: model_path_str,
            primary_layer,
            supports_layer_exit,
            layer_sessions,
        })
    }

    /// Find ONNX model candidates in priority order.
    ///
    /// Searches recognized full-model names under model_path/ and
    /// model_path/onnx/, plus the directory or legacy filename for the exact
    /// configured full layer. Lower-layer early-exit artifacts are deliberately
    /// excluded: they must never masquerade as the primary/full session.
    fn find_onnx_models<P: AsRef<Path>>(
        model_path: P,
        full_layer: usize,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        if full_layer == 0 {
            return Err(errors::config_error(
                "num_hidden_layers",
                "must be greater than zero to identify a full-layer model",
            ));
        }
        let dir = model_path.as_ref();
        let onnx_subdir = dir.join("onnx");

        let has_fa = std::env::var("ORT_CK_FLASH_ATTN_LIB")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some();
        let candidates: &[&str] = if has_fa {
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

        let search_dirs: Vec<std::path::PathBuf> = vec![dir.to_path_buf(), onnx_subdir.clone()];

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

        // Legacy flat full-layer filenames are also unambiguous.
        let full_layer_filename = format!("model_layer_{full_layer}.onnx");
        for base_dir in &search_dirs {
            let path = base_dir.join(&full_layer_filename);
            if path.exists() && !results.iter().any(|candidate| candidate == &path) {
                results.push(path);
            }
        }

        // A HuggingFace layer directory is eligible only when it names the
        // configured full layer. Lower layer directories are loaded separately.
        let full_layer_dir = onnx_subdir.join(format!("layer-{full_layer}"));
        if full_layer_dir.is_dir() {
            for candidate in candidates {
                let path = full_layer_dir.join(candidate);
                if path.exists() && !results.iter().any(|existing| existing == &path) {
                    results.push(path);
                }
            }
        }

        if results.is_empty() {
            return Err(errors::file_not_found(&format!(
                "No full-layer ONNX model found in {} (checked root, onnx/, and onnx/layer-{full_layer}/)",
                dir.display(),
            )));
        }
        Ok(results)
    }

    /// Create an ONNX Runtime session with appropriate execution provider
    fn create_session<P: AsRef<Path>>(onnx_path: P, use_cpu: bool) -> UnifiedResult<Session> {
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
                    .and_then(|b| maybe_register_custom_ops(b))
                    .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                {
                    Ok(session) => {
                        println!("INFO: Using ROCm execution provider (AMD GPU) — verified");
                        return Ok(session);
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
                        return Ok(session);
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
                        return Ok(session);
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

        Ok(session)
    }

    /// Load layer-specific ONNX sessions for early exit support.
    ///
    /// Searches for layer models in multiple locations:
    /// - model_path/model_layer_{N}.onnx  (legacy flat layout)
    /// - model_path/onnx/model_layer_{N}.onnx
    /// - model_path/onnx/layer-{N}/model_fa_fp16.onnx  (HuggingFace FA)
    /// - model_path/onnx/layer-{N}/model.onnx           (HuggingFace default)
    fn load_layer_sessions<P: AsRef<Path>>(
        model_path: P,
        use_cpu: bool,
        layers: &[usize],
        full_layer: usize,
    ) -> (bool, Vec<Option<Session>>) {
        let mut sessions = Vec::new();
        let mut any_loaded = false;
        let model_dir = model_path.as_ref();
        let onnx_dir = model_dir.join("onnx");

        let has_fa = std::env::var("ORT_CK_FLASH_ATTN_LIB")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some();

        for layer in layers {
            // The primary session owns the configured full layer. Loading it a
            // second time is wasteful and would falsely advertise layer-exit
            // support when no actual early-exit artifact exists.
            if *layer >= full_layer {
                sessions.push(None);
                continue;
            }
            let layer_filename = format!("model_layer_{}.onnx", layer);
            let hf_layer_dir = onnx_dir.join(format!("layer-{}", layer));

            let mut candidates = vec![
                model_dir.join(&layer_filename),
                onnx_dir.join(&layer_filename),
            ];
            // HuggingFace-style layer subdirectories with FA priority
            if has_fa {
                candidates.push(hf_layer_dir.join("model_fa_fp16.onnx"));
                candidates.push(hf_layer_dir.join("model_fa.onnx"));
            }
            candidates.push(hf_layer_dir.join("model.onnx"));

            let found = candidates.iter().find(|p| p.exists()).cloned();

            if let Some(ref layer_path) = found {
                println!(
                    "INFO: Loading layer-{} from {}",
                    layer,
                    layer_path.display()
                );
                match Self::create_session(layer_path, use_cpu) {
                    Ok(session) => {
                        sessions.push(Some(session));
                        any_loaded = true;
                    }
                    Err(e) => {
                        println!("WARN: Failed to load layer-{}: {:?}", layer, e);
                        sessions.push(None);
                    }
                }
            } else {
                sessions.push(None);
            }
        }

        (any_loaded, sessions)
    }

    /// Get the model configuration
    pub fn config(&self) -> &MmBertEmbeddingConfig {
        &self.config
    }

    /// Get the loaded model's advertised Matryoshka configuration.
    pub fn matryoshka_config(&self) -> &MatryoshkaConfig {
        &self.matryoshka_config
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
        self.supports_layer_exit
    }

    /// Get available early exit layers
    pub fn available_exit_layers(&self) -> Vec<usize> {
        if self.supports_layer_exit {
            let mut layers = self
                .matryoshka_config
                .layers
                .iter()
                .enumerate()
                .filter_map(|(i, &layer)| {
                    if self
                        .layer_sessions
                        .get(i)
                        .is_some_and(|s: &Option<Session>| s.is_some())
                    {
                        Some(layer)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            if !layers.contains(&self.primary_layer) {
                layers.push(self.primary_layer);
                layers.sort_unstable();
            }
            layers
        } else {
            vec![self.primary_layer]
        }
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
        self.encode_with_token_counts(texts, target_layer, target_dim)
            .map(|(embeddings, _)| embeddings)
    }

    /// Generate embeddings and return the exact tokenizer length for every
    /// input. FFI metadata must use these counts rather than whitespace-based
    /// approximations, especially for CJK and punctuation-heavy text.
    pub fn encode_with_token_counts(
        &mut self,
        texts: &[&str],
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<(Array2<f32>, Vec<usize>)> {
        if texts.is_empty() {
            return Err(UnifiedError::Validation {
                field: "texts".to_string(),
                expected: "non-empty".to_string(),
                actual: "empty".to_string(),
            });
        }

        // Tokenize
        let encodings = encode_embedding_batch_checked(
            &self.tokenizer,
            texts,
            self.config.max_position_embeddings,
            "mmBERT",
        )?;
        let token_counts = encodings.iter().map(|encoding| encoding.len()).collect();

        // Find max sequence length
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        // Prepare input tensors
        let batch_size = texts.len();
        let mut input_ids = vec![self.config.pad_token_id as i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let seq_len = encoding.len();
            for j in 0..seq_len {
                input_ids[i * max_len + j] = encoding.get_ids()[j] as i64;
                attention_mask[i * max_len + j] = 1;
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

        Ok((normalized, token_counts))
    }

    /// Run inference on the ONNX model with optional layer selection
    fn run_inference_with_layer(
        &mut self,
        target_layer: Option<usize>,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> UnifiedResult<Array2<f32>> {
        let expected_hidden = self.config.hidden_size;
        // Select session based on target layer (inline to avoid borrow issues)
        let session_idx = match target_layer {
            None => None,
            Some(layer) if layer == self.primary_layer => None,
            Some(layer) => {
                let index = self
                    .matryoshka_config
                    .layers
                    .iter()
                    .position(|&candidate| candidate == layer)
                    .filter(|&idx| {
                        self.layer_sessions
                            .get(idx)
                            .is_some_and(|session| session.is_some())
                    })
                    .ok_or_else(|| UnifiedError::Validation {
                        field: "target_layer".to_string(),
                        expected: format!("one of {:?}", self.available_exit_layers()),
                        actual: layer.to_string(),
                    })?;
                Some(index)
            }
        };

        // Get the appropriate session
        let session = if let Some(idx) = session_idx {
            self.layer_sessions
                .get_mut(idx)
                .and_then(Option::as_mut)
                .ok_or_else(|| {
                    errors::inference_error(
                        "select_layer_session",
                        "validated early-exit session became unavailable",
                    )
                })?
        } else {
            &mut self.session
        };
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Create ort tensors from ndarray - ort 2.x requires (shape, data) tuple
        let input_ids_flat: Vec<i64> = input_ids.iter().copied().collect();
        let attention_mask_flat: Vec<i64> = attention_mask.iter().copied().collect();

        let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids_flat))
            .map_err(|e: ort::Error| {
                errors::inference_error("create_input_ids_tensor", &e.to_string())
            })?;

        let attention_mask_tensor =
            Tensor::from_array(([batch_size, seq_len], attention_mask_flat)).map_err(
                |e: ort::Error| {
                    errors::inference_error("create_attention_mask_tensor", &e.to_string())
                },
            )?;

        // Run the session with inputs
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?;

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

        for name in &output_names {
            if let Some(output_value) = outputs.get(*name) {
                if let Some((dims, flat)) = try_extract_f32!(output_value) {
                    return process_embedding_output(&dims, flat, attention_mask, expected_hidden);
                }
            }
        }

        // Try first output if named outputs not found
        if let Some((_, output_value)) = outputs.iter().next() {
            if let Some((dims, flat)) = try_extract_f32!(output_value) {
                return process_embedding_output(&dims, flat, attention_mask, expected_hidden);
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

    /// Encode one input and return its exact tokenizer length.
    pub fn encode_single_with_token_count(
        &mut self,
        text: &str,
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<(Array1<f32>, usize)> {
        let (embeddings, token_counts) =
            self.encode_with_token_counts(&[text], target_layer, target_dim)?;
        Ok((embeddings.row(0).to_owned(), token_counts[0]))
    }

    /// Get model information for debugging
    pub fn model_info(&self) -> String {
        format!(
            "MmBertEmbeddingModel(path={}, hidden_size={}, layers={}, layer_exit={}, available_exits={:?})",
            self.model_path,
            self.config.hidden_size,
            self.config.num_hidden_layers,
            self.supports_layer_exit,
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
    fn test_embedding_output_shape_contract() {
        let input_shape = (2, 5);
        let hidden_size = 768;
        assert!(validate_embedding_output_shape(&[2, 768], input_shape, hidden_size).is_ok());
        assert!(validate_embedding_output_shape(&[2, 5, 768], input_shape, hidden_size).is_ok());

        for invalid in [
            vec![1, 768],
            vec![2, 0],
            vec![2, 1],
            vec![2, 1024],
            vec![1, 5, 768],
            vec![2, 4, 768],
            vec![2, 5, 0],
            vec![2, 5, 1],
            vec![2, 5, 1024],
            vec![2],
            vec![2, 5, 768, 1],
        ] {
            let error = validate_embedding_output_shape(&invalid, input_shape, hidden_size)
                .expect_err("invalid model output shape must fail closed");
            assert!(matches!(
                error,
                UnifiedError::Validation { ref field, .. }
                    if field == "embedding_output_shape"
            ));
        }
    }

    #[test]
    fn test_primary_candidates_require_a_provable_full_layer() {
        let dir = tempfile::tempdir().unwrap();
        let lower_dir = dir.path().join("onnx/layer-6");
        std::fs::create_dir_all(&lower_dir).unwrap();
        let lower_model = lower_dir.join("model.onnx");
        std::fs::write(&lower_model, b"lower-layer-placeholder").unwrap();
        std::fs::write(dir.path().join("unrecognized.onnx"), b"unknown").unwrap();

        assert!(MmBertEmbeddingModel::find_onnx_models(dir.path(), 0).is_err());
        assert!(MmBertEmbeddingModel::find_onnx_models(dir.path(), 22).is_err());

        let full_dir = dir.path().join("onnx/layer-22");
        std::fs::create_dir_all(&full_dir).unwrap();
        let full_model = full_dir.join("model.onnx");
        std::fs::write(&full_model, b"full-layer-placeholder").unwrap();

        let candidates = MmBertEmbeddingModel::find_onnx_models(dir.path(), 22).unwrap();
        assert!(candidates.contains(&full_model));
        assert!(!candidates.contains(&lower_model));
        assert!(!candidates.contains(&dir.path().join("unrecognized.onnx")));
    }

    #[test]
    fn test_primary_candidates_accept_canonical_full_model_name() {
        let dir = tempfile::tempdir().unwrap();
        let full_model = dir.path().join("model.onnx");
        std::fs::write(&full_model, b"full-model-placeholder").unwrap();

        let candidates = MmBertEmbeddingModel::find_onnx_models(dir.path(), 22).unwrap();
        assert_eq!(candidates, vec![full_model]);
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
