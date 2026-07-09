//! mmBERT-32K-YaRN Classification Model using ONNX Runtime
//!
//! Supports:
//! - Sequence classification (Intent, Jailbreak, Feedback, Factcheck)
//! - Token classification (PII detection)
//! - ROCm (AMD GPU), CUDA (NVIDIA GPU), OpenVINO (Intel), and CPU
//!
//! ## Performance (seq_len=128)
//! - ROCm MIGraphX FP16: ~2ms
//! - CPU OpenVINO FP32: ~22ms
//! - CPU ORT FP32: ~41ms

use crate::core::unified_error::{errors, UnifiedResult};
use half::f16;
use ndarray::Array2;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// Maximum sequence length for classification inference.
///
/// ModernBERT-32k has global attention layers (every 3 layers = 7–8 out of 22)
/// that scale quadratically with sequence length. At ~4000 tokens the global-attention
/// layers require roughly 6–8 GB of activation memory per batch item, reliably
/// triggering an OOM kill with no container logs. Classification tasks (intent,
/// jailbreak, PII) only need the first few hundred tokens to produce a reliable
/// signal, so we cap at 512 — matching the `max_length` field in tokenizer_config.json.
const MAX_CLASSIFICATION_SEQ_LEN: usize = 512;

// ============================================================================
// Classification Types
// ============================================================================

/// Classification result for a single input
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// Predicted class label
    pub label: String,
    /// Predicted class ID
    pub class_id: i32,
    /// Confidence score (probability)
    pub confidence: f32,
    /// All class probabilities
    pub probabilities: Vec<f32>,
}

/// Token classification result (for PII detection)
#[derive(Debug, Clone)]
pub struct TokenClassificationResult {
    /// List of detected entities
    pub entities: Vec<DetectedEntity>,
}

/// A detected entity (for PII)
#[derive(Debug, Clone)]
pub struct DetectedEntity {
    /// Entity text
    pub text: String,
    /// Entity type (e.g., "US_SSN", "EMAIL")
    pub entity_type: String,
    /// Start character offset
    pub start: usize,
    /// End character offset
    pub end: usize,
    /// Confidence score
    pub confidence: f32,
}

// ============================================================================
// Model Configuration
// ============================================================================

/// mmBERT Classifier configuration
#[derive(Debug, Clone)]
pub struct MmBertClassifierConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub num_labels: usize,
    pub id2label: HashMap<i32, String>,
    pub label2id: HashMap<String, i32>,
    pub pad_token_id: u32,
}

impl Default for MmBertClassifierConfig {
    fn default() -> Self {
        Self {
            vocab_size: 256000,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            max_position_embeddings: 32768,
            num_labels: 2,
            id2label: HashMap::new(),
            label2id: HashMap::new(),
            pad_token_id: 0,
        }
    }
}

impl MmBertClassifierConfig {
    /// Load configuration from a pretrained model directory
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let config_path = model_path.as_ref().join("config.json");

        if !config_path.exists() {
            return Err(errors::file_not_found(&config_path.display().to_string()));
        }

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| errors::file_not_found(&config_path.display().to_string()))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            errors::invalid_json(&config_path.display().to_string(), &e.to_string())
        })?;

        // Parse id2label
        let mut id2label = HashMap::new();
        let mut label2id = HashMap::new();

        if let Some(id2label_obj) = config_json.get("id2label").and_then(|v| v.as_object()) {
            for (k, v) in id2label_obj {
                if let (Ok(id), Some(label)) = (k.parse::<i32>(), v.as_str()) {
                    id2label.insert(id, label.to_string());
                    label2id.insert(label.to_string(), id);
                }
            }
        }

        let num_labels = config_json["num_labels"]
            .as_u64()
            .unwrap_or(id2label.len() as u64) as usize;

        Ok(Self {
            vocab_size: config_json["vocab_size"].as_u64().unwrap_or(256000) as usize,
            hidden_size: config_json["hidden_size"].as_u64().unwrap_or(768) as usize,
            num_hidden_layers: config_json["num_hidden_layers"].as_u64().unwrap_or(22) as usize,
            num_attention_heads: config_json["num_attention_heads"].as_u64().unwrap_or(12) as usize,
            max_position_embeddings: config_json["max_position_embeddings"]
                .as_u64()
                .unwrap_or(32768) as usize,
            num_labels,
            id2label,
            label2id,
            pad_token_id: config_json["pad_token_id"].as_u64().unwrap_or(0) as u32,
        })
    }

    /// Get label name from ID
    pub fn get_label(&self, id: i32) -> String {
        self.id2label
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{}", id))
    }
}

// ============================================================================
// Execution Provider
// ============================================================================

/// Execution provider preference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClassifierExecutionProvider {
    /// Automatic selection (ROCm > CUDA > CPU; OpenVINO requires an explicit `OpenVino` request)
    Auto,
    /// Force CPU
    Cpu,
    /// AMD GPU via ROCm/MIGraphX
    Rocm,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Intel via OpenVINO
    OpenVino,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BaselineAmdPolicy {
    CpuOnly,
}

// ============================================================================
// Sequence Classification Model
// ============================================================================

/// mmBERT Sequence Classification Model
///
/// Used for:
/// - Intent classification
/// - Jailbreak detection
/// - Feedback classification
/// - Factcheck classification
pub struct MmBertSequenceClassifier {
    session: Session,
    tokenizer: Arc<Tokenizer>,
    config: MmBertClassifierConfig,
    model_path: String,
    input_len_buckets: Vec<usize>,
}

impl MmBertSequenceClassifier {
    /// Load classifier from directory
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();

        // Load configuration
        let config = MmBertClassifierConfig::from_pretrained(&model_path)?;

        // Load tokenizer
        let tokenizer_path = model_path.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(errors::file_not_found(
                &tokenizer_path.display().to_string(),
            ));
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        // Apply truncation at load time so encode_batch never produces sequences
        // longer than MAX_CLASSIFICATION_SEQ_LEN. The tokenizer.json loaded above
        // has no truncation set (tokenizer_config.json max_length=512 is Python-side
        // metadata that the Rust tokenizers crate does not apply automatically).
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_CLASSIFICATION_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        // Find ONNX model candidates and initialize with fallback.
        let onnx_candidates = Self::find_onnx_models(&model_path, provider)?;
        let (session, onnx_path) = Self::create_session_with_fallback(
            onnx_candidates,
            provider,
            &model_path_str,
            BaselineAmdPolicy::CpuOnly,
        )?;
        println!(
            "INFO: Selected classifier ONNX file: {}",
            onnx_path.display()
        );

        let input_len_buckets = Self::sequence_input_len_buckets_for_artifact(
            &onnx_path,
            provider,
            &config,
            std::env::var("VSR_AMD_MIGRAPHX_SEQUENCE_BUCKETS")
                .ok()
                .as_deref(),
        );
        if !input_len_buckets.is_empty() {
            println!(
                "INFO: Using sequence-classifier input buckets {:?} for AMD MIGraphX sequence experiment {}",
                input_len_buckets,
                onnx_path.display(),
            );
        }

        let mut classifier = Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path_str,
            input_len_buckets,
        };
        classifier.warmup_if_requested(&onnx_path);

        Ok(classifier)
    }

    /// Find ONNX model candidates in priority order.
    fn find_onnx_models<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let has_ck_fa = crate::core::ort_migraphx::ck_flash_attention_available();
        let candidates = Self::onnx_candidate_filenames(provider, has_ck_fa);
        Self::find_onnx_models_from_candidates(model_path, &candidates, false)
    }

    fn find_token_onnx_models<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let has_ck_fa = crate::core::ort_migraphx::ck_flash_attention_available();
        let candidates = Self::token_onnx_candidate_filenames(provider, has_ck_fa);
        Self::find_onnx_models_from_candidates(model_path, &candidates, true)
    }

    fn find_onnx_models_from_candidates<P: AsRef<Path>>(
        model_path: P,
        candidates: &[&str],
        skip_sequence_optimized_fallbacks: bool,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let dir = model_path.as_ref();
        let onnx_subdir = dir.join("onnx");
        let search_dirs = [dir, onnx_subdir.as_path()];

        let mut results: Vec<std::path::PathBuf> = Vec::new();
        // Try known ONNX filenames first in both model root and `onnx/` subdirectory.
        for base_dir in search_dirs {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            for &candidate in candidates {
                let path = base_dir.join(candidate);
                if path.exists() && !results.iter().any(|p| p == &path) {
                    results.push(path);
                }
            }
        }

        // Fallback: include any .onnx file in both locations.
        for base_dir in search_dirs {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if skip_sequence_optimized_fallbacks
                        && Self::is_sequence_optimized_onnx_artifact(&path)
                    {
                        continue;
                    }
                    if path.extension().is_some_and(|ext| ext == "onnx")
                        && !results.iter().any(|p| p == &path)
                    {
                        results.push(path);
                    }
                }
            }
        }

        if results.is_empty() {
            return Err(errors::file_not_found(&format!(
                "No ONNX model found in {} (checked root and onnx/ subdir)",
                dir.display(),
            )));
        }
        Ok(results)
    }

    fn onnx_candidate_filenames(
        provider: ClassifierExecutionProvider,
        has_ck_fa: bool,
    ) -> Vec<&'static str> {
        Self::onnx_candidate_filenames_with_order(
            provider,
            has_ck_fa,
            std::env::var("VSR_AMD_SEQUENCE_PROVIDER_ORDER")
                .ok()
                .as_deref(),
        )
    }

    fn onnx_candidate_filenames_with_order(
        provider: ClassifierExecutionProvider,
        has_ck_fa: bool,
        provider_order_env: Option<&str>,
    ) -> Vec<&'static str> {
        let mut candidates = Vec::new();
        let mut push_candidate = |name: &'static str| {
            if !candidates.contains(&name) {
                candidates.push(name);
            }
        };

        if Self::prefers_amd_onnx_artifacts(provider) {
            if Self::sequence_sdpa_migraphx_first_enabled(provider_order_env) {
                push_candidate("model_sdpa_migraphx.onnx");
            }
            push_candidate("model_sdpa_fp16.onnx");
            if has_ck_fa {
                push_candidate("model_fa_fp16.onnx");
                push_candidate("model_fa.onnx");
            }
            push_candidate("model.onnx");
        } else {
            push_candidate("model.onnx");
            push_candidate("classifier.onnx");
            push_candidate("model_optimized.onnx");
            push_candidate("model_sdpa_fp16.onnx");
        }

        push_candidate("classifier.onnx");
        push_candidate("model_optimized.onnx");
        candidates
    }

    fn token_onnx_candidate_filenames(
        provider: ClassifierExecutionProvider,
        has_ck_fa: bool,
    ) -> Vec<&'static str> {
        let mut candidates = Vec::new();
        let mut push_candidate = |name: &'static str| {
            if !candidates.contains(&name) {
                candidates.push(name);
            }
        };

        if Self::prefers_amd_onnx_artifacts(provider) && Self::migraphx_token_artifacts_enabled() {
            push_candidate("model_token_sdpa.onnx");
            push_candidate("token_classifier_sdpa.onnx");
            push_candidate("model_token_sdpa_fp16.onnx");
            push_candidate("token_classifier_sdpa_fp16.onnx");
            push_candidate("model_token_eager.onnx");
            push_candidate("token_classifier_eager.onnx");
            push_candidate("model_token_eager_fp16.onnx");
            push_candidate("token_classifier_eager_fp16.onnx");
        } else {
            if Self::prefers_amd_onnx_artifacts(provider) && has_ck_fa {
                push_candidate("model_fa_fp16.onnx");
                push_candidate("model_fa.onnx");
            }
        }

        push_candidate("model.onnx");
        push_candidate("token_classifier.onnx");
        push_candidate("classifier.onnx");
        push_candidate("model_optimized.onnx");
        push_candidate("model_token_sdpa.onnx");
        push_candidate("token_classifier_sdpa.onnx");
        push_candidate("model_token_sdpa_fp16.onnx");
        push_candidate("token_classifier_sdpa_fp16.onnx");
        push_candidate("model_token_eager.onnx");
        push_candidate("token_classifier_eager.onnx");
        push_candidate("model_token_eager_fp16.onnx");
        push_candidate("token_classifier_eager_fp16.onnx");
        candidates
    }

    fn prefers_amd_onnx_artifacts(provider: ClassifierExecutionProvider) -> bool {
        match provider {
            ClassifierExecutionProvider::Rocm => true,
            ClassifierExecutionProvider::Auto => cfg!(any(feature = "migraphx", feature = "rocm")),
            ClassifierExecutionProvider::Cpu
            | ClassifierExecutionProvider::Cuda
            | ClassifierExecutionProvider::OpenVino => false,
        }
    }

    fn is_baseline_onnx_artifact(onnx_path: &Path) -> bool {
        matches!(
            onnx_path.file_name().and_then(|name| name.to_str()),
            Some("model.onnx" | "classifier.onnx" | "model_optimized.onnx")
        )
    }

    fn is_sequence_optimized_onnx_artifact(onnx_path: &Path) -> bool {
        matches!(
            onnx_path.file_name().and_then(|name| name.to_str()),
            Some(
                "model_sdpa_migraphx.onnx"
                    | "model_sdpa_fp16.onnx"
                    | "model_fa_fp16.onnx"
                    | "model_fa.onnx"
            )
        )
    }

    fn is_sequence_sdpa_onnx_artifact(onnx_path: &Path) -> bool {
        matches!(
            onnx_path.file_name().and_then(|name| name.to_str()),
            Some("model_sdpa_migraphx.onnx" | "model_sdpa_fp16.onnx")
        )
    }

    fn sequence_input_len_buckets_from_env(
        config: &MmBertClassifierConfig,
        value: Option<&str>,
    ) -> Vec<usize> {
        let max_len = MAX_CLASSIFICATION_SEQ_LEN.min(config.max_position_embeddings);
        let Some(value) = value else {
            return vec![64.min(max_len), 128.min(max_len), max_len]
                .into_iter()
                .filter(|bucket| *bucket > 0)
                .collect();
        };
        if matches!(
            value,
            "0" | "dynamic" | "none" | "false" | "FALSE" | "no" | "NO"
        ) {
            return Vec::new();
        }

        let mut buckets: Vec<usize> = value
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .filter(|bucket| *bucket > 0)
            .map(|bucket| bucket.min(max_len))
            .collect();
        buckets.sort_unstable();
        buckets.dedup();
        buckets
    }

    fn sequence_input_len_buckets_for_artifact(
        onnx_path: &Path,
        provider: ClassifierExecutionProvider,
        config: &MmBertClassifierConfig,
        buckets_env: Option<&str>,
    ) -> Vec<usize> {
        Self::sequence_input_len_buckets_for_artifact_with_order(
            onnx_path,
            provider,
            config,
            buckets_env,
            std::env::var("VSR_AMD_SEQUENCE_PROVIDER_ORDER")
                .ok()
                .as_deref(),
        )
    }

    fn sequence_input_len_buckets_for_artifact_with_order(
        onnx_path: &Path,
        provider: ClassifierExecutionProvider,
        config: &MmBertClassifierConfig,
        buckets_env: Option<&str>,
        provider_order_env: Option<&str>,
    ) -> Vec<usize> {
        if Self::prefers_amd_onnx_artifacts(provider)
            && Self::is_sequence_sdpa_onnx_artifact(onnx_path)
            && Self::sequence_sdpa_migraphx_first_enabled(provider_order_env)
        {
            Self::sequence_input_len_buckets_from_env(config, buckets_env)
        } else {
            Vec::new()
        }
    }

    fn sequence_warmup_enabled() -> bool {
        std::env::var("VSR_AMD_MIGRAPHX_WARMUP")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    }

    fn is_token_optimized_onnx_artifact(onnx_path: &Path) -> bool {
        matches!(
            onnx_path.file_name().and_then(|name| name.to_str()),
            Some(
                "model_token_sdpa.onnx"
                    | "token_classifier_sdpa.onnx"
                    | "model_token_sdpa_fp16.onnx"
                    | "token_classifier_sdpa_fp16.onnx"
                    | "model_token_eager.onnx"
                    | "token_classifier_eager.onnx"
                    | "model_token_eager_fp16.onnx"
                    | "token_classifier_eager_fp16.onnx"
            )
        )
    }

    fn experimental_migraphx_token_artifacts_enabled() -> bool {
        std::env::var("VSR_ENABLE_EXPERIMENTAL_MIGRAPHX_TOKEN_ARTIFACTS")
            .ok()
            .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
    }

    fn migraphx_token_artifacts_enabled() -> bool {
        Self::token_artifacts_migraphx_enabled(
            Self::experimental_migraphx_token_artifacts_enabled(),
            std::env::var("MIGRAPHX_MLIR_USE_SPECIFIC_OPS")
                .ok()
                .as_deref(),
            std::env::var("MIGRAPHX_DISABLE_MLIR").ok().as_deref(),
        )
    }

    fn token_artifacts_migraphx_enabled(
        experimental_enabled: bool,
        mlir_specific_ops: Option<&str>,
        disable_mlir: Option<&str>,
    ) -> bool {
        experimental_enabled
            && (Self::migraphx_attention_mlir_disabled(mlir_specific_ops)
                || Self::migraphx_mlir_disabled(disable_mlir))
    }

    fn migraphx_attention_mlir_disabled(value: Option<&str>) -> bool {
        value
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .any(|op| op.eq_ignore_ascii_case("~attention"))
            })
            .unwrap_or(false)
    }

    fn migraphx_mlir_disabled(value: Option<&str>) -> bool {
        value
            .map(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false)
    }

    fn sequence_sdpa_migraphx_first_enabled(value: Option<&str>) -> bool {
        value
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "migraphx" | "migraphx-first" | "mgx" | "mgx-first"
                )
            })
            .unwrap_or(false)
    }

    fn amd_provider_preference_for_artifact(
        onnx_path: &Path,
    ) -> crate::core::ort_migraphx::AmdProviderPreference {
        Self::amd_provider_preference_for_artifact_with_order(
            onnx_path,
            std::env::var("VSR_AMD_SEQUENCE_PROVIDER_ORDER")
                .ok()
                .as_deref(),
        )
    }

    fn amd_provider_preference_for_artifact_with_order(
        onnx_path: &Path,
        provider_order_env: Option<&str>,
    ) -> crate::core::ort_migraphx::AmdProviderPreference {
        if Self::is_sequence_sdpa_onnx_artifact(onnx_path)
            && !Self::sequence_sdpa_migraphx_first_enabled(provider_order_env)
        {
            crate::core::ort_migraphx::AmdProviderPreference::RocmFirst
        } else {
            crate::core::ort_migraphx::AmdProviderPreference::MigraphxFirst
        }
    }

    fn session_provider_for_candidate(
        onnx_path: &Path,
        provider: ClassifierExecutionProvider,
        baseline_amd_policy: BaselineAmdPolicy,
    ) -> ClassifierExecutionProvider {
        if Self::prefers_amd_onnx_artifacts(provider)
            && baseline_amd_policy == BaselineAmdPolicy::CpuOnly
            && Self::is_baseline_onnx_artifact(onnx_path)
        {
            ClassifierExecutionProvider::Cpu
        } else if Self::prefers_amd_onnx_artifacts(provider)
            && Self::is_token_optimized_onnx_artifact(onnx_path)
            && !Self::migraphx_token_artifacts_enabled()
        {
            ClassifierExecutionProvider::Cpu
        } else {
            provider
        }
    }

    /// Create session from candidates with fallback across files.
    fn create_session_with_fallback(
        onnx_candidates: Vec<std::path::PathBuf>,
        provider: ClassifierExecutionProvider,
        model_path: &str,
        baseline_amd_policy: BaselineAmdPolicy,
    ) -> UnifiedResult<(Session, std::path::PathBuf)> {
        let mut last_error: Option<String> = None;
        for onnx_path in onnx_candidates {
            let session_provider =
                Self::session_provider_for_candidate(&onnx_path, provider, baseline_amd_policy);
            if session_provider == ClassifierExecutionProvider::Cpu
                && provider != ClassifierExecutionProvider::Cpu
                && baseline_amd_policy == BaselineAmdPolicy::CpuOnly
                && Self::is_baseline_onnx_artifact(&onnx_path)
            {
                println!(
                    "WARNING: Using CPU for baseline classifier artifact {} because this raw artifact is not MIGraphX-safe on AMD paths; provide a validated optimized artifact for AMD acceleration",
                    onnx_path.display()
                );
            }

            match Self::create_session(&onnx_path, session_provider) {
                Ok(session) => return Ok((session, onnx_path)),
                Err(e) => {
                    let reason = format!("{:?}", e);
                    println!(
                        "WARN: Failed to initialize classifier session from {}: {}",
                        onnx_path.display(),
                        reason
                    );
                    last_error = Some(format!("{}: {}", onnx_path.display(), reason));
                }
            }
        }
        let detail = last_error.unwrap_or_else(|| "no ONNX candidate was loadable".to_string());
        Err(errors::model_load(model_path, &detail))
    }

    /// Create ONNX Runtime session with specified provider
    fn create_session<P: AsRef<Path>>(
        onnx_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Session> {
        let onnx_path_str = onnx_path.as_ref().display().to_string();

        match provider {
            ClassifierExecutionProvider::Cpu => {
                println!("INFO: Using CPU execution provider");
                Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))
            }
            ClassifierExecutionProvider::Rocm | ClassifierExecutionProvider::Auto => {
                #[cfg(any(feature = "migraphx", feature = "rocm"))]
                {
                    let ck_fa_lib = crate::core::ort_migraphx::ck_flash_attention_library_for_model(
                        onnx_path.as_ref(),
                    );
                    let preference = Self::amd_provider_preference_for_artifact(onnx_path.as_ref());
                    match crate::core::ort_migraphx::create_amd_session_with_preference(
                        onnx_path.as_ref(),
                        ck_fa_lib.as_deref(),
                        preference,
                    ) {
                        Ok(amd_session) => return Ok(amd_session.session),
                        Err(e) => println!("WARNING: AMD execution providers failed: {e}"),
                    }

                    println!("WARNING: All GPU execution providers failed, falling back to CPU");
                }
                #[cfg(not(any(feature = "migraphx", feature = "rocm")))]
                {
                    if matches!(provider, ClassifierExecutionProvider::Rocm) {
                        println!(
                            "WARNING: AMD GPU requested but no AMD ORT feature enabled, using CPU"
                        );
                    }
                }

                // Auto selection priority is ROCm > CUDA > CPU. OpenVINO is not
                // included in Auto because OpenVINO deployments require an explicit
                // `OpenVino` request. On a CUDA build the ROCm block above is
                // compiled out, so CUDA is tried next.
                #[cfg(feature = "cuda")]
                {
                    if matches!(provider, ClassifierExecutionProvider::Auto) {
                        use ort::execution_providers::{
                            ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                        };
                        let mem_limit = crate::core::gpu_memory::get_gpu_mem_limit();
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
                                println!(
                                    "INFO: Using CUDA execution provider (NVIDIA GPU) — verified"
                                );
                                return Ok(session);
                            }
                            Err(e) => {
                                println!("WARNING: CUDA EP failed: {}, falling back to CPU", e);
                            }
                        }
                    }
                }

                println!("INFO: Using CPU execution provider");
                Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))
            }
            ClassifierExecutionProvider::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    use ort::execution_providers::{
                        ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                    };
                    let mem_limit = crate::core::gpu_memory::get_gpu_mem_limit();
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
                        Err(e) => {
                            println!("WARNING: CUDA EP failed: {}, falling back to CPU", e);
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    println!("WARNING: CUDA requested but 'cuda' feature not enabled, using CPU");
                }

                println!("INFO: Using CPU execution provider");
                Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))
            }
            ClassifierExecutionProvider::OpenVino => {
                #[cfg(feature = "openvino")]
                {
                    use ort::execution_providers::OpenVINOExecutionProvider;
                    match Session::builder()
                        .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                        .with_execution_providers([OpenVINOExecutionProvider::default()
                            .build()
                            .error_on_failure()])
                        .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                    {
                        Ok(session) => {
                            println!("INFO: Using OpenVINO execution provider (Intel) — verified");
                            return Ok(session);
                        }
                        Err(e) => {
                            println!("WARNING: OpenVINO EP failed: {}, falling back to CPU", e);
                        }
                    }
                }
                #[cfg(not(feature = "openvino"))]
                {
                    println!(
                        "WARNING: OpenVINO requested but 'openvino' feature not enabled, using CPU"
                    );
                }

                println!("INFO: Using CPU execution provider");
                Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()))
            }
        }
    }

    /// Classify a single text
    pub fn classify(&mut self, text: &str) -> UnifiedResult<ClassificationResult> {
        let results = self.classify_batch(&[text])?;
        Ok(results.into_iter().next().unwrap())
    }

    /// Classify multiple texts in batch
    pub fn classify_batch(&mut self, texts: &[&str]) -> UnifiedResult<Vec<ClassificationResult>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        // Find max sequence length; apply both the model's architectural limit and
        // the classification safety cap. The tokenizer already enforces
        // MAX_CLASSIFICATION_SEQ_LEN via truncation, but this second guard ensures
        // correctness even if the tokenizer is replaced or called without truncation.
        let dynamic_max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let dynamic_max_len = dynamic_max_len
            .min(self.config.max_position_embeddings)
            .min(MAX_CLASSIFICATION_SEQ_LEN);
        let max_len = self
            .input_len_buckets
            .iter()
            .copied()
            .find(|bucket| *bucket >= dynamic_max_len)
            .unwrap_or(dynamic_max_len)
            .max(dynamic_max_len)
            .min(self.config.max_position_embeddings)
            .min(MAX_CLASSIFICATION_SEQ_LEN);

        // Prepare input tensors
        let batch_size = texts.len();
        let mut input_ids = vec![self.config.pad_token_id as i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let seq_len = encoding.len().min(max_len);
            let enc_attention_mask = encoding.get_attention_mask();
            for j in 0..seq_len {
                input_ids[i * max_len + j] = encoding.get_ids()[j] as i64;
                // Use the tokenizer's attention mask to correctly handle padding tokens
                // (e.g. when tokenizer has Fixed padding strategy like Fixed:512)
                attention_mask[i * max_len + j] = enc_attention_mask[j] as i64;
            }
        }

        // Create tensors
        let input_ids_tensor = Tensor::from_array(([batch_size, max_len], input_ids))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids", &e.to_string()))?;

        let attention_mask_tensor = Tensor::from_array(([batch_size, max_len], attention_mask))
            .map_err(|e: ort::Error| {
                errors::inference_error("create_attention_mask", &e.to_string())
            })?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?;

        // Extract logits (inline to avoid borrow issues)
        let logits = extract_logits_from_outputs(&outputs)?;

        // Convert to results
        let results = logits_to_classification_results(&logits, &self.config);

        Ok(results)
    }

    fn warmup_if_requested(&mut self, onnx_path: &Path) {
        if self.input_len_buckets.is_empty() || !Self::sequence_warmup_enabled() {
            return;
        }

        let started = std::time::Instant::now();
        println!(
            "INFO: Warming AMD MIGraphX sequence-classifier artifact {}",
            onnx_path.display()
        );
        for bucket in self.input_len_buckets.clone() {
            let warmup_text = Self::warmup_text_for_bucket(bucket);
            if let Err(error) = self.classify_batch(&[warmup_text.as_str()]) {
                println!(
                    "WARNING: AMD MIGraphX sequence-classifier warmup failed for {} at bucket {}: {}",
                    onnx_path.display(),
                    bucket,
                    error
                );
                return;
            }
        }
        println!(
            "INFO: AMD MIGraphX sequence-classifier warmup completed for buckets {:?} in {:.3} ms",
            self.input_len_buckets,
            started.elapsed().as_secs_f64() * 1000.0
        );
    }

    fn warmup_text_for_bucket(bucket: usize) -> String {
        let repeated_terms = if bucket <= 128 { 16 } else { 180 };
        std::iter::repeat_n("semantic-router-migraphx-warmup", repeated_terms)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get model configuration
    pub fn config(&self) -> &MmBertClassifierConfig {
        &self.config
    }

    /// Get model info string
    pub fn model_info(&self) -> String {
        format!(
            "MmBertSequenceClassifier(path={}, num_labels={}, labels={:?})",
            self.model_path,
            self.config.num_labels,
            self.config.id2label.values().collect::<Vec<_>>()
        )
    }
}

// ============================================================================
// Helper Functions (standalone to avoid borrow issues)
// ============================================================================

/// Extract logits from model output
fn extract_logits_from_outputs(outputs: &SessionOutputs<'_>) -> UnifiedResult<Array2<f32>> {
    // Try common output names
    let output_names = ["logits", "output", "predictions"];

    for name in &output_names {
        if let Some(output_value) = outputs.get(*name) {
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let flat: Vec<f32> = data.to_vec();
                    return Array2::from_shape_vec((dims[0], dims[1]), flat)
                        .map_err(|e| errors::inference_error("reshape_logits", &e.to_string()));
                }
            }
            // FP16 models (e.g. model_sdpa_fp16.onnx on AMD) can emit f16 logits.
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f16>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                if dims.len() == 2 {
                    let flat: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                    return Array2::from_shape_vec((dims[0], dims[1]), flat)
                        .map_err(|e| errors::inference_error("reshape_logits", &e.to_string()));
                }
            }
        }
    }

    // Try first output
    if let Some((_, output_value)) = outputs.iter().next() {
        if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            if dims.len() == 2 {
                let flat: Vec<f32> = data.to_vec();
                return Array2::from_shape_vec((dims[0], dims[1]), flat)
                    .map_err(|e| errors::inference_error("reshape_logits", &e.to_string()));
            }
        }
        if let Ok((shape, data)) = output_value.try_extract_tensor::<f16>() {
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            if dims.len() == 2 {
                let flat: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                return Array2::from_shape_vec((dims[0], dims[1]), flat)
                    .map_err(|e| errors::inference_error("reshape_logits", &e.to_string()));
            }
        }
    }

    Err(errors::inference_error(
        "extract_logits",
        "Failed to extract logits",
    ))
}

/// Extract token-level logits from model output
fn extract_token_logits_from_outputs(outputs: &SessionOutputs<'_>) -> UnifiedResult<Array2<f32>> {
    fn reshape_token_logits(dims: &[usize], flat: Vec<f32>) -> UnifiedResult<Array2<f32>> {
        let mut squeezed = dims.to_vec();
        // Drop singleton dimensions around the tensor (e.g. [1, 1, seq, num] or [1, seq, num, 1]).
        while squeezed.len() > 2 && squeezed.first() == Some(&1) {
            squeezed.remove(0);
        }
        while squeezed.len() > 2 && squeezed.last() == Some(&1) {
            squeezed.pop();
        }

        match squeezed.as_slice() {
            // [seq_len, num_labels]
            [seq_len, num_labels] => {
                let expected = seq_len.saturating_mul(*num_labels);
                if flat.len() < expected {
                    return Err(errors::inference_error(
                        "reshape_token_logits",
                        &format!(
                            "tensor too small for shape {:?}: data_len={}, expected={}",
                            squeezed,
                            flat.len(),
                            expected
                        ),
                    ));
                }
                Array2::from_shape_vec(
                    (*seq_len, *num_labels),
                    flat.into_iter().take(expected).collect(),
                )
                .map_err(|e| errors::inference_error("reshape_token_logits", &e.to_string()))
            }
            // [batch, seq_len, num_labels] or [seq_len, 1, num_labels]
            [a, b, c] => {
                if *b == 1 && *a > 1 {
                    // [seq_len, 1, num_labels]
                    let seq_len = *a;
                    let num_labels = *c;
                    let expected = seq_len.saturating_mul(num_labels);
                    if flat.len() < expected {
                        return Err(errors::inference_error(
                            "reshape_token_logits",
                            &format!(
                                "tensor too small for shape {:?}: data_len={}, expected={}",
                                squeezed,
                                flat.len(),
                                expected
                            ),
                        ));
                    }
                    return Array2::from_shape_vec(
                        (seq_len, num_labels),
                        flat.into_iter().take(expected).collect(),
                    )
                    .map_err(|e| errors::inference_error("reshape_token_logits", &e.to_string()));
                }

                // Treat as [batch, seq_len, num_labels], keep first batch slice.
                let seq_len = *b;
                let num_labels = *c;
                let per_batch = seq_len.saturating_mul(num_labels);
                if flat.len() < per_batch {
                    return Err(errors::inference_error(
                        "reshape_token_logits",
                        &format!(
                            "tensor too small for shape {:?}: data_len={}, expected_at_least={}",
                            squeezed,
                            flat.len(),
                            per_batch
                        ),
                    ));
                }
                Array2::from_shape_vec(
                    (seq_len, num_labels),
                    flat.into_iter().take(per_batch).collect(),
                )
                .map_err(|e| errors::inference_error("reshape_token_logits", &e.to_string()))
            }
            _ => Err(errors::inference_error(
                "reshape_token_logits",
                &format!("unsupported token logits shape: {:?}", dims),
            )),
        }
    }

    let output_names = [
        "logits",
        "output",
        "predictions",
        "output_0",
        "token_logits",
    ];
    let mut inspected_shapes: Vec<String> = Vec::new();

    macro_rules! try_output {
        ($output_name:expr, $output_value:expr) => {{
            if let Ok((shape, data)) = $output_value.try_extract_tensor::<f32>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                inspected_shapes.push(format!("{}:f32{:?}", $output_name, dims));
                if let Ok(arr) = reshape_token_logits(&dims, data.to_vec()) {
                    return Ok(arr);
                }
            }

            if let Ok((shape, data)) = $output_value.try_extract_tensor::<f16>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
                inspected_shapes.push(format!("{}:f16{:?}", $output_name, dims));
                let flat: Vec<f32> = data.iter().map(|v| v.to_f32()).collect();
                if let Ok(arr) = reshape_token_logits(&dims, flat) {
                    return Ok(arr);
                }
            }
        }};
    }

    // First try commonly used output names.
    for name in &output_names {
        if let Some(output_value) = outputs.get(*name) {
            try_output!(*name, output_value);
        }
    }

    // Then try all outputs (some exported ONNX models use non-standard names).
    for (name, output_value) in outputs.iter() {
        try_output!(name, output_value);
    }

    let detail = if inspected_shapes.is_empty() {
        "Failed to extract token logits: no f32/f16 tensor outputs were found".to_string()
    } else {
        format!(
            "Failed to extract token logits; inspected outputs: {}",
            inspected_shapes.join(", ")
        )
    };

    Err(errors::inference_error("extract_token_logits", &detail))
}

/// Convert logits to classification results
fn logits_to_classification_results(
    logits: &Array2<f32>,
    config: &MmBertClassifierConfig,
) -> Vec<ClassificationResult> {
    let mut results = Vec::with_capacity(logits.nrows());

    for row in logits.rows() {
        // Softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        // Find max (NaN-safe: treat NaN as less than any value)
        let (class_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .unwrap();

        let label = config.get_label(class_id as i32);

        results.push(ClassificationResult {
            label,
            class_id: class_id as i32,
            confidence,
            probabilities: probs,
        });
    }

    results
}

// ============================================================================
// Token Classification Model (PII Detection)
// ============================================================================

/// mmBERT Token Classification Model
///
/// Used for PII detection with BIO tagging
pub struct MmBertTokenClassifier {
    session: Session,
    tokenizer: Arc<Tokenizer>,
    config: MmBertClassifierConfig,
    model_path: String,
    pad_to_max_length: bool,
}

impl MmBertTokenClassifier {
    /// Load token classifier from directory
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();

        let config = MmBertClassifierConfig::from_pretrained(&model_path)?;

        let tokenizer_path = model_path.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(errors::file_not_found(
                &tokenizer_path.display().to_string(),
            ));
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_CLASSIFICATION_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let onnx_candidates =
            MmBertSequenceClassifier::find_token_onnx_models(&model_path, provider)?;
        let (session, onnx_path) = MmBertSequenceClassifier::create_session_with_fallback(
            onnx_candidates,
            provider,
            &model_path_str,
            BaselineAmdPolicy::CpuOnly,
        )?;
        println!(
            "INFO: Selected token-classifier ONNX file: {}",
            onnx_path.display()
        );
        let pad_to_max_length =
            MmBertSequenceClassifier::is_token_optimized_onnx_artifact(&onnx_path);

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path_str,
            pad_to_max_length,
        })
    }

    /// Detect PII entities in text
    pub fn detect_entities(&mut self, text: &str) -> UnifiedResult<TokenClassificationResult> {
        // Tokenize with offsets
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let seq_len = encoding
            .len()
            .min(self.config.max_position_embeddings)
            .min(MAX_CLASSIFICATION_SEQ_LEN);

        let input_len = if self.pad_to_max_length {
            MAX_CLASSIFICATION_SEQ_LEN
                .min(self.config.max_position_embeddings)
                .max(seq_len)
        } else {
            seq_len
        };

        // Prepare inputs. Token-specific optimized artifacts are exported with a
        // fixed 512-token contract for MIGraphX, so they receive padded inputs.
        let mut input_ids = vec![self.config.pad_token_id as i64; input_len];
        let mut attention_mask = vec![0i64; input_len];
        let enc_attention_mask = encoding.get_attention_mask();

        for i in 0..seq_len {
            input_ids[i] = encoding.get_ids()[i] as i64;
            // Use the tokenizer's attention mask to correctly handle padding tokens
            attention_mask[i] = enc_attention_mask[i] as i64;
        }

        // Create tensors (batch size 1)
        let input_ids_tensor = Tensor::from_array(([1, input_len], input_ids))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids", &e.to_string()))?;

        let attention_mask_tensor =
            Tensor::from_array(([1, input_len], attention_mask)).map_err(|e: ort::Error| {
                errors::inference_error("create_attention_mask", &e.to_string())
            })?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e: ort::Error| errors::inference_error("session_run", &e.to_string()))?;

        // Extract token logits [1, seq_len, num_labels]
        let token_logits = extract_token_logits_from_outputs(&outputs)?;
        if token_logits.nrows() < seq_len {
            return Err(errors::inference_error(
                "extract_token_logits",
                &format!(
                    "token logits length {} is shorter than tokenizer sequence length {}; selected artifact is not compatible with token classification",
                    token_logits.nrows(),
                    seq_len
                ),
            ));
        }

        // Convert to entities using BIO scheme
        let entities = bio_decode_entities(text, &encoding, &token_logits, &self.config)?;

        Ok(TokenClassificationResult { entities })
    }

    /// Get model info
    pub fn model_info(&self) -> String {
        format!(
            "MmBertTokenClassifier(path={}, num_labels={})",
            self.model_path, self.config.num_labels
        )
    }
}

/// Decode BIO tags to entities (standalone function)
fn bio_decode_entities(
    text: &str,
    encoding: &tokenizers::Encoding,
    logits: &Array2<f32>,
    config: &MmBertClassifierConfig,
) -> UnifiedResult<Vec<DetectedEntity>> {
    let mut entities = Vec::new();
    let mut current_entity: Option<(String, usize, usize, f32)> = None;

    let offsets = encoding.get_offsets();

    for (i, row) in logits.rows().into_iter().enumerate() {
        // Skip special tokens (BOS, EOS, PAD)
        if i >= offsets.len() {
            break;
        }

        let (start, end) = offsets[i];
        if start == 0 && end == 0 {
            // Special token, skip
            continue;
        }

        // Softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let probs: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        // Get predicted label (NaN-safe: treat NaN as less than any value)
        let (label_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less))
            .unwrap();

        let label = config.get_label(label_id as i32);

        // Parse BIO tag
        if let Some(stripped) = label.strip_prefix("B-") {
            // Save current entity if any
            if let Some((entity_type, ent_start, ent_end, ent_conf)) = current_entity.take() {
                if ent_start < text.len() && ent_end <= text.len() {
                    entities.push(DetectedEntity {
                        text: text[ent_start..ent_end].to_string(),
                        entity_type,
                        start: ent_start,
                        end: ent_end,
                        confidence: ent_conf,
                    });
                }
            }

            // Start new entity
            let entity_type = stripped.to_string();
            current_entity = Some((entity_type, start, end, confidence));
        } else if let Some(stripped) = label.strip_prefix("I-") {
            // Continue current entity
            if let Some((ref entity_type, ent_start, _, ref mut ent_conf)) = current_entity {
                let expected_type = stripped;
                if entity_type == expected_type {
                    current_entity = Some((
                        entity_type.clone(),
                        ent_start,
                        end,
                        (*ent_conf + confidence) / 2.0,
                    ));
                }
            }
        } else {
            // O tag - save current entity if any
            if let Some((entity_type, ent_start, ent_end, ent_conf)) = current_entity.take() {
                if ent_start < text.len() && ent_end <= text.len() {
                    entities.push(DetectedEntity {
                        text: text[ent_start..ent_end].to_string(),
                        entity_type,
                        start: ent_start,
                        end: ent_end,
                        confidence: ent_conf,
                    });
                }
            }
        }
    }

    // Save final entity if any
    if let Some((entity_type, ent_start, ent_end, ent_conf)) = current_entity {
        if ent_start < text.len() && ent_end <= text.len() {
            entities.push(DetectedEntity {
                text: text[ent_start..ent_end].to_string(),
                entity_type,
                start: ent_start,
                end: ent_end,
                confidence: ent_conf,
            });
        }
    }

    Ok(entities)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = MmBertClassifierConfig::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.max_position_embeddings, 32768);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_attention_heads, 12);
        // vocab_size varies by model (256000 for mmBERT-32K)
        assert!(config.vocab_size > 0);
    }

    #[test]
    fn test_get_label() {
        let mut config = MmBertClassifierConfig::default();
        config.id2label.insert(0, "BENIGN".to_string());
        config.id2label.insert(1, "JAILBREAK".to_string());

        assert_eq!(config.get_label(0), "BENIGN");
        assert_eq!(config.get_label(1), "JAILBREAK");
        assert_eq!(config.get_label(99), "LABEL_99");
    }

    #[test]
    fn test_classification_result_creation() {
        let result = ClassificationResult {
            label: "positive".to_string(),
            class_id: 1,
            confidence: 0.95,
            probabilities: vec![0.05, 0.95],
        };

        assert_eq!(result.label, "positive");
        assert_eq!(result.class_id, 1);
        assert!((result.confidence - 0.95).abs() < 0.001);
        assert_eq!(result.probabilities.len(), 2);
    }

    #[test]
    fn test_detected_entity_creation() {
        let entity = DetectedEntity {
            text: "john@example.com".to_string(),
            entity_type: "EMAIL".to_string(),
            start: 10,
            end: 26,
            confidence: 0.99,
        };

        assert_eq!(entity.text, "john@example.com");
        assert_eq!(entity.entity_type, "EMAIL");
        assert_eq!(entity.start, 10);
        assert_eq!(entity.end, 26);
        assert!((entity.confidence - 0.99).abs() < 0.001);
    }

    #[test]
    fn test_token_classification_result() {
        let result = TokenClassificationResult {
            entities: vec![
                DetectedEntity {
                    text: "123-45-6789".to_string(),
                    entity_type: "US_SSN".to_string(),
                    start: 0,
                    end: 11,
                    confidence: 0.98,
                },
                DetectedEntity {
                    text: "test@test.com".to_string(),
                    entity_type: "EMAIL".to_string(),
                    start: 20,
                    end: 33,
                    confidence: 0.95,
                },
            ],
        };

        assert_eq!(result.entities.len(), 2);
        assert_eq!(result.entities[0].entity_type, "US_SSN");
        assert_eq!(result.entities[1].entity_type, "EMAIL");
    }

    #[test]
    fn test_classifier_execution_provider_cpu() {
        let provider = ClassifierExecutionProvider::Cpu;
        // Should always be valid
        assert!(matches!(provider, ClassifierExecutionProvider::Cpu));
    }

    #[test]
    fn test_classifier_execution_provider_auto() {
        let provider = ClassifierExecutionProvider::Auto;
        assert!(matches!(provider, ClassifierExecutionProvider::Auto));
    }

    #[test]
    fn test_classifier_cpu_prefers_baseline_onnx_artifact() {
        let candidates = MmBertSequenceClassifier::onnx_candidate_filenames(
            ClassifierExecutionProvider::Cpu,
            true,
        );

        assert_eq!(
            candidates,
            vec![
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
                "model_sdpa_fp16.onnx",
            ]
        );
    }

    #[test]
    fn test_classifier_rocm_prefers_sdpa_artifact() {
        let candidates = MmBertSequenceClassifier::onnx_candidate_filenames(
            ClassifierExecutionProvider::Rocm,
            false,
        );

        assert_eq!(
            candidates,
            vec![
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
            ]
        );
    }

    #[test]
    fn test_classifier_rocm_prefers_ck_fa_artifact_when_enabled() {
        let candidates = MmBertSequenceClassifier::onnx_candidate_filenames(
            ClassifierExecutionProvider::Rocm,
            true,
        );

        assert_eq!(
            candidates,
            vec![
                "model_sdpa_fp16.onnx",
                "model_fa_fp16.onnx",
                "model_fa.onnx",
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
            ]
        );
    }

    #[test]
    fn test_classifier_migraphx_first_prefers_migraphx_rewritten_sdpa_artifact() {
        let candidates = MmBertSequenceClassifier::onnx_candidate_filenames_with_order(
            ClassifierExecutionProvider::Rocm,
            false,
            Some("migraphx-first"),
        );

        assert_eq!(
            candidates,
            vec![
                "model_sdpa_migraphx.onnx",
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
            ]
        );
    }

    #[test]
    fn test_sequence_baseline_artifact_uses_cpu_on_amd() {
        let provider = MmBertSequenceClassifier::session_provider_for_candidate(
            Path::new("/models/intent/onnx/model.onnx"),
            ClassifierExecutionProvider::Rocm,
            BaselineAmdPolicy::CpuOnly,
        );

        assert!(matches!(provider, ClassifierExecutionProvider::Cpu));
    }

    #[test]
    fn test_sequence_sdpa_artifact_uses_amd_on_amd() {
        let provider = MmBertSequenceClassifier::session_provider_for_candidate(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            ClassifierExecutionProvider::Rocm,
            BaselineAmdPolicy::CpuOnly,
        );

        assert!(matches!(provider, ClassifierExecutionProvider::Rocm));
    }

    #[test]
    fn test_sequence_sdpa_artifact_does_not_bucket_by_default_on_amd() {
        let config = MmBertClassifierConfig::default();
        let buckets = MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            ClassifierExecutionProvider::Rocm,
            &config,
            None,
            None,
        );

        assert!(buckets.is_empty());
    }

    #[test]
    fn test_sequence_sdpa_artifact_uses_buckets_for_migraphx_first_experiment() {
        let config = MmBertClassifierConfig::default();
        let buckets = MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            ClassifierExecutionProvider::Rocm,
            &config,
            None,
            Some("migraphx-first"),
        );

        assert_eq!(buckets, vec![64, 128, MAX_CLASSIFICATION_SEQ_LEN]);
    }

    #[test]
    fn test_sequence_buckets_can_be_disabled() {
        let config = MmBertClassifierConfig::default();
        let buckets = MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            ClassifierExecutionProvider::Rocm,
            &config,
            Some("dynamic"),
            Some("migraphx-first"),
        );

        assert!(buckets.is_empty());
    }

    #[test]
    fn test_sequence_buckets_accept_operator_override() {
        let config = MmBertClassifierConfig::default();
        let buckets = MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            ClassifierExecutionProvider::Rocm,
            &config,
            Some("64, 512, 512, 9999"),
            Some("migraphx-first"),
        );

        assert_eq!(buckets, vec![64, MAX_CLASSIFICATION_SEQ_LEN]);
    }

    #[test]
    fn test_sequence_buckets_do_not_apply_to_cpu_or_ck_fa() {
        let config = MmBertClassifierConfig::default();

        assert_eq!(
            MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
                Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
                ClassifierExecutionProvider::Cpu,
                &config,
                None,
                Some("migraphx-first"),
            ),
            Vec::<usize>::new()
        );
        assert_eq!(
            MmBertSequenceClassifier::sequence_input_len_buckets_for_artifact_with_order(
                Path::new("/models/intent/onnx/model_fa_fp16.onnx"),
                ClassifierExecutionProvider::Rocm,
                &config,
                None,
                Some("migraphx-first"),
            ),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn test_sequence_sdpa_artifact_prefers_rocm_by_default() {
        let preference = MmBertSequenceClassifier::amd_provider_preference_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            None,
        );

        assert_eq!(
            preference,
            crate::core::ort_migraphx::AmdProviderPreference::RocmFirst
        );
    }

    #[test]
    fn test_sequence_sdpa_artifact_can_opt_into_migraphx_first() {
        let preference = MmBertSequenceClassifier::amd_provider_preference_for_artifact_with_order(
            Path::new("/models/intent/onnx/model_sdpa_fp16.onnx"),
            Some("migraphx-first"),
        );

        assert_eq!(
            preference,
            crate::core::ort_migraphx::AmdProviderPreference::MigraphxFirst
        );
    }

    #[test]
    fn test_token_classifier_token_specific_artifact_uses_cpu_without_experimental_opt_in() {
        let provider = MmBertSequenceClassifier::session_provider_for_candidate(
            Path::new("/models/pii/onnx/model_token_sdpa.onnx"),
            ClassifierExecutionProvider::Rocm,
            BaselineAmdPolicy::CpuOnly,
        );

        assert!(matches!(provider, ClassifierExecutionProvider::Cpu));
    }

    #[test]
    fn test_token_artifacts_require_attention_mlir_workaround_for_migraphx() {
        assert!(!MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            true, None, None
        ));
        assert!(!MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            true,
            Some("attention,pointwise"),
            None
        ));
        assert!(MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            true,
            Some("dot, ~attention"),
            None
        ));
        assert!(MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            true,
            None,
            Some("1")
        ));
    }

    #[test]
    fn test_token_artifacts_still_require_experimental_opt_in() {
        assert!(!MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            false,
            Some("~attention"),
            None
        ));
        assert!(!MmBertSequenceClassifier::token_artifacts_migraphx_enabled(
            false,
            None,
            Some("1")
        ));
    }

    #[test]
    fn test_token_classifier_candidates_do_not_reuse_sequence_sdpa_artifact() {
        let candidates = MmBertSequenceClassifier::token_onnx_candidate_filenames(
            ClassifierExecutionProvider::Cpu,
            false,
        );

        assert_eq!(candidates[0], "model.onnx");
        assert!(!candidates.contains(&"model_sdpa_fp16.onnx"));
        assert!(!candidates.contains(&"model_fa_fp16.onnx"));
    }

    #[test]
    fn test_token_classifier_amd_keeps_baseline_before_experimental_artifact_by_default() {
        let candidates = MmBertSequenceClassifier::token_onnx_candidate_filenames(
            ClassifierExecutionProvider::Rocm,
            false,
        );

        assert_eq!(candidates[0], "model.onnx");
        assert!(
            candidates
                .iter()
                .position(|candidate| *candidate == "model_token_sdpa.onnx")
                > candidates
                    .iter()
                    .position(|candidate| *candidate == "model.onnx")
        );
        assert!(
            candidates
                .iter()
                .position(|candidate| *candidate == "model_token_eager.onnx")
                > candidates
                    .iter()
                    .position(|candidate| *candidate == "model.onnx")
        );
        assert!(!candidates.contains(&"model_sdpa_fp16.onnx"));
    }

    #[test]
    fn test_token_classifier_amd_prefers_ck_fa_artifact_when_available() {
        let candidates = MmBertSequenceClassifier::token_onnx_candidate_filenames(
            ClassifierExecutionProvider::Rocm,
            true,
        );

        assert_eq!(candidates[0], "model_fa_fp16.onnx");
        assert_eq!(candidates[1], "model_fa.onnx");
        assert!(!candidates.contains(&"model_sdpa_fp16.onnx"));
    }

    #[test]
    fn test_token_classifier_baseline_artifact_uses_cpu_on_amd() {
        let provider = MmBertSequenceClassifier::session_provider_for_candidate(
            Path::new("/models/pii/onnx/model.onnx"),
            ClassifierExecutionProvider::Rocm,
            BaselineAmdPolicy::CpuOnly,
        );

        assert!(matches!(provider, ClassifierExecutionProvider::Cpu));
    }

    #[test]
    fn test_token_optimized_artifact_requires_fixed_padding() {
        assert!(MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/model_token_sdpa.onnx")
        ));
        assert!(MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/token_classifier_sdpa.onnx")
        ));
        assert!(MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/token_classifier_sdpa_fp16.onnx")
        ));
        assert!(MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/model_token_eager.onnx")
        ));
        assert!(MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/token_classifier_eager_fp16.onnx")
        ));
        assert!(!MmBertSequenceClassifier::is_token_optimized_onnx_artifact(
            Path::new("/models/pii/onnx/model.onnx")
        ));
    }

    #[test]
    fn test_sequence_optimized_artifact_is_skipped_for_token_fallback() {
        assert!(
            MmBertSequenceClassifier::is_sequence_optimized_onnx_artifact(Path::new(
                "/models/pii/onnx/model_sdpa_fp16.onnx"
            ))
        );
        assert!(
            MmBertSequenceClassifier::is_sequence_optimized_onnx_artifact(Path::new(
                "/models/pii/onnx/model_fa_fp16.onnx"
            ))
        );
        assert!(
            !MmBertSequenceClassifier::is_sequence_optimized_onnx_artifact(Path::new(
                "/models/pii/onnx/model.onnx"
            ))
        );
    }

    #[test]
    fn test_config_with_labels() {
        let mut config = MmBertClassifierConfig {
            num_labels: 3,
            ..Default::default()
        };
        config.id2label.insert(0, "negative".to_string());
        config.id2label.insert(1, "neutral".to_string());
        config.id2label.insert(2, "positive".to_string());
        config.label2id.insert("negative".to_string(), 0);
        config.label2id.insert("neutral".to_string(), 1);
        config.label2id.insert("positive".to_string(), 2);

        assert_eq!(config.num_labels, 3);
        assert_eq!(config.id2label.len(), 3);
        assert_eq!(config.label2id.len(), 3);
        assert_eq!(config.get_label(0), "negative");
        assert_eq!(config.get_label(2), "positive");
    }

    #[test]
    fn test_classification_result_clone() {
        let result = ClassificationResult {
            label: "test".to_string(),
            class_id: 0,
            confidence: 0.8,
            probabilities: vec![0.8, 0.2],
        };

        let cloned = result.clone();
        assert_eq!(cloned.label, result.label);
        assert_eq!(cloned.class_id, result.class_id);
        assert_eq!(cloned.confidence, result.confidence);
        assert_eq!(cloned.probabilities, result.probabilities);
    }

    #[test]
    fn test_detected_entity_clone() {
        let entity = DetectedEntity {
            text: "test".to_string(),
            entity_type: "ORG".to_string(),
            start: 0,
            end: 4,
            confidence: 0.9,
        };

        let cloned = entity.clone();
        assert_eq!(cloned.text, entity.text);
        assert_eq!(cloned.entity_type, entity.entity_type);
        assert_eq!(cloned.start, entity.start);
        assert_eq!(cloned.end, entity.end);
        assert_eq!(cloned.confidence, entity.confidence);
    }

    // =========================================================================
    // Long-prompt truncation tests
    //
    // These tests verify that the tokenizer enforces MAX_CLASSIFICATION_SEQ_LEN
    // (512) regardless of raw input length, preventing the quadratic global-
    // attention OOM that was observed at ~4 000 tokens (GitHub issue #1843).
    //
    // The tests load tokenizer.json from the mmBERT-32K model on disk and are
    // therefore skipped automatically when running in CI (where model files are
    // not present) by checking the `CI` environment variable.
    // =========================================================================

    /// Return the path to the mmBERT-32K ONNX tokenizer, relative to the
    /// `onnx-binding` workspace root.
    fn mmbert_onnx_tokenizer_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("models/mmbert32k-intent-classifier-merged/onnx/tokenizer.json")
    }

    /// Load the long-prompt fixture JSON bundled in `test_data/`.
    fn load_long_prompt_fixtures() -> serde_json::Value {
        let fixture_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/long_prompt_fixtures.json");
        let raw = std::fs::read_to_string(&fixture_path)
            .unwrap_or_else(|_| panic!("fixture not found at {:?}", fixture_path));
        serde_json::from_str(&raw).expect("invalid fixture JSON")
    }

    /// Helper: tokenize `text` with the given `Tokenizer` and return the
    /// token count after the tokenizer's internal truncation has been applied.
    fn token_count_after_encode(tokenizer: &tokenizers::Tokenizer, text: &str) -> usize {
        tokenizer
            .encode(text, true)
            .expect("encode failed")
            .get_ids()
            .len()
    }

    /// Confirm the constant value that guards against OOM.
    #[test]
    fn test_max_classification_seq_len_is_512() {
        assert_eq!(
            MAX_CLASSIFICATION_SEQ_LEN, 512,
            "MAX_CLASSIFICATION_SEQ_LEN must be 512 to prevent quadratic-attention OOM"
        );
    }

    /// Verify that MmBertSequenceClassifier::load configures the tokenizer so
    /// that a ~4 000-token prompt is truncated to ≤ 512 tokens.
    ///
    /// Skipped in CI (`CI` env var set) because model files are not present.
    #[test]
    fn test_onnx_tokenizer_truncates_long_prompt() {
        if std::env::var("CI").is_ok() {
            eprintln!(
                "skipping test_onnx_tokenizer_truncates_long_prompt: CI environment detected"
            );
            return;
        }

        let tokenizer_path = mmbert_onnx_tokenizer_path();
        if !tokenizer_path.exists() {
            eprintln!(
                "skipping test_onnx_tokenizer_truncates_long_prompt: tokenizer not found at {:?}",
                tokenizer_path
            );
            return;
        }

        let fixtures = load_long_prompt_fixtures();
        let prompts = fixtures["prompts"].as_array().expect("prompts array");

        // Load and configure the tokenizer the same way MmBertSequenceClassifier::load does.
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("failed to load tokenizer");
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_CLASSIFICATION_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .expect("truncation config failed");

        for prompt in prompts {
            let id = prompt["id"].as_str().unwrap_or("?");
            let text = prompt["text"].as_str().expect("text field");
            let approx_untruncated = prompt["approx_tokens_untruncated"].as_u64().unwrap_or(0);

            let count = token_count_after_encode(&tokenizer, text);

            assert!(
                count <= MAX_CLASSIFICATION_SEQ_LEN,
                "prompt '{}' produced {} tokens (approx untruncated: {}); expected ≤ {}",
                id,
                count,
                approx_untruncated,
                MAX_CLASSIFICATION_SEQ_LEN,
            );
        }
    }

    /// Verify that a prompt long enough to OOM the model (≥ 4 000 raw tokens)
    /// does NOT reach the model with more than MAX_CLASSIFICATION_SEQ_LEN tokens.
    ///
    /// This is the direct regression test for GitHub issue #1843.
    ///
    /// Skipped in CI.
    #[test]
    fn test_onnx_4k_prompt_truncated_to_safe_length() {
        if std::env::var("CI").is_ok() {
            eprintln!("skipping test_onnx_4k_prompt_truncated_to_safe_length: CI environment");
            return;
        }

        let tokenizer_path = mmbert_onnx_tokenizer_path();
        if !tokenizer_path.exists() {
            eprintln!(
                "skipping test_onnx_4k_prompt_truncated_to_safe_length: tokenizer not found at {:?}",
                tokenizer_path
            );
            return;
        }

        let fixtures = load_long_prompt_fixtures();
        let long_prompt = fixtures["prompts"]
            .as_array()
            .and_then(|p| p.iter().find(|x| x["id"] == "long_4k"))
            .expect("long_4k fixture missing");

        let text = long_prompt["text"].as_str().expect("text");
        let approx_raw = long_prompt["approx_tokens_untruncated"]
            .as_u64()
            .unwrap_or(0);

        // Confirm the fixture is actually long enough to trigger the OOM without the fix.
        assert!(
            approx_raw > 2048,
            "fixture too short ({} est. tokens); update long_prompt_fixtures.json",
            approx_raw
        );

        // Configure tokenizer as the classifier does.
        let mut tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_CLASSIFICATION_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .expect("truncation config");

        let count = token_count_after_encode(&tokenizer, text);
        assert_eq!(
            count, MAX_CLASSIFICATION_SEQ_LEN,
            "4k prompt must produce exactly MAX_CLASSIFICATION_SEQ_LEN={} tokens after truncation, got {}",
            MAX_CLASSIFICATION_SEQ_LEN, count
        );
    }

    /// Verify that a short prompt is NOT over-truncated: a normal user message
    /// must keep all its tokens.
    ///
    /// Skipped in CI.
    #[test]
    fn test_onnx_short_prompt_not_truncated() {
        if std::env::var("CI").is_ok() {
            eprintln!("skipping test_onnx_short_prompt_not_truncated: CI environment");
            return;
        }

        let tokenizer_path = mmbert_onnx_tokenizer_path();
        if !tokenizer_path.exists() {
            eprintln!(
                "skipping test_onnx_short_prompt_not_truncated: tokenizer not found at {:?}",
                tokenizer_path
            );
            return;
        }

        let fixtures = load_long_prompt_fixtures();
        let short = fixtures["prompts"]
            .as_array()
            .and_then(|p| p.iter().find(|x| x["id"] == "short_baseline"))
            .expect("short_baseline fixture missing");

        let text = short["text"].as_str().expect("text");

        // Tokenizer without truncation limit to get the "true" token count.
        let baseline_tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
        let baseline_count = token_count_after_encode(&baseline_tokenizer, text);

        // Tokenizer with the production truncation limit.
        let mut truncating_tokenizer =
            tokenizers::Tokenizer::from_file(&tokenizer_path).expect("load tokenizer");
        truncating_tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_CLASSIFICATION_SEQ_LEN,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .expect("truncation config");
        let truncated_count = token_count_after_encode(&truncating_tokenizer, text);

        assert_eq!(
            baseline_count, truncated_count,
            "short prompt ({} tokens) should not be truncated by MAX_CLASSIFICATION_SEQ_LEN={}",
            baseline_count, MAX_CLASSIFICATION_SEQ_LEN
        );
        assert!(
            truncated_count < MAX_CLASSIFICATION_SEQ_LEN,
            "short prompt unexpectedly long: {} tokens",
            truncated_count
        );
    }
}
