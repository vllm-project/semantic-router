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

use crate::core::instance_options::InstanceOptions;
use crate::core::unified_error::{errors, UnifiedResult};
use half::f16;
use ndarray::Array2;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokenizers::{
    PostProcessor, Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy,
};

/// Conservative default for existing callers. Longer contexts are an explicit
/// deployment choice: global attention retains quadratic compute, and memory
/// use depends on the selected ONNX graph and execution provider.
const MAX_CLASSIFICATION_SEQ_LEN: usize = 512;

fn classifier_context_length(capacity: usize, requested: Option<usize>) -> UnifiedResult<usize> {
    let limit = requested.unwrap_or(MAX_CLASSIFICATION_SEQ_LEN.min(capacity));
    if capacity == 0 || limit == 0 || limit > capacity {
        return Err(errors::config_error(
            "max_sequence_length",
            &format!("requested {limit} tokens; model capacity is {capacity}"),
        ));
    }
    Ok(limit)
}

fn configure_classifier_tokenizer(tokenizer: &mut Tokenizer, limit: usize) -> UnifiedResult<()> {
    let special_tokens = tokenizer
        .get_post_processor()
        .map_or(0, |processor| processor.added_tokens(false));
    // tokenizers subtracts this count before validating truncation parameters.
    // Reject undersized budgets before that usize subtraction can underflow.
    if limit < special_tokens {
        return Err(errors::config_error(
            "max_sequence_length",
            &format!(
                "requested {limit} tokens; tokenizer requires {special_tokens} special tokens"
            ),
        ));
    }
    // Batch padding is performed below, using the model's pad token and each
    // encoding's mask. Artifact-level fixed padding must not expand this budget.
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: limit,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }))
        .map_err(|e| errors::tokenization_error(&e.to_string()))?;
    Ok(())
}

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
                .unwrap_or(MAX_CLASSIFICATION_SEQ_LEN as u64)
                as usize,
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
    max_sequence_length: usize,
}

impl MmBertSequenceClassifier {
    /// Load classifier from directory
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Self> {
        Self::load_with_context(model_path, provider, None)
    }

    /// Load with an explicit input budget, including special tokens. The graph
    /// must support this length; callers must validate its memory and latency on
    /// their execution provider before choosing a larger deployment budget.
    pub fn load_with_max_sequence_length<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
        max_sequence_length: usize,
    ) -> UnifiedResult<Self> {
        Self::load_with_context(model_path, provider, Some(max_sequence_length))
    }

    fn load_with_context<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
        requested: Option<usize>,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();

        // Load configuration
        let config = MmBertClassifierConfig::from_pretrained(&model_path)?;
        let max_sequence_length =
            classifier_context_length(config.max_position_embeddings, requested)?;

        // Load tokenizer
        let tokenizer_path = model_path.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(errors::file_not_found(
                &tokenizer_path.display().to_string(),
            ));
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        configure_classifier_tokenizer(&mut tokenizer, max_sequence_length)?;

        // Find ONNX model candidates and initialize with fallback.
        let onnx_candidates = Self::find_onnx_models(&model_path, provider)?;
        let (session, onnx_path) =
            Self::create_session_with_fallback(onnx_candidates, provider, &model_path_str)?;
        println!(
            "INFO: Selected classifier ONNX file: {}",
            onnx_path.display()
        );

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path_str,
            max_sequence_length,
        })
    }

    /// Load one owned session with an explicit provider and no provider fallback.
    pub fn load_with_options(options: &InstanceOptions) -> UnifiedResult<Self> {
        options.validate()?;
        let config = MmBertClassifierConfig::from_pretrained(&options.model_path)?;
        let mut tokenizer =
            Tokenizer::from_file(Path::new(&options.model_path).join("tokenizer.json"))
                .map_err(|e| errors::tokenization_error(&e.to_string()))?;
        options.configure_tokenizer(
            &mut tokenizer,
            MAX_CLASSIFICATION_SEQ_LEN.min(config.max_position_embeddings),
        )?;
        // Owned sessions select the standard graph deterministically. Legacy
        // environment-driven FA ranking must not change an instance's identity.
        let provider = ClassifierExecutionProvider::Cpu;
        let candidates = if options.model_file.is_some() {
            vec![]
        } else {
            Self::find_onnx_models(&options.model_path, provider)?
        };
        let graph = options.select_graph(candidates)?;
        let session = options.create_session(&graph)?;
        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: options.model_path.clone(),
        })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn finish_profiling(&mut self) -> UnifiedResult<Vec<String>> {
        self.session
            .end_profiling()
            .map(|path| vec![path])
            .map_err(|e| errors::ort_error(&e.to_string()))
    }

    /// Find ONNX model candidates in priority order.
    fn find_onnx_models<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let dir = model_path.as_ref();
        let onnx_subdir = dir.join("onnx");
        let search_dirs = [dir, onnx_subdir.as_path()];

        // Prefer FA-optimized variant when CK Flash Attention is available.
        let has_fa = std::env::var("ORT_CK_FLASH_ATTN_LIB")
            .ok()
            .filter(|s| !s.is_empty())
            .is_some();
        // The CPU provider has no FP16 kernels for these operators and runs an
        // FP16 graph by casting around each one, so there it is the slowest
        // candidate rather than the fastest. Keep it last instead of dropping it,
        // so a model that publishes only that file still loads.
        let cpu_only = match provider {
            ClassifierExecutionProvider::Cpu => true,
            // Without a GPU feature compiled in, Auto can only resolve to CPU.
            ClassifierExecutionProvider::Auto => !cfg!(any(feature = "cuda", feature = "rocm")),
            _ => false,
        };
        let candidates: &[&str] = if cpu_only {
            &[
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
                "model_sdpa_fp16.onnx",
            ]
        } else if has_fa {
            &[
                "model_fa_fp16.onnx",
                "model_fa.onnx",
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
            ]
        } else {
            &[
                "model_sdpa_fp16.onnx",
                "model.onnx",
                "classifier.onnx",
                "model_optimized.onnx",
            ]
        };

        let mut results: Vec<std::path::PathBuf> = Vec::new();
        // Try known ONNX filenames first in both model root and `onnx/` subdirectory.
        for base_dir in search_dirs {
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

        // Fallback: include any .onnx file in both locations.
        for base_dir in search_dirs {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            if let Ok(entries) = std::fs::read_dir(base_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
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

    /// Create session from candidates with fallback across files.
    fn create_session_with_fallback(
        onnx_candidates: Vec<std::path::PathBuf>,
        provider: ClassifierExecutionProvider,
        model_path: &str,
    ) -> UnifiedResult<(Session, std::path::PathBuf)> {
        let mut last_error: Option<String> = None;
        for onnx_path in onnx_candidates {
            match Self::create_session(&onnx_path, provider) {
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
                #[cfg(feature = "rocm")]
                {
                    use crate::core::gpu_memory;
                    use ort::execution_providers::{
                        ArenaExtendStrategy, MIGraphXExecutionProvider, ROCmExecutionProvider,
                    };

                    let ck_fa_lib = std::env::var("ORT_CK_FLASH_ATTN_LIB")
                        .ok()
                        .filter(|s| !s.is_empty());
                    if let Some(ref lib) = ck_fa_lib {
                        println!("INFO: CK Flash Attention custom op library: {}", lib);
                    }

                    let maybe_register_custom_ops = |builder: ort::session::builder::SessionBuilder| -> Result<ort::session::builder::SessionBuilder, ort::Error> {
                        if let Some(ref lib) = ck_fa_lib {
                            builder.with_operator_library(lib)
                        } else {
                            Ok(builder)
                        }
                    };

                    match Session::builder()
                        .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                        .with_execution_providers([MIGraphXExecutionProvider::default()
                            .build()
                            .error_on_failure()])
                        .and_then(|b| maybe_register_custom_ops(b))
                        .and_then(|b| b.commit_from_file(onnx_path.as_ref()))
                    {
                        Ok(session) => {
                            println!(
                                "INFO: Using MIGraphX execution provider (AMD GPU) — verified"
                            );
                            return Ok(session);
                        }
                        Err(e) => {
                            println!("INFO: MIGraphX EP failed to register: {}", e);
                        }
                    }

                    let mem_limit = gpu_memory::get_gpu_mem_limit();
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
                        Err(e) => {
                            println!("INFO: ROCm EP failed to register: {}", e);
                        }
                    }

                    println!("WARNING: All GPU execution providers failed, falling back to CPU");
                }

                #[cfg(not(feature = "rocm"))]
                {
                    if matches!(provider, ClassifierExecutionProvider::Rocm) {
                        println!(
                            "WARNING: ROCm requested but 'rocm' feature not enabled, using CPU"
                        );
                    }
                }

                // Auto selection priority is ROCm > CUDA > CPU. OpenVINO is not
                // part of the Auto chain; it is only used for an explicit
                // `OpenVino` request. On a CUDA build the ROCm block above is
                // compiled out, so Auto must also try CUDA here before falling
                // back to CPU. Without this,
                // Auto silently ran every classifier (PII, jailbreak, intent,
                // factcheck) on CPU even on NVIDIA GPUs, because the dedicated
                // CUDA arm below is only reached for an explicit `Cuda` request.
                // Gated to `Auto` so an explicit `Rocm` request on a CUDA build
                // still falls through to CPU rather than silently using NVIDIA.
                #[cfg(feature = "cuda")]
                {
                    if matches!(provider, ClassifierExecutionProvider::Auto) {
                        use crate::core::gpu_memory;
                        use ort::execution_providers::{
                            ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                        };
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
                    use crate::core::gpu_memory;
                    use ort::execution_providers::{
                        ArenaExtendStrategy as CudaArenaStrategy, CUDAExecutionProvider,
                    };
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
                        Err(e) => {
                            println!("WARNING: CUDA EP failed: {}, falling back to CPU", e);
                        }
                    }
                }

                #[cfg(not(feature = "cuda"))]
                println!("WARNING: CUDA requested but 'cuda' feature not enabled, using CPU");

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
                println!(
                    "WARNING: OpenVINO requested but 'openvino' feature not enabled, using CPU"
                );

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

        // The tokenizer includes special tokens within the validated budget.
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let max_len = max_len
            .min(self.config.max_position_embeddings)
            .min(self.max_sequence_length);

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
        validate_classifier_logits(&logits, batch_size, self.config.num_labels)?;

        // Convert to results
        let results = logits_to_classification_results(&logits, &self.config);

        Ok(results)
    }

    /// Get model configuration
    pub fn config(&self) -> &MmBertClassifierConfig {
        &self.config
    }

    /// Effective input budget, including special tokens.
    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
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

fn validate_classifier_logits(
    logits: &Array2<f32>,
    rows: usize,
    labels: usize,
) -> UnifiedResult<()> {
    if logits.dim() != (rows, labels) || rows == 0 || labels == 0 {
        return Err(errors::inference_error(
            "validate_logits",
            &format!(
                "expected logits shape ({rows}, {labels}), got {:?}; check the exported model task",
                logits.dim()
            ),
        ));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(errors::inference_error(
            "validate_logits",
            "model produced non-finite logits; check graph precision and input context length",
        ));
    }
    Ok(())
}

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
    max_sequence_length: usize,
}

impl MmBertTokenClassifier {
    /// Load token classifier from directory
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
    ) -> UnifiedResult<Self> {
        Self::load_with_context(model_path, provider, None)
    }

    /// Load a token classifier with a validated, explicit input budget.
    pub fn load_with_max_sequence_length<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
        max_sequence_length: usize,
    ) -> UnifiedResult<Self> {
        Self::load_with_context(model_path, provider, Some(max_sequence_length))
    }

    fn load_with_context<P: AsRef<Path>>(
        model_path: P,
        provider: ClassifierExecutionProvider,
        requested: Option<usize>,
    ) -> UnifiedResult<Self> {
        let model_path_str = model_path.as_ref().display().to_string();

        let config = MmBertClassifierConfig::from_pretrained(&model_path)?;
        let max_sequence_length =
            classifier_context_length(config.max_position_embeddings, requested)?;

        let tokenizer_path = model_path.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(errors::file_not_found(
                &tokenizer_path.display().to_string(),
            ));
        }

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        configure_classifier_tokenizer(&mut tokenizer, max_sequence_length)?;

        let onnx_candidates = MmBertSequenceClassifier::find_onnx_models(&model_path, provider)?;
        let (session, onnx_path) = MmBertSequenceClassifier::create_session_with_fallback(
            onnx_candidates,
            provider,
            &model_path_str,
        )?;
        println!(
            "INFO: Selected token-classifier ONNX file: {}",
            onnx_path.display()
        );

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            model_path: model_path_str,
            max_sequence_length,
        })
    }

    /// Own a token-classification session independently of all legacy role slots.
    pub fn load_with_options(options: &InstanceOptions) -> UnifiedResult<Self> {
        let sequence = MmBertSequenceClassifier::load_with_options(options)?;
        Ok(Self {
            session: sequence.session,
            tokenizer: sequence.tokenizer,
            config: sequence.config,
            model_path: sequence.model_path,
        })
    }

    pub fn config(&self) -> &MmBertClassifierConfig {
        &self.config
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    pub fn finish_profiling(&mut self) -> UnifiedResult<Vec<String>> {
        self.session
            .end_profiling()
            .map(|path| vec![path])
            .map_err(|e| errors::ort_error(&e.to_string()))
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
            .min(self.max_sequence_length);

        // Prepare inputs
        let mut input_ids = vec![self.config.pad_token_id as i64; seq_len];
        let mut attention_mask = vec![0i64; seq_len];
        let enc_attention_mask = encoding.get_attention_mask();

        for i in 0..seq_len {
            input_ids[i] = encoding.get_ids()[i] as i64;
            // Use the tokenizer's attention mask to correctly handle padding tokens
            attention_mask[i] = enc_attention_mask[i] as i64;
        }

        // Create tensors (batch size 1)
        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids", &e.to_string()))?;

        let attention_mask_tensor =
            Tensor::from_array(([1, seq_len], attention_mask)).map_err(|e: ort::Error| {
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
        // A sequence-classification export can have the same label count but
        // only one row. It must not silently become an empty/successful PII scan.
        validate_classifier_logits(&token_logits, seq_len, self.config.num_labels)?;

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

    /// Effective input budget, including special tokens.
    pub fn max_sequence_length(&self) -> usize {
        self.max_sequence_length
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
    // Keep the confidence sum and token count so every token has equal weight.
    let mut current_entity: Option<(String, usize, usize, f32, usize)> = None;
    let finish_entity =
        |(entity_type, start, end, confidence_sum, count): (String, usize, usize, f32, usize)| {
            text.get(start..end).and_then(|entity_text| {
                let trimmed = entity_text.trim();
                if trimmed.is_empty() {
                    return None;
                }
                // Token offsets may include surrounding whitespace. Keep the
                // public value and its UTF-8 byte offsets on the same span.
                let start = start + entity_text.len() - entity_text.trim_start().len();
                Some(DetectedEntity {
                    text: trimmed.to_string(),
                    entity_type,
                    start,
                    end: start + trimmed.len(),
                    confidence: confidence_sum / count as f32,
                })
            })
        };

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

        let tag = label
            .strip_prefix("B-")
            .map(|entity_type| (entity_type, false))
            .or_else(|| {
                label
                    .strip_prefix("I-")
                    .map(|entity_type| (entity_type, true))
            });

        if let Some((entity_type, true)) = tag {
            if let Some((current_type, _, current_end, confidence_sum, count)) =
                current_entity.as_mut()
            {
                if current_type == entity_type {
                    *current_end = end;
                    *confidence_sum += confidence;
                    *count += 1;
                    continue;
                }
            }
        }

        entities.extend(current_entity.take().and_then(&finish_entity));
        // An orphan I-tag or a changed I-tag type starts a new entity. Trained
        // classifiers can emit these without a preceding B-tag; dropping them
        // loses valid PII spans, including entire email addresses.
        if let Some((entity_type, _)) = tag {
            current_entity = Some((entity_type.to_string(), start, end, confidence, 1));
        }
    }

    // Save final entity if any
    entities.extend(current_entity.and_then(finish_entity));

    Ok(entities)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_bio_fixture(
        text: &str,
        predictions: &[(usize, usize, &str, f32)],
    ) -> Vec<DetectedEntity> {
        let labels = [
            "O",
            "B-PERSON",
            "I-PERSON",
            "B-EMAIL_ADDRESS",
            "I-EMAIL_ADDRESS",
        ];
        let config = MmBertClassifierConfig {
            num_labels: labels.len(),
            id2label: labels
                .iter()
                .enumerate()
                .map(|(i, label)| (i as i32, label.to_string()))
                .collect(),
            ..Default::default()
        };
        let encoding = tokenizers::Encoding::from_tokens(
            predictions
                .iter()
                .enumerate()
                .map(|(i, &(start, end, _, _))| tokenizers::Token {
                    id: i as u32,
                    value: text[start..end].to_string(),
                    offsets: (start, end),
                })
                .collect(),
            0,
        );
        let mut logits = Array2::zeros((predictions.len(), labels.len()));
        for (i, &(_, _, label, confidence)) in predictions.iter().enumerate() {
            let label_id = labels
                .iter()
                .position(|candidate| *candidate == label)
                .unwrap();
            logits
                .row_mut(i)
                .fill(((1.0 - confidence) / (labels.len() - 1) as f32).ln());
            logits[(i, label_id)] = confidence.ln();
        }
        bio_decode_entities(text, &encoding, &logits, &config).unwrap()
    }

    #[test]
    fn bio_orphan_email_after_unicode_keeps_utf8_byte_offsets() {
        let text = "中文🙂 alice.smith@example.com";
        // The maintained PII checkpoint emits this all-I email pattern. Zero
        // offsets model BOS/EOS/PAD; their predictions must not create entities.
        let entities = decode_bio_fixture(
            text,
            &[
                (0, 0, "B-PERSON", 0.99),
                (0, 6, "O", 0.95),
                (6, 10, "O", 0.95),
                (10, 16, "I-EMAIL_ADDRESS", 0.40),
                (16, 17, "I-EMAIL_ADDRESS", 0.85),
                (17, 22, "I-EMAIL_ADDRESS", 0.99),
                (22, 23, "I-EMAIL_ADDRESS", 0.52),
                (23, 30, "I-EMAIL_ADDRESS", 0.86),
                (30, 31, "I-EMAIL_ADDRESS", 0.74),
                (31, 34, "I-EMAIL_ADDRESS", 0.92),
                (0, 0, "B-PERSON", 0.99),
                (0, 0, "I-PERSON", 0.99),
            ],
        );
        assert_eq!(entities.len(), 1);
        let entity = &entities[0];
        assert_eq!(entity.entity_type, "EMAIL_ADDRESS");
        assert_eq!(entity.text, "alice.smith@example.com");
        assert_eq!((entity.start, entity.end), (11, text.len()));
        assert_eq!(&text[entity.start..entity.end], entity.text);
        assert!((entity.confidence - 5.28 / 7.0).abs() < 1e-6);
    }

    #[test]
    fn bio_orphan_i_can_start_at_the_first_text_token() {
        let entities =
            decode_bio_fixture("李明", &[(0, 3, "I-PERSON", 0.9), (3, 6, "I-PERSON", 0.8)]);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "李明");
        assert_eq!((entities[0].start, entities[0].end), (0, 6));
    }

    #[test]
    fn bio_trims_surrounding_unicode_whitespace_and_adjusts_byte_offsets() {
        let prefix = "前缀🙂";
        let leading = "\u{2003}\t";
        let value = "李 明";
        let trailing = "\n\u{3000}";
        let text = format!("{prefix}{leading}{value}{trailing}结束");
        let raw_end = prefix.len() + leading.len() + value.len() + trailing.len();
        let entities = decode_bio_fixture(
            &text,
            &[
                (0, prefix.len(), "O", 0.9),
                (prefix.len(), raw_end, "B-PERSON", 0.8),
                (raw_end, text.len(), "O", 0.9),
            ],
        );
        assert_eq!(entities.len(), 1);
        let entity = &entities[0];
        assert_eq!(entity.text, value);
        assert_eq!(entity.start, prefix.len() + leading.len());
        assert_eq!(entity.end, entity.start + value.len());
        assert_eq!(&text[entity.start..entity.end], value);
        assert!((entity.confidence - 0.8).abs() < 1e-6);
    }

    #[test]
    fn bio_drops_whitespace_only_entities_after_trimming() {
        let text = "\u{2003}\t \n";
        let entities = decode_bio_fixture(text, &[(0, text.len(), "I-EMAIL_ADDRESS", 0.9)]);
        assert!(entities.is_empty());
    }

    #[test]
    fn bio_i_type_change_closes_the_old_span_before_starting_another() {
        let entities = decode_bio_fixture(
            "Alice a@b.co Bob",
            &[
                (0, 5, "B-PERSON", 0.9),
                (6, 7, "I-EMAIL_ADDRESS", 0.8),
                (7, 12, "I-EMAIL_ADDRESS", 0.7),
                (13, 16, "I-PERSON", 0.9),
            ],
        );
        assert_eq!(entities.len(), 3);
        assert_eq!(entities[0].text, "Alice");
        assert_eq!(entities[0].entity_type, "PERSON");
        assert_eq!(entities[1].text, "a@b.co");
        assert_eq!(entities[1].entity_type, "EMAIL_ADDRESS");
        assert_eq!(entities[2].text, "Bob");
        assert_eq!(entities[2].entity_type, "PERSON");
        assert!((entities[1].confidence - 0.75).abs() < 1e-6);
    }

    #[test]
    fn bio_b_and_o_tags_keep_same_type_entities_separate() {
        let entities = decode_bio_fixture(
            "Alice Bob and Eve",
            &[
                (0, 5, "B-PERSON", 0.9),
                (6, 9, "B-PERSON", 0.9),
                (10, 13, "O", 0.9),
                (14, 17, "I-PERSON", 0.9),
            ],
        );
        assert_eq!(
            entities
                .iter()
                .map(|entity| entity.text.as_str())
                .collect::<Vec<_>>(),
            ["Alice", "Bob", "Eve"]
        );
    }

    #[test]
    fn bio_confidence_is_the_arithmetic_mean_of_all_entity_tokens() {
        let entities = decode_bio_fixture(
            "Alice Mary Jane Smith",
            &[
                (0, 5, "B-PERSON", 0.8),
                (6, 10, "I-PERSON", 0.6),
                (11, 15, "I-PERSON", 0.9),
                (16, 21, "I-PERSON", 0.7),
            ],
        );
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].text, "Alice Mary Jane Smith");
        assert!((entities[0].confidence - 0.75).abs() < 1e-6);
    }

    fn write_onnx_dir(files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let onnx = dir.path().join("onnx");
        std::fs::create_dir_all(&onnx).unwrap();
        for f in files {
            std::fs::write(onnx.join(f), b"").unwrap();
        }
        dir
    }

    fn first_candidate(dir: &tempfile::TempDir, provider: ClassifierExecutionProvider) -> String {
        let found = MmBertSequenceClassifier::find_onnx_models(dir.path(), provider).unwrap();
        found[0].file_name().unwrap().to_string_lossy().into_owned()
    }

    #[test]
    fn cpu_provider_ranks_the_fp32_graph_first() {
        let dir = write_onnx_dir(&["model.onnx", "model_sdpa_fp16.onnx"]);
        assert_eq!(
            first_candidate(&dir, ClassifierExecutionProvider::Cpu),
            "model.onnx"
        );
    }

    #[test]
    fn cpu_provider_still_loads_an_fp16_only_model() {
        let dir = write_onnx_dir(&["model_sdpa_fp16.onnx"]);
        assert_eq!(
            first_candidate(&dir, ClassifierExecutionProvider::Cpu),
            "model_sdpa_fp16.onnx"
        );
    }

    #[test]
    fn gpu_providers_keep_the_fp16_graph_first() {
        let dir = write_onnx_dir(&["model.onnx", "model_sdpa_fp16.onnx"]);
        for provider in [
            ClassifierExecutionProvider::Cuda,
            ClassifierExecutionProvider::Rocm,
        ] {
            assert_eq!(
                first_candidate(&dir, provider),
                "model_sdpa_fp16.onnx",
                "provider {:?} should be unchanged",
                provider
            );
        }
    }

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

    fn test_tokenizer() -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        use tokenizers::processors::bert::BertProcessing;
        let vocabulary = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "word", "tail"]
            .iter()
            .enumerate()
            .map(|(i, token)| (token.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Whitespace));
        tokenizer.with_post_processor(Some(BertProcessing::new(
            ("[SEP]".into(), 3),
            ("[CLS]".into(), 2),
        )));
        tokenizer
    }

    #[test]
    fn context_budget_is_explicit_and_bounded_by_model_capacity() {
        assert_eq!(classifier_context_length(32768, None).unwrap(), 512);
        assert_eq!(classifier_context_length(256, None).unwrap(), 256);
        for limit in [512, 513, 1025, 8192, 32768] {
            assert_eq!(
                classifier_context_length(32768, Some(limit)).unwrap(),
                limit
            );
        }
        for (capacity, limit) in [(32768, 0), (8192, 32768), (0, 512)] {
            assert!(classifier_context_length(capacity, Some(limit)).is_err());
        }
    }

    #[test]
    fn context_budget_reserves_actual_postprocessor_special_tokens() {
        let mut tokenizer = test_tokenizer();
        let error = configure_classifier_tokenizer(&mut tokenizer, 1).unwrap_err();
        assert!(error.to_string().contains("requires 2 special tokens"));
        configure_classifier_tokenizer(&mut tokenizer, 2).unwrap();
        let encoding = tokenizer.encode("word tail", true).unwrap();
        assert_eq!(encoding.get_ids(), &[2, 3]);

        let mut tokenizer = test_tokenizer();
        tokenizer.with_post_processor(None::<tokenizers::processors::bert::BertProcessing>);
        configure_classifier_tokenizer(&mut tokenizer, 1).unwrap();
        let encoding = tokenizer.encode("word tail", true).unwrap();
        assert_eq!(encoding.get_ids(), &[4]);
    }

    #[test]
    fn classification_rejects_wrong_task_shapes_and_nonfinite_logits() {
        let valid = Array2::zeros((513, 35));
        validate_classifier_logits(&valid, 513, 35).unwrap();
        assert!(validate_classifier_logits(&Array2::zeros((1, 35)), 513, 35).is_err());
        assert!(validate_classifier_logits(&valid, 513, 14).is_err());
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut logits = Array2::zeros((1, 14));
            logits[(0, 3)] = invalid;
            assert!(validate_classifier_logits(&logits, 1, 14).is_err());
        }
    }

    #[test]
    fn explicit_context_keeps_tail_and_special_tokens_through_32k() {
        for limit in [512, 513, 1025, 8192, 32768] {
            let mut tokenizer = test_tokenizer();
            configure_classifier_tokenizer(&mut tokenizer, limit).unwrap();
            let text = format!("{}tail", "word ".repeat(limit - 3));
            let encoding = tokenizer.encode(text.as_str(), true).unwrap();
            assert_eq!(encoding.len(), limit);
            assert_eq!(
                encoding.get_ids()[limit - 2],
                5,
                "tail must survive at {limit}"
            );
            assert_eq!(
                encoding.get_ids()[limit - 1],
                3,
                "SEP must count toward the limit"
            );
            let longer = format!("{}word word", text);
            let encoding = tokenizer.encode(longer.as_str(), true).unwrap();
            assert_eq!(encoding.len(), limit);
            assert_eq!(encoding.get_ids()[limit - 1], 3);
        }
    }

    #[test]
    fn artifact_fixed_padding_cannot_expand_the_input_budget() {
        let mut tokenizer = test_tokenizer();
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(4096),
            pad_id: 1,
            ..Default::default()
        }));
        configure_classifier_tokenizer(&mut tokenizer, 1025).unwrap();
        let encodings = tokenizer
            .encode_batch(vec!["word", "word tail"], true)
            .unwrap();
        assert_eq!(encodings[0].len(), 3);
        assert_eq!(encodings[1].len(), 4);
        assert!(encodings
            .iter()
            .all(|e| e.get_attention_mask().iter().all(|&v| v == 1)));
    }
}
