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
use tokenizers::Tokenizer;

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
    /// Automatic selection (ROCm > CUDA > OpenVINO > CPU)
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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        // Find ONNX model candidates and initialize with fallback.
        let onnx_candidates = Self::find_onnx_models(&model_path)?;
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
        })
    }

    /// Find ONNX model candidates in priority order.
    fn find_onnx_models<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Vec<std::path::PathBuf>> {
        let dir = model_path.as_ref();
        let onnx_subdir = dir.join("onnx");
        let search_dirs = [dir, onnx_subdir.as_path()];

        // Prefer GPU-optimized variant first, then compatibility variants.
        let candidates = [
            "model_sdpa_fp16.onnx",
            "model.onnx",
            "classifier.onnx",
            "model_optimized.onnx",
        ];

        let mut results: Vec<std::path::PathBuf> = Vec::new();
        // Try known ONNX filenames first in both model root and `onnx/` subdirectory.
        for base_dir in search_dirs {
            if !base_dir.exists() || !base_dir.is_dir() {
                continue;
            }
            for candidate in &candidates {
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
                    if let Some(ext) = path.extension() {
                        if ext == "onnx" {
                            if !results.iter().any(|p| p == &path) {
                                results.push(path);
                            }
                        }
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

                    // Try MIGraphX first (better for MI300X) — error_on_failure()
                    // ensures we get a real error instead of silent CPU fallback.
                    match Session::builder()
                        .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                        .with_execution_providers([MIGraphXExecutionProvider::default()
                            .build()
                            .error_on_failure()])
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

        // Find max sequence length
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);
        let max_len = max_len.min(self.config.max_position_embeddings);

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
                Array2::from_shape_vec((*seq_len, *num_labels), flat.into_iter().take(expected).collect())
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

    let output_names = ["logits", "output", "predictions", "output_0", "token_logits"];
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

        // Find max
        let (class_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
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

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let onnx_candidates = MmBertSequenceClassifier::find_onnx_models(&model_path)?;
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
        })
    }

    /// Detect PII entities in text
    pub fn detect_entities(&mut self, text: &str) -> UnifiedResult<TokenClassificationResult> {
        // Tokenize with offsets
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        let seq_len = encoding.len().min(self.config.max_position_embeddings);

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

        // Get predicted label
        let (label_id, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let label = config.get_label(label_id as i32);

        // Parse BIO tag
        if label.starts_with("B-") {
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
            let entity_type = label[2..].to_string();
            current_entity = Some((entity_type, start, end, confidence));
        } else if label.starts_with("I-") {
            // Continue current entity
            if let Some((ref entity_type, ent_start, _, ref mut ent_conf)) = current_entity {
                let expected_type = &label[2..];
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
    fn test_config_with_labels() {
        let mut config = MmBertClassifierConfig::default();
        config.num_labels = 3;
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
}
