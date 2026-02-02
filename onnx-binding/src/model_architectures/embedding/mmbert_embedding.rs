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
use crate::model_architectures::embedding::pooling::{l2_normalize, mean_pool_3d, truncate_dimension};
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
        let config_path = model_path.as_ref().join("config.json");

        if !config_path.exists() {
            return Err(errors::file_not_found(&config_path.display().to_string()));
        }

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| errors::file_not_found(&config_path.display().to_string()))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| errors::invalid_json(&config_path.display().to_string(), &e.to_string()))?;

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

        // Load configuration
        let config = MmBertEmbeddingConfig::from_pretrained(&model_path)?;

        // Load tokenizer
        let tokenizer_path = model_path.as_ref().join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(errors::file_not_found(&tokenizer_path.display().to_string()));
        }

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| errors::tokenization_error(&e.to_string()))?;

        // Find ONNX model file
        let onnx_path = Self::find_onnx_model(&model_path)?;

        // Create ONNX Runtime session
        let session = Self::create_session(&onnx_path, use_cpu)?;

        // Check for layer-specific ONNX files (for early exit support)
        let (supports_layer_exit, layer_sessions) = Self::load_layer_sessions(&model_path, use_cpu);

        Ok(Self {
            session,
            tokenizer: Arc::new(tokenizer),
            config,
            matryoshka_config: MatryoshkaConfig::default(),
            model_path: model_path_str,
            supports_layer_exit,
            layer_sessions,
        })
    }

    /// Find the ONNX model file in the model directory
    fn find_onnx_model<P: AsRef<Path>>(model_path: P) -> UnifiedResult<std::path::PathBuf> {
        let dir = model_path.as_ref();

        // Try common ONNX file names
        let candidates = [
            "model.onnx",
            "encoder.onnx",
            "mmbert.onnx",
            "model_optimized.onnx",
        ];

        for candidate in &candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // Look for any .onnx file
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(ext) = entry.path().extension() {
                    if ext == "onnx" {
                        return Ok(entry.path());
                    }
                }
            }
        }

        Err(errors::file_not_found(&format!(
            "No ONNX model found in {}",
            dir.display()
        )))
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
            // Try ROCm first (primary AMD GPU provider in onnxruntime-rocm)
            #[cfg(any(feature = "rocm", feature = "migraphx"))]
            {
                use ort::execution_providers::ROCmExecutionProvider;
                println!("INFO: Attempting ROCm execution provider...");
                let result = Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([ROCmExecutionProvider::default().build()])
                    .map_err(|e: ort::Error| {
                        println!("WARN: ROCm EP registration error: {}", e);
                        errors::ort_error(&e.to_string())
                    })?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()));

                if let Ok(session) = result {
                    println!("INFO: Using ROCm execution provider (AMD GPU)");
                    return Ok(session);
                } else {
                    println!("WARN: ROCm session failed, trying MIGraphX...");
                }
            }

            // Try MIGraphX as alternative
            #[cfg(feature = "migraphx")]
            {
                use ort::execution_providers::MIGraphXExecutionProvider;
                println!("INFO: Attempting MIGraphX execution provider...");
                let result = Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([
                        MIGraphXExecutionProvider::default()
                            .with_fp16(true)
                            .build()
                    ])
                    .map_err(|e: ort::Error| {
                        println!("WARN: MIGraphX EP registration error: {}", e);
                        errors::ort_error(&e.to_string())
                    })?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()));

                if let Ok(session) = result {
                    println!("INFO: Using MIGraphX execution provider (AMD GPU - FAST)");
                    return Ok(session);
                } else {
                    println!("WARN: MIGraphX failed, trying fallback...");
                }
            }

            #[cfg(feature = "cuda")]
            {
                use ort::execution_providers::CUDAExecutionProvider;
                let result = Session::builder()
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .with_execution_providers([CUDAExecutionProvider::default().build()])
                    .map_err(|e: ort::Error| errors::ort_error(&e.to_string()))?
                    .commit_from_file(onnx_path.as_ref())
                    .map_err(|e: ort::Error| errors::model_load(&onnx_path_str, &e.to_string()));

                if let Ok(session) = result {
                    println!("INFO: Using CUDA execution provider (NVIDIA GPU)");
                    return Ok(session);
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

    /// Load layer-specific ONNX sessions for early exit support
    fn load_layer_sessions<P: AsRef<Path>>(model_path: P, use_cpu: bool) -> (bool, Vec<Option<Session>>) {
        let matryoshka = MatryoshkaConfig::default();
        let mut sessions = Vec::new();
        let mut any_loaded = false;

        for layer in &matryoshka.layers {
            let layer_path = model_path.as_ref().join(format!("model_layer_{}.onnx", layer));
            if layer_path.exists() {
                match Self::create_session(&layer_path, use_cpu) {
                    Ok(session) => {
                        sessions.push(Some(session));
                        any_loaded = true;
                    }
                    Err(_) => sessions.push(None),
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
            self.matryoshka_config
                .layers
                .iter()
                .enumerate()
                .filter_map(|(i, &layer)| {
                    if self.layer_sessions.get(i).map_or(false, |s: &Option<Session>| s.is_some()) {
                        Some(layer)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![self.config.num_hidden_layers]
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
        if texts.is_empty() {
            return Err(UnifiedError::Validation {
                field: "texts".to_string(),
                expected: "non-empty".to_string(),
                actual: "empty".to_string(),
            });
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
            for j in 0..seq_len {
                input_ids[i * max_len + j] = encoding.get_ids()[j] as i64;
                attention_mask[i * max_len + j] = 1;
            }
        }

        // Create ndarray tensors
        let input_ids_array = Array2::from_shape_vec(
            (batch_size, max_len),
            input_ids.clone(),
        ).map_err(|e| errors::inference_error("create_input_ids", &e.to_string()))?;

        let attention_mask_array = Array2::from_shape_vec(
            (batch_size, max_len),
            attention_mask.clone(),
        ).map_err(|e| errors::inference_error("create_attention_mask", &e.to_string()))?;

        // Run inference - inline session selection to avoid borrow checker issues
        let embeddings = self.run_inference_with_layer(
            target_layer,
            &input_ids_array,
            &attention_mask_array,
        )?;

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

    /// Run inference on the ONNX model with optional layer selection
    fn run_inference_with_layer(
        &mut self,
        target_layer: Option<usize>,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> UnifiedResult<Array2<f32>> {
        // Select session based on target layer (inline to avoid borrow issues)
        let session_idx = if let Some(layer) = target_layer {
            if self.supports_layer_exit {
                self.matryoshka_config.layers.iter().position(|&l| l == layer)
                    .filter(|&idx| self.layer_sessions.get(idx).map_or(false, |s| s.is_some()))
            } else {
                None
            }
        } else {
            None
        };

        // Get the appropriate session
        let session = if let Some(idx) = session_idx {
            self.layer_sessions[idx].as_mut().unwrap()
        } else {
            &mut self.session
        };
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];

        // Create ort tensors from ndarray - ort 2.x requires (shape, data) tuple
        let input_ids_flat: Vec<i64> = input_ids.iter().copied().collect();
        let attention_mask_flat: Vec<i64> = attention_mask.iter().copied().collect();

        let input_ids_tensor = Tensor::from_array(([batch_size, seq_len], input_ids_flat))
            .map_err(|e: ort::Error| errors::inference_error("create_input_ids_tensor", &e.to_string()))?;

        let attention_mask_tensor = Tensor::from_array(([batch_size, seq_len], attention_mask_flat))
            .map_err(|e: ort::Error| errors::inference_error("create_attention_mask_tensor", &e.to_string()))?;

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
        let output_names = ["last_hidden_state", "sentence_embedding", "pooler_output", "embeddings"];

        for name in &output_names {
            if let Some(output_value) = outputs.get(*name) {
                if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

                    if dims.len() == 2 {
                        // Already pooled: [batch, hidden_dim]
                        let flat: Vec<f32> = data.to_vec();
                        return Array2::from_shape_vec((dims[0], dims[1]), flat)
                            .map_err(|e| errors::inference_error("reshape_output", &e.to_string()));
                    } else if dims.len() == 3 {
                        // Sequence output: [batch, seq_len, hidden_dim] - needs pooling
                        let out_batch_size = dims[0];
                        let out_seq_len = dims[1];
                        let hidden_dim = dims[2];

                        let flat: Vec<f32> = data.to_vec();
                        let hidden_states = Array3::from_shape_vec((out_batch_size, out_seq_len, hidden_dim), flat)
                            .map_err(|e| errors::inference_error("reshape_hidden_states", &e.to_string()))?;

                        // Convert attention_mask to f32
                        let attention_mask_f32: Array2<f32> = attention_mask.mapv(|x| x as f32);

                        // Mean pooling
                        return Ok(mean_pool_3d(&hidden_states, &attention_mask_f32));
                    }
                }
            }
        }

        // Try first output if named outputs not found
        if let Some((_, output_value)) = outputs.iter().next() {
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

                if dims.len() == 2 {
                    let flat: Vec<f32> = data.to_vec();
                    return Array2::from_shape_vec((dims[0], dims[1]), flat)
                        .map_err(|e| errors::inference_error("reshape_output", &e.to_string()));
                } else if dims.len() == 3 {
                    let out_batch_size = dims[0];
                    let out_seq_len = dims[1];
                    let hidden_dim = dims[2];

                    let flat: Vec<f32> = data.to_vec();
                    let hidden_states = Array3::from_shape_vec((out_batch_size, out_seq_len, hidden_dim), flat)
                        .map_err(|e| errors::inference_error("reshape_hidden_states", &e.to_string()))?;

                    let attention_mask_f32: Array2<f32> = attention_mask.mapv(|x| x as f32);
                    return Ok(mean_pool_3d(&hidden_states, &attention_mask_f32));
                }
            }
        }

        Err(errors::inference_error(
            "extract_output",
            "Failed to extract output tensor",
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
        assert!(provider == ExecutionProvider::Cpu || provider == ExecutionProvider::Rocm || provider == ExecutionProvider::Cuda);
    }

    #[test]
    fn test_config_defaults() {
        let config = MmBertEmbeddingConfig::default();
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.max_position_embeddings, 32768);
    }
}
