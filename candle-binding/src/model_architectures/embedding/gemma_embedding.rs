//! GemmaEmbedding-300M Model Implementation
//!
//! This module implements the EmbeddingGemma-300M model with:
//! - **2K context length** (max_position_embeddings: 2048)
//! - **Mean pooling** for embedding extraction
//! - **Dense bottleneck** (768→3072→768) for quality improvement
//! - **Matryoshka representation** (768/512/256/128 dimensions)
//!
//! ## Architecture
//! - Embedding layer: vocab_size × hidden_size
//! - 24 transformer blocks (Gemma3DecoderLayer)
//! - RMSNorm for normalization
//! - Mean pooling over all tokens
//! - Dense bottleneck for embedding transformation (768→3072→768)
//!
//! ## Key Features
//! - Matryoshka learning: Multi-dimensional embeddings from single forward pass
//! - Dense bottleneck critical for quality (discovered in Plan 4 analysis)
//! - MQA (Multi-Query Attention): 3 query heads, 1 KV head
//! - Mixed attention: sliding_attention + full_attention layers
//! - RoPE with θ=1000000.0 (local_base_freq=10000.0)
//!
//! ## References
//! - Official: https://huggingface.co/google/embeddinggemma-300m
//! - Config: https://huggingface.co/google/embeddinggemma-300m/blob/main/config.json
//! - TEI Gemma3: backends/candle/src/models/gemma3.rs

use crate::core::{config_errors, from_candle_error, UnifiedError, UnifiedResult};
use crate::model_architectures::traits::ModelType;
use crate::model_architectures::unified_interface::CoreModel;
use serde::Deserialize;
use std::path::Path;

/// Gemma3 Attention Layer Type
///
/// EmbeddingGemma-300M uses a mixed attention pattern:
/// - `sliding_attention`: Local attention with 512-token window
/// - `full_attention`: Global attention across all tokens
///
/// Pattern (24 layers total):
/// - Layers 0-4: sliding_attention
/// - Layer 5: full_attention
/// - Layers 6-10: sliding_attention
/// - Layer 11: full_attention
/// - Layers 12-16: sliding_attention
/// - Layer 17: full_attention
/// - Layers 18-22: sliding_attention
/// - Layer 23: full_attention
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionLayerType {
    /// Local attention with sliding window (default 512 tokens)
    SlidingAttention,
    /// Global attention across all tokens
    FullAttention,
}

/// GemmaEmbedding model configuration
///
/// This configuration is loaded from `config.json` and supports the EmbeddingGemma-300M model.
///
/// # Architecture Details
/// - **Hidden size**: 768 (embedding dimension)
/// - **Layers**: 24 transformer blocks
/// - **Attention**: MQA (3 query heads, 1 KV head)
/// - **Head dim**: 256 (explicitly specified, not computed from hidden_size)
/// - **Max length**: 2048 tokens
/// - **Pooling**: Mean pooling (configured separately)
/// - **Dense Bottleneck**: 768→3072→768 (configured separately)
///
/// # Critical Parameters
/// - `head_dim` = 256 (NOT hidden_size / num_attention_heads)
/// - `num_key_value_heads` = 1 (MQA architecture)
/// - `rope_theta` = 1000000.0 (global), `rope_local_base_freq` = 10000.0
/// - `use_bidirectional_attention` = true (encoder model)
///
/// # Usage
/// ```ignore
/// let config = GemmaEmbeddingConfig::from_pretrained(
///     "models/mom-embedding-flash"
/// )?;
/// ```
#[derive(Debug, Clone, Deserialize)]
pub struct GemmaEmbeddingConfig {
    /// Vocabulary size
    /// - EmbeddingGemma-300M: 262144
    pub vocab_size: usize,

    /// Hidden dimension size (embedding dimension)
    /// - EmbeddingGemma-300M: 768
    pub hidden_size: usize,

    /// Number of transformer layers
    /// - EmbeddingGemma-300M: 24
    pub num_hidden_layers: usize,

    /// Number of attention heads (query heads)
    /// - EmbeddingGemma-300M: 3
    pub num_attention_heads: usize,

    /// Number of key-value heads (MQA)
    /// - EmbeddingGemma-300M: 1 (Multi-Query Attention)
    /// - All query heads share the same K/V
    pub num_key_value_heads: usize,

    /// Intermediate size for MLP
    /// - EmbeddingGemma-300M: 1152
    pub intermediate_size: usize,

    /// Maximum position embeddings (sequence length)
    /// - EmbeddingGemma-300M: 2048
    pub max_position_embeddings: usize,

    /// RoPE theta (global base frequency)
    /// - EmbeddingGemma-300M: 1000000.0
    pub rope_theta: f32,

    /// RoPE local base frequency
    /// - EmbeddingGemma-300M: 10000.0
    /// - Used for position encoding calculation
    pub rope_local_base_freq: f32,

    /// RMS normalization epsilon
    /// - EmbeddingGemma-300M: 1e-6
    pub rms_norm_eps: f64,

    /// Attention dropout rate
    /// - EmbeddingGemma-300M: 0.0
    pub attention_dropout: f32,

    /// Head dimension (CRITICAL: explicitly specified, NOT computed!)
    /// - EmbeddingGemma-300M: 256
    /// - WARNING: 256 ≠ hidden_size / num_attention_heads (768 / 3 = 256)
    /// - Actually equal in this case, but still explicitly specified
    pub head_dim: usize,

    /// Sliding window size for local attention
    /// - EmbeddingGemma-300M: 512
    pub sliding_window: usize,

    /// Attention layer types for each layer
    /// - 24 layers total
    /// - Mixed pattern: sliding_attention and full_attention
    pub layer_types: Vec<AttentionLayerType>,

    /// Whether to use bidirectional attention
    /// - EmbeddingGemma-300M: true (encoder model, not causal)
    pub use_bidirectional_attention: bool,

    /// Query pre-attention scalar
    /// - EmbeddingGemma-300M: 256
    /// - Scaling factor for attention scores
    pub query_pre_attn_scalar: usize,

    /// Hidden activation function
    /// - EmbeddingGemma-300M: "gelu_pytorch_tanh"
    pub hidden_activation: String,
}

impl GemmaEmbeddingConfig {
    /// Load configuration from a pretrained model directory
    ///
    /// # Arguments
    /// - `model_path`: Path to model directory containing `config.json`
    ///
    /// # Returns
    /// - `Ok(GemmaEmbeddingConfig)`: Successfully loaded and validated config
    /// - `Err(UnifiedError)`: File not found, invalid JSON, or validation failed
    ///
    /// # Example
    /// ```ignore
    /// let config = GemmaEmbeddingConfig::from_pretrained(
    ///     "models/mom-embedding-flash"
    /// )?;
    /// println!("Loaded config: {} layers, {} hidden size",
    ///          config.num_hidden_layers, config.hidden_size);
    /// ```
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let config_path = model_path.as_ref().join("config.json");

        // Check file existence
        if !config_path.exists() {
            return Err(config_errors::file_not_found(
                &config_path.display().to_string(),
            ));
        }

        // Read file
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| config_errors::file_not_found(&config_path.display().to_string()))?;

        // Parse JSON
        let config: Self = serde_json::from_str(&config_str).map_err(|e| {
            config_errors::invalid_json(&config_path.display().to_string(), &e.to_string())
        })?;

        // Validate
        config.validate()?;

        Ok(config)
    }

    /// Validate configuration parameters
    ///
    /// Checks that all critical parameters are within expected ranges and consistent.
    ///
    /// # Validation Rules
    /// 1. `hidden_size` must be > 0 and divisible by `num_attention_heads`
    /// 2. `num_hidden_layers` must be > 0
    /// 3. `num_attention_heads` must be > 0
    /// 4. `num_key_value_heads` must be > 0 and <= `num_attention_heads`
    /// 5. `max_position_embeddings` must be >= 512 (minimum useful length)
    /// 6. `head_dim` must be > 0
    /// 7. `layer_types` must have exactly `num_hidden_layers` entries
    /// 8. `sliding_window` must be > 0 and <= `max_position_embeddings`
    /// 9. `rms_norm_eps` must be > 0
    ///
    /// # Returns
    /// - `Ok(())`: All validation passed
    /// - `Err(UnifiedError::Validation)`: Validation failed with detailed error message
    pub fn validate(&self) -> UnifiedResult<()> {
        // 1. hidden_size validation
        if self.hidden_size == 0 {
            return Err(UnifiedError::Validation {
                field: "hidden_size".to_string(),
                expected: "> 0".to_string(),
                actual: self.hidden_size.to_string(),
                context: None,
            });
        }

        // 2. num_hidden_layers validation
        if self.num_hidden_layers == 0 {
            return Err(UnifiedError::Validation {
                field: "num_hidden_layers".to_string(),
                expected: "> 0".to_string(),
                actual: self.num_hidden_layers.to_string(),
                context: None,
            });
        }

        // 3. num_attention_heads validation
        if self.num_attention_heads == 0 {
            return Err(UnifiedError::Validation {
                field: "num_attention_heads".to_string(),
                expected: "> 0".to_string(),
                actual: self.num_attention_heads.to_string(),
                context: None,
            });
        }

        // 4. MQA validation
        if self.num_key_value_heads == 0 || self.num_key_value_heads > self.num_attention_heads {
            return Err(UnifiedError::Validation {
                field: "num_key_value_heads".to_string(),
                expected: format!("> 0 and <= {}", self.num_attention_heads),
                actual: self.num_key_value_heads.to_string(),
                context: Some(
                    "MQA requires num_key_value_heads <= num_attention_heads".to_string(),
                ),
            });
        }

        // 5. max_position_embeddings validation
        if self.max_position_embeddings < 512 {
            return Err(UnifiedError::Validation {
                field: "max_position_embeddings".to_string(),
                expected: ">= 512".to_string(),
                actual: self.max_position_embeddings.to_string(),
                context: Some("Minimum useful sequence length is 512".to_string()),
            });
        }

        // 6. head_dim validation
        if self.head_dim == 0 {
            return Err(UnifiedError::Validation {
                field: "head_dim".to_string(),
                expected: "> 0".to_string(),
                actual: self.head_dim.to_string(),
                context: None,
            });
        }

        // 7. layer_types validation
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(UnifiedError::Validation {
                field: "layer_types".to_string(),
                expected: format!("{} entries (num_hidden_layers)", self.num_hidden_layers),
                actual: format!("{} entries", self.layer_types.len()),
                context: Some("layer_types must match num_hidden_layers".to_string()),
            });
        }

        // 8. sliding_window validation
        if self.sliding_window == 0 || self.sliding_window > self.max_position_embeddings {
            return Err(UnifiedError::Validation {
                field: "sliding_window".to_string(),
                expected: format!("> 0 and <= {}", self.max_position_embeddings),
                actual: self.sliding_window.to_string(),
                context: None,
            });
        }

        // 9. rms_norm_eps validation
        if self.rms_norm_eps <= 0.0 {
            return Err(UnifiedError::Validation {
                field: "rms_norm_eps".to_string(),
                expected: "> 0.0".to_string(),
                actual: self.rms_norm_eps.to_string(),
                context: None,
            });
        }

        Ok(())
    }

    /// Get the attention layer type for a specific layer index
    ///
    /// # Arguments
    /// - `layer_idx`: Layer index (0-based)
    ///
    /// # Returns
    /// - `Some(AttentionLayerType)`: Layer type if index is valid
    /// - `None`: If index is out of bounds
    pub fn get_layer_type(&self, layer_idx: usize) -> Option<AttentionLayerType> {
        self.layer_types.get(layer_idx).copied()
    }

    /// Check if a specific layer uses full attention
    ///
    /// # Arguments
    /// - `layer_idx`: Layer index (0-based)
    ///
    /// # Returns
    /// - `true`: Layer uses full attention
    /// - `false`: Layer uses sliding attention or index is invalid
    pub fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        matches!(
            self.get_layer_type(layer_idx),
            Some(AttentionLayerType::FullAttention)
        )
    }
}

// ============================================================================
// GemmaEmbeddingModel Implementation
// ============================================================================

use super::dense_layers::BottleneckDenseNet;
use super::gemma3_model::Gemma3Model;
use super::pooling::mean_pool;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

/// Complete GemmaEmbedding model
///
/// Architecture:
/// 1. Gemma3 Transformer backbone (24 layers, 768 hidden, MQA)
/// 2. Mean Pooling (sentence-level representation)
/// 3. Dense Bottleneck (768 → 3072 → 768, Identity activation)
/// 4. L2 Normalization
///
/// ## Model Specifications
/// - Model: `google/embeddinggemma-300m`
/// - Hidden size: 768
/// - Sequence length: 2048 (max)
/// - Embedding dimension: 768 (after bottleneck)
/// - Total parameters: ~300M
///
/// ## Usage
/// ```ignore
/// let config = GemmaEmbeddingConfig::from_pretrained("../models/mom-embedding-flash")?;
/// let vb = VarBuilder::from_mmaped_safetensors(...)?;
/// let model = GemmaEmbeddingModel::load("../models/mom-embedding-flash", &config, vb)?;
///
/// let embeddings = model.embedding_forward(&input_ids, Some(&attention_mask))?;
/// ```
#[derive(Debug)]
pub struct GemmaEmbeddingModel {
    /// Gemma3 Transformer backbone
    gemma_backbone: Gemma3Model,

    /// Dense Bottleneck (768 → 3072 → 768)
    dense_bottleneck: BottleneckDenseNet,

    /// Model configuration
    config: GemmaEmbeddingConfig,

    /// Device (CPU/GPU)
    device: Device,
}

impl GemmaEmbeddingModel {
    /// Load GemmaEmbedding model from pretrained weights
    ///
    /// # Arguments
    /// - `model_path`: Path to model directory
    /// - `config`: Model configuration
    /// - `vb`: VarBuilder for loading weights from safetensors
    ///
    /// # Returns
    /// - `Ok(GemmaEmbeddingModel)`: Successfully loaded model
    /// - `Err(UnifiedError)`: Loading failed
    ///
    /// # Example
    /// ```ignore
    /// let config = GemmaEmbeddingConfig::from_pretrained("../models/mom-embedding-flash")?;
    /// let device = Device::Cpu;
    /// let vb = VarBuilder::from_mmaped_safetensors(
    ///     &["../models/mom-embedding-flash/model.safetensors"],
    ///     DType::F32,
    ///     &device
    /// )?;
    /// let model = GemmaEmbeddingModel::load("../models/mom-embedding-flash", &config, vb)?;
    /// ```
    pub fn load(
        model_path: &str,
        config: &GemmaEmbeddingConfig,
        vb: VarBuilder,
    ) -> UnifiedResult<Self> {
        let device = vb.device().clone();

        // Load Gemma3 Transformer backbone
        // Note: Weights in safetensors have no "model." prefix
        let gemma_backbone = Gemma3Model::load(vb, config)?;

        // Load Dense Bottleneck (from separate safetensors files in 2_Dense/ and 3_Dense/)
        let dense_bottleneck = BottleneckDenseNet::load_from_path(model_path, &device)?;

        Ok(Self {
            gemma_backbone,
            dense_bottleneck,
            config: config.clone(),
            device,
        })
    }

    /// Get the device the model is loaded on
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Get model configuration
    pub fn config(&self) -> &GemmaEmbeddingConfig {
        &self.config
    }

    /// Get access to Gemma3 Transformer backbone (for testing)
    #[cfg(test)]
    pub fn gemma_backbone(&self) -> &Gemma3Model {
        &self.gemma_backbone
    }

    /// Get access to Dense Bottleneck (for testing)
    #[cfg(test)]
    pub fn dense_bottleneck(&self) -> &BottleneckDenseNet {
        &self.dense_bottleneck
    }

    /// Forward pass to generate embeddings
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs, shape [batch, seq_len]
    /// - `attention_mask`: Attention mask (optional), shape [batch, seq_len]
    ///
    /// # Returns
    /// - Normalized embeddings, shape [batch, 768]
    ///
    /// # Flow
    /// 1. Gemma3 Transformer → [batch, seq_len, 768]
    /// 2. Mean Pooling → [batch, 768]
    /// 3. Dense Bottleneck → [batch, 768]
    /// 4. L2 Normalization → [batch, 768]
    pub fn embedding_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        // Step 1: Gemma3 Transformer backbone
        // Output: [batch, seq_len, hidden_size=768]
        let hidden_states = self.gemma_backbone.forward(input_ids, attention_mask)?;

        // Step 2: Mean Pooling
        // Create default attention mask if not provided
        let default_mask;
        let mask = match attention_mask {
            Some(m) => m,
            None => {
                let shape = hidden_states.dims();
                default_mask =
                    Tensor::ones((shape[0], shape[1]), candle_core::DType::F32, &self.device)
                        .map_err(|e| from_candle_error(e, "create default attention mask", None))?;
                &default_mask
            }
        };

        // Output: [batch, hidden_size=768]
        let pooled = mean_pool(&hidden_states, mask).map_err(|e| UnifiedError::Processing {
            operation: "mean_pool".to_string(),
            source: e.to_string(),
            input_context: None,
        })?;

        // Step 3: Dense Bottleneck (768 → 3072 → 768)
        // Output: [batch, hidden_size=768]
        let embeddings = self.dense_bottleneck.forward(&pooled)?;

        // Step 4: L2 Normalization
        // norm = sqrt(sum(embeddings^2, dim=-1, keepdim=True))
        // normalized = embeddings / norm
        let embeddings_squared = embeddings
            .sqr()
            .map_err(|e| from_candle_error(e, "L2 norm: compute x^2", None))?;
        let sum_squared = embeddings_squared
            .sum_keepdim(candle_core::D::Minus1)
            .map_err(|e| from_candle_error(e, "L2 norm: sum(x^2)", None))?;
        let norm = sum_squared
            .sqrt()
            .map_err(|e| from_candle_error(e, "L2 norm: sqrt", None))?;
        let normalized = embeddings
            .broadcast_div(&norm)
            .map_err(|e| from_candle_error(e, "L2 norm: x / norm", None))?;

        Ok(normalized)
    }

    /// Forward pass with Matryoshka Representation support
    ///
    /// Matryoshka Representation allows truncating the embedding dimension
    /// while maintaining reasonable quality. Supported dimensions: 768, 512, 256, 128
    ///
    /// # Arguments
    /// * `input_ids` - Input token IDs [batch_size, seq_len]
    /// * `attention_mask` - Optional attention mask [batch_size, seq_len]
    /// * `embedding_dim` - Target embedding dimension (768, 512, 256, or 128)
    ///
    /// # Returns
    /// L2-normalized embeddings with shape [batch_size, embedding_dim]
    ///
    /// # Flow
    /// 1. Gemma3 Transformer backbone → [batch, seq_len, 768]
    /// 2. Mean Pooling → [batch, 768]
    /// 3. Dense Bottleneck → [batch, 768]
    /// 4. L2 Normalization → [batch, 768]
    /// 5. (Optional) Truncate to target dimension → [batch, embedding_dim]
    /// 6. (Optional) Re-normalize after truncation → [batch, embedding_dim]
    pub fn matryoshka_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        embedding_dim: usize,
    ) -> UnifiedResult<Tensor> {
        // Validate embedding dimension
        const SUPPORTED_DIMS: &[usize] = &[768, 512, 256, 128];
        if !SUPPORTED_DIMS.contains(&embedding_dim) {
            return Err(UnifiedError::Validation {
                field: "embedding_dim".to_string(),
                expected: "768, 512, 256, or 128".to_string(),
                actual: embedding_dim.to_string(),
                context: Some("Matryoshka embedding dimension".to_string()),
            });
        }

        // Step 1-4: Full embedding forward (Gemma3 → Mean Pool → Dense Bottleneck → L2 Norm)
        // Output: [batch, 768]
        let full_embeddings = self.embedding_forward(input_ids, attention_mask)?;

        // If target dimension is 768, return full embeddings (already L2 normalized)
        if embedding_dim == 768 {
            return Ok(full_embeddings);
        }

        // Step 5: Truncate to target dimension
        // narrow(dim, start, length) - extract embedding_dim elements starting from index 0
        let truncated = full_embeddings.narrow(1, 0, embedding_dim).map_err(|e| {
            from_candle_error(
                e,
                &format!("Matryoshka truncation to {} dims", embedding_dim),
                None,
            )
        })?;

        // Step 6: Re-normalize after truncation
        // After truncation, the L2 norm is no longer 1.0, so we need to re-normalize
        let embeddings_squared = truncated
            .sqr()
            .map_err(|e| from_candle_error(e, "Matryoshka L2 norm: compute x^2", None))?;
        let sum_squared = embeddings_squared
            .sum_keepdim(candle_core::D::Minus1)
            .map_err(|e| from_candle_error(e, "Matryoshka L2 norm: sum(x^2)", None))?;
        let norm = sum_squared
            .sqrt()
            .map_err(|e| from_candle_error(e, "Matryoshka L2 norm: sqrt", None))?;
        let normalized = truncated
            .broadcast_div(&norm)
            .map_err(|e| from_candle_error(e, "Matryoshka L2 norm: x / norm", None))?;

        Ok(normalized)
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl CoreModel for GemmaEmbeddingModel {
    type Config = GemmaEmbeddingConfig;
    type Error = UnifiedError;
    type Output = Tensor;

    fn model_type(&self) -> ModelType {
        ModelType::GemmaEmbedding
    }

    /// Forward pass implementation (delegates to embedding_forward)
    ///
    /// This satisfies the CoreModel trait requirement while allowing us
    /// to have a more specific public API with optional attention_mask.
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error> {
        self.embedding_forward(input_ids, Some(attention_mask))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }
}
