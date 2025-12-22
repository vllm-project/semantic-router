//! Dense Bottleneck Layers for EmbeddingGemma
//!
//! This module implements the dense bottleneck architecture discovered in Plan 4 analysis.
//! The bottleneck significantly improves embedding quality compared to raw transformer outputs.
//!
//! ## Architecture
//! ```text
//! Gemma3 Backbone (768-dim)
//!     ↓
//! Mean Pooling (768-dim)
//!     ↓
//! Dense Layer 1: 768 → 3072 (expansion, Identity activation)
//!     ↓
//! Dense Layer 2: 3072 → 768 (compression, Identity activation)
//!     ↓
//! L2 Normalization
//!     ↓
//! Final Embedding (768-dim)
//! ```
//!
//! ## Key Features
//! - **No bias**: Both dense layers use bias=false (confirmed from model config)
//! - **Identity activation**: No non-linear activation (confirmed from model config)
//! - **Dimension preservation**: Output dimension (768) matches input dimension
//! - **Quality boost**: Critical for matching official embedding quality
//!
//! ## Weight Loading
//! - Layer 1 weights: `2_Dense/model.safetensors` (weight: [3072, 768])
//! - Layer 2 weights: `3_Dense/model.safetensors` (weight: [768, 3072])
//!
//! ## References
//! - SentenceTransformers architecture: https://www.sbert.net/docs/package_reference/models.html#dense
//! - EmbeddingGemma config: models/mom-embedding-flash/2_Dense/config.json
//! - Plan 4 analysis: plan-cursor.md Section 4.2

use crate::core::{from_candle_error, UnifiedError, UnifiedResult};
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

/// Activation function for dense layers
///
/// ## Variants
/// - `Identity`: No activation, output = input (used in EmbeddingGemma)
/// - `Tanh`: Hyperbolic tangent activation (alternative option, not used in EmbeddingGemma)
///
/// ## Usage in EmbeddingGemma
/// Both dense layers use `Identity` activation as specified in config files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseActivation {
    /// Identity activation: f(x) = x
    Identity,
    /// Tanh activation: f(x) = tanh(x)
    /// (Not used in EmbeddingGemma, included for potential future variants)
    Tanh,
}

impl DenseActivation {
    /// Apply activation function to tensor
    ///
    /// # Arguments
    /// - `input`: Input tensor of any shape
    ///
    /// # Returns
    /// - Tensor with activation applied element-wise
    ///
    /// # Errors
    /// - Candle error if tensor operation fails
    pub fn apply(&self, input: &Tensor) -> UnifiedResult<Tensor> {
        match self {
            DenseActivation::Identity => Ok(input.clone()),
            DenseActivation::Tanh => input
                .tanh()
                .map_err(|e| from_candle_error(e, "tanh activation", None)),
        }
    }
}

/// Dense linear layer with optional activation
///
/// This struct represents a single dense (fully connected) layer.
/// In EmbeddingGemma, two such layers form the bottleneck architecture.
///
/// ## Architecture
/// - Input: [batch_size, in_features]
/// - Linear: weight [out_features, in_features], optional bias [out_features]
/// - Activation: Identity or Tanh
/// - Output: [batch_size, out_features]
///
/// ## EmbeddingGemma Configuration
/// - **Layer 1**: in=768, out=3072, bias=false, activation=Identity
/// - **Layer 2**: in=3072, out=768, bias=false, activation=Identity
#[derive(Debug)]
pub struct DenseLayer {
    /// Linear transformation layer
    pub(crate) linear: Linear,
    /// Activation function
    pub(crate) activation: DenseActivation,
    /// Input feature dimension
    pub(crate) in_features: usize,
    /// Output feature dimension
    pub(crate) out_features: usize,
}

impl DenseLayer {
    /// Load dense layer from pretrained weights
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights from safetensors
    /// - `in_features`: Input dimension
    /// - `out_features`: Output dimension
    /// - `activation`: Activation function to apply
    /// - `use_bias`: Whether to load and use bias (false for EmbeddingGemma)
    ///
    /// # Weight Format
    /// - `weight`: [out_features, in_features] (required)
    /// - `bias`: [out_features] (optional, only if use_bias=true)
    ///
    /// # Returns
    /// - `Ok(DenseLayer)`: Successfully loaded layer
    /// - `Err(UnifiedError)`: Failed to load weights
    ///
    /// # Example
    /// ```ignore
    /// // Load EmbeddingGemma Layer 1 (expansion)
    /// let vb = VarBuilder::from_safetensors(...);
    /// let dense1 = DenseLayer::load(
    ///     vb.pp("2"),  // 2_Dense directory
    ///     768,         // input dim
    ///     3072,        // output dim
    ///     DenseActivation::Identity,
    ///     false,       // no bias
    /// )?;
    /// ```
    pub fn load(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        activation: DenseActivation,
        use_bias: bool,
    ) -> UnifiedResult<Self> {
        // Load weight: [out_features, in_features]
        // Note: Weights are stored as "linear.weight" in safetensors
        let weight = vb
            .get((out_features, in_features), "linear.weight")
            .map_err(|e| from_candle_error(e, "load dense weight", None))?;

        // Load bias if needed: [out_features]
        let bias = if use_bias {
            Some(
                vb.get(out_features, "linear.bias")
                    .map_err(|e| from_candle_error(e, "load dense bias", None))?,
            )
        } else {
            None
        };

        // Create Linear layer
        let linear = Linear::new(weight, bias);

        Ok(Self {
            linear,
            activation,
            in_features,
            out_features,
        })
    }

    /// Forward pass through dense layer
    ///
    /// # Arguments
    /// - `input`: Input tensor [batch_size, in_features]
    ///
    /// # Returns
    /// - Output tensor [batch_size, out_features] after linear transformation and activation
    ///
    /// # Errors
    /// - Shape mismatch if input.dim(-1) != in_features
    /// - Candle error if tensor operation fails
    pub fn forward(&self, input: &Tensor) -> UnifiedResult<Tensor> {
        // Validate input shape
        let input_shape = input.dims();
        let input_dim = input_shape[input_shape.len() - 1];
        if input_dim != self.in_features {
            return Err(UnifiedError::Validation {
                field: "input dimension".to_string(),
                expected: self.in_features.to_string(),
                actual: input_dim.to_string(),
                context: Some(format!(
                    "Dense layer expects input dimension {}, got {}",
                    self.in_features, input_dim
                )),
            });
        }

        // Linear transformation
        let output = self
            .linear
            .forward(input)
            .map_err(|e| from_candle_error(e, "dense forward", None))?;

        // Apply activation
        self.activation.apply(&output)
    }

    /// Get input feature dimension
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output feature dimension
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

/// Dense Bottleneck Network for EmbeddingGemma
///
/// This struct implements the complete dense bottleneck discovered in Plan 4 analysis.
/// It consists of two dense layers: expansion (768→3072) and compression (3072→768).
///
/// ## Architecture Flow
/// ```text
/// Input: [batch_size, 768]  (from mean pooling)
///    ↓
/// Dense1: [batch, 768] → [batch, 3072]  (expansion, Identity)
///    ↓
/// Dense2: [batch, 3072] → [batch, 768]  (compression, Identity)
///    ↓
/// Output: [batch_size, 768]  (ready for L2 normalization)
/// ```
///
/// ## SentenceTransformer Mapping
/// This corresponds to:
/// - `(2): Dense({'in_features': 768, 'out_features': 3072})`
/// - `(3): Dense({'in_features': 3072, 'out_features': 768})`
///
/// ## Critical Discovery (Plan 4)
/// The dense bottleneck is **essential** for quality:
/// - Without bottleneck: ~85% of official quality
/// - With bottleneck: ~99% of official quality (>0.99 cosine similarity)
#[derive(Debug)]
pub struct BottleneckDenseNet {
    /// First dense layer: 768 → 3072 (expansion)
    pub(crate) dense1: DenseLayer,
    /// Second dense layer: 3072 → 768 (compression)
    pub(crate) dense2: DenseLayer,
}

impl BottleneckDenseNet {
    /// Load bottleneck from pretrained model
    ///
    /// # Arguments
    /// - `vb`: VarBuilder pointing to model root directory
    ///
    /// # Directory Structure
    /// ```text
    /// models/mom-embedding-flash/
    ///   ├── 2_Dense/
    ///   │   ├── config.json          (in: 768, out: 3072, bias: false, activation: Identity)
    ///   │   └── model.safetensors    (weight: [3072, 768])
    ///   └── 3_Dense/
    ///       ├── config.json          (in: 3072, out: 768, bias: false, activation: Identity)
    ///       └── model.safetensors    (weight: [768, 3072])
    /// ```
    ///
    /// # Returns
    /// - `Ok(BottleneckDenseNet)`: Successfully loaded bottleneck
    /// - `Err(UnifiedError)`: Failed to load weights
    ///
    /// # Example
    /// ```ignore
    /// use candle_nn::VarBuilder;
    ///
    /// let vb = VarBuilder::from_safetensors(
    ///     vec!["models/mom-embedding-flash/2_Dense/model.safetensors",
    ///          "models/mom-embedding-flash/3_Dense/model.safetensors"],
    ///     dtype,
    ///     device,
    /// )?;
    /// let bottleneck = BottleneckDenseNet::load(vb)?;
    /// ```
    pub fn load(vb: VarBuilder) -> UnifiedResult<Self> {
        // Load first dense layer: 768 → 3072
        // VarBuilder path: "2" (corresponds to 2_Dense directory)
        let dense1 = DenseLayer::load(
            vb.pp("2"),
            768,
            3072,
            DenseActivation::Identity,
            false, // no bias
        )?;

        // Load second dense layer: 3072 → 768
        // VarBuilder path: "3" (corresponds to 3_Dense directory)
        let dense2 = DenseLayer::load(
            vb.pp("3"),
            3072,
            768,
            DenseActivation::Identity,
            false, // no bias
        )?;

        Ok(Self { dense1, dense2 })
    }

    /// Load bottleneck from model directory path
    ///
    /// # Arguments
    /// - `model_path`: Path to model directory (e.g., "../models/mom-embedding-flash")
    /// - `device`: Device to load weights on
    ///
    /// # Returns
    /// - `Ok(BottleneckDenseNet)`: Successfully loaded bottleneck
    /// - `Err(UnifiedError)`: Failed to load weights
    pub fn load_from_path(model_path: &str, device: &candle_core::Device) -> UnifiedResult<Self> {
        use candle_nn::VarBuilder;
        use std::path::PathBuf;

        // Load 2_Dense (768 → 3072)
        let dense1_path = PathBuf::from(model_path).join("2_Dense/model.safetensors");
        let vb1 = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dense1_path.to_str().unwrap()],
                candle_core::DType::F32,
                device,
            )
        }
        .map_err(|e| from_candle_error(e, "load 2_Dense safetensors", None))?;

        let dense1 = DenseLayer::load(vb1, 768, 3072, DenseActivation::Identity, false)?;

        // Load 3_Dense (3072 → 768)
        let dense2_path = PathBuf::from(model_path).join("3_Dense/model.safetensors");
        let vb2 = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dense2_path.to_str().unwrap()],
                candle_core::DType::F32,
                device,
            )
        }
        .map_err(|e| from_candle_error(e, "load 3_Dense safetensors", None))?;

        let dense2 = DenseLayer::load(vb2, 3072, 768, DenseActivation::Identity, false)?;

        Ok(Self { dense1, dense2 })
    }

    /// Forward pass through bottleneck
    ///
    /// # Arguments
    /// - `embeddings`: Input tensor [batch_size, 768] from mean pooling
    ///
    /// # Returns
    /// - Output tensor [batch_size, 768] after bottleneck transformation
    ///
    /// # Errors
    /// - Shape mismatch if input is not [*, 768]
    /// - Candle error if tensor operations fail
    ///
    /// # Example
    /// ```ignore
    /// // After mean pooling: [batch_size, 768]
    /// let pooled = mean_pool(&hidden_states, &attention_mask)?;
    ///
    /// // Apply bottleneck
    /// let transformed = bottleneck.forward(&pooled)?;  // [batch_size, 768]
    ///
    /// // L2 normalize
    /// let normalized = l2_normalize(&transformed)?;
    /// ```
    pub fn forward(&self, embeddings: &Tensor) -> UnifiedResult<Tensor> {
        // Validate input shape
        let shape = embeddings.dims();
        let last_dim = shape[shape.len() - 1];
        if last_dim != 768 {
            return Err(UnifiedError::Validation {
                field: "input dimension".to_string(),
                expected: "768".to_string(),
                actual: last_dim.to_string(),
                context: Some(
                    "Bottleneck expects input dimension of 768 from mean pooling".to_string(),
                ),
            });
        }

        // First dense layer: 768 → 3072 (expansion)
        let expanded = self.dense1.forward(embeddings)?;

        // Second dense layer: 3072 → 768 (compression)
        let compressed = self.dense2.forward(&expanded)?;

        Ok(compressed)
    }

    /// Get the first dense layer (expansion)
    pub fn expansion_layer(&self) -> &DenseLayer {
        &self.dense1
    }

    /// Get the second dense layer (compression)
    pub fn compression_layer(&self) -> &DenseLayer {
        &self.dense2
    }
}
