//! Traditional model base class
//!
//! Provides abstract base functionality for all traditional models
//! in the dual-path architecture.

use crate::core::{ModelErrorType, UnifiedError};
use crate::model_architectures::traits::TraditionalModel;
use crate::model_error;
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};
use rayon::prelude::*;
use std::collections::HashMap;

/// Abstract base class for traditional models
pub trait TraditionalModelBase {
    /// Model configuration type
    type Config: Clone + Send + Sync;

    /// Load model with configuration
    fn load_model(config: &Self::Config, device: &Device) -> Result<Self>
    where
        Self: Sized;

    /// Forward pass through the model
    fn forward_pass(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor>;

    /// Get model embeddings for text
    fn get_embeddings(&self, text: &str) -> Result<Tensor>;

    /// Get model configuration
    fn get_config(&self) -> &Self::Config;

    /// Get model device
    fn get_device(&self) -> &Device;

    /// Check if model supports batch processing
    fn supports_batch_processing(&self) -> bool {
        true
    }

    /// Get maximum sequence length
    fn max_sequence_length(&self) -> usize {
        512
    }
}

/// Base traditional model implementation
#[derive(Debug)]
pub struct BaseTraditionalModel {
    config: BaseModelConfig,
    device: Device,
    embeddings: ModelEmbeddings,
    encoder: ModelEncoder,
    pooler: Option<ModelPooler>,
}

impl BaseTraditionalModel {
    /// Create new base traditional model
    pub fn new(config: BaseModelConfig, vb: VarBuilder, device: Device) -> Result<Self> {
        let embeddings = ModelEmbeddings::new(&config, vb.pp("embeddings"), &device)?;
        let encoder = ModelEncoder::new(&config, vb.pp("encoder"), &device)?;
        let pooler = if config.add_pooling_layer {
            Some(ModelPooler::new(&config, vb.pp("pooler"), &device)?)
        } else {
            None
        };

        Ok(Self {
            config,
            device,
            embeddings,
            encoder,
            pooler,
        })
    }

    /// Forward pass through the model
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Encoder layers
        hidden_states = self.encoder.forward(&hidden_states, attention_mask)?;

        // Optional pooling
        if let Some(pooler) = &self.pooler {
            hidden_states = pooler.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }

    /// Get embeddings for classification
    pub fn get_classification_embeddings(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let hidden_states = self.forward(input_ids, attention_mask)?;

        // Extract CLS token or apply pooling
        match self.config.pooling_strategy {
            PoolingStrategy::CLS => {
                // Take [CLS] token (first token)
                hidden_states.i((.., 0, ..))
            }
            PoolingStrategy::Mean => {
                // Mean pooling over sequence length
                self.mean_pooling(&hidden_states, attention_mask)
            }
            PoolingStrategy::Max => {
                // Max pooling over sequence length
                self.max_pooling(&hidden_states)
            }
        }
    }

    /// Batch processing for multiple inputs
    ///
    /// Uses rayon for parallel processing of independent forward passes.
    /// Thread-safe since forward() only reads model weights without modification.
    pub fn forward_batch(
        &self,
        input_batch: &[Tensor],
        attention_batch: &[Tensor],
    ) -> Result<Vec<Tensor>> {
        // Parallel processing of batch items
        input_batch
            .par_iter()
            .zip(attention_batch.par_iter())
            .map(|(input_ids, attention_mask)| self.forward(input_ids, attention_mask))
            .collect()
    }

    // Pooling strategies
    fn mean_pooling(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match hidden states dimensions
        let expanded_mask = attention_mask.unsqueeze(2)?.expand(hidden_states.shape())?;

        // Apply mask and sum
        let masked_hidden = hidden_states.mul(&expanded_mask)?;
        let sum_hidden = masked_hidden.sum_keepdim(1)?;

        // Count valid tokens
        let mask_sum = expanded_mask.sum_keepdim(1)?;
        let mask_sum = mask_sum.clamp(1e-9, f32::INFINITY)?; // Avoid division by zero

        // Average
        sum_hidden.div(&mask_sum)
    }

    fn max_pooling(&self, hidden_states: &Tensor) -> Result<Tensor> {
        hidden_states.max_keepdim(1)
    }
}

/// Model embeddings layer
#[derive(Debug)]
pub struct ModelEmbeddings {
    word_embeddings: candle_nn::Embedding,
    position_embeddings: Option<candle_nn::Embedding>,
    token_type_embeddings: Option<candle_nn::Embedding>,
    layer_norm: candle_nn::LayerNorm,
    dropout: candle_nn::Dropout,
    config: BaseModelConfig,
}

impl ModelEmbeddings {
    pub fn new(config: &BaseModelConfig, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("word_embeddings"),
        )?;

        let position_embeddings = if config.use_position_embeddings {
            Some(candle_nn::embedding(
                config.max_position_embeddings,
                config.hidden_size,
                vb.pp("position_embeddings"),
            )?)
        } else {
            None
        };

        let token_type_embeddings = if config.use_token_type_embeddings {
            Some(candle_nn::embedding(
                config.type_vocab_size,
                config.hidden_size,
                vb.pp("token_type_embeddings"),
            )?)
        } else {
            None
        };

        let layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob as f32);

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout,
            config: config.clone(),
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.shape().dims()[1];

        // Word embeddings
        let mut embeddings = self.word_embeddings.forward(input_ids)?;

        // Position embeddings
        if let Some(pos_emb) = &self.position_embeddings {
            let position_ids =
                Tensor::arange(0i64, seq_length as i64, input_ids.device())?.unsqueeze(0)?;
            let position_embeds = pos_emb.forward(&position_ids)?;
            embeddings = embeddings.add(&position_embeds)?;
        }

        // Token type embeddings
        if let Some(type_emb) = &self.token_type_embeddings {
            let token_type_ids =
                Tensor::zeros(input_ids.shape().dims(), DType::I64, input_ids.device())?;
            let token_type_embeds = type_emb.forward(&token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeds)?;
        }

        // Layer normalization and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings, false)
    }
}

/// Model encoder with transformer layers
#[derive(Debug)]
pub struct ModelEncoder {
    layers: Vec<TransformerLayer>,
    config: BaseModelConfig,
}

impl ModelEncoder {
    pub fn new(config: &BaseModelConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);

        for i in 0..config.num_hidden_layers {
            let layer = TransformerLayer::new(config, vb.pp(&format!("layer.{}", i)), device)?;
            layers.push(layer);
        }

        Ok(Self {
            layers,
            config: config.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut current_hidden = hidden_states.clone();

        for layer in &self.layers {
            current_hidden = layer.forward(&current_hidden, attention_mask)?;
        }

        Ok(current_hidden)
    }
}

/// Single transformer layer
#[derive(Debug)]
pub struct TransformerLayer {
    attention: SelfAttention,
    intermediate: candle_nn::Linear,
    output: candle_nn::Linear,
    attention_layer_norm: candle_nn::LayerNorm,
    output_layer_norm: candle_nn::LayerNorm,
    dropout: candle_nn::Dropout,
}

impl TransformerLayer {
    pub fn new(config: &BaseModelConfig, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let attention = SelfAttention::new(config, vb.pp("attention"))?;
        let intermediate = candle_nn::linear(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("intermediate.dense"),
        )?;
        let output = candle_nn::linear(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("output.dense"),
        )?;
        let attention_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attention.output.LayerNorm"),
        )?;
        let output_layer_norm = candle_nn::layer_norm(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("output.LayerNorm"),
        )?;
        let dropout = candle_nn::Dropout::new(config.hidden_dropout_prob as f32);

        Ok(Self {
            attention,
            intermediate,
            output,
            attention_layer_norm,
            output_layer_norm,
            dropout,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.dropout.forward(&attention_output, false)?;
        let attention_output = self
            .attention_layer_norm
            .forward(&(hidden_states + attention_output)?)?;

        // Feed-forward network
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let intermediate_output = match self.attention.config.hidden_act {
            ActivationFunction::Gelu => intermediate_output.gelu()?,
            ActivationFunction::Relu => intermediate_output.relu()?,
            ActivationFunction::Swish => intermediate_output.silu()?,
        };

        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self.dropout.forward(&layer_output, false)?;
        let layer_output = self
            .output_layer_norm
            .forward(&(attention_output + layer_output)?)?;

        Ok(layer_output)
    }
}

/// Self-attention mechanism
#[derive(Debug)]
pub struct SelfAttention {
    query: candle_nn::Linear,
    key: candle_nn::Linear,
    value: candle_nn::Linear,
    output: candle_nn::Linear,
    dropout: candle_nn::Dropout,
    config: BaseModelConfig,
}

impl SelfAttention {
    pub fn new(config: &BaseModelConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let query = candle_nn::linear(hidden_size, hidden_size, vb.pp("self.query"))?;
        let key = candle_nn::linear(hidden_size, hidden_size, vb.pp("self.key"))?;
        let value = candle_nn::linear(hidden_size, hidden_size, vb.pp("self.value"))?;
        let output = candle_nn::linear(hidden_size, hidden_size, vb.pp("output.dense"))?;
        let dropout = candle_nn::Dropout::new(config.attention_probs_dropout_prob as f32);

        Ok(Self {
            query,
            key,
            value,
            output,
            dropout,
            config: config.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let batch_size = hidden_states.shape().dims()[0];
        let seq_length = hidden_states.shape().dims()[1];
        let num_attention_heads = self.config.num_attention_heads;
        let attention_head_size = self.config.hidden_size / num_attention_heads;

        // Linear projections
        let query_layer = self.query.forward(hidden_states)?;
        let key_layer = self.key.forward(hidden_states)?;
        let value_layer = self.value.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query_layer = query_layer
            .reshape((
                batch_size,
                seq_length,
                num_attention_heads,
                attention_head_size,
            ))?
            .transpose(1, 2)?;

        let key_layer = key_layer
            .reshape((
                batch_size,
                seq_length,
                num_attention_heads,
                attention_head_size,
            ))?
            .transpose(1, 2)?;

        let value_layer = value_layer
            .reshape((
                batch_size,
                seq_length,
                num_attention_heads,
                attention_head_size,
            ))?
            .transpose(1, 2)?;

        // Scaled dot-product attention
        let attention_scores = query_layer.matmul(&key_layer.transpose(2, 3)?)?;
        let attention_scores = attention_scores.div(&Tensor::new(
            (attention_head_size as f32).sqrt(),
            hidden_states.device(),
        )?)?;

        // Apply attention mask
        let attention_scores = if attention_mask.rank() > 0 {
            // Apply attention mask using where_cond (candle alternative to masked_fill)
            let mask = attention_mask.unsqueeze(1)?.unsqueeze(2)?;
            let mask = mask.expand(attention_scores.shape())?;
            let zero_tensor = Tensor::zeros_like(&mask)?;
            let neg_inf_tensor = Tensor::full(
                f32::NEG_INFINITY,
                attention_scores.shape(),
                attention_scores.device(),
            )?;

            // Use where_cond: where mask==0, use neg_inf, otherwise use original scores
            let mask_condition = mask.eq(&zero_tensor)?;
            mask_condition.where_cond(&neg_inf_tensor, &attention_scores)?
        } else {
            attention_scores
        };

        // Softmax
        let attention_probs = candle_nn::ops::softmax(&attention_scores, candle_core::D::Minus1)?;
        let attention_probs = self.dropout.forward(&attention_probs, false)?;

        // Apply attention to values
        let context_layer = attention_probs.matmul(&value_layer)?;
        let context_layer = context_layer.transpose(1, 2)?.reshape((
            batch_size,
            seq_length,
            self.config.hidden_size,
        ))?;

        // Output projection
        self.output.forward(&context_layer)
    }
}

/// Optional pooling layer
#[derive(Debug)]
pub struct ModelPooler {
    dense: candle_nn::Linear,
    activation: ActivationFunction,
}

impl ModelPooler {
    pub fn new(config: &BaseModelConfig, vb: VarBuilder, _device: &Device) -> Result<Self> {
        let dense = candle_nn::linear(config.hidden_size, config.hidden_size, vb.pp("dense"))?;

        Ok(Self {
            dense,
            activation: config.pooler_activation.clone(),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Take [CLS] token
        let first_token_tensor = hidden_states.i((.., 0))?;
        let pooled_output = self.dense.forward(&first_token_tensor)?;

        match self.activation {
            ActivationFunction::Gelu => pooled_output.gelu(),
            ActivationFunction::Relu => pooled_output.relu(),
            ActivationFunction::Swish => pooled_output.silu(),
        }
    }
}

/// Base model configuration
#[derive(Debug, Clone)]
pub struct BaseModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub layer_norm_eps: f64,
    pub hidden_dropout_prob: f64,
    pub attention_probs_dropout_prob: f64,
    pub hidden_act: ActivationFunction,
    pub pooler_activation: ActivationFunction,
    pub use_position_embeddings: bool,
    pub use_token_type_embeddings: bool,
    pub add_pooling_layer: bool,
    pub pooling_strategy: PoolingStrategy,
}

impl Default for BaseModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            hidden_dropout_prob: {
                use crate::core::config_loader::GlobalConfigLoader;
                GlobalConfigLoader::load_router_config_safe().traditional_dropout_prob as f64
            },
            attention_probs_dropout_prob: {
                use crate::core::config_loader::GlobalConfigLoader;
                GlobalConfigLoader::load_router_config_safe().traditional_attention_dropout_prob
                    as f64
            },
            hidden_act: ActivationFunction::Gelu,
            pooler_activation: ActivationFunction::Gelu,
            use_position_embeddings: true,
            use_token_type_embeddings: true,
            add_pooling_layer: true,
            pooling_strategy: PoolingStrategy::CLS,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone)]
pub enum ActivationFunction {
    Gelu,
    Relu,
    Swish,
}

/// Pooling strategy for sequence representation
#[derive(Debug, Clone)]
pub enum PoolingStrategy {
    CLS,  // Use [CLS] token
    Mean, // Mean pooling
    Max,  // Max pooling
}

impl TraditionalModelBase for BaseTraditionalModel {
    type Config = BaseModelConfig;

    fn load_model(config: &Self::Config, device: &Device) -> Result<Self> {
        let vb = VarBuilder::zeros(DType::F32, device);
        Self::new(config.clone(), vb, device.clone())
    }

    fn forward_pass(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        self.forward(input_ids, attention_mask)
    }

    fn get_embeddings(&self, _text: &str) -> Result<Tensor> {
        // This would require tokenization, simplified for now
        let unified_err = model_error!(
            ModelErrorType::Traditional,
            "embedding extraction",
            "Not implemented in base class",
            "BaseTraditionalModel"
        );
        Err(candle_core::Error::from(unified_err))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn get_device(&self) -> &Device {
        &self.device
    }

    fn max_sequence_length(&self) -> usize {
        self.config.max_position_embeddings
    }
}
