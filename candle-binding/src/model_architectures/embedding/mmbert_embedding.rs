//! mmBERT Embedding Model Implementation (32K Context, 2D Matryoshka)
//!
//! This module implements the mmBERT-Embed-32K-2D-Matryoshka model, a multilingual
//! embedding model based on ModernBERT architecture with YaRN-scaled RoPE for 32K context.
//!
//! ## Model Highlights
//! - **Parameters**: 307M
//! - **Context Length**: 32,768 tokens
//! - **Languages**: 1800+ (via Glot500)
//! - **Embedding Dim**: 768 (supports 64-768 via Matryoshka)
//! - **Architecture**: ModernBERT encoder with YaRN scaling
//!
//! ## 2D Matryoshka Support (FULLY IMPLEMENTED)
//! This model supports two dimensions of flexibility:
//! 1. **Dimension Reduction** (Matryoshka): Truncate embeddings to smaller dimensions
//! 2. **Layer Reduction** (Adaptive): Use intermediate layer outputs for faster inference
//!
//! ## Architecture Details
//! - Layers: 22 transformer blocks
//! - Hidden size: 768
//! - Attention heads: 12
//! - Local attention window: 128 tokens
//! - Global attention: Every 3 layers
//! - RoPE theta: 160000 (YaRN-scaled for 32K)
//!
//! ## References
//! - Model: https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka
//! - Base: https://huggingface.co/jhu-clsp/mmBERT-base
//! - Paper: YaRN: Efficient Context Window Extension of Large Language Models

use crate::core::{config_errors, from_candle_error, UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::pooling::mean_pool;
use crate::model_architectures::traits::{
    EmbeddingPathSpecialization, LongContextEmbeddingCapable, ModelType, PoolingMethod,
};
use crate::model_architectures::unified_interface::CoreModel;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{
    embedding, layer_norm_no_bias, linear_no_bias, Embedding, LayerNorm, Linear, VarBuilder,
};
use std::path::Path;
use std::sync::Arc;

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
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
}

impl MmBertEmbeddingConfig {
    /// Load configuration from a pretrained model directory
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let config_path = model_path.as_ref().join("config.json");

        if !config_path.exists() {
            return Err(config_errors::file_not_found(
                &config_path.display().to_string(),
            ));
        }

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| config_errors::file_not_found(&config_path.display().to_string()))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            config_errors::invalid_json(&config_path.display().to_string(), &e.to_string())
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
            global_attn_every_n_layers: config_json["global_attn_every_n_layers"]
                .as_u64()
                .unwrap_or(3) as usize,
            global_rope_theta: config_json["global_rope_theta"]
                .as_f64()
                .unwrap_or(160000.0),
            local_attention: config_json["local_attention"].as_u64().unwrap_or(128) as usize,
            local_rope_theta: config_json["local_rope_theta"].as_f64().unwrap_or(160000.0),
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
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

    pub fn estimate_speedup(&self, layer: usize) -> f32 {
        22.0 / layer as f32
    }
}

// ============================================================================
// Rotary Position Embedding
// ============================================================================

#[derive(Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        config: &MmBertEmbeddingConfig,
        rope_theta: f64,
        device: &Device,
    ) -> candle_core::Result<Self> {
        let dim = config.head_dim();
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

// ============================================================================
// Attention
// ============================================================================

#[derive(Clone)]
struct MmBertAttention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl MmBertAttention {
    fn load(
        vb: VarBuilder,
        config: &MmBertEmbeddingConfig,
        rotary_emb: Arc<RotaryEmbedding>,
    ) -> candle_core::Result<Self> {
        let num_attention_heads = config.num_attention_heads;
        let attention_head_size = config.hidden_size / config.num_attention_heads;

        let qkv = linear_no_bias(config.hidden_size, config.hidden_size * 3, vb.pp("Wqkv"))?;
        let proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("Wo"))?;

        Ok(Self {
            qkv,
            proj,
            num_attention_heads,
            attention_head_size,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, seq_len, d) = hidden_states.dims3()?;
        let qkv = hidden_states
            .apply(&self.qkv)?
            .reshape((
                b,
                seq_len,
                3,
                self.num_attention_heads,
                self.attention_head_size,
            ))?
            .permute((2, 0, 3, 1, 4))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let (q, k) = self.rotary_emb.apply_rotary_emb_qkv(&q, &k)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);
        let q = (q * scale)?;

        // Ensure contiguous tensors for matmul
        let q = q.contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let att = q.matmul(&k_t)?;
        let att = att.broadcast_add(attention_mask)?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;

        // Ensure v is contiguous for matmul
        let v = v.contiguous()?;
        let xs = att.matmul(&v)?;
        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        xs.apply(&self.proj)
    }
}

// ============================================================================
// MLP
// ============================================================================

#[derive(Clone)]
struct MmBertMLP {
    wi: Linear,
    wo: Linear,
}

impl MmBertMLP {
    fn load(vb: VarBuilder, config: &MmBertEmbeddingConfig) -> candle_core::Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for MmBertMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo)
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

#[derive(Clone)]
struct MmBertLayer {
    attn: MmBertAttention,
    mlp: MmBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl MmBertLayer {
    fn load(
        vb: VarBuilder,
        config: &MmBertEmbeddingConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> candle_core::Result<Self> {
        let attn = MmBertAttention::load(vb.pp("attn"), config, rotary_emb)?;
        let mlp = MmBertMLP::load(vb.pp("mlp"), config)?;
        let attn_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("attn_norm"),
        )
        .ok();
        let mlp_norm =
            layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("mlp_norm"))?;
        Ok(Self {
            attn,
            mlp,
            attn_norm,
            mlp_norm,
            uses_local_attention,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        global_attention_mask: &Tensor,
        local_attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let attention_mask = if self.uses_local_attention {
            &global_attention_mask.broadcast_add(local_attention_mask)?
        } else {
            global_attention_mask
        };
        let xs = self.attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        xs + mlp_out
    }
}

// ============================================================================
// Attention Mask Helpers
// ============================================================================

fn prepare_4d_attention_mask(
    mask: &Tensor,
    dtype: DType,
    tgt_len: Option<usize>,
) -> candle_core::Result<Tensor> {
    let bsz = mask.dim(0)?;
    let src_len = mask.dim(1)?;
    let tgt_len = tgt_len.unwrap_or(src_len);

    let expanded_mask = mask
        .unsqueeze(1)?
        .unsqueeze(2)?
        .expand((bsz, 1, tgt_len, src_len))?
        .to_dtype(dtype)?;

    let inverted_mask = (1.0 - expanded_mask)?;
    (inverted_mask * f32::MIN as f64)?.to_dtype(dtype)
}

fn get_local_attention_mask(
    seq_len: usize,
    max_distance: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mask: Vec<_> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len).map(move |j| {
                if (j as i32 - i as i32).abs() > max_distance as i32 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (seq_len, seq_len), device)
}

// ============================================================================
// mmBERT Encoder with Layer-by-Layer Control
// ============================================================================

/// Custom ModernBERT encoder with layer-by-layer control for 2D Matryoshka
#[derive(Clone)]
struct MmBertEncoder {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<MmBertLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
}

impl MmBertEncoder {
    fn load(vb: VarBuilder, config: &MmBertEmbeddingConfig) -> candle_core::Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("embeddings.norm"),
        )?;

        let global_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.global_rope_theta,
            vb.device(),
        )?);
        let local_rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            config,
            config.local_rope_theta,
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_id in 0..config.num_hidden_layers {
            let layer_uses_local_attention = layer_id % config.global_attn_every_n_layers != 0;
            layers.push(MmBertLayer::load(
                vb.pp(format!("layers.{layer_id}")),
                config,
                if layer_uses_local_attention {
                    local_rotary_emb.clone()
                } else {
                    global_rotary_emb.clone()
                },
                layer_uses_local_attention,
            )?);
        }

        let final_norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
        })
    }

    /// Forward pass through all layers
    fn forward(&self, xs: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_to_layer(xs, mask, self.layers.len())
    }

    /// Forward pass with early exit at specified layer (1-indexed)
    /// For example, target_layer=6 will run layers 0-5 (6 layers total)
    fn forward_to_layer(
        &self,
        xs: &Tensor,
        mask: &Tensor,
        target_layer: usize,
    ) -> candle_core::Result<Tensor> {
        let seq_len = xs.shape().dims()[1];
        let global_attention_mask =
            prepare_4d_attention_mask(mask, DType::F32, None)?.to_device(xs.device())?;
        let local_attention_mask =
            get_local_attention_mask(seq_len, self.local_attention_size / 2, xs.device())?;

        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;

        // Only iterate through layers up to target_layer
        let num_layers_to_run = target_layer.min(self.layers.len());
        for layer in self.layers.iter().take(num_layers_to_run) {
            xs = layer.forward(&xs, &global_attention_mask, &local_attention_mask)?;
        }

        xs.apply(&self.final_norm)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// Complete Model
// ============================================================================

/// mmBERT Embedding Model with full 2D Matryoshka support
pub struct MmBertEmbeddingModel {
    encoder: MmBertEncoder,
    config: MmBertEmbeddingConfig,
    matryoshka_config: MatryoshkaConfig,
    device: Device,
}

impl std::fmt::Debug for MmBertEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmBertEmbeddingModel")
            .field("config", &self.config)
            .field("matryoshka_config", &self.matryoshka_config)
            .field("device", &self.device)
            .finish()
    }
}

impl MmBertEmbeddingModel {
    /// Load model from pretrained directory
    pub fn load(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        let config = MmBertEmbeddingConfig::from_pretrained(model_path)?;

        let safetensors_path = format!("{}/model.safetensors", model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path.clone()], DType::F32, device)
                .map_err(|e| {
                    from_candle_error(
                        e,
                        &format!("failed to load safetensors from {}", safetensors_path),
                        Some(model_path),
                    )
                })?
        };

        Self::load_with_vb(model_path, &config, vb, device)
    }

    /// Load model with existing VarBuilder
    pub fn load_with_vb(
        model_path: &str,
        config: &MmBertEmbeddingConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> UnifiedResult<Self> {
        // Try loading with different prefixes
        let encoder = MmBertEncoder::load(vb.clone(), config)
            .or_else(|_| MmBertEncoder::load(vb.pp("model"), config))
            .or_else(|_| MmBertEncoder::load(vb.pp("_orig_mod"), config))
            .or_else(|_| MmBertEncoder::load(vb.pp("_orig_mod.model"), config))
            .map_err(|e| from_candle_error(e, "failed to load MmBertEncoder", Some(model_path)))?;

        Ok(Self {
            encoder,
            config: config.clone(),
            matryoshka_config: MatryoshkaConfig::default(),
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &MmBertEmbeddingConfig {
        &self.config
    }

    pub fn matryoshka_config(&self) -> &MatryoshkaConfig {
        &self.matryoshka_config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_layers(&self) -> usize {
        self.encoder.num_layers()
    }

    /// Forward pass to generate embeddings (full model)
    pub fn embedding_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        self.embedding_forward_with_matryoshka(input_ids, attention_mask, None, None)
    }

    /// Forward pass with 2D Matryoshka support (layer early exit + dimension truncation)
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs
    /// - `attention_mask`: Optional attention mask
    /// - `target_layer`: Layer for early exit (1-indexed, e.g., 6 means use first 6 layers)
    /// - `target_dim`: Dimension for truncation (e.g., 256 for 33% storage)
    pub fn embedding_forward_with_matryoshka(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Tensor> {
        let (batch_size, seq_len) = input_ids
            .dims2()
            .map_err(|e| from_candle_error(e, "get input dims", None))?;

        // Validate and default target_layer
        let num_layers = self.encoder.num_layers();
        let target_layer = target_layer.unwrap_or(num_layers);
        if target_layer > num_layers || target_layer == 0 {
            return Err(UnifiedError::Validation {
                field: "target_layer".to_string(),
                expected: format!("1 to {}", num_layers),
                actual: target_layer.to_string(),
                context: Some("Layer must be between 1 and num_layers".to_string()),
            });
        }

        // Validate target_dim
        let hidden_size = self.config.hidden_size;
        let target_dim = target_dim.unwrap_or(hidden_size);
        if target_dim > hidden_size {
            return Err(UnifiedError::Validation {
                field: "target_dim".to_string(),
                expected: format!("<= {}", hidden_size),
                actual: target_dim.to_string(),
                context: None,
            });
        }

        // Create default mask if not provided
        let default_mask;
        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => {
                default_mask = Tensor::ones((batch_size, seq_len), DType::U32, &self.device)
                    .map_err(|e| from_candle_error(e, "create default mask", None))?;
                default_mask
            }
        };

        // Forward through encoder with early exit
        let hidden_states = self
            .encoder
            .forward_to_layer(input_ids, &mask, target_layer)
            .map_err(|e| {
                from_candle_error(
                    e,
                    &format!("encoder forward to layer {}", target_layer),
                    None,
                )
            })?;

        // Mean pooling
        let mask_f32 = mask
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "mask to f32", None))?;
        let embeddings =
            mean_pool(&hidden_states, &mask_f32).map_err(|e| UnifiedError::Processing {
                operation: "mean_pool".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        // Dimension truncation
        let embeddings = if target_dim < hidden_size {
            embeddings
                .narrow(1, 0, target_dim)
                .map_err(|e| from_candle_error(e, "dimension truncation", None))?
        } else {
            embeddings
        };

        // L2 normalize
        self.l2_normalize(&embeddings)
    }

    /// Convenience method for batch encoding
    pub fn encode_batch(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        texts: &[&str],
        max_length: usize,
    ) -> UnifiedResult<Tensor> {
        self.encode_batch_with_matryoshka(tokenizer, texts, max_length, None, None)
    }

    /// Batch encoding with 2D Matryoshka support
    pub fn encode_batch_with_matryoshka(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        texts: &[&str],
        max_length: usize,
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Tensor> {
        let encodings =
            tokenizer
                .encode_batch(texts.to_vec(), true)
                .map_err(|e| UnifiedError::Processing {
                    operation: "tokenize batch".to_string(),
                    source: e.to_string(),
                    input_context: None,
                })?;

        let batch_size = encodings.len();
        let seq_len = max_length.min(
            encodings
                .iter()
                .map(|e| e.get_ids().len())
                .max()
                .unwrap_or(0),
        );

        let mut input_ids_vec = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_vec = Vec::with_capacity(batch_size * seq_len);

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();

            for i in 0..seq_len {
                if i < ids.len() {
                    input_ids_vec.push(ids[i]);
                    attention_mask_vec.push(mask[i] as u32);
                } else {
                    input_ids_vec.push(0);
                    attention_mask_vec.push(0);
                }
            }
        }

        let input_ids = Tensor::from_vec(input_ids_vec, (batch_size, seq_len), &self.device)
            .map_err(|e| from_candle_error(e, "create input_ids tensor", None))?;
        let attention_mask =
            Tensor::from_vec(attention_mask_vec, (batch_size, seq_len), &self.device)
                .map_err(|e| from_candle_error(e, "create attention_mask tensor", None))?;

        self.embedding_forward_with_matryoshka(
            &input_ids,
            Some(&attention_mask),
            target_layer,
            target_dim,
        )
    }

    fn l2_normalize(&self, embeddings: &Tensor) -> UnifiedResult<Tensor> {
        let squared = embeddings
            .sqr()
            .map_err(|e| from_candle_error(e, "L2 sqr", None))?;
        let sum_squared = squared
            .sum_keepdim(1)
            .map_err(|e| from_candle_error(e, "L2 sum", None))?;
        let norm = sum_squared
            .sqrt()
            .map_err(|e| from_candle_error(e, "L2 sqrt", None))?;
        let norm_safe = (norm + 1e-12).map_err(|e| from_candle_error(e, "L2 eps", None))?;
        embeddings
            .broadcast_div(&norm_safe)
            .map_err(|e| from_candle_error(e, "L2 div", None))
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl CoreModel for MmBertEmbeddingModel {
    type Config = MmBertEmbeddingConfig;
    type Error = UnifiedError;
    type Output = Tensor;

    fn model_type(&self) -> ModelType {
        ModelType::MmBertEmbedding
    }

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

impl LongContextEmbeddingCapable for MmBertEmbeddingModel {
    fn get_max_sequence_length(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn get_embedding_dimension(&self) -> usize {
        self.config.hidden_size
    }

    fn get_pooling_method(&self) -> PoolingMethod {
        PoolingMethod::Mean
    }

    fn supports_matryoshka(&self) -> bool {
        true
    }

    fn get_matryoshka_dimensions(&self) -> Vec<usize> {
        self.matryoshka_config.dimensions.clone()
    }

    fn supports_instruction_aware(&self) -> bool {
        false
    }

    fn extract_embeddings(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        target_dim: Option<usize>,
    ) -> Result<Tensor, Self::Error> {
        let embeddings =
            mean_pool(hidden_states, attention_mask).map_err(|e| UnifiedError::Processing {
                operation: "extract_embeddings".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        if let Some(dim) = target_dim {
            if dim > self.config.hidden_size {
                return Err(UnifiedError::Validation {
                    field: "target_dim".to_string(),
                    expected: format!("<= {}", self.config.hidden_size),
                    actual: dim.to_string(),
                    context: None,
                });
            }
            embeddings
                .narrow(1, 0, dim)
                .map_err(|e| from_candle_error(e, "truncation", None))
        } else {
            Ok(embeddings)
        }
    }

    fn optimal_embedding_batch_size(&self) -> usize {
        32
    }

    fn supports_parallel_batching(&self) -> bool {
        true
    }
}

impl EmbeddingPathSpecialization for MmBertEmbeddingModel {
    fn supports_parallel(&self) -> bool {
        true
    }

    fn optimal_batch_size(&self) -> usize {
        self.optimal_embedding_batch_size()
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

    /// Test that RoPE can be computed for 32K positions with YaRN theta
    /// This verifies the mathematical foundation without requiring model weights
    #[test]
    fn test_32k_rope_computation() {
        let device = Device::Cpu;

        // Create a minimal config for 32K context
        let config = MmBertEmbeddingConfig {
            vocab_size: 256000,
            hidden_size: 768,
            num_hidden_layers: 22,
            num_attention_heads: 12,
            intermediate_size: 1152,
            max_position_embeddings: 32768, // 32K!
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160000.0, // YaRN-scaled
            local_attention: 128,
            local_rope_theta: 160000.0,
        };

        // Verify config values
        assert_eq!(config.max_position_embeddings, 32768);
        assert_eq!(config.global_rope_theta, 160000.0);
        assert_eq!(config.head_dim(), 64); // 768 / 12

        // Create RoPE embeddings for 32K positions
        let rope = RotaryEmbedding::new(DType::F32, &config, config.global_rope_theta, &device)
            .expect("Failed to create RoPE for 32K");

        // Verify sin/cos tensors have correct shape
        assert_eq!(rope.sin.dims(), &[32768, 32]); // (max_seq_len, head_dim/2)
        assert_eq!(rope.cos.dims(), &[32768, 32]);

        // Verify values at position 0 (cos=1, sin=0)
        let cos_0: Vec<f32> = rope.cos.i(0).unwrap().to_vec1().unwrap();
        let sin_0: Vec<f32> = rope.sin.i(0).unwrap().to_vec1().unwrap();
        for i in 0..cos_0.len() {
            assert!((cos_0[i] - 1.0).abs() < 1e-5, "cos[0][{}]={}", i, cos_0[i]);
            assert!(sin_0[i].abs() < 1e-5, "sin[0][{}]={}", i, sin_0[i]);
        }

        // Verify values at position 32767 (last position) are finite
        let cos_last: Vec<f32> = rope.cos.i(32767).unwrap().to_vec1().unwrap();
        let sin_last: Vec<f32> = rope.sin.i(32767).unwrap().to_vec1().unwrap();
        for i in 0..cos_last.len() {
            assert!(cos_last[i].is_finite(), "cos[32767][{}] not finite", i);
            assert!(sin_last[i].is_finite(), "sin[32767][{}] not finite", i);
            // sin²+cos² = 1
            let sum_sq = sin_last[i] * sin_last[i] + cos_last[i] * cos_last[i];
            assert!(
                (sum_sq - 1.0).abs() < 1e-4,
                "sin²+cos²={} at pos 32767",
                sum_sq
            );
        }

        println!("✓ 32K RoPE computation verified");
        println!("  - sin/cos shape: [{}, {}]", 32768, 32);
        println!("  - Position 0: cos=1, sin=0 ✓");
        println!("  - Position 32767: finite values, sin²+cos²=1 ✓");
    }

    /// Test local attention mask generation for 32K sequences
    #[test]
    fn test_32k_local_attention_mask() {
        let device = Device::Cpu;
        let seq_len = 1024; // Test with manageable size
        let local_window = 128;
        let half_window = local_window / 2;

        // Generate mask (same logic as get_local_attention_mask)
        let mask = get_local_attention_mask(seq_len, half_window, &device)
            .expect("Failed to create local attention mask");

        assert_eq!(mask.dims(), &[seq_len, seq_len]);

        let mask_data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();

        // Verify diagonal is 0 (can attend to self)
        for i in 0..seq_len {
            let idx = i * seq_len + i;
            assert_eq!(mask_data[idx], 0.0, "Diagonal should be 0");
        }

        // Verify positions outside window are -inf
        let test_pos = seq_len / 2;
        let far_pos = test_pos + half_window + 10;
        if far_pos < seq_len {
            let idx = test_pos * seq_len + far_pos;
            assert!(mask_data[idx].is_infinite() && mask_data[idx].is_sign_negative());
        }

        println!("✓ Local attention mask verified for seq_len={}", seq_len);
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
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use tokenizers::Tokenizer;

    fn get_model_path() -> Option<String> {
        std::env::var("MMBERT_MODEL_PATH").ok()
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_load_model() {
        let model_path = get_model_path().expect("MMBERT_MODEL_PATH not set");
        let device = Device::Cpu;
        let model = MmBertEmbeddingModel::load(&model_path, &device).expect("Failed to load");
        assert_eq!(model.config().hidden_size, 768);
        assert_eq!(model.config().num_hidden_layers, 22);
        assert_eq!(model.num_layers(), 22);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_layer_early_exit() {
        let model_path = get_model_path().expect("MMBERT_MODEL_PATH not set");
        let device = Device::Cpu;
        let model = MmBertEmbeddingModel::load(&model_path, &device).expect("Failed to load");

        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

        let text = "Hello world";
        let encoding = tokenizer.encode(text, true).unwrap();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let seq_len = input_ids.len();

        let input_ids = Tensor::from_vec(input_ids, (1, seq_len), &device).unwrap();
        let attention_mask = Tensor::from_vec(attention_mask, (1, seq_len), &device).unwrap();

        // Test different exit layers
        for target_layer in [3, 6, 11, 22] {
            let embeddings = model
                .embedding_forward_with_matryoshka(
                    &input_ids,
                    Some(&attention_mask),
                    Some(target_layer),
                    None,
                )
                .expect(&format!("Failed at layer {}", target_layer));

            let shape = embeddings.dims();
            assert_eq!(shape[0], 1);
            assert_eq!(shape[1], 768);

            // Check normalized
            let norm: f32 = embeddings
                .sqr()
                .unwrap()
                .sum(1)
                .unwrap()
                .sqrt()
                .unwrap()
                .to_vec1()
                .unwrap()[0];
            assert!(
                (norm - 1.0).abs() < 0.01,
                "layer {}: norm={}",
                target_layer,
                norm
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_2d_matryoshka() {
        let model_path = get_model_path().expect("MMBERT_MODEL_PATH not set");
        let device = Device::Cpu;
        let model = MmBertEmbeddingModel::load(&model_path, &device).expect("Failed to load");

        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

        let text = "Test 2D Matryoshka";
        let encoding = tokenizer.encode(text, true).unwrap();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<u32> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as u32)
            .collect();
        let seq_len = input_ids.len();

        let input_ids = Tensor::from_vec(input_ids, (1, seq_len), &device).unwrap();
        let attention_mask = Tensor::from_vec(attention_mask, (1, seq_len), &device).unwrap();

        // Test 2D combinations
        for target_layer in [6, 11, 22] {
            for target_dim in [64, 256, 768] {
                let embeddings = model
                    .embedding_forward_with_matryoshka(
                        &input_ids,
                        Some(&attention_mask),
                        Some(target_layer),
                        Some(target_dim),
                    )
                    .expect(&format!("Failed at L{}/{}d", target_layer, target_dim));

                let shape = embeddings.dims();
                assert_eq!(shape[0], 1);
                assert_eq!(
                    shape[1], target_dim,
                    "Expected dim {}, got {}",
                    target_dim, shape[1]
                );
            }
        }
    }

    /// Test 32K context length support with actual model
    /// This test verifies the model can handle extended sequences
    ///
    /// For best performance, run with release mode and native CPU optimization:
    /// ```bash
    /// MMBERT_MODEL_PATH=models/mmbert-embed-32k-2d-matryoshka \
    /// RUSTFLAGS="-C target-cpu=native" \
    /// cargo test --release --no-default-features --lib test_32k_context_length -- --ignored --nocapture
    /// ```
    ///
    /// Performance (Intel Xeon Platinum 8568Y+ with AVX512):
    /// - 512 tokens: ~1.4s (release+AVX512) vs ~44s (debug)
    /// - 1024 tokens: ~4s (release+AVX512)
    /// - 2048 tokens: ~14s (release+AVX512)
    #[test]
    #[ignore = "requires model files"]
    fn test_32k_context_length() {
        let model_path = get_model_path().expect("MMBERT_MODEL_PATH not set");
        let device = Device::Cpu;

        println!("Loading model from: {}", model_path);
        let model = MmBertEmbeddingModel::load(&model_path, &device).expect("Failed to load");

        // Verify config supports 32K
        assert_eq!(
            model.config().max_position_embeddings,
            32768,
            "Model should support 32K positions"
        );
        assert!(
            model.config().global_rope_theta >= 100000.0,
            "Model should use YaRN-scaled RoPE theta"
        );

        println!("✓ Config verified:");
        println!(
            "  - max_position_embeddings: {}",
            model.config().max_position_embeddings
        );
        println!(
            "  - global_rope_theta: {} (YaRN)",
            model.config().global_rope_theta
        );

        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");

        // Test sequence lengths - release+AVX512 makes longer tests feasible
        // Debug mode: only test 128, 512 (O(n²) attention is too slow)
        // Release+AVX512: can test up to 2048+ in reasonable time
        #[cfg(debug_assertions)]
        let test_lengths = vec![128, 512];
        #[cfg(not(debug_assertions))]
        let test_lengths = vec![128, 512, 1024, 2048];

        for target_len in test_lengths {
            let base_text = "This is a test sentence for context length verification. ";
            let long_text = base_text.repeat(target_len / 8);

            let encoding = tokenizer
                .encode(long_text.as_str(), true)
                .expect("Failed to encode");

            let seq_len = encoding.get_ids().len().min(target_len);
            let input_ids: Vec<u32> = encoding.get_ids()[..seq_len].to_vec();
            let attention_mask: Vec<u32> = vec![1u32; seq_len];

            let input_ids = Tensor::from_vec(input_ids, (1, seq_len), &device)
                .expect("Failed to create input_ids tensor");
            let attention_mask = Tensor::from_vec(attention_mask, (1, seq_len), &device)
                .expect("Failed to create attention_mask tensor");

            println!("Testing seq_len={}...", seq_len);

            let start = std::time::Instant::now();
            let embeddings = model
                .embedding_forward(&input_ids, Some(&attention_mask))
                .expect(&format!("Failed at seq_len={}", seq_len));
            let elapsed = start.elapsed();

            let shape = embeddings.dims();
            assert_eq!(shape[0], 1);
            assert_eq!(shape[1], 768);

            // Verify normalized
            let norm: f32 = embeddings
                .sqr()
                .unwrap()
                .sum(1)
                .unwrap()
                .sqrt()
                .unwrap()
                .to_vec1()
                .unwrap()[0];
            assert!((norm - 1.0).abs() < 0.01, "norm={}", norm);

            println!(
                "  ✓ shape={:?}, norm={:.4}, time={:.2}s",
                shape,
                norm,
                elapsed.as_secs_f32()
            );
        }

        println!("✅ Context length test passed (32K support verified via config)");
    }

    /// Test that config is correctly loaded for 32K YaRN model
    /// This test runs without requiring the full model weights
    #[test]
    fn test_32k_config_loading() {
        // Try to load config from default model path
        let model_path = std::env::var("MMBERT_MODEL_PATH")
            .unwrap_or_else(|_| "../models/mmbert-embed-32k-2d-matryoshka".to_string());

        let config_path = format!("{}/config.json", model_path);
        if !std::path::Path::new(&config_path).exists() {
            println!("Skipping config test - model not found at: {}", model_path);
            println!("To run this test, download the model first:");
            println!("  make download-mmbert-embedding");
            return;
        }

        let config =
            MmBertEmbeddingConfig::from_pretrained(&model_path).expect("Failed to load config");

        // Verify 32K YaRN parameters
        assert_eq!(
            config.max_position_embeddings, 32768,
            "max_position_embeddings should be 32768"
        );
        assert_eq!(
            config.global_rope_theta, 160000.0,
            "global_rope_theta should be 160000 (YaRN-scaled)"
        );
        assert_eq!(
            config.local_rope_theta, 160000.0,
            "local_rope_theta should be 160000"
        );
        assert_eq!(config.hidden_size, 768, "hidden_size should be 768");
        assert_eq!(
            config.num_hidden_layers, 22,
            "num_hidden_layers should be 22"
        );
        assert_eq!(
            config.num_attention_heads, 12,
            "num_attention_heads should be 12"
        );
        assert_eq!(config.local_attention, 128, "local_attention should be 128");
        assert_eq!(
            config.global_attn_every_n_layers, 3,
            "global_attn_every_n_layers should be 3"
        );
        assert!(
            config.vocab_size >= 200000,
            "vocab_size should be >= 200000 (mmBERT)"
        );

        println!("✅ 32K YaRN config loaded and verified:");
        println!(
            "   - max_position_embeddings: {}",
            config.max_position_embeddings
        );
        println!(
            "   - global_rope_theta: {} (YaRN 4x scaling)",
            config.global_rope_theta
        );
        println!("   - vocab_size: {} (Gemma 2 tokenizer)", config.vocab_size);
        println!("   - hidden_size: {}", config.hidden_size);
        println!("   - num_layers: {}", config.num_hidden_layers);
        println!("   - num_heads: {}", config.num_attention_heads);
        println!("   - local_attention: {} tokens", config.local_attention);
    }
}
