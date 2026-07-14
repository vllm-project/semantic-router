//! Multi-Modal Embedding Model Implementation
//!
//! Implements the `multi-modal-embed-small` model from
//! <https://huggingface.co/llm-semantic-router/multi-modal-embed-small>
//!
//! ## Architecture (~120M parameters)
//! - **Text Encoder**: MiniLM-L6-v2 (22M params, 6 layers, 384-dim output)
//! - **Image Encoder**: SigLIP-base-patch16-512 (86M params, 768-dim → 384 via projection)
//! - **Audio Encoder**: Whisper-tiny encoder (8M params, 4 layers, 384-dim output)
//! - **Output**: 384-dim L2-normalized embeddings in a shared semantic space
//!
//! ## Features
//! - MRL (Matryoshka Representation Learning): truncate to 32, 64, 128, 256, 384
//! - 2DMSE: early exit at any text encoder layer (0–5)
//! - Cross-modal retrieval (text↔image, text↔audio, image↔audio)
//!
//! ## Weight Layout (model.safetensors / model.pt)
//! - `text_encoder.encoder.*` → MiniLM weights
//! - `image_encoder.vision_encoder.*` → SigLIP vision weights
//! - `image_encoder.projection.{weight,bias}` → 768→384 linear
//! - `audio_encoder.encoder.*` → Whisper encoder weights

use crate::core::{config_errors, from_candle_error, UnifiedError, UnifiedResult};
use crate::model_architectures::embedding::pooling::mean_pool;
use crate::model_architectures::traits::{
    EmbeddingPathSpecialization, LongContextEmbeddingCapable, ModelType, PoolingMethod,
};
use crate::model_architectures::unified_interface::CoreModel;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{embedding, layer_norm, linear, Embedding, LayerNorm, Linear, VarBuilder};
use std::path::Path;

// ============================================================================
// Configuration
// ============================================================================

/// Multi-modal embedding model configuration
#[derive(Debug, Clone)]
pub struct MultiModalEmbeddingConfig {
    pub embedding_dim: usize,
    pub text_hidden_size: usize,
    pub text_num_layers: usize,
    pub text_num_heads: usize,
    pub text_intermediate_size: usize,
    pub text_vocab_size: usize,
    pub text_max_position_embeddings: usize,
    pub text_type_vocab_size: usize,
    pub text_layer_norm_eps: f64,
    pub image_hidden_size: usize,
    pub image_patch_size: usize,
    pub image_size: usize,
    pub image_num_layers: usize,
    pub image_num_heads: usize,
    pub image_intermediate_size: usize,
    pub audio_hidden_size: usize,
    pub audio_num_layers: usize,
    pub audio_num_heads: usize,
    pub audio_num_mel_bins: usize,
    pub audio_max_source_positions: usize,
    pub matryoshka_dims: Vec<usize>,
}

impl Default for MultiModalEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 384,
            // MiniLM-L6-v2
            text_hidden_size: 384,
            text_num_layers: 6,
            text_num_heads: 12,
            text_intermediate_size: 1536,
            text_vocab_size: 30522,
            text_max_position_embeddings: 512,
            text_type_vocab_size: 2,
            text_layer_norm_eps: 1e-12,
            // SigLIP-base-patch16-512
            image_hidden_size: 768,
            image_patch_size: 16,
            image_size: 512,
            image_num_layers: 12,
            image_num_heads: 12,
            image_intermediate_size: 3072,
            // Whisper-tiny encoder
            audio_hidden_size: 384,
            audio_num_layers: 4,
            audio_num_heads: 6,
            audio_num_mel_bins: 80,
            audio_max_source_positions: 1500,
            matryoshka_dims: vec![384, 256, 128, 64, 32],
        }
    }
}

impl MultiModalEmbeddingConfig {
    /// Load configuration from a pretrained model directory.
    /// Falls back to defaults for any missing fields.
    pub fn from_pretrained<P: AsRef<Path>>(model_path: P) -> UnifiedResult<Self> {
        let config_path = model_path.as_ref().join("config.json");

        if !config_path.exists() {
            return Ok(Self::default());
        }

        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|_| config_errors::file_not_found(&config_path.display().to_string()))?;

        let v: serde_json::Value = serde_json::from_str(&config_str).map_err(|e| {
            config_errors::invalid_json(&config_path.display().to_string(), &e.to_string())
        })?;

        let mut cfg = Self::default();
        if let Some(d) = v["embedding_dim"].as_u64() {
            cfg.embedding_dim = d as usize;
        }
        if let Some(d) = v["text_hidden_size"].as_u64() {
            cfg.text_hidden_size = d as usize;
        }
        if let Some(d) = v["text_num_layers"].as_u64() {
            cfg.text_num_layers = d as usize;
        }
        if let Some(dims) = v["matryoshka_dims"].as_array() {
            cfg.matryoshka_dims = dims
                .iter()
                .filter_map(|d| d.as_u64().map(|x| x as usize))
                .collect();
        }
        Ok(cfg)
    }

    pub fn num_image_patches(&self) -> usize {
        (self.image_size / self.image_patch_size).pow(2)
    }
}

/// Matryoshka configuration for multi-modal model
#[derive(Debug, Clone)]
pub struct MultiModalMatryoshkaConfig {
    pub dimensions: Vec<usize>,
    pub layers: Vec<usize>,
}

impl Default for MultiModalMatryoshkaConfig {
    fn default() -> Self {
        Self {
            dimensions: vec![384, 256, 128, 64, 32],
            layers: vec![1, 2, 3, 4, 5, 6],
        }
    }
}

impl MultiModalMatryoshkaConfig {
    pub fn validate_dimension(&self, dim: usize) -> bool {
        self.dimensions.contains(&dim)
    }

    pub fn validate_layer(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }
}

// ============================================================================
// MiniLM Text Encoder (BERT-like, 6 layers)
// ============================================================================

#[derive(Clone)]
struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
}

impl BertEmbeddings {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let word_embeddings = embedding(
            config.text_vocab_size,
            config.text_hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = embedding(
            config.text_max_position_embeddings,
            config.text_hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = embedding(
            config.text_type_vocab_size,
            config.text_hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = layer_norm(
            config.text_hidden_size,
            config.text_layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let position_ids =
            Tensor::arange(0u32, seq_len as u32, input_ids.device())?.unsqueeze(0)?;

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let pos_emb = self.position_embeddings.forward(&position_ids)?;
        let tok_emb = self.token_type_embeddings.forward(token_type_ids)?;

        let emb = (word_emb + pos_emb)?.broadcast_add(&tok_emb)?;
        emb.apply(&self.layer_norm)
    }
}

#[derive(Clone)]
struct BertSelfAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl BertSelfAttention {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.text_hidden_size;
        let query = linear(h, h, vb.pp("query"))?;
        let key = linear(h, h, vb.pp("key"))?;
        let value = linear(h, h, vb.pp("value"))?;
        let num_heads = config.text_num_heads;
        let head_dim = h / num_heads;
        Ok(Self {
            query,
            key,
            value,
            num_heads,
            head_dim,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let (b, seq_len, _) = hidden_states.dims3()?;
        let q = hidden_states
            .apply(&self.query)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = hidden_states
            .apply(&self.key)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = hidden_states
            .apply(&self.value)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            * scale)?;
        let scores = scores.broadcast_add(attention_mask)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?;
        out.transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
    }
}

#[derive(Clone)]
struct BertSelfOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertSelfOutput {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.text_hidden_size;
        Ok(Self {
            dense: linear(h, h, vb.pp("dense"))?,
            layer_norm: layer_norm(h, config.text_layer_norm_eps, vb.pp("LayerNorm"))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input: &Tensor) -> candle_core::Result<Tensor> {
        let out = hidden_states.apply(&self.dense)?;
        (out + input)?.apply(&self.layer_norm)
    }
}

#[derive(Clone)]
struct BertIntermediate {
    dense: Linear,
}

impl BertIntermediate {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(
                config.text_hidden_size,
                config.text_intermediate_size,
                vb.pp("dense"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        hidden_states.apply(&self.dense)?.gelu_erf()
    }
}

#[derive(Clone)]
struct BertOutput {
    dense: Linear,
    layer_norm: LayerNorm,
}

impl BertOutput {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        Ok(Self {
            dense: linear(
                config.text_intermediate_size,
                config.text_hidden_size,
                vb.pp("dense"),
            )?,
            layer_norm: layer_norm(
                config.text_hidden_size,
                config.text_layer_norm_eps,
                vb.pp("LayerNorm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input: &Tensor) -> candle_core::Result<Tensor> {
        let out = hidden_states.apply(&self.dense)?;
        (out + input)?.apply(&self.layer_norm)
    }
}

#[derive(Clone)]
struct BertLayer {
    attention_self: BertSelfAttention,
    attention_output: BertSelfOutput,
    intermediate: BertIntermediate,
    output: BertOutput,
}

impl BertLayer {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let attn_vb = vb.pp("attention");
        Ok(Self {
            attention_self: BertSelfAttention::load(attn_vb.pp("self"), config)?,
            attention_output: BertSelfOutput::load(attn_vb.pp("output"), config)?,
            intermediate: BertIntermediate::load(vb.pp("intermediate"), config)?,
            output: BertOutput::load(vb.pp("output"), config)?,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let attn_out = self.attention_self.forward(hidden_states, attention_mask)?;
        let attn_out = self.attention_output.forward(&attn_out, hidden_states)?;
        let intermediate_out = self.intermediate.forward(&attn_out)?;
        self.output.forward(&intermediate_out, &attn_out)
    }
}

#[derive(Clone)]
struct MiniLMEncoder {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
}

impl MiniLMEncoder {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let embeddings = BertEmbeddings::load(vb.pp("embeddings"), config)?;
        let mut layers = Vec::with_capacity(config.text_num_layers);
        for i in 0..config.text_num_layers {
            layers.push(BertLayer::load(
                vb.pp(format!("encoder.layer.{}", i)),
                config,
            )?);
        }
        Ok(Self { embeddings, layers })
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_to_layer(input_ids, attention_mask, self.layers.len())
    }

    fn forward_to_layer(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        target_layer: usize,
    ) -> candle_core::Result<Tensor> {
        let token_type_ids = Tensor::zeros_like(input_ids)?;
        let mut xs = self.embeddings.forward(input_ids, &token_type_ids)?;

        // Build 4D causal mask from [batch, seq] → [batch, 1, 1, seq]
        let ext_mask = prepare_bert_attention_mask(attention_mask, xs.dtype())?;

        let num_layers = target_layer.min(self.layers.len());
        for layer in self.layers.iter().take(num_layers) {
            xs = layer.forward(&xs, &ext_mask)?;
        }
        Ok(xs)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

fn prepare_bert_attention_mask(mask: &Tensor, dtype: DType) -> candle_core::Result<Tensor> {
    let mask = mask.unsqueeze(1)?.unsqueeze(2)?;
    let mask = mask.to_dtype(dtype)?;
    let inverted = (1.0 - mask)?;
    inverted * f32::MIN as f64
}

// ============================================================================
// SigLIP Vision Encoder
// ============================================================================

/// Per-channel mean for SigLIP image normalization.
///
/// From SigLIP base / SigLIP2 `preprocessor_config.json`. The Go-side
/// `decodeAndResizeImage` in semantic-router.go produces CHW float32 pixels
/// in `[0, 1]`. SigLIP was trained on inputs in `[-1, 1]`, normalized via
/// `(x - mean) / std` per channel. We apply that normalization here, in the
/// Rust encoder, so the Go side stays unchanged and the activations land in
/// the distribution the model expects.
const IMAGE_MEAN: [f32; 3] = [0.5, 0.5, 0.5];

/// Per-channel std for SigLIP image normalization. See `IMAGE_MEAN`.
const IMAGE_STD: [f32; 3] = [0.5, 0.5, 0.5];

#[derive(Clone)]
struct SigLIPPatchEmbedding {
    projection: candle_nn::Conv2d,
    position_embedding: Embedding,
    num_patches: usize,
}

impl SigLIPPatchEmbedding {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let num_patches = config.num_image_patches();
        let projection = candle_nn::conv2d(
            3,
            config.image_hidden_size,
            config.image_patch_size,
            candle_nn::Conv2dConfig {
                stride: config.image_patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
        let position_embedding = embedding(
            num_patches,
            config.image_hidden_size,
            vb.pp("position_embedding"),
        )?;

        Ok(Self {
            projection,
            position_embedding,
            num_patches,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        let patch_emb = self.projection.forward(pixel_values)?;
        let (b, c, h, w) = patch_emb.dims4()?;
        let patch_emb = patch_emb.reshape((b, c, h * w))?.transpose(1, 2)?;
        let position_ids = Tensor::arange(0u32, self.num_patches as u32, patch_emb.device())?;
        let pos_emb = self.position_embedding.forward(&position_ids)?;
        patch_emb.broadcast_add(&pos_emb)
    }
}

#[derive(Clone)]
struct SigLIPSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl SigLIPSelfAttention {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.image_hidden_size;
        let num_heads = config.image_num_heads;
        let head_dim = h / num_heads;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("q_proj"))?,
            k_proj: linear(h, h, vb.pp("k_proj"))?,
            v_proj: linear(h, h, vb.pp("v_proj"))?,
            out_proj: linear(h, h, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let q = xs
            .apply(&self.q_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = xs
            .apply(&self.k_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = xs
            .apply(&self.v_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            * scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?;
        out.transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

#[derive(Clone)]
struct SigLIPMLP {
    fc1: Linear,
    fc2: Linear,
}

impl SigLIPMLP {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        Ok(Self {
            fc1: linear(
                config.image_hidden_size,
                config.image_intermediate_size,
                vb.pp("fc1"),
            )?,
            fc2: linear(
                config.image_intermediate_size,
                config.image_hidden_size,
                vb.pp("fc2"),
            )?,
        })
    }
}

impl Module for SigLIPMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        xs.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

#[derive(Clone)]
struct SigLIPEncoderLayer {
    self_attn: SigLIPSelfAttention,
    layer_norm1: LayerNorm,
    mlp: SigLIPMLP,
    layer_norm2: LayerNorm,
}

impl SigLIPEncoderLayer {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: SigLIPSelfAttention::load(vb.pp("self_attn"), config)?,
            layer_norm1: layer_norm(config.image_hidden_size, 1e-6, vb.pp("layer_norm1"))?,
            mlp: SigLIPMLP::load(vb.pp("mlp"), config)?,
            layer_norm2: layer_norm(config.image_hidden_size, 1e-6, vb.pp("layer_norm2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs.clone();
        let xs = xs.apply(&self.layer_norm1)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = xs.clone();
        let xs = xs.apply(&self.layer_norm2)?;
        let xs = xs.apply(&self.mlp)?;
        xs + residual
    }
}

#[derive(Clone)]
struct SigLIPVisionEncoder {
    patch_embedding: SigLIPPatchEmbedding,
    layers: Vec<SigLIPEncoderLayer>,
    post_layernorm: LayerNorm,
    head: SigLIPHead,
}

/// MultiheadAttention compatible with PyTorch's nn.MultiheadAttention combined-QKV
/// in_proj_weight format. Mirrors candle-transformers/src/models/siglip.rs:292.
/// Used by SigLIP-base / SigLIP2 attentional probe pooling.
#[derive(Clone, Debug)]
struct SigLIPHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl SigLIPHeadAttention {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.image_hidden_size;
        let num_heads = config.image_num_heads;
        let w_in_proj = vb.get((3 * h, h), "in_proj_weight")?.chunk(3, 0)?;
        let b_in_proj = vb.get(3 * h, "in_proj_bias")?.chunk(3, 0)?;
        let q_proj = Linear::new(w_in_proj[0].clone(), Some(b_in_proj[0].clone()));
        let k_proj = Linear::new(w_in_proj[1].clone(), Some(b_in_proj[1].clone()));
        let v_proj = Linear::new(w_in_proj[2].clone(), Some(b_in_proj[2].clone()));
        let out_proj = linear(h, h, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let (b, n, c) = x.dims3()?;
        x.reshape((b, n, self.num_heads, c / self.num_heads))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn recombine_heads(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let (b, n_heads, n_tokens, c_per_head) = x.dims4()?;
        x.transpose(1, 2)?
            .reshape((b, n_tokens, n_heads * c_per_head))
    }

    fn forward(
        &self,
        q: &candle_core::Tensor,
        k: &candle_core::Tensor,
        v: &candle_core::Tensor,
    ) -> candle_core::Result<candle_core::Tensor> {
        use candle_core::Module;
        let q = self.q_proj.forward(&q.contiguous()?)?;
        let k = self.k_proj.forward(&k.contiguous()?)?;
        let v = self.v_proj.forward(&v.contiguous()?)?;

        let q = self.separate_heads(&q)?;
        let k = self.separate_heads(&k)?;
        let v = self.separate_heads(&v)?;

        let (_, _, _, c_per_head) = q.dims4()?;
        let attn = (q.matmul(&k.t()?)? / (c_per_head as f64).sqrt())?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        let out = attn.matmul(&v)?;
        self.recombine_heads(&out)?.apply(&self.out_proj)
    }
}

/// MLP for the attentional pooling head. Matches candle-transformers/src/models/siglip.rs:444
/// in shape, uses gelu_erf for the activation consistent with the rest of this file's
/// SigLIP encoder MLPs.
#[derive(Clone, Debug)]
struct SigLIPHeadMlp {
    fc1: Linear,
    fc2: Linear,
}

impl SigLIPHeadMlp {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let fc1 = candle_nn::linear(
            config.image_hidden_size,
            config.image_intermediate_size,
            vb.pp("fc1"),
        )?;
        let fc2 = candle_nn::linear(
            config.image_intermediate_size,
            config.image_hidden_size,
            vb.pp("fc2"),
        )?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        xs.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)
    }
}

/// SigLIP attentional probe pooling head. Mirrors HF transformers'
/// SiglipMultiheadAttentionPoolingHead (modeling_siglip.py) and
/// candle-transformers/src/models/siglip.rs:351's MultiheadAttentionPoolingHead.
/// Replaces the incorrect BERT-style mean+linear+tanh that was previously here.
#[derive(Clone, Debug)]
struct SigLIPHead {
    probe: candle_core::Tensor,
    attention: SigLIPHeadAttention,
    layernorm: LayerNorm,
    mlp: SigLIPHeadMlp,
}

impl SigLIPHead {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let mlp = SigLIPHeadMlp::load(vb.pp("mlp"), config)?;
        let layernorm = layer_norm(config.image_hidden_size, 1e-6, vb.pp("layernorm"))?;
        let probe = vb.get((1, 1, config.image_hidden_size), "probe")?;
        let attention = SigLIPHeadAttention::load(vb.pp("attention"), config)?;
        Ok(Self {
            probe,
            attention,
            layernorm,
            mlp,
        })
    }

    fn forward(&self, xs: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        use candle_core::IndexOp;
        let batch_size = xs.dim(0)?;
        let probe = self.probe.repeat((batch_size, 1, 1))?;
        let xs = self.attention.forward(&probe, xs, xs)?;
        let residual = &xs;
        let xs = xs.apply(&self.layernorm)?;
        let xs = self.mlp.forward(&xs)?;
        (xs + residual)?.i((.., 0))
    }
}

impl SigLIPVisionEncoder {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let patch_embedding = SigLIPPatchEmbedding::load(vb.pp("embeddings"), config)?;
        let mut layers = Vec::with_capacity(config.image_num_layers);
        for i in 0..config.image_num_layers {
            layers.push(SigLIPEncoderLayer::load(
                vb.pp(format!("encoder.layers.{}", i)),
                config,
            )?);
        }
        let post_layernorm = layer_norm(config.image_hidden_size, 1e-6, vb.pp("post_layernorm"))?;
        let head = SigLIPHead::load(vb.pp("head"), config)?;
        Ok(Self {
            patch_embedding,
            layers,
            post_layernorm,
            head,
        })
    }

    /// Returns pooler_output: [batch, hidden_size]
    ///
    /// SigLIP-base uses `SiglipMultiheadAttentionPoolingHead` (learned probe +
    /// cross-attention + LayerNorm + MLP + residual). Replaces the prior
    /// BERT-style `mean + Linear + tanh` which produced garbage cosines.
    /// Mirrors candle-transformers/src/models/siglip.rs:373.
    ///
    /// The pooling head is required: genuine SigLIP checkpoints always ship
    /// the head weights, and `load` fails loudly when they are missing. A
    /// prior version kept `head: Option<SigLIPHead>` with a silent mean-pool
    /// fallback on the `None` branch; mean-pooled penultimate activations are
    /// not in the trained pooled-output space, so every downstream cosine
    /// threshold misreads them while nothing logs or errors. A missing head
    /// is a misload to surface at load time, not a degraded mode to serve.
    fn forward(&self, pixel_values: &Tensor) -> candle_core::Result<Tensor> {
        // SigLIP expects inputs in [-1, 1], normalized via (x - mean) / std
        // per channel using IMAGE_MEAN / IMAGE_STD from the model's
        // preprocessor_config.json. The Go-side image decoder produces pixels
        // in [0, 1]; apply the normalization here so activations sit in the
        // input distribution the encoder was trained on.
        let device = pixel_values.device();
        let dtype = pixel_values.dtype();
        // Shape [1, 3, 1, 1] broadcasts cleanly against [B, 3, H, W].
        let mean = Tensor::from_slice(&IMAGE_MEAN, (1, 3, 1, 1), device)?.to_dtype(dtype)?;
        let std = Tensor::from_slice(&IMAGE_STD, (1, 3, 1, 1), device)?.to_dtype(dtype)?;
        let normalized = pixel_values
            .broadcast_sub(&mean)?
            .broadcast_div(&std)?
            .contiguous()?;

        let mut xs = self.patch_embedding.forward(&normalized)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        xs = xs.apply(&self.post_layernorm)?;
        self.head.forward(&xs)
    }
}

// ============================================================================
// Whisper Audio Encoder
// ============================================================================

#[derive(Clone)]
struct WhisperConv {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
}

impl WhisperConv {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let conv1 = candle_nn::conv1d(
            config.audio_num_mel_bins,
            config.audio_hidden_size,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            config.audio_hidden_size,
            config.audio_hidden_size,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;
        Ok(Self { conv1, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv1.forward(xs)?.gelu_erf()?;
        self.conv2.forward(&xs)?.gelu_erf()
    }
}

#[derive(Clone)]
struct WhisperSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl WhisperSelfAttention {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.audio_hidden_size;
        let num_heads = config.audio_num_heads;
        let head_dim = h / num_heads;
        Ok(Self {
            q_proj: linear(h, h, vb.pp("self_attn.q_proj"))?,
            k_proj: linear(h, h, vb.pp("self_attn.k_proj"))?,
            v_proj: linear(h, h, vb.pp("self_attn.v_proj"))?,
            out_proj: linear(h, h, vb.pp("self_attn.out_proj"))?,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, n, _) = xs.dims3()?;
        let q = xs
            .apply(&self.q_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = xs
            .apply(&self.k_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = xs
            .apply(&self.v_proj)?
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).powf(-0.5);
        let attn = (q
            .contiguous()?
            .matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            * scale)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?;
        out.transpose(1, 2)?
            .reshape((b, n, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

#[derive(Clone)]
struct WhisperEncoderLayer {
    self_attn: WhisperSelfAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl WhisperEncoderLayer {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let h = config.audio_hidden_size;
        let intermediate = h * 4; // Whisper-tiny uses 4x expansion
        Ok(Self {
            self_attn: WhisperSelfAttention::load(vb.clone(), config)?,
            self_attn_layer_norm: layer_norm(h, 1e-5, vb.pp("self_attn_layer_norm"))?,
            fc1: linear(h, intermediate, vb.pp("fc1"))?,
            fc2: linear(intermediate, h, vb.pp("fc2"))?,
            final_layer_norm: layer_norm(h, 1e-5, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs.clone();
        let xs = xs.apply(&self.self_attn_layer_norm)?;
        let xs = self.self_attn.forward(&xs)?;
        let xs = (xs + residual)?;

        let residual = xs.clone();
        let xs = xs.apply(&self.final_layer_norm)?;
        let xs = xs.apply(&self.fc1)?.gelu_erf()?.apply(&self.fc2)?;
        xs + residual
    }
}

#[derive(Clone)]
struct WhisperEncoder {
    conv: WhisperConv,
    embed_positions: Embedding,
    layers: Vec<WhisperEncoderLayer>,
    layer_norm: LayerNorm,
}

impl WhisperEncoder {
    fn load(vb: VarBuilder, config: &MultiModalEmbeddingConfig) -> candle_core::Result<Self> {
        let conv = WhisperConv::load(vb.clone(), config)?;
        let embed_positions = embedding(
            config.audio_max_source_positions,
            config.audio_hidden_size,
            vb.pp("embed_positions"),
        )?;
        let mut layers = Vec::with_capacity(config.audio_num_layers);
        for i in 0..config.audio_num_layers {
            layers.push(WhisperEncoderLayer::load(
                vb.pp(format!("layers.{}", i)),
                config,
            )?);
        }
        let layer_norm = layer_norm(config.audio_hidden_size, 1e-5, vb.pp("layer_norm"))?;
        Ok(Self {
            conv,
            embed_positions,
            layers,
            layer_norm,
        })
    }

    /// Input: mel spectrogram [batch, n_mels, time_frames]
    /// Output: [batch, seq_len, hidden_size]
    fn forward(&self, mel: &Tensor) -> candle_core::Result<Tensor> {
        let xs = self.conv.forward(mel)?;
        // conv output: [batch, hidden, seq_len] → [batch, seq_len, hidden]
        let xs = xs.transpose(1, 2)?;
        let seq_len = xs.dim(1)?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_emb = self.embed_positions.forward(&position_ids)?;
        let mut xs = xs.broadcast_add(&pos_emb)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        xs.apply(&self.layer_norm)
    }
}

// ============================================================================
// Complete Multi-Modal Embedding Model
// ============================================================================

/// Multi-modal embedding model combining text, image, and audio encoders.
pub struct MultiModalEmbeddingModel {
    text_encoder: MiniLMEncoder,
    image_encoder: SigLIPVisionEncoder,
    image_projection: Linear,
    audio_encoder: Option<WhisperEncoder>,
    config: MultiModalEmbeddingConfig,
    matryoshka_config: MultiModalMatryoshkaConfig,
    device: Device,
}

impl std::fmt::Debug for MultiModalEmbeddingModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiModalEmbeddingModel")
            .field("config", &self.config)
            .field("matryoshka_config", &self.matryoshka_config)
            .field("device", &self.device)
            .finish()
    }
}

impl MultiModalEmbeddingModel {
    /// Load model from a pretrained directory that contains `model.safetensors`
    /// and optionally `config.json`.
    pub fn load(model_path: &str, device: &Device) -> UnifiedResult<Self> {
        let config = MultiModalEmbeddingConfig::from_pretrained(model_path)?;

        let safetensors_path = format!("{}/model.safetensors", model_path);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&safetensors_path),
                DType::F32,
                device,
            )
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

    /// Load model with an existing VarBuilder (enables custom weight prefix mapping).
    pub fn load_with_vb(
        _model_path: &str,
        config: &MultiModalEmbeddingConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> UnifiedResult<Self> {
        // Text encoder: weights under text_encoder.encoder.*
        let text_encoder = MiniLMEncoder::load(vb.pp("text_encoder.encoder"), config)
            .map_err(|e| from_candle_error(e, "failed to load MiniLM text encoder", None))?;

        // Image encoder: weights under image_encoder.vision_encoder.*
        let image_encoder =
            SigLIPVisionEncoder::load(vb.pp("image_encoder.vision_encoder"), config)
                .map_err(|e| from_candle_error(e, "failed to load SigLIP image encoder", None))?;

        // Image projection: 768 → 384
        let image_projection = linear(
            config.image_hidden_size,
            config.embedding_dim,
            vb.pp("image_encoder.projection"),
        )
        .map_err(|e| from_candle_error(e, "failed to load image projection", None))?;

        // Audio encoder: weights under audio_encoder.encoder.* (optional)
        let audio_encoder = WhisperEncoder::load(vb.pp("audio_encoder.encoder"), config).ok();

        Ok(Self {
            text_encoder,
            image_encoder,
            image_projection,
            audio_encoder,
            config: config.clone(),
            matryoshka_config: MultiModalMatryoshkaConfig::default(),
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &MultiModalEmbeddingConfig {
        &self.config
    }

    pub fn matryoshka_config(&self) -> &MultiModalMatryoshkaConfig {
        &self.matryoshka_config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn num_text_layers(&self) -> usize {
        self.text_encoder.num_layers()
    }

    // ─── Text encoding ───

    /// Encode text input_ids and attention_mask → [batch, embedding_dim] normalized embedding.
    pub fn encode_text(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        self.encode_text_with_matryoshka(input_ids, attention_mask, None, None)
    }

    /// Encode text with 2DMSE support (layer early exit + dimension truncation).
    pub fn encode_text_with_matryoshka(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        target_layer: Option<usize>,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Tensor> {
        let (batch_size, seq_len) = input_ids
            .dims2()
            .map_err(|e| from_candle_error(e, "get text input dims", None))?;

        let num_layers = self.text_encoder.num_layers();
        let target_layer = target_layer.unwrap_or(num_layers);
        if target_layer > num_layers || target_layer == 0 {
            return Err(UnifiedError::Validation {
                field: "target_layer".to_string(),
                expected: format!("1 to {}", num_layers),
                actual: target_layer.to_string(),
                context: Some("Text encoder layer for 2DMSE".to_string()),
            });
        }

        let target_dim = target_dim.unwrap_or(self.config.embedding_dim);
        if target_dim > self.config.embedding_dim {
            return Err(UnifiedError::Validation {
                field: "target_dim".to_string(),
                expected: format!("<= {}", self.config.embedding_dim),
                actual: target_dim.to_string(),
                context: None,
            });
        }

        let default_mask;
        let mask = match attention_mask {
            Some(m) => m.clone(),
            None => {
                default_mask = Tensor::ones((batch_size, seq_len), DType::U32, &self.device)
                    .map_err(|e| from_candle_error(e, "create default mask", None))?;
                default_mask
            }
        };

        let hidden_states = self
            .text_encoder
            .forward_to_layer(input_ids, &mask, target_layer)
            .map_err(|e| from_candle_error(e, "text encoder forward", None))?;

        let mask_f32 = mask
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "mask to f32", None))?;
        let embeddings =
            mean_pool(&hidden_states, &mask_f32).map_err(|e| UnifiedError::Processing {
                operation: "mean_pool text".to_string(),
                source: e.to_string(),
                input_context: None,
            })?;

        let embeddings = if target_dim < self.config.embedding_dim {
            embeddings
                .narrow(1, 0, target_dim)
                .map_err(|e| from_candle_error(e, "dimension truncation", None))?
        } else {
            embeddings
        };

        self.l2_normalize(&embeddings)
    }

    // ─── Image encoding ───

    /// Encode image pixel_values [batch, 3, H, W] → [batch, embedding_dim] normalized embedding.
    pub fn encode_image(&self, pixel_values: &Tensor) -> UnifiedResult<Tensor> {
        self.encode_image_with_dim(pixel_values, None)
    }

    /// Encode image with optional MRL dimension truncation.
    pub fn encode_image_with_dim(
        &self,
        pixel_values: &Tensor,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Tensor> {
        let target_dim = target_dim.unwrap_or(self.config.embedding_dim);

        let pooler_output = self
            .image_encoder
            .forward(pixel_values)
            .map_err(|e| from_candle_error(e, "SigLIP forward", None))?;

        // Project 768 → 384
        let projected = pooler_output
            .apply(&self.image_projection)
            .map_err(|e| from_candle_error(e, "image projection", None))?;

        let projected = if target_dim < self.config.embedding_dim {
            projected
                .narrow(1, 0, target_dim)
                .map_err(|e| from_candle_error(e, "image dim truncation", None))?
        } else {
            projected
        };

        self.l2_normalize(&projected)
    }

    // ─── Audio encoding ───

    /// Encode audio mel spectrogram [batch, n_mels, time] → [batch, embedding_dim] normalized embedding.
    pub fn encode_audio(&self, mel_spectrogram: &Tensor) -> UnifiedResult<Tensor> {
        self.encode_audio_with_dim(mel_spectrogram, None)
    }

    /// Encode audio with optional MRL dimension truncation.
    pub fn encode_audio_with_dim(
        &self,
        mel_spectrogram: &Tensor,
        target_dim: Option<usize>,
    ) -> UnifiedResult<Tensor> {
        let target_dim = target_dim.unwrap_or(self.config.embedding_dim);

        let encoder = self.audio_encoder.as_ref().ok_or_else(|| {
            from_candle_error(
                candle_core::Error::Msg("Audio encoder not loaded (weights not available)".into()),
                "audio encoding requires Whisper weights",
                None,
            )
        })?;
        let hidden_states = encoder
            .forward(mel_spectrogram)
            .map_err(|e| from_candle_error(e, "Whisper encoder forward", None))?;

        // Mean pooling over time
        let embeddings = hidden_states
            .mean(1)
            .map_err(|e| from_candle_error(e, "audio mean pool", None))?;

        let embeddings = if target_dim < self.config.embedding_dim {
            embeddings
                .narrow(1, 0, target_dim)
                .map_err(|e| from_candle_error(e, "audio dim truncation", None))?
        } else {
            embeddings
        };

        self.l2_normalize(&embeddings)
    }

    // ─── Batch helpers ───

    /// Batch encode texts using the tokenizer.
    pub fn encode_text_batch(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        texts: &[&str],
        max_length: usize,
    ) -> UnifiedResult<Tensor> {
        self.encode_text_batch_with_matryoshka(tokenizer, texts, max_length, None, None)
    }

    /// Batch encode texts with 2DMSE.
    pub fn encode_text_batch_with_matryoshka(
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
                    attention_mask_vec.push(mask[i]);
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

        self.encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask),
            target_layer,
            target_dim,
        )
    }

    // ─── Utilities ───

    fn l2_normalize(&self, embeddings: &Tensor) -> UnifiedResult<Tensor> {
        let squared = embeddings
            .sqr()
            .map_err(|e| from_candle_error(e, "L2 sqr", None))?;
        let sum_squared = squared
            .sum_keepdim(D::Minus1)
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

impl CoreModel for MultiModalEmbeddingModel {
    type Config = MultiModalEmbeddingConfig;
    type Error = UnifiedError;
    type Output = Tensor;

    fn model_type(&self) -> ModelType {
        ModelType::MultiModalEmbedding
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error> {
        self.encode_text(input_ids, Some(attention_mask))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }
}

impl LongContextEmbeddingCapable for MultiModalEmbeddingModel {
    fn get_max_sequence_length(&self) -> usize {
        self.config.text_max_position_embeddings
    }

    fn get_embedding_dimension(&self) -> usize {
        self.config.embedding_dim
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
            if dim > self.config.embedding_dim {
                return Err(UnifiedError::Validation {
                    field: "target_dim".to_string(),
                    expected: format!("<= {}", self.config.embedding_dim),
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
        64
    }

    fn supports_parallel_batching(&self) -> bool {
        true
    }
}

impl EmbeddingPathSpecialization for MultiModalEmbeddingModel {
    fn supports_parallel(&self) -> bool {
        true
    }

    fn optimal_batch_size(&self) -> usize {
        64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_image_config() -> MultiModalEmbeddingConfig {
        MultiModalEmbeddingConfig {
            image_hidden_size: 8,
            image_patch_size: 4,
            image_size: 8, // 4 patches
            image_num_layers: 0,
            image_num_heads: 2,
            image_intermediate_size: 16,
            ..Default::default()
        }
    }

    /// Tensor map covering exactly what SigLIPVisionEncoder::load reads for a
    /// zero-layer encoder, optionally including the pooling-head weights.
    fn encoder_tensors(
        config: &MultiModalEmbeddingConfig,
        with_head: bool,
    ) -> std::collections::HashMap<String, Tensor> {
        let h = config.image_hidden_size;
        let p = config.image_patch_size;
        let i = config.image_intermediate_size;
        let n = config.num_image_patches();
        let dev = Device::Cpu;
        let mut m = std::collections::HashMap::new();
        let mut put = |name: &str, dims: Vec<usize>| {
            m.insert(
                name.to_string(),
                Tensor::zeros(dims, candle_core::DType::F32, &dev).unwrap(),
            );
        };
        put("embeddings.patch_embedding.weight", vec![h, 3, p, p]);
        put("embeddings.patch_embedding.bias", vec![h]);
        put("embeddings.position_embedding.weight", vec![n, h]);
        put("post_layernorm.weight", vec![h]);
        put("post_layernorm.bias", vec![h]);
        if with_head {
            put("head.probe", vec![1, 1, h]);
            put("head.attention.in_proj_weight", vec![3 * h, h]);
            put("head.attention.in_proj_bias", vec![3 * h]);
            put("head.attention.out_proj.weight", vec![h, h]);
            put("head.attention.out_proj.bias", vec![h]);
            put("head.layernorm.weight", vec![h]);
            put("head.layernorm.bias", vec![h]);
            put("head.mlp.fc1.weight", vec![i, h]);
            put("head.mlp.fc1.bias", vec![i]);
            put("head.mlp.fc2.weight", vec![h, i]);
            put("head.mlp.fc2.bias", vec![h]);
        }
        m
    }

    #[test]
    fn test_siglip_vision_encoder_loads_with_head_weights() {
        let config = tiny_image_config();
        let vb = VarBuilder::from_tensors(
            encoder_tensors(&config, true),
            candle_core::DType::F32,
            &Device::Cpu,
        );
        SigLIPVisionEncoder::load(vb, &config).expect("encoder with full head weights should load");
    }

    /// A checkpoint without the pooling-head weights must fail to load. The
    /// prior Option-based load swallowed the missing head and silently
    /// mean-pooled at forward time, producing embeddings outside the trained
    /// pooled-output space with nothing logged.
    #[test]
    fn test_siglip_vision_encoder_requires_pooling_head() {
        let config = tiny_image_config();
        let vb = VarBuilder::from_tensors(
            encoder_tensors(&config, false),
            candle_core::DType::F32,
            &Device::Cpu,
        );
        assert!(
            SigLIPVisionEncoder::load(vb, &config).is_err(),
            "encoder load must fail loudly when head weights are missing"
        );
    }

    #[test]
    fn test_default_config() {
        let config = MultiModalEmbeddingConfig::default();
        assert_eq!(config.embedding_dim, 384);
        assert_eq!(config.text_hidden_size, 384);
        assert_eq!(config.text_num_layers, 6);
        assert_eq!(config.image_hidden_size, 768);
        assert_eq!(config.audio_hidden_size, 384);
        assert_eq!(config.num_image_patches(), 1024); // (512/16)^2
    }

    #[test]
    fn test_matryoshka_config() {
        let config = MultiModalMatryoshkaConfig::default();
        assert!(config.validate_dimension(384));
        assert!(config.validate_dimension(128));
        assert!(config.validate_dimension(32));
        assert!(!config.validate_dimension(100));
        assert!(config.validate_layer(6));
        assert!(config.validate_layer(1));
        assert!(!config.validate_layer(7));
    }

    #[test]
    fn test_bert_attention_mask_shape() {
        let device = Device::Cpu;
        let mask = Tensor::ones((2, 10), DType::U32, &device).unwrap();
        let ext = prepare_bert_attention_mask(&mask, DType::F32).unwrap();
        assert_eq!(ext.dims(), &[2, 1, 1, 10]);
    }

    #[test]
    fn test_siglip_image_normalization_constants() {
        // SigLIP image normalization maps [0, 1] -> [-1, 1] via
        // (x - 0.5) / 0.5. Validate the arithmetic on a known input that
        // spans the full [0, 1] range INCLUDING the 0.5 midpoint, using
        // the IMAGE_MEAN / IMAGE_STD constants and the same broadcast
        // pattern as SigLIPVisionEncoder::forward.
        let device = Device::Cpu;
        // [1, 3, 1, 3] tensor with values 0.0, 0.5, 1.0 per channel to
        // exercise both endpoints and the midpoint.
        let pixels = Tensor::from_slice(
            &[
                0.0f32, 0.5, 1.0, // channel 0
                0.0, 0.5, 1.0, // channel 1
                0.0, 0.5, 1.0, // channel 2
            ],
            (1, 3, 1, 3),
            &device,
        )
        .unwrap();
        let mean = Tensor::from_slice(&IMAGE_MEAN, (1, 3, 1, 1), &device).unwrap();
        let std = Tensor::from_slice(&IMAGE_STD, (1, 3, 1, 1), &device).unwrap();
        let normalized = pixels
            .broadcast_sub(&mean)
            .unwrap()
            .broadcast_div(&std)
            .unwrap();
        let out: Vec<f32> = normalized.flatten_all().unwrap().to_vec1().unwrap();
        // Expected per-element: 0.0 -> -1.0, 0.5 -> 0.0, 1.0 -> +1.0,
        // all within tol.
        let expected = [-1.0f32, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
        for (got, want) in out.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "normalized pixel {} differs from expected {}",
                got,
                want
            );
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::io::Read as IoRead;
    use tokenizers::Tokenizer;

    fn get_model_path() -> Option<String> {
        std::env::var("MULTIMODAL_MODEL_PATH").ok()
    }

    fn load_model_and_tokenizer() -> (MultiModalEmbeddingModel, Tokenizer) {
        let model_path = get_model_path().expect("MULTIMODAL_MODEL_PATH not set");
        let device = Device::Cpu;
        let model =
            MultiModalEmbeddingModel::load(&model_path, &device).expect("Failed to load model");
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path).expect("Failed to load tokenizer");
        (model, tokenizer)
    }

    fn encode_text_str(
        model: &MultiModalEmbeddingModel,
        tokenizer: &Tokenizer,
        text: &str,
    ) -> Tensor {
        let encoding = tokenizer.encode(text, true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let device = model.device();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();
        model
            .encode_text(&input_ids, Some(&attention_mask))
            .expect("encode_text failed")
    }

    fn to_vec(t: &Tensor) -> Vec<f32> {
        t.squeeze(0).unwrap().to_vec1::<f32>().unwrap()
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            0.0
        }
    }

    fn assert_unit_norm(t: &Tensor, label: &str) {
        let v = to_vec(t);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "{}: expected unit norm, got {}",
            label,
            norm
        );
    }

    // ----------------------------------------------------------------
    // Model loading
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_load_model() {
        let model_path = get_model_path().expect("MULTIMODAL_MODEL_PATH not set");
        let device = Device::Cpu;
        let model = MultiModalEmbeddingModel::load(&model_path, &device).expect("Failed to load");
        assert_eq!(model.config().embedding_dim, 384);
        assert_eq!(model.num_text_layers(), 6);
    }

    // ----------------------------------------------------------------
    // Text-only tests
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_text_embedding() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let emb = encode_text_str(&model, &tokenizer, "A photo of a cat");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "text embedding");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_semantic_similarity() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let cat = to_vec(&encode_text_str(&model, &tokenizer, "A fluffy orange cat"));
        let dog = to_vec(&encode_text_str(
            &model,
            &tokenizer,
            "A golden retriever dog",
        ));
        let car = to_vec(&encode_text_str(&model, &tokenizer, "A red sports car"));

        let cat_dog = cosine_sim(&cat, &dog);
        let cat_car = cosine_sim(&cat, &car);

        println!("cat-dog: {:.4}, cat-car: {:.4}", cat_dog, cat_car);
        assert!(
            cat_dog > cat_car,
            "cat-dog ({:.4}) should be more similar than cat-car ({:.4})",
            cat_dog,
            cat_car,
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_matryoshka_dims() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let encoding = tokenizer.encode("Hello world", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let device = model.device();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();

        for dim in [32, 64, 128, 256, 384] {
            let emb = model
                .encode_text_with_matryoshka(&input_ids, Some(&attention_mask), None, Some(dim))
                .unwrap_or_else(|e| panic!("Failed at dim {}: {:?}", dim, e));
            assert_eq!(emb.dims(), &[1, dim], "dim mismatch for target {}", dim);
            assert_unit_norm(&emb, &format!("text MRL dim={}", dim));
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_2dmse_layer_early_exit() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let encoding = tokenizer.encode("Test layer exit", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let device = model.device();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();

        for layer in 1..=6 {
            let emb = model
                .encode_text_with_matryoshka(&input_ids, Some(&attention_mask), Some(layer), None)
                .unwrap_or_else(|e| panic!("Failed at layer {}: {:?}", layer, e));
            assert_eq!(emb.dims(), &[1, 384]);
            assert_unit_norm(&emb, &format!("text layer={}", layer));
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_batch_encoding() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let texts = vec![
            "A fluffy cat",
            "A large dog",
            "A fast car",
            "A beautiful sunset",
        ];
        let batch_emb = model
            .encode_text_batch(&tokenizer, &texts, 512)
            .expect("batch encoding failed");
        assert_eq!(batch_emb.dims(), &[4, 384]);

        for i in 0..4 {
            let row = batch_emb.get(i).unwrap();
            let norm: f32 = row
                .sqr()
                .unwrap()
                .sum(0)
                .unwrap()
                .sqrt()
                .unwrap()
                .to_vec0()
                .unwrap();
            assert!(
                (norm - 1.0).abs() < 0.01,
                "batch row {} norm={:.4}",
                i,
                norm
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_consistency() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let emb1 = to_vec(&encode_text_str(&model, &tokenizer, "Hello world"));
        let emb2 = to_vec(&encode_text_str(&model, &tokenizer, "Hello world"));

        let sim = cosine_sim(&emb1, &emb2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "same text should produce identical embeddings, got sim={:.6}",
            sim
        );
    }

    // ----------------------------------------------------------------
    // Image helpers (synthetic + real)
    // ----------------------------------------------------------------

    /// Create a synthetic 512x512 pixel tensor for testing.
    fn make_test_image(device: &Device, r: f32, g: f32, b: f32) -> Tensor {
        let h = 512usize;
        let w = 512usize;
        let mut data = Vec::with_capacity(3 * h * w);
        for _ in 0..h * w {
            data.push(r);
        }
        for _ in 0..h * w {
            data.push(g);
        }
        for _ in 0..h * w {
            data.push(b);
        }
        Tensor::from_vec(data, (1, 3, h, w), device).unwrap()
    }

    /// Copyright-free Wikimedia Commons images:
    ///   - Tuxedo_kitten.jpg    : Public Domain (author: TimVickers)
    ///   - 1Cute-doggy.jpg      : CC0 1.0 Universal (author: X posid)
    ///   - 1908_Ford_Model_T.jpg: Public Domain (published 1908, pre-1930)
    const WIKI_CAT_URL: &str =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Tuxedo_kitten.jpg/512px-Tuxedo_kitten.jpg";
    const WIKI_DOG_URL: &str =
        "https://upload.wikimedia.org/wikipedia/commons/a/a7/1Cute-doggy.jpg";
    const WIKI_CAR_URL: &str =
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/1908_Ford_Model_T.jpg/960px-1908_Ford_Model_T.jpg";

    fn download_bytes(url: &str) -> Vec<u8> {
        let resp = ureq::get(url).call().expect("HTTP download failed");
        let mut buf = Vec::new();
        resp.into_reader()
            .read_to_end(&mut buf)
            .expect("read body failed");
        buf
    }

    /// Decode JPEG/PNG bytes → resize to 512×512 → [1, 3, 512, 512] tensor in [0, 1].
    fn image_bytes_to_tensor(bytes: &[u8], device: &Device) -> Tensor {
        use image::GenericImageView;
        let img = image::load_from_memory(bytes).expect("image decode failed");
        let img = img.resize_exact(512, 512, image::imageops::FilterType::Triangle);
        let (w, h) = img.dimensions();
        assert_eq!((w, h), (512, 512));
        let rgb = img.to_rgb8();
        let raw = rgb.as_raw();
        let mut chw = vec![0f32; 3 * 512 * 512];
        for y in 0..512usize {
            for x in 0..512usize {
                let idx = (y * 512 + x) * 3;
                chw[y * 512 + x] = raw[idx] as f32 / 255.0;
                chw[512 * 512 + y * 512 + x] = raw[idx + 1] as f32 / 255.0;
                chw[2 * 512 * 512 + y * 512 + x] = raw[idx + 2] as f32 / 255.0;
            }
        }
        Tensor::from_vec(chw, (1, 3, 512, 512), device).unwrap()
    }

    /// Simulate OpenAI API flow: raw bytes → base64 encode → base64 decode → tensor.
    fn base64_roundtrip_to_tensor(image_bytes: &[u8], device: &Device) -> Tensor {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(image_bytes);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .expect("base64 decode failed");
        image_bytes_to_tensor(&decoded, device)
    }

    // ----------------------------------------------------------------
    // Synthetic image tests
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_image_embedding_shape_and_norm() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let img = make_test_image(model.device(), 0.5, 0.3, 0.7);

        let emb = model.encode_image(&img).expect("encode_image failed");
        assert_eq!(emb.dims(), &[1, 384], "image embedding should be [1, 384]");
        assert_unit_norm(&emb, "image embedding");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_image_matryoshka_dims() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let img = make_test_image(model.device(), 0.4, 0.4, 0.4);

        for dim in [32, 64, 128, 256, 384] {
            let emb = model
                .encode_image_with_dim(&img, Some(dim))
                .unwrap_or_else(|e| panic!("image MRL dim={} failed: {:?}", dim, e));
            assert_eq!(emb.dims(), &[1, dim]);
            assert_unit_norm(&emb, &format!("image MRL dim={}", dim));
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_image_different_inputs_differ() {
        let (model, _tokenizer) = load_model_and_tokenizer();

        let red_img = make_test_image(model.device(), 1.0, 0.0, 0.0);
        let blue_img = make_test_image(model.device(), 0.0, 0.0, 1.0);

        let red_emb = to_vec(&model.encode_image(&red_img).unwrap());
        let blue_emb = to_vec(&model.encode_image(&blue_img).unwrap());

        let sim = cosine_sim(&red_emb, &blue_emb);
        assert!(
            sim < 0.999,
            "different images should produce different embeddings, got sim={:.6}",
            sim
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_image_consistency() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let img = make_test_image(model.device(), 0.5, 0.5, 0.5);

        let emb1 = to_vec(&model.encode_image(&img).unwrap());
        let emb2 = to_vec(&model.encode_image(&img).unwrap());

        let sim = cosine_sim(&emb1, &emb2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "same image should produce identical embeddings, got sim={:.6}",
            sim
        );
    }

    // ----------------------------------------------------------------
    // Real image tests (Wikimedia Commons, copyright-free)
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files and network"]
    fn test_real_image_cat_embedding() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAT_URL);
        let tensor = image_bytes_to_tensor(&bytes, model.device());

        let emb = model
            .encode_image(&tensor)
            .expect("encode cat image failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "real cat image");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_real_image_dog_embedding() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_DOG_URL);
        let tensor = image_bytes_to_tensor(&bytes, model.device());

        let emb = model
            .encode_image(&tensor)
            .expect("encode dog image failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "real dog image");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_real_image_car_embedding() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAR_URL);
        let tensor = image_bytes_to_tensor(&bytes, model.device());

        let emb = model
            .encode_image(&tensor)
            .expect("encode car image failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "real car image");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_real_images_produce_distinct_embeddings() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let cat_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_CAT_URL),
                    device,
                ))
                .unwrap(),
        );
        let dog_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_DOG_URL),
                    device,
                ))
                .unwrap(),
        );
        let car_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_CAR_URL),
                    device,
                ))
                .unwrap(),
        );

        let cat_dog = cosine_sim(&cat_emb, &dog_emb);
        let cat_car = cosine_sim(&cat_emb, &car_emb);
        let dog_car = cosine_sim(&dog_emb, &car_emb);

        println!(
            "real images: cat-dog={:.4}, cat-car={:.4}, dog-car={:.4}",
            cat_dog, cat_car, dog_car
        );

        assert!(cat_dog < 0.999, "cat and dog should differ");
        assert!(cat_car < 0.999, "cat and car should differ");
        assert!(dog_car < 0.999, "dog and car should differ");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_real_image_matryoshka_dims() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAT_URL);
        let tensor = image_bytes_to_tensor(&bytes, model.device());

        for dim in [32, 64, 128, 256, 384] {
            let emb = model
                .encode_image_with_dim(&tensor, Some(dim))
                .unwrap_or_else(|e| panic!("real image MRL dim={} failed: {:?}", dim, e));
            assert_eq!(emb.dims(), &[1, dim]);
            assert_unit_norm(&emb, &format!("real image MRL dim={}", dim));
        }
    }

    // ----------------------------------------------------------------
    // Base64-encoded image tests (OpenAI API style)
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files and network"]
    fn test_base64_image_embedding() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAT_URL);
        let tensor = base64_roundtrip_to_tensor(&bytes, model.device());

        let emb = model
            .encode_image(&tensor)
            .expect("base64 image encode failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "base64 cat image");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_base64_vs_raw_identical_embeddings() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAT_URL);
        let device = model.device();

        let raw_tensor = image_bytes_to_tensor(&bytes, device);
        let b64_tensor = base64_roundtrip_to_tensor(&bytes, device);

        let raw_emb = to_vec(&model.encode_image(&raw_tensor).unwrap());
        let b64_emb = to_vec(&model.encode_image(&b64_tensor).unwrap());

        let sim = cosine_sim(&raw_emb, &b64_emb);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "base64 roundtrip must produce identical embedding, got sim={:.6}",
            sim
        );
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_base64_all_images_consistency() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        for (name, url) in [
            ("cat", WIKI_CAT_URL),
            ("dog", WIKI_DOG_URL),
            ("car", WIKI_CAR_URL),
        ] {
            let bytes = download_bytes(url);
            let raw_emb = to_vec(
                &model
                    .encode_image(&image_bytes_to_tensor(&bytes, device))
                    .unwrap(),
            );
            let b64_emb = to_vec(
                &model
                    .encode_image(&base64_roundtrip_to_tensor(&bytes, device))
                    .unwrap(),
            );
            let sim = cosine_sim(&raw_emb, &b64_emb);
            println!("{} raw vs base64: sim={:.6}", name, sim);
            assert!(
                (sim - 1.0).abs() < 1e-5,
                "{}: base64 roundtrip must match raw, got sim={:.6}",
                name,
                sim
            );
        }
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_base64_with_data_uri_prefix() {
        use base64::Engine;
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_CAT_URL);
        let device = model.device();

        let b64_str = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let data_uri = format!("data:image/jpeg;base64,{}", b64_str);

        let payload = data_uri
            .strip_prefix("data:image/jpeg;base64,")
            .expect("strip prefix");
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(payload)
            .expect("decode");
        let tensor = image_bytes_to_tensor(&decoded, device);

        let emb = model
            .encode_image(&tensor)
            .expect("data-uri base64 encode failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "data-uri base64 cat");

        let raw_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(&bytes, device))
                .unwrap(),
        );
        let uri_emb = to_vec(&emb);
        let sim = cosine_sim(&raw_emb, &uri_emb);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "data-uri flow should match raw, sim={:.6}",
            sim
        );
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_base64_matryoshka_dims() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let bytes = download_bytes(WIKI_DOG_URL);
        let tensor = base64_roundtrip_to_tensor(&bytes, model.device());

        for dim in [32, 64, 128, 256, 384] {
            let emb = model
                .encode_image_with_dim(&tensor, Some(dim))
                .unwrap_or_else(|e| panic!("base64 MRL dim={} failed: {:?}", dim, e));
            assert_eq!(emb.dims(), &[1, dim]);
            assert_unit_norm(&emb, &format!("base64 dog MRL dim={}", dim));
        }
    }

    // ----------------------------------------------------------------
    // Cross-modal tests (text + real image)
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_cross_modal_text_image_same_space() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let text_emb = to_vec(&encode_text_str(&model, &tokenizer, "A solid red image"));
        let img_emb = to_vec(
            &model
                .encode_image(&make_test_image(model.device(), 1.0, 0.0, 0.0))
                .unwrap(),
        );

        assert_eq!(
            text_emb.len(),
            img_emb.len(),
            "text and image should produce same-dimensional embeddings"
        );

        let sim = cosine_sim(&text_emb, &img_emb);
        assert!(
            sim.is_finite() && (-1.0..=1.0).contains(&sim),
            "similarity should be finite in [-1,1], got {}",
            sim
        );
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_cross_modal_real_image_text_retrieval() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let cat_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_CAT_URL),
                    device,
                ))
                .unwrap(),
        );
        let dog_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_DOG_URL),
                    device,
                ))
                .unwrap(),
        );
        let car_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_CAR_URL),
                    device,
                ))
                .unwrap(),
        );

        let queries = ["a photo of a cat", "a photo of a dog", "a photo of a car"];
        let image_embs = [("cat", &cat_emb), ("dog", &dog_emb), ("car", &car_emb)];

        for query in &queries {
            let text_emb = to_vec(&encode_text_str(&model, &tokenizer, query));
            for (name, img_emb) in &image_embs {
                let sim = cosine_sim(&text_emb, img_emb);
                println!("  \"{}\" vs {}: {:.4}", query, name, sim);
                assert!(sim.is_finite(), "cross-modal sim must be finite");
            }
        }
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_cross_modal_base64_image_text() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let cat_bytes = download_bytes(WIKI_CAT_URL);
        let cat_b64_tensor = base64_roundtrip_to_tensor(&cat_bytes, device);
        let cat_emb = to_vec(&model.encode_image(&cat_b64_tensor).unwrap());

        let text_emb = to_vec(&encode_text_str(&model, &tokenizer, "a photo of a kitten"));

        assert_eq!(text_emb.len(), cat_emb.len());
        let sim = cosine_sim(&text_emb, &cat_emb);
        println!("text('kitten') vs base64(cat): sim={:.4}", sim);
        assert!(sim.is_finite());
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_cross_modal_text_image_mrl_dimensions_match() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let cat_bytes = download_bytes(WIKI_CAT_URL);
        let img_tensor = image_bytes_to_tensor(&cat_bytes, device);

        let encoding = tokenizer.encode("a cute kitten", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();

        for dim in [64, 128, 256, 384] {
            let text_emb = model
                .encode_text_with_matryoshka(&input_ids, Some(&attention_mask), None, Some(dim))
                .unwrap();
            let img_emb = model.encode_image_with_dim(&img_tensor, Some(dim)).unwrap();

            assert_eq!(text_emb.dims()[1], dim);
            assert_eq!(img_emb.dims()[1], dim);

            let sim = cosine_sim(&to_vec(&text_emb), &to_vec(&img_emb));
            assert!(
                sim.is_finite(),
                "cross-modal sim at dim={} should be finite",
                dim
            );
            println!("  cross-modal dim={}: sim={:.4}", dim, sim);
        }
    }

    // ----------------------------------------------------------------
    // Audio tests
    // ----------------------------------------------------------------

    /// Create a synthetic mel spectrogram for testing.
    fn make_test_mel(device: &Device, n_mels: usize, frames: usize, value: f32) -> Tensor {
        let data = vec![value; n_mels * frames];
        Tensor::from_vec(data, (1, n_mels, frames), device).unwrap()
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_audio_embedding_shape_and_norm() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let mel = make_test_mel(model.device(), 80, 3000, 0.0);

        let emb = model.encode_audio(&mel).expect("encode_audio failed");
        assert_eq!(emb.dims(), &[1, 384]);
        assert_unit_norm(&emb, "audio embedding");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_audio_matryoshka_dims() {
        let (model, _tokenizer) = load_model_and_tokenizer();
        let mel = make_test_mel(model.device(), 80, 3000, 0.1);

        for dim in [32, 64, 128, 256, 384] {
            let emb = model
                .encode_audio_with_dim(&mel, Some(dim))
                .unwrap_or_else(|e| panic!("audio MRL dim={} failed: {:?}", dim, e));
            assert_eq!(emb.dims(), &[1, dim]);
            assert_unit_norm(&emb, &format!("audio MRL dim={}", dim));
        }
    }

    // ----------------------------------------------------------------
    // All three modalities together
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_all_modalities_shared_space_synthetic() {
        let (model, tokenizer) = load_model_and_tokenizer();

        let text_emb = to_vec(&encode_text_str(&model, &tokenizer, "A person speaking"));
        let img_emb = to_vec(
            &model
                .encode_image(&make_test_image(model.device(), 0.6, 0.4, 0.3))
                .unwrap(),
        );
        let audio_emb = to_vec(
            &model
                .encode_audio(&make_test_mel(model.device(), 80, 3000, 0.5))
                .unwrap(),
        );

        assert_eq!(text_emb.len(), 384);
        assert_eq!(img_emb.len(), 384);
        assert_eq!(audio_emb.len(), 384);

        let ti = cosine_sim(&text_emb, &img_emb);
        let ta = cosine_sim(&text_emb, &audio_emb);
        let ia = cosine_sim(&img_emb, &audio_emb);

        println!("text-image: {:.4}", ti);
        println!("text-audio: {:.4}", ta);
        println!("image-audio: {:.4}", ia);

        assert!(ti.is_finite(), "text-image sim must be finite");
        assert!(ta.is_finite(), "text-audio sim must be finite");
        assert!(ia.is_finite(), "image-audio sim must be finite");
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_all_modalities_shared_space_real_image() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let text_emb = to_vec(&encode_text_str(&model, &tokenizer, "A cute kitten"));
        let img_emb = to_vec(
            &model
                .encode_image(&image_bytes_to_tensor(
                    &download_bytes(WIKI_CAT_URL),
                    device,
                ))
                .unwrap(),
        );
        let audio_emb = to_vec(
            &model
                .encode_audio(&make_test_mel(device, 80, 3000, 0.5))
                .unwrap(),
        );

        assert_eq!(text_emb.len(), 384);
        assert_eq!(img_emb.len(), 384);
        assert_eq!(audio_emb.len(), 384);

        let ti = cosine_sim(&text_emb, &img_emb);
        let ta = cosine_sim(&text_emb, &audio_emb);
        let ia = cosine_sim(&img_emb, &audio_emb);

        println!(
            "real: text-image={:.4}, text-audio={:.4}, image-audio={:.4}",
            ti, ta, ia
        );

        assert!(ti.is_finite());
        assert!(ta.is_finite());
        assert!(ia.is_finite());
    }

    #[test]
    #[ignore = "requires model files and network"]
    fn test_all_modalities_base64_image() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let device = model.device();

        let text_emb = to_vec(&encode_text_str(&model, &tokenizer, "A fluffy puppy"));
        let dog_bytes = download_bytes(WIKI_DOG_URL);
        let img_emb = to_vec(
            &model
                .encode_image(&base64_roundtrip_to_tensor(&dog_bytes, device))
                .unwrap(),
        );
        let audio_emb = to_vec(
            &model
                .encode_audio(&make_test_mel(device, 80, 3000, -0.1))
                .unwrap(),
        );

        assert_eq!(text_emb.len(), 384);
        assert_eq!(img_emb.len(), 384);
        assert_eq!(audio_emb.len(), 384);

        let ti = cosine_sim(&text_emb, &img_emb);
        let ta = cosine_sim(&text_emb, &audio_emb);
        let ia = cosine_sim(&img_emb, &audio_emb);

        println!(
            "base64: text-image={:.4}, text-audio={:.4}, image-audio={:.4}",
            ti, ta, ia
        );

        assert!(ti.is_finite());
        assert!(ta.is_finite());
        assert!(ia.is_finite());
    }

    // ----------------------------------------------------------------
    // Validation / error handling
    // ----------------------------------------------------------------

    #[test]
    #[ignore = "requires model files"]
    fn test_text_invalid_layer_rejected() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let encoding = tokenizer.encode("Hi", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let device = model.device();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();

        let result = model.encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask),
            Some(0), // layer=0 is invalid (1-indexed)
            None,
        );
        assert!(result.is_err(), "layer=0 should be rejected");

        let result = model.encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask),
            Some(99), // exceeds number of layers
            None,
        );
        assert!(result.is_err(), "layer=99 should be rejected");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_text_invalid_dim_rejected() {
        let (model, tokenizer) = load_model_and_tokenizer();
        let encoding = tokenizer.encode("Hi", true).unwrap();
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let seq_len = ids.len();
        let device = model.device();
        let input_ids = Tensor::from_vec(ids, (1, seq_len), device).unwrap();
        let attention_mask = Tensor::from_vec(mask, (1, seq_len), device).unwrap();

        let result = model.encode_text_with_matryoshka(
            &input_ids,
            Some(&attention_mask),
            None,
            Some(999), // exceeds embedding_dim
        );
        assert!(result.is_err(), "dim=999 should be rejected");
    }

    /// Shape-only smoke test for the new SigLIPHead attentional probe pooling.
    ///
    /// Builds a `SigLIPHead` from synthetic randomly-initialized weights and
    /// confirms the forward pass collapses a `[batch, num_patches, hidden]`
    /// patch-token tensor down to `[batch, hidden]`. This exercises the head's
    /// own state-dict shape contract (probe, attention.in_proj_*,
    /// attention.out_proj.*, layernorm.*, mlp.fc1.*, mlp.fc2.*) and the
    /// forward arithmetic without requiring any model file on disk. It does
    /// NOT exercise the encoder-to-head wiring in `SigLIPVisionEncoder`;
    /// that integration is covered by the `#[ignore]`-gated diagnostic
    /// tests on the verification branch.
    ///
    /// Empirical correctness against real SigLIP weights is verified on the
    /// `candle-binding-siglip-pooling-fix-wip` branch via `#[ignore]`-gated
    /// diagnostic tests.
    #[test]
    fn test_siglip_head_forward_shape() {
        use std::collections::HashMap;

        let device = Device::Cpu;
        // Small synthetic dimensions; head_dim must divide hidden_size evenly.
        let hidden = 32usize;
        let num_heads = 4usize;
        let intermediate = 64usize;
        let batch = 2usize;
        let num_patches = 16usize;

        let config = MultiModalEmbeddingConfig {
            image_hidden_size: hidden,
            image_num_heads: num_heads,
            image_intermediate_size: intermediate,
            ..MultiModalEmbeddingConfig::default()
        };

        // Populate a backend HashMap with the exact state-dict keys SigLIPHead::load
        // expects under prefix "head.*". Random init for every tensor.
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let randn =
            |shape: &[usize]| -> Tensor { Tensor::randn(0.0f32, 1.0, shape, &device).unwrap() };
        let zeros =
            |shape: &[usize]| -> Tensor { Tensor::zeros(shape, DType::F32, &device).unwrap() };
        let ones =
            |shape: &[usize]| -> Tensor { Tensor::ones(shape, DType::F32, &device).unwrap() };

        tensors.insert("head.probe".into(), randn(&[1, 1, hidden]));
        tensors.insert(
            "head.attention.in_proj_weight".into(),
            randn(&[3 * hidden, hidden]),
        );
        tensors.insert("head.attention.in_proj_bias".into(), zeros(&[3 * hidden]));
        tensors.insert(
            "head.attention.out_proj.weight".into(),
            randn(&[hidden, hidden]),
        );
        tensors.insert("head.attention.out_proj.bias".into(), zeros(&[hidden]));
        tensors.insert("head.layernorm.weight".into(), ones(&[hidden]));
        tensors.insert("head.layernorm.bias".into(), zeros(&[hidden]));
        tensors.insert("head.mlp.fc1.weight".into(), randn(&[intermediate, hidden]));
        tensors.insert("head.mlp.fc1.bias".into(), zeros(&[intermediate]));
        tensors.insert("head.mlp.fc2.weight".into(), randn(&[hidden, intermediate]));
        tensors.insert("head.mlp.fc2.bias".into(), zeros(&[hidden]));

        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let head = SigLIPHead::load(vb.pp("head"), &config)
            .expect("SigLIPHead::load with synthetic weights should succeed");

        let xs = Tensor::randn(0.0f32, 1.0, (batch, num_patches, hidden), &device).unwrap();
        let out = head
            .forward(&xs)
            .expect("SigLIPHead::forward should succeed");

        assert_eq!(
            out.dims(),
            &[batch, hidden],
            "SigLIPHead.forward should pool [batch, num_patches, hidden] to [batch, hidden]"
        );
    }
}
