//! ModernBERT model implementation
//!
//! ModernBERT is a modernized bidirectional encoder-only Transformer model
//! supporting extended context windows up to 32K tokens via YaRN RoPE scaling.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    embedding, layer_norm_no_bias, linear, linear_no_bias, ops::softmax, Embedding, LayerNorm,
    Linear, Module, VarBuilder,
};
use serde::Deserialize;

use core::f32;
use std::collections::HashMap;
use std::sync::Arc;

use crate::model_architectures::attention::chunked_sdpa::{
    chunked_sdpa, prepare_padding_mask, ChunkedSdpaConfig, ATTN_QUERY_BLOCK,
};

// Flash Attention support (optional, requires flash-attn feature)
#[cfg(feature = "flash-attn")]
use candle_flash_attn::flash_attn;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
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
    #[serde(default)]
    #[serde(flatten)]
    pub classifier_config: Option<ClassifierConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Copy, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClassifierPooling {
    #[default]
    CLS,
    MEAN,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClassifierConfig {
    pub id2label: HashMap<String, String>,
    pub label2id: HashMap<String, String>,
    pub classifier_pooling: ClassifierPooling,
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, config: &Config, rope_theta: f64, dev: &Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let max_seq_len = config.max_position_embeddings;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &self.cos, &self.sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &self.cos, &self.sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Clone)]
struct ModernBertAttention {
    qkv: Linear,
    proj: Linear,
    num_attention_heads: usize,
    attention_head_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    use_flash_attn: bool,
}

impl ModernBertAttention {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        use_flash_attn: bool,
    ) -> Result<Self> {
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
            use_flash_attn,
        })
    }

    /// Project hidden states into `(b, heads, seq, head_dim)` q/k/v with RoPE applied.
    ///
    /// RoPE runs on the full q/k (cheap, O(seq*d)) before any chunking, so sliced
    /// query blocks keep position-correct rotations.
    fn project_qkv(&self, hidden_states: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq_len, _) = hidden_states.dims3()?;
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
        let (q, k) = self
            .rotary_emb
            .apply_rotary_emb_qkv(&qkv.get(0)?, &qkv.get(1)?)?;
        Ok((q, k, qkv.get(2)?))
    }

    /// Bidirectional attention over `(b, seq, hidden)` hidden states.
    ///
    /// `pad_mask` is the `(b, 1, 1, seq)` additive padding mask (`0` for real tokens,
    /// large negative for padding), broadcast over query positions. A local layer
    /// attends within `±window` of each query, a global layer to every key.
    /// `block_size` is the query-block size handed to the shared kernel.
    fn forward(
        &self,
        hidden_states: &Tensor,
        pad_mask: &Tensor,
        uses_local_attention: bool,
        window: usize,
        block_size: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, d) = hidden_states.dims3()?;
        let (q, k, v) = self.project_qkv(hidden_states)?;

        let scale = (self.attention_head_size as f64).powf(-0.5);

        // Memory-bounded attention: the shared kernel walks the query dimension in
        // blocks, so the full (b, heads, seq, seq) score matrix is never materialized.
        // A local layer maps to a sliding window; a global layer attends to every key.
        // The softmax scale is folded into the kernel.
        let cfg = ChunkedSdpaConfig {
            block_size,
            window: if uses_local_attention {
                Some(window)
            } else {
                None
            },
            causal: false,
            scale,
        };

        // Use Flash Attention if enabled, otherwise use the shared chunked kernel
        let xs = if self.use_flash_attn {
            #[cfg(feature = "flash-attn")]
            {
                // Flash Attention path
                // Flash Attention expects: [batch, seq_len, num_heads, head_dim]
                // Flash Attention requires f16/bf16, but we have F32
                // Convert to f16, run Flash Attention, then convert back to F32
                // `flash_attn` computes softmax(Q @ K^T . softmax_scale) @ V, so it
                // applies the scale itself and takes `q` unscaled. The upstream
                // candle-transformers model pre-scales `q` for its dense path only;
                // carrying that over here scaled twice, landing at 1/d.
                let q_flash = q.transpose(1, 2)?; // [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
                let k_flash = k.transpose(1, 2)?;
                let v_flash = v.transpose(1, 2)?;

                // Convert to f16 for Flash Attention
                let q_flash_f16 = q_flash.to_dtype(DType::F16)?;
                let k_flash_f16 = k_flash.to_dtype(DType::F16)?;
                let v_flash_f16 = v_flash.to_dtype(DType::F16)?;

                let softmax_scale = 1.0 / (self.attention_head_size as f32).sqrt();
                // ModernBERT is bidirectional (non-causal)
                match flash_attn(
                    &q_flash_f16,
                    &k_flash_f16,
                    &v_flash_f16,
                    softmax_scale,
                    false,
                ) {
                    Ok(attn_output_f16) => {
                        // Convert back to F32 and transpose back to [batch, num_heads, seq_len, head_dim]
                        attn_output_f16.to_dtype(DType::F32)?.transpose(1, 2)?
                    }
                    Err(e) => {
                        // Flash Attention failed, fallback to standard attention
                        eprintln!(
                            "⚠️  Flash Attention failed, falling back to standard attention: {}",
                            e
                        );
                        chunked_sdpa(&q, &k, &v, Some(pad_mask), &cfg)?
                    }
                }
            }
            #[cfg(not(feature = "flash-attn"))]
            {
                // flash-attn feature not enabled, use the chunked kernel
                chunked_sdpa(&q, &k, &v, Some(pad_mask), &cfg)?
            }
        } else {
            chunked_sdpa(&q, &k, &v, Some(pad_mask), &cfg)?
        };

        let xs = xs.transpose(1, 2)?.reshape((b, seq_len, d))?;
        let xs = xs.apply(&self.proj)?;
        let xs = xs.reshape((b, seq_len, d))?;

        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertMLP {
    wi: Linear,
    wo: Linear,
}

impl ModernBertMLP {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let wi = linear_no_bias(
            config.hidden_size,
            config.intermediate_size * 2,
            vb.pp("Wi"),
        )?;
        let wo = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("Wo"))?;
        Ok(Self { wi, wo })
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.wi)?;
        let xs = xs.chunk(2, D::Minus1)?;
        let xs = (&xs[0].gelu_erf()? * &xs[1])?.apply(&self.wo)?; // GeGLU
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertLayer {
    attn: ModernBertAttention,
    mlp: ModernBertMLP,
    attn_norm: Option<LayerNorm>,
    mlp_norm: LayerNorm,
    uses_local_attention: bool,
}

impl ModernBertLayer {
    fn load(
        vb: VarBuilder,
        config: &Config,
        rotary_emb: Arc<RotaryEmbedding>,
        uses_local_attention: bool,
    ) -> Result<Self> {
        let attn = ModernBertAttention::load(
            vb.pp("attn"),
            config,
            rotary_emb,
            cfg!(feature = "flash-attn"),
        )?;
        let mlp = ModernBertMLP::load(vb.pp("mlp"), config)?;
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
        pad_mask: &Tensor,
        window: usize,
        block_size: usize,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = xs.clone();
        if let Some(norm) = &self.attn_norm {
            xs = xs.apply(norm)?;
        }

        let xs = self
            .attn
            .forward(&xs, pad_mask, self.uses_local_attention, window, block_size)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.mlp_norm)?.apply(&self.mlp)?;
        let xs = (xs + mlp_out)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertHead {
    dense: Linear,
    norm: LayerNorm,
}

impl ModernBertHead {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let dense = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("dense"))?;
        let norm = layer_norm_no_bias(config.hidden_size, config.layer_norm_eps, vb.pp("norm"))?;
        Ok(Self { dense, norm })
    }
}

impl Module for ModernBertHead {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.dense)?.gelu_erf()?.apply(&self.norm)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertDecoder {
    decoder: Linear,
}

impl ModernBertDecoder {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let decoder_weights = vb.get(
            (config.vocab_size, config.hidden_size),
            "model.embeddings.tok_embeddings.weight",
        )?;
        let decoder_bias = vb.get(config.vocab_size, "decoder.bias")?;
        let decoder = Linear::new(decoder_weights, Some(decoder_bias));
        Ok(Self { decoder })
    }
}

impl Module for ModernBertDecoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.decoder)?;
        Ok(xs)
    }
}

// ModernBERT backbone
#[derive(Clone)]
pub struct ModernBert {
    word_embeddings: Embedding,
    norm: LayerNorm,
    layers: Vec<ModernBertLayer>,
    final_norm: LayerNorm,
    local_attention_size: usize,
}

impl ModernBert {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let word_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embeddings.tok_embeddings"),
        )?;
        let norm = layer_norm_no_bias(
            config.hidden_size,
            config.layer_norm_eps,
            vb.pp("model.embeddings.norm"),
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
            layers.push(ModernBertLayer::load(
                vb.pp(format!("model.layers.{layer_id}")),
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
            vb.pp("model.final_norm"),
        )?;

        Ok(Self {
            word_embeddings,
            norm,
            layers,
            final_norm,
            local_attention_size: config.local_attention,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // (b, 1, 1, seq) additive padding mask, broadcast over query positions. The
        // previous (b, 1, seq, seq) expansion and the (seq, seq) sliding-window band
        // were both O(seq^2); the window is now applied inside the kernel per block.
        let pad_mask = prepare_padding_mask(mask, DType::F32)?.to_device(xs.device())?;
        let window = self.local_attention_size / 2;
        let mut xs = xs.apply(&self.word_embeddings)?.apply(&self.norm)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &pad_mask, window, ATTN_QUERY_BLOCK)?;
        }
        let xs = xs.apply(&self.final_norm)?;
        Ok(xs)
    }
}

// ModernBERT for the fill-mask task
#[derive(Clone)]
pub struct ModernBertForMaskedLM {
    model: ModernBert,
    decoder: ModernBertDecoder,
    head: ModernBertHead,
}

impl ModernBertForMaskedLM {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let decoder = ModernBertDecoder::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            decoder,
            head,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let xs = self
            .model
            .forward(xs, mask)?
            .apply(&self.head)?
            .apply(&self.decoder)?;
        Ok(xs)
    }
}

#[derive(Clone)]
pub struct ModernBertClassifier {
    classifier: Linear,
}

impl ModernBertClassifier {
    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        // The decoder weights are tied with the embeddings layer weights
        let classifier = linear(
            config.hidden_size,
            config
                .classifier_config
                .as_ref()
                .map(|cc| cc.id2label.len())
                .unwrap_or_default(),
            vb.pp("classifier"),
        )?;
        Ok(Self { classifier })
    }
}

impl Module for ModernBertClassifier {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.classifier)?;
        softmax(&xs, D::Minus1)
    }
}

#[derive(Clone)]
pub struct ModernBertForSequenceClassification {
    model: ModernBert,
    head: ModernBertHead,
    classifier: ModernBertClassifier,
    classifier_pooling: ClassifierPooling,
}

impl ModernBertForSequenceClassification {
    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let model = ModernBert::load(vb.clone(), config)?;
        let classifier = ModernBertClassifier::load(vb.clone(), config)?;
        let head = ModernBertHead::load(vb.pp("head"), config)?;
        Ok(Self {
            model,
            head,
            classifier,
            classifier_pooling: config
                .classifier_config
                .as_ref()
                .map(|cc| cc.classifier_pooling)
                .unwrap_or_default(),
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let output = self.model.forward(xs, mask)?;
        let last_hidden_state = match self.classifier_pooling {
            ClassifierPooling::CLS => output.i((.., 0, ..))?,
            ClassifierPooling::MEAN => {
                let unsqueezed_mask = &mask.unsqueeze(D::Minus1)?.to_dtype(DType::F32)?;
                let sum_output = output.broadcast_mul(unsqueezed_mask)?.sum(1)?;
                sum_output.broadcast_div(&mask.sum_keepdim(1)?.to_dtype(DType::F32)?)?
            }
        };
        let xs = self
            .head
            .forward(&last_hidden_state)?
            .apply(&self.classifier)?;
        Ok(xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny ModernBERT config: enough heads and layers to exercise the attention
    /// block, small enough to run on CPU without model assets.
    fn tiny_config() -> Config {
        Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            intermediate_size: 64,
            max_position_embeddings: 256,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160000.0,
            local_attention: 8, // window = 4 each side
            local_rope_theta: 160000.0,
            classifier_config: None,
        }
    }

    /// Build an attention block with deterministic random weights.
    fn make_test_attention(config: &Config, device: &Device) -> ModernBertAttention {
        let hidden = config.hidden_size;
        let wqkv = Tensor::randn(0f32, 0.2f32, (hidden * 3, hidden), device).unwrap();
        let wo = Tensor::randn(0f32, 0.2f32, (hidden, hidden), device).unwrap();
        let rotary = Arc::new(
            RotaryEmbedding::new(DType::F32, config, config.global_rope_theta, device).unwrap(),
        );
        ModernBertAttention {
            qkv: Linear::new(wqkv, None),
            proj: Linear::new(wo, None),
            num_attention_heads: config.num_attention_heads,
            attention_head_size: hidden / config.num_attention_heads,
            rotary_emb: rotary,
            use_flash_attn: false,
        }
    }

    /// The dense attention this module used before the chunked kernel: a
    /// `(b, 1, seq, seq)` padding mask expansion plus a materialized `(seq, seq)`
    /// sliding-window band, added to the full `(b, heads, seq, seq)` score matrix.
    fn dense_reference_attention(
        attn: &ModernBertAttention,
        hidden_states: &Tensor,
        raw_mask: &Tensor,
        uses_local_attention: bool,
        window: usize,
    ) -> Tensor {
        let (b, seq_len, d) = hidden_states.dims3().unwrap();
        let device = hidden_states.device();

        // prepare_4d_attention_mask: (b, seq) -> (b, 1, seq, seq)
        let expanded = raw_mask
            .unsqueeze(1)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .expand((b, 1, seq_len, seq_len))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let global_mask = ((1.0 - expanded).unwrap() * f32::MIN as f64).unwrap();

        // get_local_attention_mask: (seq, seq) band, -inf outside |i - j| <= window
        let attention_mask = if uses_local_attention {
            let band: Vec<f32> = (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if (j as i32 - i as i32).abs() > window as i32 {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let band = Tensor::from_slice(&band, (seq_len, seq_len), device).unwrap();
            global_mask.broadcast_add(&band).unwrap()
        } else {
            global_mask
        };

        let qkv = hidden_states
            .apply(&attn.qkv)
            .unwrap()
            .reshape((
                b,
                seq_len,
                3,
                attn.num_attention_heads,
                attn.attention_head_size,
            ))
            .unwrap()
            .permute((2, 0, 3, 1, 4))
            .unwrap();
        let q = qkv.get(0).unwrap();
        let k = qkv.get(1).unwrap();
        let v = qkv.get(2).unwrap();
        let (q, k) = attn.rotary_emb.apply_rotary_emb_qkv(&q, &k).unwrap();

        let scale = (attn.attention_head_size as f64).powf(-0.5);
        let q = (q * scale).unwrap();

        // compute_standard_attention
        let att = q
            .matmul(
                &k.transpose(D::Minus2, D::Minus1)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            )
            .unwrap();
        let att = att.broadcast_add(&attention_mask).unwrap();
        let att = softmax(&att, D::Minus1).unwrap();
        let xs = att.matmul(&v.contiguous().unwrap()).unwrap();

        let xs = xs
            .transpose(1, 2)
            .unwrap()
            .reshape((b, seq_len, d))
            .unwrap();
        xs.apply(&attn.proj).unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    /// All-real padding mask in the raw `(b, seq)` form the backbone receives.
    fn all_real_mask(b: usize, seq_len: usize, device: &Device) -> Tensor {
        Tensor::ones((b, seq_len), DType::F32, device).unwrap()
    }

    #[test]
    fn test_chunked_attention_matches_dense() {
        let device = Device::Cpu;
        let config = tiny_config();
        let attn = make_test_attention(&config, &device);
        let window = config.local_attention / 2;

        // Cover global + local layers over several block sizes: a divisor of the
        // sequence, a non-divisor, a single block, and a block smaller than the window.
        for &uses_local in &[false, true] {
            for &seq_len in &[1usize, 5, 16, 40] {
                let hidden =
                    Tensor::randn(0f32, 1f32, (1, seq_len, config.hidden_size), &device).unwrap();
                let raw_mask = all_real_mask(1, seq_len, &device);
                let pad_mask = prepare_padding_mask(&raw_mask, DType::F32).unwrap();
                let reference =
                    dense_reference_attention(&attn, &hidden, &raw_mask, uses_local, window);

                for &block in &[1usize, 3, 8, 16, ATTN_QUERY_BLOCK] {
                    let chunked = attn
                        .forward(&hidden, &pad_mask, uses_local, window, block)
                        .unwrap();
                    let diff = max_abs_diff(&chunked, &reference);
                    assert!(
                        diff < 1e-4,
                        "local={} seq={} block={}: max|Δ|={}",
                        uses_local,
                        seq_len,
                        block,
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_chunked_attention_matches_dense_with_padding() {
        let device = Device::Cpu;
        let config = tiny_config();
        let attn = make_test_attention(&config, &device);
        let window = config.local_attention / 2;
        let seq_len = 24;

        // Last 7 positions are padding.
        let mut mask_vec = vec![1f32; seq_len];
        for m in mask_vec.iter_mut().skip(seq_len - 7) {
            *m = 0.0;
        }
        let raw_mask = Tensor::from_vec(mask_vec, (1, seq_len), &device).unwrap();
        let pad_mask = prepare_padding_mask(&raw_mask, DType::F32).unwrap();
        let hidden = Tensor::randn(0f32, 1f32, (1, seq_len, config.hidden_size), &device).unwrap();

        for &uses_local in &[false, true] {
            let reference =
                dense_reference_attention(&attn, &hidden, &raw_mask, uses_local, window);
            for &block in &[3usize, 8, ATTN_QUERY_BLOCK] {
                let chunked = attn
                    .forward(&hidden, &pad_mask, uses_local, window, block)
                    .unwrap();
                let diff = max_abs_diff(&chunked, &reference);
                assert!(
                    diff < 1e-4,
                    "padding local={} block={}: max|Δ|={}",
                    uses_local,
                    block,
                    diff
                );
            }
        }
    }
}
