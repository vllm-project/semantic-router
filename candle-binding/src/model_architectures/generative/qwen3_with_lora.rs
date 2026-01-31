//! Qwen3 Model with LoRA Support
//!
//! This is a modified version of the official candle-transformers Qwen3 model
//! that includes LoRA (Low-Rank Adaptation) hooks for fine-tuning.
//!
//! Key modifications from official implementation:
//! 1. Added LoRA adapter fields to Qwen3Attention and Qwen3MLP
//! 2. Modified forward passes to apply LoRA deltas: output += LoRA_B(LoRA_A(input)) * scaling
//! 3. Added `inject_lora_adapters()` method to dynamically load adapters
//!
//! Based on: huggingface/candle @ candle-transformers/src/models/qwen3.rs

use crate::model_architectures::lora::LoRAAdapter;
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    kv_cache::KvCache, linear, linear_no_bias, rms_norm, Activation, Embedding, Linear, RmsNorm,
    VarBuilder,
};
use candle_transformers::utils::repeat_kv;
use std::collections::HashMap;
use std::sync::Arc;

// Re-export Config from official implementation
pub use candle_transformers::models::qwen3::Config;

#[derive(Debug, Clone)]
struct Qwen3RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl Qwen3RotaryEmbedding {
    fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(DType::F32)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Qwen3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,

    // LoRA adapters
    gate_proj_lora: Option<Arc<LoRAAdapter>>,
    up_proj_lora: Option<Arc<LoRAAdapter>>,
    down_proj_lora: Option<Arc<LoRAAdapter>>,
}

impl Qwen3MLP {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
            gate_proj_lora: None,
            up_proj_lora: None,
            down_proj_lora: None,
        })
    }
}

impl Module for Qwen3MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Apply gate projection with LoRA
        let mut gate = self.gate_proj.forward(x)?;
        if let Some(lora) = &self.gate_proj_lora {
            let delta = lora.forward(x, false)?;
            gate = (gate + delta)?;
        }
        let lhs = gate.apply(&self.act_fn)?;

        // Apply up projection with LoRA
        let mut up = self.up_proj.forward(x)?;
        if let Some(lora) = &self.up_proj_lora {
            let delta = lora.forward(x, false)?;
            up = (up + delta)?;
        }

        // Combine gate and up
        let combined = (lhs * up)?;

        // Apply down projection with LoRA
        let mut output = self.down_proj.forward(&combined)?;
        if let Some(lora) = &self.down_proj_lora {
            let delta = lora.forward(&combined, false)?;
            output = (output + delta)?;
        }

        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct Qwen3Attention {
    // Base projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    // Norms
    q_norm: RmsNorm,
    k_norm: RmsNorm,

    // Hyper params
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,

    // Utils
    rotary_emb: Arc<Qwen3RotaryEmbedding>,
    kv_cache: KvCache,

    // LoRA adapters
    q_proj_lora: Option<Arc<LoRAAdapter>>,
    k_proj_lora: Option<Arc<LoRAAdapter>>,
    v_proj_lora: Option<Arc<LoRAAdapter>>,
    o_proj_lora: Option<Arc<LoRAAdapter>>,
}

impl Qwen3Attention {
    fn new(cfg: &Config, rotary_emb: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        if cfg.use_sliding_window {
            candle_core::bail!("sliding window is not supported")
        }

        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        // Check if model uses bias
        let use_bias = cfg.attention_bias;

        let q_proj = if use_bias {
            linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj"))?
        };
        let k_proj = if use_bias {
            linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?
        };
        let v_proj = if use_bias {
            linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        } else {
            linear_no_bias(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?
        };
        let o_proj = if use_bias {
            linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?
        } else {
            linear_no_bias(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?
        };

        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        let hidden_size = head_dim * cfg.num_attention_heads;
        let kv_cache = KvCache::new(2, 512);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            q_proj_lora: None,
            k_proj_lora: None,
            v_proj_lora: None,
            o_proj_lora: None,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        // 1. Projections with LoRA
        let mut q = self.q_proj.forward(x)?;
        if let Some(lora) = &self.q_proj_lora {
            let delta = lora.forward(x, false)?;
            q = (q + delta)?;
        }

        let mut k = self.k_proj.forward(x)?;
        if let Some(lora) = &self.k_proj_lora {
            let delta = lora.forward(x, false)?;
            k = (k + delta)?;
        }

        let mut v = self.v_proj.forward(x)?;
        if let Some(lora) = &self.v_proj_lora {
            let delta = lora.forward(x, false)?;
            v = (v + delta)?;
        }

        // 2. Reshape: (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Per-head RMSNorm
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        // 4. RoPE
        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // 5. Accumulate KV cache
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // 6. GQA repeat_kv
        let k = repeat_kv(k, self.num_kv_groups)?;
        let v = repeat_kv(v, self.num_kv_groups)?;

        // 7. Attention score
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        // 8. Output projection with LoRA
        let reshaped = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
        let mut output = self.o_proj.forward(&reshaped)?;
        if let Some(lora) = &self.o_proj_lora {
            let delta = lora.forward(&reshaped, false)?;
            output = (output + delta)?;
        }

        Ok(output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, rotary: Arc<Qwen3RotaryEmbedding>, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3Attention::new(cfg, rotary, vb.pp("self_attn"))?;
        let mlp = Qwen3MLP::new(cfg, vb.pp("mlp"))?;
        let ln1 = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let ln2 = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let rotary = Arc::new(Qwen3RotaryEmbedding::new(vb.dtype(), cfg, vb.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, rotary.clone(), vb_l.pp(i))?);
        }
        Ok(Self {
            embed_tokens,
            layers,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn clear_kv_cache(&mut self) {
        for l in &mut self.layers {
            l.clear_kv_cache();
        }
    }

    /// Snapshot KV caches for all layers (clone, cheap metadata copy)
    pub fn kv_cache_snapshot(&self) -> Vec<candle_nn::kv_cache::KvCache> {
        self.layers
            .iter()
            .map(|l| l.self_attn.kv_cache.clone())
            .collect()
    }

    /// Restore KV caches from a prior snapshot
    pub fn kv_cache_restore(&mut self, caches: &[candle_nn::kv_cache::KvCache]) {
        for (layer, cache) in self.layers.iter_mut().zip(caches.iter()) {
            layer.self_attn.kv_cache = cache.clone();
        }
    }

    /// Process prefix tokens and return for caching
    ///
    /// This processes the prefix through the model and the KV cache is automatically
    /// maintained. The caller can then continue with suffix tokens.
    ///
    /// Returns the prefix length for future reference.
    pub fn process_prefix(&mut self, prefix_tokens: &[u32]) -> Result<usize> {
        let prefix_len = prefix_tokens.len();

        // Create tensor from prefix tokens
        let input = Tensor::new(prefix_tokens, &self.device)?.unsqueeze(0)?;

        // Forward pass (KV cache is accumulated automatically)
        self.forward(&input, 0)?;

        Ok(prefix_len)
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;

        let causal = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };

        for layer in &mut self.layers {
            h = layer.forward(&h, causal.as_ref(), offset)?;
        }
        self.norm.forward(&h)
    }

    /// Inject LoRA adapters into the model
    ///
    /// # Arguments
    /// - `adapters`: HashMap of adapters indexed by "layers.{idx}.{module}.{projection}"
    ///   Example keys: "layers.0.self_attn.q_proj", "layers.0.mlp.gate_proj"
    pub fn inject_lora_adapters(&mut self, adapters: HashMap<String, Arc<LoRAAdapter>>) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // Inject attention LoRA adapters
            if let Some(adapter) = adapters.get(&format!("layers.{}.self_attn.q_proj", layer_idx)) {
                layer.self_attn.q_proj_lora = Some(adapter.clone());
            }
            if let Some(adapter) = adapters.get(&format!("layers.{}.self_attn.k_proj", layer_idx)) {
                layer.self_attn.k_proj_lora = Some(adapter.clone());
            }
            if let Some(adapter) = adapters.get(&format!("layers.{}.self_attn.v_proj", layer_idx)) {
                layer.self_attn.v_proj_lora = Some(adapter.clone());
            }
            if let Some(adapter) = adapters.get(&format!("layers.{}.self_attn.o_proj", layer_idx)) {
                layer.self_attn.o_proj_lora = Some(adapter.clone());
            }

            // Inject MLP LoRA adapters
            if let Some(adapter) = adapters.get(&format!("layers.{}.mlp.gate_proj", layer_idx)) {
                layer.mlp.gate_proj_lora = Some(adapter.clone());
            }
            if let Some(adapter) = adapters.get(&format!("layers.{}.mlp.up_proj", layer_idx)) {
                layer.mlp.up_proj_lora = Some(adapter.clone());
            }
            if let Some(adapter) = adapters.get(&format!("layers.{}.mlp.down_proj", layer_idx)) {
                layer.mlp.down_proj_lora = Some(adapter.clone());
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    base: Model,
    lm_head: Linear,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base = Model::new(cfg, vb.clone())?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(base.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { base, lm_head })
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (_, l) = input.dims2()?;
        self.base
            .forward(input, offset)?
            .narrow(1, l - 1, 1)?
            .apply(&self.lm_head)
    }

    /// Forward over all time steps (returns logits for each position)
    pub fn forward_all(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let h = self.base.forward(input, offset)?;
        h.apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base.clear_kv_cache();
    }

    /// Snapshot KV caches for reuse (e.g., to reuse prompt prefix)
    pub fn kv_cache_snapshot(&self) -> Vec<candle_nn::kv_cache::KvCache> {
        self.base.kv_cache_snapshot()
    }

    /// Restore KV caches from a prior snapshot
    pub fn kv_cache_restore(&mut self, caches: &[candle_nn::kv_cache::KvCache]) {
        self.base.kv_cache_restore(caches);
    }

    /// Process prefix tokens for caching
    pub fn process_prefix(&mut self, prefix_tokens: &[u32]) -> Result<usize> {
        self.base.process_prefix(prefix_tokens)
    }

    /// Inject LoRA adapters into the base model
    pub fn inject_lora_adapters(&mut self, adapters: HashMap<String, Arc<LoRAAdapter>>) {
        self.base.inject_lora_adapters(adapters);
    }
}
