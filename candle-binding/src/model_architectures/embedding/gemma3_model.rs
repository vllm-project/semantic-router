//! Gemma3 Transformer Backbone for EmbeddingGemma-300M
//!
//! This module implements the core Gemma3 Transformer model used as the backbone
//! for EmbeddingGemma-300M. It includes:
//! - **RmsNorm**: Root Mean Square Layer Normalization
//! - **RotaryEmbedding**: Rotary Position Embeddings (RoPE) with local base frequency
//! - **Gemma3Attention**: Multi-Query Attention (MQA) with mixed attention pattern
//! - **Gemma3MLP**: Feed-forward network with gelu_pytorch_tanh activation
//! - **Gemma3Layer**: Complete transformer layer (pre-norm architecture)
//! - **Gemma3Model**: Full model with 24 transformer layers
//!
//! ## Architecture (EmbeddingGemma-300M)
//! - Layers: 24 transformer blocks
//! - Hidden size: 768
//! - Attention: MQA (3 query heads, 1 KV head)
//! - Head dimension: 256 (explicitly specified)
//! - MLP intermediate size: 1152
//! - Max sequence length: 2048
//! - RoPE: theta=1000000.0, local_base_freq=10000.0
//! - Mixed attention: Sliding window (512) + Full attention
//!
//! ## Key Differences from Qwen3
//! 1. **MQA vs GQA**: Gemma3 uses Multi-Query Attention (1 KV head) instead of Grouped Query Attention (8 KV heads)
//! 2. **Mixed Attention**: Alternating between sliding window (512) and full attention
//! 3. **Bidirectional Attention**: No causal masking (encoder model, not decoder)
//! 4. **gelu_pytorch_tanh**: Different MLP activation function
//! 5. **RoPE Local Base Freq**: 10000.0 (in addition to global theta=1000000.0)
//!
//! ## References
//! - TEI Gemma3: https://github.com/huggingface/text-embeddings-inference/blob/main/backends/candle/src/models/gemma3.rs
//! - Official model: https://huggingface.co/google/embeddinggemma-300m

use super::gemma_embedding::{AttentionLayerType, GemmaEmbeddingConfig};
use crate::core::{config_errors, from_candle_error, ModelErrorType, UnifiedError, UnifiedResult};
use crate::model_architectures::attention::chunked_sdpa::{
    chunked_sdpa, ChunkedSdpaConfig, ATTN_QUERY_BLOCK,
};
use candle_core::{Device, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, Module, VarBuilder};

// ============================================================================
// Helper Functions
// ============================================================================

// ============================================================================
// RmsNorm - Reused from Qwen3 (same implementation)
// ============================================================================

/// Root Mean Square Layer Normalization
///
/// RmsNorm normalizes the input by the root mean square of the activations,
/// providing a simpler alternative to LayerNorm without centering.
///
/// # Formula
/// ```text
/// RmsNorm(x) = (x / RMS(x)) * weight
/// where RMS(x) = sqrt(mean(x^2) + eps)
/// ```
///
/// # Usage in Gemma3
/// - Applied before attention (input_layernorm)
/// - Applied before MLP (post_attention_layernorm)
/// - Applied after all transformer layers (final norm)
#[derive(Debug)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Create a new RmsNorm layer
    ///
    /// # Arguments
    /// - `weight`: Learnable scale parameter, shape [hidden_size]
    /// - `eps`: Epsilon for numerical stability (typically 1e-6)
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    /// Load RmsNorm from VarBuilder
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights
    /// - `hidden_size`: Dimension of the input/output
    /// - `eps`: Epsilon for numerical stability
    pub fn load(vb: VarBuilder, hidden_size: usize, eps: f64) -> UnifiedResult<Self> {
        let weight = vb
            .get(hidden_size, "weight")
            .map_err(|e| config_errors::missing_field("weight", &format!("RmsNorm: {}", e)))?;
        Ok(Self::new(weight, eps))
    }

    /// Apply RMS normalization
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [..., hidden_size]
    ///
    /// # Returns
    /// Normalized tensor with same shape as input
    pub fn forward(&self, x: &Tensor) -> UnifiedResult<Tensor> {
        let x_squared = x
            .sqr()
            .map_err(|e| from_candle_error(e, "RmsNorm: compute x^2", None))?;
        let mean_squared = x_squared
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| from_candle_error(e, "RmsNorm: compute mean(x^2)", None))?;
        let mean_plus_eps = (mean_squared + self.eps)
            .map_err(|e| from_candle_error(e, "RmsNorm: add epsilon", None))?;
        let rms = mean_plus_eps
            .sqrt()
            .map_err(|e| from_candle_error(e, "RmsNorm: compute sqrt", None))?;
        let normalized = x
            .broadcast_div(&rms)
            .map_err(|e| from_candle_error(e, "RmsNorm: normalize (x / rms)", None))?;
        // Gemma3 scales by (1.0 + weight). See huggingface/transformers#29402.
        let one_plus_weight = (&self.weight + 1.0)
            .map_err(|e| from_candle_error(e, "RmsNorm: 1.0 + weight", None))?;
        normalized
            .broadcast_mul(&one_plus_weight)
            .map_err(|e| from_candle_error(e, "RmsNorm: scale by (1.0 + weight)", None))
    }
}

// ============================================================================
// RotaryEmbedding - Gemma3-specific (with local_base_freq)
// ============================================================================

/// Rotary Position Embedding (RoPE) Cache for Gemma3
///
/// Gemma3 uses RoPE with two frequency parameters:
/// - `rope_theta` (global): 1000000.0 (for long context)
/// - `rope_local_base_freq`: 10000.0 (for local position encoding)
///
/// # RoPE Formula
/// ```text
/// freq_i = 1.0 / (local_base_freq^(2i/d))  for i in [0, d/2)
/// cos_cached[pos, i] = cos(pos * freq_i)
/// sin_cached[pos, i] = sin(pos * freq_i)
/// ```
///
/// # Application to Q and K
/// ```text
/// Q_rope = [Q_even * cos - Q_odd * sin, Q_odd * cos + Q_even * sin]
/// K_rope = [K_even * cos - K_odd * sin, K_odd * cos + K_even * sin]
/// ```
#[derive(Debug)]
pub struct RotaryEmbeddingCache {
    cos_cached: Tensor, // [max_seq_len, head_dim]
    sin_cached: Tensor, // [max_seq_len, head_dim]
    head_dim: usize,
}

impl RotaryEmbeddingCache {
    /// Create a new RotaryEmbeddingCache
    ///
    /// # Arguments
    /// - `head_dim`: Dimension of each attention head (must be even)
    /// - `max_seq_len`: Maximum sequence length
    /// - `rope_local_base_freq`: Local base frequency (10000.0 for Gemma3)
    /// - `device`: Device to store the cache
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_local_base_freq: f32,
        device: &Device,
    ) -> UnifiedResult<Self> {
        if !head_dim.is_multiple_of(2) {
            return Err(UnifiedError::Validation {
                field: "head_dim".to_string(),
                expected: "even number".to_string(),
                actual: head_dim.to_string(),
                context: Some("RoPE requires even head dimension".to_string()),
            });
        }

        // Step 1: Compute frequency for each dimension pair
        // freq_i = 1.0 / (local_base_freq^(2i/d))  for i in [0, d/2)
        let half_dim = head_dim / 2;
        let mut freqs = Vec::with_capacity(half_dim);

        for i in 0..half_dim {
            let exponent = (2 * i) as f64 / head_dim as f64;
            let freq = 1.0 / (rope_local_base_freq as f64).powf(exponent);
            freqs.push(freq);
        }

        // Convert freqs to tensor: [head_dim/2]
        // Convert f64 to f32 for tensor creation
        let freqs_f32: Vec<f32> = freqs.iter().map(|&f| f as f32).collect();
        let freqs_tensor = Tensor::from_vec(freqs_f32, (half_dim,), device)
            .map_err(|e| from_candle_error(e, "RoPE: create freqs tensor", None))?;

        // Step 2: Expand freqs to [head_dim] by concatenating with itself
        // This is critical: Python repeats the first half, not interleaves
        // freqs_expanded = [freq[0], freq[1], ..., freq[63], freq[0], freq[1], ..., freq[63]]  (for head_dim=128)
        let freqs_expanded = Tensor::cat(&[&freqs_tensor, &freqs_tensor], 0)
            .map_err(|e| from_candle_error(e, "RoPE: expand freqs", None))?;

        // Step 3: Create position tensor: [max_seq_len]
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let position_tensor = Tensor::from_vec(positions, (max_seq_len,), device)
            .map_err(|e| from_candle_error(e, "RoPE: create position tensor", None))?;

        // Step 4: Compute outer product: position[i] * freq[j]
        // position_tensor: [max_seq_len] -> [max_seq_len, 1]
        // freqs_expanded: [head_dim] -> [1, head_dim]
        // result: [max_seq_len, head_dim]
        let position_expanded = position_tensor
            .unsqueeze(1)
            .map_err(|e| from_candle_error(e, "RoPE: unsqueeze position", None))?;
        let freqs_expanded_2d = freqs_expanded
            .unsqueeze(0)
            .map_err(|e| from_candle_error(e, "RoPE: unsqueeze freqs", None))?;

        let angles = position_expanded
            .broadcast_mul(&freqs_expanded_2d)
            .map_err(|e| from_candle_error(e, "RoPE: compute angles", None))?;

        // Step 5: Precompute cos and sin
        let cos_cached = angles
            .cos()
            .map_err(|e| from_candle_error(e, "RoPE: compute cos", None))?;
        let sin_cached = angles
            .sin()
            .map_err(|e| from_candle_error(e, "RoPE: compute sin", None))?;

        Ok(Self {
            cos_cached,
            sin_cached,
            head_dim,
        })
    }

    /// Apply rotary position embedding to query or key tensor
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [batch, num_heads, seq_len, head_dim]
    /// - `position_ids`: Position indices, shape [batch, seq_len]
    ///
    /// # Returns
    /// Tensor with RoPE applied, shape [batch, num_heads, seq_len, head_dim]
    pub fn apply_rotary_emb(&self, x: &Tensor, position_ids: &Tensor) -> UnifiedResult<Tensor> {
        let (batch_size, _num_heads, seq_len, head_dim) = x
            .dims4()
            .map_err(|e| from_candle_error(e, "RoPE apply: get x dims", None))?;

        if head_dim != self.head_dim {
            return Err(UnifiedError::Validation {
                field: "head_dim".to_string(),
                expected: self.head_dim.to_string(),
                actual: head_dim.to_string(),
                context: Some("RoPE head_dim mismatch".to_string()),
            });
        }

        // Step 1: Extract cos and sin for the given positions
        // position_ids: [batch, seq_len]
        // cos_cached: [max_seq_len, head_dim]
        // We need: [batch, 1, seq_len, head_dim] for broadcasting

        // Flatten position_ids to [batch * seq_len]
        let positions_flat = position_ids
            .flatten_all()
            .map_err(|e| from_candle_error(e, "RoPE apply: flatten positions", None))?;

        // Index cos and sin: [batch * seq_len, head_dim]
        let cos_selected = self
            .cos_cached
            .index_select(&positions_flat, 0)
            .map_err(|e| from_candle_error(e, "RoPE apply: index cos", None))?;
        let sin_selected = self
            .sin_cached
            .index_select(&positions_flat, 0)
            .map_err(|e| from_candle_error(e, "RoPE apply: index sin", None))?;

        // Reshape to [batch, seq_len, head_dim]
        let cos_reshaped = cos_selected
            .reshape((batch_size, seq_len, head_dim))
            .map_err(|e| from_candle_error(e, "RoPE apply: reshape cos", None))?;
        let sin_reshaped = sin_selected
            .reshape((batch_size, seq_len, head_dim))
            .map_err(|e| from_candle_error(e, "RoPE apply: reshape sin", None))?;

        // Unsqueeze to [batch, 1, seq_len, head_dim] for broadcasting
        let cos = cos_reshaped
            .unsqueeze(1)
            .map_err(|e| from_candle_error(e, "RoPE apply: unsqueeze cos", None))?;
        let sin = sin_reshaped
            .unsqueeze(1)
            .map_err(|e| from_candle_error(e, "RoPE apply: unsqueeze sin", None))?;

        // Step 2: Apply RoPE following Python Gemma official implementation
        // Python: rotate_half(x) = cat([-x2, x1]), where x1=x[..., :half], x2=x[..., half:]
        // Python: x_embed = (x * cos) + (rotate_half(x) * sin)

        let half_dim = head_dim / 2;

        // Step 2.1: Compute x * cos
        let x_cos = x
            .broadcast_mul(&cos)
            .map_err(|e| from_candle_error(e, "RoPE apply: x * cos", None))?;

        // Step 2.2: Compute rotate_half(x)
        // x1: first half [0:half_dim]
        let x1 = x
            .narrow(3, 0, half_dim)
            .map_err(|e| from_candle_error(e, "RoPE apply: narrow x1", None))?;

        // x2: second half [half_dim:head_dim]
        let x2 = x
            .narrow(3, half_dim, half_dim)
            .map_err(|e| from_candle_error(e, "RoPE apply: narrow x2", None))?;

        // rotate_half(x) = cat([-x2, x1])
        let neg_x2 = x2
            .neg()
            .map_err(|e| from_candle_error(e, "RoPE apply: negate x2", None))?;
        let rotate_half_x = Tensor::cat(&[neg_x2, x1], 3)
            .map_err(|e| from_candle_error(e, "RoPE apply: cat rotate_half", None))?;

        // Step 2.3: Compute rotate_half(x) * sin
        let rotate_half_x_sin = rotate_half_x
            .broadcast_mul(&sin)
            .map_err(|e| from_candle_error(e, "RoPE apply: rotate_half(x) * sin", None))?;

        // Step 2.4: x_embed = (x * cos) + (rotate_half(x) * sin)
        x_cos
            .add(&rotate_half_x_sin)
            .map_err(|e| from_candle_error(e, "RoPE apply: x*cos + rotate_half(x)*sin", None))
    }
}

// ============================================================================
// Gemma3 MLP (Feed-Forward Network)
// ============================================================================

/// Gemma3 MLP (Feed-Forward Network)
///
/// Architecture:
/// ```text
/// hidden_states [batch, seq_len, 768]
///   ↓ gate_proj (768 → 1152)
///   ↓ gelu_pytorch_tanh
///   ↓ down_proj (1152 → 768)
/// output [batch, seq_len, 768]
/// ```
///
/// # Key Differences from Qwen3
/// - **Activation**: gelu_pytorch_tanh (not SwiGLU)
/// - **No up_proj**: Single gate projection (not gated)
#[derive(Debug)]
pub struct Gemma3MLP {
    gate_proj: Linear,
    up_proj: Linear, // Added: for SwiGLU activation
    down_proj: Linear,
}

impl Gemma3MLP {
    /// Load Gemma3MLP from VarBuilder
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights
    /// - `config`: GemmaEmbeddingConfig
    pub fn load(vb: VarBuilder, config: &GemmaEmbeddingConfig) -> UnifiedResult<Self> {
        let gate_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("gate_proj"),
        )
        .map_err(|e| from_candle_error(e, "Gemma3MLP: load gate_proj", None))?;

        let up_proj = linear_no_bias(
            config.hidden_size,
            config.intermediate_size,
            vb.pp("up_proj"),
        )
        .map_err(|e| from_candle_error(e, "Gemma3MLP: load up_proj", None))?;

        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )
        .map_err(|e| from_candle_error(e, "Gemma3MLP: load down_proj", None))?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// Forward pass through MLP
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    /// Output tensor, shape [batch, seq_len, hidden_size]
    pub fn forward(&self, x: &Tensor) -> UnifiedResult<Tensor> {
        let gate_output = self
            .gate_proj
            .forward(x)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: gate_proj", None))?;
        let gate_activated = Self::gelu_pytorch_tanh(&gate_output)?;
        let up_output = self
            .up_proj
            .forward(x)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: up_proj", None))?;
        let gated = gate_activated
            .mul(&up_output)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: gate * up", None))?;
        self.down_proj
            .forward(&gated)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: down_proj", None))
    }

    /// Helper function to compute tensor statistics
    fn compute_tensor_stats(tensor: &Tensor) -> (f32, f32, f32, f32) {
        let vec = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let count = vec.len() as f32;
        let sum: f32 = vec.iter().sum();
        let mean = sum / count;
        let variance: f32 = vec.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count;
        let std = variance.sqrt();
        let min = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        (mean, std, min, max)
    }

    /// GELU activation with PyTorch's tanh approximation
    ///
    /// Formula: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    fn gelu_pytorch_tanh(x: &Tensor) -> UnifiedResult<Tensor> {
        const SQRT_2_OVER_PI: f64 = 0.7978845608028654; // sqrt(2/π)
        const COEFF: f64 = 0.044715;

        // x^3
        let x_cubed = x
            .powf(3.0)
            .map_err(|e| from_candle_error(e, "GELU: compute x^3", None))?;

        // 0.044715 * x^3
        let coeff_x_cubed = (x_cubed * COEFF)
            .map_err(|e| from_candle_error(e, "GELU: multiply coeff * x^3", None))?;

        // x + 0.044715 * x^3
        let inner = x
            .add(&coeff_x_cubed)
            .map_err(|e| from_candle_error(e, "GELU: x + coeff * x^3", None))?;

        // sqrt(2/π) * (x + 0.044715 * x^3)
        let scaled = (inner * SQRT_2_OVER_PI)
            .map_err(|e| from_candle_error(e, "GELU: scale inner", None))?;

        // tanh(...)
        let tanh_result = scaled
            .tanh()
            .map_err(|e| from_candle_error(e, "GELU: tanh", None))?;

        // 1 + tanh(...)
        let one_plus_tanh =
            (tanh_result + 1.0).map_err(|e| from_candle_error(e, "GELU: 1 + tanh", None))?;

        // x * (1 + tanh(...))
        let x_times_result = x
            .broadcast_mul(&one_plus_tanh)
            .map_err(|e| from_candle_error(e, "GELU: x * (1 + tanh)", None))?;

        // 0.5 * x * (1 + tanh(...))
        (x_times_result * 0.5).map_err(|e| from_candle_error(e, "GELU: final multiply 0.5", None))
    }
}

// ============================================================================
// Gemma3 Attention (Multi-Query Attention with Mixed Pattern)
// ============================================================================

/// Gemma3 Multi-Query Attention (MQA)
///
/// # Architecture (EmbeddingGemma-300M)
/// - Q heads: 3 (`num_attention_heads`)
/// - KV heads: 1 (`num_key_value_heads`) - **Multi-Query Attention**
/// - Head dimension: 256 (explicitly specified)
/// - Scaling: 1/sqrt(256) ≈ 0.0625
///
/// # MQA (Multi-Query Attention)
/// Unlike GQA where multiple Q heads share a group of KV heads, MQA has all Q heads
/// share a SINGLE set of K and V:
/// ```text
/// GQA (Qwen3): Q[16 heads] × K[8 heads] × V[8 heads] (repeat K/V 2x)
/// MQA (Gemma3): Q[3 heads]  × K[1 head]  × V[1 head]  (repeat K/V 3x)
/// ```
///
/// # Mixed Attention Pattern
/// - **Sliding Attention**: Local attention with 512-token window
/// - **Full Attention**: Global attention across all tokens
/// - Pattern: Layers 0-4, 6-10, 12-16, 18-22 use sliding; Layers 5, 11, 17, 23 use full
///
/// # Bidirectional Attention
/// - No causal masking (encoder model, not decoder)
/// - Attention mask only for padding
#[derive(Debug)]
pub struct Gemma3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm, // RMSNorm for query states (after projection, before RoPE)
    k_norm: RmsNorm, // RMSNorm for key states (after projection, before RoPE)
    rope_cache_global: RotaryEmbeddingCache, // base=1000000, for full_attention
    rope_cache_local: RotaryEmbeddingCache, // base=10000, for sliding_attention
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    hidden_size: usize,
    attention_type: AttentionLayerType,
    sliding_window: usize,
    layer_idx: usize, // Layer index for debugging
}

impl Gemma3Attention {
    /// Load Gemma3Attention from VarBuilder
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights
    /// - `config`: GemmaEmbeddingConfig
    /// - `layer_idx`: Index of this layer (for determining attention type)
    pub fn load(
        vb: VarBuilder,
        config: &GemmaEmbeddingConfig,
        layer_idx: usize,
    ) -> UnifiedResult<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;

        // Validate MQA configuration
        if num_key_value_heads != 1 {
            return Err(UnifiedError::Model {
                model_type: ModelErrorType::Embedding,
                operation: "Gemma3Attention: validate MQA".to_string(),
                context: Some(format!(
                    "EmbeddingGemma expects MQA (num_key_value_heads=1), got {}",
                    num_key_value_heads
                )),
                source: "".to_string(),
            });
        }

        // Load projection layers (no bias)
        let q_proj = linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: load q_proj", None))?;

        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: load k_proj", None))?;

        let v_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: load v_proj", None))?;

        let o_proj = linear_no_bias(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: load o_proj", None))?;

        // Load Q/K RMSNorm layers (Gemma3-specific: normalize Q/K after projection, before RoPE)
        // Both norms operate on head_dim (256 for embeddinggemma-300m)
        let q_norm = RmsNorm::load(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;

        let k_norm = RmsNorm::load(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;

        // Create two RoPE caches for different attention types
        // Global RoPE: base=rope_theta (1000000.0) for full_attention layers
        let rope_cache_global = RotaryEmbeddingCache::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            vb.device(),
        )?;

        // Local RoPE: base=rope_local_base_freq (10000.0) for sliding_attention layers
        let rope_cache_local = RotaryEmbeddingCache::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_local_base_freq,
            vb.device(),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope_cache_global,
            rope_cache_local,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            hidden_size,
            attention_type: config
                .get_layer_type(layer_idx)
                .unwrap_or(AttentionLayerType::FullAttention),
            sliding_window: config.sliding_window,
            layer_idx,
        })
    }

    /// Forward pass through attention
    ///
    /// # Arguments
    /// - `hidden_states`: Input tensor, shape [batch, seq_len, hidden_size]
    /// - `attention_mask`: Optional padding mask, shape [batch, seq_len] (1 for valid, 0 for padding)
    ///
    /// # Returns
    /// Output tensor, shape [batch, seq_len, hidden_size]
    /// `pad_mask` is an optional `(b, 1, 1, seq)` additive padding mask (`0` for
    /// real tokens, large negative for padding). Causality and the sliding window
    /// are applied inside [`Self::compute_attention`].
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        pad_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states
            .dims3()
            .map_err(|e| from_candle_error(e, "Gemma3Attention: get hidden_states dims", None))?;

        // Step 1: Project Q, K, V
        // Q: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads * head_dim]
        // K: [batch, seq_len, hidden_size] -> [batch, seq_len, num_kv_heads * head_dim]
        // V: [batch, seq_len, hidden_size] -> [batch, seq_len, num_kv_heads * head_dim]
        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: q_proj", None))?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: k_proj", None))?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: v_proj", None))?;

        // Step 2: Reshape to multi-head format
        // Q: [batch, seq_len, num_heads, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: reshape Q", None))?;
        let k = k
            .reshape((batch_size, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: reshape K", None))?;
        let v = v
            .reshape((batch_size, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: reshape V", None))?;

        // Step 3: Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q
            .transpose(1, 2)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: transpose Q", None))?;
        let k = k
            .transpose(1, 2)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: transpose K", None))?;
        let v = v
            .transpose(1, 2)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: transpose V", None))?;

        // Step 3.5: Apply Q Norm and K Norm (Gemma3-specific)
        // This is a KEY difference from standard attention: normalize Q/K AFTER projection, BEFORE RoPE
        // Q/K shape: [batch, num_heads, seq_len, head_dim]
        // RmsNorm is applied along the last dimension (head_dim)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Step 4: Apply RoPE to Q and K
        // Generate position IDs: [0, 1, 2, ..., seq_len-1]
        let positions: Vec<u32> = (0..seq_len as u32).collect();
        let position_tensor = Tensor::from_vec(positions, (seq_len,), q.device())
            .map_err(|e| from_candle_error(e, "Gemma3Attention: create position tensor", None))?;

        // Repeat for batch: [batch, seq_len]
        let position_ids = position_tensor
            .unsqueeze(0)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: unsqueeze positions", None))?
            .repeat(&[batch_size, 1])
            .map_err(|e| from_candle_error(e, "Gemma3Attention: repeat positions", None))?;

        // Select RoPE cache based on attention type
        // Full attention: use global RoPE (base=1000000)
        // Sliding attention: use local RoPE (base=10000)
        let rope_cache = match self.attention_type {
            AttentionLayerType::FullAttention => &self.rope_cache_global,
            AttentionLayerType::SlidingAttention => &self.rope_cache_local,
        };
        let q_rope = rope_cache.apply_rotary_emb(&q, &position_ids)?;
        let k_rope = rope_cache.apply_rotary_emb(&k, &position_ids)?;

        // Step 5: Repeat K and V for MQA (1 → 3 heads)
        // K: [batch, 1, seq_len, head_dim] -> [batch, 3, seq_len, head_dim]
        // V: [batch, 1, seq_len, head_dim] -> [batch, 3, seq_len, head_dim]
        let k_repeated = k_rope
            .repeat(&[1, self.num_attention_heads, 1, 1])
            .map_err(|e| from_candle_error(e, "Gemma3Attention: repeat K for MQA", None))?;
        let v_repeated = v
            .repeat(&[1, self.num_attention_heads, 1, 1])
            .map_err(|e| from_candle_error(e, "Gemma3Attention: repeat V for MQA", None))?;

        // Step 6: Compute attention (causal; windowed on sliding layers)
        let attn_output = self.compute_attention(&q_rope, &k_repeated, &v_repeated, pad_mask)?;

        // Step 7: Reshape back to [batch, seq_len, num_heads * head_dim]
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: transpose attn output", None))?
            .reshape((
                batch_size,
                seq_len,
                self.num_attention_heads * self.head_dim,
            ))
            .map_err(|e| from_candle_error(e, "Gemma3Attention: reshape attn output", None))?;

        // Step 8: Output projection
        self.o_proj
            .forward(&attn_output)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: o_proj", None))
    }

    /// Memory-bounded attention for this layer.
    ///
    /// Every layer is causal, which `Gemma3Model::forward` used to express as a
    /// `(1, 1, seq, seq)` lower-triangular mask. A sliding layer additionally keeps
    /// only the `sliding_window` most recent keys, `[i - sliding_window + 1, i]`:
    /// the causal triangle intersected with a `±(sliding_window - 1)` band. The
    /// shared kernel applies both per query block, so neither the `(seq, seq)`
    /// window mask nor the `(b, heads, seq, seq)` score matrix is materialized.
    fn compute_attention(
        &self,
        q: &Tensor,                // [batch, num_heads, seq_len, head_dim]
        k: &Tensor,                // [batch, num_heads, seq_len, head_dim]
        v: &Tensor,                // [batch, num_heads, seq_len, head_dim]
        pad_mask: Option<&Tensor>, // [batch, 1, 1, seq_len], additive
    ) -> UnifiedResult<Tensor> {
        let window = match self.attention_type {
            AttentionLayerType::SlidingAttention => Some(self.sliding_window.saturating_sub(1)),
            AttentionLayerType::FullAttention => None,
        };
        let cfg = ChunkedSdpaConfig {
            block_size: ATTN_QUERY_BLOCK,
            window,
            causal: true,
            scale: (self.head_dim as f64).powf(-0.5),
            q_offset: 0,
        };
        chunked_sdpa(q, k, v, pad_mask, &cfg)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: chunked attention", None))
    }
}

/// Gemma3 Transformer Layer (Pre-Norm Architecture)
///
/// Architecture:
/// ```text
/// hidden_states [batch, seq_len, 768]
///   ├→ residual (save)
///   ↓
///   RmsNorm (input_layernorm)
///   ↓
///   Gemma3Attention
///   ↓
///   residual + attention_output
///   ├→ residual (save)
///   ↓
///   RmsNorm (post_attention_layernorm)
///   ↓
///   Gemma3MLP
///   ↓
///   residual + mlp_output
/// output [batch, seq_len, 768]
/// ```
#[derive(Debug)]
pub struct Gemma3Layer {
    input_layernorm: RmsNorm,
    self_attn: Gemma3Attention,
    post_attention_layernorm: RmsNorm,
    pre_feedforward_layernorm: RmsNorm, // Added: norm before MLP
    mlp: Gemma3MLP,
    post_feedforward_layernorm: RmsNorm, // Added: norm after MLP
    layer_idx: usize,                    // Layer index for debugging
}

impl Gemma3Layer {
    /// Load Gemma3Layer from VarBuilder
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights
    /// - `config`: GemmaEmbeddingConfig
    /// - `layer_idx`: Index of this layer
    pub fn load(
        vb: VarBuilder,
        config: &GemmaEmbeddingConfig,
        layer_idx: usize,
    ) -> UnifiedResult<Self> {
        let input_layernorm = RmsNorm::load(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let self_attn = Gemma3Attention::load(vb.pp("self_attn"), config, layer_idx)?;

        let post_attention_layernorm = RmsNorm::load(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let pre_feedforward_layernorm = RmsNorm::load(
            vb.pp("pre_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        let mlp = Gemma3MLP::load(vb.pp("mlp"), config)?;

        let post_feedforward_layernorm = RmsNorm::load(
            vb.pp("post_feedforward_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            mlp,
            post_feedforward_layernorm,
            layer_idx,
        })
    }

    /// Forward pass through transformer layer
    ///
    /// # Arguments
    /// - `hidden_states`: Input tensor, shape [batch, seq_len, hidden_size]
    /// - `attention_mask`: Optional padding mask, shape [batch, seq_len]
    ///
    /// # Returns
    /// Output tensor, shape [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        pad_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        // ============ Attention Block ============
        // Step 1: Save residual
        let residual = hidden_states.clone();

        // Step 2: Pre-norm (RmsNorm before attention)
        let hidden_states = self.input_layernorm.forward(hidden_states)?;

        // Step 3: Self-attention (causal and, on sliding layers, windowed inside)
        let mut hidden_states = self.self_attn.forward(&hidden_states, pad_mask)?;

        // Step 4: Post-attention LayerNorm (CRITICAL: this was missing!)
        hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;

        // Step 5: First residual connection
        let hidden_states = residual
            .add(&hidden_states)
            .map_err(|e| from_candle_error(e, "Gemma3Layer: attention residual add", None))?;

        // ============ MLP Block ============
        // Step 6: Save residual
        let residual = hidden_states.clone();

        // Step 7: Pre-feedforward norm (before MLP)
        let hidden_states = self.pre_feedforward_layernorm.forward(&hidden_states)?;

        // Step 8: MLP
        let hidden_states = self.mlp.forward(&hidden_states)?;

        // Step 9: Post-feedforward norm (after MLP)
        let hidden_states = self.post_feedforward_layernorm.forward(&hidden_states)?;

        // Step 10: Second residual connection
        let output = residual
            .add(&hidden_states)
            .map_err(|e| from_candle_error(e, "Gemma3Layer: MLP residual add", None))?;

        Ok(output)
    }
}

/// Gemma3 Model - Complete Transformer Backbone
///
/// This is the core transformer model used as the backbone for EmbeddingGemma-300M.
/// After this model, Mean Pooling and Dense Bottleneck are applied.
///
/// # Architecture
/// ```text
/// Input IDs [batch, seq_len]
///   ↓
/// Token Embeddings [batch, seq_len, hidden_size=768]
///   ↓
/// 24× Gemma3Layer (RmsNorm → Attention+Residual → RmsNorm → MLP+Residual)
///   ↓
/// Final RmsNorm
/// Output [batch, seq_len, 768]
/// ```
///
/// # Usage
/// ```ignore
/// let model = Gemma3Model::load(vb, &config)?;
/// let output = model.forward(&input_ids, &attention_mask)?;
/// // output: [batch, seq_len, 768]
/// ```
#[derive(Debug)]
pub struct Gemma3Model {
    embeddings: Embedding,
    layers: Vec<Gemma3Layer>,
    norm: RmsNorm,
    config: GemmaEmbeddingConfig,
}

impl Gemma3Model {
    /// Load Gemma3Model from VarBuilder
    ///
    /// # Arguments
    /// - `vb`: VarBuilder for loading weights
    /// - `config`: GemmaEmbeddingConfig
    pub fn load(vb: VarBuilder, config: &GemmaEmbeddingConfig) -> UnifiedResult<Self> {
        // Load token embeddings
        let embeddings =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))
                .map_err(|e| from_candle_error(e, "Gemma3Model: load embeddings", None))?;

        // Load transformer layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            let layer =
                Gemma3Layer::load(vb.pp(format!("layers.{}", layer_idx)), config, layer_idx)?;
            layers.push(layer);
        }

        // Load final norm
        let norm = RmsNorm::load(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            embeddings,
            layers,
            norm,
            config: config.clone(),
        })
    }

    /// Forward pass through Gemma3 model
    ///
    /// # Arguments
    /// - `input_ids`: Token IDs, shape [batch, seq_len]
    /// - `attention_mask`: Optional padding mask, shape [batch, seq_len] (1 for valid, 0 for padding)
    ///
    /// # Returns
    /// Hidden states, shape [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _attention_mask: Option<&Tensor>, // Reserved for future padding mask support
    ) -> UnifiedResult<Tensor> {
        // Step 1: Token embeddings with scaling
        // CRITICAL: Gemma3 uses Gemma3TextScaledWordEmbedding which scales by sqrt(hidden_size)
        // This is done inside embed_tokens.forward() in Python, we need to do it manually here
        let mut hidden_states = self
            .embeddings
            .forward(input_ids)
            .map_err(|e| from_candle_error(e, "Gemma3Model: embeddings forward", None))?;

        // Apply embedding scaling: hidden_states *= sqrt(hidden_size)
        // Python uses Gemma3TextScaledWordEmbedding which does this automatically
        let embed_scale = (self.config.hidden_size as f64).sqrt();
        hidden_states = (hidden_states * embed_scale)
            .map_err(|e| from_candle_error(e, "Gemma3Model: apply embedding scale", None))?;

        // Step 2: Pass through transformer layers. Causal masking (and the sliding
        // window) is applied per query block inside each attention layer, so no
        // `(seq, seq)` mask is built here. The padding mask stays reserved.
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states =
                layer
                    .forward(&hidden_states, None)
                    .map_err(|e| UnifiedError::Model {
                        model_type: ModelErrorType::Embedding,
                        operation: format!("Gemma3Model: layer {} forward", layer_idx),
                        context: Some(format!("Failed to process transformer layer {}", layer_idx)),
                        source: e.to_string(),
                    })?;
        }

        // Step 3: Final normalization
        let output = self.norm.forward(&hidden_states)?;

        Ok(output)
    }

    /// Get model configuration
    pub fn config(&self) -> &GemmaEmbeddingConfig {
        &self.config
    }

    /// Get model device
    pub fn device(&self) -> Device {
        self.embeddings.embeddings().device().clone()
    }
}

#[cfg(test)]
mod chunked_attention_tests {
    //! The kernel must reproduce the dense attention it replaced (issue #3382): a
    //! `(b, heads, seq, seq)` score matrix scaled by `1/sqrt(head_dim)`, a causal
    //! `-inf` triangle, on sliding layers a one-sided `-1e9` window
    //! `[i - sliding_window + 1, i]` (skipped when `seq <= sliding_window`), softmax,
    //! then `@ V` in f32.
    use super::*;

    const SLIDING_WINDOW: usize = 16;

    fn make_attention(attention_type: AttentionLayerType, device: &Device) -> Gemma3Attention {
        let (heads, head_dim, hidden) = (3usize, 8usize, 24usize);
        let lin = |out: usize, inp: usize| {
            Linear::new(
                Tensor::randn(0f32, 0.2f32, (out, inp), device).unwrap(),
                None,
            )
        };
        let norm = || RmsNorm::new(Tensor::randn(1f32, 0.1f32, head_dim, device).unwrap(), 1e-6);
        Gemma3Attention {
            q_proj: lin(heads * head_dim, hidden),
            k_proj: lin(head_dim, hidden),
            v_proj: lin(head_dim, hidden),
            o_proj: lin(hidden, heads * head_dim),
            q_norm: norm(),
            k_norm: norm(),
            rope_cache_global: RotaryEmbeddingCache::new(head_dim, 1024, 1_000_000.0, device)
                .unwrap(),
            rope_cache_local: RotaryEmbeddingCache::new(head_dim, 1024, 10_000.0, device).unwrap(),
            num_attention_heads: heads,
            num_key_value_heads: 1,
            head_dim,
            hidden_size: hidden,
            attention_type,
            sliding_window: SLIDING_WINDOW,
            layer_idx: 0,
        }
    }

    /// The pre-migration forward: same projections, norms and RoPE as
    /// `Gemma3Attention::forward`, dense attention in the middle.
    fn dense_reference(attn: &Gemma3Attention, hidden_states: &Tensor) -> Tensor {
        let (b, seq, _) = hidden_states.dims3().unwrap();
        let device = hidden_states.device();
        let heads = attn.num_attention_heads;
        let hd = attn.head_dim;
        let q = attn
            .q_proj
            .forward(hidden_states)
            .unwrap()
            .reshape((b, seq, heads, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = attn
            .k_proj
            .forward(hidden_states)
            .unwrap()
            .reshape((b, seq, 1, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = attn
            .v_proj
            .forward(hidden_states)
            .unwrap()
            .reshape((b, seq, 1, hd))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let q = attn.q_norm.forward(&q).unwrap();
        let k = attn.k_norm.forward(&k).unwrap();
        let positions = Tensor::from_vec((0..seq as u32).collect::<Vec<_>>(), (seq,), device)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .repeat(&[b, 1])
            .unwrap();
        let rope = match attn.attention_type {
            AttentionLayerType::FullAttention => &attn.rope_cache_global,
            AttentionLayerType::SlidingAttention => &attn.rope_cache_local,
        };
        let q = rope.apply_rotary_emb(&q, &positions).unwrap();
        let k = rope
            .apply_rotary_emb(&k, &positions)
            .unwrap()
            .repeat(&[1, heads, 1, 1])
            .unwrap();
        let v = v.repeat(&[1, heads, 1, 1]).unwrap();

        // Dense attention exactly as the removed code wrote it.
        let scores = (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() / (hd as f64).sqrt()).unwrap();
        let mut mask = vec![0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                if j > i {
                    mask[i * seq + j] = f32::NEG_INFINITY;
                } else if matches!(attn.attention_type, AttentionLayerType::SlidingAttention)
                    && seq > SLIDING_WINDOW
                    && j + SLIDING_WINDOW <= i
                {
                    mask[i * seq + j] = -1e9;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (1, 1, seq, seq), device).unwrap();
        let scores = scores.broadcast_add(&mask).unwrap();
        let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();
        let ctx = probs
            .matmul(&v)
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .reshape((b, seq, heads * hd))
            .unwrap();
        attn.o_proj.forward(&ctx).unwrap()
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

    #[test]
    fn test_chunked_attention_matches_dense() {
        let device = Device::Cpu;
        // Below the window, above it, and above the kernel's query block.
        for layer_type in [
            AttentionLayerType::FullAttention,
            AttentionLayerType::SlidingAttention,
        ] {
            let attn = make_attention(layer_type, &device);
            for &seq in &[3usize, 40, ATTN_QUERY_BLOCK + 88] {
                let xs = Tensor::randn(0f32, 1f32, (2, seq, attn.hidden_size), &device).unwrap();
                let got = attn.forward(&xs, None).unwrap();
                let want = dense_reference(&attn, &xs);
                let diff = max_abs_diff(&got, &want);
                assert!(diff < 1e-4, "{:?} seq={}: max|Δ|={}", layer_type, seq, diff);
            }
        }
    }
}
