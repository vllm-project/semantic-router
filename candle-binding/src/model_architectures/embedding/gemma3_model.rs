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
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, Embedding, Linear, Module, VarBuilder};

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a causal attention mask (lower triangular)
///
/// # Arguments
/// - `seq_len`: Sequence length
/// - `device`: Device to create the mask on
///
/// # Returns
/// Causal mask tensor, shape [1, 1, seq_len, seq_len]
/// - 0.0 for positions that can attend
/// - -inf for positions that should be masked
///
/// Example for seq_len=4:
/// ```
/// [[0,  -inf, -inf, -inf],
///  [0,   0,   -inf, -inf],
///  [0,   0,    0,   -inf],
///  [0,   0,    0,    0  ]]
/// ```
fn create_causal_mask(seq_len: usize, device: &Device) -> UnifiedResult<Tensor> {
    // Create a lower triangular matrix filled with 0s
    let mut mask_data = vec![0.0f32; seq_len * seq_len];

    // Fill upper triangle with -inf
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    // Create tensor [seq_len, seq_len]
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)
        .map_err(|e| from_candle_error(e, "create_causal_mask: create tensor", None))?;

    // Reshape to [1, 1, seq_len, seq_len] for broadcasting
    mask.unsqueeze(0)
        .and_then(|t| t.unsqueeze(0))
        .map_err(|e| from_candle_error(e, "create_causal_mask: unsqueeze", None))
}

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
///
/// # Precision
/// Uses f64 for critical calculations to match Python implementation.
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
        // Using f64 precision for RMS normalization (same as Qwen3)
        // This achieves >0.99 cosine similarity with Python reference

        // Step 1: Convert input to f64
        let x_f64 = x
            .to_dtype(DType::F64)
            .map_err(|e| from_candle_error(e, "RmsNorm: x to f64", None))?;

        // Step 2: Square the input
        let x_squared = x_f64
            .sqr()
            .map_err(|e| from_candle_error(e, "RmsNorm: compute x^2", None))?;

        // Step 3: Compute mean along last dimension, keeping dimension
        let mean_squared = x_squared
            .mean_keepdim(candle_core::D::Minus1)
            .map_err(|e| from_candle_error(e, "RmsNorm: compute mean(x^2)", None))?;

        // Step 4: Add epsilon and take square root
        let mean_plus_eps = (mean_squared + self.eps)
            .map_err(|e| from_candle_error(e, "RmsNorm: add epsilon", None))?;
        let rms = mean_plus_eps
            .sqrt()
            .map_err(|e| from_candle_error(e, "RmsNorm: compute sqrt", None))?;

        // Step 5: Normalize by dividing by RMS
        let normalized_f64 = x_f64
            .broadcast_div(&rms)
            .map_err(|e| from_candle_error(e, "RmsNorm: normalize (x / rms)", None))?;

        // Step 6: Convert weight to f64 and apply Gemma3-specific scaling
        // CRITICAL: Gemma3 uses (1.0 + weight) instead of just weight!
        // See: https://github.com/huggingface/transformers/pull/29402
        // output = normalized * (1.0 + weight)
        let weight_f64 = self
            .weight
            .to_dtype(DType::F64)
            .map_err(|e| from_candle_error(e, "RmsNorm: weight to f64", None))?;
        let one_plus_weight =
            (weight_f64 + 1.0).map_err(|e| from_candle_error(e, "RmsNorm: 1.0 + weight", None))?;
        let output_f64 = normalized_f64
            .broadcast_mul(&one_plus_weight)
            .map_err(|e| from_candle_error(e, "RmsNorm: scale by (1.0 + weight)", None))?;

        // Step 7: Convert back to f32 for subsequent layers
        output_f64
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "RmsNorm: output to f32", None))
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
        if head_dim % 2 != 0 {
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
// Helper Functions for F64 Precision
// ============================================================================

/// Helper function to perform Linear forward with f64 precision
///
/// This function temporarily converts Linear weights to f64 for computation,
/// which helps reduce floating-point accumulation errors in deep networks.
///
/// # Arguments
/// - `linear`: The Linear layer
/// - `x`: Input tensor (should be f64)
///
/// # Returns
/// Output tensor in f64 precision
fn linear_forward_f64(linear: &Linear, x: &Tensor) -> UnifiedResult<Tensor> {
    // Convert weight to f64
    let weight_f64 = linear
        .weight()
        .to_dtype(DType::F64)
        .map_err(|e| from_candle_error(e, "linear_forward_f64: convert weight to f64", None))?;

    // Transpose weight for matmul
    let weight_t = weight_f64
        .t()
        .map_err(|e| from_candle_error(e, "linear_forward_f64: transpose weight", None))?;

    // Compute: x @ weight^T using broadcast_matmul for proper 3D @ 2D handling
    let output = x
        .broadcast_matmul(&weight_t)
        .map_err(|e| from_candle_error(e, "linear_forward_f64: broadcast_matmul", None))?;

    // Add bias if present
    if let Some(bias) = linear.bias() {
        let bias_f64 = bias
            .to_dtype(DType::F64)
            .map_err(|e| from_candle_error(e, "linear_forward_f64: convert bias to f64", None))?;
        output
            .broadcast_add(&bias_f64)
            .map_err(|e| from_candle_error(e, "linear_forward_f64: add bias", None))
    } else {
        Ok(output)
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

    /// Forward pass through MLP (using f64 precision to reduce accumulation error)
    ///
    /// # Arguments
    /// - `x`: Input tensor, shape [batch, seq_len, hidden_size]
    ///
    /// # Returns
    /// Output tensor, shape [batch, seq_len, hidden_size]
    pub fn forward(&self, x: &Tensor) -> UnifiedResult<Tensor> {
        // Convert input to f64 for higher precision
        let x_f64 = x
            .to_dtype(DType::F64)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: convert input to f64", None))?;

        // Step 1: gate_proj: [batch, seq_len, 768] -> [batch, seq_len, 1152] (f64)
        let gate_output = linear_forward_f64(&self.gate_proj, &x_f64)?;

        // Step 2: gelu_pytorch_tanh activation on gate_output
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        let gate_activated = Self::gelu_pytorch_tanh(&gate_output)?;

        // Step 3: up_proj: [batch, seq_len, 768] -> [batch, seq_len, 1152] (f64)
        let up_output = linear_forward_f64(&self.up_proj, &x_f64)?;

        // Step 4: Element-wise multiplication (GeGLU gating)
        let gated = gate_activated
            .mul(&up_output)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: gate * up", None))?;

        // Step 5: down_proj: [batch, seq_len, 1152] -> [batch, seq_len, 768] (f64)
        let output_f64 = linear_forward_f64(&self.down_proj, &gated)?;

        // Convert back to f32 for subsequent layers
        output_f64
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "Gemma3MLP: convert output to f32", None))
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
            &vb.device(),
        )?;

        // Local RoPE: base=rope_local_base_freq (10000.0) for sliding_attention layers
        let rope_cache_local = RotaryEmbeddingCache::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_local_base_freq,
            &vb.device(),
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

    /// Forward pass through attention (using f64 precision to reduce accumulation error)
    ///
    /// # Arguments
    /// - `hidden_states`: Input tensor, shape [batch, seq_len, hidden_size]
    /// - `attention_mask`: Optional padding mask, shape [batch, seq_len] (1 for valid, 0 for padding)
    ///
    /// # Returns
    /// Output tensor, shape [batch, seq_len, hidden_size]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states
            .dims3()
            .map_err(|e| from_candle_error(e, "Gemma3Attention: get hidden_states dims", None))?;

        // Convert input to f64 for higher precision
        let hidden_states_f64 = hidden_states
            .to_dtype(DType::F64)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: convert input to f64", None))?;

        // Step 1: Project Q, K, V (in f64 precision)
        // Q: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads * head_dim]
        // K: [batch, seq_len, hidden_size] -> [batch, seq_len, num_kv_heads * head_dim]
        // V: [batch, seq_len, hidden_size] -> [batch, seq_len, num_kv_heads * head_dim]
        let q = linear_forward_f64(&self.q_proj, &hidden_states_f64)?;
        let k = linear_forward_f64(&self.k_proj, &hidden_states_f64)?;
        let v = linear_forward_f64(&self.v_proj, &hidden_states_f64)?;

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

        // Step 6: Compute attention based on attention type
        let attn_output = match self.attention_type {
            AttentionLayerType::SlidingAttention => {
                self.compute_sliding_attention(&q_rope, &k_repeated, &v_repeated, attention_mask)?
            }
            AttentionLayerType::FullAttention => {
                self.compute_full_attention(&q_rope, &k_repeated, &v_repeated, attention_mask)?
            }
        };

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

        // Step 8: Output projection (in f64) - convert attn_output to f64 first
        let attn_output_f64 = attn_output.to_dtype(DType::F64).map_err(|e| {
            from_candle_error(
                e,
                "Gemma3Attention: convert attn_output to f64 for o_proj",
                None,
            )
        })?;
        let output_f64 = linear_forward_f64(&self.o_proj, &attn_output_f64)?;

        // Convert back to f32 for subsequent layers
        let output = output_f64
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "Gemma3Attention: convert output to f32", None))?;

        Ok(output)
    }

    /// Compute full (global) attention
    fn compute_full_attention(
        &self,
        q: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        k: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        v: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        attention_mask: Option<&Tensor>, // [batch, seq_len]
    ) -> UnifiedResult<Tensor> {
        // Standard scaled dot-product attention
        // scores = (Q @ K^T) / sqrt(head_dim)
        // attn = softmax(scores) @ V
        let scale = (self.head_dim as f64).sqrt();

        // Q @ K^T: [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        //       -> [batch, num_heads, seq_len, seq_len]
        let k_t = k
            .transpose(2, 3)
            .map_err(|e| from_candle_error(e, "FullAttention: transpose K", None))?;
        let attn_scores = q
            .matmul(&k_t)
            .map_err(|e| from_candle_error(e, "FullAttention: Q @ K^T", None))?;

        // Scale by 1/sqrt(head_dim) (standard attention scaling)
        let attn_scores = (attn_scores / scale)
            .map_err(|e| from_candle_error(e, "FullAttention: scale scores", None))?;

        // Apply causal mask (attention_mask is now [1, 1, seq_len, seq_len] causal mask)
        // Mask values: 0 for allowed positions, -inf for masked positions
        // Add mask to scores: allowed positions remain unchanged, masked positions become -inf
        let attn_scores = if let Some(mask) = attention_mask {
            // Broadcasting: [batch, num_heads, seq_len, seq_len] + [1, 1, seq_len, seq_len]
            attn_scores
                .broadcast_add(mask)
                .map_err(|e| from_candle_error(e, "FullAttention: apply causal mask", None))?
        } else {
            attn_scores
        };
        // Softmax over last dimension
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)
            .map_err(|e| from_candle_error(e, "FullAttention: softmax", None))?;

        // attn_weights @ V: [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        //                -> [batch, num_heads, seq_len, head_dim]
        // Note: Convert V to F32 to match attn_weights dtype
        let v_f32 = v
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "FullAttention: convert V to F32", None))?;
        let output = attn_weights
            .matmul(&v_f32)
            .map_err(|e| from_candle_error(e, "FullAttention: attn @ V", None))?;

        Ok(output)
    }

    /// Compute sliding window attention
    fn compute_sliding_attention(
        &self,
        q: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        k: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        v: &Tensor,                      // [batch, num_heads, seq_len, head_dim]
        attention_mask: Option<&Tensor>, // [batch, seq_len]
    ) -> UnifiedResult<Tensor> {
        // Sliding window attention with window size = sliding_window
        // Each token can only attend to tokens within the window
        // Implementation: Uses sliding window mask for efficient computation

        // If sequence length <= window size, use full attention
        let seq_len = q
            .dim(2)
            .map_err(|e| from_candle_error(e, "SlidingAttention: get seq_len", None))?;

        if seq_len <= self.sliding_window {
            return self.compute_full_attention(q, k, v, attention_mask);
        }

        // Otherwise, apply sliding window mask
        // Create sliding window mask: each position can attend to [pos - window, pos]
        let window_mask = self.create_sliding_window_mask(seq_len, q.device())?;

        // Compute attention with window mask
        let scale = (self.head_dim as f64).sqrt();

        let k_t = k
            .transpose(2, 3)
            .map_err(|e| from_candle_error(e, "SlidingAttention: transpose K", None))?;
        let mut attn_scores = q
            .matmul(&k_t)
            .map_err(|e| from_candle_error(e, "SlidingAttention: Q @ K^T", None))?;

        // Scale
        attn_scores = (attn_scores / scale)
            .map_err(|e| from_candle_error(e, "SlidingAttention: scale scores", None))?;

        // Apply window mask
        attn_scores = attn_scores
            .broadcast_add(&window_mask)
            .map_err(|e| from_candle_error(e, "SlidingAttention: apply window mask", None))?;

        // Apply causal mask if provided (attention_mask is now [1, 1, seq_len, seq_len] causal mask)
        if let Some(mask) = attention_mask {
            attn_scores = attn_scores
                .broadcast_add(mask)
                .map_err(|e| from_candle_error(e, "SlidingAttention: apply causal mask", None))?;
        }

        // Softmax
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_scores)
            .map_err(|e| from_candle_error(e, "SlidingAttention: softmax", None))?;

        // attn @ V (convert V to F32 to match attn_weights dtype)
        let v_f32 = v
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "SlidingAttention: convert V to F32", None))?;
        attn_weights
            .matmul(&v_f32)
            .map_err(|e| from_candle_error(e, "SlidingAttention: attn @ V", None))
    }

    /// Create sliding window mask
    ///
    /// Returns a mask of shape [1, 1, seq_len, seq_len] where:
    /// - 0.0 for positions within the window
    /// - -1e9 for positions outside the window (avoid -inf to prevent NaN)
    fn create_sliding_window_mask(&self, seq_len: usize, device: &Device) -> UnifiedResult<Tensor> {
        const LARGE_NEGATIVE: f32 = -1e9;
        let mut mask_data = vec![LARGE_NEGATIVE; seq_len * seq_len];

        for i in 0..seq_len {
            let window_start = if i >= self.sliding_window {
                i - self.sliding_window + 1
            } else {
                0
            };
            let window_end = i + 1; // Inclusive of current position

            for j in window_start..window_end {
                mask_data[i * seq_len + j] = 0.0;
            }
        }

        let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)
            .map_err(|e| from_candle_error(e, "create_sliding_window_mask: from_vec", None))?;

        // Unsqueeze to [1, 1, seq_len, seq_len]
        mask.unsqueeze(0)
            .map_err(|e| from_candle_error(e, "create_sliding_window_mask: unsqueeze 0", None))?
            .unsqueeze(0)
            .map_err(|e| from_candle_error(e, "create_sliding_window_mask: unsqueeze 1", None))
    }

    /// Apply padding mask to attention scores
    ///
    /// # Arguments
    /// - `attn_scores`: Attention scores, shape [batch, num_heads, seq_len, seq_len]
    /// - `attention_mask`: Padding mask, shape [batch, seq_len] (1 for valid, 0 for padding)
    ///
    /// # Returns
    /// Masked attention scores with -inf for padded positions
    fn apply_padding_mask(
        &self,
        attn_scores: &Tensor,
        attention_mask: &Tensor,
    ) -> UnifiedResult<Tensor> {
        // attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        let mask = attention_mask
            .unsqueeze(1)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: unsqueeze 1", None))?
            .unsqueeze(1)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: unsqueeze 2", None))?;

        // Convert mask: 1 -> 0.0, 0 -> -inf
        // IMPORTANT: Avoid 0 * -inf = NaN!
        // Strategy: (1 - mask) * -1e9 where -1e9 is a large negative number (not -inf)
        let mask_f32 = mask
            .to_dtype(DType::F32)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: mask to f32", None))?;

        // (1 - mask): gives 1 for padding (0), 0 for valid (1)
        let one_tensor = Tensor::ones_like(&mask_f32)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: create ones", None))?;
        let inverted_mask = one_tensor
            .sub(&mask_f32)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: 1 - mask", None))?;

        // Use a large negative number instead of -inf to avoid NaN
        // -1e9 is effectively -inf for softmax but avoids 0 * -inf = NaN
        const LARGE_NEGATIVE: f64 = -1e9;
        let neg_mask = (inverted_mask * LARGE_NEGATIVE).map_err(|e| {
            from_candle_error(e, "apply_padding_mask: multiply large negative", None)
        })?;

        // Add to attention scores
        attn_scores
            .broadcast_add(&neg_mask)
            .map_err(|e| from_candle_error(e, "apply_padding_mask: add to scores", None))
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
        attention_mask: Option<&Tensor>,
    ) -> UnifiedResult<Tensor> {
        // ============ Attention Block ============
        // Step 1: Save residual
        let residual = hidden_states.clone();

        // Step 2: Pre-norm (RmsNorm before attention)
        let hidden_states = self.input_layernorm.forward(hidden_states)?;

        // Step 3: Self-attention
        let mut hidden_states = self.self_attn.forward(&hidden_states, attention_mask)?;

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
                Gemma3Layer::load(vb.pp(&format!("layers.{}", layer_idx)), config, layer_idx)?;
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

        // Step 1.5: Create causal attention mask
        // CRITICAL: Gemma3 uses causal attention (lower triangular mask)
        // Each token can only attend to itself and previous tokens
        let seq_len = hidden_states
            .dim(1)
            .map_err(|e| from_candle_error(e, "Gemma3Model: get seq_len", None))?;
        let causal_mask = create_causal_mask(seq_len, hidden_states.device())?;

        // Step 2: Pass through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(&hidden_states, Some(&causal_mask))
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
