//! Unified Pooling Implementations for Embedding Models
//!
//! This module provides pooling functions to aggregate token-level representations
//! into sentence-level embeddings.
//!
//! ## Supported Pooling Methods
//! - **Mean Pooling**: Average all token embeddings (weighted by attention mask)
//!   - Used by: GemmaEmbedding, BERT
//!   - Best for: General-purpose embeddings
//!
//! - **Last Token Pooling**: Use the last valid token's embedding
//!   - Used by: Qwen3-Embedding
//!   - Best for: Causal language models, instruction-following
//!
//! - **CLS Pooling**: Use the first token ([CLS]) embedding
//!   - Used by: Original BERT, some fine-tuned models
//!   - Best for: Models trained with CLS token supervision
//!
//! ## References
//! - Qwen3 Official: https://github.com/qwenlm/qwen3-embedding
//! - TEI Implementation: backends/candle/src/models/qwen3.rs
//! - GemmaEmbedding: https://huggingface.co/google/embeddinggemma-300m

use anyhow::Result;
use candle_core::{IndexOp, Tensor};

/// Mean pooling implementation
///
/// Averages all token embeddings weighted by the attention mask.
///
/// ## Algorithm
/// 1. Expand attention_mask: [batch, seq_len] -> [batch, seq_len, hidden]
/// 2. Apply mask: masked_hidden = hidden_states * mask_expanded
/// 3. Sum over sequence: sum_hidden = sum(masked_hidden, dim=1)
/// 4. Count valid tokens: sum_mask = sum(mask_expanded, dim=1)
/// 5. Average: embeddings = sum_hidden / sum_mask
///
/// ## Arguments
/// - `hidden_states`: Token representations `[batch_size, seq_len, hidden_size]`
/// - `attention_mask`: Valid token mask `[batch_size, seq_len]`, dtype: F32
///
/// ## Return
/// - `Ok(Tensor)`: Sentence embeddings `[batch_size, hidden_size]`
/// - `Err`: If tensor operations fail or dimensions mismatch
///
/// ## Example
/// ```rust,ignore
/// let hidden = Tensor::randn(0f32, 1., (2, 10, 768), &device)?;
/// let mask = Tensor::ones((2, 10), DType::F32, &device)?;
/// let embeddings = mean_pool(&hidden, &mask)?;
/// assert_eq!(embeddings.dims(), &[2, 768]);
/// ```
///
/// ## References
/// - TEI implementation: backends/candle/src/models/mod.rs
/// - Official GemmaEmbedding: uses mean pooling
pub fn mean_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Algorithm:
    // 1. Expand attention_mask: [batch, seq_len] -> [batch, seq_len, hidden]
    // 2. Apply mask: masked_hidden = hidden_states * mask_expanded
    // 3. Sum over sequence: sum_hidden = sum(masked_hidden, dim=1)
    // 4. Count valid tokens: sum_mask = sum(mask_expanded, dim=1)
    // 5. Average: embeddings = sum_hidden / sum_mask

    // Step 1: Expand attention_mask to match hidden_states dimensions
    let mask_expanded = attention_mask
        .unsqueeze(2)? // [batch, seq_len, 1]
        .expand(hidden_states.dims())? // [batch, seq_len, hidden]
        .to_dtype(hidden_states.dtype())?; // Match dtype

    // Step 2: Apply mask to hidden states
    let masked_hidden = hidden_states.mul(&mask_expanded)?;

    // Step 3: Sum over sequence dimension (dim=1)
    let sum_hidden = masked_hidden.sum(1)?; // [batch, hidden]

    // Step 4: Count valid tokens
    let sum_mask = mask_expanded.sum(1)?; // [batch, hidden]

    // Step 5: Average (handle division by zero gracefully)
    // Note: sum_mask should never be zero if attention_mask is valid
    let embeddings = sum_hidden.div(&sum_mask)?;

    Ok(embeddings)
}

/// Last token pooling implementation
///
/// Extracts the embedding of the last valid token for each sequence.
///
/// ## Algorithm
/// 1. Calculate sequence lengths: lengths = sum(attention_mask, dim=1) - 1
/// 2. For each batch: gather hidden_states[batch_idx, lengths[batch_idx], :]
/// 3. Stack all batch embeddings
///
/// ## Arguments
/// - `hidden_states`: Token representations `[batch_size, seq_len, hidden_size]`
/// - `attention_mask`: Valid token mask `[batch_size, seq_len]`, dtype: F32
///
/// ## Return
/// - `Ok(Tensor)`: Sentence embeddings `[batch_size, hidden_size]`
/// - `Err`: If tensor operations fail or sequence length is 0
///
/// ## Example
/// ```rust,ignore
/// let hidden = Tensor::randn(0f32, 1., (2, 10, 768), &device)?;
/// // First sequence: 5 valid tokens, second: 8 valid tokens
/// let mask = Tensor::new(
///     &[[1f32, 1., 1., 1., 1., 0., 0., 0., 0., 0.],
///       [1f32, 1., 1., 1., 1., 1., 1., 1., 0., 0.]],
///     &device
/// )?;
/// let embeddings = last_token_pool(&hidden, &mask)?;
/// assert_eq!(embeddings.dims(), &[2, 768]);
/// ```
///
/// ## References
/// - Qwen3 Official: https://github.com/qwenlm/qwen3-embedding
/// - TEI Qwen3: backends/candle/src/models/qwen3.rs (last_token_pool)
pub fn last_token_pool(hidden_states: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // Algorithm (following official Qwen3-Embedding implementation):
    // 1. Check if left padding: attention_mask[:, -1].sum() == batch_size
    // 2. If left padding: return hidden_states[:, -1]
    // 3. If right padding: calculate lengths and gather accordingly
    //
    // Reference: https://github.com/qwenlm/qwen3-embedding (last_token_pool)

    let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;

    // Step 1: Check if left padding
    // left_padding = (attention_mask[:, -1].sum() == batch_size)
    let last_col_mask = attention_mask.narrow(1, seq_len - 1, 1)?; // [batch, 1]
    let last_col_mask_f32 = last_col_mask.to_dtype(candle_core::DType::F32)?;
    let last_col_sum = last_col_mask_f32.sum_all()?.to_scalar::<f32>()?;
    let is_left_padding = (last_col_sum as usize) == batch_size;

    if is_left_padding {
        // Step 2a: For left padding, directly return the last token position
        // hidden_states[:, -1, :] in Python notation
        let last_token_embeddings = hidden_states
            .narrow(1, seq_len - 1, 1)? // [batch, 1, hidden]
            .squeeze(1)?; // [batch, hidden]
        Ok(last_token_embeddings)
    } else {
        // Step 2b: For right padding, calculate sequence lengths and gather
        // sequence_lengths = attention_mask.sum(dim=1) - 1
        let sequence_lengths = attention_mask
            .sum(1)? // [batch_size] (no keepdim)
            .to_dtype(candle_core::DType::U32)? // Convert to U32 for indexing
            .to_vec1::<u32>()? // Extract to Vec
            .iter()
            .map(|&len| {
                // Handle edge case: if length is 0, use 0 instead of underflow
                if len > 0 {
                    (len - 1) as usize
                } else {
                    0
                }
            })
            .collect::<Vec<_>>();

        // Step 3: Extract the last valid token for each batch
        // Python equivalent: last_hidden_states[torch.arange(batch_size), sequence_lengths]
        let mut embeddings = Vec::new();
        for (batch_idx, &seq_idx) in sequence_lengths.iter().enumerate() {
            let embedding = hidden_states
                .i((batch_idx, seq_idx))? // [hidden_size]
                .unsqueeze(0)?; // [1, hidden_size]
            embeddings.push(embedding);
        }

        // Step 4: Concatenate all batch embeddings: [batch_size, hidden_size]
        Ok(Tensor::cat(&embeddings, 0)?)
    }
}

/// CLS token pooling implementation
///
/// Extracts the first token ([CLS]) embedding for each sequence.
///
/// ## Algorithm
/// 1. Simply return hidden_states[:, 0, :]
///
/// ## Arguments
/// - `hidden_states`: Token representations `[batch_size, seq_len, hidden_size]`
///
/// ## Return
/// - `Ok(Tensor)`: Sentence embeddings `[batch_size, hidden_size]`
/// - `Err`: If tensor operations fail
///
/// ## Example
/// ```rust,ignore
/// let hidden = Tensor::randn(0f32, 1., (2, 10, 768), &device)?;
/// let embeddings = cls_pool(&hidden)?;
/// assert_eq!(embeddings.dims(), &[2, 768]);
/// ```
///
/// ## Note
/// This method does not use attention_mask since it only selects the first token.
pub fn cls_pool(hidden_states: &Tensor) -> Result<Tensor> {
    // Algorithm:
    // Simply extract the first token ([CLS]) for each batch
    // hidden_states[:, 0, :] in Python notation

    // Extract first token: [batch_size, 0, :] -> [batch_size, hidden_size]
    // Using narrow to select index 0 along dimension 1 (sequence dimension)
    let cls_embeddings = hidden_states
        .narrow(1, 0, 1)? // [batch_size, 1, hidden_size]
        .squeeze(1)?; // [batch_size, hidden_size]

    Ok(cls_embeddings)
}

// Tests are in pooling_test.rs (following project convention)
// Run tests with: cargo test --lib pooling
// Run performance tests with: cargo test --lib pooling -- --ignored
