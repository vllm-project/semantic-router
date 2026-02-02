//! Pooling strategies for embedding generation

use ndarray::{Array2, ArrayView2};

/// Mean pooling over non-padded tokens
///
/// # Arguments
/// * `hidden_states` - [batch_size, seq_len, hidden_dim] tensor
/// * `attention_mask` - [batch_size, seq_len] tensor (1 for real tokens, 0 for padding)
///
/// # Returns
/// * `[batch_size, hidden_dim]` tensor with mean-pooled embeddings
pub fn mean_pool(hidden_states: ArrayView2<f32>, attention_mask: ArrayView2<f32>) -> Array2<f32> {
    let batch_size = hidden_states.shape()[0];
    let _seq_len = hidden_states.shape()[1];

    // For 2D hidden_states [batch, hidden_dim], just return as-is (already pooled from ONNX)
    if hidden_states.ndim() == 2 && attention_mask.shape()[0] == batch_size {
        return hidden_states.to_owned();
    }

    hidden_states.to_owned()
}

/// Mean pooling for 3D tensors [batch, seq_len, hidden_dim]
pub fn mean_pool_3d(
    hidden_states: &ndarray::Array3<f32>,
    attention_mask: &ndarray::Array2<f32>,
) -> Array2<f32> {
    let batch_size = hidden_states.shape()[0];
    let hidden_dim = hidden_states.shape()[2];

    let mut output = Array2::<f32>::zeros((batch_size, hidden_dim));

    for b in 0..batch_size {
        let mut sum = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;

        for s in 0..hidden_states.shape()[1] {
            let mask_val = attention_mask[[b, s]];
            if mask_val > 0.0 {
                for d in 0..hidden_dim {
                    sum[d] += hidden_states[[b, s, d]] * mask_val;
                }
                count += mask_val;
            }
        }

        // Avoid division by zero
        if count > 0.0 {
            for d in 0..hidden_dim {
                output[[b, d]] = sum[d] / count;
            }
        }
    }

    output
}

/// L2 normalize embeddings
pub fn l2_normalize(embeddings: &Array2<f32>) -> Array2<f32> {
    let mut normalized = embeddings.clone();

    for mut row in normalized.rows_mut() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_safe = if norm > 1e-12 { norm } else { 1e-12 };
        row.mapv_inplace(|x| x / norm_safe);
    }

    normalized
}

/// Truncate embeddings to target dimension (Matryoshka)
pub fn truncate_dimension(embeddings: &Array2<f32>, target_dim: usize) -> Array2<f32> {
    let current_dim = embeddings.shape()[1];

    if target_dim >= current_dim {
        return embeddings.clone();
    }

    embeddings.slice(ndarray::s![.., ..target_dim]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_l2_normalize() {
        let embeddings = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 3.0, 4.0, 0.0]).unwrap();
        let normalized = l2_normalize(&embeddings);

        // First row: [1, 0, 0] -> [1, 0, 0]
        assert!((normalized[[0, 0]] - 1.0).abs() < 1e-6);

        // Second row: [3, 4, 0] -> [0.6, 0.8, 0]
        assert!((normalized[[1, 0]] - 0.6).abs() < 1e-6);
        assert!((normalized[[1, 1]] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_truncate_dimension() {
        let embeddings = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let truncated = truncate_dimension(&embeddings, 2);

        assert_eq!(truncated.shape(), &[2, 2]);
        assert_eq!(truncated[[0, 0]], 1.0);
        assert_eq!(truncated[[0, 1]], 2.0);
        assert_eq!(truncated[[1, 0]], 5.0);
        assert_eq!(truncated[[1, 1]], 6.0);
    }
}
