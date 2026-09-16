//! Tests for traditional base model implementation

use super::base_model::*;
use crate::test_fixtures::{fixtures::*, test_utils::*};
use rstest::*;

/// Test BaseModelConfig default values
#[rstest]
fn test_base_model_base_model_config_default() {
    let config = BaseModelConfig::default();

    // Test BERT-base default values
    assert_eq!(config.vocab_size, 30522);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_hidden_layers, 12);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.max_position_embeddings, 512);
    assert_eq!(config.type_vocab_size, 2);
    assert_eq!(config.layer_norm_eps, 1e-12);

    // Test boolean flags
    assert!(config.use_position_embeddings);
    assert!(config.use_token_type_embeddings);
    assert!(config.add_pooling_layer);

    // Test enums
    assert!(matches!(config.hidden_act, ActivationFunction::Gelu));
    assert!(matches!(config.pooler_activation, ActivationFunction::Gelu));
    assert!(matches!(config.pooling_strategy, PoolingStrategy::CLS));

    println!("BaseModelConfig default values test passed");
}

/// Dense reference for `SelfAttention::forward` (issue #3382): full
/// `(b, heads, seq, seq)` scores, `-inf` where the `(b, seq)` mask is 0, softmax,
/// `@ V`, then the output projection.
fn dense_self_attention_reference(
    tensors: &std::collections::HashMap<String, candle_core::Tensor>,
    heads: usize,
    xs: &candle_core::Tensor,
    mask: &candle_core::Tensor,
) -> candle_core::Tensor {
    use candle_core::{Module, Tensor};
    let (b, seq, hidden) = xs.dims3().unwrap();
    let head_dim = hidden / heads;
    let linear = |name: &str| {
        candle_nn::Linear::new(
            tensors[&format!("{name}.weight")].clone(),
            Some(tensors[&format!("{name}.bias")].clone()),
        )
    };
    let split = |x: &Tensor| {
        x.reshape((b, seq, heads, head_dim))
            .unwrap()
            .transpose(1, 2)
            .unwrap()
            .contiguous()
            .unwrap()
    };
    let q = split(&linear("self.query").forward(xs).unwrap());
    let k = split(&linear("self.key").forward(xs).unwrap());
    let v = split(&linear("self.value").forward(xs).unwrap());
    let mut scores =
        (q.matmul(&k.transpose(2, 3).unwrap()).unwrap() / (head_dim as f64).sqrt()).unwrap();
    if mask.rank() > 0 {
        let m = mask
            .unsqueeze(1)
            .unwrap()
            .unsqueeze(2)
            .unwrap()
            .expand(scores.shape())
            .unwrap();
        let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), xs.device()).unwrap();
        scores = m
            .eq(&Tensor::zeros_like(&m).unwrap())
            .unwrap()
            .where_cond(&neg_inf, &scores)
            .unwrap();
    }
    let probs = candle_nn::ops::softmax(&scores, candle_core::D::Minus1).unwrap();
    let ctx = probs
        .matmul(&v)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .reshape((b, seq, hidden))
        .unwrap();
    linear("output.dense").forward(&ctx).unwrap()
}

/// The chunked kernel must reproduce the dense attention it replaced (issue #3382),
/// with a padded and an unmasked (rank-0 mask) batch, below and above the kernel's
/// query block.
#[rstest]
fn test_self_attention_matches_dense_reference() {
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use std::collections::HashMap;

    let device = Device::Cpu;
    let config = BaseModelConfig {
        hidden_size: 16,
        num_attention_heads: 2,
        ..Default::default()
    };
    let hidden = config.hidden_size;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in ["self.query", "self.key", "self.value", "output.dense"] {
        let w = Tensor::randn(0f32, 0.2f32, (hidden, hidden), &device).unwrap();
        let b = Tensor::randn(0f32, 0.2f32, hidden, &device).unwrap();
        tensors.insert(format!("{name}.weight"), w);
        tensors.insert(format!("{name}.bias"), b);
    }
    let vb = VarBuilder::from_tensors(tensors.clone(), DType::F32, &device);
    let attn = SelfAttention::new(&config, vb).unwrap();

    for &seq in &[3usize, 40, 600] {
        let xs = Tensor::randn(0f32, 1f32, (2, seq, hidden), &device).unwrap();
        let mut mask_data = vec![1u32; 2 * seq];
        for c in (seq - seq / 3)..seq {
            mask_data[seq + c] = 0;
        }
        let padded = Tensor::from_vec(mask_data, (2, seq), &device).unwrap();
        let no_mask = Tensor::new(1u32, &device).unwrap();
        for (label, mask) in [("padded", &padded), ("unmasked", &no_mask)] {
            let got = attn.forward(&xs, mask).unwrap();
            let want =
                dense_self_attention_reference(&tensors, config.num_attention_heads, &xs, mask);
            let diff = (got - &want)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-4, "{label} seq={seq}: max|Δ|={diff}");
        }
    }
}
