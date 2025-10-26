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
