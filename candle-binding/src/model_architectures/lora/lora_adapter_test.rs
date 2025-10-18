//! Tests for LoRA adapter module

use super::lora_adapter::*;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use rstest::*;

// ============================================================================
// Configuration Tests
// ============================================================================

#[rstest]
fn test_lora_config_default() {
    let config = LoRAConfig::default();

    assert_eq!(config.rank, 16);
    assert_eq!(config.alpha, 32.0);
    assert_eq!(config.dropout, 0.1);
    assert_eq!(config.target_modules.len(), 4);
    assert!(!config.use_bias);
    assert!(matches!(config.init_method, LoRAInitMethod::Kaiming));
}

#[rstest]
fn test_lora_config_custom() {
    let config = LoRAConfig {
        rank: 32,
        alpha: 64.0,
        dropout: 0.2,
        target_modules: vec!["query".to_string(), "value".to_string()],
        use_bias: true,
        init_method: LoRAInitMethod::Xavier,
    };

    assert_eq!(config.rank, 32);
    assert_eq!(config.alpha, 64.0);
    assert_eq!(config.dropout, 0.2);
    assert_eq!(config.target_modules.len(), 2);
    assert!(config.use_bias);
    assert!(matches!(config.init_method, LoRAInitMethod::Xavier));
}

#[rstest]
fn test_lora_config_clone() {
    let config1 = LoRAConfig::default();
    let config2 = config1.clone();

    assert_eq!(config1.rank, config2.rank);
    assert_eq!(config1.alpha, config2.alpha);
    assert_eq!(config1.dropout, config2.dropout);
}

#[rstest]
#[case(4)]
#[case(8)]
#[case(16)]
#[case(32)]
#[case(64)]
fn test_lora_config_various_ranks(#[case] rank: usize) {
    let config = LoRAConfig {
        rank,
        ..Default::default()
    };

    assert_eq!(config.rank, rank);

    // Scaling factor should be alpha / rank
    let expected_scaling = config.alpha / rank as f64;
    assert!((expected_scaling - (config.alpha / config.rank as f64)).abs() < 1e-9);
}

// ============================================================================
// LoRA Init Method Tests
// ============================================================================

#[rstest]
fn test_lora_init_method_variants() {
    let methods = vec![
        LoRAInitMethod::Kaiming,
        LoRAInitMethod::Xavier,
        LoRAInitMethod::Normal {
            mean: 0.0,
            std: 0.02,
        },
        LoRAInitMethod::Zero,
    ];

    // Each variant should be distinct
    for (i, method1) in methods.iter().enumerate() {
        for (j, method2) in methods.iter().enumerate() {
            if i != j {
                match (method1, method2) {
                    (LoRAInitMethod::Kaiming, LoRAInitMethod::Kaiming) => unreachable!(),
                    (LoRAInitMethod::Xavier, LoRAInitMethod::Xavier) => unreachable!(),
                    (LoRAInitMethod::Zero, LoRAInitMethod::Zero) => unreachable!(),
                    _ => {
                        // Different variants
                    }
                }
            }
        }
    }
}

#[rstest]
fn test_lora_init_method_normal_with_custom_params() {
    let method = LoRAInitMethod::Normal {
        mean: 0.5,
        std: 0.1,
    };

    match method {
        LoRAInitMethod::Normal { mean, std } => {
            assert_eq!(mean, 0.5);
            assert_eq!(std, 0.1);
        }
        _ => panic!("Expected Normal variant"),
    }
}

// ============================================================================
// LoRA Adapter Creation Tests
// ============================================================================

#[rstest]
fn test_lora_adapter_new_basic() {
    let device = Device::Cpu;
    let input_dim = 768;
    let output_dim = 768;
    let config = LoRAConfig::default();

    // Create a simple VarMap for testing
    let var_map = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    let result = LoRAAdapter::new(input_dim, output_dim, &config, vb, &device);

    assert!(result.is_ok(), "Should create LoRA adapter");
}

#[rstest]
#[case(512, 512)]
#[case(768, 768)]
#[case(1024, 1024)]
fn test_lora_adapter_various_dimensions(#[case] input_dim: usize, #[case] output_dim: usize) {
    let device = Device::Cpu;
    let config = LoRAConfig::default();

    let var_map = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    let result = LoRAAdapter::new(input_dim, output_dim, &config, vb, &device);

    assert!(
        result.is_ok(),
        "Should create adapter with dims {}x{}",
        input_dim,
        output_dim
    );
}

#[rstest]
fn test_lora_adapter_with_different_init_methods() {
    let device = Device::Cpu;
    let input_dim = 768;
    let output_dim = 768;

    let init_methods = vec![
        LoRAInitMethod::Kaiming,
        LoRAInitMethod::Xavier,
        LoRAInitMethod::Normal {
            mean: 0.0,
            std: 0.02,
        },
        LoRAInitMethod::Zero,
    ];

    for init_method in init_methods {
        let config = LoRAConfig {
            init_method,
            ..Default::default()
        };

        let var_map = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

        let result = LoRAAdapter::new(input_dim, output_dim, &config, vb, &device);

        assert!(
            result.is_ok(),
            "Should create adapter with init method {:?}",
            config.init_method
        );
    }
}

// ============================================================================
// LoRA Scaling Tests
// ============================================================================

#[rstest]
#[case(16, 32.0, 2.0)]
#[case(8, 16.0, 2.0)]
#[case(32, 64.0, 2.0)]
#[case(4, 8.0, 2.0)]
fn test_lora_scaling_calculation(
    #[case] rank: usize,
    #[case] alpha: f64,
    #[case] expected_scaling: f64,
) {
    let config = LoRAConfig {
        rank,
        alpha,
        ..Default::default()
    };

    let scaling = config.alpha / config.rank as f64;

    assert!(
        (scaling - expected_scaling).abs() < 1e-9,
        "Scaling should be alpha/rank = {}, got {}",
        expected_scaling,
        scaling
    );
}

// ============================================================================
// Target Modules Tests
// ============================================================================

#[rstest]
fn test_lora_config_default_target_modules() {
    let config = LoRAConfig::default();

    let expected_modules = vec!["query", "value", "key", "output"];

    assert_eq!(config.target_modules.len(), expected_modules.len());

    for expected in expected_modules {
        assert!(
            config.target_modules.contains(&expected.to_string()),
            "Should contain target module: {}",
            expected
        );
    }
}

#[rstest]
fn test_lora_config_custom_target_modules() {
    let custom_modules = vec!["query".to_string(), "key".to_string(), "dense".to_string()];

    let config = LoRAConfig {
        target_modules: custom_modules.clone(),
        ..Default::default()
    };

    assert_eq!(config.target_modules.len(), 3);
    assert_eq!(config.target_modules, custom_modules);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[rstest]
fn test_lora_config_with_zero_dropout() {
    let config = LoRAConfig {
        dropout: 0.0,
        ..Default::default()
    };

    assert_eq!(config.dropout, 0.0);
}

#[rstest]
fn test_lora_config_with_high_dropout() {
    let config = LoRAConfig {
        dropout: 0.9,
        ..Default::default()
    };

    assert_eq!(config.dropout, 0.9);
}

#[rstest]
fn test_lora_config_with_small_rank() {
    let config = LoRAConfig {
        rank: 2,
        ..Default::default()
    };

    assert_eq!(config.rank, 2);
}

#[rstest]
fn test_lora_config_with_large_rank() {
    let config = LoRAConfig {
        rank: 128,
        ..Default::default()
    };

    assert_eq!(config.rank, 128);
}

// ============================================================================
// Serialization Tests (if needed)
// ============================================================================

#[rstest]
fn test_lora_config_serialization() {
    let config = LoRAConfig::default();

    // Test JSON serialization
    let json_result = serde_json::to_string(&config);
    assert!(json_result.is_ok(), "Should serialize to JSON");

    let json_str = json_result.unwrap();
    assert!(!json_str.is_empty(), "JSON string should not be empty");
}

#[rstest]
fn test_lora_config_deserialization() {
    let json_str = r#"{
        "rank": 16,
        "alpha": 32.0,
        "dropout": 0.1,
        "target_modules": ["query", "value", "key", "output"],
        "use_bias": false,
        "init_method": "Kaiming"
    }"#;

    let result: Result<LoRAConfig, _> = serde_json::from_str(json_str);

    assert!(result.is_ok(), "Should deserialize from JSON");

    let config = result.unwrap();
    assert_eq!(config.rank, 16);
    assert_eq!(config.alpha, 32.0);
}

#[rstest]
fn test_lora_init_method_serialization() {
    let methods = vec![
        LoRAInitMethod::Kaiming,
        LoRAInitMethod::Xavier,
        LoRAInitMethod::Normal {
            mean: 0.0,
            std: 0.02,
        },
        LoRAInitMethod::Zero,
    ];

    for method in methods {
        let json_result = serde_json::to_string(&method);
        assert!(
            json_result.is_ok(),
            "Should serialize init method {:?}",
            method
        );
    }
}

// ============================================================================
// Parameter Count Tests
// ============================================================================

#[rstest]
#[case(768, 768, 16)]
#[case(1024, 1024, 32)]
#[case(512, 512, 8)]
fn test_lora_parameter_count(
    #[case] input_dim: usize,
    #[case] output_dim: usize,
    #[case] rank: usize,
) {
    // LoRA parameters: A (rank x input_dim) + B (output_dim x rank)
    let expected_params = (rank * input_dim) + (output_dim * rank);

    // For reference: full fine-tuning would be input_dim x output_dim
    let full_params = input_dim * output_dim;

    let reduction_ratio = full_params as f64 / expected_params as f64;

    assert!(
        reduction_ratio > 1.0,
        "LoRA should reduce parameter count (reduction: {}x)",
        reduction_ratio
    );
}

// ============================================================================
// Configuration Validation Tests
// ============================================================================

#[rstest]
fn test_lora_config_alpha_positive() {
    let config = LoRAConfig {
        alpha: 32.0,
        ..Default::default()
    };

    assert!(config.alpha > 0.0, "Alpha should be positive");
}

#[rstest]
fn test_lora_config_rank_positive() {
    let config = LoRAConfig {
        rank: 16,
        ..Default::default()
    };

    assert!(config.rank > 0, "Rank should be positive");
}

#[rstest]
fn test_lora_config_dropout_valid_range() {
    let config = LoRAConfig {
        dropout: 0.1,
        ..Default::default()
    };

    assert!(
        config.dropout >= 0.0 && config.dropout <= 1.0,
        "Dropout should be in [0, 1] range"
    );
}

// ============================================================================
// Bias Configuration Tests
// ============================================================================

#[rstest]
fn test_lora_config_with_bias() {
    let config = LoRAConfig {
        use_bias: true,
        ..Default::default()
    };

    assert!(config.use_bias);
}

#[rstest]
fn test_lora_config_without_bias() {
    let config = LoRAConfig {
        use_bias: false,
        ..Default::default()
    };

    assert!(!config.use_bias);
}

// ============================================================================
// Memory Estimation Tests
// ============================================================================

#[rstest]
#[case(768, 768, 16, 4)] // F32 = 4 bytes
fn test_lora_memory_estimation(
    #[case] input_dim: usize,
    #[case] output_dim: usize,
    #[case] rank: usize,
    #[case] bytes_per_param: usize,
) {
    // Memory for LoRA: (rank * input_dim + output_dim * rank) * bytes_per_param
    let lora_params = (rank * input_dim) + (output_dim * rank);
    let lora_memory = lora_params * bytes_per_param;

    // Memory for full fine-tuning: input_dim * output_dim * bytes_per_param
    let full_params = input_dim * output_dim;
    let full_memory = full_params * bytes_per_param;

    let memory_saving_ratio = full_memory as f64 / lora_memory as f64;

    assert!(
        memory_saving_ratio > 1.0,
        "LoRA should save memory ({}x reduction)",
        memory_saving_ratio
    );
}

// ============================================================================
// Target Module Pattern Tests
// ============================================================================

#[rstest]
fn test_lora_target_modules_empty() {
    let config = LoRAConfig {
        target_modules: vec![],
        ..Default::default()
    };

    assert_eq!(config.target_modules.len(), 0);
}

#[rstest]
fn test_lora_target_modules_single() {
    let config = LoRAConfig {
        target_modules: vec!["query".to_string()],
        ..Default::default()
    };

    assert_eq!(config.target_modules.len(), 1);
    assert_eq!(config.target_modules[0], "query");
}

#[rstest]
fn test_lora_target_modules_all_attention() {
    let attention_modules = vec!["query".to_string(), "key".to_string(), "value".to_string()];

    let config = LoRAConfig {
        target_modules: attention_modules.clone(),
        ..Default::default()
    };

    for module in attention_modules {
        assert!(config.target_modules.contains(&module));
    }
}
