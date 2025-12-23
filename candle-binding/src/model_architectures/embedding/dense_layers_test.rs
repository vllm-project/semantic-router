//! Unit tests for Dense Bottleneck layers
//!
//! ## Test Coverage
//! - DenseActivation functions
//! - DenseLayer construction and forward pass (using manually created weights)
//! - BottleneckDenseNet architecture validation
//! - Input/output shape validation
//!
//! ## Testing Strategy
//! - Use `rstest` for parameterized tests
//! - Use manually created test weights (not loading from actual model files)
//! - Focus on shape validation and mathematical correctness

use crate::core::UnifiedError;
use crate::model_architectures::embedding::dense_layers::{
    BottleneckDenseNet, DenseActivation, DenseLayer,
};
use candle_core::Tensor;
use candle_nn::Linear;
use rstest::*;
use serial_test::serial;

// Import test fixture
use crate::test_fixtures::fixtures::test_device;

/// Test DenseActivation::Identity
#[rstest]
#[case::simple_values(vec![1.0, 2.0, 3.0])]
#[case::negative_values(vec![-1.0, -2.0, -3.0])]
#[case::mixed_values(vec![-1.5, 0.0, 1.5, 2.5])]
#[case::zero(vec![0.0])]
fn test_dense_activation_identity(#[case] input_vec: Vec<f32>) {
    let device = test_device();
    let input = Tensor::new(input_vec.as_slice(), &device).unwrap();
    let activation = DenseActivation::Identity;

    let output = activation.apply(&input).unwrap();

    // Identity should preserve values exactly
    let output_vec: Vec<f32> = output.to_vec1().unwrap();
    assert_eq!(
        output_vec, input_vec,
        "Identity activation should preserve input"
    );
}

/// Test DenseActivation::Tanh
#[rstest]
#[case::zero(0.0, 0.0, 1e-6)]
#[case::positive_one(1.0, 0.7615942, 1e-5)]
#[case::negative_one(-1.0, -0.7615942, 1e-5)]
#[case::large_positive(5.0, 0.9999092, 1e-5)]
#[case::large_negative(-5.0, -0.9999092, 1e-5)]
fn test_dense_activation_tanh(#[case] input: f32, #[case] expected: f32, #[case] tolerance: f32) {
    let device = test_device();
    let input_tensor = Tensor::new(&[input], &device).unwrap();
    let activation = DenseActivation::Tanh;

    let output = activation.apply(&input_tensor).unwrap();

    let output_value: Vec<f32> = output.to_vec1().unwrap();
    assert!(
        (output_value[0] - expected).abs() < tolerance,
        "tanh({}) = {}, expected {}, diff = {}",
        input,
        output_value[0],
        expected,
        (output_value[0] - expected).abs()
    );
}

/// Test DenseActivation::Tanh symmetry
#[rstest]
fn test_dense_activation_tanh_symmetry() {
    let device = test_device();
    let input = Tensor::new(&[1.0f32, -1.0, 2.0, -2.0], &device).unwrap();
    let activation = DenseActivation::Tanh;

    let output = activation.apply(&input).unwrap();
    let output_vec: Vec<f32> = output.to_vec1().unwrap();

    // Tanh should be antisymmetric: tanh(-x) = -tanh(x)
    assert!(
        (output_vec[0] + output_vec[1]).abs() < 1e-6,
        "tanh(1) + tanh(-1) should be ~0"
    );
    assert!(
        (output_vec[2] + output_vec[3]).abs() < 1e-6,
        "tanh(2) + tanh(-2) should be ~0"
    );
}

/// Test DenseActivation::Tanh saturation
#[rstest]
fn test_dense_activation_tanh_saturation() {
    let device = test_device();
    let input = Tensor::new(&[10.0f32, -10.0], &device).unwrap();
    let activation = DenseActivation::Tanh;

    let output = activation.apply(&input).unwrap();
    let output_vec: Vec<f32> = output.to_vec1().unwrap();

    // Tanh saturates at ±1 for large inputs
    assert!(
        (output_vec[0] - 1.0).abs() < 1e-4,
        "tanh(10) should be close to 1.0"
    );
    assert!(
        (output_vec[1] + 1.0).abs() < 1e-4,
        "tanh(-10) should be close to -1.0"
    );
}

/// Test DenseLayer input dimension validation
///
/// **Purpose**: Verify that DenseLayer correctly validates input dimensions
/// **Strategy**: Create a layer with known dimensions and test with various input shapes
#[rstest]
#[case::correct_dim(768, true)]
#[case::wrong_dim_512(512, false)]
#[case::wrong_dim_1024(1024, false)]
fn test_dense_layer_input_validation(#[case] input_dim: usize, #[case] should_pass: bool) {
    let device = test_device();

    // Create a simple linear layer manually for testing
    // This simulates a DenseLayer with in_features=768, out_features=3072
    let weight = Tensor::randn(0f32, 1.0f32, (3072, 768), &device).unwrap();
    let linear = Linear::new(weight, None);

    let layer = DenseLayer {
        linear,
        activation: DenseActivation::Identity,
        in_features: 768,
        out_features: 3072,
    };

    // Create input with specified dimension
    let input = Tensor::randn(0f32, 1.0f32, (1, input_dim), &device).unwrap();

    let result = layer.forward(&input);

    if should_pass {
        assert!(
            result.is_ok(),
            "Should accept input with correct dimension {}",
            input_dim
        );
        let output = result.unwrap();
        assert_eq!(output.dims(), &[1, 3072], "Output shape mismatch");
    } else {
        assert!(
            result.is_err(),
            "Should reject input with incorrect dimension {}",
            input_dim
        );
        if let Err(UnifiedError::Validation {
            field,
            expected,
            actual,
            ..
        }) = result
        {
            assert_eq!(field, "input dimension");
            assert_eq!(expected, "768");
            assert_eq!(actual, input_dim.to_string());
        } else {
            panic!("Expected Validation error, got: {:?}", result);
        }
    }
}

/// Test DenseLayer forward pass with Identity activation
#[rstest]
#[case::batch_1(1, 768, 3072)]
#[case::batch_4(4, 768, 3072)]
#[case::batch_16(16, 768, 3072)]
fn test_dense_layer_forward_identity(
    #[case] batch_size: usize,
    #[case] in_features: usize,
    #[case] out_features: usize,
) {
    let device = test_device();

    // Create weight and layer
    let weight = Tensor::randn(0f32, 1.0f32, (out_features, in_features), &device).unwrap();
    let linear = Linear::new(weight, None);

    let layer = DenseLayer {
        linear,
        activation: DenseActivation::Identity,
        in_features,
        out_features,
    };

    // Create random input
    let input = Tensor::randn(0f32, 1.0f32, (batch_size, in_features), &device).unwrap();

    // Forward pass
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.dims(), &[batch_size, out_features]);
}

/// Test DenseLayer forward pass with Tanh activation
#[rstest]
fn test_dense_layer_forward_tanh() {
    let device = test_device();
    let in_features = 768;
    let out_features = 3072;

    // Create weight and layer
    let weight = Tensor::randn(0f32, 1.0f32, (out_features, in_features), &device).unwrap();
    let linear = Linear::new(weight, None);

    let layer = DenseLayer {
        linear,
        activation: DenseActivation::Tanh,
        in_features,
        out_features,
    };

    // Create input
    let input = Tensor::randn(0f32, 1.0f32, (2, in_features), &device).unwrap();

    // Forward pass
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.dims(), &[2, out_features]);

    // Verify Tanh saturation: all values should be in range [-1, 1]
    let output_vec: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();
    for &val in output_vec.iter() {
        assert!(
            val >= -1.0 && val <= 1.0,
            "Tanh output {} out of range [-1, 1]",
            val
        );
    }
}

/// Test DenseLayer with bias
#[rstest]
fn test_dense_layer_with_bias() {
    let device = test_device();
    let in_features = 768;
    let out_features = 3072;

    // Create weight and bias
    let weight = Tensor::randn(0f32, 1.0f32, (out_features, in_features), &device).unwrap();
    let bias = Tensor::randn(0f32, 1.0f32, (out_features,), &device).unwrap();
    let linear = Linear::new(weight, Some(bias));

    let layer = DenseLayer {
        linear,
        activation: DenseActivation::Identity,
        in_features,
        out_features,
    };

    // Create input
    let input = Tensor::randn(0f32, 1.0f32, (1, in_features), &device).unwrap();

    // Forward pass
    let output = layer.forward(&input).unwrap();

    // Verify output shape
    assert_eq!(output.dims(), &[1, out_features]);
}

/// Test DenseLayer accessor methods
#[rstest]
fn test_dense_layer_accessors() {
    let device = test_device();
    let in_features = 768;
    let out_features = 3072;

    let weight = Tensor::randn(0f32, 1.0f32, (out_features, in_features), &device).unwrap();
    let linear = Linear::new(weight, None);

    let layer = DenseLayer {
        linear,
        activation: DenseActivation::Identity,
        in_features,
        out_features,
    };

    assert_eq!(layer.in_features(), in_features);
    assert_eq!(layer.out_features(), out_features);
}

/// Test BottleneckDenseNet input validation
///
/// **Purpose**: Verify that BottleneckDenseNet validates input dimension (must be 768)
#[rstest]
#[case::correct_768(768, true)]
#[case::wrong_512(512, false)]
#[case::wrong_1024(1024, false)]
#[case::wrong_3072(3072, false)]
fn test_bottleneck_input_validation(#[case] input_dim: usize, #[case] should_pass: bool) {
    let device = test_device();

    // Create BottleneckDenseNet with manually constructed layers
    let weight1 = Tensor::randn(0f32, 1.0f32, (3072, 768), &device).unwrap();
    let linear1 = Linear::new(weight1, None);
    let dense1 = DenseLayer {
        linear: linear1,
        activation: DenseActivation::Identity,
        in_features: 768,
        out_features: 3072,
    };

    let weight2 = Tensor::randn(0f32, 1.0f32, (768, 3072), &device).unwrap();
    let linear2 = Linear::new(weight2, None);
    let dense2 = DenseLayer {
        linear: linear2,
        activation: DenseActivation::Identity,
        in_features: 3072,
        out_features: 768,
    };

    let bottleneck = BottleneckDenseNet { dense1, dense2 };

    // Create input with specified dimension
    let input = Tensor::randn(0f32, 1.0f32, (1, input_dim), &device).unwrap();

    let result = bottleneck.forward(&input);

    if should_pass {
        assert!(result.is_ok(), "Should accept input with dimension 768");
        let output = result.unwrap();
        assert_eq!(output.dims(), &[1, 768], "Output should be [1, 768]");
    } else {
        assert!(
            result.is_err(),
            "Should reject input with dimension {}",
            input_dim
        );
        if let Err(UnifiedError::Validation {
            field,
            expected,
            actual,
            ..
        }) = result
        {
            assert_eq!(field, "input dimension");
            assert_eq!(expected, "768");
            assert_eq!(actual, input_dim.to_string());
        } else {
            panic!("Expected Validation error, got: {:?}", result);
        }
    }
}

/// Test BottleneckDenseNet forward pass with various batch sizes
///
/// **Purpose**: Verify that bottleneck correctly handles different batch sizes
/// **Expected**: Input [batch, 768] → Output [batch, 768]
#[rstest]
#[case::batch_1(1)]
#[case::batch_2(2)]
#[case::batch_4(4)]
#[case::batch_8(8)]
#[case::batch_16(16)]
fn test_bottleneck_forward_batch_sizes(#[case] batch_size: usize) {
    let device = test_device();

    // Create BottleneckDenseNet
    let weight1 = Tensor::randn(0f32, 1.0f32, (3072, 768), &device).unwrap();
    let linear1 = Linear::new(weight1, None);
    let dense1 = DenseLayer {
        linear: linear1,
        activation: DenseActivation::Identity,
        in_features: 768,
        out_features: 3072,
    };

    let weight2 = Tensor::randn(0f32, 1.0f32, (768, 3072), &device).unwrap();
    let linear2 = Linear::new(weight2, None);
    let dense2 = DenseLayer {
        linear: linear2,
        activation: DenseActivation::Identity,
        in_features: 3072,
        out_features: 768,
    };

    let bottleneck = BottleneckDenseNet { dense1, dense2 };

    // Create input
    let input = Tensor::randn(0f32, 1.0f32, (batch_size, 768), &device).unwrap();

    // Forward pass
    let output = bottleneck.forward(&input).unwrap();

    // Verify output shape: should preserve batch dimension, output 768 features
    assert_eq!(output.dims(), &[batch_size, 768]);
}

/// Test BottleneckDenseNet accessor methods
#[rstest]
fn test_bottleneck_accessors() {
    let device = test_device();

    // Create BottleneckDenseNet
    let weight1 = Tensor::randn(0f32, 1.0f32, (3072, 768), &device).unwrap();
    let linear1 = Linear::new(weight1, None);
    let dense1 = DenseLayer {
        linear: linear1,
        activation: DenseActivation::Identity,
        in_features: 768,
        out_features: 3072,
    };

    let weight2 = Tensor::randn(0f32, 1.0f32, (768, 3072), &device).unwrap();
    let linear2 = Linear::new(weight2, None);
    let dense2 = DenseLayer {
        linear: linear2,
        activation: DenseActivation::Identity,
        in_features: 3072,
        out_features: 768,
    };

    let bottleneck = BottleneckDenseNet { dense1, dense2 };

    // Test accessors
    assert_eq!(bottleneck.expansion_layer().in_features(), 768);
    assert_eq!(bottleneck.expansion_layer().out_features(), 3072);
    assert_eq!(bottleneck.compression_layer().in_features(), 3072);
    assert_eq!(bottleneck.compression_layer().out_features(), 768);
}

/// Test BottleneckDenseNet dimension preservation
///
/// **Purpose**: Verify that bottleneck preserves the input dimension (768)
/// **Architecture**: 768 → 3072 → 768
#[rstest]
fn test_bottleneck_dimension_preservation() {
    let device = test_device();

    // Create BottleneckDenseNet
    let weight1 = Tensor::randn(0f32, 1.0f32, (3072, 768), &device).unwrap();
    let linear1 = Linear::new(weight1, None);
    let dense1 = DenseLayer {
        linear: linear1,
        activation: DenseActivation::Identity,
        in_features: 768,
        out_features: 3072,
    };

    let weight2 = Tensor::randn(0f32, 1.0f32, (768, 3072), &device).unwrap();
    let linear2 = Linear::new(weight2, None);
    let dense2 = DenseLayer {
        linear: linear2,
        activation: DenseActivation::Identity,
        in_features: 3072,
        out_features: 768,
    };

    let bottleneck = BottleneckDenseNet { dense1, dense2 };

    // Test with multiple batch sizes
    for batch_size in [1, 2, 4, 8] {
        let input = Tensor::randn(0f32, 1.0f32, (batch_size, 768), &device).unwrap();
        let output = bottleneck.forward(&input).unwrap();

        // Input and output should have same dimensions
        assert_eq!(
            input.dims(),
            output.dims(),
            "Bottleneck should preserve dimensions for batch size {}",
            batch_size
        );
    }
}

// =============================================================================
// Real Model Loading Tests
// =============================================================================

/// Test loading Dense Bottleneck from actual model files
#[rstest]
#[serial]
fn test_dense_bottleneck_load_from_path() {
    use candle_core::{DType, Tensor};

    let model_path = "../models/mom-embedding-flash";
    let device = test_device();

    println!("\n=== Loading Dense Bottleneck from Path ===");
    let bottleneck: BottleneckDenseNet =
        BottleneckDenseNet::load_from_path(model_path, &device).expect("Failed to load bottleneck");
    println!("  ✅ Loaded successfully");

    // Create test input: [batch=2, dim=768]
    let input = Tensor::ones((2, 768), DType::F32, &device).expect("Failed to create input");
    println!("\n=== Forward pass ===");
    println!("  Input shape: {:?}", input.dims());
    println!(
        "  Input mean: {:.6}",
        input.mean_all().unwrap().to_scalar::<f32>().unwrap()
    );

    let output = bottleneck.forward(&input).expect("Forward pass failed");
    println!("  Output shape: {:?}", output.dims());

    let output_vec = output.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let has_nan = output_vec.iter().any(|x| x.is_nan());
    let has_inf = output_vec.iter().any(|x| x.is_infinite());

    println!("  Output contains NaN: {}", has_nan);
    println!("  Output contains Inf: {}", has_inf);

    assert!(!has_nan, "❌ Dense Bottleneck produces NaN!");
    assert!(!has_inf, "❌ Dense Bottleneck produces Inf!");

    let sum: f32 = output_vec.iter().sum();
    let mean = sum / output_vec.len() as f32;
    println!("  Output mean: {:.6}", mean);
    println!("  ✅ Dense Bottleneck works correctly");
}
