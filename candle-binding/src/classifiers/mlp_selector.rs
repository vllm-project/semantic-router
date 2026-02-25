//! MLP (Multi-Layer Perceptron) Selector
//!
//! GPU-accelerated neural network for model selection.
//! Reference: FusionFactory (arXiv:2507.10540) - Query-level fusion via tailored LLM routers
//!
//! ## Architecture
//! - Training: Done in Python using PyTorch (src/training/model_selection/ml_model_selection/)
//! - Inference: Done in Rust using Candle for GPU acceleration
//!
//! ## Model Format
//! Models are loaded from JSON files with the following structure:
//! - `layers`: List of layer definitions (linear, relu, batch_norm, dropout)
//! - `model_names`: List of model names for classification
//! - `hidden_sizes`: Hidden layer dimensions
//! - `feature_dim`: Input feature dimension
//!
//! ## Mixed Precision Support
//! Supports multiple data types for inference:
//! - `f32`: Full precision (default, best accuracy)
//! - `f16`: Half precision (faster on GPUs with tensor cores)
//! - `bf16`: BFloat16 (good balance of range and speed)
//!
//! Use `with_dtype()` or `from_json_with_dtype()` to specify precision.

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// Supported data types for mixed precision inference
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MLPDType {
    /// Full precision (32-bit float) - default, best accuracy
    F32,
    /// Half precision (16-bit float) - faster on GPUs with tensor cores
    F16,
    /// BFloat16 - good balance of dynamic range and speed
    BF16,
}

impl MLPDType {
    /// Convert to Candle DType
    fn to_candle_dtype(&self) -> DType {
        match self {
            MLPDType::F32 => DType::F32,
            MLPDType::F16 => DType::F16,
            MLPDType::BF16 => DType::BF16,
        }
    }

    /// Parse from string (e.g., from config)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "f32" | "float32" | "fp32" => Ok(MLPDType::F32),
            "f16" | "float16" | "fp16" | "half" => Ok(MLPDType::F16),
            "bf16" | "bfloat16" => Ok(MLPDType::BF16),
            _ => Err(format!("Unknown dtype '{}'. Supported: f32, f16, bf16", s)),
        }
    }
}

impl Default for MLPDType {
    fn default() -> Self {
        MLPDType::F32
    }
}

/// MLP layer definition from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum LayerDef {
    #[serde(rename = "linear")]
    Linear {
        in_features: usize,
        out_features: usize,
        weight: Vec<Vec<f64>>,
        bias: Option<Vec<f64>>,
    },
    #[serde(rename = "relu")]
    ReLU,
    #[serde(rename = "batch_norm")]
    BatchNorm {
        num_features: usize,
        weight: Option<Vec<f64>>,
        bias: Option<Vec<f64>>,
        running_mean: Option<Vec<f64>>,
        running_var: Option<Vec<f64>>,
        eps: Option<f64>,
    },
    #[serde(rename = "dropout")]
    Dropout { p: f64 },
}

/// MLP model data from JSON
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPModelData {
    pub algorithm: String,
    pub trained: bool,
    pub model_names: Vec<String>,
    pub feature_dim: usize,
    pub n_classes: usize,
    pub hidden_sizes: Vec<usize>,
    pub dropout: f64,
    pub layers: Vec<LayerDef>,
}

/// Compiled MLP layer for inference
enum CompiledLayer {
    Linear(Tensor, Tensor), // (weight, bias)
    ReLU,
    BatchNorm {
        weight: Tensor,
        bias: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        eps: f64,
    },
    Dropout, // No-op during inference
}

/// MLP Selector for model selection
pub struct MLPSelector {
    layers: Vec<CompiledLayer>,
    model_names: Vec<String>,
    feature_dim: usize,
    device: Device,
    dtype: MLPDType,
    trained: bool,
}

impl MLPSelector {
    /// Create a new MLP selector (untrained, f32 precision)
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            model_names: Vec::new(),
            feature_dim: 0,
            device: Device::Cpu,
            dtype: MLPDType::F32,
            trained: false,
        }
    }

    /// Create MLP selector with specific device (f32 precision)
    pub fn with_device(device: Device) -> Self {
        Self {
            layers: Vec::new(),
            model_names: Vec::new(),
            feature_dim: 0,
            device,
            dtype: MLPDType::F32,
            trained: false,
        }
    }

    /// Create MLP selector with specific device and dtype for mixed precision
    pub fn with_device_and_dtype(device: Device, dtype: MLPDType) -> Self {
        Self {
            layers: Vec::new(),
            model_names: Vec::new(),
            feature_dim: 0,
            device,
            dtype,
            trained: false,
        }
    }

    /// Get the current dtype
    pub fn dtype(&self) -> MLPDType {
        self.dtype
    }

    /// Check if model is trained/loaded
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Get model names
    pub fn model_names(&self) -> &[String] {
        &self.model_names
    }

    /// Select model for a query embedding
    /// Supports mixed precision inference based on configured dtype
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        if !self.trained {
            return Err("Model not trained/loaded".to_string());
        }

        if query.len() != self.feature_dim {
            return Err(format!(
                "Feature dimension mismatch: expected {}, got {}",
                self.feature_dim,
                query.len()
            ));
        }

        // Convert query to tensor with configured dtype for mixed precision
        let query_f32: Vec<f32> = query.iter().map(|&x| x as f32).collect();
        let x = Tensor::from_vec(query_f32, (1, self.feature_dim), &self.device)
            .map_err(|e| format!("Failed to create input tensor: {}", e))?;

        // Convert to target dtype if not f32 (for mixed precision)
        let x = if self.dtype != MLPDType::F32 {
            x.to_dtype(self.dtype.to_candle_dtype())
                .map_err(|e| format!("Failed to convert input to {:?}: {}", self.dtype, e))?
        } else {
            x
        };

        // Forward pass through layers
        let output = self.forward(x)?;

        // Convert output back to f32 for argmax (always use f32 for final computation)
        let output_f32 = if self.dtype != MLPDType::F32 {
            output
                .to_dtype(DType::F32)
                .map_err(|e| format!("Failed to convert output to f32: {}", e))?
        } else {
            output
        };

        // Get argmax
        let output_vec = output_f32
            .squeeze(0)
            .map_err(|e| format!("Squeeze failed: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| format!("Failed to get output: {}", e))?;

        let (max_idx, _) = output_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .ok_or("Empty output")?;

        if max_idx < self.model_names.len() {
            Ok(self.model_names[max_idx].clone())
        } else {
            Err(format!(
                "Invalid class index: {} (have {} classes)",
                max_idx,
                self.model_names.len()
            ))
        }
    }

    /// Forward pass through all layers
    fn forward(&self, mut x: Tensor) -> Result<Tensor, String> {
        for layer in &self.layers {
            x = match layer {
                CompiledLayer::Linear(weight, bias) => {
                    // x @ weight.T + bias
                    let weight_t = weight.t().map_err(|e| format!("Transpose failed: {}", e))?;
                    let out = x
                        .matmul(&weight_t)
                        .map_err(|e| format!("Matmul failed: {}", e))?;
                    out.broadcast_add(bias)
                        .map_err(|e| format!("Bias add failed: {}", e))?
                }
                CompiledLayer::ReLU => x.relu().map_err(|e| format!("ReLU failed: {}", e))?,
                CompiledLayer::BatchNorm {
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    eps,
                } => {
                    // Batch normalization: (x - mean) / sqrt(var + eps) * weight + bias
                    let x_norm = x
                        .broadcast_sub(running_mean)
                        .map_err(|e| format!("BN sub failed: {}", e))?;
                    // Create eps tensor with matching dtype for mixed precision
                    let eps_tensor = Tensor::new(&[*eps as f32], &self.device)
                        .unwrap()
                        .to_dtype(self.dtype.to_candle_dtype())
                        .map_err(|e| format!("BN eps dtype conversion: {}", e))?;
                    let var_eps = running_var
                        .broadcast_add(&eps_tensor)
                        .map_err(|e| format!("BN var_eps failed: {}", e))?;
                    let std = var_eps
                        .sqrt()
                        .map_err(|e| format!("BN sqrt failed: {}", e))?;
                    let x_norm = x_norm
                        .broadcast_div(&std)
                        .map_err(|e| format!("BN div failed: {}", e))?;
                    let x_scaled = x_norm
                        .broadcast_mul(weight)
                        .map_err(|e| format!("BN scale failed: {}", e))?;
                    x_scaled
                        .broadcast_add(bias)
                        .map_err(|e| format!("BN shift failed: {}", e))?
                }
                CompiledLayer::Dropout => x, // No-op during inference
            };
        }
        Ok(x)
    }

    /// Load model from JSON string (f32 precision)
    pub fn from_json(json: &str) -> Result<Self, String> {
        Self::from_json_with_dtype(json, Device::Cpu, MLPDType::F32)
    }

    /// Load model from JSON with specific device (f32 precision)
    pub fn from_json_with_device(json: &str, device: Device) -> Result<Self, String> {
        Self::from_json_with_dtype(json, device, MLPDType::F32)
    }

    /// Load model from JSON with specific device and dtype for mixed precision
    ///
    /// # Arguments
    /// * `json` - JSON string containing the model data
    /// * `device` - Device to load model on (CPU, CUDA, Metal)
    /// * `dtype` - Data type for inference (F32, F16, BF16)
    ///
    /// # Example
    /// ```ignore
    /// let model = MLPSelector::from_json_with_dtype(json, Device::Cpu, MLPDType::F16)?;
    /// ```
    pub fn from_json_with_dtype(
        json: &str,
        device: Device,
        dtype: MLPDType,
    ) -> Result<Self, String> {
        let data: MLPModelData =
            serde_json::from_str(json).map_err(|e| format!("JSON parse error: {}", e))?;

        if data.algorithm != "mlp" {
            return Err(format!(
                "Invalid algorithm: expected 'mlp', got '{}'",
                data.algorithm
            ));
        }

        Self::from_model_data(data, device, dtype)
    }

    /// Build selector from model data with mixed precision support
    fn from_model_data(
        data: MLPModelData,
        device: Device,
        dtype: MLPDType,
    ) -> Result<Self, String> {
        let target_dtype = dtype.to_candle_dtype();
        let mut layers = Vec::new();

        for layer_def in &data.layers {
            match layer_def {
                LayerDef::Linear {
                    in_features,
                    out_features,
                    weight,
                    bias,
                } => {
                    // Flatten weight matrix and create tensor in f32 first
                    let weight_flat: Vec<f32> = weight
                        .iter()
                        .flat_map(|row| row.iter().map(|&x| x as f32))
                        .collect();

                    let weight_tensor =
                        Tensor::from_vec(weight_flat, (*out_features, *in_features), &device)
                            .map_err(|e| format!("Failed to create weight tensor: {}", e))?;

                    let bias_tensor = if let Some(b) = bias {
                        let bias_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(bias_f32, *out_features, &device)
                            .map_err(|e| format!("Failed to create bias tensor: {}", e))?
                    } else {
                        Tensor::zeros(*out_features, DType::F32, &device)
                            .map_err(|e| format!("Failed to create zero bias: {}", e))?
                    };

                    // Convert to target dtype for mixed precision
                    let weight_tensor = weight_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("Failed to convert weight to {:?}: {}", dtype, e))?;
                    let bias_tensor = bias_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("Failed to convert bias to {:?}: {}", dtype, e))?;

                    layers.push(CompiledLayer::Linear(weight_tensor, bias_tensor));
                }
                LayerDef::ReLU => {
                    layers.push(CompiledLayer::ReLU);
                }
                LayerDef::BatchNorm {
                    num_features,
                    weight,
                    bias,
                    running_mean,
                    running_var,
                    eps,
                } => {
                    // Create tensors in f32 first, then convert for mixed precision
                    let weight_tensor = if let Some(w) = weight {
                        let w_f32: Vec<f32> = w.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(w_f32, *num_features, &device)
                            .map_err(|e| format!("BN weight error: {}", e))?
                    } else {
                        Tensor::ones(*num_features, DType::F32, &device)
                            .map_err(|e| format!("BN weight ones error: {}", e))?
                    };

                    let bias_tensor = if let Some(b) = bias {
                        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(b_f32, *num_features, &device)
                            .map_err(|e| format!("BN bias error: {}", e))?
                    } else {
                        Tensor::zeros(*num_features, DType::F32, &device)
                            .map_err(|e| format!("BN bias zeros error: {}", e))?
                    };

                    let mean_tensor = if let Some(m) = running_mean {
                        let m_f32: Vec<f32> = m.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(m_f32, *num_features, &device)
                            .map_err(|e| format!("BN mean error: {}", e))?
                    } else {
                        Tensor::zeros(*num_features, DType::F32, &device)
                            .map_err(|e| format!("BN mean zeros error: {}", e))?
                    };

                    let var_tensor = if let Some(v) = running_var {
                        let v_f32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
                        Tensor::from_vec(v_f32, *num_features, &device)
                            .map_err(|e| format!("BN var error: {}", e))?
                    } else {
                        Tensor::ones(*num_features, DType::F32, &device)
                            .map_err(|e| format!("BN var ones error: {}", e))?
                    };

                    // Convert to target dtype for mixed precision
                    let weight_tensor = weight_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("BN weight dtype conversion: {}", e))?;
                    let bias_tensor = bias_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("BN bias dtype conversion: {}", e))?;
                    let mean_tensor = mean_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("BN mean dtype conversion: {}", e))?;
                    let var_tensor = var_tensor
                        .to_dtype(target_dtype)
                        .map_err(|e| format!("BN var dtype conversion: {}", e))?;

                    layers.push(CompiledLayer::BatchNorm {
                        weight: weight_tensor,
                        bias: bias_tensor,
                        running_mean: mean_tensor,
                        running_var: var_tensor,
                        eps: eps.unwrap_or(1e-5),
                    });
                }
                LayerDef::Dropout { .. } => {
                    layers.push(CompiledLayer::Dropout);
                }
            }
        }

        Ok(Self {
            layers,
            model_names: data.model_names,
            feature_dim: data.feature_dim,
            device,
            dtype,
            trained: data.trained,
        })
    }

    /// Serialize model to JSON
    pub fn to_json(&self) -> Result<String, String> {
        // Note: Full serialization requires converting tensors back to vectors
        // For now, return minimal info (full model should be saved from Python)
        let data = serde_json::json!({
            "algorithm": "mlp",
            "trained": self.trained,
            "model_names": self.model_names,
            "feature_dim": self.feature_dim,
            "n_classes": self.model_names.len(),
            "layers": [], // Layers not serialized from Rust
        });
        serde_json::to_string(&data).map_err(|e| format!("JSON serialize error: {}", e))
    }
}

impl Default for MLPSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_new() {
        let selector = MLPSelector::new();
        assert!(!selector.is_trained());
    }

    #[test]
    fn test_mlp_from_json_minimal() {
        let json = r#"{
            "algorithm": "mlp",
            "trained": true,
            "model_names": ["model_a", "model_b"],
            "feature_dim": 4,
            "n_classes": 2,
            "hidden_sizes": [8],
            "dropout": 0.1,
            "layers": [
                {
                    "type": "linear",
                    "in_features": 4,
                    "out_features": 8,
                    "weight": [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                    "bias": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                },
                {"type": "relu"},
                {
                    "type": "linear",
                    "in_features": 8,
                    "out_features": 2,
                    "weight": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
                    "bias": [0.1, 0.2]
                }
            ]
        }"#;

        let selector = MLPSelector::from_json(json).unwrap();
        assert!(selector.is_trained());
        assert_eq!(selector.model_names().len(), 2);

        // Test prediction
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let result = selector.select(&query);
        assert!(result.is_ok());
    }
}
