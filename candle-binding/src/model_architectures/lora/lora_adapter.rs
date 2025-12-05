//! LoRA adapter core implementation
//!
//! This module provides the core LoRA (Low-Rank Adaptation) adapter implementation
//! for parameter-efficient fine-tuning of transformer models.

use candle_core::{Device, Result, Tensor};
use candle_nn::{Dropout, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LoRA adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// LoRA rank (typically 4, 8, 16, 32, 64)
    pub rank: usize,
    /// LoRA alpha parameter for scaling
    pub alpha: f64,
    /// Dropout rate for LoRA layers
    pub dropout: f64,
    /// Target modules to apply LoRA to
    pub target_modules: Vec<String>,
    /// Whether to use bias in LoRA layers
    pub use_bias: bool,
    /// Initialization method for LoRA weights
    pub init_method: LoRAInitMethod,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            target_modules: vec![
                "query".to_string(),
                "value".to_string(),
                "key".to_string(),
                "output".to_string(),
            ],
            use_bias: false,
            init_method: LoRAInitMethod::Kaiming,
        }
    }
}

/// LoRA weight initialization methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoRAInitMethod {
    /// Kaiming/He initialization
    Kaiming,
    /// Xavier/Glorot initialization
    Xavier,
    /// Normal distribution initialization
    Normal { mean: f64, std: f64 },
    /// Zero initialization for B matrix
    Zero,
}

/// Core LoRA adapter implementation
#[derive(Debug)]
pub struct LoRAAdapter {
    /// Low-rank matrix A (rank x input_dim)
    lora_a: Linear,
    /// Low-rank matrix B (output_dim x rank)
    lora_b: Linear,
    /// Dropout layer
    dropout: Dropout,
    /// Scaling factor (alpha / rank)
    scaling: f64,
    /// Configuration
    config: LoRAConfig,
}

impl LoRAAdapter {
    /// Create a new LoRA adapter
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        config: &LoRAConfig,
        vb: VarBuilder,
        _device: &Device,
    ) -> Result<Self> {
        // Create LoRA A matrix (rank x input_dim)
        // CRITICAL: Must load from pretrained weights, NOT initialize!
        let lora_a = {
            let weight = vb.get((config.rank, input_dim), "lora_A.weight")
                .map_err(|e| {
                    eprintln!("❌ FATAL: Failed to load lora_A.weight: {}", e);
                    eprintln!("   This means LoRA weights are not being loaded - inference will be incorrect!");
                    e
                })?;

            let bias = if config.use_bias {
                Some(vb.get(config.rank, "lora_A.bias")?)
            } else {
                None
            };

            Linear::new(weight, bias)
        };

        // Create LoRA B matrix (output_dim x rank) - Must load from pretrained weights
        let lora_b = {
            let weight = vb
                .get((output_dim, config.rank), "lora_B.weight")
                .map_err(|e| {
                    eprintln!("❌ FATAL: Failed to load lora_B.weight: {}", e);
                    e
                })?;

            let bias = if config.use_bias {
                Some(vb.get(output_dim, "lora_B.bias")?)
            } else {
                None
            };

            Linear::new(weight, bias)
        };

        // Create dropout layer
        let dropout = Dropout::new(config.dropout as f32);

        // Calculate scaling factor
        let scaling = config.alpha / config.rank as f64;

        Ok(Self {
            lora_a,
            lora_b,
            dropout,
            scaling,
            config: config.clone(),
        })
    }

    /// Forward pass through LoRA adapter
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        // x -> LoRA_A -> dropout -> LoRA_B -> scale
        let hidden = self.lora_a.forward(x)?;
        let hidden = self.dropout.forward(&hidden, train)?;
        let output = self.lora_b.forward(&hidden)?;

        // Apply scaling
        output.affine(self.scaling, 0.0)
    }

    /// Get LoRA configuration
    pub fn config(&self) -> &LoRAConfig {
        &self.config
    }

    /// Get scaling factor
    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    /// Merge LoRA weights into base model weights
    pub fn merge_weights(&self, base_weight: &Tensor) -> Result<Tensor> {
        // Get LoRA weights
        let lora_a_weight = self.lora_a.weight();
        let lora_b_weight = self.lora_b.weight();

        // Compute LoRA delta: B @ A * scaling
        let lora_delta = lora_b_weight.matmul(lora_a_weight)?;
        let scaled_delta = lora_delta.affine(self.scaling, 0.0)?;

        // Add to base weights
        base_weight.add(&scaled_delta)
    }

    /// Extract LoRA weights for saving
    pub fn extract_weights(&self) -> Result<LoRAWeights> {
        Ok(LoRAWeights {
            lora_a: self.lora_a.weight().clone(),
            lora_b: self.lora_b.weight().clone(),
            lora_a_bias: self.lora_a.bias().cloned(),
            lora_b_bias: self.lora_b.bias().cloned(),
            config: self.config.clone(),
        })
    }

    /// Load LoRA weights
    pub fn load_weights(&mut self, weights: &LoRAWeights) -> Result<()> {
        // Note: In a real implementation, we would need to update the Linear layers
        // This is a simplified version showing the interface
        self.config = weights.config.clone();
        self.scaling = self.config.alpha / self.config.rank as f64;
        Ok(())
    }

    /// Get parameter count
    pub fn parameter_count(&self) -> usize {
        let lora_a_params = self.config.rank * self.lora_a.weight().shape().dims()[1];
        let lora_b_params = self.lora_b.weight().shape().dims()[0] * self.config.rank;

        let bias_params = if self.config.use_bias {
            self.config.rank + self.lora_b.weight().shape().dims()[0]
        } else {
            0
        };

        lora_a_params + lora_b_params + bias_params
    }

    /// Calculate compression ratio compared to full fine-tuning
    pub fn compression_ratio(&self, full_model_params: usize) -> f64 {
        let lora_params = self.parameter_count();
        full_model_params as f64 / lora_params as f64
    }
}

/// LoRA weights for serialization
#[derive(Debug, Clone)]
pub struct LoRAWeights {
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub lora_a_bias: Option<Tensor>,
    pub lora_b_bias: Option<Tensor>,
    pub config: LoRAConfig,
}

/// Multi-layer LoRA adapter for transformer blocks
#[derive(Debug)]
pub struct MultiLayerLoRAAdapter {
    /// LoRA adapters for each layer
    adapters: HashMap<String, LoRAAdapter>,
    /// Global configuration
    config: LoRAConfig,
}

impl MultiLayerLoRAAdapter {
    /// Create multi-layer LoRA adapter
    pub fn new(
        layer_configs: HashMap<String, (usize, usize)>, // layer_name -> (input_dim, output_dim)
        config: &LoRAConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let mut adapters = HashMap::new();

        for (layer_name, (input_dim, output_dim)) in layer_configs {
            if config
                .target_modules
                .iter()
                .any(|target| layer_name.contains(target))
            {
                let layer_vb = vb.pp(&layer_name);
                let adapter = LoRAAdapter::new(input_dim, output_dim, config, layer_vb, device)?;
                adapters.insert(layer_name, adapter);
            }
        }

        Ok(Self {
            adapters,
            config: config.clone(),
        })
    }

    /// Forward pass through specific layer adapter
    pub fn forward_layer(
        &self,
        layer_name: &str,
        x: &Tensor,
        train: bool,
    ) -> Result<Option<Tensor>> {
        if let Some(adapter) = self.adapters.get(layer_name) {
            Ok(Some(adapter.forward(x, train)?))
        } else {
            Ok(None)
        }
    }

    /// Get all layer names with LoRA adapters
    pub fn layer_names(&self) -> Vec<&String> {
        self.adapters.keys().collect()
    }

    /// Get total parameter count across all layers
    pub fn total_parameter_count(&self) -> usize {
        self.adapters
            .values()
            .map(|adapter| adapter.parameter_count())
            .sum()
    }

    /// Merge all LoRA weights into base model
    pub fn merge_all_weights(
        &self,
        base_weights: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut merged_weights = HashMap::new();

        for (layer_name, base_weight) in base_weights {
            if let Some(adapter) = self.adapters.get(layer_name) {
                let merged_weight = adapter.merge_weights(base_weight)?;
                merged_weights.insert(layer_name.clone(), merged_weight);
            } else {
                merged_weights.insert(layer_name.clone(), base_weight.clone());
            }
        }

        Ok(merged_weights)
    }
}

/// LoRA adapter factory for creating adapters with different configurations
pub struct LoRAAdapterFactory;

impl LoRAAdapterFactory {
    /// Create adapter for BERT-like models
    pub fn create_bert_adapter(
        hidden_size: usize,
        config: &LoRAConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<HashMap<String, LoRAAdapter>> {
        let mut adapters = HashMap::new();

        // Create adapters for attention layers
        for module in &["query", "key", "value", "output"] {
            if config.target_modules.contains(&module.to_string()) {
                let adapter_vb = vb.pp(&format!("attention.{}", module));
                let adapter =
                    LoRAAdapter::new(hidden_size, hidden_size, config, adapter_vb, device)?;
                adapters.insert(module.to_string(), adapter);
            }
        }

        // Create adapters for feed-forward layers
        if config.target_modules.contains(&"intermediate".to_string()) {
            let adapter_vb = vb.pp("intermediate.dense");
            let adapter =
                LoRAAdapter::new(hidden_size, hidden_size * 4, config, adapter_vb, device)?;
            adapters.insert("intermediate".to_string(), adapter);
        }

        if config.target_modules.contains(&"output".to_string()) {
            let adapter_vb = vb.pp("output.dense");
            let adapter =
                LoRAAdapter::new(hidden_size * 4, hidden_size, config, adapter_vb, device)?;
            adapters.insert("output_dense".to_string(), adapter);
        }

        Ok(adapters)
    }

    /// Create adapter for classification head
    pub fn create_classification_adapter(
        input_size: usize,
        num_classes: usize,
        config: &LoRAConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<LoRAAdapter> {
        LoRAAdapter::new(input_size, num_classes, config, vb, device)
    }

    /// Create task-specific adapters for multi-task learning
    pub fn create_multitask_adapters(
        input_size: usize,
        task_configs: &HashMap<String, usize>, // task_name -> num_classes
        config: &LoRAConfig,
        vb: VarBuilder,
        device: &Device,
    ) -> Result<HashMap<String, LoRAAdapter>> {
        let mut adapters = HashMap::new();

        for (task_name, &num_classes) in task_configs {
            let task_vb = vb.pp(task_name);
            let adapter = LoRAAdapter::new(input_size, num_classes, config, task_vb, device)?;
            adapters.insert(task_name.clone(), adapter);
        }

        Ok(adapters)
    }
}

/// LoRA training utilities
pub struct LoRATrainingUtils;

impl LoRATrainingUtils {
    /// Calculate effective learning rate for LoRA parameters
    pub fn calculate_effective_lr(base_lr: f64, config: &LoRAConfig) -> f64 {
        // LoRA typically uses higher learning rates due to lower rank
        let rank_factor = (config.rank as f64 / 16.0).sqrt();
        let alpha_factor = config.alpha / 32.0;
        base_lr * rank_factor * alpha_factor
    }

    /// Estimate memory savings compared to full fine-tuning
    pub fn estimate_memory_savings(
        full_model_params: usize,
        lora_params: usize,
        batch_size: usize,
        sequence_length: usize,
    ) -> MemorySavings {
        let full_memory_mb =
            Self::estimate_training_memory(full_model_params, batch_size, sequence_length);
        let lora_memory_mb =
            Self::estimate_training_memory(lora_params, batch_size, sequence_length);

        let savings_mb = full_memory_mb - lora_memory_mb;
        let savings_ratio = savings_mb / full_memory_mb;

        MemorySavings {
            full_training_memory_mb: full_memory_mb,
            lora_training_memory_mb: lora_memory_mb,
            memory_savings_mb: savings_mb,
            memory_savings_ratio: savings_ratio,
        }
    }

    fn estimate_training_memory(params: usize, batch_size: usize, sequence_length: usize) -> f64 {
        // Simplified memory estimation for training
        let model_memory = params as f64 * 4.0 / 1024.0 / 1024.0; // 4 bytes per parameter
        let gradient_memory = model_memory; // Gradients same size as model
        let optimizer_memory = model_memory * 2.0; // Adam optimizer states
        let activation_memory =
            batch_size as f64 * sequence_length as f64 * 768.0 * 4.0 / 1024.0 / 1024.0;

        model_memory + gradient_memory + optimizer_memory + activation_memory
    }
}

/// Memory savings analysis
#[derive(Debug, Clone)]
pub struct MemorySavings {
    pub full_training_memory_mb: f64,
    pub lora_training_memory_mb: f64,
    pub memory_savings_mb: f64,
    pub memory_savings_ratio: f64,
}
