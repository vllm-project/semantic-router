//! Dual-Path Configuration System
//!
//! This module provides unified configuration management for both Traditional and LoRA paths.
//! It supports intelligent defaults, validation, and path-specific optimizations.

use crate::core::{config_errors, UnifiedError};
use crate::model_architectures::traits::ModelType;
use crate::validation_error;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Unified configuration for dual-path architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualPathConfig {
    /// Traditional model configuration
    pub traditional: TraditionalConfig,
    /// LoRA model configuration
    pub lora: LoRAConfig,
    /// Embedding model configuration
    pub embedding: EmbeddingConfig,
    /// Global settings
    pub global: GlobalConfig,
}

/// Traditional model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraditionalConfig {
    /// Model path
    pub model_path: PathBuf,
    /// Use CPU instead of GPU
    pub use_cpu: bool,
    /// Batch size for traditional processing
    pub batch_size: usize,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Maximum sequence length
    pub max_sequence_length: usize,
}

/// LoRA model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    /// Base model path
    pub base_model_path: PathBuf,
    /// LoRA adapter paths for different tasks
    pub adapter_paths: LoRAAdapterPaths,
    /// LoRA rank
    pub rank: usize,
    /// LoRA alpha
    pub alpha: f32,
    /// LoRA dropout
    pub dropout: f32,
    /// Parallel batch size
    pub parallel_batch_size: usize,
    /// High confidence threshold (0.99+)
    pub confidence_threshold: f32,
}

/// LoRA adapter paths for different tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAAdapterPaths {
    /// Intent classification adapter
    pub intent: Option<PathBuf>,
    /// PII detection adapter
    pub pii: Option<PathBuf>,
    /// Security detection adapter
    pub security: Option<PathBuf>,
}

/// Embedding model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Batch size for Qwen3 embedding model
    pub qwen3_batch_size: usize,
    /// Batch size for Gemma embedding model
    pub gemma_batch_size: usize,
    /// Maximum sequence length for embeddings
    pub max_sequence_length: usize,
    /// Enable performance monitoring for embedding models
    pub enable_performance_tracking: bool,
}

/// Global configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalConfig {
    /// Device preference
    pub device_preference: DevicePreference,
    /// Path selection strategy
    pub path_selection: PathSelectionStrategy,
    /// Performance optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

/// Device preference for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DevicePreference {
    /// Prefer GPU if available
    GPU,
    /// Force CPU usage
    CPU,
    /// Automatic selection
    Auto,
}

/// Path selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathSelectionStrategy {
    /// Always use LoRA path
    AlwaysLoRA,
    /// Always use Traditional path
    AlwaysTraditional,
    /// Automatic selection based on requirements
    Automatic,
    /// Performance-based selection
    PerformanceBased,
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// Conservative optimization
    Conservative,
    /// Balanced optimization
    Balanced,
    /// Aggressive optimization
    Aggressive,
}

/// Processing priority for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingPriority {
    /// Minimize latency
    Latency,
    /// Maximize throughput
    Throughput,
    /// Maximize accuracy
    Accuracy,
    /// Balanced approach
    Balanced,
}

impl Default for DualPathConfig {
    fn default() -> Self {
        Self {
            traditional: TraditionalConfig::default(),
            lora: LoRAConfig::default(),
            embedding: EmbeddingConfig::default(),
            global: GlobalConfig::default(),
        }
    }
}

impl Default for TraditionalConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("models/traditional/modernbert"),
            use_cpu: false,
            batch_size: 16,
            confidence_threshold: 0.0, // Will be set dynamically based on model performance
            max_sequence_length: 512,
        }
    }
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            base_model_path: PathBuf::from("models/lora/base"),
            adapter_paths: LoRAAdapterPaths::default(),
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            parallel_batch_size: 32,
            confidence_threshold: 0.0, // Will be set dynamically based on model performance
        }
    }
}

impl Default for LoRAAdapterPaths {
    fn default() -> Self {
        Self {
            intent: Some(PathBuf::from("models/lora/adapters/intent")),
            pii: Some(PathBuf::from("models/lora/adapters/pii")),
            security: Some(PathBuf::from("models/lora/adapters/security")),
        }
    }
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            // Qwen3: larger model, smaller batch size for memory efficiency
            qwen3_batch_size: 8,
            // Gemma: smaller model, can handle larger batches
            gemma_batch_size: 16,
            // Maximum sequence length: 32K for Qwen3, 8K for Gemma
            max_sequence_length: 32768,
            // Disable performance tracking by default to reduce log noise
            enable_performance_tracking: false,
        }
    }
}

impl Default for GlobalConfig {
    fn default() -> Self {
        Self {
            device_preference: DevicePreference::Auto,
            path_selection: PathSelectionStrategy::Automatic,
            optimization_level: OptimizationLevel::Balanced,
            enable_monitoring: true,
        }
    }
}

impl DualPathConfig {
    /// Create configuration for specific model type
    pub fn for_model_type(model_type: ModelType) -> Self {
        let mut config = Self::default();
        match model_type {
            ModelType::Traditional => {
                config.global.path_selection = PathSelectionStrategy::AlwaysTraditional;
            }
            ModelType::LoRA => {
                config.global.path_selection = PathSelectionStrategy::AlwaysLoRA;
            }
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding => {
                //   Embedding models use automatic selection
                // Selection is handled by UnifiedClassifier::select_embedding_model()
                config.global.path_selection = PathSelectionStrategy::Automatic;
            }
        }
        config
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), UnifiedError> {
        // Validate traditional config
        if !self.traditional.model_path.exists() {
            return Err(config_errors::file_not_found(&format!(
                "Traditional model path does not exist: {:?}",
                self.traditional.model_path
            )));
        }

        // Validate LoRA config
        if !self.lora.base_model_path.exists() {
            return Err(config_errors::file_not_found(&format!(
                "LoRA base model path does not exist: {:?}",
                self.lora.base_model_path
            )));
        }

        // Validate LoRA parameters
        if self.lora.rank == 0 {
            return Err(validation_error!("lora_rank", "greater than 0", "0"));
        }

        if self.lora.alpha <= 0.0 {
            return Err(validation_error!(
                "lora_alpha",
                "positive value",
                &self.lora.alpha.to_string()
            ));
        }

        if self.lora.dropout < 0.0 || self.lora.dropout > 1.0 {
            return Err(validation_error!(
                "lora_dropout",
                "between 0.0 and 1.0",
                &self.lora.dropout.to_string()
            ));
        }

        Ok(())
    }

    /// Get optimal batch size for given model type
    pub fn optimal_batch_size(&self, model_type: ModelType) -> usize {
        match model_type {
            ModelType::Traditional => self.traditional.batch_size,
            ModelType::LoRA => self.lora.parallel_batch_size,
            ModelType::Qwen3Embedding => self.embedding.qwen3_batch_size,
            ModelType::GemmaEmbedding => self.embedding.gemma_batch_size,
        }
    }

    /// Get confidence threshold for given model type
    pub fn confidence_threshold(&self, model_type: ModelType) -> f32 {
        match model_type {
            ModelType::Traditional => self.traditional.confidence_threshold,
            ModelType::LoRA => self.lora.confidence_threshold,
            ModelType::Qwen3Embedding | ModelType::GemmaEmbedding => {
                //  Embedding models don't produce classification confidence
                // Embeddings are vector representations, not classification predictions
                // Return 0.0 as embeddings don't have confidence scores
                0.0
            }
        }
    }
}

/// Configuration builder for fluent API
pub struct ConfigBuilder {
    config: DualPathConfig,
}

impl ConfigBuilder {
    /// Create new builder with defaults
    pub fn new() -> Self {
        Self {
            config: DualPathConfig::default(),
        }
    }

    /// Set traditional model path
    pub fn traditional_model_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.traditional.model_path = path.into();
        self
    }

    /// Set LoRA base model path
    pub fn lora_base_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.config.lora.base_model_path = path.into();
        self
    }

    /// Set LoRA rank
    pub fn lora_rank(mut self, rank: usize) -> Self {
        self.config.lora.rank = rank;
        self
    }

    /// Set device preference
    pub fn device_preference(mut self, preference: DevicePreference) -> Self {
        self.config.global.device_preference = preference;
        self
    }

    /// Set path selection strategy
    pub fn path_selection(mut self, strategy: PathSelectionStrategy) -> Self {
        self.config.global.path_selection = strategy;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<DualPathConfig, UnifiedError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

impl Default for ConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}
