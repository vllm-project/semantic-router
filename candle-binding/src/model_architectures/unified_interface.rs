//! Unified Model Interface - Simplified Trait Architecture
//!
//! This module provides simplified, unified

use crate::model_architectures::traits::ModelType;
use candle_core::{Device, Tensor};
use std::error::Error;
use std::fmt::Debug;

/// Core model interface
///
/// This trait contains only the essential methods that every model must implement.
/// It reduces complexity by focusing on the core functionality needed for inference.
pub trait CoreModel: Send + Sync + Debug {
    /// Configuration type for this model
    type Config: Clone + Send + Sync + Debug;

    /// Error type for this model
    type Error: Error + Send + Sync + 'static;

    /// Output type for forward pass
    type Output: Send + Sync + Debug;

    /// Get the model type (Traditional or LoRA)
    fn model_type(&self) -> ModelType;

    /// Forward pass through the model
    ///
    /// This is the core inference method that all models must implement.
    /// It takes tokenized input and attention mask, returns model-specific output.
    fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Self::Output, Self::Error>;

    /// Get model configuration
    ///
    /// Provides access to the model's configuration for introspection
    /// and compatibility checks.
    fn get_config(&self) -> &Self::Config;
}

/// Path specialization trait
///
/// This trait provides path-specific optimizations and characteristics.
/// It consolidates the functionality from both Traditional and LoRA specific traits.
pub trait PathSpecialization: CoreModel {
    /// Check if model supports parallel processing
    ///
    /// - Traditional models: typically false (sequential processing)
    /// - LoRA models: typically true (parallel multi-task processing)
    fn supports_parallel(&self) -> bool;

    /// Get confidence threshold for this model type
    ///
    /// Returns the minimum confidence score for reliable predictions.
    /// Different model types may have different reliability characteristics.
    fn get_confidence_threshold(&self) -> f32;

    /// Get optimal batch size for this model
    ///
    /// Returns the recommended batch size for optimal performance.
    /// Takes into account memory constraints and processing characteristics.
    fn optimal_batch_size(&self) -> usize;
}

/// Optional trait for models that support loading from configuration
///
/// This trait is separate from CoreModel to allow for models that are
/// created through other means (e.g., factory patterns, builders).
pub trait ConfigurableModel: CoreModel {
    /// Load model from configuration and device
    ///
    /// This method creates a new instance of the model from configuration.
    /// It's optional because some models may use different construction patterns.
    fn load(config: &Self::Config, device: &Device) -> Result<Self, Self::Error>
    where
        Self: Sized;
}

/// Convenience trait that combines all unified interface traits
///
/// This trait provides a single bound for code that needs the full
/// unified interface functionality.
pub trait UnifiedModel: CoreModel + PathSpecialization + ConfigurableModel {}

// Blanket implementation for any type that implements all three traits
impl<T> UnifiedModel for T where T: CoreModel + PathSpecialization + ConfigurableModel {}

/// Model capability flags for runtime introspection
///
/// This struct provides a way to query model capabilities at runtime
/// without needing to know the specific model type.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCapabilities {
    /// Model type (Traditional or LoRA)
    pub model_type: ModelType,

    /// Supports parallel processing
    pub supports_parallel: bool,

    /// Confidence threshold
    pub confidence_threshold: f32,

    /// Optimal batch size
    pub optimal_batch_size: usize,

    /// Supports configuration-based loading
    pub supports_config_loading: bool,
}

impl ModelCapabilities {
    /// Create capabilities from a model instance
    pub fn from_model<M: PathSpecialization>(model: &M) -> Self {
        Self {
            model_type: model.model_type(),
            supports_parallel: model.supports_parallel(),
            confidence_threshold: model.get_confidence_threshold(),
            optimal_batch_size: model.optimal_batch_size(),
            supports_config_loading: false, // Will be true if model also implements ConfigurableModel
        }
    }

    /// Create capabilities from a configurable model instance
    pub fn from_configurable_model<M: UnifiedModel>(model: &M) -> Self {
        Self {
            model_type: model.model_type(),
            supports_parallel: model.supports_parallel(),
            confidence_threshold: model.get_confidence_threshold(),
            optimal_batch_size: model.optimal_batch_size(),
            supports_config_loading: true,
        }
    }
}
