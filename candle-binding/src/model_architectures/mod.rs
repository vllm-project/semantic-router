//! # Model Architectures

#![allow(dead_code)]

pub mod lora;
pub mod traditional;

// Core model modules
pub mod config;
pub mod model_factory;
pub mod routing;
pub mod traits;
pub mod unified_interface;

// Re-export types from traits module
pub use traits::{FineTuningType, ModelType, TaskType};

// Re-export unified interface (new simplified traits)
pub use unified_interface::{
    ConfigurableModel, CoreModel, ModelCapabilities, PathSpecialization, UnifiedModel,
};

// Re-export routing functionality
pub use routing::{DualPathRouter, ProcessingRequirements};

// Re-export config functionality
pub use config::PathSelectionStrategy;

// Re-export model factory functionality
pub use model_factory::{DualPathModel, ModelFactory, ModelFactoryConfig, ModelOutput};
