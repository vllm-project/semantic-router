//! # Model Architectures

#![allow(dead_code)]

pub mod embedding;
pub mod generative; // NEW: Generative/causal language models (Qwen3ForCausalLM)
pub mod lora;
pub mod prefix_cache; // NEW: Prefix caching for fixed prompts
pub mod traditional; // NEW: Embedding models (Qwen3, Gemma)

// Core model modules
pub mod config;
pub mod model_factory;
pub mod routing;
pub mod traits;
pub mod unified_interface;

// Re-export types from traits module
pub use traits::{
    EmbeddingPathSpecialization, // Embedding path specialization
    FineTuningType,
    LongContextEmbeddingCapable,
    ModelType,
    PoolingMethod,
    TaskType,
};

// Re-export unified interface (new simplified traits)
pub use unified_interface::{
    ConfigurableModel, CoreModel, ModelCapabilities, PathSpecialization, UnifiedModel,
};

// Re-export routing functionality
pub use routing::{DualPathRouter, ProcessingRequirements};

// Re-export config functionality
pub use config::PathSelectionStrategy;

// Re-export model factory functionality
pub use model_factory::{
    DualPathModel,
    EmbeddingOutput, // Embedding model output
    ModelFactory,
    ModelFactoryConfig,
    ModelOutput,
};

// Re-export embedding module pooling functions
pub use embedding::pooling::{cls_pool, last_token_pool, mean_pool};

// Test modules
#[cfg(test)]
pub mod model_factory_test;
#[cfg(test)]
pub mod routing_test;
#[cfg(test)]
#[cfg(test)]
pub mod unified_interface_test;
