//! # Semantic Router - Modular Dual-Path Classification Engine
//!
//! A high-performance, modular text classification system built with Rust and Candle.
//! Features unified trait architecture, dual-path model support, and comprehensive
//! error handling with extensible design for future model integrations.

// Core modules
pub mod classifiers;
pub mod core;
pub mod model_architectures;
pub mod utils;

// C FFI interface
pub mod ffi;

// Test fixtures and utilities (only available in test builds)
#[cfg(test)]
pub mod test_fixtures;

// Public re-exports for backward compatibility
pub use core::similarity::BertSimilarity;
pub use model_architectures::traditional::bert::TraditionalBertClassifier as BertClassifier;

// Specific re-exports to avoid naming conflicts
pub use classifiers::unified::DualPathUnifiedClassifier;
pub use model_architectures::lora::{
    LoRAAdapter, LoRABertClassifier, LoRAConfig, LoRAMultiTaskResult,
};
pub use model_architectures::traditional::{base_model, TraditionalBertClassifier};

// C FFI functions re-exported
pub use ffi::*;
