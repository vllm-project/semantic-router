//! ONNX Runtime Semantic Router Library
//!
//! This library provides ONNX Runtime-based embedding generation with 2D Matryoshka support.
//! It supports AMD GPU (ROCm), NVIDIA GPU (CUDA), and CPU inference via ONNX Runtime.
//!
//! ## Features
//! - **AMD GPU Support**: Via ROCm execution provider
//! - **NVIDIA GPU Support**: Via CUDA execution provider  
//! - **2D Matryoshka**: Layer early exit + dimension truncation for flexible performance/quality tradeoffs
//! - **Multilingual**: 1800+ languages via mmBERT base

pub mod core;
pub mod ffi;
pub mod model_architectures;

// Re-export commonly used types
pub use core::unified_error::{UnifiedError, UnifiedResult};
pub use model_architectures::embedding::mmbert_embedding::{
    MatryoshkaConfig, MmBertEmbeddingConfig, MmBertEmbeddingModel,
};
