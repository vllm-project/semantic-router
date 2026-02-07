//! ONNX Runtime Semantic Router Library
//!
//! This library provides ONNX Runtime-based embedding and classification with mmBERT-32K-YaRN.
//! It supports AMD GPU (ROCm), NVIDIA GPU (CUDA), OpenVINO (Intel), and CPU inference.
//!
//! ## Features
//! - **AMD GPU Support**: Via ROCm/MIGraphX execution provider (~2ms latency)
//! - **NVIDIA GPU Support**: Via CUDA execution provider
//! - **Intel CPU Support**: Via OpenVINO execution provider (~22ms latency)
//! - **CPU Support**: Via default ONNX Runtime (~41ms latency)
//! - **2D Matryoshka**: Layer early exit + dimension truncation for embeddings
//! - **Multilingual**: 1800+ languages via mmBERT base
//! - **Classification**: Intent, Jailbreak, Feedback, Factcheck, PII detection

pub mod core;
pub mod ffi;
pub mod model_architectures;

// Re-export commonly used types
pub use core::unified_error::{UnifiedError, UnifiedResult};

// Embedding types
pub use model_architectures::embedding::mmbert_embedding::{
    MatryoshkaConfig, MmBertEmbeddingConfig, MmBertEmbeddingModel,
};

// Classification types
pub use model_architectures::classification::{
    ClassificationResult, ClassifierExecutionProvider, DetectedEntity,
    MmBertClassifierConfig, MmBertSequenceClassifier, MmBertTokenClassifier,
    TokenClassificationResult,
};
