//! Embedding models using ONNX Runtime

pub mod mmbert_embedding;
pub mod pooling;

pub use mmbert_embedding::{MatryoshkaConfig, MmBertEmbeddingConfig, MmBertEmbeddingModel};
