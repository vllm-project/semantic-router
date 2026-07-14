//! Embedding models using ONNX Runtime

pub mod mmbert_embedding;
pub mod multimodal_embedding;
pub(crate) mod multimodal_output;
pub mod pooling;
pub(crate) mod tokenizer_contract;

pub use mmbert_embedding::{MatryoshkaConfig, MmBertEmbeddingConfig, MmBertEmbeddingModel};
pub use multimodal_embedding::{MultiModalConfig, MultiModalEmbeddingModel};
