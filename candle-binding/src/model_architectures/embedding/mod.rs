//! Embedding Model Architectures
//!
//! This module contains implementations of long-context embedding models:
//! - **Qwen3-Embedding**: 32K context, last-token pooling, instruction-aware
//! - **GemmaEmbedding**: 2K context, mean pooling, Matryoshka representation
//! - **mmBERT-Embedding**: 32K context, mean pooling, 2D Matryoshka (dimension + layer)
//!
//! ## Module Structure
//! - `pooling`: Unified pooling implementations (mean, last-token, CLS)
//! - `qwen3_embedding`: Qwen3-Embedding-0.6B model implementation
//! - `gemma_embedding`: GemmaEmbedding-300M model implementation
//! - `mmbert_embedding`: mmBERT-Embed-32K-2D-Matryoshka model implementation
//! - `dense_layers`: Dense bottleneck for GemmaEmbedding quality improvement
//!
//! ## Design Principles
//! - **Modularity**: Shared pooling functions, model-specific configurations
//! - **Performance**: Optimized for 32K sequence length (Qwen3/mmBERT) and batch processing
//! - **Production-ready**: Comprehensive error handling and validation
//!
//! ## References
//! - Qwen3-Embedding: https://github.com/qwenlm/qwen3-embedding
//! - GemmaEmbedding: https://huggingface.co/google/embeddinggemma-300m
//! - mmBERT-Embedding: https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka
//! - TEI Qwen3: backends/candle/src/models/qwen3.rs
//! - TEI Gemma3: backends/candle/src/models/gemma3.rs

// Pooling module - shared pooling implementations
pub mod pooling;

// Qwen3-Embedding model
pub mod qwen3_embedding;

// Continuous batching for embeddings
pub mod continuous_batch_scheduler;

// Qwen3 with continuous batching
pub mod qwen3_batched;

// GemmaEmbedding model
pub mod gemma_embedding;

// Dense bottleneck for GemmaEmbedding
pub mod dense_layers;

// Gemma3 Transformer backbone for GemmaEmbedding
pub mod gemma3_model;

// mmBERT Embedding model (32K context, 2D Matryoshka)
pub mod mmbert_embedding;

// Re-exports for convenience
pub use dense_layers::{BottleneckDenseNet, DenseActivation, DenseLayer};
pub use gemma3_model::{
    Gemma3Attention, Gemma3Layer, Gemma3MLP, Gemma3Model, RmsNorm as Gemma3RmsNorm,
    RotaryEmbeddingCache as Gemma3RoPE,
};
pub use pooling::{cls_pool, last_token_pool, mean_pool};

// Model-specific re-exports
pub use qwen3_embedding::Qwen3EmbeddingConfig;
pub use qwen3_embedding::Qwen3EmbeddingModel;

// Continuous batching re-exports
pub use continuous_batch_scheduler::{ContinuousBatchConfig, ContinuousBatchScheduler};
pub use qwen3_batched::Qwen3EmbeddingModelBatched;

// GemmaEmbedding re-exports
pub use gemma_embedding::AttentionLayerType;
pub use gemma_embedding::GemmaEmbeddingConfig;
pub use gemma_embedding::GemmaEmbeddingModel;

// mmBERT Embedding re-exports
pub use mmbert_embedding::MatryoshkaConfig;
pub use mmbert_embedding::MmBertEmbeddingConfig;
pub use mmbert_embedding::MmBertEmbeddingModel;

// Pooling tests
#[cfg(test)]
mod pooling_test;

// Qwen3-Embedding tests
#[cfg(test)]
mod qwen3_embedding_test;

// GemmaEmbedding tests
#[cfg(test)]
mod gemma_embedding_test;

// Dense bottleneck tests
#[cfg(test)]
mod dense_layers_test;

// Gemma3 model tests
#[cfg(test)]
mod gemma3_model_test;
