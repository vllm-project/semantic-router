//! Model Architecture Traits and Type Definitions

use crate::model_architectures::unified_interface::CoreModel;
use anyhow::Result;
use candle_core::Tensor;
use std::fmt::Debug;

/// Model type enumeration for multi-path routing
///
/// Supports both classification models (Traditional, LoRA) and embedding models
/// (Qwen3Embedding, GemmaEmbedding, MmBertEmbedding) with distinct characteristics for intelligent routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Traditional BERT fine-tuning path - stable and reliable for classification
    Traditional,
    /// LoRA parameter-efficient path - high performance for classification
    LoRA,
    /// Qwen3 embedding model - high quality, up to 32K context length
    ///
    /// Characteristics:
    /// - Max sequence length: 32,768 tokens
    /// - Hidden size: 1024
    /// - Pooling: Last Token
    /// - Latency: ~30ms (512 tokens)
    /// - Best for: Long documents, high quality requirements
    Qwen3Embedding,
    /// Gemma embedding model - fast inference, up to 8K context length
    ///
    /// Characteristics:
    /// - Max sequence length: 8,192 tokens
    /// - Hidden size: 768
    /// - Pooling: Mean
    /// - Matryoshka support: 768/512/256/128
    /// - Latency: ~20ms (512 tokens)
    /// - Best for: Short to medium documents, latency-sensitive applications
    GemmaEmbedding,
    /// mmBERT embedding model - multilingual, 32K context, 2D Matryoshka
    ///
    /// Characteristics:
    /// - Max sequence length: 32,768 tokens
    /// - Hidden size: 768 (supports 64-768 via Matryoshka)
    /// - Pooling: Mean
    /// - Languages: 1800+ (via Glot500)
    /// - 2D Matryoshka: Dimension reduction + Layer reduction
    /// - Latency: 1.6-3.1× faster than BGE-M3 (FA2 advantage)
    /// - Best for: Multilingual RAG, long-document retrieval, edge deployment
    MmBertEmbedding,
}

/// Task type enumeration for multi-task processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Intent classification task
    Intent,
    /// PII (Personally Identifiable Information) detection
    PII,
    /// Security/Jailbreak detection
    Security,
    /// Basic classification task
    Classification,
    /// Token-level classification
    TokenClassification,
}

/// Fine-tuning type for traditional models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FineTuningType {
    /// Full model fine-tuning
    Full,
    /// Head-only fine-tuning
    HeadOnly,
    /// Layer-wise fine-tuning
    LayerWise,
}

/// LoRA-capable model trait - for high-performance parameter-efficient models
pub trait LoRACapable: CoreModel {
    /// Get LoRA rank (typically 16, 32, 64)
    fn get_lora_rank(&self) -> usize;

    /// Check if supports multi-task parallel processing
    fn supports_multi_task_parallel(&self) -> bool {
        true
    }

    /// Get available task adapters
    fn get_task_adapters(&self) -> Vec<TaskType>;
}

/// Traditional model trait - for stable, reliable fine-tuned models
pub trait TraditionalModel: CoreModel {
    /// Fine-tuning configuration
    type FineTuningConfig: Clone + Send + Sync + std::fmt::Debug;

    /// Get fine-tuning type used for this model
    fn get_fine_tuning_type(&self) -> FineTuningType;

    /// Check if supports single-task processing
    fn supports_single_task(&self) -> bool {
        true
    }

    /// Get model head configuration
    fn get_head_config(&self) -> Option<&Self::FineTuningConfig>;

    /// Check if model has classification head
    fn has_classification_head(&self) -> bool;

    /// Check if model has token classification head
    fn has_token_classification_head(&self) -> bool;

    /// Process single task with high reliability
    fn sequential_forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
        task: TaskType,
    ) -> Result<Self::Output, Self::Error>;

    /// Get optimal batch size for sequential processing
    fn optimal_sequential_batch_size(&self) -> usize {
        16 // Conservative batch size for stability
    }

    /// Estimate sequential processing time
    fn estimate_sequential_time(&self, batch_size: usize) -> f32 {
        // Traditional models: stable 4.567s baseline for standard batch
        let base_time = 4567.0; // milliseconds
        (batch_size as f32 / 4.0) * base_time
    }

    /// Get model stability score (0.0 to 1.0)
    fn stability_score(&self) -> f32 {
        0.98 // Traditional models are highly stable
    }

    /// Check if model is production-ready
    fn is_production_ready(&self) -> bool {
        true // Traditional models are always production-ready
    }

    /// Get backward compatibility version
    fn compatibility_version(&self) -> &str;
}

/// Pooling method enumeration for embedding models
///
/// Different embedding models use different pooling strategies to aggregate
/// token-level representations into a single sentence embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PoolingMethod {
    /// Mean pooling - average all token representations
    ///
    /// Used by: BERT, GemmaEmbedding
    /// Formula: mean(hidden_states * attention_mask) / sum(attention_mask)
    Mean,

    /// Last token pooling - use the last valid token
    ///
    /// Used by: Qwen3-Embedding
    /// Formula: hidden_states[batch_idx, sequence_lengths[batch_idx]]
    LastToken,

    /// CLS token pooling - use the first token ([CLS])
    ///
    /// Used by: Original BERT models
    /// Formula: hidden_states[:, 0, :]
    CLS,
}

/// Long-context embedding model trait
///
/// This trait defines the interface for embedding models that support
/// long sequences (up to 32K tokens for Qwen3) and advanced features like
/// Matryoshka representation learning.
///
/// ## Design Philosophy
/// - **Extensibility**: Supports both Qwen3 (32K, last-token pooling) and
///   GemmaEmbedding (2K, mean pooling, Matryoshka)
/// - **Performance**: Provides metadata for optimal batch sizing and parallel processing
/// - **Production-ready**: Clear error handling and configuration validation
///
/// ## Example
/// ```rust,ignore
/// impl LongContextEmbeddingCapable for Qwen3EmbeddingModel {
///     fn get_max_sequence_length(&self) -> usize { 32768 }
///     fn get_embedding_dimension(&self) -> usize { 768 }
///     fn get_pooling_method(&self) -> PoolingMethod { PoolingMethod::LastToken }
///     fn supports_matryoshka(&self) -> bool { false }
/// }
/// ```
pub trait LongContextEmbeddingCapable: CoreModel {
    /// Get maximum supported sequence length
    ///
    /// ## Return
    /// - Qwen3: 32768 tokens (32K context)
    /// - GemmaEmbedding: 2048 tokens (2K context)
    fn get_max_sequence_length(&self) -> usize;

    /// Get embedding dimension (output vector size)
    ///
    /// ## Return
    /// - Qwen3: 768 dimensions
    /// - GemmaEmbedding: 768 dimensions (full), 512/256/128 (Matryoshka)
    fn get_embedding_dimension(&self) -> usize;

    /// Get pooling method used by this model
    ///
    /// ## Return
    /// - Qwen3: `PoolingMethod::LastToken`
    /// - GemmaEmbedding: `PoolingMethod::Mean`
    fn get_pooling_method(&self) -> PoolingMethod;

    /// Check if model supports Matryoshka representation learning
    ///
    /// Matryoshka models can produce embeddings of multiple dimensions
    /// from a single forward pass by truncating the output vector.
    ///
    /// ## Return
    /// - `true`: Model supports Matryoshka (e.g., GemmaEmbedding)
    /// - `false`: Model uses fixed dimension (e.g., Qwen3)
    ///
    /// ## Default
    /// Returns `false` for models without Matryoshka support.
    fn supports_matryoshka(&self) -> bool {
        false
    }

    /// Get available Matryoshka dimensions
    ///
    /// ## Return
    /// - GemmaEmbedding: `vec![768, 512, 256, 128]`
    /// - Qwen3: `vec![768]` (only full dimension)
    ///
    /// ## Default
    /// Returns a single-element vector containing the full embedding dimension.
    fn get_matryoshka_dimensions(&self) -> Vec<usize> {
        vec![self.get_embedding_dimension()]
    }

    /// Check if model supports instruction-aware embeddings
    ///
    /// Instruction-aware models can take an instruction prefix to improve
    /// task-specific performance (e.g., "query:" or "passage:").
    ///
    /// ## Return
    /// - `true`: Model benefits from instruction prefixes (e.g., Qwen3)
    /// - `false`: Model does not use instructions
    ///
    /// ## Default
    /// Returns `false` for models without instruction support.
    fn supports_instruction_aware(&self) -> bool {
        false
    }

    /// Extract embeddings from hidden states using model-specific pooling
    ///
    /// This is the core method that implements the pooling strategy.
    ///
    /// ## Arguments
    /// - `hidden_states`: Token-level representations `[batch_size, seq_len, hidden_size]`
    /// - `attention_mask`: Valid token mask `[batch_size, seq_len]`
    /// - `target_dim`: Optional dimension for Matryoshka truncation
    ///
    /// ## Return
    /// - `Ok(Tensor)`: Sentence embeddings `[batch_size, target_dim or embedding_dim]`
    /// - `Err`: If pooling fails or target_dim is invalid
    ///
    /// ## Implementation Note
    /// This method will be implemented in the concrete model types (Qwen3, Gemma)
    /// using the pooling functions from `embedding::pooling` module.
    fn extract_embeddings(
        &self,
        hidden_states: &Tensor,
        attention_mask: &Tensor,
        target_dim: Option<usize>,
    ) -> Result<Tensor, Self::Error>;

    /// Get optimal batch size for embedding generation
    ///
    /// ## Return
    /// Recommended batch size based on model size and sequence length capacity.
    ///
    /// ## Default
    /// Returns 32 for balanced throughput and memory usage.
    fn optimal_embedding_batch_size(&self) -> usize {
        32
    }

    /// Check if model supports parallel batch processing
    ///
    /// ## Return
    /// - `true`: Model can process multiple batches in parallel
    /// - `false`: Model requires sequential processing
    ///
    /// ## Default
    /// Returns `true` for most embedding models.
    fn supports_parallel_batching(&self) -> bool {
        true
    }
}

/// Embedding path specialization trait
///
/// This trait provides metadata and optimization hints specifically for embedding models.
/// Unlike `PathSpecialization` (used for classification models with confidence scores),
/// this trait focuses on embedding-specific characteristics like dimension support,
/// pooling strategies, and sequence length handling.
///
/// ## Design Rationale
/// Embedding models do not produce confidence scores, so they cannot implement
/// the standard `PathSpecialization` trait. This trait provides an alternative
/// interface tailored to embedding generation requirements.
///
/// ## Example
/// ```rust,ignore
/// impl EmbeddingPathSpecialization for Qwen3EmbeddingModel {
///     fn supports_parallel(&self) -> bool { true }
///     fn optimal_batch_size(&self) -> usize { 32 }
/// }
/// ```
pub trait EmbeddingPathSpecialization: CoreModel {
    /// Check if model supports parallel batch processing
    ///
    /// ## Return
    /// - `true`: Model can process multiple batches concurrently (default)
    /// - `false`: Model requires sequential processing
    ///
    /// ## Use Case
    /// This helps the router decide whether to use parallel or sequential processing
    /// for batch embedding generation.
    fn supports_parallel(&self) -> bool {
        true
    }

    /// Get optimal batch size for this embedding model
    ///
    /// ## Return
    /// Recommended batch size that balances throughput and memory usage.
    ///
    /// ## Typical Values
    /// - Qwen3: 32 (long sequences consume more memory)
    /// - Gemma: 64 (shorter sequences allow larger batches)
    ///
    /// ## Default
    /// Returns 32 for balanced performance.
    fn optimal_batch_size(&self) -> usize {
        32
    }
}
