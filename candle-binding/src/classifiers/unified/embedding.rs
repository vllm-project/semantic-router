//! Embedding-model selection and performance accounting.

use super::{DualPathUnifiedClassifier, UnifiedClassifierError};
use crate::model_architectures::traits::ModelType;

const MAX_SEQUENCE_LENGTH: usize = 32_768;
const SHORT_SEQUENCE_LIMIT: usize = 512;
const MEDIUM_SEQUENCE_LIMIT: usize = 2_048;
const HIGH_PRIORITY: f32 = 0.7;
const MATRYOSHKA_LATENCY_PRIORITY: f32 = 0.5;
const FULL_EMBEDDING_DIMENSION: usize = 768;

/// Requirements used to select an embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingRequirements {
    /// Input length in tokens. Qwen3 handles the longest inputs.
    pub sequence_length: usize,
    /// Preference for embedding quality in the inclusive range 0.0-1.0.
    pub quality_priority: f32,
    /// Preference for low latency in the inclusive range 0.0-1.0.
    pub latency_priority: f32,
    /// Optional Matryoshka output dimension.
    pub target_dimension: Option<usize>,
}

impl DualPathUnifiedClassifier {
    /// Record latency and sequence-length statistics for an embedding request.
    pub fn update_embedding_stats(
        &self,
        model_type: ModelType,
        processing_time_ms: f32,
        sequence_length: usize,
    ) {
        let mut stats = self.performance_stats.write();
        stats.embedding_total_requests += 1;

        match model_type {
            ModelType::Qwen3Embedding => {
                stats.qwen3_usage += 1;
                stats.qwen3_total_time_ms += processing_time_ms;
                stats.avg_qwen3_sequence_length = incremental_average(
                    stats.avg_qwen3_sequence_length,
                    stats.qwen3_usage,
                    sequence_length,
                );
            }
            ModelType::GemmaEmbedding => {
                stats.gemma_usage += 1;
                stats.gemma_total_time_ms += processing_time_ms;
                stats.avg_gemma_sequence_length = incremental_average(
                    stats.avg_gemma_sequence_length,
                    stats.gemma_usage,
                    sequence_length,
                );
            }
            _ => {}
        }
    }

    /// Select the preferred embedding model for the supplied requirements.
    ///
    /// Model availability and fallback are execution concerns handled by the
    /// embedding FFI after this deterministic policy decision.
    pub fn select_embedding_model(
        &self,
        requirements: &EmbeddingRequirements,
    ) -> Result<ModelType, UnifiedClassifierError> {
        validate_sequence_length(requirements.sequence_length)?;
        let selected = preferred_embedding_model(requirements);

        if self.config.embedding.enable_performance_tracking {
            println!(
                "[Embedding Router] Model {:?} selected for seq_len={} (quality={:.2}, latency={:.2}, target_dim={:?})",
                selected,
                requirements.sequence_length,
                requirements.quality_priority,
                requirements.latency_priority,
                requirements.target_dimension
            );
        }

        Ok(selected)
    }
}

fn validate_sequence_length(sequence_length: usize) -> Result<(), UnifiedClassifierError> {
    if sequence_length > MAX_SEQUENCE_LENGTH {
        return Err(UnifiedClassifierError::ProcessingError(format!(
            "Sequence length {} exceeds maximum supported length of 32K tokens. Consider splitting the input into smaller chunks.",
            sequence_length
        )));
    }
    Ok(())
}

fn preferred_embedding_model(requirements: &EmbeddingRequirements) -> ModelType {
    let preferred = if requirements.sequence_length <= SHORT_SEQUENCE_LIMIT {
        if requirements.quality_priority > HIGH_PRIORITY {
            ModelType::Qwen3Embedding
        } else if requirements.latency_priority > HIGH_PRIORITY {
            ModelType::GemmaEmbedding
        } else {
            ModelType::Qwen3Embedding
        }
    } else if requirements.sequence_length <= MEDIUM_SEQUENCE_LIMIT {
        ModelType::GemmaEmbedding
    } else {
        ModelType::Qwen3Embedding
    };

    match requirements.target_dimension {
        Some(dimension)
            if dimension < FULL_EMBEDDING_DIMENSION
                && requirements.latency_priority > MATRYOSHKA_LATENCY_PRIORITY =>
        {
            ModelType::GemmaEmbedding
        }
        _ => preferred,
    }
}

fn incremental_average(current: f32, count: u64, sequence_length: usize) -> f32 {
    let count = count as f32;
    (current * (count - 1.0) + sequence_length as f32) / count
}
