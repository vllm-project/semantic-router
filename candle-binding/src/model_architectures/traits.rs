//! Model Architecture Traits and Type Definitions

use crate::model_architectures::unified_interface::CoreModel;
use anyhow::Result;
use candle_core::Tensor;
use std::fmt::Debug;

/// Model type enumeration for dual-path routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    /// Traditional fine-tuning path - stable and reliable
    Traditional,
    /// LoRA parameter-efficient path - high performance
    LoRA,
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
