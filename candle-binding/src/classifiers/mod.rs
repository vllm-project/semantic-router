//! # Classification Systems - Dual-Path Classifier Implementation

#![allow(dead_code)]

pub mod lora;
pub mod traditional;

pub mod unified;

// Re-export key types from unified module
pub use unified::{DualPathUnifiedClassifier, EmbeddingRequirements, UnifiedClassifierError};

/// Classification task types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassificationTask {
    /// Intent classification
    Intent,
    /// PII (Personally Identifiable Information) detection
    PII,
    /// Security/Jailbreak detection
    Security,
}

/// Classification result with dual-path support
#[derive(Debug, Clone)]
pub struct DualPathResult {
    /// Which path was used for classification
    pub path_used: crate::model_architectures::ModelType,
    /// Task-specific results
    pub results: Vec<TaskResult>,
    /// Overall confidence
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
}

/// Individual task result
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Task type
    pub task: ClassificationTask,
    /// Classification result
    pub result: String,
    /// Confidence score
    pub confidence: f32,
}

// Test modules
#[cfg(test)]
pub mod unified_test;
