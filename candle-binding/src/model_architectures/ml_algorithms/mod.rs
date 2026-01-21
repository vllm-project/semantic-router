//! # ML-based Model Selection Algorithms
//!
//! This module implements machine learning algorithms for intelligent model selection,
//! ported from Go to Rust with identical behavior as specified in PR 1119.
//!
//! ## Algorithms Implemented
//!
//! - **KNN (K-Nearest Neighbors)**: Selects models based on similarity-weighted voting
//!   from historical query-model performance data.
//!
//! - **KMeans**: Clusters queries and routes them to the best-performing model per cluster.
//!   Implements the Avengers-Pro performance-efficiency scoring (arXiv:2508.12631).
//!
//! - **SVM (Support Vector Machine)**: One-vs-all classification using kernel functions
//!   (RBF, linear, polynomial).
//!
//! ## Reference Papers
//!
//! - Avengers-Pro (arXiv:2508.12631): Performance-efficiency optimized routing via clustering
//! - FusionFactory (arXiv:2507.10540): Multi-LLM fusion framework
//!
//! ## Algorithm Alignment
//!
//! All implementations match the Go implementations in `src/semantic-router/pkg/modelselection/`
//! to ensure consistent behavior across languages.

pub mod kmeans;
pub mod knn;
pub mod svm;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod go_model_tests;

// Re-exports
pub use kmeans::{KMeansModelData, KMeansSelector};
pub use knn::{KNNModelData, KNNSelector};
pub use svm::{SVMModelData, SVMSelector};

use serde::{Deserialize, Serialize};

/// Training record for model selection algorithms
/// Matches Go TrainingRecord struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRecord {
    /// Original query text
    #[serde(default)]
    pub query_text: String,
    /// Embedding of the query
    pub query_embedding: Vec<f64>,
    /// Decision/category name for category-specific training
    #[serde(default)]
    pub decision_name: String,
    /// Model that was selected
    pub selected_model: String,
    /// Response latency in nanoseconds
    pub response_latency_ns: i64,
    /// Response quality score (0-1)
    pub response_quality: f64,
    /// Whether the request was successful
    pub success: bool,
    /// Unix timestamp when this record was created
    pub timestamp: i64,
}

impl TrainingRecord {
    /// Get response latency in milliseconds
    pub fn response_latency_ms(&self) -> f64 {
        self.response_latency_ns as f64 / 1_000_000.0
    }
}

/// Model selection result
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected model name
    pub model_name: String,
    /// Model index in refs array
    pub model_index: usize,
    /// Confidence/vote score
    pub score: f64,
}

/// Model reference for selection
#[derive(Debug, Clone)]
pub struct ModelRef {
    /// Primary model name
    pub model: String,
    /// Optional LoRA adapter name
    pub lora_name: Option<String>,
}

impl ModelRef {
    /// Get the effective model name (LoRA name if present, otherwise model name)
    pub fn get_name(&self) -> &str {
        self.lora_name.as_deref().unwrap_or(&self.model)
    }
}

/// Selection context for model selection
/// Matches Go's SelectionContext in pkg/selection/selector.go
#[derive(Debug, Clone)]
pub struct SelectionContext {
    /// Raw query text - the user's input query
    pub query: String,
    /// Query embedding vector (precomputed)
    pub query_embedding: Vec<f64>,
    /// Decision/category name for category-specific selection
    pub decision_name: String,
    /// Cost weight (0.0-1.0): higher values prefer cheaper models
    pub cost_weight: f64,
    /// Quality weight (0.0-1.0): higher values prefer higher-quality models
    pub quality_weight: f64,
}

impl Default for SelectionContext {
    fn default() -> Self {
        Self {
            query: String::new(),
            query_embedding: Vec::new(),
            decision_name: String::new(),
            cost_weight: 0.3,    // Default: 30% cost consideration
            quality_weight: 0.7, // Default: 70% quality consideration
        }
    }
}

impl SelectionContext {
    /// Create a new SelectionContext with query and embedding
    pub fn new(query: impl Into<String>, embedding: Vec<f64>) -> Self {
        Self {
            query: query.into(),
            query_embedding: embedding,
            ..Default::default()
        }
    }

    /// Set the decision/category name
    pub fn with_decision(mut self, decision_name: impl Into<String>) -> Self {
        self.decision_name = decision_name.into();
        self
    }

    /// Set cost/quality weights
    pub fn with_weights(mut self, cost_weight: f64, quality_weight: f64) -> Self {
        self.cost_weight = cost_weight.clamp(0.0, 1.0);
        self.quality_weight = quality_weight.clamp(0.0, 1.0);
        self
    }
}

/// Trait for model selectors
pub trait ModelSelector: Send + Sync {
    /// Algorithm name
    fn name(&self) -> &str;

    /// Select the best model from refs based on selection context
    fn select(&self, ctx: &SelectionContext, refs: &[ModelRef]) -> Option<SelectionResult>;

    /// Train/update the selector with new training data
    fn train(&mut self, data: &[TrainingRecord]) -> Result<(), String>;

    /// Get training data count
    fn training_count(&self) -> usize;

    /// Check if the selector is trained
    fn is_trained(&self) -> bool;
}

// =============================================================================
// Numerical Utilities (matches Go gonum utilities)
// =============================================================================

/// Compute cosine similarity between two vectors
/// Formula: dot(a, b) / (||a|| * ||b||)
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute Euclidean distance between two vectors
/// Formula: sqrt(sum((a_i - b_i)^2))
pub fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return f64::MAX;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Normalize a vector to unit length (L2 normalization)
pub fn normalize_l2(v: &[f64]) -> Vec<f64> {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm == 0.0 {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Dot product of two vectors
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Softmax with numerical stability
pub fn softmax(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return vec![];
    }

    // Find max for numerical stability
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let exp_vals: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exp_vals.iter().sum();

    if sum > 0.0 {
        exp_vals.iter().map(|v| v / sum).collect()
    } else {
        exp_vals
    }
}

// Note: Integration tests are in the external tests.rs file
