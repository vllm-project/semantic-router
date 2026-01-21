//! # K-Nearest Neighbors (KNN) Selector
//!
//! Production implementation of KNN for model selection.
//! Matches the Go implementation in `selector.go`.
//!
//! ## Algorithm
//!
//! 1. Find K nearest neighbors by cosine similarity to the query embedding
//! 2. Vote with similarity, quality, and latency weighting
//! 3. Select the model with the highest weighted vote
//!
//! ## Weighting Formula (RouteLLM-inspired)
//!
//! For each neighbor:
//! - Base weight = cosine_similarity
//! - Quality factor = (1 + quality_score) if success, 0.5 otherwise
//! - Efficiency bonus = 1 / (1 + normalized_latency)
//!
//! Final weight = base_weight * quality_factor * efficiency_bonus

use std::collections::HashMap;
use std::sync::RwLock;

use serde::{Deserialize, Serialize};

use super::{
    cosine_similarity, ModelRef, ModelSelector, SelectionContext, SelectionResult, TrainingRecord,
};

/// Serializable KNN model data (matches Go KNNModelData)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNModelData {
    pub version: String,
    pub algorithm: String,
    pub training: Vec<TrainingRecord>,
    pub trained: bool,
    pub k: i32,
}

/// KNN Selector implementation
pub struct KNNSelector {
    /// Number of neighbors to consider
    k: usize,
    /// Training records
    training: RwLock<Vec<TrainingRecord>>,
    /// Maximum training size
    max_size: usize,
}

impl KNNSelector {
    /// Create a new KNN selector with specified K
    pub fn new(k: usize) -> Self {
        let k = if k == 0 { 3 } else { k };
        Self {
            k,
            training: RwLock::new(Vec::with_capacity(10000)),
            max_size: 10000,
        }
    }

    /// Load from JSON data
    pub fn load_from_json(&mut self, data: &[u8]) -> Result<(), String> {
        let model_data: KNNModelData =
            serde_json::from_slice(data).map_err(|e| format!("Failed to parse KNN JSON: {}", e))?;

        if model_data.k > 0 {
            self.k = model_data.k as usize;
        }

        let mut training = self.training.write().map_err(|e| e.to_string())?;
        *training = model_data.training;

        Ok(())
    }

    /// Save to JSON data
    pub fn save_to_json(&self) -> Result<Vec<u8>, String> {
        let training = self.training.read().map_err(|e| e.to_string())?;
        let model_data = KNNModelData {
            version: "1.0".to_string(),
            algorithm: "knn".to_string(),
            training: training.clone(),
            trained: !training.is_empty(),
            k: self.k as i32,
        };
        serde_json::to_vec(&model_data).map_err(|e| format!("Failed to serialize KNN: {}", e))
    }

    /// Build a map from model name to ref index
    fn build_model_index(refs: &[ModelRef]) -> HashMap<String, usize> {
        refs.iter()
            .enumerate()
            .map(|(i, r)| (r.get_name().to_string(), i))
            .collect()
    }
}

impl ModelSelector for KNNSelector {
    fn name(&self) -> &str {
        "knn"
    }

    fn train(&mut self, data: &[TrainingRecord]) -> Result<(), String> {
        let mut training = self.training.write().map_err(|e| e.to_string())?;
        training.extend(data.iter().cloned());

        // Keep only recent records
        if training.len() > self.max_size {
            let drain_count = training.len() - self.max_size;
            training.drain(0..drain_count);
        }

        Ok(())
    }

    fn training_count(&self) -> usize {
        self.training.read().map(|t| t.len()).unwrap_or(0)
    }

    fn is_trained(&self) -> bool {
        self.training_count() > 0
    }

    fn select(&self, ctx: &SelectionContext, refs: &[ModelRef]) -> Option<SelectionResult> {
        if refs.is_empty() {
            return None;
        }
        if refs.len() == 1 {
            return Some(SelectionResult {
                model_name: refs[0].get_name().to_string(),
                model_index: 0,
                score: 1.0,
            });
        }

        let training = self.training.read().ok()?;
        if training.is_empty() || ctx.query_embedding.is_empty() {
            // No training data or embedding, select first model
            return Some(SelectionResult {
                model_name: refs[0].get_name().to_string(),
                model_index: 0,
                score: 0.0,
            });
        }

        // Find K nearest neighbors
        struct Neighbor<'a> {
            record: &'a TrainingRecord,
            similarity: f64,
        }

        let mut neighbors: Vec<Neighbor> = training
            .iter()
            .filter(|r| !r.query_embedding.is_empty())
            .map(|record| {
                let sim = cosine_similarity(&ctx.query_embedding, &record.query_embedding);
                Neighbor {
                    record,
                    similarity: sim,
                }
            })
            .collect();

        // Sort by similarity (descending)
        neighbors.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top K
        let k = self.k.min(neighbors.len());
        neighbors.truncate(k);

        // Find max latency for normalization
        let max_latency_ms: f64 = neighbors
            .iter()
            .map(|n| n.record.response_latency_ms())
            .fold(1000.0, f64::max);

        // Vote with similarity, quality, and latency weighting
        let model_index = Self::build_model_index(refs);
        let mut votes: HashMap<String, f64> = HashMap::new();

        for neighbor in &neighbors {
            if model_index.contains_key(&neighbor.record.selected_model) {
                // Base weight from similarity
                let mut weight = neighbor.similarity;

                // Quality factor: success + quality score
                if neighbor.record.success {
                    weight *= 1.0 + neighbor.record.response_quality;
                } else {
                    weight *= 0.5;
                }

                // Latency factor: faster models get bonus (RouteLLM-style cost penalty)
                // efficiency = 1 / (1 + normalized_latency)
                // This gives: 0ms → 1.0, 500ms → 0.67, 1000ms → 0.5, 2000ms → 0.33
                let latency_ms = neighbor.record.response_latency_ms();
                let normalized_latency = latency_ms / max_latency_ms;
                let efficiency_bonus = 1.0 / (1.0 + normalized_latency);
                weight *= efficiency_bonus;

                *votes
                    .entry(neighbor.record.selected_model.clone())
                    .or_insert(0.0) += weight;
            }
        }

        // Find best model
        let best = votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));

        if let Some((model, vote)) = best {
            if let Some(&idx) = model_index.get(model) {
                return Some(SelectionResult {
                    model_name: model.clone(),
                    model_index: idx,
                    score: *vote,
                });
            }
        }

        // Fallback to first model
        Some(SelectionResult {
            model_name: refs[0].get_name().to_string(),
            model_index: 0,
            score: 0.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_records() -> Vec<TrainingRecord> {
        vec![
            TrainingRecord {
                query_text: "What is 2+2?".to_string(),
                query_embedding: vec![1.0, 0.0, 0.0],
                decision_name: "math".to_string(),
                selected_model: "model_a".to_string(),
                response_latency_ns: 100_000_000, // 100ms
                response_quality: 0.9,
                success: true,
                timestamp: 0,
            },
            TrainingRecord {
                query_text: "Calculate 3*4".to_string(),
                query_embedding: vec![0.9, 0.1, 0.0],
                decision_name: "math".to_string(),
                selected_model: "model_a".to_string(),
                response_latency_ns: 120_000_000, // 120ms
                response_quality: 0.85,
                success: true,
                timestamp: 1,
            },
            TrainingRecord {
                query_text: "Write a function".to_string(),
                query_embedding: vec![0.0, 1.0, 0.0],
                decision_name: "coding".to_string(),
                selected_model: "model_b".to_string(),
                response_latency_ns: 200_000_000, // 200ms
                response_quality: 0.95,
                success: true,
                timestamp: 2,
            },
        ]
    }

    #[test]
    fn test_knn_creation() {
        let knn = KNNSelector::new(5);
        assert_eq!(knn.name(), "knn");
        assert_eq!(knn.training_count(), 0);
        assert!(!knn.is_trained());
    }

    #[test]
    fn test_knn_training() {
        let mut knn = KNNSelector::new(3);
        let records = create_test_records();
        knn.train(&records).unwrap();
        assert_eq!(knn.training_count(), 3);
        assert!(knn.is_trained());
    }

    #[test]
    fn test_knn_selection() {
        let mut knn = KNNSelector::new(3);
        let records = create_test_records();
        knn.train(&records).unwrap();

        let refs = vec![
            ModelRef {
                model: "model_a".to_string(),
                lora_name: None,
            },
            ModelRef {
                model: "model_b".to_string(),
                lora_name: None,
            },
        ];

        // Query similar to model_a training data (math query)
        let ctx =
            SelectionContext::new("What is 5+5?", vec![0.95, 0.05, 0.0]).with_decision("math");

        let result = knn.select(&ctx, &refs);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.model_name, "model_a");

        // Query similar to model_b training data (coding query)
        let ctx = SelectionContext::new("Write a Python script", vec![0.0, 0.95, 0.05])
            .with_decision("coding");

        let result = knn.select(&ctx, &refs);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.model_name, "model_b");
    }

    #[test]
    fn test_knn_serialization() {
        let mut knn = KNNSelector::new(5);
        let records = create_test_records();
        knn.train(&records).unwrap();

        let json = knn.save_to_json().unwrap();

        let mut knn2 = KNNSelector::new(3);
        knn2.load_from_json(&json).unwrap();

        assert_eq!(knn2.k, 5);
        assert_eq!(knn2.training_count(), 3);
    }
}
