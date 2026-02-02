//! KNN (K-Nearest Neighbors) implementation using Linfa
//!
//! Inference-only implementation. Training is done in Python (src/training/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.
//!
//! Uses quality-weighted voting: neighbors with higher quality scores have more influence.
//! This ensures we select models that PERFORM BEST, not just which was selected.

use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// KNN Selector using Linfa's Ball Tree for efficient nearest neighbor search
/// Implements query-level fusion for LLM routing as per FusionFactory
#[derive(Debug)]
pub struct KNNSelector {
    k: usize,
    embeddings: Option<Array2<f64>>,
    labels: Vec<String>,
    qualities: Vec<f64>,   // Quality score for each training sample
    latencies: Vec<i64>,   // Latency in nanoseconds for each sample
    trained: bool,
}

/// Model data for JSON serialization
#[derive(Debug, Serialize, Deserialize)]
pub struct KNNModelData {
    pub algorithm: String,
    pub trained: bool,
    pub k: usize,
    pub embeddings: Vec<Vec<f64>>,
    pub labels: Vec<String>,
    #[serde(default)]
    pub qualities: Vec<f64>,
    #[serde(default)]
    pub latencies: Vec<i64>,
}

impl KNNSelector {
    /// Create a new KNN selector with specified k
    pub fn new(k: usize) -> Self {
        Self {
            k,
            embeddings: None,
            labels: Vec::new(),
            qualities: Vec::new(),
            latencies: Vec::new(),
            trained: false,
        }
    }

    /// Select the best model for a query embedding using Linfa Ball Tree
    /// Uses QUALITY-WEIGHTED voting: models with higher quality get more vote weight
    /// This ensures we pick the model that PERFORMS BEST, not just which was selected
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        if !self.trained {
            return Err("Model not trained".to_string());
        }

        let embeddings = self.embeddings.as_ref().unwrap();

        // Build Ball Tree for efficient O(log n) search using Linfa
        let ball_tree = BallTree::new()
            .from_batch(embeddings, L2Dist)
            .map_err(|e| format!("Failed to build Ball Tree: {}", e))?;

        let query_arr = Array1::from_vec(query.to_vec());

        // Use Linfa to find K nearest neighbors efficiently
        let k = self.k.min(self.labels.len());
        let neighbors = ball_tree
            .k_nearest(query_arr.view(), k)
            .map_err(|e| format!("KNN search failed: {}", e))?;

        // Quality+Speed weighted voting among neighbors
        // Formula: vote_weight = 0.9 * quality + 0.1 * speed_factor
        // Matches global QualityWeight=0.9 hyperparameter (90% quality, 10% speed)
        const QUALITY_WEIGHT: f64 = 0.9;
        const SPEED_WEIGHT: f64 = 0.1;

        // Compute min/max latency for normalization
        let max_latency = self.latencies.iter().cloned().max().unwrap_or(1) as f64;
        let min_latency = self.latencies.iter().cloned().min().unwrap_or(1) as f64;
        let latency_range = (max_latency - min_latency).max(1.0);

        let mut model_scores: HashMap<&str, f64> = HashMap::new();
        let mut model_counts: HashMap<&str, i32> = HashMap::new();

        for (_point, idx) in neighbors.iter() {
            let model: &str = &self.labels[*idx];
            let quality = if *idx < self.qualities.len() {
                self.qualities[*idx].max(0.01)
            } else {
                0.5
            };

            let latency = if *idx < self.latencies.len() {
                self.latencies[*idx] as f64
            } else {
                (min_latency + max_latency) / 2.0 // Default to median
            };

            let normalized_latency = (latency - min_latency) / latency_range;
            let speed_factor = 1.0 - normalized_latency; // 1.0 for fastest, 0.0 for slowest
            let vote_weight = QUALITY_WEIGHT * quality + SPEED_WEIGHT * speed_factor;

            *model_scores.entry(model).or_insert(0.0) += vote_weight;
            *model_counts.entry(model).or_insert(0) += 1;
        }

        // Pick model with highest weighted score
        model_scores
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(model, _)| model.to_string())
            .ok_or_else(|| "No votes found".to_string())
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Save model to JSON
    pub fn to_json(&self) -> Result<String, String> {
        let embeddings_vec: Vec<Vec<f64>> = self
            .embeddings
            .as_ref()
            .map(|e| e.rows().into_iter().map(|r| r.to_vec()).collect())
            .unwrap_or_default();

        let data = KNNModelData {
            algorithm: "knn".to_string(),
            trained: self.trained,
            k: self.k,
            embeddings: embeddings_vec,
            labels: self.labels.clone(),
            qualities: self.qualities.clone(),
            latencies: self.latencies.clone(),
        };

        serde_json::to_string_pretty(&data).map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Load model from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        let data: KNNModelData =
            serde_json::from_str(json).map_err(|e| format!("JSON parse failed: {}", e))?;

        let mut selector = Self::new(data.k);

        if !data.embeddings.is_empty() {
            let dim = data.embeddings[0].len();
            let n = data.embeddings.len();
            let flat: Vec<f64> = data.embeddings.into_iter().flatten().collect();

            selector.embeddings = Some(
                Array2::from_shape_vec((n, dim), flat)
                    .map_err(|e| format!("Failed to restore embeddings: {}", e))?,
            );
            selector.labels = data.labels;
            selector.qualities = data.qualities;
            selector.latencies = data.latencies;
            selector.trained = data.trained;
        }

        Ok(selector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model_json() -> String {
        r#"{
            "algorithm": "knn",
            "trained": true,
            "k": 3,
            "embeddings": [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.1]
            ],
            "labels": ["model-a", "model-a", "model-b", "model-b"],
            "qualities": [0.9, 0.85, 0.95, 0.88],
            "latencies": [100, 110, 200, 190]
        }"#.to_string()
    }

    #[test]
    fn test_knn_load_and_select() {
        let json = create_test_model_json();
        let selector = KNNSelector::from_json(&json).unwrap();

        assert!(selector.is_trained());
        assert_eq!(selector.k, 3);

        // Query closer to model-a cluster
        let result = selector.select(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result, "model-a");

        // Query closer to model-b cluster
        let result = selector.select(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(result, "model-b");
    }

    #[test]
    fn test_knn_quality_weighted_voting() {
        // Two neighbors of model-a with low quality, one neighbor of model-b with high quality
        let json = r#"{
            "algorithm": "knn",
            "trained": true,
            "k": 3,
            "embeddings": [
                [1.0, 0.0, 0.0],
                [1.0, 0.1, 0.0],
                [0.9, 0.0, 0.0]
            ],
            "labels": ["model-a", "model-a", "model-b"],
            "qualities": [0.3, 0.3, 0.95],
            "latencies": [100, 110, 200]
        }"#;

        let selector = KNNSelector::from_json(json).unwrap();

        // Query is closest to all 3 neighbors
        // Without quality weighting: model-a wins (2 votes vs 1)
        // With quality weighting: model-b wins (0.95 > 0.3+0.3=0.6)
        let result = selector.select(&[0.95, 0.05, 0.0]).unwrap();
        assert_eq!(result, "model-b", "Quality-weighted voting should pick higher quality model");
    }

    #[test]
    fn test_knn_json_roundtrip() {
        let json = r#"{
            "algorithm": "knn",
            "trained": true,
            "k": 5,
            "embeddings": [[1.0, 2.0, 3.0]],
            "labels": ["test-model"],
            "qualities": [0.85],
            "latencies": [500]
        }"#;

        let selector = KNNSelector::from_json(json).unwrap();
        let exported = selector.to_json().unwrap();
        let restored = KNNSelector::from_json(&exported).unwrap();

        assert_eq!(restored.k, 5);
        assert!(restored.is_trained());
        assert_eq!(restored.qualities.len(), 1);
        assert_eq!(restored.latencies.len(), 1);
    }
}
