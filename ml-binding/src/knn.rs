//! KNN (K-Nearest Neighbors) implementation using Linfa
//!
//! Inference-only implementation. Training is done in Python (src/training/model_selection/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.
//!
//! Uses quality-weighted voting: neighbors with higher quality scores have more influence.
//! This ensures we select models that PERFORM BEST, not just which was selected.

use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// KNN Selector using Linfa's Ball Tree for efficient nearest neighbor search
/// Implements query-level fusion for LLM routing as per FusionFactory
#[derive(Debug)]
pub struct KNNSelector {
    k: usize,
    embeddings: Option<Array2<f64>>,
    labels: Vec<String>,
    qualities: Vec<f64>, // Quality score for each training sample
    latencies: Vec<i64>, // Latency in nanoseconds for each sample
    trained: bool,
}

/// Model data for JSON serialization
#[derive(Debug, Serialize, Deserialize)]
pub struct KNNModelData {
    pub algorithm: String,
    #[serde(default)]
    pub format_version: Option<u32>,
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

        let embeddings = self.embeddings.as_ref().ok_or("Missing KNN embeddings")?;
        if query.len() != embeddings.ncols() || query.iter().any(|x| !x.is_finite()) {
            return Err(format!(
                "Expected {} finite KNN features, got {}",
                embeddings.ncols(),
                query.len()
            ));
        }
        let normalized = embeddings.map_axis(ndarray::Axis(1), |row| {
            row.iter().map(|x| x * x).sum::<f64>().sqrt()
        });
        let mut features = embeddings.clone();
        for (mut row, norm) in features.rows_mut().into_iter().zip(normalized) {
            if norm > 0.0 {
                row /= norm;
            }
        }
        let norm = query.iter().map(|x| x * x).sum::<f64>().sqrt();
        let query_arr =
            Array1::from_iter(query.iter().map(|x| if norm > 0.0 { x / norm } else { *x }));
        let ball_tree = BallTree::new()
            .from_batch(&features, L2Dist)
            .map_err(|e| format!("Failed to build Ball Tree: {}", e))?;
        let k = self.k.min(self.labels.len());
        let nearest = ball_tree
            .k_nearest(query_arr.view(), k)
            .map_err(|e| format!("KNN search failed: {}", e))?;
        let squared_distance = |idx: usize| {
            features
                .row(idx)
                .iter()
                .zip(query_arr.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        };
        let radius = nearest
            .iter()
            .map(|(_, idx)| squared_distance(*idx))
            .fold(0.0, f64::max)
            .sqrt()
            + 1e-12;
        // Include every boundary tie before sorting by distance and sample index.
        let neighbors = ball_tree
            .within_range(query_arr.view(), radius)
            .map_err(|e| format!("KNN radius search failed: {}", e))?;
        let mut indices: Vec<usize> = neighbors.iter().map(|(_, idx)| *idx).collect();
        indices.sort_by(|a, b| {
            squared_distance(*a)
                .total_cmp(&squared_distance(*b))
                .then_with(|| a.cmp(b))
        });
        let mut scores = BTreeMap::<&str, f64>::new();
        for idx in indices.into_iter().take(k) {
            // Identical to Python: latencies are integer nanoseconds, with a
            // fixed 10-second scale independent of unrelated training samples.
            let speed = 1.0 / (1.0 + self.latencies[idx] as f64 / 10_000_000_000.0);
            let weight = 0.9 * self.qualities[idx] + 0.1 * speed;
            *scores.entry(&self.labels[idx]).or_default() += weight;
        }
        // BTreeMap iteration gives a stable lexicographic tie break.
        let mut winner = None;
        let mut best_score = f64::NEG_INFINITY;
        for (name, score) in scores {
            if score > best_score {
                winner = Some(name.to_string());
                best_score = score;
            }
        }
        winner.ok_or_else(|| "No votes found".into())
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
            format_version: Some(2),
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
        let mut data: KNNModelData =
            serde_json::from_str(json).map_err(|e| format!("JSON parse failed: {}", e))?;

        if data.algorithm != "knn" || !matches!(data.format_version, None | Some(1) | Some(2)) {
            return Err("Unsupported KNN artifact format".into());
        }
        let n = data.embeddings.len();
        let dim = data.embeddings.first().map_or(0, Vec::len);
        if data.qualities.is_empty() {
            data.qualities = vec![0.5; n];
        }
        if data.latencies.is_empty() {
            data.latencies = vec![0; n];
        }
        if data.k == 0
            || n == 0
            || dim == 0
            || data.labels.len() != n
            || data.qualities.len() != n
            || data.latencies.len() != n
            || data
                .embeddings
                .iter()
                .any(|v| v.len() != dim || v.iter().any(|x| !x.is_finite()))
            || data.qualities.iter().any(|x| !x.is_finite())
            || data.latencies.iter().any(|x| *x < 0)
            || data.labels.iter().any(String::is_empty)
        {
            return Err("Invalid KNN sample shapes or values".into());
        }
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
        }"#
        .to_string()
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
        assert_eq!(
            result, "model-b",
            "Quality-weighted voting should pick higher quality model"
        );
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
