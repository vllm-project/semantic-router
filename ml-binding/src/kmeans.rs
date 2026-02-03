//! KMeans clustering implementation using linfa-nn
//!
//! Inference-only implementation. Training is done in Python (src/training/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.
//!
//! Uses linfa-nn for efficient centroid lookup during inference.

use linfa_nn::{distance::L2Dist, BallTree, NearestNeighbour};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// KMeans Selector using Linfa's Ball Tree for efficient centroid lookup
#[derive(Debug)]
pub struct KMeansSelector {
    num_clusters: usize,
    centroids: Option<Array2<f64>>,
    cluster_models: Vec<String>,
    trained: bool,
}

/// Model data for JSON serialization
#[derive(Debug, Serialize, Deserialize)]
pub struct KMeansModelData {
    pub algorithm: String,
    pub trained: bool,
    pub num_clusters: usize,
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<String>,
}

impl KMeansSelector {
    /// Create a new KMeans selector with specified number of clusters
    pub fn new(num_clusters: usize) -> Self {
        Self {
            num_clusters,
            centroids: None,
            cluster_models: Vec::new(),
            trained: false,
        }
    }

    /// Select the best model for a query embedding using Linfa BallTree
    /// Uses linfa-nn for efficient O(log k) nearest centroid lookup
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        if !self.trained {
            return Err("Model not trained".to_string());
        }

        let centroids = self.centroids.as_ref().unwrap();
        let query_arr = Array1::from_vec(query.to_vec());

        // Use Linfa BallTree for efficient nearest centroid search
        let ball_tree = BallTree::new()
            .from_batch(centroids, L2Dist)
            .map_err(|e| format!("Failed to build BallTree from centroids: {}", e))?;

        // Find nearest centroid using Linfa's k_nearest
        let neighbors = ball_tree
            .k_nearest(query_arr.view(), 1)
            .map_err(|e| format!("Nearest centroid search failed: {}", e))?;

        let nearest_cluster = neighbors
            .first()
            .map(|(_, idx)| *idx)
            .ok_or_else(|| "No nearest centroid found".to_string())?;

        self.cluster_models
            .get(nearest_cluster)
            .cloned()
            .ok_or_else(|| "No model assigned to cluster".to_string())
    }

    /// Check if model is trained
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Save model to JSON
    pub fn to_json(&self) -> Result<String, String> {
        let centroids_vec: Vec<Vec<f64>> = self
            .centroids
            .as_ref()
            .map(|c| c.rows().into_iter().map(|r| r.to_vec()).collect())
            .unwrap_or_default();

        let data = KMeansModelData {
            algorithm: "kmeans".to_string(),
            trained: self.trained,
            num_clusters: self.num_clusters,
            centroids: centroids_vec,
            cluster_models: self.cluster_models.clone(),
        };

        serde_json::to_string_pretty(&data).map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Load model from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        let data: KMeansModelData =
            serde_json::from_str(json).map_err(|e| format!("JSON parse failed: {}", e))?;

        let mut selector = Self::new(data.num_clusters);

        if !data.centroids.is_empty() {
            let dim = data.centroids[0].len();
            let n = data.centroids.len();
            let flat: Vec<f64> = data.centroids.into_iter().flatten().collect();

            selector.centroids = Some(
                Array2::from_shape_vec((n, dim), flat)
                    .map_err(|e| format!("Failed to restore centroids: {}", e))?,
            );
            selector.cluster_models = data.cluster_models;
            selector.trained = data.trained;
        }

        Ok(selector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_load_and_select() {
        // Pre-trained model with 2 clusters
        let json = r#"{
            "algorithm": "kmeans",
            "trained": true,
            "num_clusters": 2,
            "centroids": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ],
            "cluster_models": ["model-a", "model-b"]
        }"#;

        let selector = KMeansSelector::from_json(json).unwrap();
        assert!(selector.is_trained());

        // Query closer to first centroid
        let result = selector.select(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result, "model-a");

        // Query closer to second centroid
        let result = selector.select(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(result, "model-b");
    }

    #[test]
    fn test_kmeans_json_roundtrip() {
        let json = r#"{
            "algorithm": "kmeans",
            "trained": true,
            "num_clusters": 3,
            "centroids": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ],
            "cluster_models": ["model-a", "model-b", "model-c"]
        }"#;

        let selector = KMeansSelector::from_json(json).unwrap();
        let exported = selector.to_json().unwrap();
        let restored = KMeansSelector::from_json(&exported).unwrap();

        assert!(restored.is_trained());
        assert_eq!(restored.num_clusters, 3);
    }
}
