//! KNN (K-Nearest Neighbors) implementation using Linfa
//! Aligned with FusionFactory (arXiv:2507.10540) query-level fusion approach

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
    trained: bool,
}

/// Training record for KNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNTrainingRecord {
    pub embedding: Vec<f64>,
    pub model: String,
}

/// Model data for JSON serialization
#[derive(Debug, Serialize, Deserialize)]
pub struct KNNModelData {
    pub algorithm: String,
    pub trained: bool,
    pub k: usize,
    pub embeddings: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}

impl KNNSelector {
    /// Create a new KNN selector with specified k
    pub fn new(k: usize) -> Self {
        Self {
            k,
            embeddings: None,
            labels: Vec::new(),
            trained: false,
        }
    }

    /// Train the KNN model with training records
    /// Stores embeddings for later Ball Tree search
    pub fn train(&mut self, records: Vec<KNNTrainingRecord>) -> Result<(), String> {
        if records.is_empty() {
            return Err("No training records provided".to_string());
        }

        let dim = records[0].embedding.len();
        let n = records.len();

        // Build embeddings matrix
        let mut data = Vec::with_capacity(n * dim);
        self.labels.clear();

        for record in &records {
            if record.embedding.len() != dim {
                return Err(format!(
                    "Inconsistent embedding dimension: expected {}, got {}",
                    dim,
                    record.embedding.len()
                ));
            }
            data.extend(&record.embedding);
            self.labels.push(record.model.clone());
        }

        self.embeddings = Some(
            Array2::from_shape_vec((n, dim), data)
                .map_err(|e| format!("Failed to create embeddings matrix: {}", e))?,
        );
        self.trained = true;

        Ok(())
    }

    /// Select the best model for a query embedding using Linfa Ball Tree
    /// Implements query-level routing per FusionFactory's tailored LLM routers
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

        // Vote among neighbors - FusionFactory style query-level fusion
        // k_nearest returns Vec<(point, index)>
        let mut votes: HashMap<&str, usize> = HashMap::new();
        for (_point, idx) in neighbors.iter() {
            let model: &str = &self.labels[*idx];
            *votes.entry(model).or_insert(0) += 1;
        }

        votes
            .into_iter()
            .max_by_key(|(_, count)| *count)
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
            selector.trained = data.trained;
        }

        Ok(selector)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_train_and_select() {
        let mut selector = KNNSelector::new(3);

        let records = vec![
            KNNTrainingRecord {
                embedding: vec![1.0, 0.0, 0.0],
                model: "model-a".to_string(),
            },
            KNNTrainingRecord {
                embedding: vec![1.0, 0.1, 0.0],
                model: "model-a".to_string(),
            },
            KNNTrainingRecord {
                embedding: vec![0.0, 1.0, 0.0],
                model: "model-b".to_string(),
            },
            KNNTrainingRecord {
                embedding: vec![0.0, 1.0, 0.1],
                model: "model-b".to_string(),
            },
        ];

        selector.train(records).unwrap();
        assert!(selector.is_trained());

        // Query closer to model-a cluster
        let result = selector.select(&[0.9, 0.1, 0.0]).unwrap();
        assert_eq!(result, "model-a");

        // Query closer to model-b cluster
        let result = selector.select(&[0.1, 0.9, 0.0]).unwrap();
        assert_eq!(result, "model-b");
    }

    #[test]
    fn test_knn_json_roundtrip() {
        let mut selector = KNNSelector::new(5);
        let records = vec![
            KNNTrainingRecord {
                embedding: vec![1.0, 2.0, 3.0],
                model: "test-model".to_string(),
            },
        ];
        selector.train(records).unwrap();

        let json = selector.to_json().unwrap();
        let restored = KNNSelector::from_json(&json).unwrap();

        assert_eq!(restored.k, 5);
        assert!(restored.is_trained());
    }
}
