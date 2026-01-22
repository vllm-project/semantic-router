//! KMeans clustering implementation using linfa-clustering

use linfa::prelude::*;
use linfa_clustering::KMeans;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// KMeans Selector using Linfa's clustering implementation
#[derive(Debug)]
pub struct KMeansSelector {
    num_clusters: usize,
    centroids: Option<Array2<f64>>,
    cluster_models: Vec<String>,
    trained: bool,
}

/// Training record for KMeans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansTrainingRecord {
    pub embedding: Vec<f64>,
    pub model: String,
    pub quality: f64,
    pub latency_ns: i64,
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

    /// Train the KMeans model with training records
    pub fn train(&mut self, records: Vec<KMeansTrainingRecord>) -> Result<(), String> {
        if records.is_empty() {
            return Err("No training records provided".to_string());
        }

        let dim = records[0].embedding.len();
        let n = records.len();

        // Build embeddings matrix
        let mut data = Vec::with_capacity(n * dim);
        for record in &records {
            if record.embedding.len() != dim {
                return Err(format!(
                    "Inconsistent embedding dimension: expected {}, got {}",
                    dim,
                    record.embedding.len()
                ));
            }
            data.extend(&record.embedding);
        }

        let embeddings = Array2::from_shape_vec((n, dim), data)
            .map_err(|e| format!("Failed to create embeddings matrix: {}", e))?;

        // Create dataset for Linfa
        let dataset = DatasetBase::from(embeddings.clone());

        // Fit KMeans
        let k = self.num_clusters.min(n);
        let model = KMeans::params(k)
            .max_n_iterations(100)
            .tolerance(1e-4)
            .fit(&dataset)
            .map_err(|e| format!("KMeans fitting failed: {}", e))?;

        self.centroids = Some(model.centroids().to_owned());

        // Assign points to clusters
        let predictions = model.predict(&dataset);

        // Assign best model to each cluster based on quality scores
        self.cluster_models = self.assign_cluster_models(&records, &predictions, k);
        self.trained = true;

        Ok(())
    }

    /// Assign the best model to each cluster based on quality scores
    fn assign_cluster_models(
        &self,
        records: &[KMeansTrainingRecord],
        labels: &Array1<usize>,
        k: usize,
    ) -> Vec<String> {
        // For each cluster, track model quality scores
        let mut cluster_model_scores: Vec<HashMap<String, (f64, i32)>> = vec![HashMap::new(); k];

        for (i, record) in records.iter().enumerate() {
            let cluster = labels[i];
            if cluster < k {
                let entry = cluster_model_scores[cluster]
                    .entry(record.model.clone())
                    .or_insert((0.0, 0));
                entry.0 += record.quality;
                entry.1 += 1;
            }
        }

        // Select best model for each cluster (highest average quality)
        cluster_model_scores
            .into_iter()
            .map(|scores| {
                scores
                    .into_iter()
                    .map(|(model, (total_quality, count))| {
                        (model, total_quality / count as f64)
                    })
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(model, _)| model)
                    .unwrap_or_else(|| "unknown".to_string())
            })
            .collect()
    }

    /// Select the best model for a query embedding
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        if !self.trained {
            return Err("Model not trained".to_string());
        }

        let centroids = self.centroids.as_ref().unwrap();
        let query_arr = Array1::from_vec(query.to_vec());

        // Find nearest centroid using Euclidean distance
        let mut min_dist = f64::MAX;
        let mut nearest_cluster = 0;

        for (i, centroid) in centroids.axis_iter(Axis(0)).enumerate() {
            let diff = &query_arr - &centroid;
            let dist = diff.mapv(|x| x * x).sum().sqrt();
            if dist < min_dist {
                min_dist = dist;
                nearest_cluster = i;
            }
        }

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
    fn test_kmeans_train_and_select() {
        let mut selector = KMeansSelector::new(2);

        let records = vec![
            KMeansTrainingRecord {
                embedding: vec![1.0, 0.0, 0.0],
                model: "model-a".to_string(),
                quality: 0.9,
                latency_ns: 100,
            },
            KMeansTrainingRecord {
                embedding: vec![1.1, 0.1, 0.0],
                model: "model-a".to_string(),
                quality: 0.85,
                latency_ns: 110,
            },
            KMeansTrainingRecord {
                embedding: vec![0.0, 1.0, 0.0],
                model: "model-b".to_string(),
                quality: 0.95,
                latency_ns: 200,
            },
            KMeansTrainingRecord {
                embedding: vec![0.1, 1.1, 0.0],
                model: "model-b".to_string(),
                quality: 0.88,
                latency_ns: 190,
            },
        ];

        selector.train(records).unwrap();
        assert!(selector.is_trained());

        // Should work without panicking
        let _result = selector.select(&[0.9, 0.1, 0.0]);
    }
}
