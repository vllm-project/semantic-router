//! # KMeans Selector
//!
//! Production implementation of KMeans clustering for model selection.
//! Based on Avengers-Pro framework (arXiv:2508.12631) that routes queries
//! to models based on performance-efficiency score.
//!
//! ## Algorithm
//!
//! 1. Cluster query embeddings using K-means with k-means++ initialization
//! 2. Track per-model performance and latency within each cluster
//! 3. Route new queries to the best model for their assigned cluster
//!
//! ## Performance-Efficiency Score (Avengers-Pro)
//!
//! For each model in a cluster:
//! - Performance = success_rate * average_quality
//! - Efficiency = 1 / (1 + normalized_latency)
//! - Score = (1 - efficiency_weight) * performance + efficiency_weight * efficiency
//!
//! Default efficiency_weight = 0.3 (70% performance, 30% efficiency)

use std::collections::HashMap;
use std::sync::RwLock;

use rand::Rng;
use serde::{Deserialize, Serialize};

use super::{
    euclidean_distance, ModelRef, ModelSelector, SelectionContext, SelectionResult, TrainingRecord,
};

/// Stats for a model within a cluster
#[derive(Debug, Clone, Default)]
struct ClusterModelStats {
    success_count: usize,
    total_count: usize,
    total_latency_ms: f64,
    total_quality: f64,
}

/// Serializable KMeans model data (matches Go KMeansModelData)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeansModelData {
    pub version: String,
    pub algorithm: String,
    pub training: Vec<TrainingRecord>,
    pub trained: bool,
    pub num_clusters: i32,
    pub centroids: Vec<Vec<f64>>,
    pub cluster_models: Vec<String>,
    #[serde(default)]
    pub efficiency_weight: f64,
}

/// KMeans Selector implementation
pub struct KMeansSelector {
    /// Number of clusters
    pub num_clusters: usize,
    /// Cluster centroids (num_clusters x embedding_dim)
    centroids: RwLock<Vec<Vec<f64>>>,
    /// Best model for each cluster
    cluster_models: RwLock<Vec<String>>,
    /// Per-cluster per-model stats
    cluster_stats: RwLock<HashMap<usize, HashMap<String, ClusterModelStats>>>,
    /// Training records
    training: RwLock<Vec<TrainingRecord>>,
    /// Efficiency weight (0-1): 0 = pure performance, 1 = pure efficiency
    pub efficiency_weight: f64,
    /// Whether the model is trained
    trained: RwLock<bool>,
    /// Maximum training size
    max_size: usize,
}

impl KMeansSelector {
    /// Create a new KMeans selector
    pub fn new(num_clusters: usize) -> Self {
        let num_clusters = if num_clusters == 0 { 4 } else { num_clusters };
        Self {
            num_clusters,
            centroids: RwLock::new(Vec::new()),
            cluster_models: RwLock::new(vec![String::new(); num_clusters]),
            cluster_stats: RwLock::new(HashMap::new()),
            training: RwLock::new(Vec::with_capacity(10000)),
            efficiency_weight: 0.3, // Default: 70% performance, 30% efficiency
            trained: RwLock::new(false),
            max_size: 10000,
        }
    }

    /// Create a KMeans selector with custom efficiency weight
    pub fn with_efficiency(num_clusters: usize, efficiency_weight: f64) -> Self {
        let mut selector = Self::new(num_clusters);
        selector.efficiency_weight = efficiency_weight.max(0.0).min(1.0);
        selector
    }

    /// Load from JSON data
    pub fn load_from_json(&mut self, data: &[u8]) -> Result<(), String> {
        let model_data: KMeansModelData = serde_json::from_slice(data)
            .map_err(|e| format!("Failed to parse KMeans JSON: {}", e))?;

        if model_data.num_clusters > 0 {
            self.num_clusters = model_data.num_clusters as usize;
        }
        if model_data.efficiency_weight > 0.0 {
            self.efficiency_weight = model_data.efficiency_weight;
        }

        {
            let mut cluster_models = self.cluster_models.write().map_err(|e| e.to_string())?;
            if !model_data.cluster_models.is_empty() {
                *cluster_models = model_data.cluster_models;
            }
        }

        {
            let mut centroids = self.centroids.write().map_err(|e| e.to_string())?;
            *centroids = model_data.centroids;
        }

        {
            let mut training = self.training.write().map_err(|e| e.to_string())?;
            *training = model_data.training;
        }

        {
            let mut trained = self.trained.write().map_err(|e| e.to_string())?;
            let centroids = self.centroids.read().map_err(|e| e.to_string())?;
            *trained = model_data.trained && !centroids.is_empty();
        }

        Ok(())
    }

    /// Save to JSON data
    pub fn save_to_json(&self) -> Result<Vec<u8>, String> {
        let training = self.training.read().map_err(|e| e.to_string())?;
        let centroids = self.centroids.read().map_err(|e| e.to_string())?;
        let cluster_models = self.cluster_models.read().map_err(|e| e.to_string())?;
        let trained = self.trained.read().map_err(|e| e.to_string())?;

        let model_data = KMeansModelData {
            version: "1.0".to_string(),
            algorithm: "kmeans".to_string(),
            training: training.clone(),
            trained: *trained,
            num_clusters: self.num_clusters as i32,
            centroids: centroids.clone(),
            cluster_models: cluster_models.clone(),
            efficiency_weight: self.efficiency_weight,
        };
        serde_json::to_vec(&model_data).map_err(|e| format!("Failed to serialize KMeans: {}", e))
    }

    /// Find the nearest cluster for an embedding
    fn find_nearest_cluster(&self, embedding: &[f64]) -> usize {
        let centroids = self.centroids.read().unwrap();
        if centroids.is_empty() {
            return 0;
        }

        let mut best_cluster = 0;
        let mut best_dist = f64::MAX;

        for (i, centroid) in centroids.iter().enumerate() {
            let dist = euclidean_distance(embedding, centroid);
            if dist < best_dist {
                best_dist = dist;
                best_cluster = i;
            }
        }

        best_cluster
    }

    /// Run K-means clustering with k-means++ initialization
    fn run_kmeans(&self) {
        let training = self.training.read().unwrap();

        // Get valid training records
        let valid_records: Vec<&TrainingRecord> = training
            .iter()
            .filter(|r| !r.query_embedding.is_empty())
            .collect();

        if valid_records.len() < self.num_clusters {
            return;
        }

        let emb_dim = valid_records[0].query_embedding.len();

        // Ensure all embeddings have the same dimension
        let valid_records: Vec<&TrainingRecord> = valid_records
            .into_iter()
            .filter(|r| r.query_embedding.len() == emb_dim)
            .collect();

        if valid_records.len() < self.num_clusters {
            return;
        }

        // Initialize centroids using k-means++
        let mut rng = rand::thread_rng();
        let mut new_centroids: Vec<Vec<f64>> = Vec::with_capacity(self.num_clusters);

        // First centroid: random selection
        let first_idx = rng.gen_range(0..valid_records.len());
        new_centroids.push(valid_records[first_idx].query_embedding.clone());

        // Remaining centroids: weighted by distance squared
        for _ in 1..self.num_clusters {
            let mut distances: Vec<f64> = Vec::with_capacity(valid_records.len());
            let mut total_dist = 0.0;

            for record in &valid_records {
                let min_dist = new_centroids
                    .iter()
                    .map(|c| euclidean_distance(&record.query_embedding, c))
                    .fold(f64::MAX, f64::min);
                let dist_sq = min_dist * min_dist;
                distances.push(dist_sq);
                total_dist += dist_sq;
            }

            // Select next centroid with probability proportional to distance squared
            if total_dist > 0.0 {
                let target = rng.gen::<f64>() * total_dist;
                let mut cumulative = 0.0;
                let mut selected_idx = 0;

                for (i, &d) in distances.iter().enumerate() {
                    cumulative += d;
                    if cumulative >= target {
                        selected_idx = i;
                        break;
                    }
                }
                new_centroids.push(valid_records[selected_idx].query_embedding.clone());
            } else {
                // Fallback to sequential
                let idx = new_centroids.len() % valid_records.len();
                new_centroids.push(valid_records[idx].query_embedding.clone());
            }
        }

        // Run K-means iterations
        const MAX_ITERATIONS: usize = 100;
        const TOLERANCE: f64 = 1e-6;

        for _ in 0..MAX_ITERATIONS {
            // Assign points to clusters
            let assignments: Vec<usize> = valid_records
                .iter()
                .map(|r| {
                    let mut best_cluster = 0;
                    let mut best_dist = f64::MAX;
                    for (i, centroid) in new_centroids.iter().enumerate() {
                        let dist = euclidean_distance(&r.query_embedding, centroid);
                        if dist < best_dist {
                            best_dist = dist;
                            best_cluster = i;
                        }
                    }
                    best_cluster
                })
                .collect();

            // Update centroids
            let mut cluster_sums: Vec<Vec<f64>> = vec![vec![0.0; emb_dim]; self.num_clusters];
            let mut cluster_counts: Vec<usize> = vec![0; self.num_clusters];

            for (i, record) in valid_records.iter().enumerate() {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                for (j, val) in record.query_embedding.iter().enumerate() {
                    cluster_sums[cluster][j] += val;
                }
            }

            let mut updated_centroids: Vec<Vec<f64>> = Vec::with_capacity(self.num_clusters);
            for i in 0..self.num_clusters {
                if cluster_counts[i] > 0 {
                    let centroid: Vec<f64> = cluster_sums[i]
                        .iter()
                        .map(|v| v / cluster_counts[i] as f64)
                        .collect();
                    updated_centroids.push(centroid);
                } else {
                    updated_centroids.push(new_centroids[i].clone());
                }
            }

            // Check convergence
            let diff: f64 = new_centroids
                .iter()
                .zip(updated_centroids.iter())
                .map(|(old, new)| euclidean_distance(old, new).powi(2))
                .sum::<f64>()
                .sqrt();

            new_centroids = updated_centroids;

            if diff < TOLERANCE {
                break;
            }
        }

        // Update centroids
        {
            let mut centroids = self.centroids.write().unwrap();
            *centroids = new_centroids;
        }

        // Update cluster models
        self.update_cluster_models(&valid_records);

        // Mark as trained
        {
            let mut trained = self.trained.write().unwrap();
            *trained = true;
        }
    }

    /// Update best model for each cluster using performance-efficiency score
    fn update_cluster_models(&self, records: &[&TrainingRecord]) {
        let centroids = self.centroids.read().unwrap();

        // Build comprehensive stats per model per cluster
        let mut stats: HashMap<usize, HashMap<String, ClusterModelStats>> = HashMap::new();

        for record in records {
            let cluster = {
                let mut best_cluster = 0;
                let mut best_dist = f64::MAX;
                for (i, centroid) in centroids.iter().enumerate() {
                    let dist = euclidean_distance(&record.query_embedding, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = i;
                    }
                }
                best_cluster
            };

            let cluster_stats = stats.entry(cluster).or_insert_with(HashMap::new);
            let model_stats = cluster_stats
                .entry(record.selected_model.clone())
                .or_insert_with(ClusterModelStats::default);

            model_stats.total_count += 1;
            model_stats.total_latency_ms += record.response_latency_ms();
            if record.success {
                model_stats.success_count += 1;
                model_stats.total_quality += record.response_quality;
            }
        }

        // Assign best model to each cluster using performance-efficiency score
        let mut cluster_models = self.cluster_models.write().unwrap();
        *cluster_models = vec![String::new(); self.num_clusters];

        for cluster in 0..self.num_clusters {
            let cluster_stats = match stats.get(&cluster) {
                Some(s) => s,
                None => continue,
            };

            let mut best_model = String::new();
            let mut best_score: f64 = -1.0;

            // Find max latency for normalization
            let max_latency: f64 = cluster_stats
                .values()
                .filter(|s| s.total_count > 0)
                .map(|s| s.total_latency_ms / s.total_count as f64)
                .fold(1.0, f64::max);

            for (model, model_stats) in cluster_stats {
                if model_stats.total_count < 3 {
                    continue; // Require minimum samples
                }

                // Performance score: success_rate * average_quality
                let success_rate =
                    model_stats.success_count as f64 / model_stats.total_count as f64;
                let avg_quality = if model_stats.success_count > 0 {
                    model_stats.total_quality / model_stats.success_count as f64
                } else {
                    0.5
                };
                let performance = success_rate * avg_quality;

                // Efficiency score: inverse of normalized latency
                let avg_latency = model_stats.total_latency_ms / model_stats.total_count as f64;
                let normalized_latency = avg_latency / max_latency;
                let efficiency = 1.0 / (1.0 + normalized_latency);

                // Combined performance-efficiency score (Avengers-Pro formula)
                let score = (1.0 - self.efficiency_weight) * performance
                    + self.efficiency_weight * efficiency;

                if score > best_score {
                    best_score = score;
                    best_model = model.clone();
                }
            }

            cluster_models[cluster] = best_model;
        }

        // Update cluster_stats
        {
            let mut cluster_stats_lock = self.cluster_stats.write().unwrap();
            *cluster_stats_lock = stats;
        }
    }

    /// Build a map from model name to ref index
    fn build_model_index(refs: &[ModelRef]) -> HashMap<String, usize> {
        refs.iter()
            .enumerate()
            .map(|(i, r)| (r.get_name().to_string(), i))
            .collect()
    }
}

impl ModelSelector for KMeansSelector {
    fn name(&self) -> &str {
        "kmeans"
    }

    fn train(&mut self, data: &[TrainingRecord]) -> Result<(), String> {
        {
            let mut training = self.training.write().map_err(|e| e.to_string())?;
            training.extend(data.iter().cloned());

            // Keep only recent records
            if training.len() > self.max_size {
                let drain_count = training.len() - self.max_size;
                training.drain(0..drain_count);
            }
        }

        // Train if we have enough records
        let training_len = self.training.read().map(|t| t.len()).unwrap_or(0);
        if training_len >= self.num_clusters {
            self.run_kmeans();
        }

        Ok(())
    }

    fn training_count(&self) -> usize {
        self.training.read().map(|t| t.len()).unwrap_or(0)
    }

    fn is_trained(&self) -> bool {
        self.trained.read().map(|t| *t).unwrap_or(false)
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

        let trained = *self.trained.read().ok()?;
        let centroids = self.centroids.read().ok()?;

        if !trained || centroids.is_empty() || ctx.query_embedding.is_empty() {
            // Not trained or no embedding, select first model
            return Some(SelectionResult {
                model_name: refs[0].get_name().to_string(),
                model_index: 0,
                score: 0.0,
            });
        }

        let cluster = self.find_nearest_cluster(&ctx.query_embedding);
        let model_index = Self::build_model_index(refs);
        let cluster_models = self.cluster_models.read().ok()?;

        if cluster < cluster_models.len() && !cluster_models[cluster].is_empty() {
            if let Some(&idx) = model_index.get(&cluster_models[cluster]) {
                return Some(SelectionResult {
                    model_name: cluster_models[cluster].clone(),
                    model_index: idx,
                    score: 1.0,
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
            // Cluster 1: math queries -> model_a
            TrainingRecord {
                query_text: "Calculate 2+2".to_string(),
                query_embedding: vec![1.0, 0.0, 0.0],
                decision_name: "math".to_string(),
                selected_model: "model_a".to_string(),
                response_latency_ns: 100_000_000,
                response_quality: 0.9,
                success: true,
                timestamp: 0,
            },
            TrainingRecord {
                query_text: "What is 3*4?".to_string(),
                query_embedding: vec![0.9, 0.1, 0.0],
                decision_name: "math".to_string(),
                selected_model: "model_a".to_string(),
                response_latency_ns: 110_000_000,
                response_quality: 0.85,
                success: true,
                timestamp: 1,
            },
            TrainingRecord {
                query_text: "Solve for x".to_string(),
                query_embedding: vec![0.95, 0.05, 0.0],
                decision_name: "math".to_string(),
                selected_model: "model_a".to_string(),
                response_latency_ns: 105_000_000,
                response_quality: 0.88,
                success: true,
                timestamp: 2,
            },
            // Cluster 2: coding queries -> model_b
            TrainingRecord {
                query_text: "Write a function".to_string(),
                query_embedding: vec![0.0, 1.0, 0.0],
                decision_name: "coding".to_string(),
                selected_model: "model_b".to_string(),
                response_latency_ns: 200_000_000,
                response_quality: 0.95,
                success: true,
                timestamp: 3,
            },
            TrainingRecord {
                query_text: "Debug this code".to_string(),
                query_embedding: vec![0.1, 0.9, 0.0],
                decision_name: "coding".to_string(),
                selected_model: "model_b".to_string(),
                response_latency_ns: 190_000_000,
                response_quality: 0.92,
                success: true,
                timestamp: 4,
            },
            TrainingRecord {
                query_text: "Refactor the class".to_string(),
                query_embedding: vec![0.05, 0.95, 0.0],
                decision_name: "coding".to_string(),
                selected_model: "model_b".to_string(),
                response_latency_ns: 195_000_000,
                response_quality: 0.93,
                success: true,
                timestamp: 5,
            },
        ]
    }

    #[test]
    fn test_kmeans_creation() {
        let kmeans = KMeansSelector::new(3);
        assert_eq!(kmeans.name(), "kmeans");
        assert_eq!(kmeans.training_count(), 0);
    }

    #[test]
    fn test_kmeans_training() {
        let mut kmeans = KMeansSelector::new(2);
        let records = create_test_records();
        kmeans.train(&records).unwrap();
        assert_eq!(kmeans.training_count(), 6);
    }

    #[test]
    fn test_kmeans_efficiency_weight() {
        let kmeans = KMeansSelector::with_efficiency(3, 0.7);
        assert!((kmeans.efficiency_weight - 0.7).abs() < 1e-6);

        // Test clamping
        let kmeans = KMeansSelector::with_efficiency(3, 1.5);
        assert!((kmeans.efficiency_weight - 1.0).abs() < 1e-6);

        let kmeans = KMeansSelector::with_efficiency(3, -0.5);
        assert!((kmeans.efficiency_weight - 0.0).abs() < 1e-6);
    }
}
