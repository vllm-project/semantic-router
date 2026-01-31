//! SVM (Support Vector Machine) using Linfa
//!
//! Supports both Linear and RBF kernels for one-vs-all multiclass classification.
//! Aligned with FusionFactory (arXiv:2507.10540) query-level fusion approach.
//!
//! - Linear kernel: f(x) = w·x - rho (fast, good for high-dim data)
//! - RBF kernel: f(x) = Σ(αᵢ·exp(-γ||x-xᵢ||²)) - rho (flexible boundaries)
//!
//! Training uses quality scores to determine the BEST model for each query,
//! rather than blindly using which model was selected.

use linfa::prelude::*;
use linfa_svm::Svm;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Kernel type for SVM
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KernelType {
    Linear,
    Rbf,
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::Rbf // RBF is better for high-dimensional embeddings
    }
}

/// Linear SVM classifier - stores weight vector for fast inference
#[derive(Clone)]
struct LinearClassifier {
    model_name: String,
    weights: Array1<f64>,
    rho: f64,
}

impl LinearClassifier {
    /// Decision function: f(x) = w·x - rho
    #[inline]
    fn decision_function(&self, x: &Array1<f64>) -> f64 {
        self.weights.dot(x) - self.rho
    }
}

/// RBF SVM classifier - stores alpha and training data for kernel computation
#[derive(Clone)]
struct RbfClassifier {
    model_name: String,
    alpha: Vec<f64>,
    support_vectors: Array2<f64>, // Training samples used for kernel
    rho: f64,
    gamma: f64,
}

impl RbfClassifier {
    /// RBF kernel: k(x, y) = exp(-γ||x-y||²)
    #[inline]
    fn rbf_kernel(&self, x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let diff = x - y;
        let sq_dist: f64 = diff.iter().map(|d| d * d).sum();
        (-self.gamma * sq_dist).exp()
    }

    /// Decision function: f(x) = Σ(αᵢ·k(x, xᵢ)) - rho
    fn decision_function(&self, x: &Array1<f64>) -> f64 {
        let mut sum = 0.0;
        for (i, alpha_i) in self.alpha.iter().enumerate() {
            let x_i = self.support_vectors.row(i);
            let x_i_owned = x_i.to_owned();
            sum += alpha_i * self.rbf_kernel(x, &x_i_owned);
        }
        sum - self.rho
    }
}

/// Unified classifier that can be either Linear or RBF
#[derive(Clone)]
enum Classifier {
    Linear(LinearClassifier),
    Rbf(RbfClassifier),
}

impl Classifier {
    fn model_name(&self) -> &str {
        match self {
            Classifier::Linear(c) => &c.model_name,
            Classifier::Rbf(c) => &c.model_name,
        }
    }

    fn decision_function(&self, x: &Array1<f64>) -> f64 {
        match self {
            Classifier::Linear(c) => c.decision_function(x),
            Classifier::Rbf(c) => c.decision_function(x),
        }
    }
}

/// SVM Selector for LLM routing
pub struct SVMSelector {
    classifiers: Vec<Classifier>,
    model_names: Vec<String>,
    trained: bool,
    kernel_type: KernelType,
    gamma: f64, // For RBF kernel
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMTrainingRecord {
    pub embedding: Vec<f64>,
    pub model: String,
    pub quality: f64,      // Response quality score (0-1)
    pub latency_ns: i64,   // Response latency in nanoseconds
}

#[derive(Debug, Serialize, Deserialize)]
struct LinearClassifierData {
    model_name: String,
    weights: Vec<f64>,
    rho: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RbfClassifierData {
    model_name: String,
    alpha: Vec<f64>,
    support_vectors: Vec<Vec<f64>>,
    rho: f64,
    gamma: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SVMModelData {
    pub algorithm: String,
    pub trained: bool,
    pub model_names: Vec<String>,
    pub kernel_type: KernelType,
    pub gamma: f64,
    #[serde(default)]
    pub linear_classifiers: Vec<LinearClassifierData>,
    #[serde(default)]
    pub rbf_classifiers: Vec<RbfClassifierData>,
}

impl SVMSelector {
    pub fn new() -> Self {
        Self::with_kernel(KernelType::Rbf, 1.0) // RBF with gamma=1.0 for high-dim normalized embeddings
    }

    pub fn with_kernel(kernel_type: KernelType, gamma: f64) -> Self {
        Self {
            classifiers: Vec::new(),
            model_names: Vec::new(),
            trained: false,
            kernel_type,
            gamma,
        }
    }

    /// Create with RBF kernel. Gamma defaults to 1.0 which works well for high-dimensional normalized embeddings.
    pub fn with_rbf(gamma: Option<f64>) -> Self {
        Self {
            classifiers: Vec::new(),
            model_names: Vec::new(),
            trained: false,
            kernel_type: KernelType::Rbf,
            gamma: gamma.unwrap_or(1.0), // 1.0 is good for high-dim normalized embeddings
        }
    }

    fn normalize_vector(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v.to_vec()
        }
    }

    /// Train one-vs-all SVM classifiers
    /// Uses quality + latency weighted training:
    /// - High quality records have more influence on decision boundaries
    /// - Low latency records have more influence (faster = better)
    /// Implemented via oversampling: duplicate records proportional to their weight
    pub fn train(&mut self, records: Vec<SVMTrainingRecord>) -> Result<(), String> {
        if records.is_empty() {
            return Err("No training records provided".to_string());
        }

        // Calculate weights based on quality and latency
        // Formula: weight = 0.9 * quality + 0.1 * speed_factor
        // This matches the global QualityWeight=0.9 hyperparameter (90% quality, 10% speed)
        let max_latency = records.iter().map(|r| r.latency_ns).max().unwrap_or(1) as f64;
        let min_latency = records.iter().map(|r| r.latency_ns).min().unwrap_or(1) as f64;
        let latency_range = (max_latency - min_latency).max(1.0);

        const QUALITY_WEIGHT: f64 = 0.9;  // 90% quality
        const SPEED_WEIGHT: f64 = 0.1;    // 10% speed

        // Compute weights for all records
        let weights: Vec<f64> = records.iter().map(|r| {
            let normalized_latency = (r.latency_ns as f64 - min_latency) / latency_range;
            let speed_factor = 1.0 - normalized_latency; // 1.0 for fastest, 0.0 for slowest
            let weight = QUALITY_WEIGHT * r.quality + SPEED_WEIGHT * speed_factor;
            weight.max(0.01) // Minimum weight to include all samples
        }).collect();

        // Normalize weights to [1, 5] range for oversampling
        let max_weight = weights.iter().cloned().fold(0.0_f64, f64::max);
        let min_weight = weights.iter().cloned().fold(f64::MAX, f64::min);
        let weight_range = (max_weight - min_weight).max(0.01);

        // Build weighted training set via oversampling
        // Higher weight = more copies of the record
        let mut weighted_records: Vec<&SVMTrainingRecord> = Vec::new();
        for (i, record) in records.iter().enumerate() {
            let normalized_weight = (weights[i] - min_weight) / weight_range;
            let copies = (1.0 + normalized_weight * 4.0).round() as usize; // 1-5 copies
            for _ in 0..copies {
                weighted_records.push(record);
            }
        }

        eprintln!("SVM training: {} records -> {} weighted samples (quality+latency weighting)",
            records.len(), weighted_records.len());

        if weighted_records.is_empty() {
            return Err("No training records after weighting".to_string());
        }

        let dim = weighted_records[0].embedding.len();
        let n = weighted_records.len();

        // For normalized embeddings (L2 norm = 1), squared distance between
        // orthogonal vectors ≈ 2. Using gamma=1.0 gives exp(-1.0*2)≈0.14 for
        // orthogonal vectors, providing sharper class boundaries.
        // gamma=1.0 is good for high-dimensional data where vectors tend to be
        // nearly orthogonal due to curse of dimensionality.

        // Normalize embeddings
        let normalized: Vec<Vec<f64>> = weighted_records
            .iter()
            .map(|r| Self::normalize_vector(&r.embedding))
            .collect();

        // Build training matrix and labels
        let mut data = Vec::with_capacity(n * dim);
        let mut labels: Vec<String> = Vec::with_capacity(n);

        for (i, record) in weighted_records.iter().enumerate() {
            if record.embedding.len() != dim {
                return Err("Inconsistent embedding dimensions".to_string());
            }
            data.extend(normalized[i].iter().cloned());
            labels.push(record.model.clone());
        }

        let embeddings = Array2::from_shape_vec((n, dim), data)
            .map_err(|e| format!("Failed to create embeddings matrix: {}", e))?;

        // Get unique model names
        let unique_models: Vec<String> = {
            let mut models: Vec<String> = labels.iter().cloned().collect();
            models.sort();
            models.dedup();
            models
        };
        self.model_names = unique_models.clone();
        self.classifiers.clear();

        // Train one-vs-all classifiers
        for model_name in &unique_models {
            let binary_labels: Vec<bool> = labels.iter().map(|m| m == model_name).collect();
            let labels_arr = Array1::from_vec(binary_labels);
            let dataset = Dataset::new(embeddings.clone(), labels_arr);

            // Train with appropriate kernel
            let train_result = match self.kernel_type {
                KernelType::Linear => Svm::<_, bool>::params().linear_kernel().fit(&dataset),
                KernelType::Rbf => Svm::<_, bool>::params()
                    .gaussian_kernel(self.gamma)
                    .fit(&dataset),
            };

            match train_result {
                Ok(svm) => {
                    let alpha = svm.alpha.clone();
                    let rho = svm.rho;

                    match self.kernel_type {
                        KernelType::Linear => {
                            // Compute weight vector: w = Σ(αᵢ·xᵢ)
                            let mut weights = Array1::zeros(dim);
                            for (i, &alpha_i) in alpha.iter().enumerate() {
                                let x_i = embeddings.row(i);
                                weights = weights + &(x_i.to_owned() * alpha_i);
                            }

                            self.classifiers.push(Classifier::Linear(LinearClassifier {
                                model_name: model_name.clone(),
                                weights,
                                rho,
                            }));
                        }
                        KernelType::Rbf => {
                            // For RBF, we need to store all training samples
                            self.classifiers.push(Classifier::Rbf(RbfClassifier {
                                model_name: model_name.clone(),
                                alpha,
                                support_vectors: embeddings.clone(),
                                rho,
                                gamma: self.gamma,
                            }));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Warning: SVM training failed for {}: {}", model_name, e);
                }
            }
        }

        self.trained = !self.classifiers.is_empty();

        if self.trained {
            Ok(())
        } else {
            Err("Failed to train any SVM classifiers".to_string())
        }
    }

    /// Select best model using SVM decision function scores
    pub fn select(&self, query: &[f64]) -> Result<String, String> {
        if !self.trained {
            return Err("Model not trained".to_string());
        }

        if self.classifiers.is_empty() {
            return self
                .model_names
                .first()
                .cloned()
                .ok_or_else(|| "No models available".to_string());
        }

        let normalized_query = Self::normalize_vector(query);
        let query_arr = Array1::from_vec(normalized_query);

        let mut best_model = String::new();
        let mut best_score = f64::NEG_INFINITY;

        for classifier in &self.classifiers {
            let score = classifier.decision_function(&query_arr);

            if score > best_score {
                best_score = score;
                best_model = classifier.model_name().to_string();
            }
        }

        if !best_model.is_empty() {
            Ok(best_model)
        } else {
            self.model_names
                .first()
                .cloned()
                .ok_or_else(|| "No model selected".to_string())
        }
    }

    pub fn is_trained(&self) -> bool {
        self.trained
    }

    pub fn kernel_type(&self) -> KernelType {
        self.kernel_type
    }

    /// Save model to JSON
    pub fn to_json(&self) -> Result<String, String> {
        let mut linear_classifiers = Vec::new();
        let mut rbf_classifiers = Vec::new();

        for classifier in &self.classifiers {
            match classifier {
                Classifier::Linear(c) => {
                    linear_classifiers.push(LinearClassifierData {
                        model_name: c.model_name.clone(),
                        weights: c.weights.to_vec(),
                        rho: c.rho,
                    });
                }
                Classifier::Rbf(c) => {
                    rbf_classifiers.push(RbfClassifierData {
                        model_name: c.model_name.clone(),
                        alpha: c.alpha.clone(),
                        support_vectors: c
                            .support_vectors
                            .rows()
                            .into_iter()
                            .map(|r| r.to_vec())
                            .collect(),
                        rho: c.rho,
                        gamma: c.gamma,
                    });
                }
            }
        }

        let data = SVMModelData {
            algorithm: "svm".to_string(),
            trained: self.trained,
            model_names: self.model_names.clone(),
            kernel_type: self.kernel_type,
            gamma: self.gamma,
            linear_classifiers,
            rbf_classifiers,
        };

        serde_json::to_string_pretty(&data)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Load model from JSON (no retraining needed!)
    pub fn from_json(json: &str) -> Result<Self, String> {
        let data: SVMModelData =
            serde_json::from_str(json).map_err(|e| format!("JSON parse failed: {}", e))?;

        let mut classifiers = Vec::new();

        // Load linear classifiers
        for c in data.linear_classifiers {
            classifiers.push(Classifier::Linear(LinearClassifier {
                model_name: c.model_name,
                weights: Array1::from_vec(c.weights),
                rho: c.rho,
            }));
        }

        // Load RBF classifiers
        for c in data.rbf_classifiers {
            let n = c.support_vectors.len();
            let dim = if n > 0 { c.support_vectors[0].len() } else { 0 };
            let flat: Vec<f64> = c.support_vectors.into_iter().flatten().collect();
            let support_vectors = if n > 0 && dim > 0 {
                Array2::from_shape_vec((n, dim), flat).unwrap_or_else(|_| Array2::zeros((0, 0)))
            } else {
                Array2::zeros((0, 0))
            };

            classifiers.push(Classifier::Rbf(RbfClassifier {
                model_name: c.model_name,
                alpha: c.alpha,
                support_vectors,
                rho: c.rho,
                gamma: c.gamma,
            }));
        }

        Ok(Self {
            classifiers,
            model_names: data.model_names,
            trained: data.trained,
            kernel_type: data.kernel_type,
            gamma: data.gamma,
        })
    }
}

impl Default for SVMSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svm_linear_basic() {
        let mut selector = SVMSelector::with_kernel(KernelType::Linear, 1.0); // Explicit Linear kernel

        let records = vec![
            SVMTrainingRecord { embedding: vec![1.0, 0.0, 0.0], model: "model-a".to_string(), quality: 0.9, latency_ns: 100 },
            SVMTrainingRecord { embedding: vec![0.9, 0.1, 0.0], model: "model-a".to_string(), quality: 0.85, latency_ns: 110 },
            SVMTrainingRecord { embedding: vec![0.8, 0.2, 0.0], model: "model-a".to_string(), quality: 0.88, latency_ns: 105 },
            SVMTrainingRecord { embedding: vec![0.0, 1.0, 0.0], model: "model-b".to_string(), quality: 0.92, latency_ns: 200 },
            SVMTrainingRecord { embedding: vec![0.1, 0.9, 0.0], model: "model-b".to_string(), quality: 0.90, latency_ns: 190 },
            SVMTrainingRecord { embedding: vec![0.2, 0.8, 0.0], model: "model-b".to_string(), quality: 0.87, latency_ns: 210 },
            SVMTrainingRecord { embedding: vec![0.0, 0.0, 1.0], model: "model-c".to_string(), quality: 0.95, latency_ns: 150 },
            SVMTrainingRecord { embedding: vec![0.0, 0.1, 0.9], model: "model-c".to_string(), quality: 0.93, latency_ns: 160 },
            SVMTrainingRecord { embedding: vec![0.0, 0.2, 0.8], model: "model-c".to_string(), quality: 0.91, latency_ns: 155 },
        ];

        selector.train(records).unwrap();
        assert!(selector.is_trained());
        assert_eq!(selector.kernel_type(), KernelType::Linear);

        let result_a = selector.select(&[0.95, 0.05, 0.0]).unwrap();
        let result_b = selector.select(&[0.05, 0.95, 0.0]).unwrap();
        let result_c = selector.select(&[0.0, 0.05, 0.95]).unwrap();

        assert_eq!(result_a, "model-a");
        assert_eq!(result_b, "model-b");
        assert_eq!(result_c, "model-c");
    }

    #[test]
    fn test_svm_rbf_basic() {
        let mut selector = SVMSelector::with_kernel(KernelType::Rbf, 1.0);

        let records = vec![
            SVMTrainingRecord { embedding: vec![1.0, 0.0, 0.0], model: "model-a".to_string(), quality: 0.9, latency_ns: 100 },
            SVMTrainingRecord { embedding: vec![0.9, 0.1, 0.0], model: "model-a".to_string(), quality: 0.85, latency_ns: 110 },
            SVMTrainingRecord { embedding: vec![0.8, 0.2, 0.0], model: "model-a".to_string(), quality: 0.88, latency_ns: 105 },
            SVMTrainingRecord { embedding: vec![0.0, 1.0, 0.0], model: "model-b".to_string(), quality: 0.92, latency_ns: 200 },
            SVMTrainingRecord { embedding: vec![0.1, 0.9, 0.0], model: "model-b".to_string(), quality: 0.90, latency_ns: 190 },
            SVMTrainingRecord { embedding: vec![0.2, 0.8, 0.0], model: "model-b".to_string(), quality: 0.87, latency_ns: 210 },
            SVMTrainingRecord { embedding: vec![0.0, 0.0, 1.0], model: "model-c".to_string(), quality: 0.95, latency_ns: 150 },
            SVMTrainingRecord { embedding: vec![0.0, 0.1, 0.9], model: "model-c".to_string(), quality: 0.93, latency_ns: 160 },
            SVMTrainingRecord { embedding: vec![0.0, 0.2, 0.8], model: "model-c".to_string(), quality: 0.91, latency_ns: 155 },
        ];

        selector.train(records).unwrap();
        assert!(selector.is_trained());
        assert_eq!(selector.kernel_type(), KernelType::Rbf);

        // RBF should also classify well-separated data
        let result_a = selector.select(&[0.95, 0.05, 0.0]).unwrap();
        let result_b = selector.select(&[0.05, 0.95, 0.0]).unwrap();
        let result_c = selector.select(&[0.0, 0.05, 0.95]).unwrap();

        assert_eq!(result_a, "model-a");
        assert_eq!(result_b, "model-b");
        assert_eq!(result_c, "model-c");
    }

    #[test]
    fn test_svm_quality_filtering() {
        let mut selector = SVMSelector::new();

        // Mix of high and low quality samples
        // Low quality samples (quality < 0.5) should be filtered out
        let records = vec![
            SVMTrainingRecord { embedding: vec![1.0, 0.0], model: "good-model".to_string(), quality: 0.9, latency_ns: 100 },
            SVMTrainingRecord { embedding: vec![0.9, 0.1], model: "good-model".to_string(), quality: 0.85, latency_ns: 110 },
            SVMTrainingRecord { embedding: vec![0.8, 0.2], model: "good-model".to_string(), quality: 0.88, latency_ns: 105 },
            SVMTrainingRecord { embedding: vec![0.0, 1.0], model: "good-model".to_string(), quality: 0.92, latency_ns: 200 },
            SVMTrainingRecord { embedding: vec![0.1, 0.9], model: "good-model".to_string(), quality: 0.90, latency_ns: 190 },
            SVMTrainingRecord { embedding: vec![0.2, 0.8], model: "good-model".to_string(), quality: 0.87, latency_ns: 210 },
            // These low-quality samples should be filtered
            SVMTrainingRecord { embedding: vec![0.5, 0.5], model: "bad-model".to_string(), quality: 0.1, latency_ns: 500 },
            SVMTrainingRecord { embedding: vec![0.4, 0.6], model: "bad-model".to_string(), quality: 0.2, latency_ns: 600 },
            SVMTrainingRecord { embedding: vec![0.6, 0.4], model: "bad-model".to_string(), quality: 0.15, latency_ns: 550 },
            // Add more high-quality to ensure we have enough
            SVMTrainingRecord { embedding: vec![0.7, 0.3], model: "good-model".to_string(), quality: 0.8, latency_ns: 120 },
            SVMTrainingRecord { embedding: vec![0.3, 0.7], model: "good-model".to_string(), quality: 0.82, latency_ns: 130 },
            SVMTrainingRecord { embedding: vec![0.95, 0.05], model: "good-model".to_string(), quality: 0.95, latency_ns: 90 },
        ];

        selector.train(records).unwrap();
        assert!(selector.is_trained());

        // Should predict good-model because bad-model samples were filtered
        let result = selector.select(&[0.5, 0.5]).unwrap();
        assert_eq!(result, "good-model", "Quality filtering should exclude low-quality samples");
    }

    #[test]
    fn test_svm_linear_serialization() {
        let mut selector = SVMSelector::with_kernel(KernelType::Linear, 1.0); // Explicit Linear for this test

        let records = vec![
            SVMTrainingRecord { embedding: vec![1.0, 0.0], model: "a".to_string(), quality: 0.9, latency_ns: 100 },
            SVMTrainingRecord { embedding: vec![0.9, 0.1], model: "a".to_string(), quality: 0.85, latency_ns: 110 },
            SVMTrainingRecord { embedding: vec![0.8, 0.2], model: "a".to_string(), quality: 0.88, latency_ns: 105 },
            SVMTrainingRecord { embedding: vec![0.0, 1.0], model: "b".to_string(), quality: 0.92, latency_ns: 200 },
            SVMTrainingRecord { embedding: vec![0.1, 0.9], model: "b".to_string(), quality: 0.90, latency_ns: 190 },
            SVMTrainingRecord { embedding: vec![0.2, 0.8], model: "b".to_string(), quality: 0.87, latency_ns: 210 },
        ];

        selector.train(records).unwrap();
        let json = selector.to_json().unwrap();
        let loaded = SVMSelector::from_json(&json).unwrap();

        assert!(loaded.is_trained());
        assert_eq!(loaded.kernel_type(), KernelType::Linear);
        assert_eq!(
            selector.select(&[0.95, 0.05]).unwrap(),
            loaded.select(&[0.95, 0.05]).unwrap()
        );
    }

    #[test]
    fn test_svm_rbf_serialization() {
        let mut selector = SVMSelector::with_kernel(KernelType::Rbf, 0.5);

        let records = vec![
            SVMTrainingRecord { embedding: vec![1.0, 0.0], model: "a".to_string(), quality: 0.9, latency_ns: 100 },
            SVMTrainingRecord { embedding: vec![0.9, 0.1], model: "a".to_string(), quality: 0.85, latency_ns: 110 },
            SVMTrainingRecord { embedding: vec![0.8, 0.2], model: "a".to_string(), quality: 0.88, latency_ns: 105 },
            SVMTrainingRecord { embedding: vec![0.0, 1.0], model: "b".to_string(), quality: 0.92, latency_ns: 200 },
            SVMTrainingRecord { embedding: vec![0.1, 0.9], model: "b".to_string(), quality: 0.90, latency_ns: 190 },
            SVMTrainingRecord { embedding: vec![0.2, 0.8], model: "b".to_string(), quality: 0.87, latency_ns: 210 },
        ];

        selector.train(records).unwrap();
        let json = selector.to_json().unwrap();
        let loaded = SVMSelector::from_json(&json).unwrap();

        assert!(loaded.is_trained());
        assert_eq!(loaded.kernel_type(), KernelType::Rbf);
        assert_eq!(
            selector.select(&[0.95, 0.05]).unwrap(),
            loaded.select(&[0.95, 0.05]).unwrap()
        );
    }

    #[test]
    fn test_svm_multiclass() {
        let mut selector = SVMSelector::new(); // Uses default RBF kernel
        let mut records = Vec::new();

        for i in 0..20 {
            let offset = (i as f64) * 0.05;
            let quality = 0.8 + (i as f64) * 0.01; // Varying quality
            records.push(SVMTrainingRecord {
                embedding: vec![1.0 + offset, 1.0 + offset],
                model: "top-right".to_string(),
                quality,
                latency_ns: 100 + i * 10,
            });
            records.push(SVMTrainingRecord {
                embedding: vec![-1.0 - offset, 1.0 + offset],
                model: "top-left".to_string(),
                quality,
                latency_ns: 100 + i * 10,
            });
            records.push(SVMTrainingRecord {
                embedding: vec![-1.0 - offset, -1.0 - offset],
                model: "bottom-left".to_string(),
                quality,
                latency_ns: 100 + i * 10,
            });
            records.push(SVMTrainingRecord {
                embedding: vec![1.0 + offset, -1.0 - offset],
                model: "bottom-right".to_string(),
                quality,
                latency_ns: 100 + i * 10,
            });
        }

        selector.train(records).unwrap();
        assert!(selector.is_trained());

        assert_eq!(selector.select(&[5.0, 5.0]).unwrap(), "top-right");
        assert_eq!(selector.select(&[-5.0, 5.0]).unwrap(), "top-left");
        assert_eq!(selector.select(&[-5.0, -5.0]).unwrap(), "bottom-left");
        assert_eq!(selector.select(&[5.0, -5.0]).unwrap(), "bottom-right");
    }
}
