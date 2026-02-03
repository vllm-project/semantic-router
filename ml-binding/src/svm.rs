//! SVM (Support Vector Machine) inference implementation
//!
//! Inference-only implementation. Training is done in Python (src/training/ml_model_selection/).
//! Models are loaded from JSON files trained by the Python scripts.
//!
//! Supports both Linear and RBF kernels for one-vs-all multiclass classification.
//!
//! - Linear kernel: f(x) = w·x - rho (fast, good for high-dim data)
//! - RBF kernel: f(x) = Σ(αᵢ·exp(-γ||x-xᵢ||²)) - rho (flexible boundaries)

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
    fn test_svm_linear_load_and_select() {
        // Pre-trained linear SVM model
        let json = r#"{
            "algorithm": "svm",
            "trained": true,
            "model_names": ["model-a", "model-b", "model-c"],
            "kernel_type": "Linear",
            "gamma": 1.0,
            "linear_classifiers": [
                {"model_name": "model-a", "weights": [1.0, 0.0, 0.0], "rho": 0.0},
                {"model_name": "model-b", "weights": [0.0, 1.0, 0.0], "rho": 0.0},
                {"model_name": "model-c", "weights": [0.0, 0.0, 1.0], "rho": 0.0}
            ],
            "rbf_classifiers": []
        }"#;

        let selector = SVMSelector::from_json(json).unwrap();
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
    fn test_svm_rbf_load_and_select() {
        // Pre-trained RBF SVM model with support vectors
        let json = r#"{
            "algorithm": "svm",
            "trained": true,
            "model_names": ["model-a", "model-b"],
            "kernel_type": "Rbf",
            "gamma": 1.0,
            "linear_classifiers": [],
            "rbf_classifiers": [
                {
                    "model_name": "model-a",
                    "alpha": [1.0],
                    "support_vectors": [[1.0, 0.0]],
                    "rho": 0.0,
                    "gamma": 1.0
                },
                {
                    "model_name": "model-b",
                    "alpha": [1.0],
                    "support_vectors": [[0.0, 1.0]],
                    "rho": 0.0,
                    "gamma": 1.0
                }
            ]
        }"#;

        let selector = SVMSelector::from_json(json).unwrap();
        assert!(selector.is_trained());
        assert_eq!(selector.kernel_type(), KernelType::Rbf);

        // Query closer to model-a's support vector
        let result_a = selector.select(&[0.95, 0.05]).unwrap();
        assert_eq!(result_a, "model-a");

        // Query closer to model-b's support vector
        let result_b = selector.select(&[0.05, 0.95]).unwrap();
        assert_eq!(result_b, "model-b");
    }

    #[test]
    fn test_svm_json_roundtrip() {
        let json = r#"{
            "algorithm": "svm",
            "trained": true,
            "model_names": ["a", "b"],
            "kernel_type": "Linear",
            "gamma": 1.0,
            "linear_classifiers": [
                {"model_name": "a", "weights": [1.0, 0.0], "rho": 0.1},
                {"model_name": "b", "weights": [0.0, 1.0], "rho": 0.1}
            ],
            "rbf_classifiers": []
        }"#;

        let selector = SVMSelector::from_json(json).unwrap();
        let exported = selector.to_json().unwrap();
        let loaded = SVMSelector::from_json(&exported).unwrap();

        assert!(loaded.is_trained());
        assert_eq!(loaded.kernel_type(), KernelType::Linear);
        assert_eq!(
            selector.select(&[0.95, 0.05]).unwrap(),
            loaded.select(&[0.95, 0.05]).unwrap()
        );
    }
}
