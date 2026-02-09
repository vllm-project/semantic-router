//! ML Binding for Semantic Router
//!
//! Inference-only library for ML-based model selection.
//!
//! ## Architecture
//! - **Training**: Done in Python (src/training/ml_model_selection/) using scikit-learn/PyTorch
//! - **Inference**: Done in Rust via FFI to Go, using linfa-nn and candle for efficient inference
//!
//! ## Algorithms
//! - KNN (K-Nearest Neighbors): Quality-weighted voting among neighbors
//! - KMeans: Nearest centroid lookup with pre-trained cluster assignments
//! - SVM: Decision function scoring with Linear or RBF kernels
//! - MLP (Multi-Layer Perceptron): GPU-accelerated neural network via Candle
//!
//! Reference: FusionFactory (arXiv:2507.10540) - Query-level fusion via tailored LLM routers
//!
//! Models are loaded from JSON files trained by the Python scripts.

pub mod knn;
pub mod kmeans;
pub mod svm;
pub mod mlp;
pub mod ffi;

// Re-exports for convenience
pub use knn::KNNSelector;
pub use kmeans::KMeansSelector;
pub use svm::SVMSelector;
pub use mlp::MLPSelector;
