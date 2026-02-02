//! ML Binding for Semantic Router
//!
//! Inference-only library for ML-based model selection.
//!
//! ## Architecture
//! - **Training**: Done in Python (src/training/ml_model_selection/) using scikit-learn
//! - **Inference**: Done in Rust via FFI to Go, using linfa-nn for efficient Ball Tree lookups
//!
//! ## Algorithms
//! - KNN (K-Nearest Neighbors): Quality-weighted voting among neighbors
//! - KMeans: Nearest centroid lookup with pre-trained cluster assignments
//! - SVM: Decision function scoring with Linear or RBF kernels
//!
//! Models are loaded from JSON files trained by the Python scripts.

pub mod knn;
pub mod kmeans;
pub mod svm;
pub mod ffi;

// Re-exports for convenience
pub use knn::KNNSelector;
pub use kmeans::KMeansSelector;
pub use svm::SVMSelector;
