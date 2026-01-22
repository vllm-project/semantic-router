//! ML Binding for Semantic Router
//!
//! This library provides Linfa-based ML algorithms for model selection:
//! - KNN (K-Nearest Neighbors) via linfa-nn
//! - KMeans clustering via linfa-clustering
//! - SVM (Support Vector Machine) via linfa-svm
//!
//! MLP and Matrix Factorization remain in Go implementation.

pub mod knn;
pub mod kmeans;
pub mod svm;
pub mod ffi;

// Re-exports for convenience
pub use knn::KNNSelector;
pub use kmeans::KMeansSelector;
pub use svm::SVMSelector;
