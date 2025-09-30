//! Traditional Fine-Tuning Models

#![allow(dead_code)]
#![allow(unused_imports)]

// Traditional model modules
pub mod bert;

pub mod base_model;
pub mod modernbert;
// Re-export main traditional models
pub use bert::TraditionalBertClassifier;

// Re-export traditional models
pub use base_model::*;

// Test modules (only compiled in test builds)
#[cfg(test)]
pub mod base_model_test;
#[cfg(test)]
pub mod bert_test;
#[cfg(test)]
pub mod modernbert_test;
