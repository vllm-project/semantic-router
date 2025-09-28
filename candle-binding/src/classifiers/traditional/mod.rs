//! Traditional Classifiers
//!
//! This module contains traditional classification implementations that provide
//! stable, reliable performance with full backward compatibility.

#![allow(dead_code)]

// Traditional classifier modules
pub mod batch_processor;
pub mod modernbert_classifier;

// Re-export classifier types
pub use batch_processor::*;
pub use modernbert_classifier::*;
