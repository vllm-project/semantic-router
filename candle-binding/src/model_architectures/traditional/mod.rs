//! Traditional Fine-Tuning Models
//!
//! This module provides traditional fine-tuned encoder models including:
//! - BERT: Classic bidirectional encoder
//! - DeBERTa v3: Enhanced disentangled attention
//! - ModernBERT: Modern architecture with Flash Attention (supports both standard and mmBERT variants)
//!
//! mmBERT (multilingual ModernBERT) is supported through the ModernBERT implementation
//! using `ModernBertVariant::Multilingual` or the `MmBertClassifier` type alias.

#![allow(dead_code)]
#![allow(unused_imports)]

// Traditional model modules
pub mod bert;
pub mod deberta_v3;

pub mod base_model;
pub mod modernbert;

// Local copy of candle-transformers models with Flash Attention support
pub mod candle_models;

// Re-export main traditional models
pub use bert::TraditionalBertClassifier;
pub use deberta_v3::DebertaV3Classifier;

// Re-export ModernBERT and mmBERT (mmBERT is a type alias for ModernBERT with Multilingual variant)
pub use modernbert::{
    MmBertClassifier, MmBertTokenClassifier, ModernBertVariant, TraditionalModernBertClassifier,
    TraditionalModernBertTokenClassifier,
};

// Re-export local candle_models (ModernBERT with Flash Attention support)
// Note: candle_models/mod.rs already re-exports these from modernbert
pub use candle_models::{
    Config, ModernBert, ModernBertForMaskedLM, ModernBertForSequenceClassification,
};

// Re-export traditional models
pub use base_model::*;

// Test modules (only compiled in test builds)
#[cfg(test)]
pub mod base_model_test;
#[cfg(test)]
pub mod bert_test;
#[cfg(test)]
pub mod deberta_v3_test;
#[cfg(test)]
pub mod modernbert_test;
