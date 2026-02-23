//! Local copy of ModernBERT from candle-transformers with Flash Attention support
//!
//! This module contains a local copy of ModernBERT implementation from candle-transformers
//! with added Flash Attention support for improved performance on long sequences.

pub mod modernbert;

// Re-export the main types
pub use modernbert::{
    ClassifierConfig, ClassifierPooling, Config, ModernBert, ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
};
