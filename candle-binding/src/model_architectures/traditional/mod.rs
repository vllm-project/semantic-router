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
// IMPORTANT: This local copy is necessary because:
// 1. Flash Attention 2 support: The upstream candle-transformers ModernBERT doesn't support
//    Flash Attention 2, which is required for efficient 32K context processing
// 2. Extended context support: We need custom handling for Extended32K variant with YaRN RoPE scaling
// 3. Runtime max_position_embeddings override: We need to override config values at runtime
//    for Extended32K models, which requires mutable Config fields
//
// This creates a type shadowing situation where local types (Config, ModernBert, etc.) are used
// instead of upstream types. This is intentional but should be documented.
//
// Using upstream Config + extend only ModernBertAttention it's not possible because upstream types have private fields.
//   - ModernBertAttention is `struct` (not `pub struct`) - cannot access it
//   - ModernBertLayer is also private - cannot replace attention layer
//   - ModernBert contains `layers: Vec<ModernBertLayer>` - cannot replace layers
//
// To sync with upstream changes:
//   1. Check upstream: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/modernbert.rs
//   2. Compare with local: candle-binding/src/model_architectures/traditional/candle_models/modernbert.rs
//   3. Manually merge upstream changes while preserving Flash Attention support
//   4. Run tests to ensure compatibility
//
// If upstream candle-transformers adds Flash Attention 2 support, we should consider migrating back.
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
// NOTE: These types shadow upstream candle_transformers::models::modernbert types.
// This is intentional - see comment above about why we need a local copy.
// Code should use these re-exported types, not direct imports from candle_transformers.
pub use candle_models::{
    ClassifierConfig, ClassifierPooling, Config, ModernBert, ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
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
