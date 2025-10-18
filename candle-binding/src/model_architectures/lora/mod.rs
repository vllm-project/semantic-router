//! LoRA (Low-Rank Adaptation) Models
//!
//! This module contains LoRA-based parameter-efficient fine-tuning implementations.
//! These models provide high-performance processing with ultra-high confidence.

#![allow(dead_code)]

// Core LoRA modules
pub mod bert_lora;
pub mod lora_adapter;

// Re-export main LoRA models
pub use bert_lora::{LoRABertClassifier, LoRAMultiTaskResult};

// Re-export LoRA adapter functionality
pub use lora_adapter::*;

// Test modules (only compiled in test builds)
#[cfg(test)]
pub mod bert_lora_test;
#[cfg(test)]
pub mod lora_adapter_test;
