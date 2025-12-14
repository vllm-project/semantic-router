//! Generative Model Architectures
//!
//! This module contains the Qwen3 Multi-LoRA adapter system for
//! generative classification with dynamic adapter switching.
//!
//! ## Models
//! - **Qwen3MultiLoRAClassifier**: Multi-adapter system with official Qwen3 + LoRA
//!
//! ## Features
//! - Multiple LoRA adapters per base model
//! - Dynamic adapter switching (no model reload required)
//! - Category classification via generative inference
//! - ChatML format prompt support
//! - Efficient KV cache management
//! - Thread-safe FFI interface for Go

pub mod continuous_batch_scheduler;
pub mod qwen3_guard;
pub mod qwen3_multi_lora_classifier;
pub mod qwen3_with_lora;

pub use continuous_batch_scheduler::{BatchSchedulerConfig, ContinuousBatchScheduler};
pub use qwen3_guard::{GuardGenerationResult, Qwen3GuardConfig, Qwen3GuardModel};
pub use qwen3_multi_lora_classifier::{
    AdapterLabelMapping, MultiAdapterClassificationResult, Qwen3MultiLoRAClassifier,
};
