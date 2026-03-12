//! # FFI (Foreign Function Interface) Module

#![allow(dead_code)]

// FFI modules
pub mod classify; //  core classification functions
pub mod classify_bert; //  BERT/LoRA text classifiers
pub mod classify_hallucination; //  hallucination detection + NLI
pub mod classify_mmbert; //  mmBERT 32K classifiers
pub mod classify_modernbert; //  ModernBERT + DeBERTa + fact-check + feedback
pub mod embedding; //  embedding functions
pub mod embedding_multimodal; //  batched embedding + multimodal
pub mod embedding_routing; //  smart routing and model-type embedding
pub mod embedding_similarity; //  similarity calculations + models info
pub mod generative_classifier; // Qwen3 LoRA generative classifier
pub mod init; //  initialization functions (core + modernbert)
pub mod init_classifiers; //  specialized classifier initialization
pub mod init_mmbert; //  mmBERT initialization functions
pub mod memory; //  memory management functions
pub mod mlp; // MLP selector for model selection (GPU-accelerated)
pub mod similarity; //  similarity functions
pub mod tokenization; //  tokenization function
pub mod types; //  C structure definitions
pub mod validation; //  parameter validation functions

pub mod memory_safety; // Dual-path memory safety system
pub mod state_manager; // Global state management system

// Re-export types and functions
pub use classify::*;
pub use classify_bert::*;
pub use classify_hallucination::*;
pub use classify_mmbert::*;
pub use classify_modernbert::*;
pub use embedding::*; // Intelligent embedding functions
pub use embedding_multimodal::*; // Batched embedding + multimodal
pub use embedding_routing::*; // Smart routing and model-type embedding
pub use embedding_similarity::*; // Similarity calculations + models info
pub use generative_classifier::*; // Qwen3 LoRA generative classifier functions
pub use init::*;
pub use init_classifiers::*;
pub use init_mmbert::*;
pub use memory::*;
pub use mlp::*; // MLP selector FFI functions

pub use similarity::*;
pub use tokenization::*;
pub use types::*;
pub use validation::*;

pub use memory_safety::*;
pub use state_manager::*;

#[cfg(test)]
pub mod classify_test;
#[cfg(test)]
pub mod embedding_test;
#[cfg(test)]
pub mod init_test;
#[cfg(test)]
pub mod memory_safety_test;
#[cfg(test)]
pub mod oncelock_concurrent_test;
#[cfg(test)]
pub mod state_manager_test;
#[cfg(test)]
pub mod validation_test;
