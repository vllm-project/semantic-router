//! # FFI (Foreign Function Interface) Module

#![allow(dead_code)]

// FFI modules
pub mod classify; //  classification functions
pub mod embedding; //  embedding functions
pub mod generative_classifier; // Qwen3 LoRA generative classifier
pub mod init; //  initialization functions
pub mod memory; //  memory management functions
pub mod similarity; //  similarity functions
pub mod tokenization; //  tokenization function
pub mod types; //  C structure definitions
pub mod validation; //  parameter validation functions

pub mod memory_safety; // Dual-path memory safety system
pub mod state_manager; // Global state management system

// Re-export types and functions
pub use classify::*;
pub use embedding::*; // Intelligent embedding functions
pub use generative_classifier::*; // Qwen3 LoRA generative classifier functions
pub use init::*;
pub use memory::*;

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
