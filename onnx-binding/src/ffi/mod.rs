//! Foreign Function Interface (FFI) for Go bindings

pub mod embedding;
pub mod memory;
pub mod types;

pub use embedding::*;
pub use memory::*;
pub use types::*;
