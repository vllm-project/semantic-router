//! Foreign Function Interface (FFI) for Go bindings

pub mod classification;
pub mod embedding;
pub mod memory;
pub mod types;

pub use classification::*;
pub use embedding::*;
pub use memory::*;
pub use types::*;
