//! Foreign Function Interface (FFI) for Go bindings

use crate::core::unified_error::UnifiedError;

pub(super) const EMBEDDING_INPUT_TOO_LONG_STATUS: i32 = -3;

pub(super) fn embedding_error_status(error: &UnifiedError) -> i32 {
    if error.is_input_too_long() {
        EMBEDDING_INPUT_TOO_LONG_STATUS
    } else {
        -1
    }
}

mod c_string_array;
pub mod classification;
pub mod embedding;
mod init_once;
pub mod memory;
#[cfg(test)]
mod memory_test;
pub mod multimodal;
pub mod types;
pub mod unified;

pub use classification::*;
pub use embedding::*;
pub use memory::*;
pub use multimodal::*;
pub use types::*;
pub use unified::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::unified_error::errors;

    #[test]
    fn embedding_status_distinguishes_context_overflow_from_internal_errors() {
        assert_eq!(
            embedding_error_status(&errors::input_too_long("mmBERT", 33, 32)),
            EMBEDDING_INPUT_TOO_LONG_STATUS
        );
        assert_eq!(
            embedding_error_status(&errors::tokenization_error("invalid input")),
            -1
        );
        assert_eq!(
            embedding_error_status(&errors::inference_error("run", "failure")),
            -1
        );
    }
}
