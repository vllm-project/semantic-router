//! Unified error types for the ONNX binding library

use std::fmt;

/// Unified error type for the library
#[derive(Debug)]
pub enum UnifiedError {
    /// Model loading or initialization error
    ModelLoad { model_path: String, source: String },
    /// Configuration error
    Config { field: String, message: String },
    /// Inference error
    Inference { operation: String, source: String },
    /// Tokenization error
    Tokenization { source: String },
    /// The tokenizer output exceeds the selected embedding model's context.
    InputTooLong {
        model: String,
        count: usize,
        maximum: usize,
    },
    /// Validation error
    Validation {
        field: String,
        expected: String,
        actual: String,
    },
    /// File not found
    FileNotFound { path: String },
    /// Invalid JSON
    InvalidJson { path: String, source: String },
    /// ONNX Runtime error
    OrtError { source: String },
    /// Generic error
    Other { message: String },
}

impl fmt::Display for UnifiedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnifiedError::ModelLoad { model_path, source } => {
                write!(f, "Failed to load model from '{}': {}", model_path, source)
            }
            UnifiedError::Config { field, message } => {
                write!(f, "Configuration error for '{}': {}", field, message)
            }
            UnifiedError::Inference { operation, source } => {
                write!(f, "Inference error during '{}': {}", operation, source)
            }
            UnifiedError::Tokenization { source } => {
                write!(f, "Tokenization error: {}", source)
            }
            UnifiedError::InputTooLong {
                model,
                count,
                maximum,
            } => write!(
                f,
                "{} input has {} tokens; maximum context is {}",
                model, count, maximum
            ),
            UnifiedError::Validation {
                field,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Validation error for '{}': expected {}, got {}",
                    field, expected, actual
                )
            }
            UnifiedError::FileNotFound { path } => {
                write!(f, "File not found: {}", path)
            }
            UnifiedError::InvalidJson { path, source } => {
                write!(f, "Invalid JSON in '{}': {}", path, source)
            }
            UnifiedError::OrtError { source } => {
                write!(f, "ONNX Runtime error: {}", source)
            }
            UnifiedError::Other { message } => {
                write!(f, "{}", message)
            }
        }
    }
}

impl std::error::Error for UnifiedError {}

impl UnifiedError {
    /// Returns true only for request input that exceeds a model context window.
    pub fn is_input_too_long(&self) -> bool {
        matches!(self, Self::InputTooLong { .. })
    }
}

/// Result type alias using UnifiedError
pub type UnifiedResult<T> = Result<T, UnifiedError>;

/// Helper functions for creating errors
pub mod errors {
    use super::UnifiedError;

    pub fn model_load(model_path: &str, source: &str) -> UnifiedError {
        UnifiedError::ModelLoad {
            model_path: model_path.to_string(),
            source: source.to_string(),
        }
    }

    pub fn config_error(field: &str, message: &str) -> UnifiedError {
        UnifiedError::Config {
            field: field.to_string(),
            message: message.to_string(),
        }
    }

    pub fn inference_error(operation: &str, source: &str) -> UnifiedError {
        UnifiedError::Inference {
            operation: operation.to_string(),
            source: source.to_string(),
        }
    }

    pub fn tokenization_error(source: &str) -> UnifiedError {
        UnifiedError::Tokenization {
            source: source.to_string(),
        }
    }

    pub fn input_too_long(model: &str, count: usize, maximum: usize) -> UnifiedError {
        UnifiedError::InputTooLong {
            model: model.to_string(),
            count,
            maximum,
        }
    }

    pub fn file_not_found(path: &str) -> UnifiedError {
        UnifiedError::FileNotFound {
            path: path.to_string(),
        }
    }

    pub fn invalid_json(path: &str, source: &str) -> UnifiedError {
        UnifiedError::InvalidJson {
            path: path.to_string(),
            source: source.to_string(),
        }
    }

    pub fn ort_error(source: &str) -> UnifiedError {
        UnifiedError::OrtError {
            source: source.to_string(),
        }
    }

    pub fn validation(field: &str, expected: &str, actual: &str) -> UnifiedError {
        UnifiedError::Validation {
            field: field.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_too_long_is_the_only_context_limit_error() {
        let limit = errors::input_too_long("mmBERT", 33, 32);
        assert!(limit.is_input_too_long());
        assert_eq!(
            limit.to_string(),
            "mmBERT input has 33 tokens; maximum context is 32"
        );

        assert!(!errors::tokenization_error("invalid input").is_input_too_long());
        assert!(!errors::inference_error("run", "failure").is_input_too_long());
    }
}
