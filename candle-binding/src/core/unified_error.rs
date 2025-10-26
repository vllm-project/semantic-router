//! Unified Error Handling System
//!
//! This module provides a comprehensive error handling system that replaces
//! scattered candle_core::Error::Msg usage with a structured, consistent approach.
//! Eliminates 50+ error handling duplication instances across the codebase.

use std::fmt;

/// Unified error type for all candle-binding operations
#[derive(Debug)]
pub enum UnifiedError {
    /// Configuration-related errors (file loading, parsing, validation)
    Configuration {
        operation: String,
        source: ConfigErrorType,
        context: Option<String>,
    },

    /// Model-related errors (loading, initialization, inference)
    Model {
        model_type: ModelErrorType,
        operation: String,
        source: String,
        context: Option<String>,
    },

    /// Processing errors (tensor operations, batch processing, computations)
    Processing {
        operation: String,
        source: String,
        input_context: Option<String>,
    },

    /// FFI-related errors (C interface, memory management)
    FFI {
        function: String,
        reason: String,
        safety_info: Option<String>,
    },

    /// I/O errors (file operations, network, device access)
    IO {
        operation: String,
        path: Option<String>,
        source: std::io::Error,
    },

    /// Validation errors (input validation, parameter checks)
    Validation {
        field: String,
        expected: String,
        actual: String,
        context: Option<String>,
    },

    /// Threading and concurrency errors
    Concurrency { operation: String, reason: String },

    /// External library errors (candle, tokenizers, etc.)
    External {
        library: String,
        operation: String,
        error: String,
    },
}

/// Configuration error subtypes
#[derive(Debug)]
pub enum ConfigErrorType {
    FileNotFound(String),
    ParseError(String),
    MissingField(String),
    InvalidData(String),
    SchemaValidation(String),
}

/// Model error subtypes
#[derive(Debug)]
pub enum ModelErrorType {
    Traditional,
    LoRA,
    ModernBERT,
    Tokenizer,
    Classifier,
    Similarity,
    Embedding, // For Qwen3/Gemma embedding models
}

impl fmt::Display for UnifiedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnifiedError::Configuration {
                operation,
                source,
                context,
            } => {
                write!(f, "Configuration error in '{}': {}", operation, source)?;
                if let Some(ctx) = context {
                    write!(f, " (context: {})", ctx)?;
                }
                Ok(())
            }
            UnifiedError::Model {
                model_type,
                operation,
                source,
                context,
            } => {
                write!(
                    f,
                    "Model error ({:?}) in '{}': {}",
                    model_type, operation, source
                )?;
                if let Some(ctx) = context {
                    write!(f, " (context: {})", ctx)?;
                }
                Ok(())
            }
            UnifiedError::Processing {
                operation,
                source,
                input_context,
            } => {
                write!(f, "Processing error in '{}': {}", operation, source)?;
                if let Some(ctx) = input_context {
                    write!(f, " (input: {})", ctx)?;
                }
                Ok(())
            }
            UnifiedError::FFI {
                function,
                reason,
                safety_info,
            } => {
                write!(f, "FFI error in '{}': {}", function, reason)?;
                if let Some(info) = safety_info {
                    write!(f, " (safety: {})", info)?;
                }
                Ok(())
            }
            UnifiedError::IO {
                operation,
                path,
                source,
            } => {
                write!(f, "I/O error in '{}': {}", operation, source)?;
                if let Some(p) = path {
                    write!(f, " (path: {})", p)?;
                }
                Ok(())
            }
            UnifiedError::Validation {
                field,
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Validation error for '{}': expected '{}', got '{}'",
                    field, expected, actual
                )?;
                if let Some(ctx) = context {
                    write!(f, " (context: {})", ctx)?;
                }
                Ok(())
            }
            UnifiedError::Concurrency { operation, reason } => {
                write!(f, "Concurrency error in '{}': {}", operation, reason)
            }
            UnifiedError::External {
                library,
                operation,
                error,
            } => {
                write!(
                    f,
                    "External error in {} during '{}': {}",
                    library, operation, error
                )
            }
        }
    }
}

impl fmt::Display for ConfigErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigErrorType::FileNotFound(path) => write!(f, "file not found: {}", path),
            ConfigErrorType::ParseError(msg) => write!(f, "parse error: {}", msg),
            ConfigErrorType::MissingField(field) => write!(f, "missing required field: {}", field),
            ConfigErrorType::InvalidData(msg) => write!(f, "invalid data: {}", msg),
            ConfigErrorType::SchemaValidation(msg) => {
                write!(f, "schema validation failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for UnifiedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            UnifiedError::IO { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Result type alias for unified error handling
pub type UnifiedResult<T> = Result<T, UnifiedError>;

/// Trait for converting errors with additional context
pub trait ErrorUnification<T> {
    /// Convert to UnifiedError with context
    fn with_config_context(self, operation: &str, context: Option<&str>) -> UnifiedResult<T>;
    fn with_model_context(
        self,
        model_type: ModelErrorType,
        operation: &str,
        context: Option<&str>,
    ) -> UnifiedResult<T>;
    fn with_processing_context(
        self,
        operation: &str,
        input_context: Option<&str>,
    ) -> UnifiedResult<T>;
    fn with_ffi_context(self, function: &str, safety_info: Option<&str>) -> UnifiedResult<T>;
}

impl<T, E> ErrorUnification<T> for Result<T, E>
where
    E: fmt::Display,
{
    fn with_config_context(self, operation: &str, context: Option<&str>) -> UnifiedResult<T> {
        self.map_err(|e| UnifiedError::Configuration {
            operation: operation.to_string(),
            source: ConfigErrorType::InvalidData(e.to_string()),
            context: context.map(|s| s.to_string()),
        })
    }

    fn with_model_context(
        self,
        model_type: ModelErrorType,
        operation: &str,
        context: Option<&str>,
    ) -> UnifiedResult<T> {
        self.map_err(|e| UnifiedError::Model {
            model_type,
            operation: operation.to_string(),
            source: e.to_string(),
            context: context.map(|s| s.to_string()),
        })
    }

    fn with_processing_context(
        self,
        operation: &str,
        input_context: Option<&str>,
    ) -> UnifiedResult<T> {
        self.map_err(|e| UnifiedError::Processing {
            operation: operation.to_string(),
            source: e.to_string(),
            input_context: input_context.map(|s| s.to_string()),
        })
    }

    fn with_ffi_context(self, function: &str, safety_info: Option<&str>) -> UnifiedResult<T> {
        self.map_err(|e| UnifiedError::FFI {
            function: function.to_string(),
            reason: e.to_string(),
            safety_info: safety_info.map(|s| s.to_string()),
        })
    }
}

/// Convert UnifiedError to candle_core::Error for backward compatibility
impl From<UnifiedError> for candle_core::Error {
    fn from(err: UnifiedError) -> Self {
        candle_core::Error::Msg(err.to_string())
    }
}

/// Convert from std::io::Error
impl From<std::io::Error> for UnifiedError {
    fn from(err: std::io::Error) -> Self {
        UnifiedError::IO {
            operation: "I/O operation".to_string(),
            path: None,
            source: err,
        }
    }
}

/// Convert from serde_json::Error
impl From<serde_json::Error> for UnifiedError {
    fn from(err: serde_json::Error) -> Self {
        UnifiedError::Configuration {
            operation: "JSON parsing".to_string(),
            source: ConfigErrorType::ParseError(err.to_string()),
            context: None,
        }
    }
}

/// Convenience macros for common error patterns

/// Create a configuration error
#[macro_export]
macro_rules! config_error {
    ($operation:expr, $msg:expr) => {
        UnifiedError::Configuration {
            operation: $operation.to_string(),
            source: ConfigErrorType::InvalidData($msg.to_string()),
            context: None,
        }
    };
    ($operation:expr, $msg:expr, $context:expr) => {
        UnifiedError::Configuration {
            operation: $operation.to_string(),
            source: ConfigErrorType::InvalidData($msg.to_string()),
            context: Some($context.to_string()),
        }
    };
}

/// Create a model error
#[macro_export]
macro_rules! model_error {
    ($model_type:expr, $operation:expr, $msg:expr) => {
        UnifiedError::Model {
            model_type: $model_type,
            operation: $operation.to_string(),
            source: $msg.to_string(),
            context: None,
        }
    };
    ($model_type:expr, $operation:expr, $msg:expr, $context:expr) => {
        UnifiedError::Model {
            model_type: $model_type,
            operation: $operation.to_string(),
            source: $msg.to_string(),
            context: Some($context.to_string()),
        }
    };
}

/// Create a processing error
#[macro_export]
macro_rules! processing_error {
    ($operation:expr, $msg:expr) => {
        UnifiedError::Processing {
            operation: $operation.to_string(),
            source: $msg.to_string(),
            input_context: None,
        }
    };
    ($operation:expr, $msg:expr, $input:expr) => {
        UnifiedError::Processing {
            operation: $operation.to_string(),
            source: $msg.to_string(),
            input_context: Some($input.to_string()),
        }
    };
}

/// Create an FFI error
#[macro_export]
macro_rules! ffi_error {
    ($function:expr, $msg:expr) => {
        UnifiedError::FFI {
            function: $function.to_string(),
            reason: $msg.to_string(),
            safety_info: None,
        }
    };
    ($function:expr, $msg:expr, $safety:expr) => {
        UnifiedError::FFI {
            function: $function.to_string(),
            reason: $msg.to_string(),
            safety_info: Some($safety.to_string()),
        }
    };
}

/// Create a validation error
#[macro_export]
macro_rules! validation_error {
    ($field:expr, $expected:expr, $actual:expr) => {
        UnifiedError::Validation {
            field: $field.to_string(),
            expected: $expected.to_string(),
            actual: $actual.to_string(),
            context: None,
        }
    };
    ($field:expr, $expected:expr, $actual:expr, $context:expr) => {
        UnifiedError::Validation {
            field: $field.to_string(),
            expected: $expected.to_string(),
            actual: $actual.to_string(),
            context: Some($context.to_string()),
        }
    };
}

/// Utility functions for common error conversions

/// Convert candle_core::Error to UnifiedError with context
pub fn from_candle_error(
    err: candle_core::Error,
    operation: &str,
    _context: Option<&str>,
) -> UnifiedError {
    UnifiedError::External {
        library: "candle-core".to_string(),
        operation: operation.to_string(),
        error: err.to_string(),
    }
}

/// Convert any error to processing error
pub fn to_processing_error<E: fmt::Display>(err: E, operation: &str) -> UnifiedError {
    UnifiedError::Processing {
        operation: operation.to_string(),
        source: err.to_string(),
        input_context: None,
    }
}

/// Convert any error to model error
pub fn to_model_error<E: fmt::Display>(
    err: E,
    model_type: ModelErrorType,
    operation: &str,
) -> UnifiedError {
    UnifiedError::Model {
        model_type,
        operation: operation.to_string(),
        source: err.to_string(),
        context: None,
    }
}

/// Create a concurrency error
pub fn concurrency_error(operation: &str, reason: &str) -> UnifiedError {
    UnifiedError::Concurrency {
        operation: operation.to_string(),
        reason: reason.to_string(),
    }
}

/// Predefined error builders for common scenarios

/// Configuration file loading errors
pub mod config_errors {
    use super::*;

    pub fn file_not_found(path: &str) -> UnifiedError {
        UnifiedError::Configuration {
            operation: "config file loading".to_string(),
            source: ConfigErrorType::FileNotFound(path.to_string()),
            context: None,
        }
    }

    pub fn missing_field(field: &str, file: &str) -> UnifiedError {
        UnifiedError::Configuration {
            operation: "config validation".to_string(),
            source: ConfigErrorType::MissingField(field.to_string()),
            context: Some(format!("in file: {}", file)),
        }
    }

    pub fn invalid_json(file: &str, error: &str) -> UnifiedError {
        UnifiedError::Configuration {
            operation: "JSON parsing".to_string(),
            source: ConfigErrorType::ParseError(error.to_string()),
            context: Some(format!("file: {}", file)),
        }
    }
}

/// Model operation errors
pub mod model_errors {
    use super::*;

    pub fn load_failure(model_type: ModelErrorType, path: &str, error: &str) -> UnifiedError {
        UnifiedError::Model {
            model_type,
            operation: "model loading".to_string(),
            source: error.to_string(),
            context: Some(format!("path: {}", path)),
        }
    }

    pub fn inference_failure(
        model_type: ModelErrorType,
        input_info: &str,
        error: &str,
    ) -> UnifiedError {
        UnifiedError::Model {
            model_type,
            operation: "model inference".to_string(),
            source: error.to_string(),
            context: Some(format!("input: {}", input_info)),
        }
    }

    pub fn tokenizer_failure(error: &str) -> UnifiedError {
        UnifiedError::Model {
            model_type: ModelErrorType::Tokenizer,
            operation: "tokenization".to_string(),
            source: error.to_string(),
            context: None,
        }
    }
}

/// Processing operation errors
pub mod processing_errors {
    use super::*;

    pub fn tensor_operation(operation: &str, error: &str) -> UnifiedError {
        UnifiedError::Processing {
            operation: format!("tensor {}", operation),
            source: error.to_string(),
            input_context: None,
        }
    }

    pub fn batch_processing(batch_size: usize, error: &str) -> UnifiedError {
        UnifiedError::Processing {
            operation: "batch processing".to_string(),
            source: error.to_string(),
            input_context: Some(format!("batch_size: {}", batch_size)),
        }
    }

    pub fn empty_input(operation: &str) -> UnifiedError {
        UnifiedError::Processing {
            operation: operation.to_string(),
            source: "empty input provided".to_string(),
            input_context: None,
        }
    }
}
