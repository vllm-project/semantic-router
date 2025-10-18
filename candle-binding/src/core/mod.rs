//! # Core Business Logic Layer

// Core modules
pub mod config_loader;
pub mod similarity;
pub mod tokenization;
pub mod unified_error;

// Re-export main similarity functionality for backward compatibility
pub use similarity::{normalize_l2, BertSimilarity};

// Re-export unified configuration loader
pub use config_loader::{
    load_id2label_from_config, load_intent_labels, load_labels_from_model_config,
    load_modernbert_num_classes, load_pii_labels, load_security_labels, load_token_config,
    LoRAConfigData, ModelConfig, UnifiedConfigLoader,
};

pub use unified_error::{
    concurrency_error, config_errors, from_candle_error, model_errors, processing_errors,
    to_model_error, to_processing_error, ConfigErrorType, ErrorUnification, ModelErrorType,
    UnifiedError, UnifiedResult,
};

pub use tokenization::{
    create_bert_compatibility_tokenizer, create_c_tokenization_error,
    create_lora_compatibility_tokenizer, create_modernbert_compatibility_tokenizer,
    create_tokenizer, detect_tokenization_strategy, tokenization_result_to_c, tokenize_text_compat,
    BatchTokenizationResult, CTokenizationResult, DualPathTokenizer, TokenDataType,
    TokenizationConfig, TokenizationResult, TokenizationStrategy, UnifiedTokenizer,
};

// Test modules (only compiled in test builds)
#[cfg(test)]
pub mod config_loader_test;
#[cfg(test)]
pub mod similarity_test;
#[cfg(test)]
pub mod tokenization_test;
#[cfg(test)]
pub mod unified_error_test;
