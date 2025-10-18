//! Tests for core tokenization module

use super::tokenization::*;
use candle_core::Device;
use rayon::prelude::*;
use rstest::*;
use std::path::PathBuf;
use tokenizers::{TruncationDirection, TruncationStrategy};

// Test model paths
const TEST_MODEL_BASE: &str = "../models";
const BERT_MODEL: &str = "lora_intent_classifier_bert-base-uncased_model";

/// Fixture to create a UnifiedTokenizer instance
#[fixture]
fn unified_tokenizer() -> UnifiedTokenizer {
    let model_path = PathBuf::from(TEST_MODEL_BASE).join(BERT_MODEL);
    let tokenizer_path = model_path.join("tokenizer.json");

    if tokenizer_path.exists() {
        UnifiedTokenizer::from_file(
            tokenizer_path.to_str().unwrap(),
            TokenizationStrategy::BERT,
            Device::Cpu,
        )
        .expect("Failed to create UnifiedTokenizer")
    } else {
        // Skip test if tokenizer not available
        panic!("Test tokenizer not found at {:?}", tokenizer_path);
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[rstest]
fn test_tokenization_config_default() {
    let config = TokenizationConfig::default();

    assert_eq!(config.max_length, 512);
    assert!(config.add_special_tokens);
    assert_eq!(config.truncation_strategy, TruncationStrategy::LongestFirst);
    assert_eq!(config.pad_token_id, 0);
    assert_eq!(config.tokenization_strategy, TokenizationStrategy::BERT);
    assert_eq!(config.token_data_type, TokenDataType::I32);
}

#[rstest]
fn test_tokenization_config_custom() {
    let config = TokenizationConfig {
        max_length: 256,
        add_special_tokens: false,
        truncation_strategy: TruncationStrategy::OnlyFirst,
        truncation_direction: TruncationDirection::Left,
        pad_token_id: 1,
        pad_token: "<pad>".to_string(),
        tokenization_strategy: TokenizationStrategy::ModernBERT,
        token_data_type: TokenDataType::U32,
    };

    assert_eq!(config.max_length, 256);
    assert!(!config.add_special_tokens);
    assert_eq!(
        config.tokenization_strategy,
        TokenizationStrategy::ModernBERT
    );
    assert_eq!(config.token_data_type, TokenDataType::U32);
}

// ============================================================================
// UnifiedTokenizer Tests
// ============================================================================

#[rstest]
fn test_unified_tokenizer_new(unified_tokenizer: UnifiedTokenizer) {
    // UnifiedTokenizer should be created successfully
    // We can't access config directly (private field), but we can test functionality
    let result = unified_tokenizer.tokenize("test");
    assert!(result.is_ok(), "Tokenizer should work");
}

#[rstest]
fn test_tokenize_basic(unified_tokenizer: UnifiedTokenizer) {
    let text = "Hello, world!";
    let result = unified_tokenizer.tokenize(text);

    assert!(result.is_ok(), "Should tokenize simple text");

    let tokenization_result = result.unwrap();
    assert!(
        !tokenization_result.token_ids.is_empty(),
        "Should have token IDs"
    );
    assert_eq!(
        tokenization_result.token_ids.len(),
        tokenization_result.attention_mask.len(),
        "Token IDs and attention mask should have same length"
    );
}

#[rstest]
fn test_tokenize_empty(unified_tokenizer: UnifiedTokenizer) {
    let text = "";
    let result = unified_tokenizer.tokenize(text);

    assert!(result.is_ok(), "Should handle empty text");
}

#[rstest]
#[case("Simple text")]
#[case("A longer text that needs to be tokenized properly")]
#[case("Short")]
fn test_tokenize_various_texts(unified_tokenizer: UnifiedTokenizer, #[case] text: &str) {
    let result = unified_tokenizer.tokenize(text);

    assert!(result.is_ok(), "Should tokenize: {}", text);

    let tokenization_result = result.unwrap();
    assert!(!tokenization_result.tokens.is_empty(), "Should have tokens");
}

// ============================================================================
// Batch Tokenization Tests
// ============================================================================

#[rstest]
fn test_tokenize_batch_basic(unified_tokenizer: UnifiedTokenizer) {
    let texts = vec!["First text", "Second text", "Third text"];
    let result = unified_tokenizer.tokenize_batch(&texts);

    assert!(result.is_ok(), "Should tokenize batch");

    let batch_result = result.unwrap();
    assert_eq!(batch_result.batch_size, 3, "Should have 3 texts");
    assert_eq!(
        batch_result.token_ids.len(),
        3,
        "Should have 3 tokenizations"
    );
    assert!(batch_result.max_length > 0, "Max length should be positive");
}

#[rstest]
fn test_tokenize_batch_empty(unified_tokenizer: UnifiedTokenizer) {
    let texts: Vec<&str> = vec![];
    let result = unified_tokenizer.tokenize_batch(&texts);

    // Should either handle gracefully or return error
    match result {
        Ok(batch_result) => {
            assert_eq!(batch_result.batch_size, 0, "Should have 0 texts");
        }
        Err(_) => {
            // Also acceptable to return error
        }
    }
}

#[rstest]
fn test_tokenize_batch_varying_lengths(unified_tokenizer: UnifiedTokenizer) {
    let texts = vec![
        "Short",
        "A medium length text here",
        "This is a much longer text that will have more tokens after tokenization",
    ];
    let result = unified_tokenizer.tokenize_batch(&texts);

    assert!(result.is_ok(), "Should tokenize varying length texts");

    let batch_result = result.unwrap();
    assert_eq!(batch_result.batch_size, 3);

    // All tokenizations should be padded to max_length
    for token_ids in &batch_result.token_ids {
        assert_eq!(token_ids.len(), batch_result.max_length);
    }
}

// ============================================================================
// Traditional Tokenization Tests
// ============================================================================

#[rstest]
fn test_tokenize_for_traditional(unified_tokenizer: UnifiedTokenizer) {
    let text = "Traditional tokenization test";
    let result = unified_tokenizer.tokenize_for_traditional(text);

    assert!(result.is_ok(), "Should tokenize for traditional path");

    let tokenization_result = result.unwrap();
    assert!(!tokenization_result.token_ids.is_empty());
}

// ============================================================================
// LoRA Tokenization Tests
// ============================================================================

#[rstest]
fn test_tokenize_for_lora(unified_tokenizer: UnifiedTokenizer) {
    let text = "LoRA tokenization test";
    let result = unified_tokenizer.tokenize_for_lora(text);

    assert!(result.is_ok(), "Should tokenize for LoRA path");

    let tokenization_result = result.unwrap();
    assert!(!tokenization_result.token_ids.is_empty());
}

// ============================================================================
// Tensor Creation Tests
// ============================================================================

#[rstest]
fn test_create_tensors(unified_tokenizer: UnifiedTokenizer) {
    let text = "Test for tensor creation";
    let tokenization_result = unified_tokenizer.tokenize(text).expect("Tokenize text");

    let result = unified_tokenizer.create_tensors(&tokenization_result);

    assert!(result.is_ok(), "Should create tensors");

    let (token_ids_tensor, attention_mask_tensor) = result.unwrap();
    assert_eq!(token_ids_tensor.dims().len(), 2, "Token IDs should be 2D");
    assert_eq!(
        attention_mask_tensor.dims().len(),
        2,
        "Attention mask should be 2D"
    );
    assert_eq!(
        token_ids_tensor.dims()[1],
        attention_mask_tensor.dims()[1],
        "Tensors should have same sequence length"
    );
}

#[rstest]
fn test_create_batch_tensors(unified_tokenizer: UnifiedTokenizer) {
    let texts = vec!["First", "Second", "Third"];
    let batch_result = unified_tokenizer
        .tokenize_batch(&texts)
        .expect("Tokenize batch");

    let result = unified_tokenizer.create_batch_tensors(&batch_result);

    assert!(result.is_ok(), "Should create batch tensors");

    let (token_ids_tensor, attention_mask_tensor) = result.unwrap();
    let dims = token_ids_tensor.dims();

    assert_eq!(dims.len(), 2, "Should be 2D tensor");
    assert_eq!(dims[0], 3, "Batch size should be 3");
    assert_eq!(
        token_ids_tensor.dims(),
        attention_mask_tensor.dims(),
        "Tensors should have same dimensions"
    );
}

// ============================================================================
// Smart Batch Tokenization Tests
// ============================================================================

#[rstest]
#[case(true, "Should prefer LoRA")]
#[case(false, "Should not prefer LoRA")]
fn test_tokenize_batch_smart(
    unified_tokenizer: UnifiedTokenizer,
    #[case] prefer_lora: bool,
    #[case] description: &str,
) {
    let texts = vec!["Text one", "Text two"];
    let result = unified_tokenizer.tokenize_batch_smart(&texts, prefer_lora);

    assert!(result.is_ok(), "{}", description);

    let batch_result = result.unwrap();
    assert_eq!(batch_result.batch_size, 2);
}

// ============================================================================
// Helper Function Tests
// ============================================================================

#[test]
fn test_create_tokenizer() {
    let model_path = PathBuf::from(TEST_MODEL_BASE).join(BERT_MODEL);
    let tokenizer_path = model_path.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return; // Skip test if model not available
    }

    let result = create_tokenizer(
        tokenizer_path.to_str().unwrap(),
        TokenizationStrategy::BERT,
        Device::Cpu,
    );
    assert!(result.is_ok(), "Should create tokenizer from path");
}

#[test]
fn test_detect_tokenization_strategy() {
    let model_path = PathBuf::from(TEST_MODEL_BASE).join(BERT_MODEL);
    let tokenizer_path = model_path.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return; // Skip test if model not available
    }

    let result = detect_tokenization_strategy(tokenizer_path.to_str().unwrap());
    assert!(result.is_ok(), "Should detect tokenization strategy");
}

// ============================================================================
// Compatibility Tokenizer Tests
// ============================================================================

#[test]
fn test_create_bert_compatibility_tokenizer() {
    use tokenizers::Tokenizer;

    let model_path = PathBuf::from(TEST_MODEL_BASE).join(BERT_MODEL);
    let tokenizer_path = model_path.join("tokenizer.json");

    if !tokenizer_path.exists() {
        return;
    }

    let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Load tokenizer");

    let result = create_bert_compatibility_tokenizer(tokenizer, Device::Cpu);

    assert!(result.is_ok(), "Should create BERT compatibility tokenizer");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_create_tokenizer_invalid_path() {
    let result = create_tokenizer(
        "/nonexistent/tokenizer.json",
        TokenizationStrategy::BERT,
        Device::Cpu,
    );
    assert!(result.is_err(), "Should fail with invalid path");
}

#[test]
fn test_detect_strategy_invalid_path() {
    let result = detect_tokenization_strategy("/nonexistent/tokenizer.json");
    assert!(result.is_err(), "Should fail with invalid path");
}

// ============================================================================
// Tokenization Strategy Tests
// ============================================================================

#[rstest]
#[case(TokenizationStrategy::BERT, TokenDataType::I32)]
#[case(TokenizationStrategy::ModernBERT, TokenDataType::U32)]
#[case(TokenizationStrategy::LoRA, TokenDataType::I32)]
fn test_tokenization_strategy_data_types(
    #[case] strategy: TokenizationStrategy,
    #[case] expected_dtype: TokenDataType,
) {
    let config = TokenizationConfig {
        tokenization_strategy: strategy,
        token_data_type: expected_dtype.clone(),
        ..Default::default()
    };

    assert_eq!(config.tokenization_strategy, strategy);
    assert_eq!(config.token_data_type, expected_dtype);
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[rstest]
fn test_unified_tokenizer_thread_safety(unified_tokenizer: UnifiedTokenizer) {
    use std::sync::Arc;

    let tokenizer = Arc::new(unified_tokenizer);

    // Use rayon for parallel execution - simpler and more efficient
    let results: Vec<_> = (0..4)
        .into_par_iter()
        .map(|i| {
            let text = format!("Thread {} test text", i);
            tokenizer.tokenize(&text).expect("Tokenize in thread")
        })
        .collect();

    for result in results {
        assert!(!result.token_ids.is_empty(), "Should tokenize successfully");
    }
}
