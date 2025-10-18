//! Tests for core similarity module

use super::similarity::*;
use candle_core::{Device, Tensor};
use rayon::prelude::*;
use rstest::*;
use std::path::PathBuf;

// Test model paths
const TEST_MODEL_BASE: &str = "../models";
const BERT_MODEL: &str = "lora_intent_classifier_bert-base-uncased_model";

/// Fixture to create a BertSimilarity instance
#[fixture]
fn bert_similarity() -> BertSimilarity {
    let model_path = PathBuf::from(TEST_MODEL_BASE).join(BERT_MODEL);

    if model_path.exists() {
        BertSimilarity::new(model_path.to_str().unwrap(), true)
            .expect("Failed to create BertSimilarity")
    } else {
        // Skip test if model not available
        panic!("Test model not found at {:?}", model_path);
    }
}

// ============================================================================
// Initialization Tests
// ============================================================================

#[rstest]
fn test_bert_similarity_new(bert_similarity: BertSimilarity) {
    assert!(bert_similarity.device().is_cpu(), "Should use CPU device");
}

#[rstest]
fn test_bert_similarity_tokenizer(bert_similarity: BertSimilarity) {
    let tokenizer = bert_similarity.tokenizer();
    assert!(
        tokenizer.get_vocab_size(true) > 0,
        "Tokenizer should have vocabulary"
    );
}

#[rstest]
fn test_bert_similarity_is_gpu(bert_similarity: BertSimilarity) {
    assert!(!bert_similarity.is_gpu(), "Should be using CPU");
}

// ============================================================================
// Tokenization Tests
// ============================================================================

#[rstest]
fn test_tokenize_text_basic(bert_similarity: BertSimilarity) {
    let text = "Hello, world!";
    let result = bert_similarity.tokenize_text(text, None);

    assert!(result.is_ok(), "Should tokenize simple text");

    let (token_ids, tokens) = result.unwrap();
    assert!(!token_ids.is_empty(), "Token IDs should not be empty");
    assert!(!tokens.is_empty(), "Tokens should not be empty");
}

#[rstest]
fn test_tokenize_text_empty(bert_similarity: BertSimilarity) {
    let text = "";
    let result = bert_similarity.tokenize_text(text, None);

    assert!(result.is_ok(), "Should handle empty text");
}

#[rstest]
#[case("Simple text", None)]
#[case(
    "A longer text that might need truncation when the max length is set",
    Some(20)
)]
#[case("Short", Some(512))]
fn test_tokenize_text_with_max_length(
    bert_similarity: BertSimilarity,
    #[case] text: &str,
    #[case] max_length: Option<usize>,
) {
    let result = bert_similarity.tokenize_text(text, max_length);

    assert!(
        result.is_ok(),
        "Should tokenize text with max_length {:?}",
        max_length
    );

    let (token_ids, _tokens) = result.unwrap();

    if let Some(max_len) = max_length {
        assert!(
            token_ids.len() <= max_len,
            "Token IDs length should be <= max_length"
        );
    }
}

// ============================================================================
// Embedding Generation Tests
// ============================================================================

#[rstest]
fn test_get_embedding(bert_similarity: BertSimilarity) {
    let text = "This is a test sentence for embedding.";
    let result = bert_similarity.get_embedding(text, None);

    assert!(result.is_ok(), "Should generate embedding");

    let embedding = result.unwrap();
    let dims = embedding.dims();
    // get_embedding returns [batch_size, hidden_dim] = [1, 768]
    assert_eq!(
        dims.len(),
        2,
        "Embedding should be 2D tensor (batch format)"
    );
    assert_eq!(dims[0], 1, "Batch size should be 1");
    assert!(dims[1] > 0, "Hidden dimension should be positive");
}

#[rstest]
fn test_get_embedding_consistency(bert_similarity: BertSimilarity) {
    let text = "Consistency test sentence.";

    // Generate embedding twice
    let embedding1 = bert_similarity
        .get_embedding(text, None)
        .expect("First embedding");
    let embedding2 = bert_similarity
        .get_embedding(text, None)
        .expect("Second embedding");

    // Should produce identical embeddings for same input
    assert_eq!(
        embedding1.dims(),
        embedding2.dims(),
        "Embeddings should have same dimensions"
    );

    // Convert to Vec for comparison (squeeze batch dimension)
    let vec1: Vec<f32> = embedding1
        .squeeze(0)
        .expect("Squeeze")
        .to_vec1()
        .expect("Convert to vec1");
    let vec2: Vec<f32> = embedding2
        .squeeze(0)
        .expect("Squeeze")
        .to_vec1()
        .expect("Convert to vec2");

    for (i, (v1, v2)) in vec1.iter().zip(vec2.iter()).enumerate() {
        assert!(
            (v1 - v2).abs() < 1e-6,
            "Embeddings should be identical at position {}: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

#[rstest]
fn test_get_embedding_different_texts(bert_similarity: BertSimilarity) {
    let text1 = "The cat sits on the mat.";
    let text2 = "A dog runs in the park.";

    let embedding1 = bert_similarity
        .get_embedding(text1, None)
        .expect("First embedding");
    let embedding2 = bert_similarity
        .get_embedding(text2, None)
        .expect("Second embedding");

    // Embeddings should be different for different texts (squeeze batch dimension)
    let vec1: Vec<f32> = embedding1
        .squeeze(0)
        .expect("Squeeze")
        .to_vec1()
        .expect("Convert to vec1");
    let vec2: Vec<f32> = embedding2
        .squeeze(0)
        .expect("Squeeze")
        .to_vec1()
        .expect("Convert to vec2");

    let mut differences = 0;
    for (v1, v2) in vec1.iter().zip(vec2.iter()) {
        if (v1 - v2).abs() > 1e-6 {
            differences += 1;
        }
    }

    assert!(
        differences > vec1.len() / 10,
        "Embeddings should be substantially different (found {} differences out of {})",
        differences,
        vec1.len()
    );
}

#[rstest]
fn test_get_embedding_with_max_length(bert_similarity: BertSimilarity) {
    let long_text = "This is a very long text that will be truncated. ".repeat(20);
    let result = bert_similarity.get_embedding(&long_text, Some(128));

    assert!(result.is_ok(), "Should generate embedding with max_length");
}

// ============================================================================
// Similarity Calculation Tests
// ============================================================================

#[rstest]
fn test_calculate_similarity_identical(bert_similarity: BertSimilarity) {
    let text = "Identical text";

    let similarity = bert_similarity
        .calculate_similarity(text, text, None)
        .expect("Calculate similarity");

    assert!(
        (similarity - 1.0).abs() < 0.01,
        "Identical text should have similarity ~1.0, got {}",
        similarity
    );
}

#[rstest]
fn test_calculate_similarity_similar_texts(bert_similarity: BertSimilarity) {
    let text1 = "Machine learning is fascinating.";
    let text2 = "AI and machine learning are interesting.";

    let similarity = bert_similarity
        .calculate_similarity(text1, text2, None)
        .expect("Calculate similarity");

    assert!(
        similarity > 0.3,
        "Similar texts should have reasonable similarity, got {}",
        similarity
    );
}

#[rstest]
fn test_calculate_similarity_dissimilar_texts(bert_similarity: BertSimilarity) {
    let text1 = "The weather is sunny today.";
    let text2 = "Quantum physics is complex.";

    let similarity = bert_similarity
        .calculate_similarity(text1, text2, None)
        .expect("Calculate similarity");

    assert!(
        similarity < 0.9 && similarity > -1.0,
        "Dissimilar texts should have lower similarity, got {}",
        similarity
    );
}

#[rstest]
#[case("Hello", "Hi", 0.0)] // Should be somewhat similar
#[case("Cat", "Dog", 0.0)] // Should be somewhat similar (both animals)
#[case("Apple", "Computer", -1.0)] // Can vary greatly
fn test_calculate_similarity_various_pairs(
    bert_similarity: BertSimilarity,
    #[case] text1: &str,
    #[case] text2: &str,
    #[case] min_similarity: f32,
) {
    let similarity = bert_similarity
        .calculate_similarity(text1, text2, None)
        .expect("Calculate similarity");

    assert!(
        similarity >= min_similarity && similarity <= 1.0,
        "Similarity should be between {} and 1.0, got {}",
        min_similarity,
        similarity
    );
}

// ============================================================================
// Most Similar Finding Tests
// ============================================================================

#[rstest]
fn test_find_most_similar(bert_similarity: BertSimilarity) {
    let query = "Machine learning algorithms";
    let candidates = vec![
        "AI and deep learning",
        "Cooking recipes",
        "Neural networks",
        "Weather forecast",
    ];

    let result = bert_similarity.find_most_similar(query, &candidates, None);

    assert!(result.is_ok(), "Should find most similar");

    let (most_similar_idx, similarity) = result.unwrap();

    // Should find either "AI and deep learning" (0) or "Neural networks" (2)
    assert!(
        most_similar_idx == 0 || most_similar_idx == 2,
        "Should find AI-related text, got index {}",
        most_similar_idx
    );

    assert!(
        similarity > 0.3,
        "Similarity should be reasonably high, got {}",
        similarity
    );
}

#[rstest]
fn test_find_most_similar_single_candidate(bert_similarity: BertSimilarity) {
    let query = "Test query";
    let candidates = vec!["Single candidate"];

    let result = bert_similarity.find_most_similar(query, &candidates, None);

    assert!(result.is_ok(), "Should handle single candidate");

    let (most_similar_idx, _) = result.unwrap();
    assert_eq!(most_similar_idx, 0, "Should return the only candidate");
}

#[rstest]
fn test_find_most_similar_with_max_length(bert_similarity: BertSimilarity) {
    let query = "Short query";
    let long_text = "This is a very long candidate text that will be truncated. ".repeat(10);
    let candidates_data = vec![long_text.as_str(), "Short match"];

    let result = bert_similarity.find_most_similar(query, &candidates_data, Some(64));

    assert!(result.is_ok(), "Should handle max_length parameter");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_new_with_invalid_path() {
    let result = BertSimilarity::new("/nonexistent/path", true);
    assert!(result.is_err(), "Should fail with invalid path");
}

#[rstest]
fn test_find_most_similar_empty_candidates(bert_similarity: BertSimilarity) {
    let query = "Test query";
    let candidates: Vec<&str> = vec![];

    let result = bert_similarity.find_most_similar(query, &candidates, None);

    // Depending on implementation, this might error or return None
    // Adjust assertion based on actual behavior
    assert!(
        result.is_err() || result.unwrap().1 == 0.0,
        "Should handle empty candidates"
    );
}

// ============================================================================
// L2 Normalization Tests
// ============================================================================

#[test]
fn test_normalize_l2() {
    let device = Device::Cpu;
    let data = vec![3.0_f32, 4.0_f32]; // L2 norm = 5.0
                                       // normalize_l2 expects 2D tensor (batch format: [batch_size, dim])
    let tensor = Tensor::from_slice(&data, (1, 2), &device).expect("Create tensor");

    let normalized = normalize_l2(&tensor).expect("Normalize");
    let vec: Vec<f32> = normalized
        .squeeze(0)
        .expect("Squeeze")
        .to_vec1()
        .expect("To vec");

    // After normalization: [3/5, 4/5] = [0.6, 0.8]
    assert!(
        (vec[0] - 0.6).abs() < 0.01,
        "First component should be ~0.6"
    );
    assert!(
        (vec[1] - 0.8).abs() < 0.01,
        "Second component should be ~0.8"
    );

    // Check L2 norm is 1.0
    let l2_norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((l2_norm - 1.0).abs() < 0.01, "L2 norm should be ~1.0");
}

#[test]
fn test_normalize_l2_zero_vector() {
    let device = Device::Cpu;
    let data = vec![0.0_f32, 0.0_f32];
    let tensor = Tensor::from_slice(&data, 2, &device).expect("Create tensor");

    let result = normalize_l2(&tensor);

    // Should handle zero vector gracefully (either error or return zeros)
    match result {
        Ok(normalized) => {
            let vec: Vec<f32> = normalized.to_vec1().expect("To vec");
            assert!(
                vec.iter().all(|x| x.is_nan() || *x == 0.0),
                "Should handle zero vector"
            );
        }
        Err(_) => {
            // Also acceptable to return an error
        }
    }
}

// ============================================================================
// Concurrency Tests
// ============================================================================

#[rstest]
fn test_bert_similarity_thread_safety(bert_similarity: BertSimilarity) {
    use std::sync::Arc;

    let similarity = Arc::new(bert_similarity);

    // Use rayon for parallel execution - simpler and more efficient
    let embeddings: Vec<_> = (0..4)
        .into_par_iter()
        .map(|i| {
            let text = format!("Thread {} test text", i);
            similarity
                .get_embedding(&text, None)
                .expect("Generate embedding in thread")
        })
        .collect();

    for embedding in embeddings {
        assert!(embedding.dims()[0] > 0, "Should generate valid embedding");
    }
}
