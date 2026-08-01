use candle_binding::ffi::types::{
    ClassificationResult, EmbeddingResult, TokenizationResult,
    get_struct_size, get_struct_align
};
use std::mem;

#[test]
fn test_classification_result_layout() {
    assert_eq!(mem::size_of::<ClassificationResult>(), get_struct_size::<ClassificationResult>());
    assert_eq!(mem::align_of::<ClassificationResult>(), get_struct_align::<ClassificationResult>());
    assert_eq!(mem::size_of::<ClassificationResult>(), 16);
}

#[test]
fn test_embedding_result_layout() {
    assert_eq!(mem::size_of::<EmbeddingResult>(), get_struct_size::<EmbeddingResult>());
    assert_eq!(mem::align_of::<EmbeddingResult>(), get_struct_align::<EmbeddingResult>());
}

#[test]
fn test_tokenization_result_layout() {
    assert_eq!(mem::size_of::<TokenizationResult>(), get_struct_size::<TokenizationResult>());
    assert_eq!(mem::align_of::<TokenizationResult>(), get_struct_align::<TokenizationResult>());
}
