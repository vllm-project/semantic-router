//! Cross-language ABI layout receipt.
//!
//! Go redeclares every shared FFI struct by hand in its cgo preamble while
//! Rust owns the `#[repr(C)]` definitions, so a field reorder on either side
//! compiles cleanly and corrupts memory at runtime. Each function here exports
//! the Rust-side layout — `[size, align, offset(field 0), offset(field 1), …]`
//! in declaration order — so the Go test suite can assert byte-level agreement
//! across the cgo boundary without model artifacts (issues #2477 / #2396).
//!
//! Contract: callers pass an output buffer and its capacity; the function
//! returns the number of values in the receipt. Values are written only when
//! the buffer is non-null and large enough, so a `(null, 0)` probe call is a
//! safe way to learn the required capacity.

use std::mem::{align_of, offset_of, size_of};

use super::classification::{ClassificationResultFFI, PIIEntityFFI, PIIResultFFI};
use super::types::{
    BatchSimilarityResult, EmbeddingModelInfo, EmbeddingModelsInfoResult, EmbeddingResult,
    EmbeddingSimilarityResult, MultiModalEmbeddingResult, SimilarityMatch,
};

macro_rules! layout_receipt {
    ($fn_name:ident, $ty:ty, [$($field:ident),+ $(,)?]) => {
        /// # Safety
        /// `out` must be null or valid for `cap` writes of `usize`.
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(out: *mut usize, cap: usize) -> usize {
            let values = [
                size_of::<$ty>(),
                align_of::<$ty>(),
                $(offset_of!($ty, $field),)+
            ];
            if !out.is_null() && cap >= values.len() {
                for (i, v) in values.iter().enumerate() {
                    unsafe { *out.add(i) = *v };
                }
            }
            values.len()
        }
    };
}

layout_receipt!(
    abi_layout_embedding_result,
    EmbeddingResult,
    [data, length, error, model_type, sequence_length, processing_time_ms]
);

layout_receipt!(
    abi_layout_embedding_similarity_result,
    EmbeddingSimilarityResult,
    [similarity, model_type, processing_time_ms, error]
);

layout_receipt!(abi_layout_similarity_match, SimilarityMatch, [index, similarity]);

layout_receipt!(
    abi_layout_batch_similarity_result,
    BatchSimilarityResult,
    [matches, num_matches, model_type, processing_time_ms, error]
);

layout_receipt!(
    abi_layout_embedding_model_info,
    EmbeddingModelInfo,
    [
        model_name,
        is_loaded,
        max_sequence_length,
        default_dimension,
        model_path,
        supports_layer_exit,
        available_layers
    ]
);

layout_receipt!(
    abi_layout_embedding_models_info_result,
    EmbeddingModelsInfoResult,
    [models, num_models, error]
);

layout_receipt!(
    abi_layout_classification_result,
    ClassificationResultFFI,
    [
        label,
        class_id,
        confidence,
        num_classes,
        probabilities,
        processing_time_ms,
        error
    ]
);

layout_receipt!(
    abi_layout_pii_entity,
    PIIEntityFFI,
    [text, entity_type, start, end, confidence]
);

layout_receipt!(
    abi_layout_pii_result,
    PIIResultFFI,
    [entities, num_entities, processing_time_ms, error, error_message]
);

layout_receipt!(
    abi_layout_multimodal_embedding_result,
    MultiModalEmbeddingResult,
    [data, length, error, modality, processing_time_ms]
);

#[cfg(test)]
mod tests {
    use super::*;

    /// Every receipt must report its full length from a (null, 0) probe, write
    /// exactly that many values into a large-enough buffer, and describe a
    /// struct whose size is at least the end of its last field.
    fn assert_receipt(f: unsafe extern "C" fn(*mut usize, usize) -> usize, min_fields: usize) {
        let probe = unsafe { f(std::ptr::null_mut(), 0) };
        assert!(probe >= 2 + min_fields, "receipt too short: {probe}");

        let mut buf = vec![usize::MAX; probe];
        let written = unsafe { f(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(written, probe);
        assert!(buf.iter().all(|&v| v != usize::MAX || v == 0) || buf[0] > 0);

        let (size, align, offsets) = (buf[0], buf[1], &buf[2..]);
        assert!(size > 0 && align > 0);
        assert!(size % align == 0, "size {size} not a multiple of align {align}");
        for offset in offsets {
            assert!(*offset < size, "field offset {offset} outside struct of size {size}");
        }
    }

    #[test]
    fn test_all_layout_receipts_are_sane() {
        assert_receipt(abi_layout_embedding_result, 6);
        assert_receipt(abi_layout_embedding_similarity_result, 4);
        assert_receipt(abi_layout_similarity_match, 2);
        assert_receipt(abi_layout_batch_similarity_result, 5);
        assert_receipt(abi_layout_embedding_model_info, 7);
        assert_receipt(abi_layout_embedding_models_info_result, 3);
        assert_receipt(abi_layout_classification_result, 7);
        assert_receipt(abi_layout_pii_entity, 5);
        assert_receipt(abi_layout_pii_result, 5);
        assert_receipt(abi_layout_multimodal_embedding_result, 5);
    }

    /// Undersized buffers must be left untouched while still reporting the
    /// required capacity.
    #[test]
    fn test_receipt_respects_capacity() {
        let mut buf = [usize::MAX; 1];
        let needed = unsafe { abi_layout_embedding_result(buf.as_mut_ptr(), buf.len()) };
        assert!(needed > 1);
        assert_eq!(buf[0], usize::MAX, "undersized buffer must not be written");
    }
}
