//! Input decoding and ranking for the batch similarity FFI.

use super::types::{BatchSimilarityResult, SimilarityMatch};
use ndarray::Array2;
use std::ffi::{c_char, CStr};

/// # Safety
/// `candidates` must reference `count` readable pointers. Its non-null entries
/// must be NUL-terminated strings that remain readable for `'a`.
pub(super) unsafe fn parse_candidates<'a>(
    candidates: *const *const c_char,
    count: usize,
) -> Result<Vec<&'a str>, String> {
    let mut texts = Vec::with_capacity(count);
    for i in 0..count {
        let pointer = unsafe { *candidates.add(i) };
        if pointer.is_null() {
            return Err(format!("null candidate at index {}", i));
        }
        let text = unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .map_err(|error| format!("invalid UTF-8 in candidate {}: {}", i, error))?;
        texts.push(text);
    }
    Ok(texts)
}

pub(super) fn rank_candidates(
    embeddings: &Array2<f32>,
    count: usize,
    top_k: i32,
) -> Vec<SimilarityMatch> {
    let query_embedding = embeddings.row(0);
    let mut similarities = Vec::with_capacity(count);
    for i in 0..count {
        let candidate_embedding = embeddings.row(i + 1);
        let dot_product: f32 = query_embedding
            .iter()
            .zip(candidate_embedding.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_query: f32 = query_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_candidate: f32 = candidate_embedding
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        let similarity = if norm_query > 0.0 && norm_candidate > 0.0 {
            dot_product / (norm_query * norm_candidate)
        } else {
            0.0
        };
        similarities.push((i, similarity));
    }
    similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = if top_k <= 0 || top_k as usize > count {
        count
    } else {
        top_k as usize
    };
    similarities
        .iter()
        .take(k)
        .map(|(index, similarity)| SimilarityMatch {
            index: *index as i32,
            similarity: *similarity,
        })
        .collect()
}

pub(super) fn batch_result(
    matches: Vec<SimilarityMatch>,
    processing_time_ms: f32,
) -> BatchSimilarityResult {
    BatchSimilarityResult {
        num_matches: matches.len() as i32,
        matches: Box::into_raw(matches.into_boxed_slice()) as *mut SimilarityMatch,
        model_type: 0,
        processing_time_ms,
        error: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::ffi::CString;

    #[test]
    fn ranking_preserves_top_k_zero_norms_and_stable_ties() {
        let embeddings = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]];
        for top_k in [-1, 0, 5] {
            let matches = rank_candidates(&embeddings, 4, top_k);
            let values: Vec<_> = matches.iter().map(|m| (m.index, m.similarity)).collect();
            assert_eq!(values, vec![(1, 1.0), (0, 0.0), (3, 0.0), (2, -1.0)]);
        }
        let matches = rank_candidates(&embeddings, 4, 2);
        assert_eq!(matches.len(), 2);
        assert_eq!((matches[0].index, matches[1].index), (1, 0));
    }

    #[test]
    fn candidate_decoding_preserves_order_and_reports_invalid_entries() {
        let first = CString::new("first").unwrap();
        let second = CString::new("second").unwrap();
        let mut pointers = [first.as_ptr(), second.as_ptr()];
        assert_eq!(
            unsafe { parse_candidates(pointers.as_ptr(), 2) }.unwrap(),
            ["first", "second"]
        );
        pointers[1] = std::ptr::null();
        assert_eq!(
            unsafe { parse_candidates(pointers.as_ptr(), 2) }.unwrap_err(),
            "null candidate at index 1"
        );
        let invalid = CString::new(vec![0xff]).unwrap();
        pointers[1] = invalid.as_ptr();
        assert!(unsafe { parse_candidates(pointers.as_ptr(), 2) }
            .unwrap_err()
            .starts_with("invalid UTF-8 in candidate 1:"));
    }
}
