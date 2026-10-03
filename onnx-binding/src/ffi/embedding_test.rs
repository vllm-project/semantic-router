use super::*;
use ndarray::array;

#[test]
fn ranking_preserves_top_k_zero_norms_and_stable_ties() {
    let embeddings = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0], [0.0, 0.0]];
    for top_k in [-1, 0, 5] {
        let matches = rank_similarities(&embeddings, top_k);
        let values: Vec<_> = matches.iter().map(|m| (m.index, m.similarity)).collect();
        assert_eq!(values, vec![(1, 1.0), (0, 0.0), (3, 0.0), (2, -1.0)]);
    }
    let matches = rank_similarities(&embeddings, 2);
    assert_eq!(matches.len(), 2);
    assert_eq!((matches[0].index, matches[1].index), (1, 0));
}

#[test]
fn candidate_decoding_preserves_order_and_rejects_invalid_entries() {
    let first = CString::new("first").unwrap();
    let second = CString::new("second").unwrap();
    let mut pointers = [first.as_ptr(), second.as_ptr()];
    assert_eq!(
        unsafe { parse_candidates(pointers.as_ptr(), 2) }.unwrap(),
        ["first", "second"]
    );
    pointers[1] = std::ptr::null();
    assert!(unsafe { parse_candidates(pointers.as_ptr(), 2) }.is_none());
    let invalid = CString::new(vec![0xff]).unwrap();
    pointers[1] = invalid.as_ptr();
    assert!(unsafe { parse_candidates(pointers.as_ptr(), 2) }.is_none());
}
