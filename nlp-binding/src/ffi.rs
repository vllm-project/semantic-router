//! C FFI interface for BM25 and N-gram classifiers.
//!
//! Follows the same convention as candle-binding:
//! - `#[no_mangle] pub extern "C" fn` for all exports
//! - `#[repr(C)]` for all shared structs
//! - CStr/CString for string passing
//! - Explicit free functions for all Rust-allocated memory

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::Mutex;

use crate::bm25_classifier::Bm25Classifier;
use crate::ngram_classifier::NgramClassifier;

// ---------------------------------------------------------------------------
// Global state (thread-safe via Mutex)
// ---------------------------------------------------------------------------
use std::sync::OnceLock;

static BM25_CLASSIFIERS: OnceLock<Mutex<std::collections::HashMap<u64, Bm25Classifier>>> =
    OnceLock::new();
static NGRAM_CLASSIFIERS: OnceLock<Mutex<std::collections::HashMap<u64, NgramClassifier>>> =
    OnceLock::new();

fn bm25_map() -> &'static Mutex<std::collections::HashMap<u64, Bm25Classifier>> {
    BM25_CLASSIFIERS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn ngram_map() -> &'static Mutex<std::collections::HashMap<u64, NgramClassifier>> {
    NGRAM_CLASSIFIERS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// C-compatible result structs
// ---------------------------------------------------------------------------

/// Result of a classification operation.
#[repr(C)]
pub struct ClassifyResult {
    /// Whether a match was found.
    pub matched: bool,
    /// Rule name (caller must free with `free_classify_result`).
    pub rule_name: *mut c_char,
    /// Array of matched keyword strings (caller must free).
    pub matched_keywords: *mut *mut c_char,
    /// Array of scores/similarities per matched keyword.
    pub scores: *mut f32,
    /// Number of matched keywords.
    pub match_count: i32,
    /// Total keywords in the matched rule.
    pub total_keywords: i32,
}

/// Ordered results from an all-match classification operation.
#[repr(C)]
pub struct ClassifyResults {
    /// Contiguous array of results owned by Rust.
    pub results: *mut ClassifyResult,
    /// Number of elements in `results`.
    pub count: i32,
}

impl ClassifyResults {
    fn from_vec(mut results: Vec<ClassifyResult>) -> Self {
        if results.is_empty() {
            return Self {
                results: ptr::null_mut(),
                count: 0,
            };
        }

        let count = results.len();
        let ptr = results.as_mut_ptr();
        std::mem::forget(results);
        Self {
            results: ptr,
            count: count as i32,
        }
    }
}

impl ClassifyResult {
    fn empty() -> Self {
        ClassifyResult {
            matched: false,
            rule_name: ptr::null_mut(),
            matched_keywords: ptr::null_mut(),
            scores: ptr::null_mut(),
            match_count: 0,
            total_keywords: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: Rust string -> C string (caller frees)
// ---------------------------------------------------------------------------
fn to_c_string(s: &str) -> *mut c_char {
    CString::new(s)
        .map(|cs| cs.into_raw())
        .unwrap_or(ptr::null_mut())
}

fn to_classify_result(
    rule_name: &str,
    matched_keywords: &[String],
    scores: &[f32],
    match_count: usize,
    total_keywords: usize,
) -> ClassifyResult {
    let count = matched_keywords.len();
    let keyword_ptrs: Vec<*mut c_char> = matched_keywords
        .iter()
        .map(|keyword| to_c_string(keyword))
        .collect();

    let keyword_array = if count == 0 {
        ptr::null_mut()
    } else {
        let ptr =
            unsafe { libc::malloc(count * std::mem::size_of::<*mut c_char>()) as *mut *mut c_char };
        if !ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(keyword_ptrs.as_ptr(), ptr, count) };
        }
        ptr
    };

    let scores_ptr = if count == 0 {
        ptr::null_mut()
    } else {
        let ptr = unsafe { libc::malloc(count * std::mem::size_of::<f32>()) as *mut f32 };
        if !ptr.is_null() {
            unsafe { std::ptr::copy_nonoverlapping(scores.as_ptr(), ptr, count) };
        }
        ptr
    };

    ClassifyResult {
        matched: true,
        rule_name: to_c_string(rule_name),
        matched_keywords: keyword_array,
        scores: scores_ptr,
        match_count: match_count as i32,
        total_keywords: total_keywords as i32,
    }
}

/// Convert a C string pointer to a Rust &str. Returns None on null or invalid UTF-8.
unsafe fn from_c_str<'a>(ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    CStr::from_ptr(ptr).to_str().ok()
}

// ===========================================================================
// BM25 Classifier FFI
// ===========================================================================

/// Create a new BM25 classifier instance. Returns a handle (>0) or 0 on error.
#[no_mangle]
pub extern "C" fn bm25_classifier_new() -> u64 {
    let id = next_id();
    let classifier = Bm25Classifier::new();
    bm25_map().lock().unwrap().insert(id, classifier);
    id
}

/// Add a rule to a BM25 classifier.
///
/// # Arguments
/// * `handle` - Classifier handle from `bm25_classifier_new`.
/// * `name` - Rule name (C string).
/// * `operator` - "AND", "OR", or "NOR" (C string).
/// * `keywords` - Array of keyword C strings.
/// * `num_keywords` - Length of the keywords array.
/// * `threshold` - BM25 score threshold for a keyword to count as matched.
/// * `case_sensitive` - Whether matching is case-sensitive.
///
/// # Safety
/// Non-null string pointers must reference valid NUL-terminated strings, and
/// `keywords` must reference `num_keywords` readable pointers.
#[no_mangle]
pub unsafe extern "C" fn bm25_classifier_add_rule(
    handle: u64,
    name: *const c_char,
    operator: *const c_char,
    keywords: *const *const c_char,
    num_keywords: i32,
    threshold: f32,
    case_sensitive: bool,
) -> bool {
    let name = match unsafe { from_c_str(name) } {
        Some(s) => s.to_string(),
        None => return false,
    };
    let operator = match unsafe { from_c_str(operator) } {
        Some(s) => s.to_string(),
        None => return false,
    };

    if keywords.is_null() || num_keywords <= 0 {
        return false;
    }

    let kw_slice = unsafe { std::slice::from_raw_parts(keywords, num_keywords as usize) };
    let kw_vec: Vec<String> = kw_slice
        .iter()
        .filter_map(|&ptr| unsafe { from_c_str(ptr) }.map(|s| s.to_string()))
        .collect();

    if kw_vec.len() != num_keywords as usize {
        return false;
    }

    let mut map = bm25_map().lock().unwrap();
    if let Some(classifier) = map.get_mut(&handle) {
        classifier.add_rule(name, operator, kw_vec, threshold, case_sensitive);
        true
    } else {
        false
    }
}

/// Classify text using a BM25 classifier.
///
/// # Safety
/// A non-null `text` pointer must reference a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn bm25_classifier_classify(
    handle: u64,
    text: *const c_char,
) -> ClassifyResult {
    let text = match unsafe { from_c_str(text) } {
        Some(s) => s,
        None => return ClassifyResult::empty(),
    };

    let map = bm25_map().lock().unwrap();
    let classifier = match map.get(&handle) {
        Some(c) => c,
        None => return ClassifyResult::empty(),
    };

    match classifier.classify(text) {
        Some(result) => to_classify_result(
            &result.rule_name,
            &result.matched_keywords,
            &result.scores,
            result.match_count,
            result.total_keywords,
        ),
        None => ClassifyResult::empty(),
    }
}

/// Classify text and return every matching BM25 rule in declaration order.
///
/// # Safety
/// A non-null `text` pointer must reference a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn bm25_classifier_classify_all(
    handle: u64,
    text: *const c_char,
) -> ClassifyResults {
    let text = match unsafe { from_c_str(text) } {
        Some(s) => s,
        None => return ClassifyResults::from_vec(Vec::new()),
    };

    let map = bm25_map().lock().unwrap();
    let classifier = match map.get(&handle) {
        Some(c) => c,
        None => return ClassifyResults::from_vec(Vec::new()),
    };
    ClassifyResults::from_vec(
        classifier
            .classify_all(text)
            .into_iter()
            .map(|result| {
                to_classify_result(
                    &result.rule_name,
                    &result.matched_keywords,
                    &result.scores,
                    result.match_count,
                    result.total_keywords,
                )
            })
            .collect(),
    )
}

/// Destroy a BM25 classifier and free its resources.
#[no_mangle]
pub extern "C" fn bm25_classifier_free(handle: u64) {
    bm25_map().lock().unwrap().remove(&handle);
}

// ===========================================================================
// N-gram Classifier FFI
// ===========================================================================

/// Create a new N-gram classifier instance. Returns a handle (>0) or 0 on error.
#[no_mangle]
pub extern "C" fn ngram_classifier_new() -> u64 {
    let id = next_id();
    let classifier = NgramClassifier::new();
    ngram_map().lock().unwrap().insert(id, classifier);
    id
}

/// Add a rule to an N-gram classifier.
///
/// # Arguments
/// * `handle` - Classifier handle from `ngram_classifier_new`.
/// * `name` - Rule name (C string).
/// * `operator` - "AND", "OR", or "NOR" (C string).
/// * `keywords` - Array of keyword C strings.
/// * `num_keywords` - Length of the keywords array.
/// * `threshold` - Similarity threshold (0.0-1.0) for n-gram matching.
/// * `case_sensitive` - Whether matching is case-sensitive.
/// * `arity` - N-gram arity (2 for bigrams, 3 for trigrams, etc.).
///
/// # Safety
/// Non-null string pointers must reference valid NUL-terminated strings, and
/// `keywords` must reference `num_keywords` readable pointers.
#[no_mangle]
pub unsafe extern "C" fn ngram_classifier_add_rule(
    handle: u64,
    name: *const c_char,
    operator: *const c_char,
    keywords: *const *const c_char,
    num_keywords: i32,
    threshold: f32,
    case_sensitive: bool,
    arity: i32,
) -> bool {
    let name = match unsafe { from_c_str(name) } {
        Some(s) => s.to_string(),
        None => return false,
    };
    let operator = match unsafe { from_c_str(operator) } {
        Some(s) => s.to_string(),
        None => return false,
    };

    if keywords.is_null() || num_keywords <= 0 {
        return false;
    }

    let kw_slice = unsafe { std::slice::from_raw_parts(keywords, num_keywords as usize) };
    let kw_vec: Vec<String> = kw_slice
        .iter()
        .filter_map(|&ptr| unsafe { from_c_str(ptr) }.map(|s| s.to_string()))
        .collect();

    if kw_vec.len() != num_keywords as usize {
        return false;
    }

    let mut map = ngram_map().lock().unwrap();
    if let Some(classifier) = map.get_mut(&handle) {
        classifier.add_rule(
            name,
            operator,
            kw_vec,
            threshold,
            case_sensitive,
            arity as usize,
        );
        true
    } else {
        false
    }
}

/// Classify text using an N-gram classifier.
///
/// # Safety
/// A non-null `text` pointer must reference a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ngram_classifier_classify(
    handle: u64,
    text: *const c_char,
) -> ClassifyResult {
    let text = match unsafe { from_c_str(text) } {
        Some(s) => s,
        None => return ClassifyResult::empty(),
    };

    let map = ngram_map().lock().unwrap();
    let classifier = match map.get(&handle) {
        Some(c) => c,
        None => return ClassifyResult::empty(),
    };

    match classifier.classify(text) {
        Some(result) => to_classify_result(
            &result.rule_name,
            &result.matched_keywords,
            &result.similarities,
            result.match_count,
            result.total_keywords,
        ),
        None => ClassifyResult::empty(),
    }
}

/// Classify text and return every matching N-gram rule in declaration order.
///
/// # Safety
/// A non-null `text` pointer must reference a valid NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ngram_classifier_classify_all(
    handle: u64,
    text: *const c_char,
) -> ClassifyResults {
    let text = match unsafe { from_c_str(text) } {
        Some(s) => s,
        None => return ClassifyResults::from_vec(Vec::new()),
    };

    let map = ngram_map().lock().unwrap();
    let classifier = match map.get(&handle) {
        Some(c) => c,
        None => return ClassifyResults::from_vec(Vec::new()),
    };
    ClassifyResults::from_vec(
        classifier
            .classify_all(text)
            .into_iter()
            .map(|result| {
                to_classify_result(
                    &result.rule_name,
                    &result.matched_keywords,
                    &result.similarities,
                    result.match_count,
                    result.total_keywords,
                )
            })
            .collect(),
    )
}

/// Destroy an N-gram classifier and free its resources.
#[no_mangle]
pub extern "C" fn ngram_classifier_free(handle: u64) {
    ngram_map().lock().unwrap().remove(&handle);
}

// ===========================================================================
// Memory management
// ===========================================================================

/// Free a ClassifyResult returned by `bm25_classifier_classify` or `ngram_classifier_classify`.
#[no_mangle]
pub extern "C" fn free_classify_result(result: ClassifyResult) {
    free_classify_result_inner(result);
}

fn free_classify_result_inner(result: ClassifyResult) {
    unsafe {
        if !result.rule_name.is_null() {
            let _ = CString::from_raw(result.rule_name);
        }

        if !result.matched_keywords.is_null() && result.match_count > 0 {
            let count = result.match_count as usize;
            let kw_slice = std::slice::from_raw_parts(result.matched_keywords, count);
            for &ptr in kw_slice {
                if !ptr.is_null() {
                    let _ = CString::from_raw(ptr);
                }
            }
            libc::free(result.matched_keywords as *mut libc::c_void);
        }

        if !result.scores.is_null() {
            libc::free(result.scores as *mut libc::c_void);
        }
    }
}

/// Free a ClassifyResults returned by an all-match operation.
#[no_mangle]
pub extern "C" fn free_classify_results(results: ClassifyResults) {
    if results.results.is_null() || results.count <= 0 {
        return;
    }
    unsafe {
        let owned = Vec::from_raw_parts(
            results.results,
            results.count as usize,
            results.count as usize,
        );
        for result in owned {
            free_classify_result_inner(result);
        }
    }
}
