//! Dealloc-layout regression tests for the FFI free functions
//!
//! `Box::from_raw` must reconstruct the exact type that was leaked with
//! `Box::into_raw`. The arrays handed across the FFI boundary are allocated
//! as `Box<[T]>` via `into_boxed_slice()`, so reclaiming them through the
//! element pointer (`Box::from_raw(slice.as_mut_ptr())`) deallocates with a
//! single-element layout. That is undefined behavior whenever the slice holds
//! more than one element, but the default system allocator absorbs the size
//! mismatch silently (`free()` takes no size), so an ordinary round-trip test
//! passes against the broken form.
//!
//! These tests install a layout-tracking allocator for the test binary that
//! records the layout of every allocation made while a tracking window is
//! armed and compares it with the layout supplied at deallocation. A free
//! through the element pointer is reported as a layout mismatch and fails the
//! test; the full-length `slice_from_raw_parts_mut` form passes.

use super::embedding::{free_batch_similarity_result, free_embedding_models_info};
use super::types::{
    BatchSimilarityResult, EmbeddingModelInfo, EmbeddingModelsInfoResult, SimilarityMatch,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::CString;
use std::sync::Mutex;

/// One allocation observed while the tracking window is armed.
#[derive(Debug, Clone, Copy)]
struct AllocRecord {
    ptr: usize,
    size: usize,
    align: usize,
}

/// A tracked pointer freed with a layout differing from its allocation layout.
// The fields are read only through the Debug-formatted assertion message, which
// dead-code analysis does not count as a use; allow rather than drop them, since
// the byte counts are the whole diagnostic value when this fires.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct LayoutMismatch {
    ptr: usize,
    alloc_size: usize,
    alloc_align: usize,
    dealloc_size: usize,
    dealloc_align: usize,
}

static TRACKED: Mutex<Vec<AllocRecord>> = Mutex::new(Vec::new());
static MISMATCHED: Mutex<Vec<LayoutMismatch>> = Mutex::new(Vec::new());
/// (ptr, size) for every tracked pointer freed with its exact allocation layout.
static FREED_OK: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());
/// Serializes the tests in this module so their tracking windows don't overlap.
static WINDOW_GUARD: Mutex<()> = Mutex::new(());

thread_local! {
    /// True only on the thread that armed the current tracking window, so the
    /// hooks never race with allocations made by other test threads or the
    /// libtest harness.
    static WINDOW_OWNER: Cell<bool> = const { Cell::new(false) };
    /// Reentrancy guard: the bookkeeping Vecs may themselves allocate while
    /// the hook holds the lock; those inner allocations must not be recorded.
    static IN_HOOK: Cell<bool> = const { Cell::new(false) };
}

/// Only the window-owner thread records, and never reentrantly. `try_with`
/// keeps the hook safe during thread teardown after TLS destruction.
fn should_record() -> bool {
    WINDOW_OWNER.try_with(|w| w.get()).unwrap_or(false)
        && !IN_HOOK.try_with(|g| g.get()).unwrap_or(true)
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct LayoutTrackingAllocator;

unsafe impl GlobalAlloc for LayoutTrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() && should_record() {
            IN_HOOK.with(|g| g.set(true));
            locked(&TRACKED).push(AllocRecord {
                ptr: ptr as usize,
                size: layout.size(),
                align: layout.align(),
            });
            IN_HOOK.with(|g| g.set(false));
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if should_record() {
            IN_HOOK.with(|g| g.set(true));
            let entry = {
                let mut tracked = locked(&TRACKED);
                tracked
                    .iter()
                    .rposition(|r| r.ptr == ptr as usize)
                    .map(|pos| tracked.remove(pos))
            };
            if let Some(record) = entry {
                if record.size == layout.size() && record.align == layout.align() {
                    locked(&FREED_OK).push((record.ptr, record.size));
                } else {
                    locked(&MISMATCHED).push(LayoutMismatch {
                        ptr: record.ptr,
                        alloc_size: record.size,
                        alloc_align: record.align,
                        dealloc_size: layout.size(),
                        dealloc_align: layout.align(),
                    });
                }
            }
            IN_HOOK.with(|g| g.set(false));
        }
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static GLOBAL: LayoutTrackingAllocator = LayoutTrackingAllocator;

fn arm() {
    locked(&TRACKED).clear();
    locked(&MISMATCHED).clear();
    locked(&FREED_OK).clear();
    WINDOW_OWNER.with(|w| w.set(true));
}

struct WindowReport {
    mismatched: Vec<LayoutMismatch>,
    freed_ok: Vec<(usize, usize)>,
}

fn disarm() -> WindowReport {
    WINDOW_OWNER.with(|w| w.set(false));
    WindowReport {
        mismatched: std::mem::take(&mut *locked(&MISMATCHED)),
        freed_ok: std::mem::take(&mut *locked(&FREED_OK)),
    }
}

fn assert_full_layout_free(report: &WindowReport, ptr: usize, expected_size: usize, what: &str) {
    assert!(
        report.mismatched.is_empty(),
        "{} freed with wrong layout (UB): {:?}",
        what,
        report.mismatched
    );
    assert!(
        report
            .freed_ok
            .iter()
            .any(|&(p, size)| p == ptr && size == expected_size),
        "{} (ptr {:#x}, {} bytes) was not freed with its full allocation layout; \
         freed_ok = {:?}",
        what,
        ptr,
        expected_size,
        report.freed_ok
    );
}

/// Mirrors the producer in `find_most_similar_batch`: the matches array is
/// allocated as `Vec<SimilarityMatch>` -> `into_boxed_slice()` -> `Box::into_raw`.
#[test]
fn test_free_batch_similarity_result_uses_full_slice_layout() {
    let _guard = WINDOW_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const NUM_MATCHES: usize = 4;

    arm();
    let top_matches: Vec<SimilarityMatch> = (0..NUM_MATCHES)
        .map(|i| SimilarityMatch {
            index: i as i32,
            similarity: 1.0 - 0.1 * i as f32,
        })
        .collect();
    let matches_ptr = Box::into_raw(top_matches.into_boxed_slice()) as *mut SimilarityMatch;

    let mut result = BatchSimilarityResult {
        matches: matches_ptr,
        num_matches: NUM_MATCHES as i32,
        model_type: 0,
        processing_time_ms: 0.0,
        error: false,
    };
    free_batch_similarity_result(&mut result);
    let report = disarm();

    assert_full_layout_free(
        &report,
        matches_ptr as usize,
        NUM_MATCHES * std::mem::size_of::<SimilarityMatch>(),
        "BatchSimilarityResult.matches array",
    );
    assert!(result.matches.is_null());
    assert_eq!(result.num_matches, 0);
}

/// Mirrors the producer in `get_embedding_models_info`: the models array is
/// allocated as `Vec<EmbeddingModelInfo>` -> `into_boxed_slice()` ->
/// `Box::into_raw`, with `CString::into_raw` name/path fields.
#[test]
fn test_free_embedding_models_info_uses_full_slice_layout() {
    let _guard = WINDOW_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    const NUM_MODELS: usize = 2;

    arm();
    let models_vec: Vec<EmbeddingModelInfo> = ["qwen3", "gemma"]
        .iter()
        .map(|name| EmbeddingModelInfo {
            model_name: CString::new(*name).unwrap().into_raw(),
            is_loaded: true,
            max_sequence_length: 8192,
            default_dimension: 768,
            model_path: CString::new(format!("/models/{}", name))
                .unwrap()
                .into_raw(),
        })
        .collect();
    let models_ptr = Box::into_raw(models_vec.into_boxed_slice()) as *mut EmbeddingModelInfo;

    let mut result = EmbeddingModelsInfoResult {
        models: models_ptr,
        num_models: NUM_MODELS as i32,
        error: false,
    };
    free_embedding_models_info(&mut result);
    let report = disarm();

    assert_full_layout_free(
        &report,
        models_ptr as usize,
        NUM_MODELS * std::mem::size_of::<EmbeddingModelInfo>(),
        "EmbeddingModelsInfoResult.models array",
    );
    assert!(result.models.is_null());
    assert_eq!(result.num_models, 0);
}
