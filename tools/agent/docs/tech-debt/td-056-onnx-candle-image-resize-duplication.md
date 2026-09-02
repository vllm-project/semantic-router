# TD056: onnx-binding/candle-binding Image Resize Implementation Duplication

## Status

Open

## Owner Plan

Unassigned - newly surfaced by #2166. Topically adjacent to
[TD042](td-042-ffi-embedding-structure-debt.md)'s owner plan
(PL-0032: Architecture Debt Consolidation), given both concern
`candle-binding/src/ffi/embedding.rs`.

## Release Relevance

Not release-blocking. Correctness risk only surfaces if the two copies
silently diverge in a future change.

## Scope

`candle-binding/src/ffi/embedding.rs::decode_resize_to_chw_f32` and
`onnx-binding/src/ffi/multimodal.rs::decode_resize_to_chw_f32`.

## Summary

Issue #2166 fixed onnx-binding's image resize to match candle-binding's by
copying candle-binding's `decode_resize_to_chw_f32` verbatim into
onnx-binding, rather than sharing one implementation. The function now
exists as two independently-compiled copies, kept in sync only by a
code comment and by both Cargo.tomls independently pinning
`image = "0.25"`. Nothing in the build or test pipeline enforces that
the two copies stay identical.

## Evidence

- `candle-binding/src/ffi/embedding.rs` (`decode_resize_to_chw_f32`,
  private fn).
- `onnx-binding/src/ffi/multimodal.rs` (`decode_resize_to_chw_f32`,
  private fn; doc comment references candle-binding and #2166).
- Both Cargo.tomls independently declare `image = "0.25"` with no
  shared version-pinning mechanism between the two crates.
- No test or gate compares the two source files or their compiled
  behavior beyond the golden-image fixture added in #2166, which only
  catches a behavioral drift if someone runs it - not a source-level
  divergence between the two files.

## Why It Matters

This is structurally the same failure mode #2166 fixed: two
implementations that are supposed to agree, with nothing but a comment
enforcing it. If either crate's `image` dependency is later bumped
independently, or the function is edited in one file without the
other, the two bindings can silently diverge again on a different
code path than the original bug.

## Desired End State

One implementation, not two.

## Exit Criteria

- Either a small shared crate (no candle-core/CUDA dependency) exists
  and both bindings depend on it instead of maintaining separate
  copies, or a CI check exists that reads both source files and fails
  when the two `decode_resize_to_chw_f32` bodies diverge, as a lighter
  interim measure.
- The interim code comment cross-referencing #2166 in both files can
  be removed once either exit criterion lands, since enforcement
  replaces documentation.
