# TD042: FFI Embedding Structure Exception

## Status

Open

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Non-release debt.

## Scope

`candle-binding/src/ffi/embedding.rs` and its exception in
`tools/agent/structure-rules.yaml`.

## Summary

The Rust embedding FFI file owns too many model families and entry points to
meet the shared file and function-size limits. The structure gate therefore
ignores the whole file, which also prevents it from detecting new growth.

## Evidence

- [`embedding.rs`](../../../../candle-binding/src/ffi/embedding.rs) contains
  initialization, text and multimodal encoding, batching, similarity, and
  lifecycle entry points in one module.
- [`structure-rules.yaml`](../../../../tools/agent/structure-rules.yaml) lists
  the file under `ignore_globs`.

## Why It Matters

A file-wide exception hides whether a narrow FFI change adds new structural
debt. It also makes ownership and isolated testing harder because unrelated
model paths share one module.

## Desired End State

Split the entry points into feature-owned modules behind a small FFI surface,
then apply the shared structure rules directly. Keep exported ABI behavior
stable while extracting implementation details.

## Exit Criteria

- The file is removed from `ignore_globs`.
- Every resulting module passes the shared file, function, and nesting limits.
- Existing FFI tests cover the exported behavior after extraction.
