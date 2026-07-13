# TD047: ONNX FFI Exports Lack an Enforced Unsafe-Boundary Contract

## Status

Open

## Owner Plan

PL0038 Router Hardening Audit

## Release Relevance

High - native FFI safety and CI signal

## Scope

`onnx-binding/src/ffi/{classification,embedding,multimodal,unified}.rs` and the
ONNX binding Clippy gate.

## Summary

The ONNX binding exports C ABI functions that dereference caller-owned raw
pointers while remaining safe Rust functions. Clippy therefore cannot prove a
clear unsafe-call boundary, and the binding has accumulated related raw-slice
reconstruction and module-layout findings. The normal Rust and Go test suites
pass, but strict Clippy cannot currently act as a release gate.

## Evidence

- On 2026-07-13, `cargo clippy --lib --tests -- -D warnings` reported 68 errors
  for the library target and 69 for the test target. The dominant finding was
  `clippy::not_unsafe_ptr_arg_deref` across classification, embedding,
  multimodal, and unified exports; additional findings covered
  `cast_slice_from_raw_parts`, `ptr_offset_with_cast`, and
  `items_after_test_module`.
- `cargo test --lib` passed all 68 tests in the same checkout, so this is a
  static safety-contract/gate gap rather than a failing behavioral suite.
- [onnx-binding/src/ffi](../../../onnx-binding/src/ffi) contains the affected
  ABI boundary.

## Why It Matters

Raw-pointer validity is a caller obligation. Expressing the exports as safe
functions hides that obligation from Rust callers and makes narrow native
runtime changes inherit a noisy, non-actionable lint baseline. Suppressing the
lint globally would also hide new pointer-lifetime defects.

## Desired End State

Every pointer-dereferencing export has one explicit unsafe ABI boundary with a
documented safety contract, pointer parsing and owned-slice reclamation are
centralized in small audited helpers, tests call those exports through explicit
unsafe blocks or safe adapters, and test modules do not interrupt production
items.

## Exit Criteria

- `cargo clippy --lib --tests -- -D warnings` passes in `onnx-binding` without a
  crate-wide allow for raw-pointer dereference findings.
- Every affected export documents nullability, lengths, allocation ownership,
  and lifetime requirements.
- Negative tests cover null pointers, invalid lengths, invalid UTF-8, repeated
  free, and shape/cardinality overflow at each shared helper seam.
- `cargo test --lib` and the Go binding tests continue to pass under the strict
  gate.
