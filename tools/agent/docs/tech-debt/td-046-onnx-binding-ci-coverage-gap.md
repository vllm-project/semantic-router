# TD046: ONNX Binding Changes Lack Mandatory Runtime Coverage

## Status

Open.

## Owner Plan

[PL-0039 Domain CI Architecture](../plans/pl-0039-domain-ci-architecture.md)

## Release Relevance

Not release-blocking for the CI architecture transition, but relevant to any
change under `onnx-binding/`.

## Scope

Mandatory PR validation for the ONNX native binding.

## Summary

Native binding changes now route through the existing core `make test` receipt
instead of a second workflow that repeated `make test-binding-minimal`.
That receipt builds Candle, ML, and NLP libraries and runs Candle-backed Go
tests, but it does not build or test `onnx-binding/`.

## Evidence

- `tools/make/build-run-test.mk` makes the CI `test` target depend on
  `test-binding-minimal`.
- `tools/make/rust.mk` implements `test-binding-minimal` by running tests only
  under `candle-binding/`.
- The same file's `rust-ci` target builds Candle, ML, and NLP libraries but has
  no ONNX build or test command.
- The former `.github/workflows/bindings-test.yml` invoked only
  `make test-binding-minimal`, so its `Bindings` label overstated coverage and
  did not close the ONNX gap.

## Why It Matters

A green core receipt for an ONNX-only change currently proves repository and
Candle compatibility, not that the changed ONNX binding compiles or behaves
correctly.

## Desired End State

The core/native validation path has a deterministic, CPU-compatible ONNX build
and test receipt that runs for `onnx-binding/**` without restoring a duplicate
Candle-only workflow.

## Exit Criteria

- A documented Make target builds and tests the ONNX binding on supported CI
  runners.
- `onnx-binding/**` classifier fixtures select that target.
- The target is mandatory for affected PRs and represented accurately in CI
  naming and contributor documentation.
