# TD043: candle-binding/semantic-router.go Carries Pre-Existing Cyclomatic Complexity Debt Inside the Diff-Scoped Lint Window

## Status

Open

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

`candle-binding/semantic-router.go` cyclop lint gating (per-function cyclomatic complexity, max 12)

## Summary

`candle-binding/semantic-router.go` carries pre-existing cyclop
violations that predate any PR touching the file: on the `main` branch
baseline, three functions exceed the `max-complexity: 12` threshold
configured in `tools/linter/go/.golangci.agent.yml`. These fire whenever
any PR touches the file, including narrow additive or refactoring
changes such as the multimodal FFI plumbing in this PR.

This is the same shape as the parallel structure-rules debt in
`candle-binding/src/ffi/embedding.rs` tracked as TD-042, and mirrors the
existing convention for `src/semantic-router/pkg/extproc/processor_res_cache.go`
(cyclop-only path-based exception in `.golangci.agent.yml` with no other
linters relaxed). The companion commit in this PR adds the same shape of
exception entry for `candle-binding/semantic-router.go`.

## Evidence

- [candle-binding/semantic-router.go](../../../candle-binding/semantic-router.go) - the file carrying the pre-existing debt
- [tools/linter/go/.golangci.agent.yml](../../../tools/linter/go/.golangci.agent.yml) - the gate definition (cyclop `max-complexity: 12`) and the new path-based exclusion
- `processor_res_cache.go` (sibling exclusion) - the precedent for the cyclop-only path-based exception pattern
- First PR to surface this debt as a CI gate failure: the candle-binding image-preprocessing fix (this PR)

## Why It Matters

- Narrow additive or refactoring PRs to `candle-binding/semantic-router.go` (new FFI entry points, signature cleanups, doc improvements) get blocked by cyclop findings that predate them and have no relationship to the change under review.
- The two functions added or modified by this PR (`MultiModalEncodeImageFromBytes` after refactor, plus the helper extraction in the FFI layer) pass per-function complexity on their own; the gate failure comes entirely from three pre-existing functions in unrelated code paths.
- Reviewers and agents lose signal about whether a PR introduced new complexity or merely intersected the historical backlog.

## Desired End State

The pre-existing cyclomatic complexity debt in `candle-binding/semantic-router.go` is retired by explicit refactoring PRs (function decomposition, switch-statement simplification, or extraction of helpers) that bring each of the three functions below the `max-complexity: 12` threshold. Once all three are below threshold, the `.golangci.agent.yml` cyclop exception for this file can be removed and this debt entry retired.

## Exit Criteria

- `make agent-ci-lint CHANGED_FILES="...,candle-binding/semantic-router.go,..."` no longer requires the cyclop path exception for this file.
- The cyclop exception entry for `candle-binding/semantic-router.go` is removed from `tools/linter/go/.golangci.agent.yml`.
- The three functions below have been refactored to pass cyclop's `max-complexity: 12` threshold on their own.

## Affected Pre-Existing Findings (as of 2026-06-03, baseline `main`)

Per-function on `main` (all predate this PR; the FFI changes in this PR are in unrelated functions and do not touch any of these):

- `GetEmbedding2DMatryoshka` (line 1548): calculated cyclomatic complexity 13 (limit 12)
- `CalculateSimilarityBatch` (line 1769): calculated cyclomatic complexity 14 (limit 12)
- `ClassifyBatchWithLoRA` (line 3497): calculated cyclomatic complexity 13 (limit 12)

These findings predate the PR opening this debt entry and are unrelated to the diff under review.
