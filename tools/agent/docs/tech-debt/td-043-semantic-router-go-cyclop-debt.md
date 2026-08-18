# TD043: Go Binding Complexity Exception

## Status

Open

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Non-release debt.

## Scope

`candle-binding/semantic-router.go` and its `cyclop` exclusion in
`tools/linter/go/.golangci.agent.yml`.

## Summary

The Go binding facade contains several functions above the repository's
cyclomatic-complexity limit. A path-level lint exclusion keeps unrelated
binding changes moving, but it also applies to new code in the same file.

## Evidence

- [`semantic-router.go`](../../../../candle-binding/semantic-router.go) combines
  initialization, embeddings, similarity, classifiers, LoRA, and batch APIs.
- [`.golangci.agent.yml`](../../../../tools/linter/go/.golangci.agent.yml)
  excludes `cyclop` for the entire file.

## Why It Matters

The broad exclusion weakens regression detection and makes the public binding
facade expensive to review. Complex dispatch functions also couple otherwise
independent native features.

## Desired End State

Extract family-specific dispatch and validation helpers so every public entry
point stays below the shared complexity limit without changing its external
contract.

## Exit Criteria

- The file passes `cyclop` with the repository limit.
- Its path-level `cyclop` exclusion is removed.
- Focused Go and native-binding tests cover the extracted branches.
