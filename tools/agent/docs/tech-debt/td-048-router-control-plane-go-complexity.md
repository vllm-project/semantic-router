# TD048: Go Complexity Debt Ratchet

## Status

Open

## Owner Plan

[PL-0032: Architecture Debt Consolidation](../plans/pl-0032-architecture-scorecard-ratchet.md)

## Release Relevance

Non-release debt.

## Scope

Every Go declaration recorded in
[`complexity-debt.yaml`](../../../linter/go/complexity-debt.yaml), across all
tracked Go modules and every satisfiable build context.

## Summary

The repository contains declarations above the shared cyclomatic, cognitive,
nesting, function-size, or interface-size limits. The exact debt
inventory is machine-readable and frozen: a changed declaration may keep its
recorded metric temporarily, but it may not add or increase debt. Any
improvement must tighten the inventory in the same change.

The inventory is not a lint exclusion. GolangCI continues to evaluate every
enabled complexity dimension, and the ratchet reports the known, new,
worsened, improved, and stale totals independently.

The one-time bootstrap is anchored by
[`complexity-baseline.freeze.yaml`](../../../linter/go/complexity-baseline.freeze.yaml).
That marker and the initial manifest must be introduced in the same commit; it
commits to the initial manifest, config, tool, and identity-parser contract.
Later commits resolve that immutable introduction from Git history. Once the
manifest exists on the target branch, the target-branch tip is canonical.

## Evidence

- [`complexity-debt.yaml`](../../../linter/go/complexity-debt.yaml) records
  each declaration, metric, configured limit, owner, and debt identifier.
- [`go_complexity_identity.py`](../../scripts/go_complexity_identity.py)
  resolves findings to receiver-aware Go declarations and normalized AST sites.
- [`go_complexity_manifest.py`](../../scripts/go_complexity_manifest.py)
  owns deterministic inventory parsing, serialization, and fail-closed tool and
  configuration contracts.
- [`go_complexity_ratchet.py`](../../scripts/go_complexity_ratchet.py) rejects
  new or widened allowances and requires improvements to be retained.
- [`go_complexity_config.py`](../../scripts/go_complexity_config.py) rejects
  disabled linters, higher thresholds, new exclusions, reduced analysis
  coverage, and unproven tool or setting changes. Stricter settings still
  require the synchronized manifest to pass the exact finding contract.
- [`go_complexity_source_policy.py`](../../scripts/go_complexity_source_policy.py)
  rejects new complexity `nolint` directives and build constraints that hide
  changed source from the lint context.
- [`.golangci.agent.yml`](../../../linter/go/.golangci.agent.yml) keeps the
  shared limits unchanged and preserves same-line diagnostics from every
  enabled complexity linter.

## Why It Matters

Complex request, persistence, validation, and streaming flows are harder to
reason about and more likely to couple unrelated lifecycle policy. Broad path
exclusions would let that debt grow invisibly. An exact declaration-level
inventory keeps the current gap visible while making every changed file obey a
strict no-growth contract.

## Desired End State

- Request and persistence orchestrators delegate to narrow validation,
  transition, serialization, and storage helpers.
- Protocol codecs model streaming behavior through explicit state transitions
  instead of deeply nested conditional flows.
- Repository interfaces are split by consumer capability at real package seams.
- Native binding facades delegate family-specific dispatch and validation to
  narrow helpers.
- Production and test declarations pass the shared limits without debt entries
  or path-level exclusions.

## Burn-Down Order

1. Retire declarations with cognitive complexity above 35, cyclomatic
   complexity above 20, or nesting above 7.
2. Reduce the highest-concentration management, control-plane, publication,
   identity, protocol, quota, and extproc packages.
3. Extract reusable test fixtures and assertion helpers from complex tests.
4. Continue removing or lowering entries until the inventory is empty.

Each reduction must include focused tests for the extracted seam and a matching
manifest reduction. Removing one entry never creates budget for another.

## Exit Criteria

- `complexity-debt.yaml` contains no entries.
- The changed-file Go gate passes with the shared limits, complete same-line
  diagnostics, and no complexity debt manifest allowances.
- The debt ratchet and its manifest bootstrap path are removed after the final
  entry is retired; ordinary GolangCI enforcement remains.
