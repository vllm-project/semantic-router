# Technical Debt Entries

This directory stores only current open technical debt. Closed debt is removed
from the current tree and remains available through git history.

## When to Create or Update a Debt Entry

- Create a TD when the repo has a durable architecture or harness gap that will
  outlive the current change.
- Update a TD when current source evidence changes the scope, owner, or exit
  criteria.
- Every open TD must have exactly one owner plan.

## What Belongs in a Debt Entry

- one unresolved gap per file
- status, owner plan, and release relevance
- concrete source evidence
- why the gap matters
- desired end state
- exit criteria

## What Does Not Belong in a Debt Entry

- task progress that belongs in a plan
- daily GitHub issue or PR state
- execution rules that belong in plans or governance docs
- historical closed items

## Debt Entry Versus Other Governance Files

- TD: unresolved gap.
- Plan: active execution owner.
- Maintainer board: generated daily issue/PR operating state.

Rule of thumb:

- if it is a gap, use TD
- if it is current execution, use a plan
- if it changes daily, keep it under `.agent-harness/maintainer/`

## Debt Entry Template

Every debt entry should include:

- `# TDxxx: <title>`
- `## Status`
- `## Owner Plan`
- `## Release Relevance`
- `## Scope`
- `## Summary`
- `## Evidence`
- `## Why It Matters`
- `## Desired End State`
- `## Exit Criteria`

## Open Debt By Owner Plan

### PL0032 Architecture Debt Consolidation

- [TD006 Structural Rule Exceptions Still Cover Active Code](td-006-structural-rule-exceptions.md)
- [TD016 Fleet Sim Subtree Still Diverges from the Shared Ruff Contract](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [TD017 Fleet Sim Migration Still Depends on Relaxed Structure Gates](td-017-fleet-sim-structure-gate-migration-gap.md)
- [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](td-020-classification-subsystem-boundary-collapse.md)
- [TD027 Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy](td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
- [TD042 FFI Embedding File Carries Pre-Existing Structure-Rules Debt](td-042-ffi-embedding-structure-debt.md)
- [TD043 candle-binding/semantic-router.go Carries Pre-Existing Cyclomatic Complexity Debt](td-043-semantic-router-go-cyclop-debt.md)
- [TD044 Router-Owned Model Lifecycle Is Split Across Download, Runtime, and API Surfaces](td-044-router-model-lifecycle-split.md)

## Retired Debt Policy

Retired TD files are removed from this directory. Keep the ID in commit history
instead of carrying a long closed-items table in the active maintainer surface.
