# Technical Debt

This directory contains unresolved architecture gaps only. Completed entries
are removed and remain available through Git history.

## When to Create or Update a Debt Entry

Create an entry when a concrete code or validation gap will outlive the change
that discovered it. Update it when current source changes the evidence, scope,
owner, release relevance, or exit criteria.

## What Belongs in a Debt Entry

- one unresolved gap;
- current source or gate evidence;
- why the gap matters and the desired end state;
- testable exit criteria;
- one active owner plan and release relevance.

## What Does Not Belong in a Debt Entry

- progress reports or completed work;
- branch, commit, or pull-request notes;
- daily issue state;
- operating rules that belong in governance docs;
- execution tasks that belong in a plan.

## Debt Entry Versus Other Governance Files

- Debt entry: the unresolved architecture gap.
- Execution plan: current work that owns or reduces the gap.
- Maintainer board: changing issue and pull-request state under the gitignored
  `.agent-harness/maintainer/` directory.

## Debt Entry Template

Every entry uses these sections:

- `Status`
- `Owner Plan`
- `Release Relevance`
- `Scope`
- `Summary`
- `Evidence`
- `Why It Matters`
- `Desired End State`
- `Exit Criteria`

## Open Debt By Owner Plan

### PL-0032: Architecture Debt Consolidation

- [TD006: Structural rule exceptions](td-006-structural-rule-exceptions.md)
- [TD016: Fleet-sim Ruff contract](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [TD017: Fleet-sim structure gates](td-017-fleet-sim-structure-gate-migration-gap.md)
- [TD020: Classification subsystem boundaries](td-020-classification-subsystem-boundary-collapse.md)
- [TD027: Fleet-sim optimizer boundaries](td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)
- [TD042: FFI embedding structure](td-042-ffi-embedding-structure-debt.md)
- [TD043: Go binding complexity](td-043-semantic-router-go-cyclop-debt.md)
- [TD044: Flow tool-state durability](td-044-flow-tool-state-durability-gap.md)
- [TD045: Reviewed content moderation](td-045-reviewed-content-moderation.md)
- [TD046: ONNX binding CI coverage](td-046-onnx-binding-ci-coverage-gap.md)
- [TD047: Response-cache polarity guard surface mirrors](td-047-response-cache-polarity-guard-surface-mirrors.md)

If a gap becomes release-critical, move ownership to the active release plan
and update both indexes in the same change.
