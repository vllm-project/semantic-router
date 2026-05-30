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

### PL0033 v0.3 Themis Release Closure

- [TD015 Weak Typing Still Leaks Through DSL YAML Helpers and Dashboard Config Utilities](td-015-weakly-typed-config-and-dsl-contracts.md)
- [TD028 Operator Config Contract Still Collapses Across CRD Schema, Webhook Validation, Canonical Translation, and Sample Fixtures](td-028-operator-config-contract-boundary-collapse.md)
- [TD030 Dashboard Frontend Config and Interaction Surfaces Still Collapse Route Shell, Page Orchestration, and Large UI Containers](td-030-dashboard-frontend-config-and-interaction-slice-collapse.md)
- [TD031 Router Runtime Bootstrap and Shared Service Registry Still Depend on Process-Wide Globals](td-031-router-runtime-bootstrap-and-shared-service-registry-global-state.md)
- [TD032 Training and Evaluation Artifact Contracts Still Drift Across Dashboard, Runtime, and Scripts](td-032-training-evaluation-artifact-contract-drift.md)
- [TD033 Native Binding Runtime Parity and Lifecycle Still Diverge Across Candle and ONNX Backends](td-033-native-binding-runtime-parity-and-lifecycle-gap.md)
- [TD034 Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract](td-034-runtime-and-dashboard-state-durability-and-telemetry-contract.md)
- [TD037 Dev Integration Environment Ownership and Shared-Suite Topology Still Diverge Across CLI, Kind, and CI](td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- [TD039 Control Planes Still Depend on Router Runtime Internals Instead of Contract-Owned Seams](td-039-control-plane-contract-ownership-collapse.md)

### PL0032 Architecture Debt Consolidation

- [TD006 Structural Rule Exceptions Still Cover Active Code](td-006-structural-rule-exceptions.md)
- [TD016 Fleet Sim Subtree Still Diverges from the Shared Ruff Contract](td-016-fleet-sim-shared-ruff-contract-gap.md)
- [TD017 Fleet Sim Migration Still Depends on Relaxed Structure Gates](td-017-fleet-sim-structure-gate-migration-gap.md)
- [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](td-020-classification-subsystem-boundary-collapse.md)
- [TD027 Fleet Sim Optimizer and Public Surface Boundaries Still Collapse Analytical Sizing, DES Verification, and Export Policy](td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md)

## Retired Debt Policy

Retired TD files are removed from this directory. Keep the ID in commit history
instead of carrying a long closed-items table in the active maintainer surface.
