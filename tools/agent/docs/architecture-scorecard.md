# Architecture Status

This page is a compact index of current architecture risk. It intentionally
avoids subjective numeric scores and completed-work receipts; the linked debt
entries carry the evidence and exit criteria.

## Current Posture

| Area | Current gap | Source |
|---|---|---|
| Repository modularity | Active files still require structure exceptions | [TD006](tech-debt/td-006-structural-rule-exceptions.md) |
| Fleet Simulator | Lint, structure, optimizer, and export boundaries differ from the shared target | [TD016](tech-debt/td-016-fleet-sim-shared-ruff-contract-gap.md), [TD017](tech-debt/td-017-fleet-sim-structure-gate-migration-gap.md), [TD027](tech-debt/td-027-fleet-sim-optimizer-and-public-surface-boundary-collapse.md) |
| Classification | Construction and request-time responsibilities remain concentrated | [TD020](tech-debt/td-020-classification-subsystem-boundary-collapse.md) |
| Native bindings | Rust structure exceptions and recorded Go declaration debt increase review cost | [TD042](tech-debt/td-042-ffi-embedding-structure-debt.md), [TD048](tech-debt/td-048-router-control-plane-go-complexity.md) |
| Router Flow state | Redis-backed tool state lacks integration and deployment validation | [TD044](tech-debt/td-044-flow-tool-state-durability-gap.md) |
| Community automation | Content moderation has no reviewed implementation | [TD045](tech-debt/td-045-reviewed-content-moderation.md) |
| ONNX validation | ONNX-only changes lack mandatory runtime coverage | [TD046](tech-debt/td-046-onnx-binding-ci-coverage-gap.md) |
| Router control plane | Management, access, protocol, quota, and usage declarations exceed shared Go complexity limits | [TD048](tech-debt/td-048-router-control-plane-go-complexity.md) |

There is no active release-readiness row. Release risks belong in the active
release plan when a milestone is opened.

## Update Rules

- Add or change a row only when current source or a validation gate changes.
- Put the detailed evidence and testable exit criteria in one debt entry.
- Remove a closed entry from this page and the debt index in the same change.
- Do not use plans, pull requests, or benchmark logs as substitutes for current
  source evidence.

## Owners

- [PL-0032: Architecture Debt Consolidation](plans/pl-0032-architecture-scorecard-ratchet.md)
- [Technical Debt Index](tech-debt/README.md)
