---
name: harness-governance
category: fragment
description: Human-readable docs, executable manifests, and contributor interfaces that define the repository's shared agent contract.
---

# Harness Governance

## Trigger

- Load this fragment when editing shared harness docs, manifests, or contributor-facing harness wrappers

## Required Surfaces

- `harness_docs`

## Conditional Surfaces

- `harness_exec`
- `contributor_interface`
- `ci_e2e`

## Stop Conditions

- The harness change would leave the indexed docs and executable rules inconsistent

## Must Read

- [docs/agent/governance.md](../../../../docs/agent/governance.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [tools/agent/repo-manifest.yaml](../../repo-manifest.yaml)
- [tools/agent/task-matrix.yaml](../../task-matrix.yaml)

## Standard Commands

- `make agent-validate`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Canonical harness docs, manifests, and contributor surfaces stay aligned and discoverable
- Durable architecture or harness gaps are promoted into the debt register when they are not retired in the same change
