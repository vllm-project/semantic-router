---
name: harness-governance
category: fragment
description: Maintains the repository's shared agent contract by updating human-readable docs, executable manifests, and contributor-facing harness wrappers. Use when editing AGENTS.md, repo-manifest.yaml, task-matrix.yaml, governance docs, or any contributor-facing harness interface.
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

## Workflow

1. Read governance docs and repo manifest to understand current contract state
2. Edit shared harness docs, manifests, or contributor-facing wrappers
3. Run `make agent-validate` to check alignment between docs and manifests
4. Run `make agent-ci-gate CHANGED_FILES="..."` to verify all surfaces pass
5. Promote any durable gaps into indexed debt entries if not retired in this change

## Must Read

- [docs/agent/governance.md](../../../../docs/agent/governance.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)
- [tools/agent/repo-manifest.yaml](../../repo-manifest.yaml)
- [tools/agent/task-matrix.yaml](../../task-matrix.yaml)

## Standard Commands

- `make agent-validate`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Canonical harness docs, manifests, and contributor surfaces stay aligned and discoverable
- Durable architecture or harness gaps are promoted into indexed debt entries when they are not retired in the same change
