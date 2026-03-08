---
name: harness-contract-change
category: primary
description: Use when changing the repository's agent contract, docs index, manifests, validation scripts, or contributor-facing harness wrappers.
---

# Harness Contract Change

## Trigger

- Change `AGENTS.md`, `docs/agent/*`, `tools/agent/*`, `tools/make/agent.mk`, or harness-facing CI/workflow classification
- Change contributor-facing wrappers that explain the harness, such as `README.md`, `CONTRIBUTING.md`, or the PR template

## Required Surfaces

- `harness_docs`

## Conditional Surfaces

- `harness_exec`
- `contributor_interface`
- `ci_e2e`

## Stop Conditions

- The edit would create a second conflicting source of truth instead of updating the canonical one
- The rule change cannot be enforced or validated in the same change

## Must Read

- [docs/agent/README.md](../../../../docs/agent/README.md)
- [docs/agent/governance.md](../../../../docs/agent/governance.md)
- [docs/agent/plans/README.md](../../../../docs/agent/plans/README.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)

## Standard Commands

- `make agent-validate`
- `make agent-scorecard`
- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Harness docs, manifests, scripts, and contributor wrappers remain aligned
- The change improves discoverability, source-of-truth clarity, or mechanical enforcement of the harness
- Any durable code/spec divergence discovered during the harness change is recorded in the tech-debt register instead of being left only in PR text
