---
name: harness-contract-change
category: primary
description: Modifies the repository's agent contract including AGENTS.md, docs index, manifests, validation scripts, and contributor-facing harness wrappers. Use when updating agent documentation, changing repo manifests, editing validation scripts, modifying CI/workflow classification, or updating contributor-facing guides like README.md, CONTRIBUTING.md, or the PR template.
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

## Workflow

1. Read agent README, governance docs, and tech debt register for current contract state
2. Modify agent contract docs, manifests, validation scripts, or contributor wrappers
3. Run `make agent-validate` to check alignment between docs and manifests
4. Run `make agent-ci-gate CHANGED_FILES="..."` to verify all surfaces pass
5. Record any durable code/spec divergence as indexed debt entries

## Gotchas

- Do not add a prose-only harness rule when the same invariant can be enforced in manifests, scripts, or CI.
- When routing, validation, or contributor workflow changes, update the executable layer and the human-readable layer in the same patch.

## Must Read

- [docs/agent/README.md](../../../../docs/agent/README.md)
- [docs/agent/governance.md](../../../../docs/agent/governance.md)
- [docs/agent/plans/README.md](../../../../docs/agent/plans/README.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)

## Standard Commands

- `make agent-validate`
- `make agent-scorecard`
- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Harness docs, manifests, scripts, and contributor wrappers remain aligned
- The change improves discoverability, source-of-truth clarity, or mechanical enforcement of the harness
- Any durable code/spec divergence discovered during the harness change is recorded in the matching indexed debt entry instead of being left only in PR text
