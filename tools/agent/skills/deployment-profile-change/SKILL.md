---
name: deployment-profile-change
category: primary
description: Modifies non-operator Kubernetes deployment manifests, profile-owned stack resources, or profile-specific platform wiring. Use when changing deploy/kubernetes stack manifests outside CRDs, such as response-api, ai-gateway, routing-strategies, observability, or streaming profiles.
---

# Deployment Profile Change

## Trigger

- Change non-operator Kubernetes deployment manifests or profile-owned stack resources
- Change deploy/kubernetes stack wiring that affects a named profile but is not a CRD or operator API change

## Workflow

1. Read change surfaces and the E2E selection playbook for the affected deployment profile
2. Modify the stack manifests or profile-owned platform wiring
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify profile selection and surface alignment
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate the affected constraints
5. Confirm stack manifests and the affected profile validation path still describe the same behavior

## Gotchas

- Do not funnel stack-manifest edits into the operator skill just because they live under `deploy/kubernetes`; CRD control-plane changes and stack-resource changes are separate contracts.
- Manifest-only edits still need profile-aware validation when they change response-api, authz, observability, or routing stack behavior.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/playbooks/e2e-selection.md](../../../../docs/agent/playbooks/e2e-selection.md)
- [tools/agent/e2e-profile-map.yaml](../../e2e-profile-map.yaml)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Deployment manifests, profile-owned resources, and affected E2E expectations stay aligned
- Operator-only CRD behavior remains separated from stack-manifest changes
