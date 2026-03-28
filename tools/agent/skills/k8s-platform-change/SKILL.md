---
name: k8s-platform-change
category: primary
description: Modifies Kubernetes-facing operator, CRD, deployment-profile, or DSL translation behavior for semantic-router platform integration. Use when changing operator APIs or controllers, deployment stack manifests, profile-owned platform wiring, or router-to-Kubernetes translation layers.
---

# Kubernetes Platform Change

## Trigger

- Change operator APIs, CRDs, or controller-facing config translation
- Change deployment-profile manifests or profile-owned platform wiring under `deploy/kubernetes/**`
- Change router-to-Kubernetes DSL translation or config bridging layers

## Workflow

1. Read change surfaces, module boundaries, and the E2E selection playbook for the affected Kubernetes-facing surface
2. Modify the operator, deployment profile, or DSL translation behavior
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces and affected profile coverage
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate the affected constraints
5. Verify control-plane behavior, stack-manifest behavior, and DSL translation still describe the same platform contract

## Gotchas

- Do not hide control-plane and stack-resource differences; they can change in the same loop, but they should remain explicit in the final report.
- Manifest-only edits still need profile-aware validation when they change response-api, authz, observability, or routing stack behavior.
- Router config representation debt belongs to `config-platform-change`; Kubernetes-facing translation and deployment ownership belong here.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/e2e-selection.md](../../../../docs/agent/playbooks/e2e-selection.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Operator APIs, DSL translation, deployment manifests, and affected Kubernetes validation paths stay aligned
- Profile-owned resources and control-plane behavior remain explicit even when they change in the same loop
