---
name: k8s-platform-runtime
category: fragment
description: Maintains Kubernetes-facing operator, deployment-profile, and DSL translation implementation slices for semantic-router platform integration. Use when the task changes operator APIs or CRDs, deployment manifests, profile-owned stack resources, or router-to-Kubernetes translation behavior.
---

# Kubernetes Platform Runtime

## Trigger

- The primary skill touches operator APIs, CRDs, deployment manifests, profile-owned stack resources, or router-to-Kubernetes translation behavior

## Workflow

1. Read change surfaces, module boundaries, and the E2E selection playbook for the affected Kubernetes-facing path
2. Modify operator, deployment-profile, or DSL translation behavior as one platform slice
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to confirm affected profile coverage
4. Verify operator APIs, DSL translation, deployment manifests, and profile-aware validation still describe the same platform contract

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/e2e-selection.md](../../../../docs/agent/playbooks/e2e-selection.md)
- [tools/agent/e2e-profile-map.yaml](../../e2e-profile-map.yaml)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Operator APIs, DSL translation, deployment manifests, and profile-aware validation stay aligned on the same platform contract
