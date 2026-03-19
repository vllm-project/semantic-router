---
name: deployment-profile-stack
category: fragment
description: Modifies non-operator Kubernetes deployment manifests, profile-owned stack resources, and profile-specific platform wiring. Use when a primary skill touches deploy/kubernetes stack manifests outside operator CRDs.
---

# Deployment Profile Stack

## Trigger

- The primary skill touches deploy/kubernetes stack manifests or profile-owned resources outside operator CRDs

## Workflow

1. Read the E2E selection playbook and profile map for the affected stack
2. Modify the deployment manifests or profile-owned resources
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to confirm the profile mapping
4. Confirm manifests and profile-owned validation paths still describe the same behavior

## Must Read

- [docs/agent/playbooks/e2e-selection.md](../../../../docs/agent/playbooks/e2e-selection.md)
- [tools/agent/e2e-profile-map.yaml](../../e2e-profile-map.yaml)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Deployment manifests and the profile-owned validation path stay aligned
