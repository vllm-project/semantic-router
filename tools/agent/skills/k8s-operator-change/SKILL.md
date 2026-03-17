---
name: k8s-operator-change
category: primary
description: Modifies Kubernetes operator APIs, CRD schemas, and control-plane reconciliation behavior for semantic-router deployments. Use when updating operator controller logic, changing CRD field definitions, modifying config translation to Kubernetes resources, or adjusting deployment control-plane behavior.
---

# Kubernetes Operator Change

## Trigger

- Change operator APIs, CRDs, or controller-facing config translation
- Change Kubernetes deployment control-plane behavior for semantic-router

## Workflow

1. Read change surfaces, feature-complete checklist, and module boundaries for operator context
2. Modify operator APIs, CRDs, or controller-facing config translation
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints
5. Verify operator APIs, CRDs, and router-facing translation stay aligned

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Operator APIs, CRDs, and router-facing translation stay aligned
- Relevant Kubernetes-facing validation is updated when behavior changes
