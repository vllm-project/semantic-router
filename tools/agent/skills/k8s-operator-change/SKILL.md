---
name: k8s-operator-change
category: primary
description: Use when changing operator APIs, CRDs, or Kubernetes control-plane behavior for semantic-router deployments.
---

# Kubernetes Operator Change

## Trigger

- Change operator APIs, CRDs, or controller-facing config translation
- Change Kubernetes deployment control-plane behavior for semantic-router

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
