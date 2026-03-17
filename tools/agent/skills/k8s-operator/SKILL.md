---
name: k8s-operator
category: fragment
description: Manages Kubernetes operator APIs, Custom Resource Definitions (CRDs), and control-plane reconciliation logic for the semantic router. Use when modifying CRD schemas, updating operator controller logic, or changing how the router integrates with the Kubernetes API.
---

# Kubernetes Operator

## Trigger

- The primary skill touches operator APIs, CRDs, or Kubernetes control-plane logic

## Workflow

1. Read change surfaces and module boundaries to understand operator integration points
2. Modify operator APIs, CRDs, or controller logic
3. Run `make generate-crd` to regenerate CRD manifests from API changes
4. Verify operator APIs, CRDs, and controller-facing config agree on the same contract

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)

## Standard Commands

- `make generate-crd`

## Acceptance

- Operator APIs, CRDs, and controller-facing config agree on the same contract
