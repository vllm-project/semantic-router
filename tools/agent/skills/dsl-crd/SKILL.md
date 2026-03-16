---
name: dsl-crd
category: fragment
description: Maintains the translation layer between the router DSL configuration and Kubernetes CRD manifests, including DSL emission, parsing, and config-to-k8s mapping. Use when modifying how router config is translated to Kubernetes resources, updating DSL parsers, or changing CRD field mappings.
---

# DSL CRD

## Trigger

- The primary skill touches DSL emission/parsing or router-to-Kubernetes translation

## Workflow

1. Read change surfaces doc to understand DSL/CRD translation dependencies
2. Modify DSL emission, parsing, or router-to-Kubernetes translation code
3. Run `make test-semantic-router` to validate router-side behavior
4. Run `make build-e2e` to verify end-to-end Kubernetes translation
5. Confirm DSL and Kubernetes translation layers remain aligned with the router config contract

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make test-semantic-router`
- `make build-e2e`

## Acceptance

- DSL and Kubernetes translation layers remain aligned with the router config contract
