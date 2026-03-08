---
name: dsl-crd
category: fragment
description: DSL, CRD, and Kubernetes translation details for router config changes.
---

# DSL CRD

## Trigger

- The primary skill touches DSL emission/parsing or Kubernetes route conversion

## Required Surfaces

- `dsl_crd`

## Conditional Surfaces

- `router_config_contract`
- `docs_examples`

## Stop Conditions

- DSL/CRD compatibility policy for the new behavior is unclear

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make test-semantic-router`
- `make build-e2e`

## Acceptance

- DSL/CRD translations remain aligned with the router config contract
