---
name: router-service-platform
category: fragment
description: Modifies router-side API, authz, memory, provider, storage, and service-support modules outside config, decision, selection, and extproc plugin chains. Use when a primary skill touches apiserver, authz, rate-limit, memory, response API, provider adapters, or related runtime service-support packages.
---

# Router Service Platform

## Trigger

- The primary skill touches router-side API, authz, memory, provider, storage, or service-support packages outside the config and plugin chains

## Workflow

1. Read module boundaries and the Go router playbook for the affected service-layer seam
2. Modify the router-side service or API code
3. Run `make test-semantic-router` to verify behavior
4. Confirm service-layer code and affected validation paths describe the same contract

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Router-side service modules and their validation paths agree on the same contract
