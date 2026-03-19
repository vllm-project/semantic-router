---
name: router-service-platform-change
category: primary
description: Modifies router-side API, authz, memory, provider, storage, or runtime service modules outside config, decision, selection, and extproc plugin chains. Use when changing apiserver endpoints, authz or rate-limit policy code, memory or response storage flows, provider adapters, or other router service-platform modules.
---

# Router Service Platform Change

## Trigger

- Change router-side API, service, storage, provider, or support modules outside config, decision, selection, and extproc plugin chains
- Change apiserver, authz, rate-limit, memory, response API, storage, or provider-adapter behavior

## Workflow

1. Read change surfaces, module boundaries, and the Go router playbook for the affected service layer
2. Modify the router-side service or API seam and identify whether profile-owned behavior changed
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify surface alignment
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate the affected constraints
5. Update any impacted local or CI E2E expectation when the changed module is profile-owned

## Gotchas

- These modules are not generic fallback territory anymore; if the change is in this surface, keep the service contract explicit instead of hand-waving it as cross-stack glue.
- Storage, authz, and API seams often look local in code but still change E2E behavior through response-api, memory, or authz-rbac profiles.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Router-side service modules, API seams, and affected validation paths stay aligned
- Profile-owned behavior changes update the relevant local or CI E2E expectation
