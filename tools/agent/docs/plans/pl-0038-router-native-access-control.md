# PL-0038: Router-Native Access Control

## Goal

Ship one Router-owned management and inference-access contract, with the Dashboard
as an optional API client and identical enforcement semantics across Docker and
Kubernetes.

## Scope

- Keep the human-authored Router manifest on `version: v0.3`. Physical connections
  remain in `providers.models`, semantic metadata remains in
  `routing.modelCards`, reusable routing logic remains in `recipes`, and callable
  Mixture-of-Models plus complete Decision assignments remain in `entrypoints`.
- Expose per-Model invocation policy as structured
  `providers.models[].control.retry` and `providers.models[].control.timeout`, with
  exact string pricing under `providers.models[].pricing`.
- Derive process capabilities from static YAML plus optional configured stores and
  services. File-only routing has no database dependency; PostgreSQL enables durable
  resource authority and the optional Management API; PostgreSQL plus Valkey enables
  Router-native inference access and global quotas. There is no serialized
  deployment-mode selector.
- Keep dynamic Users, Teams, API keys, access policies, quota policies, counters,
  usage, and audit out of Router YAML and Kubernetes resources. PostgreSQL owns
  desired state and durable facts; Valkey owns applied projections, global counters,
  admission state, settlement idempotency, and the ingestion stream.
- Authenticate API keys, authorize discovery and invocation, admit quota, and settle
  authoritative usage inside the Router on every public inference path. The current
  request may cross an actual-token or actual-cost limit; the next request is denied.
- Support at least 10,000 independent API keys without request-time PostgreSQL joins
  or per-key gateway configuration.
- Keep Provider Integrations in the control-plane application. They compile typed
  provider inputs into provider-neutral immutable backends; the data plane depends
  only on installed wire codecs and credential adapters.
- Keep Recipe authoring connection-free. Entrypoints select one Recipe, assign Models
  to its readable Decision names, and optionally define Router-owned priority
  fallback with bounded safe-failure classes.
- Publish the versioned `/management/v1` OpenAPI contract for Dashboard, CLI,
  automation, and independent consoles. Keep manifest, HTTP API, and PostgreSQL
  schema evolution independently versioned and explicitly upgraded.
- Cover Dashboard and Playground authorization, delegated inference, Agent Builder,
  usage/cost visibility, provider onboarding, topology, and accessible responsive UX
  through the same Router APIs.
- Validate file-only and durable Docker/Kubernetes deployment, recovery, scale,
  streaming, tools, images, protocol codecs, fallback, accounting, and publication.

## Non-Goals

- Making Dashboard availability part of inference availability.
- Storing dynamic identity, policy, quota, or usage state in YAML, ConfigMaps, custom
  resources, xDS, or gateway routes.
- Introducing another persistent Model-to-Recipe association beside Entrypoint
  assignments.
- Using inference API keys as Management API credentials.
- Letting gateway retries choose another logical Model or create unaccounted work.
- Maintaining multiple serving-time readers or writers for one public contract.

## Exit Criteria

- The strict v0.3 manifest, Management OpenAPI, generated clients, schema, examples,
  and Dashboard forms expose the same Model control, pricing, Recipe, Entrypoint,
  assignment, and fallback contract.
- File-only startup compiles one immutable manifest without PostgreSQL, Valkey,
  Management mutations, or native API-key state.
- Durable routing seeds only an empty PostgreSQL authority from the manifest; every
  later change is an explicit revisioned Management mutation or import.
- Every access-enabled inference endpoint, including discovery, streaming,
  Playground, direct Model tests, and Mixture-of-Models, uses one Router access
  runtime and one effective-policy evaluator.
- Multiple Router replicas enforce the same API-key lifecycle, model visibility,
  RPM, actual-token, actual-cost, and concurrency state through Valkey.
- Settlement records every internal dispatch exactly once, permits the crossing
  request, blocks the next request while over limit, and fences unknown usage instead
  of treating it as zero.
- Provider onboarding changes control-plane Integration composition without adding a
  product-provider branch to the Dashboard or inference runtime.
- Entrypoint publication validates the complete Recipe and assignment graph;
  priority fallback advances only before visible output on Router-proven safe
  evidence and preserves one request deadline and dispatch ledger.
- The Dashboard remains removable without changing authentication, authorization,
  quota, routing, accounting, or Management automation.
- The API-key and usage views agree with live quota state and durable actual-cost
  accounting at key, User, Team, and namespace scope.
- Docker and Kubernetes pass readiness, restart, replica-loss, store-failure,
  restrictive-policy, recovery, and forward-schema-upgrade scenarios.
- A 10,000-key capacity gate, complete buffered/streaming codec matrix, remote
  hardware-backed regression, repository gates, and pull-request CI pass.

## Task List

- [ ] `RAC-01` Close the v0.3 manifest, Management OpenAPI, generated-client, schema,
  import/export, and documentation contract gates.
- [ ] `RAC-02` Close PostgreSQL desired-state, Valkey publication, API-key lifecycle,
  effective policy, global admission, actual settlement, reconciliation, usage,
  rollup, and audit tests.
- [ ] `RAC-03` Close Model control/pricing, Provider Integration compilation,
  ProviderCredential dispatch, Recipe/Entrypoint publication, priority fallback,
  protocol codec, and direct-inference tests.
- [ ] `RAC-04` Close Management authentication, authorization, invitations,
  Team/User inheritance, delegated inference, scoped list/detail/statistics, and
  independent-console contracts.
- [ ] `RAC-05` Close Dashboard, Playground, Agent Builder, topology, cost/quota,
  responsive layout, accessibility, and permission-visibility regression coverage.
- [ ] `RAC-06` Close file-only and durable Docker/Kubernetes composition, schema
  upgrade ordering, readiness, failure, recovery, and operator documentation.
- [ ] `RAC-07` Run `make perf-access-capacity` against an isolated Valkey for the
  reproducible 10,000-key projection, multi-replica admission, usage-lag, memory,
  Redis-operation, and Router-replica failover report; then run the separate full
  protocol, Router/Envoy HTTP, store-failure, remote hardware-backed, repository,
  and pull-request gates.

## Next Action

Run the contract and documentation gates, then close the smallest failing runtime
or end-to-end group in Task List order. Record durable gaps in the linked debt item
instead of adding transitional behavior to the target contract.

## Operating Rules

- PostgreSQL is the desired-state and durable-ledger authority; Valkey is the applied
  runtime and global-counter authority.
- Public inference paths never depend on Dashboard availability or request-time SQL
  joins.
- Human YAML and DSL contain readable names, not generated IDs, revisions, backend
  identities, catalog digests, or secret material.
- Public behavior changes require API-level negative authorization and failure-mode
  coverage as well as successful-path tests.
- Main processors, handlers, config loaders, and CLI commands remain small
  orchestrators over narrow modules.
- Private deployment details and credentials remain outside tracked artifacts.

## Related Docs

- [Router-Native Access Control and Quota Accounting](../../../../website/docs/proposals/router-native-access-control.md)
- [Resource Contracts](../../../../website/docs/proposals/router-native-access-control-contracts.md)
- [Provider Integration Registry](../../../../website/docs/proposals/router-native-access-control-provider-catalog.md)
- [Model Runtime](../../../../website/docs/proposals/router-native-access-control-model-runtime.md)
- [Quota Runtime](../../../../website/docs/proposals/router-native-access-control-quota-runtime.md)
- [Management API](../../../../website/docs/proposals/router-native-access-control-management-api.md)
- [Authorization](../../../../website/docs/proposals/router-native-access-control-authorization.md)
- [Deployment](../../../../website/docs/proposals/router-native-access-control-deployment.md)
- [Neutral Protocol](../../../../website/docs/proposals/multi-protocol-adaptor.md)
- [Agent and Playground Builder](../../../../website/docs/proposals/router-native-agent-runtime.md)
- [Upgrade and rollback](../../../../website/docs/installation/upgrade-rollback.md)
