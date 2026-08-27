# PL-0038: Replaceable Control Plane and Inference Enforcement

## Goal

Ship one clean inference architecture in which Envoy owns transport, ExtProc owns
semantic selection and execution of compiled policy, a replaceable control plane owns
all product desired state, and the optional Playground Agent is not part of Router.
Docker and Kubernetes must enforce the same API-key, authorization, quota, routing,
usage, and accounting semantics.

## Scope

- Keep human Router configuration on `version: v0.3`: physical connections in
  `providers.models`, connection-free metadata in `routing.modelCards`, reusable
  routing logic in `recipes`, and callable Mixture-of-Models plus complete Decision
  assignments in `entrypoints`.
- Keep per-Model reliability in structured `providers.models[].control.retry` and
  `providers.models[].control.timeout`, with exact string pricing in
  `providers.models[].pricing`.
- Keep dynamic Dashboard members, Users, Teams, invitations, API keys, AccessPolicy,
  RateLimitPolicy/Budget, provider catalog, routing desired state, usage, and audit in
  the replaceable control plane. PostgreSQL is authoritative desired state and ledger.
- Publish immutable routing and access snapshots to ExtProc. Keep keys and policies
  out of YAML, Envoy routes, xDS, ConfigMaps, and CRDs. Valkey owns applied projections,
  global counters, admission state, settlement idempotency, and usage ingestion.
- Execute API-key verification, discovery/invocation authorization, quota admission,
  and response-actual settlement in ExtProc without request-time PostgreSQL or
  synchronous control-plane calls.
- Return a logical DispatchPlan from ExtProc. Envoy or an external gateway owns
  upstream route/cluster resolution, credentials, connection pools, health, timeout,
  safe retry/fallback, streaming, and backend forwarding.
- Support at least 10,000 API keys and 1,000 Models without one route/resource per key,
  without mandatory one-cluster-per-Model, and without one shared Router reverse-proxy
  cluster.
- Keep Provider Integrations in the control-plane application. They compile authoring
  input into immutable ExtProc and gateway projections; the data plane has no product
  provider catalog.
- Publish the versioned `/management/v1` OpenAPI from the control plane for Dashboard,
  CLI, automation, and independent consoles. Router exposes only inference, ExtProc,
  projection/status, health, and metrics contracts.
- Run Chat and Builder over one optional control-plane Agent kernel. Every model step
  streams standard `/v1/chat/completions` through Envoy; every Builder tool uses the
  control-plane API. Router config, binary, and OpenAPI contain no Agent resources.
- Validate file-only and dynamic Docker/Kubernetes deployment, recovery, scale,
  streaming, tools, images, neutral codecs, fallback, accounting, and publication.

## Non-Goals

- Making Dashboard, control-plane, or Agent availability part of direct inference
  availability.
- Storing dynamic product state in Router YAML or per-resource infrastructure config.
- Letting ExtProc reverse-proxy Model traffic or letting Envoy evaluate semantic
  routing policy.
- Hosting User/Team/key CRUD, invitations, provider catalog, Agent sessions, Tools, or
  Skills in Router Management.
- Introducing another Model-to-Recipe authority beside Entrypoint assignments.
- Using inference API keys as control-plane credentials.
- Retrying after visible output, unknown usage, or any attempt without known-zero
  gateway evidence.
- Maintaining compatibility readers, dual writers, hidden Dashboard authority, or
  migration-only branches in steady-state serving code.
- Adding a dynamic KnowledgeBase resource; the current gap remains TD047.

## Exit Criteria

- Static v0.3 YAML, control-plane OpenAPI, generated clients, schema, examples, and
  Dashboard forms expose one Model/Recipe/Entrypoint contract with readable authoring
  values and no generated runtime identity.
- Router config has no Agent, control-plane listener, Dashboard identity, or
  PostgreSQL desired-state configuration. File-only startup needs no PostgreSQL or
  Valkey.
- Router binary and Router OpenAPI contain no product CRUD or Agent endpoints. The
  reference control plane can be replaced by another implementation of the published
  API and snapshot contracts.
- Every access-enabled inference endpoint uses one ExtProc access evaluator and one
  immutable applied revision. Dashboard outage and request-time PostgreSQL loss do not
  bypass or interrupt already-applied enforcement.
- Multiple ExtProc replicas enforce identical API-key lifecycle, visibility, RPM,
  response-actual token/cost, and concurrency through Valkey.
- ExtProc performs no upstream Model HTTP call. Bundled Envoy and external gateway
  adapters dispatch the selected logical Model and return authenticated attempt and
  usage evidence.
- Priority fallback uses only gateway-proven known-zero transitions, preserves one
  deadline, supports required codec transitions only when the adapter proves them,
  and records every attempt exactly once.
- Provider onboarding changes control-plane Integration composition without a product
  provider branch in Dashboard presentation or data-plane selection code.
- Chat and Builder both stream standard OpenAI-compatible SSE, obey the selected key's
  policy and quota, and write ordinary logs/usage. Builder cannot publish without an
  explicit immutable-plan confirmation.
- API-key detail, live quota, usage, and cost agree at key, User, Team, and namespace
  scope under concurrency and across replicas.
- Docker and Kubernetes pass readiness, restart, replica loss, store failure,
  restrictive mutation, recovery, and forward-schema-upgrade tests.
- 10,000-key and 1,000-Model capacity gates, the full buffered/streaming codec matrix,
  remote hardware regression, repository gates, and PR CI pass.

## Task List

- [ ] `RAC-01` Close strict v0.3 parse/export/migration and remove Agent/control-plane
  product configuration from Router YAML.
- [ ] `RAC-02` Extract product CRUD, identity exchange, policy compilation, routing
  desired state, migrations, usage queries, and audit from Router composition into the
  reference control-plane service without compatibility forwarding.
- [ ] `RAC-03` Close immutable routing/access publication, ExtProc verification,
  effective policy, global admission, response-actual settlement, reconciliation,
  usage rollup, and audit contracts.
- [ ] `RAC-04` Replace Router backend invocation with gateway-owned DispatchPlan
  execution; close Envoy/external-gateway adapters, ProviderCredential injection,
  control/timeout, fallback, terminal evidence, and neutral codec matrix.
- [ ] `RAC-05` Move Agent sessions, Skills, Tools, Tool Sources, web search, artifacts,
  and Builder publication coordination into the optional control-plane Agent service;
  remove `/management/v1/agent-*` and Router Agent packages.
- [ ] `RAC-06` Close Dashboard identity, invitations, Team/User inheritance, API-key
  UX, delegated inference, scoped usage/statistics/logs, Playground, topology,
  responsive layout, accessibility, and negative permission coverage.
- [ ] `RAC-07` Close file-only and dynamic Docker/Kubernetes composition, migration
  ordering, readiness, failure, recovery, and operator docs.
- [ ] `RAC-08` Run isolated 10,000-key and 1,000-Model capacity tests, full protocol and
  Router/Envoy HTTP suites, store-failure scenarios, remote AMD regression, repository
  gates, and PR CI.

## Next Action

Complete `RAC-05` dependency extraction first because Router-hosted Agent state is a
known architecture violation and currently expands Router config and Management API.
Then complete `RAC-02` using the same control-plane boundary before claiming the
proposal is implemented.

## Operating Rules

- PostgreSQL is control-plane desired state and durable ledger; Valkey is applied
  runtime state and global-counter authority.
- Public inference never depends on Dashboard, Agent, synchronous control-plane calls,
  or request-time SQL joins.
- Envoy transports; ExtProc selects and executes compiled policy; the control plane
  authors and publishes; Agent orchestrates optional UX only.
- Human YAML and DSL contain readable names, not generated IDs, revisions, backend
  identities, catalog digests, or secrets.
- Removed layouts use an explicit offline migrator that is never imported by serving,
  validation, publication, or control-plane steady-state code.
- Public behavior changes require API-level negative authorization and failure-mode
  coverage as well as success tests.
- Private deployment details and credentials remain outside tracked artifacts.

## Related Docs

- [Access Control and Quota Accounting](../../../../website/docs/proposals/router-native-access-control.md)
- [Resource Contracts](../../../../website/docs/proposals/router-native-access-control-contracts.md)
- [Provider Integration Registry](../../../../website/docs/proposals/router-native-access-control-provider-catalog.md)
- [Model Dispatch](../../../../website/docs/proposals/router-native-access-control-model-runtime.md)
- [Quota Runtime](../../../../website/docs/proposals/router-native-access-control-quota-runtime.md)
- [Control-plane API](../../../../website/docs/proposals/router-native-access-control-management-api.md)
- [Authorization](../../../../website/docs/proposals/router-native-access-control-authorization.md)
- [Deployment](../../../../website/docs/proposals/router-native-access-control-deployment.md)
- [Neutral Protocol](../../../../website/docs/proposals/multi-protocol-adaptor.md)
- [Optional Agent Harness](../../../../website/docs/proposals/router-native-agent-runtime.md)
- [Upgrade and rollback](../../../../website/docs/installation/upgrade-rollback.md)
