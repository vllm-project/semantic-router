# PL-0038: Router-Native Access Control

## Goal

Ship the Router-owned managed control plane and inference access runtime defined by
the router-native access-control proposal, with the Dashboard as an optional API
client and identical behavior across Docker and Kubernetes deployments.

## Scope

- Versioned Management API and canonical resource contracts.
- PostgreSQL desired state, usage ledger, rollups, invitations, and audit.
- Valkey projections, exact global counters, settlement idempotency, and usage
  ingestion.
- Inference authentication, model visibility, authorization, admission, actual
  usage settlement, and cost accounting across every public protocol path.
- Model, Recipe, and Entrypoint management, provider credentials, execution policy,
  pricing, per-decision priority fallback, and immutable published routing snapshots.
- Dashboard and Playground migration to Management and inference APIs only.
- One neutral protocol IR and immutable buffered/streaming Codec Registry for every
  public-client and backend-format pair.
- Router-native Agent Profiles, Skills, Tools, durable sessions, Playground Builder,
  probes/evals, and human-confirmed publication.
- API-key cost analytics, live cost quota, and arbitrary bounded sliding windows such
  as an eight-hour currency budget.
- One `vllm-sr serve` launch path with no virtual-model command, recipe/model operand,
  or launch-time routing authoring.
- Docker, Kubernetes, CLI, forward-only schema migration, observability, website
  documentation, and end-to-end tests.

## Non-Goals

- Retaining Dashboard-owned enforcement or any parallel rate-limit implementation.
- Storing dynamic users, keys, policies, or counters in Router YAML, ConfigMaps,
  custom resources, or gateway routes.
- Adding a separate ModelPool or Mixture resource. Entrypoint assignments are the
  only persistent Model-to-Recipe association.
- Using inference API keys as management credentials.

## Exit Criteria

- All normative proposal contracts are implemented as one runtime contract, with
  explicit and testable forward-only PostgreSQL schema migration.
- Direct clients, Playground, discovery, streaming, and non-streaming requests use
  one shared access runtime and produce consistent authorization and accounting.
- Multiple Router replicas enforce globally consistent key state and quota counters.
- The Dashboard has no authoritative access-control store or inference proxy path.
- API-key Usage/detail agree on actual cost, and live currency budget state agrees
  with global settlement across replicas.
- Every authorized reader can open Entrypoint topology, while every routing mutation
  is independently denied without `routing.manage`.
- Priority fallback preserves one request deadline, advances only on proven-safe
  evidence, and records every attempted Model without gateway-hidden retries.
- Standalone, managed Docker, and managed Kubernetes modes satisfy their documented
  dependency and failure semantics.
- A remote AMD deployment passes management, inference, quota, settlement, routing,
  resilience, and UX end-to-end scenarios.
- The complete three-format buffered and streaming codec matrix passes with no
  reachable pair-specific translator or protocol-specific accounting path.
- A real Builder session survives reconnect, builds and evaluates a Recipe using
  live dynamic schemas, waits for immutable confirmation, publishes an Entrypoint,
  and invokes it through ordinary discovery and inference.
- Required repository gates and pull-request CI pass.

## Task List

- [x] `RAC-01` Inventory current changes and map every proposal requirement to an
  owning module, contract, test, migration, or removal.
- [ ] `RAC-02` Implement canonical managed-mode configuration, resource schemas,
  OpenAPI contracts, schema migrations, and store interfaces. The public v0.4
  manifest has readable `models`, model-free `recipes`, and callable `entrypoints`.
  ModelCard metadata stays separate from connections at the authoring boundary;
  generated identity, catalog provenance, and compiled backends stay internal.
- [ ] `RAC-03` Implement Management authentication, typed authorization, identity,
  team, invitation, API-key, policy, provider-credential, scoped statistics,
  usage, and audit services. Statistics must require `usage.read`, omit each
  independently denied resource field, preserve exact decimal counts, and use a
  fixed number of indexed aggregate queries rather than entity pagination.
- [ ] `RAC-04` Implement policy and routing compilation, revision publication, Valkey
  projection, invalidation, and replica-consistent refresh behavior.
- [ ] `RAC-05` Implement shared inference authentication, discovery and invocation
  authorization, global admission, actual usage settlement, and cost accounting.
- [ ] `RAC-06` Implement Model execution policy and pricing, provider dispatch, and
  the Model/Recipe/Entrypoint resource boundary. The
  application-installed Integration Registry is a control-plane extension seam;
  immutable snapshots use only stable protocol/credential adapters and compiled
  non-secret backend values. Integration form fields are consumed by a typed
  control-plane compiler and never enter the data-plane snapshot as an open-ended
  product field map. Decision assignments embed priority-tier Model references and
  a closed safe-fallback policy; Router, not Envoy retry configuration, owns every
  cross-Model transition and dispatch record. Model mutation uses a sparse ETag PATCH;
  execution/pricing edits preserve server-owned compiled backends, and only explicit
  whole-backend replacement invokes credential-use authorization. Provider display
  metadata includes a validated generic icon descriptor so installing a compatible
  Integration never requires a Dashboard provider lookup-table change.
- [ ] `RAC-07` Refactor the Dashboard and Playground into versioned API clients with
  scoped UX, delegated inference sessions, and no data-plane authority. Add the
  routing role matrix, topology/create/assignment gates, API-key cost/detail views,
  live currency budget editor, one-of inherited/existing/inline key quota creation,
  priority-fallback editor, and a shared accessible
  icon/action system. Models and every Recipe-scoped Signals/Projections/Decisions
  editor must use Router resources and revisions directly; remove the Dashboard
  Recipe-draft store and every routing-config mutation shim. Audit every active
  navigation item, primary/secondary button, table action, and empty state for a
  restrained semantic icon without turning labels into icon-only controls.
  Access Usage and the Dashboard home compose Router statistics with Router Usage;
  neither may enumerate all identity, key, or policy pages for summary cards.
  Access entity tables and form selectors must keep independent state. User, Team,
  key, Access Policy, and Rate Limit Policy selectors use bounded server-side search
  plus keyset cursors, hydrate an already-selected resource by ID, and never treat a
  first page of 100 resources as the complete directory.
  Dashboard invitations must materialize a namespace-scoped Dashboard role and a
  separate User-scoped consumer role so automatic first-key onboarding and delegated
  Playground calls work without broadening Dashboard mutation authority. After
  acceptance, the Dashboard shows one authenticated welcome moment before handing
  the same non-persisted secret to the standard API-key delivery/detail flow.
  - [x] Migrate Builder to one selected Router-managed, model-free Recipe document:
    read access is view-only, `routing.manage` owns save/duplicate, immutable
    distribution Recipes are duplicate-only, imports are compiler-projected
    through `recipeDocuments`, and deployment/Model/Entrypoint
    authoring is absent from the Builder client.
- [ ] `RAC-08` Implement standalone and managed Docker/Kubernetes orchestration,
  health semantics, safe bootstrap, and forward-only schema migration. Remove the
  virtual-model command group and Model/Recipe operands or algorithm override from
  serve, plus the stack-scoped active-Recipe pointer/mount/recovery and
  `--recipe-env` second authority; `vllm-sr serve` is the only launch command, and
  its optional `--config` selects exactly one immutable v0.4 bootstrap manifest.
  That flag does not select or author a Model, Recipe, algorithm, or active routing
  state.
  Standalone compiles one read-only manifest and exposes no dynamic
  `/config/router*`, Recipe, knowledge-base, or runtime-sync mutation surface;
  managed authoring uses only the versioned resource Management API.
  Any configuration import that targets a managed installation must call ordinary
  versioned Model, Recipe, and Entrypoint Management APIs; it may not rewrite the
  mounted runtime manifest or create a second activation path.
- [ ] `RAC-09` Add unit, contract, integration, scale, fault, and end-to-end coverage,
  plus user-facing website documentation and examples. Cover cost windows, key-cost
  breakdown/detail parity, fallback failure classes/deadlines/accounting, role-based
  topology and mutation visibility, CLI absence, Dashboard interaction/a11y, and
  server-searched Access selectors with directories larger than one page.
- [ ] `RAC-10` Deploy to the remote AMD environment, validate the complete chain,
  resolve regressions, and complete the required pull-request gates.
- [ ] `RAC-11` Replace canonical-provider and pairwise translation with the neutral
  protocol IR, immutable Codec Registry, explicit capability/fidelity policy,
  buffered and stream engines, and typed errors. Integrate every public path,
  BackendInvoker, plugins, authoritative usage settlement, and readiness; delete the
  former Responses/Anthropic translation branches and cover the complete codec matrix.
- [ ] `RAC-12` Implement Router-native Agent Profiles, Skills, Tool Sources, durable
  sessions/turns/events/artifacts, leased resumable workers, dynamic Router tools,
  context checkpoints, immutable publication approval, generated Management client,
  the vLLM-SR Agent management page, and Playground Builder mode. Remove the
  Dashboard-owned MCP authority and NL Builder endpoint, then pass deterministic CI
  and the real-model publish/discover/invoke acceptance scenario.

## Next Action

Close the remaining release gates: shared-access coverage across every installed
wire format, exact multi-replica quota and reconciliation, priority-fallback image
coverage, Agent worker replacement and reconnect, Docker and Kubernetes receipts,
the 10,000-key capacity target, and the complete remote regression suite. Regenerate
all contracts remotely, run the real-model publish/discover/invoke acceptance path,
then update the pull request and production only from the validated artifacts.

## Operating Rules

- PostgreSQL is the managed desired-state and durable-ledger authority; Valkey is
  the applied runtime and global-counter authority.
- PostgreSQL is also the Agent queue, lease, fence, event, artifact, checkpoint,
  and publication-plan authority. Valkey may accelerate wake-up and event delivery,
  but loss of that optional notification path cannot lose or duplicate a Turn.
- Public inference paths never depend on Dashboard availability or request-time SQL
  joins.
- Main processor, handler, config, and CLI files remain orchestrators; new behavior
  belongs in narrow modules with explicit seams.
- Public behavior changes require contract and end-to-end assertions, including
  negative authorization and failure-mode cases.
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
