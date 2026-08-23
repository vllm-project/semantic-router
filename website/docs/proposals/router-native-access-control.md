---
title: Router-Native Access Control and Quota Accounting
description: Defines a scalable Router-owned control plane and data plane for inference identities, API keys, model grants, global quotas, usage, and audit.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-08-22

Normative appendices: [resources](./router-native-access-control-contracts),
[Provider catalog](./router-native-access-control-provider-catalog),
[Model runtime](./router-native-access-control-model-runtime),
[neutral protocol](./multi-protocol-adaptor),
[quota runtime](./router-native-access-control-quota-runtime),
[Management API](./router-native-access-control-management-api),
[authorization](./router-native-access-control-authorization), and
[deployment](./router-native-access-control-deployment), plus the
[Agent and Playground Builder](./router-native-agent-runtime).

## Problem

Inference access is a Router responsibility. Because clients call it without the Dashboard, authentication, visibility, quota, and accounting cannot depend on a Dashboard process.

A Dashboard-owned proxy, key checks, quota engine, and authoritative tables would create four structural problems:

- bypassing the Dashboard bypasses the policy boundary;
- Dashboard availability becomes inference availability;
- multiple Router replicas do not share one authoritative enforcement state; and
- another console or automation client cannot manage the same product contract
  without depending on Dashboard internals.

The design must support at least 10,000 API keys with independent model visibility and quota. That state must not expand into Router YAML, gateway routes, xDS, ConfigMaps, or one custom resource per key, which would couple every mutation to configuration distribution and reloads.

## Decision summary

This proposal makes the following decisions:

1. Semantic Router owns the public inference access boundary and the versioned
   Management API.
2. PostgreSQL is the authoritative desired-state store for identities, keys,
   policies, Models, Recipes, Entrypoints, usage, and audit.
3. Valkey or Redis is the applied runtime store for credential projections,
   compiled policies and routing snapshots, global counters, settlement idempotency,
   and the durable usage-ingestion stream.
4. The Dashboard is an optional Management API client. It never registers or
   proxies public inference routes and never reads access-control tables directly.
5. Managed mode uses PostgreSQL as its only desired-state authority; Router YAML and
   Kubernetes resources contain infrastructure bootstrap only. Standalone mode may
   use one local routing manifest compiled by the same validator/snapshot compiler,
   but exposes no dynamic routing or access Management mutations.
6. API-key authentication, model discovery, invocation authorization, and quota
   admission use the same compiled effective policy.
7. Request counts are admitted before inference. Token quotas use authoritative
   response usage only: the current request may cross a token limit, and the next
   request is blocked.
8. An Entrypoint is the callable Mixture-of-Models. Decision assignments are embedded
   directly in its action; pools and mixtures are derived views, not resources.
9. Default standalone Docker requires neither PostgreSQL nor Valkey. Managed mode
   requires both stores even if API-key enforcement is off, and enabling access
   requires managed mode. Kubernetes uses the same modes and semantics.
10. Runtime code accepts exactly the contracts defined here. Each resource has one
    authoritative writer, one validated read path, and one publication path.
11. Provider products are application-installed control-plane Integrations. Their
    Definitions and compiler plugins render canonical Model backends; the inference
    data plane receives only a stable wire-format ID, credential-adapter ID, semantic
    capabilities, and compiled non-secret settings. It contains no product-provider
    switch or Dashboard catalog.
12. Every public and backend format uses one neutral request/response/event IR and an
    immutable Codec Registry. Formats compose through the IR; pair-specific
    translators and protocol-specific accounting paths do not exist.
13. Playground Builder is a durable Router Agent workflow over the same versioned
    Management and inference APIs. It may prepare a Recipe publication but only an
    authorized human may commit the exact reviewed plan.

## Goals

- Enforce API-key authentication and authorization on every public inference path.
- Keep `GET /v1/models`, invocation, Playground, and direct-model testing consistent.
- Make key disable, expiry, grant reduction, and quota reduction globally effective
  across Router replicas without configuration reloads.
- Support reusable policies and a distinct policy per key without request-time SQL
  joins.
- Support exact rolling RPM, actual-token and actual-cost quotas, daily windows, and
  concurrency through one extensible rule model.
- Make per-decision Model fallback explicit, priority ordered, safe before visible
  output, and fully accounted by Router rather than hidden in gateway retries.
- Preserve complete per-request accounting for streaming, non-streaming, and
  multi-dispatch Mixture-of-Models execution.
- Expose stable OpenAPI contracts so the Dashboard, CLI, automation, and independent
  consoles have equal management capability.
- Keep the Docker deployment small while preserving a direct path to stateless
  Kubernetes scale-out.
- Make failure and consistency semantics visible in APIs, health endpoints, and the
  Dashboard.
- Let an operator add an ordinary compatible Provider through the control-plane
  Integration Registry, without changing the Dashboard or inference runtime.
- Preserve tools, multimodal content, reasoning, stop semantics, streaming events,
  and authoritative usage across a complete client/backend protocol codec matrix.
- Let users describe, tune, probe, evaluate, review, and publish a Mixture-of-Models
  through a durable Playground Builder session without creating a second Recipe or
  inference authority.

## Non-goals

- Storing inference API keys in static Router configuration or Kubernetes objects.
- Making the Dashboard an inference proxy, policy engine, or source of truth.
- Using inference credentials as management credentials.
- Conflating inference credentials with the separately encrypted
  ProviderCredentials used by the Router to call model backends.
- Choosing or scheduling a physical model replica; access grants target logical
  Router resources.
- Treating the analytics ledger as the source of live quota remaining.
- Providing an in-memory or SQLite enforcement mode with different quota semantics.
- Hiding approximate rate-limit algorithms behind labels that imply exact windows.

## Product and trust boundaries

The following diagram is managed mode. Standalone omits Dashboard, Management,
PostgreSQL, Valkey, projector, and writers and loads one locally compiled routing
snapshot before serving.

```mermaid
flowchart LR
    Client["Inference client"] --> Gateway["Public gateway"]
    Gateway --> Access["Router access runtime"]
    Access --> Runtime["Semantic routing runtime"]
    Runtime --> Backend["Model backend"]
    Access <--> Hot["Valkey runtime state"]
    Runtime -->|"actual usage"| Access

    Dashboard["Optional Dashboard"] --> Management["Router Management API"]
    CLI["CLI / automation / custom console"] --> Management
    Agent["Router Agent workers"] --> Management
    Agent --> Access
    Integrations["Provider Integration Registry"] --> Management
    Management --> PG["PostgreSQL desired state"]
    PG --> Projector["Policy projector"]
    Projector --> Hot
    Hot --> Stream["Usage stream"]
    Stream --> Writer["Usage writer"]
    Writer --> PG
```

| Component | Owns | Must not own |
| --- | --- | --- |
| Public gateway | Listener, transport filtering, access-service calls, and forwarding | API-key records, policy compilation, quota state, or usage truth |
| Router access runtime | Credential verification, trusted principal context, grants, admission, settlement, and global quota decisions | Browser sessions or Dashboard presentation state |
| Router routing runtime | Entrypoint resolution, signals, projections, decisions, algorithms, plugins, neutral protocol processing, codec dispatch, and backend invocation | Product-provider catalogs, Management identity authentication, or mutable policy storage |
| Router Management API | Versioned CRUD, Provider Integration catalog and compilation, effective-policy evaluation, policy publication, usage queries, and audit | Public inference proxying or product-specific data-plane branches |
| Router Agent runtime | Durable sessions, leased turns, trusted Skills, authorized Tools, probes/evals, and immutable publication plans | Autonomous publication, backend credentials, or a second Recipe/inference path |
| PostgreSQL | Authoritative identities, routing resources, policies, revisions, ledger, rollups, and audit | Per-request hot-path reads |
| Valkey/Redis | Applied credential/policy projections, compiled routing snapshots, global counters, idempotency, and ingestion stream | Long-term analytics or the only copy of desired state |
| Dashboard | Product UX over the Management and inference APIs | Direct database access or a required data-plane hop |

One Router image can expose the ExtProc gRPC service, internal authentication and
quota gRPC services, Management HTTP, health, and metrics. When managed access is
enabled, every ingress adapter executes access authentication, authorization, and
quota admission before semantic ExtProc. After verification, the Router removes the
inference `Authorization` header; backend dispatch injects a separate
ProviderCredential. No access-enabled adapter may bypass the shared `AccessRuntime`.
Standalone and managed routing-only modes do not start `AccessRuntime`; their public
discovery and invocation paths still share the same Entrypoint resolver and active
routing snapshot.

### Two identity planes

Management identity and inference identity are intentionally different:

- A **ManagementPrincipal** authenticates to the Management API through OIDC,
  mTLS, or a service account. Its ManagementRole controls administrative actions.
- A Router **User** consumes model service. It may belong to teams and own API keys.
- A Dashboard account is one possible login UX for a ManagementPrincipal. It may be
  linked to one Router User, but the link is explicit rather than inferred from an
  email address.
- An **InferenceAPIKey** authenticates only to public inference APIs unless an
  explicit, separately issued management credential says otherwise.

This keeps Dashboard login state, ManagementRole authority, TeamRole membership, and
inference access policy from sharing an ambiguous `role` field.

## Terminology

| Term | Meaning |
| --- | --- |
| `Namespace` | The top-level isolation boundary for management, policies, routing resources, and analytics. |
| `DashboardAccount` | Optional browser-login identity and session UX; never an inference principal or Router authority by itself. |
| `ManagementPrincipal` | An actor authorized to call Management APIs. |
| `ManagementRole` | A permission preset scoped through a Management role binding. |
| `ServiceAccount` | A non-human ManagementPrincipal used by automation. |
| `User` | A model-service consumer identity owned by the Router control plane. |
| `Team` | A collection of Users with defaults and optionally shared hard caps. |
| `TeamMembership` | A User's membership and TeamRole in one Team. |
| `TeamRole` | Membership authority inside one Team; independent of ManagementRole. |
| `RoutingRole` | A typed routing-context value derived from Router-owned subject state; it grants no Management capability. |
| `InferenceAPIKey` | A stable logical key resource used for ownership, policy, usage, and URLs. |
| `APIKeyCredentialVersion` | One secret version for an InferenceAPIKey; rotation creates another version. |
| `DelegatedInferenceSession` | A short-lived session linking a Management session, User, and permitted logical key. |
| `DelegatedInferenceCredential` | The non-revealable Bearer secret issued for one DelegatedInferenceSession. |
| `AccessPolicy` | A reusable set of explicit discover/invoke grants. The Dashboard may label it **Access Group**. |
| `RateLimitPolicy` | A reusable ordered set of quota rules. The Dashboard may label it **Budget**. |
| `AccessPolicyBinding` | Attaches an AccessPolicy to a key, User, or Team; it owns no quota counter. |
| `RateLimitBinding` | Attaches a RateLimitPolicy to a key, User, or Team and owns its counters. |
| `ModelGrant` | Permission on a stable Entrypoint or Model identifier. |
| `QuotaCounter` | Live enforcement state in Valkey. |
| `UsageEvent` | An immutable accounting fact persisted in the analytics ledger. |
| `ProviderCredential` | A secret used by the Router to call a backend; never an inference key. |
| `ProviderIntegration` | An application-installed control-plane extension that contributes one immutable Provider Definition. |
| `ProviderDefinition` | Product metadata, origin rules, typed UX fields, compiler configuration, and references to stable adapters. |
| `WireFormat` | A stable data-plane contract such as OpenAI Chat or Anthropic Messages, resolved through the immutable Codec Registry; never a product catalog entry. |
| `CredentialAdapter` | A stable secret materializer such as Bearer or `x-api-key`; never a product catalog entry. |
| `TenantContext` | Router-issued trusted request identity and policy context. |

The routing `authz` signal consumes a verified TenantContext as a routing fact.
API-key authentication and Model authorization belong to the AccessRuntime before
Recipe evaluation.

## Routing resource boundary

This section defines the complete human authoring and internal publication boundary.
Runtime code consumes this contract through the canonical validator and snapshot
compiler.

Only three persistent routing concepts are needed:

- **Model**: one logical model with a semantic `card`, one or more human-readable
  `connections`, and optional advanced runtime and pricing settings.
- **Recipe**: reusable signals, projections, decisions, algorithms, and plugins.
- **Entrypoint**: a callable Mixture-of-Models that selects a Recipe and assigns
  Models to that Recipe's readable decision names.

The manifest represents those concepts directly as top-level `models`, `recipes`,
and `entrypoints` collections. Human YAML, DSL, and Dashboard authoring use readable
names. They never expose resource UIDs, revision hashes, backend IDs, catalog digests,
compiled adapters, or database keys. The importer resolves names within one Namespace
and generates publication identity internally. The Management API still returns
stable resource IDs, revisions, and ETags for safe automation and optimistic
concurrency; those values are never serialized back into human source.

This boundary does not expose connections to Recipe authors. Every Model read has a
permission-filtered **ModelCardView** projection containing semantic identity,
modality, capabilities, context, reasoning family, quality, LoRAs, and tags. Recipe
and DSL editors never receive an origin, ProviderCredential, authentication field,
or compiled backend. `ModelCardView` has no independent CRUD or persistence: it is
derived from the same Model revision, so card metadata cannot drift from the Model
selected by an Entrypoint.

The DSL may render that projection as a `MODEL` card block for authoring context, but
the block contains no connection, credential, UID, or revision syntax. Saving a
Recipe mutates only `Recipe.document`; executable Model selection remains an
Entrypoint assignment action.

An Entrypoint is the product's Mixture-of-Models. Its decision assignments are the
only stored mapping from a Recipe to Models; aggregate Model views are calculated
from those assignments. Immutable UIDs, revisions, compiled backends, and catalog
provenance exist only after validation at the internal snapshot boundary.

```yaml
version: v0.4
billing_currency: USD
models:
  - name: local/fast
    card:
      description: Fast general model
      capabilities: [chat, tools]
      modality: text
    connections:
      - provider: vllm
        interface: chat
        endpoint: http://model:8000/v1
        model: fast-model
    runtime:
      max_retries: 1
      request_timeout: 60s
      stream_timeout: 10m
    pricing:
      input_cost_per_million_tokens: "0.10"
      output_cost_per_million_tokens: "0.40"
  - name: remote/frontier
    card:
      description: Deep reasoning model
      capabilities: [chat, tools, reasoning]
      modality: text
    connections:
      - provider: openai-compatible
        interface: chat
        endpoint: https://models.example.com/v1
        model: frontier-model
        credential: frontier-api

recipes:
  - name: balance
    description: Match each request to the right capability.
    document:
      signals:
        complexity:
          - name: workload
            threshold: 0.15
            easy: {candidates: [short direct answer]}
            hard: {candidates: [multi-step analysis with trade-offs]}
      decisions:
        - name: Simple
          rules:
            operator: NOT
            conditions: [{type: complexity, name: workload}]
        - name: Complex
          rules:
            operator: AND
            conditions: [{type: complexity, name: workload}]

entrypoints:
  - name: vllm-sr/blend
    aliases: [blend]
    recipe: balance
    assignments:
      Simple:
        models: [{model: local/fast}]
      Complex:
        models:
          - model: remote/frontier
            priority: 0
            reasoning: {enabled: true, effort: high}
          - model: local/fast
            priority: 1
        fallback:
          strategy: priority
          on: [unavailable, overloaded]
```

The common Entrypoint form is `name + recipe + assignments`; the Dashboard presents
exactly that flow. Each assignment selects Models by readable name and may add a
priority fallback. Empty arrays, zero values, effective defaults, and
compiler-owned fields are omitted from authoring exports.

One Model reference inside that value has this complete v0.4 shape:

```yaml
model: remote/frontier
priority: 0
weight: "1"
lora: code-specialist
reasoning:
  enabled: true
  effort: high
  description: Use deliberate reasoning for this role.
```

Only `model` is required. `priority` is an integer from 0 through 31 and defaults
to 0; lower numbers are preferred. `weight` is a positive canonical decimal used by
algorithms that select or sample several assigned Models; it defaults to `"1"`.
`lora` selects an adapter declared by the pinned Model revision. `reasoning` is
an assignment-local invocation control because the same logical Model may serve a
reasoning role in one decision and a direct-answer role in another. `effort` must be
accepted by the Model's reasoning family, and `description` is a bounded runtime
instruction rather than display metadata. Model endpoint, capability, execution,
pricing, and provider credential fields remain on Model; algorithm orchestration
remains on Recipe. Publication rejects assignment fields that the pinned Model or
Recipe decision cannot consume. There is no second assignment or pool resource.

Fallback is absent by default. `strategy: priority` requires at least two contiguous
priority tiers beginning at zero and a bounded non-empty `on` set drawn from
`unavailable`, `overloaded`, and `timeout`. It is valid only for a Recipe decision
whose dispatch cardinality is one. Multiple references at the active priority remain
the Recipe algorithm's weighted candidate set; Models in a lower priority are never
sampled while a higher tier is eligible. A multi-dispatch decision keeps every Model
at priority zero and cannot silently reinterpret required parallel Models as backups.

Router first applies the selected Model's own bounded safe retries. It advances to
the next priority only before a client-visible byte and only when the backend adapter
proves the configured failure class and `known_zero` billable usage. An ambiguous
timeout, partial stream, policy denial, caller error, or unknown usage is terminal.
Every attempt, skipped-unhealthy tier, fallback transition, and final Model revision
is recorded in the dispatch ledger. Envoy supplies transport and endpoint health but
does not perform cross-Model retries, because hidden retries would break Model
identity, quota evidence, and cost accounting.

Provider selection is an authoring concern inside Model create/import. The control
plane resolves a `provider_id` from the active Integration Registry, validates the
chosen origin and credential, and compiles a backend containing one `wire_format`,
canonical origin, provider model ID, non-secret connection values, and an optional
ProviderCredential UID. `provider_id` remains only for attribution and immutable
credential binding. At dispatch BackendInvoker resolves that wire format in the
immutable Codec Registry,
not on a product name. The complete extension and rolling-update contract is in the
[Provider catalog appendix](./router-native-access-control-provider-catalog).

In managed mode, PostgreSQL owns Model, Recipe, and Entrypoint desired state. Draft
Models and Recipes can be edited independently, but they do not enter the data plane.
Publishing an Entrypoint validates the complete referenced chain, compiles a
content-addressed routing snapshot, stages it in Valkey, and atomically advances the
namespace routing pointer only after every active Router replica can load it. Only
resources reachable from a published Entrypoint enter that snapshot.

In managed mode, YAML is an authoring/import manifest for Management API clients, not
a second runtime source of truth. A Dashboard, custom console, or automation imports
Models, Recipes, and Entrypoints through the ordinary resource APIs with the same
ETags, validation, outbox, and publication gates as interactive edits. Built-in
Recipes are versioned artifacts installed through that same control-plane path.

Standalone mode has no PostgreSQL, Valkey, dynamic access control, or routing
Management mutations. One local manifest is its sole routing authority and is
compiled into the identical immutable snapshot shape before readiness. Switching
between modes is an explicit export/import and restart, never live dual authority or
a runtime fallback.

Entrypoint-local rules can select a different Recipe or assignment set from trusted
claims without adding another resource layer:

```yaml
entrypoints:
  - name: vllm-sr/blend
    aliases: [blend]
    rules:
      - name: premium
        matches:
          - claim:
              name: routing_tier
              exact: premium
        recipe: balance
        assignments:
          Simple: {models: [{model: local/fast}]}
          Complex: {models: [{model: remote/frontier}]}
      - name: default
        recipe: balance
        assignments:
          Simple: {models: [{model: local/fast}]}
          Complex: {models: [{model: local/fast}]}
```

Entrypoint matching and access authorization remain separate:

- AccessPolicy decides whether the caller may discover or invoke the Entrypoint.
- The Entrypoint resolver then selects one Recipe and assignment set from the trusted
  TenantContext and normalized inference path.
- Entrypoint rules never contain API keys, Users, Teams, quota, or another global
  grant table.

Routing claims have one authoritative source. In managed-access mode, a namespace
defines a bounded typed claim schema, and the Management API stores values against a
Key, User, or Team subject. Effective values resolve field by field at Key, then User,
then context Team; a Team-owned key resolves Key then owner Team. The policy projector
validates and compiles those values into the key's active Valkey projection, and AuthN
copies only that compiled set into the signed, request-bound TenantContext. Client
headers, request bodies, model names, Dashboard sessions, and provider responses can
never set or override them. A value controls routing only and never implies model
access or Management authority.

The initial schema permits at most 16 namespaced string, boolean, or bounded-integer
claims and exact comparisons. `routing_tier: premium` is an ordinary schema-validated
value, not a hard-coded header. Claim-schema or subject-value changes use their own
Management permission, revision, audit, and key-policy publication fan-out. Managed
routing-only and standalone modes have no authenticated subject source, so they reject
claim matchers at publication and use path rules or distinct Entrypoints instead.

The initial matcher surface is intentionally small: exact trusted-claim match, exact
path, and segment-aware path prefix. Raw client identity headers, query parameters,
regular expressions, arbitrary request bodies, and general expression languages are
not matchers. Matchers inside one rule are ANDed. More exact identity predicates
outrank fewer predicates, exact path outranks prefix, and a longer prefix outranks a
shorter prefix. Equally specific rules with different actions are rejected at
publication, so list order never changes routing.

The compiled resolver returns `matched`, `claimed_no_match`, `unclaimed`, or
`ambiguous`. A claimed Entrypoint with no matching rule returns the same
nondisclosing `404` as a forbidden or nonexistent model; it never falls through to
a default Recipe or concrete Model. Ambiguity fails publication and fails closed if
encountered at runtime.

The access layer grants `discover` and `invoke` on the resolved Entrypoint identity. It does not
duplicate or mutate assignments. Recipe and Entrypoint CRUD remain separate in the
[Management API contract](./router-native-access-control-management-api).
`resolve` is a permission-checked dry run over path and, in managed access, an optional
subject context. A subject is required only for claim rules. Managed routing-only
evaluates path/default rules without a subject; standalone has no Management resolve.
Overrides are separately authorized simulations. The response reports outcome,
Recipe, assignments, and explanation without invocation. When access is enabled,
`GET /v1/models` first
applies AccessPolicy, then includes only Entrypoints whose resolver is visible for the
caller and optional `for_path`; invocation uses that same resolver. When access is
disabled, discovery exposes only published aliases from the active snapshot and
invocation uses the same resolver without credential policy. There is no separate
pool or mixture resource API.

## Canonical resource and API contracts

The normative resource relationships, PostgreSQL schema, credential lifecycle,
policy inheritance, counter ownership, and Valkey projection live in the
[resource contract appendix](./router-native-access-control-contracts). Management
identity exchange, delegated inference, exact endpoints, and response rules live in
the [Management API appendix](./router-native-access-control-management-api).
Permissions, exact role presets, scope containment, and operation authorization live
in the [Management authorization appendix](./router-native-access-control-authorization).
In
particular:

- one logical InferenceAPIKey owns independently rotatable credential versions;
- Access selects Key, then User, then context-Team policy;
- quota selects one inherited allocation and additionally enforces explicit shared
  hard caps;
- reusable policy definitions never imply shared counters;
- counter identity is `binding_id + rule_id`;
- model grants use explicit stable Entrypoint or Model IDs; and
- the OpenAPI contract, not Dashboard internals, is the management product surface.

## Data-plane request flow

The following flow applies when managed access is enabled. Standalone has no
inference identity or quota resources and begins from its already compiled routing
snapshot.

1. The public gateway removes client-supplied identity and policy headers.
2. Router AuthN identifies an API-key or delegated-session credential by its public
   prefix, loads the corresponding Valkey projection, verifies HMAC in constant
   time, and checks credential/session, key, User, Team, membership, expiry, and deny
   barriers.
3. Router AuthZ loads the compiled revision and resolves the requested resource with
   the same evaluator used for discovery.
4. The Router creates a typed TenantContext containing only the admission ID,
   namespace, key, User, Team, compiled effective routing context, and policy revision.
   When it crosses a process boundary it is signed, audience-bound, and
   may start work only within a short window. Accepted work keeps an in-process
   principal for the request lifetime; the context is not reusable authentication.
5. Quota admission atomically checks every applicable rule and consumes known
   request/concurrency units.
6. Semantic routing resolves the Entrypoint rule, Recipe, and assignments, then
   journals each bounded dispatch intent before invoking the allowed backend path.
7. The dispatch journal and request-scoped ledger aggregate authoritative input and
   output usage across every backend dispatch made by the Recipe.
8. On terminal response, settlement atomically applies actual token usage, records
   idempotency, and appends the UsageEvent to the stream.
9. Response metadata identifies the limiting rule and whether the values are the
   admission snapshot or terminal settlement. Live detail remains in the Management
   API.

Invalid credentials return `401`. Valid credentials without resource access receive
the nondisclosing `404`. An enforced quota returns `429` with a bounded `Retry-After`.
An unavailable required runtime store returns `503`, not `429`.

## Global quota algorithms

Admission is one Valkey Function or Lua operation:

1. load the effective binding IDs and ordered rules;
2. remove expired rolling-window entries;
3. check every enforced request, token, calendar, concurrency, and hard-cap rule;
4. return a denial without consuming any counter if any rule fails;
5. consume RPM and concurrency only after every check succeeds; and
6. return the limiting rule's `limit`, `used`, `remaining`, and `reset_at`.

`sliding_log` provides exact "the last 60 seconds" semantics. Requests use a sorted
set of admitted request IDs. Token rules use timestamped settlement IDs with token
amounts and a running total maintained atomically while expired members are removed.

`token_bucket` or GCRA provides an explicit O(1) alternative for very high request
rates only. `calendar_window` handles day or month allocations with a named timezone.
The API and UI always show the selected algorithm and reset semantics.

Every HTTP attempt receives a Router-generated `admission_id`. Internal retries of
the same admission and request digest do not consume RPM twice; a conflicting reuse
is rejected and audited. A client `Idempotency-Key` avoids a second charge only when
the Router can return the stored terminal response without another dispatch;
otherwise a new HTTP attempt gets a new admission and is charged. A quota-denied
request consumes nothing. Once dispatched, an upstream failure consumes one request.
`Retry-After` reflects the latest reset among rules that denied admission.

All window timestamps come from Valkey server time, never a Router Pod or client
clock. Concurrency and pending accounting use an admission lease stored in a sorted
set by deadline. Long requests heartbeat the lease. Admission and settlement scripts
scan opportunistically, while a reaper atomically turns an expired, unsettled lease
into a persistent `unknown` fence. A lease never disappears by TTL alone.

## Actual-token accounting

Token limits never reserve a prompt estimate before inference:

```text
before request: check only already settled token usage
after response: settle authoritative provider-reported usage
crossing request: allowed to finish
next request: denied with 429 while the rule remains over limit
```

Actual-only overshoot equals the real token total of every admitted but unsettled
request. It has a calculable upper bound only when both `max_concurrency` and an
enforced per-request maximum charge for the corresponding metric exist. Output quota
needs a generated-token cap; input quota also needs request-body, model-context, and
multimodal-input bounds plus a bound on internal dispatch count; total quota needs all
of them. The Recipe validator must prove those bounds for a limited Entrypoint.
Strict zero-overshoot token admission would require an estimate or reservation and is
not claimed.

The Router requests usage from every streaming provider, normalizes provider wire
formats, and counts all internal Recipe dispatches. A multi-model external request
consumes RPM once and token quota once using the sum of its dispatch ledger. The
ledger pins Model pricing revisions and keeps per-dispatch token/cost breakdowns.

Settlement uses the Router-generated admission ID:

- the same admission ID and the same canonical usage is an idempotent success;
- the same admission ID with different usage is rejected and raises an accounting
  alert; and
- the first settlement updates every token counter, writes a settlement marker, and
  appends one request-level UsageEvent atomically.

Streaming settlement occurs before the terminal stream marker is committed. Client
disconnect does not cancel bounded upstream usage collection. Non-streaming,
streaming, and looper execution all pass through the same finalizer.

Pre-stream headers contain only admission-time limit/remaining/reset and declare
`x-ratelimit-snapshot: admission`; they exclude that response's tokens. Non-streaming
headers may carry terminal values after settlement. Streaming returns them in an
opt-in final event/trailer when supported, else via admission ID in quota/request detail.

The default `input_tokens`, `output_tokens`, and `total_tokens` meters count the
sum of authoritative backend usage across the dispatch ledger, so every routed model
token is charged exactly once. Optional `served_input_tokens`,
`served_output_tokens`, and `served_total_tokens` meter the canonical public
request/response separately; `cost` prices actual backend billing buckets.
A cache hit is known zero for backend tokens and may
settle stored canonical served usage when a served-token rule exists. A failure
proven to occur before inference is known zero. A normal backend response is known
actual. A partial generation, disconnect, or cache record without the usage required
by an applicable rule is unknown.

Every provider and short-circuit adapter emits one explicit usage state:
`known_zero`, `known_actual`, or `unknown`; absence of a wire field is never
interpreted implicitly. A backend that cannot produce valid authoritative usage is
not eligible for an enforced token-limited public Entrypoint. Unknown usage affecting
an `enforce` rule marks the lease unknown, fences those bindings, and makes subsequent
requests for that scope fail with `503` until reconciliation. Shadow-only unknown is
recorded as incomplete but never changes admission or availability. No estimated
value is silently mixed into an actual counter.

## Usage and audit pipeline

The same settlement operation performs:

```text
counter updates + settled marker + XADD usage-stream
```

Router usage workers consume with a consumer group and batch PostgreSQL writes. A
compact settlement table supplies global request-id idempotency while time-partitioned
event tables retain analytics. A stream item is acknowledged only after both records
commit. This is at-least-once delivery with exactly-once logical persistence.

Quota counters and the analytics ledger have different purposes:

- quota APIs read the live counter engine, never reconstruct `remaining` from usage;
- Usage pages query immutable events and verified rollups;
- audit records management actions, reveal, authentication anomalies, policy
  publication, and accounting faults; and
- optional request/response payload logging is a separate retention-controlled
  facility. Quota correctness never depends on payload logs or log scraping.

Raw request, dispatch, and attempt facts use one aligned UTC-month partition
hierarchy. Every Router replica takes the same PostgreSQL advisory-lock contract;
writers create a missing event month inside their ledger transaction, while one
bounded maintenance pass creates the configured future months or retires a complete
old month. A pass inspects at most 32 retirement candidates and drops at most one
aligned month, rotating blocked candidates so metadata locks and transaction work
stay bounded. No API key, User, or Team becomes a physical partition.

The event and dispatch inserts mark a durable minute bucket dirty in the same
transaction. Minute refresh moves that exact dependency to the hour queue, hour to
day, and day completion clears it, always in the transaction that replaces the
rollup. A crash before queue clear rolls back the refresh; a crash after commit leaves
the next grain durably discoverable. This avoids full raw-table reconciliation scans
and is safe when multiple replicas claim the same work.

One-minute rollups retain high-resolution short-range charts; hourly and daily
rollups serve long ranges. Queries select a grain from the requested time range,
return the grain in the response, and use bounded keyset pagination plus event-date
partition pruning for raw logs. Request detail first resolves the permanent
settlement directory, then reads exactly one event, dispatch, and attempt partition.
This keeps hundreds of Users and multi-terabyte token totals usable without scanning
raw history.

Raw usage and audit are retained indefinitely by default. Raw usage deletion is an
explicit operator opt-in and applies only to complete months after every dirty rollup
dependency has cleared and no inference replay, open unknown-usage fence, or
unfinished reconciliation still references the month. Settlement rows are permanent
digest tombstones: a matching late stream redelivery remains idempotent after raw
retirement, while a different digest is still an accounting conflict. Retired raw
request detail returns not found; aggregate rollups remain queryable. Request and
response payload capture is disabled by default and, when enabled, uses an
independent bounded encrypted-payload policy. An analytical sink or object archive
can be added later without changing enforcement.

### Authenticated outcome feedback

Post-response feedback is an inference-plane operation, not a Management mutation.
`POST /v1/router/outcomes` is mounted only on the public inference listener and
requires an API-key or delegated-inference credential plus an `Idempotency-Key`.
The Dashboard calls that endpoint with its bounded delegated inference session; it
does not proxy feedback through a Dashboard store or a Management service identity.

The Router derives namespace, logical API-key, User, Team, and source from the
authenticated session. Caller-supplied provenance is ignored. It accepts a bounded
verdict payload, loads the durable replay record, and requires exact namespace and
logical-key ownership. A Model-targeted outcome must name the Model revision actually
served by that replay. A missing replay, another subject's replay, or a different
Model returns the same nondisclosing not-found response.

The idempotency key is hashed with the logical key and replay identity. One
PostgreSQL transaction claims the unique digest, appends the immutable outcome, and
enqueues any learning projection work. Concurrent submissions to different Router
replicas therefore produce one logical outcome. Failed transactions release the
claim; a committed duplicate returns the original receipt. A fixed Router-owned
Valkey abuse limit applies per logical key across replicas, but feedback does not
consume inference request/token/cost quota because it performs no Model dispatch.
Adaptive state is rebuilt from durable outcomes and published through the same
revisioned learning boundary; process-local maps are never an authority.

## Desired-to-applied consistency

Every Management API mutation follows a revisioned outbox protocol:

1. validate the request and its `If-Match` revision;
2. in one PostgreSQL transaction, mutate desired state, increment the namespace and
   affected-resource revision, append an audit event, and append `policy_outbox`;
3. a projector claims outbox rows, compiles effective key policies or a complete
   routing snapshot, and atomically writes Valkey projections with
   compare-by-revision;
4. update the applied watermark only after all projection writes succeed; and
5. return the desired and applied revision in the mutation response.

Permission expansion, key creation, and quota expansion become usable only after the
new revision is applied. The API may wait for a bounded interval and return `200`
when applied, or return `202` with `state: pending` and an operation URL.

Restrictive mutations require a deny barrier:

1. commit a pending restrictive mutation and outbox record;
2. install the affected namespace, credential, key, User, Team, membership, grant,
   binding, or rule deny barrier in Valkey;
3. project the reduced policy;
4. activate and finalize the publication operation; and
5. remove the barrier only after the applied watermark reaches that revision.

The Management API does not report restrictive success until the barrier exists. If
Valkey is unavailable, the mutation remains pending or fails; it never claims global
enforcement. Reconciliation is idempotent and removes stale conservative barriers
only after proving the desired revision.

Projectors stage immutable per-key policy blobs and pending pointers. Each expansion
references a publication ID that remains inactive until every affected key is staged;
one shared publication gate then makes the complete expansion visible. Restrictions
install deny barriers before staging and retain them through activation. Active
pointers always select one complete old or new blob.

Outbox rows are ordered per aggregate and applied with revision compare-and-set.
Parallel workers may stage independent aggregates, but a contiguous namespace
watermark advances across a lower operation only when it is successfully applied,
superseded by a later fully applied revision, or explicitly rolled back in
PostgreSQL by a new applied revision. A merely `failed` operation blocks the
watermark and cannot release a deny barrier. Overlapping operations serialize
through their aggregate revision. Operation status, per-key applied revision,
publication state, and the contiguous watermark are distinct. PostgreSQL can rebuild
policy projections, but live quota state follows the separate recovery contract and
never becomes ready from a policy watermark alone.

Routing publication has its own contiguous watermark, replica acknowledgements, and
active snapshot pointer. An access expansion that grants a new or changed Entrypoint
waits for routing activation first. A routing restriction or deletion installs
resource deny barriers before the old snapshot can disappear. At activation, any
active replica that did not acknowledge the staged snapshot is forced out of
readiness; a new replica becomes ready only after loading the active snapshot. No
replica may observe an access grant whose referenced routing object is absent.

## Bootstrap, deployment, and recovery contract

The normative Router configuration, Docker-first stack, Kubernetes topology,
readiness, failure behavior, persistence guarantees, and Valkey catastrophic-loss
procedure live in the
[deployment appendix](./router-native-access-control-deployment). Standalone
deployments add no stores. Managed deployments use PostgreSQL and a shared highly
available single-writer Valkey; API-key access requires managed mode. Dashboard and
observability remain optional.

## Dashboard client and interaction contract

The Dashboard renders server-authorized capabilities; it does not infer authority
from a coarse `readonly` flag or hide Router data by account label. A principal with
the complete `routing.read` conjunction for an Entrypoint and its dependencies can
always open topology. Creating or editing an Entrypoint, choosing a Recipe, assigning
Models, configuring priority fallback, and publishing additionally require the exact
`routing.manage` expression returned by the Management contract. Cluster
administrators include both sets. Role-matrix tests cover cluster administrator,
namespace operator, Team administrator, and read-only User for every visible route,
button, direct URL, and API mutation—not only navigation.

Managed onboarding has one routing flow: connect Models, inspect or author a Recipe,
create an Entrypoint, then assign one or more configured Models to each decision.
Fallback controls stay collapsed until a decision has more than one eligible Model;
enabling them reveals simple priority tiers and failure behavior without exposing
gateway internals. Built-in Recipes use the same resources and endpoints as custom
ones.

The product shell uses one small semantic icon registry for navigation, primary
creation actions, row affordances, status, copy, edit, and destructive operations.
Icons have accessible names, come from the repository's product asset system, and
never encode permission or status without text. Emoji, provider-specific hard-coded
button art, and ornamental icons are excluded. Tables share aligned columns and one
action hierarchy; detail/modal actions reuse the same labels and icon meanings.
Cost appears in the main Usage view, API-key breakdown, and API-key detail beside its
live budget—not in a separate analytics implementation.

The Access Usage page composes the immutable Usage ledger with one scoped
`/management/v1/statistics` snapshot for control-plane cardinalities. It never
downloads every User, Team, key, Access Policy, or Rate Limit Policy to calculate
header cards. Counts remain exact decimal strings on the wire, and cards whose
read permission is absent are omitted rather than rendered as zero. Entity tables
continue to use their keyset-paginated list endpoints. Form selectors do not reuse a
visible table page as a directory cache: they issue bounded, debounced server
searches, advance opaque cursors on demand, and hydrate any previously selected
resource by its detail endpoint. This keeps ownership, Team membership, Access
Policy, and Budget editing complete beyond the first page without downloading the
full directory.

## Security properties

- The public gateway strips spoofable principal, Team, grant, quota, and revision
  headers before the Router creates trusted context.
- TenantContext is signed, short lived, request bound, and accepted only from the
  internal listener path.
- Inference Authorization, TenantContext, principal, Team, grant, and quota metadata
  are removed before backend dispatch; only a separate ProviderCredential is added.
- API-key HMAC pepper, reveal KEK, TenantContext signing key, provider credentials,
  and Management credentials have separate secret material and rotation lifecycles.
- Key lookup uses a public `kid`; authentication compares HMAC in constant time.
- Raw key reveal is optional, independently authorized, non-cacheable, and audited.
- Revocation and restrictive policy changes use deny barriers and fail closed.
- Management list and analytics queries are scope-filtered by the server, not only by
  Dashboard navigation.
- Access policy publication validates every Entrypoint, Model, decision ID, and quota
  partition before making a revision usable.
- Audit and telemetry redact Authorization, secrets, ciphertext, database URLs, and
  internal signed context.
- Public inference and private Management listeners have distinct exposure and
  authentication policies.

## Scale model

Ten thousand keys are a routine control-plane size:

- credential lookup is one indexed `kid` read from Valkey;
- compiled policies avoid request-time PostgreSQL joins;
- 2-4 KiB per compiled key is tens of MiB before datastore overhead;
- policy definitions are reusable while binding-owned counters prevent accidental
  sharing;
- list APIs use keyset pagination and bounded filters;
- Access directory search is server-side, scope-filtered before matching,
  cursor-bound, and covered with multi-page fixtures;
- control-plane summary cards use one indexed, permission-projected aggregate
  statement instead of walking entity pages;
- usage writes are streamed and batched, not synchronous per-request SQL writes;
- raw usage is partitioned and charts use verified rollups; and
- key or policy mutations never modify gateway configuration or restart the Router.

Request rate, not key count, sets the Valkey capacity requirement. Operators size
Valkey from peak admissions, settlements, rolling-window cardinality, and the maximum
PostgreSQL outage backlog. Exact sliding logs consume memory proportional to events
inside the window; very high-rate policies should use an explicitly selected O(1)
algorithm when exact event history is not required.

## Deliberate tradeoffs and risks

- Actual-only token accounting permits concurrent in-flight overshoot; only the
  combination of concurrency and per-request generation caps gives a calculable
  upper bound, never a prepaid ceiling.
- Exact rolling windows consume memory proportional to events in the active window.
- The first production contract requires a highly available single-writer Valkey;
  arbitrary cross-slot clustered admission is not claimed.
- Revealable credentials increase the consequence of KEK or privileged-account
  compromise and therefore remain separately configurable and authorized.
- A Team or User policy edit can fan out to thousands of compiled key projections;
  batch throughput, lag, and deny barriers are product metrics.
- Stream persistence settings define a real durability/latency tradeoff.
- A terminal streaming settlement failure cannot retract content already delivered;
  the unknown fence prevents that ambiguity from spreading to later requests.
- Live quota and asynchronous analytics intentionally have different freshness; APIs
  expose `asOf`, applied revision, rollup freshness, and ingestion lag.
- Request logs may be sampled and retained briefly, but usage facts cannot be sampled.
- Per-key metric labels are forbidden; high-cardinality detail belongs in bounded
  Management queries over usage and audit stores.

## Schema lifecycle and replacement boundary

The runtime has one configuration, API, and persistence contract. It contains no
dual reader, dual writer, hidden Dashboard authority, mounted routing mutation path,
or historical configuration translator. The v0.4 runtime accepts only a
`version: v0.4` manifest. `vllm-sr config migrate` is a separate offline tool that
accepts exactly v0.3, produces and validates a new v0.4 file, and is never imported
by the runtime. There is no dual-read deployment mode.

### Contract versioning and upgrade

The manifest version and Management HTTP API version evolve independently. A
manifest upgrade changes the compiled Router contract; it does not silently select
an HTTP API version. Management clients explicitly select `/management/v1` and
`application/vnd.vllm-semantic-router.management.v1+json` in `Accept` and
`Content-Type`, and bind to the matching published OpenAPI document. Within v1,
responses may gain fields and requests may gain only optional fields with documented
defaults. Removing, renaming, reinterpreting, or changing the default of a field
requires `/management/v2` and a new media type. Clients never silently fall back
between versions.

Standalone v0.3 manifests are converted offline before rollout. Managed upgrades
also run the release's forward-only PostgreSQL migration before new replicas become
ready. The operational sequence, validation boundary, and rollback requirements are
defined in [Upgrade and rollback](../installation/upgrade-rollback.md) and the
[deployment contract](./router-native-access-control-deployment.md).

PostgreSQL uses ordinary forward-only schema migrations. A managed Router verifies
the schema before readiness and never performs destructive automatic conversion.
Valkey state is a rebuildable projection of PostgreSQL desired state and durable
usage evidence; a new runtime epoch is published only after policy and routing
snapshots validate together.

The Dashboard uses only Management and inference APIs. Public discovery, Playground,
direct inference, topology, access policy, quota, usage, and audit all exercise the
same Router authority. `vllm-sr serve --config` may select one immutable standalone
manifest; it never selects a Recipe, authors a Model, or creates a second active
routing pointer. Built-in Recipes are immutable Namespace resources installed by the
Router and customized by duplication.

## Validation matrix

| Area | Required validation |
| --- | --- |
| Credential lifecycle | Create, reveal permission, overlap rotation, expiry, disable, enable, renew, delete, concurrent revoke, and secret redaction. |
| Ownership | User-owned, Team-owned, context Team, membership removal, disabled owner, and one-of validation. |
| Model policy | Authenticated discovery, unauthenticated denial, Entrypoint invoke, forbidden/nonexistent nondisclosure, direct Model pinning, and candidate-model escape prevention. |
| Provider integration | Strict Integration Registry construction, deterministic composition, duplicate/unknown capability failure, fixed/user-supplied origin, secret-field rejection, catalog rolling activation, stale discovery revision, credential binding, compatible Provider added without Dashboard/data-plane code, and new wire-codec rollout before Model publication. |
| Protocol matrix | Every buffered and streaming source/target codec pair; text, multilingual, images, reasoning, tools and ordering, structured output, stop semantics, authoritative usage/cache buckets, typed errors, bounded preservation, lossy rejection, malformed frames, finalization, cancellation, and proof that no pair-specific translator is reachable. |
| Inheritance | Key override, User override, Team inheritance, override removal, shared counter ownership, and cumulative hard cap. |
| RPM | More than 12 requests in an exact rolling minute, boundary timestamps, concurrent admission, and idempotent retry. |
| Tokens | Actual input/output/total usage, crossing request allowed, next request denied, reset, overshoot bound only with concurrency plus generation caps, and unknown-usage reconciliation. |
| Cost | Eight-hour sliding and calendar budgets, crossing request then next-request denial, exact decimal arithmetic, API-key breakdown/detail parity, live remaining/reset, inherited bindings, unpriced/incomplete/fenced state, and multi-currency separation. |
| Execution shapes | Managed/standalone Model digest parity; defaults/bounds and Dashboard Advanced round-trip; only proven-pre-inference retry, no retry after a visible byte, total request/stream timeout; priority fallback selection, same-Model retry exhaustion, unavailable/overloaded/proven-zero timeout, deadline preservation, no fallback after output or unknown usage; four exclusive billing buckets, cache inheritance, explicit zero/unpriced state, pinned historical price revision; and non-streaming, streaming, disconnect, fusion, workflow, and looper accounting. |
| Consistency | Staged expansion gate, restrictive deny barrier, routing snapshot acknowledgements, access/routing dependency order, contiguous watermarks, failed-operation blocking, overlapping mutations, lost projector, duplicate outbox delivery, policy-only rebuild, and stale revision conflict. |
| Replicas | Identical result from every Router replica with no sticky session and no local cache dependence. |
| Usage | Counter/ledger agreement, duplicate stream delivery, PostgreSQL outage backlog, rollup reconciliation, retention, and cursor pagination. |
| Outcome feedback | Public-listener authentication, delegated-session use, exact logical-key replay ownership, served-Model match, bounded payload, cross-replica duplicate submission, failed-claim retry, global abuse limit, and learning projection rebuild. |
| Docker | Standalone manifest with no stores, managed routing with access off, managed access, embedded/external stores, migration ordering, secret files, restart persistence, and optional Dashboard absence. |
| Kubernetes | Standalone manifest, managed HPA scale, routing revision rollout, Pod loss, migration Job, NetworkPolicy, Management isolation, store failover, projector contention, and PDB behavior. |
| Management RBAC | Every permission and subject scope at API level, including self-service and forbidden cross-Team queries. |
| Dashboard capability UX | Topology for every fully authorized reader; Entrypoint/assignment/publish mutation only for exact manage authority; direct-URL/API denial; aligned accessible icons/actions; and no coarse account-label heuristic. |
| CLI surface | `vllm-sr serve`, optional immutable-bootstrap selection through `--config`, infrastructure options, and proof that Model/Recipe operands, model command group, catalog materialization, and launch-time algorithm override are absent. |
| Management identity | OIDC and local-issuer exchange, nonce replay, audience, expiry, principal linking, broker actor chain, invitation onboarding, session disable, and service accounts. |
| Delegated inference | Playground session creation/revocation, current-policy resolution, direct Model grant, counter sharing, usage attribution, and invalidation after key/User/Team/session disable. |
| Agent and Builder | Dynamic catalog revision, Skill loading, Tool permission composition, durable events, idempotent turns, lease fencing, reconnect/resume, cancellation, context checkpoints, ETag conflicts, probe/eval artifacts, immutable approval, publication rollout, discovery, and direct invocation of the published Entrypoint. |
| Schema lifecycle | Fresh-schema and forward-only upgrade coverage; operator import receipts, explicit resets, policy equivalence, credential verification, quota state, usage totals, rollback backup, and proof that no duplicate Dashboard authority exists. |

Performance gates use realistic policy cardinality: 10,000 keys, independent compiled
policies, hundreds of Users, multiple Team-shared counters, exact and O(1) rules, and
concurrent Router replicas. The benchmark reports p50/p95/p99 admission latency,
Valkey operations, memory per key and rolling event, projector throughput, usage lag,
and behavior during store failover.

## Acceptance criteria

- Every public inference endpoint rejects missing credentials when access is enabled.
- Dashboard removal or outage has no effect on inference authentication, discovery,
  invocation, quota, or accounting.
- `/v1/models` and invocation demonstrably use the same policy evaluator.
- A restrictive mutation is globally enforced before the API reports success.
- Exact RPM, actual-token, and actual-cost receipts match the live quota endpoint under
  concurrency and across replicas.
- API-key Usage and detail show the same actual cost, while live cost quota reports
  exact independent remaining and reset state for arbitrary bounded windows.
- Priority fallback advances only through the configured tiers on Router-proven safe
  evidence, preserves the original deadline, and leaves a complete dispatch trail.
- Streaming and every internal Mixture-of-Models dispatch are accounted once without
  estimates or silent zero usage.
- Outcome feedback is accepted only for the caller's durable replay and served Model;
  the same logical submission is recorded once across replicas and survives restart.
- A custom console can implement the complete product lifecycle using published
  Management OpenAPI only.
- An ordinary compatible Provider can be registered in the control-plane application
  and appears in every Management client without a Dashboard or inference-runtime
  product change.
- A Dashboard cookie has no Router authority until a valid identity exchange, and
  Playground uses a short-lived delegated credential against the public inference
  listener with no proxy, shared key, or quota exception.
- Standalone `vllm-sr serve` adds no PostgreSQL or Valkey dependency; managed mode
  always declares both stores and access-enabled mode cannot start without them.
- `vllm-sr serve` starts the selected deployment topology. The optional `--config`
  flag selects one immutable bootstrap manifest; routing resources are authored
  through the manifest in standalone mode or the Management API in managed mode.
- The Dashboard reads built-in Recipes through the same Router Management API as an
  independent console; it never shells out to, mirrors, or exports a CLI model
  catalog.
- Router and schema-migration images carry only canonical built-in Recipe assets.
  The distribution contains no Models, virtual Models, Model pools, or recommended
  assignments. Dashboard images, ConfigMaps, CRDs, and Helm values carry no copy;
  immutable provenance makes source version, asset digest, source Recipe revision,
  and projected Recipe digest observable through ordinary Recipe reads and audit.
- Access-enabled Docker and Kubernetes use one semantic implementation.
- No API key, User, policy, usage event, or audit event is represented in Router YAML,
  gateway routes, xDS, ConfigMaps, or per-resource Kubernetes custom resources.
- Dashboard inference uses the Router public listener, while Dashboard management uses
  generated Management API clients.
- Router AccessRuntime exclusively owns inference identity, authorization, and global
  quota enforcement; Dashboard packages remain stateless clients of that authority.
- Every installed wire format passes the full buffered and streaming codec matrix;
  usage settlement consumes the same neutral response record before client encoding.
- A Builder session can survive reconnect and worker replacement, use only authorized
  dynamic Router tools, wait for an explicit immutable approval, publish a complete
  Recipe and Entrypoint, and invoke it through ordinary discovery and inference.

## Related documentation

- [API server](../api/apiserver) documents the Management listener surface.
- [Security hardening](../installation/security-hardening) defines baseline secret,
  listener, and production deployment practices.
- [Docker installation](../installation/docker) and
  [Kubernetes operator installation](../installation/k8s/operator) provide the
  deployment entry points governed by this proposal.

## Resolved operational defaults

- Raw usage, request-log, and audit records are retained indefinitely by default.
  Request and response payload capture is disabled by default and, when enabled, has
  its own explicit operator retention. Partition removal is opt-in and must pass the
  replay, reconciliation, and aggregate-safety gates defined above.
- Exact `sliding_log` enforcement is the default. There is no universal traffic-rate
  threshold for changing semantics: an operator selects `token_bucket` only after a
  representative capacity test demonstrates that the exact policy cannot meet the
  deployment's latency or throughput objective.
- Revealable credentials are disabled until an operator explicitly configures the
  reveal key-encryption key and enables the capability. Authentication remains based
  on the stored HMAC, so disabling reveal never weakens or interrupts ordinary keys.
- `max_usage_backlog` defaults to 1,000,000 stream entries. Reaching the bound fails
  admission closed instead of losing accounting evidence. Operators size the bound
  from their measured request rate, maximum supported PostgreSQL outage, and Valkey
  memory budget, then validate that choice in the deployment capacity gate.
