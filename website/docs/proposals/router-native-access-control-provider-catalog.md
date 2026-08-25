---
title: Router-Native Provider Catalog Appendix
description: Defines the control-plane Provider Integration registry, backend compilation, immutable catalog publication, and runtime wire-format contract.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for provider integration in
[Router-Native Access Control and Quota Accounting](./router-native-access-control).
The [Management API](./router-native-access-control-management-api) owns catalog
reads and discovery. The [Model runtime](./router-native-access-control-model-runtime)
owns compiled backend execution. The
[resource contract](./router-native-access-control-contracts) owns persistence and
ProviderCredential lifecycle.

## Decision

A Provider is a control-plane integration, not a product branch in the inference
runtime and not user-authored Router configuration.

The architecture has five separate concepts:

| Concept | Plane | Responsibility |
| --- | --- | --- |
| Provider Integration | Control | Application-installed extension that contributes exactly one Provider Definition to the registry. |
| Provider Definition | Control | Stable identity, display metadata, origin rule, typed form fields, discovery reference, and one or more typed inference interfaces. |
| Backend compiler | Control | Strictly validate Provider inputs and compile one complete, canonical, non-secret backend connection. |
| Wire codec | Data | Implement one stable inference format, neutral request/response/event translation, typed errors, and authoritative usage extraction. |
| Credential adapter | Data | Materialize one encrypted ProviderCredential version into bounded authentication state such as Bearer or `x-api-key`. |

`provider_id` is attribution and an immutable credential-binding value. It never
selects a product switch. An Integration interface selects a stable `wire_format`
and a control-plane compiler; the compiled backend contains exactly one wire format.
`credential.adapter_id` selects data-plane secret materialization.

An ordinary Provider that uses installed compiler, discovery, wire format, and
credential capabilities is added by registering a control-plane Integration. A Provider
with unusual configuration semantics adds one narrow backend-compiler plugin to the
control-plane application. A genuinely new inference wire or authentication
protocol requires one neutral codec, deployed before an Integration may
reference it.

There is no external catalog manifest, per-provider bootstrap field, per-provider
ConfigMap, Provider CRD, or Dashboard-owned catalog. Application composition is the
single extension boundary.

## Integration contract

An Integration returns one immutable Definition. The registry owns its revision and
rejects a caller-supplied revision, duplicate Provider or interface ID, unknown
compiler, discovery adapter, credential adapter, or wire format, invalid
origin, unsafe path or header, and non-canonical field schema.

Conceptually, an Integration contributes the following typed value. This schematic is
an application-extension contract, not Router YAML, Recipe DSL, a ConfigMap, or a
Management payload. End users never author these adapter and compiler fields:

```yaml
id: example
order: 100
display:
  name: Example
  description: Connect Example models.
  category: Model APIs
  icon:
    source: lobe
    value: example
    color: true
  monogram: E
  accent: "#5b8cff"
interfaces:
  - id: chat
    label: Chat Completions
    default: true
    wire_format: openai.chat.v1
    compiler:
      adapter_id: static.backend.v1
      config:
        path: /chat/completions
        headers:
          X-Example-Version: "1"
    capabilities: [image_input, streaming, tools]
  - id: responses
    label: Responses API
    wire_format: openai.responses.v1
    compiler:
      adapter_id: static.backend.v1
      config:
        path: /responses
    capabilities: [file_input, image_input, reasoning, streaming, tools]
credential:
  mode: required
  adapter_id: bearer
  label: API key
origin:
  mode: fixed
  default_url: https://api.example.com/v1
discovery:
  adapter_id: openai.models.v1
  path: /models
  headers:
    X-Example-Version: "1"
capabilities: [image_input, reasoning, streaming, tools]
```

`display.icon` is a validated presentation descriptor owned by the Integration,
not a Provider lookup table in the Dashboard. `source` is `lobe`, `asset`, or
`url`; asset paths are confined to the application origin and remote icons require
an absolute credential-free HTTPS URL. The Dashboard applies one pinned renderer
and falls back to the monogram if an icon cannot load.

The Definition has no secret-valued connection field. `credential` describes one
secret prompt and creates or references a ProviderCredential. Connection fields are
typed non-secret compiler inputs. They are consumed at the control-plane compiler
boundary and never become an open-ended map in the Router snapshot. Authorization,
cookie, proxy, host, content-length, transfer-encoding, and other credential or
transport-framing headers are forbidden in Definition compiler and discovery
configuration.

`static.backend.v1` accepts a literal canonical path and bounded non-secret headers.
It accepts no Provider-specific form fields and covers normal fixed-origin and
user-supplied-origin integrations. A Provider that needs a typed field registers a
compiler that declares how the field becomes a canonical connection:

```yaml
compiler:
  adapter_id: regional.backend.v1
  config:
    path_template: /regions/{region}/chat/completions
connection_fields:
  - name: region
    label: Region
    kind: select
    required: true
    advanced: true
    options:
      - {value: global, label: Global}
      - {value: eu, label: Europe}
```

Compiler-specific configuration is opaque only to the Integration envelope. The
selected plugin strictly decodes and validates it during registry construction. The
compiler returns typed `path` and `headers` values or fails closed. The common
control plane independently normalizes the origin, applies egress policy, and
validates compiler output before attaching the credential reference. A compiler
cannot return credentials, choose a credential, open a socket, mutate desired state,
or publish directly to the data plane.

### Origin and credential modes

- `origin.mode: fixed` requires one canonical public base URL and renders no URL
  input. This is the normal **API key only** experience.
- `origin.mode: user_supplied` renders a required base URL. The Management API
  canonicalizes it and applies backend-egress policy before any network access.
- `credential.mode: none` forbids a ProviderCredential.
- `credential.mode: optional` permits a no-auth backend or one ProviderCredential.
- `credential.mode: required` requires one ProviderCredential before probe, import,
  or Model publication.

The Definition's `credential.adapter_id` is copied into immutable credential
metadata. A later Integration revision cannot reinterpret an encrypted secret under
another authentication scheme.

## Registry composition and publication

The control-plane application explicitly constructs one immutable Integration
Registry from:

- built-in and application-provided Provider Integrations;
- installed backend compilers and discovery adapters; and
- the stable wire formats and credential adapter capability IDs supported by the
  data plane.

The executable application constructor is the only place that selects the shipped
Integration set. It injects typed `Integration` and `BackendCompiler` values into
process composition. A runtime with a Management store derives installed wire formats,
credential, and discovery capability IDs from its immutable registries, then
constructs and validates the catalog before opening desired-state stores or serving
Management traffic. Missing Integrations, compilers, or referenced adapters fail
startup. The runtime contains no fallback Provider list and never loads a
Provider product manifest. A different application can therefore compose another
Integration set without editing Router orchestration or inference code.

The registry evaluates every Integration exactly once, canonicalizes Definitions,
computes one SHA-256 revision for each Definition, sorts them deterministically, and
computes a content-addressed catalog revision. Unknown capabilities and duplicate or
invalid identities fail process composition. No request, tenant, file watcher, or
Dashboard action can mutate this registry.

When a Management store is configured, the catalog coordinator:

1. validates the application registry and its plane-specific capability digests;
2. stores the immutable catalog value in `provider_catalog_revisions` in PostgreSQL;
3. stages that revision while control-plane rollout groups acknowledge compiler and
   discovery compatibility and data-plane rollout groups acknowledge wire-format and
   credential compatibility; and
4. advances the singleton active catalog pointer only after the declared gate passes.

On a genuinely empty durable store, Router startup converges the unique
application-installed catalog without a separate bootstrap client. A replica uses the
singleton generation as a compare-and-swap token, stages only that installed revision,
ACKs only its declared rollout-group memberships, and activates only after the complete
declared gate is compatible. Concurrent replicas reread stage or activation conflicts;
a replica blocked on another group's missing ACK leaves its own ACK durable so the peer
can finish the same rollout. An already desired or active different revision is never
replaced by this cold-start path. Restarts only reconcile the established revision, and
all later catalog lifecycle changes remain explicit Management operations.

`provider_catalog_state` stores desired and active revisions.
`provider_catalog_replica_acks` stores bounded renewable compatibility leases with a
plane and rollout-group identity. Activation never infers a required set from Pods
that happened to report. Deployments declare stable rollout groups, so autoscaling
does not turn ephemeral replica names into catalog configuration. Mixed application
versions block activation until the group is homogeneous or the previous lease
expires.

Historical catalog revisions remain long enough to validate pagination cursors,
discovery claims, audit, and in-flight Operations. This state is small and changes
only when the control-plane application changes; it is not projected to Valkey for
inference.

Every Management list or detail response includes `catalogRevision`. Keyset cursors
and signed discovery revisions bind the same revision. A request sent to another
Management replica therefore retains exact semantics during a rolling deployment.
An incompatible replica is unready instead of serving another meaning for the same
revision.

## Model compilation

Model create, bulk import, and update submit `providerId`, a Provider model ID,
optional ProviderCredential ID, and schema-approved non-secret connection values.
The control plane resolves them against one active catalog revision and compiles:

```yaml
provider_id: example
wire_format: openai.chat.v1
origin: https://api.example.com/v1
provider_model_id: example-3
provider_credential_id: 2f2a80c4-8fc4-4ce1-86c9-0dbfeff73f1d
connection:
  path: /chat/completions
  headers:
    X-Example-Version: "1"
weight: "1"
```

The backend compiler receives its strictly validated Definition configuration and
schema-approved non-secret connection values. A compiler-independent validator
checks its result, then the common control plane attaches the normalized origin and
validated ProviderCredential reference. No form field, display value, compiler ID,
discovery configuration, or default-URL rule is published to the Router.

The immutable Model revision records the catalog revision used to compile it.
Routing publication verifies every Router rollout group supports each named wire
format and credential adapter. The routing snapshot contains only compiled
provider-neutral backend values, one stable wire format per backend, and the
credential adapter ID.

At dispatch, BackendInvoker resolves the compiled wire format in the immutable Codec
Registry. When a credential is
present, it verifies provider and origin equality, pins one credential version,
loads that credential's adapter by ID, decrypts only in process, materializes bounded
authentication state, and erases plaintext. The Provider name is never an execution
branch.

## Discovery and Dashboard

The Dashboard calls `GET /management/v1/providers` and renders the returned safe
catalog view. It owns reusable visual components, not product definitions, logos,
authentication headers, origins, or provider-specific forms. A newly registered
compatible Integration therefore appears in Add Model without a Dashboard release.
Any CLI, automation, or independent console receives the same contract.

`POST /management/v1/providers/{providerId}:discover-models` invokes the Definition's
control-plane discovery adapter. Discovery is a separate registry because listing
models is not inference dispatch. An adapter receives only validated origin,
non-secret connection values, and a pinned credential handle. Every outbound target
passes egress policy before and after DNS resolution; secret-bearing redirects are
disabled. Results are bounded and normalized into one catalog-item schema.

The signed discovery revision binds namespace, actor authority, catalog revision,
Provider ID and Definition revision, normalized origin and connection digest,
credential and version, returned item IDs, and expiry. Bulk import accepts selected
item IDs and safe Model overrides; request fields cannot replace origin, adapters,
or credential.

Model verification uses the same compiled backend, wire codec,
credential resolver, and egress transport as inference. The control plane sends one
bounded, non-streaming, one-token neutral request and lets the selected codec produce
the provider wire shape. OpenAI-compatible and Anthropic-style
backends therefore share one probe path without Provider product branches. The
probe honors the Model's saved retry policy and the operation timeout, reports only
availability, latency, and check time, closes every response body, and never returns
provider credentials or upstream response content.

## Versioned evolution

Integration changes use one of four paths:

1. **Compatible Provider:** register or update one control-plane Integration that
   references installed capabilities. No data-plane or Dashboard code changes.
2. **New configuration shape:** add one backend-compiler plugin to the control-plane
   application, then register the Integration that references it.
3. **New discovery shape:** add one control-plane discovery plugin, then register the
   Integration that references it.
4. **New inference or authentication protocol:** deploy the neutral codec or
   credential adapter to every Router rollout group, observe capability readiness,
   then release the Integration and publish Models that use it.

Changing a fixed origin, credential adapter, selected interface or wire format, or compiler input
produces a new Provider Definition revision and new Model revision. Existing
ProviderCredentials remain bound to their stored provider, origin, and credential
adapter. Existing published Models keep their compiled backend until explicitly
revised. Removing an Integration blocks new creates and imports but does not silently
reinterpret an already published snapshot; credential rotation for that pinned
binding remains possible. Runtime dispatch continues through the pinned neutral codec,
credential adapter, and compiled backend contract.

## File-backed and persistent deployments

With a Management store, Router builds the catalog from the application Integration Registry, persists
the content-addressed snapshot, and exposes Provider Management APIs. Docker and
Kubernetes use the same application composition and revision-acknowledgement
protocol. Provider count is independent of the number of namespaces, Users, or API
keys; no provider resource is mounted into either environment.

File-backed authoring uses the same Integration Registry and backend compilers in the
CLI or compiler process. Its final local routing snapshot already contains one wire
format, canonical origin, compiled non-secret connection, and a bootstrap
credential reference. Provider catalog APIs, discovery Operations, PostgreSQL
catalog tables, and dynamic activation are not started.

Both deployment shapes derive catalog provenance identically from canonical Integration
Definitions and registry-owned revisions. The inference runtime never reopens
provider configuration or selects behavior by Provider product.

## Security invariants

- Integrations and Definitions contain no API keys, tokens, passwords, cookies,
  encrypted blobs, or references to inference API keys.
- Definitions and compilers cannot emit authorization, credential, proxy, cookie,
  host, content-length, transfer-encoding, or other forbidden headers.
- ProviderCredential provider, credential adapter, catalog revision, and canonical
  origin are immutable and included in encryption additional authenticated data.
- Product metadata never reaches inference authorization or routing decisions.
- Unknown compiler, protocol, credential, or discovery adapter IDs fail registry
  composition; unknown compiled runtime adapters fail Router readiness.
- Catalog read permission exposes integration metadata only. Credential metadata,
  credential use, and network discovery require independent permissions.
