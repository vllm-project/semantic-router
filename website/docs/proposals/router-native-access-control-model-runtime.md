---
title: Router-Native Model Runtime Appendix
description: Specifies Model execution, retry, timeout, pricing, and cost-accounting contracts for Router-native managed routing.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for Model execution and pricing in
[Router-Native Access Control and Quota Accounting](./router-native-access-control).
The [resource contract](./router-native-access-control-contracts) owns persistence and
usage records; the [Management API](./router-native-access-control-management-api)
owns Model CRUD and Dashboard-facing fields.

## One readable Model, one compiled revision

Users configure one Model value. The semantic `card` is deliberately separate from
`connections`, so Recipe and DSL authors can reason about capability without seeing
endpoint or credential details. Runtime and pricing are optional advanced settings:

```yaml
name: local/primary
card:
  description: General model with tool support
  capabilities: [chat, tools]
  modality: text
connections:
  - provider: vllm
    interface: chat
    endpoint: http://model:8000/v1
    model: primary
runtime:
  max_retries: 2
  request_timeout: 5m
  stream_timeout: 15m
pricing:
  input_cost_per_million_tokens: "0.50"
  output_cost_per_million_tokens: "1.50"
  cache_read_cost_per_million_tokens: "0.05"
  cache_write_cost_per_million_tokens: "0.625"
```

Managed CRUD/import/export and standalone authoring use this same shape. Empty and
default-only fields are omitted. A fixed-origin Integration lets `endpoint` disappear;
a no-auth connection omits `credential`. The authoring value never contains generated
IDs, revisions, catalog hashes, compiled backend data, or secret material.
`interface` is the optional readable Provider API style; omission selects the
Provider's single declared default.

Publication resolves readable names, validates the Provider Integration, creates
immutable identity and revision state, and compiles each connection into a
provider-neutral backend. That internal revision pins catalog provenance, protocol
adapter, canonical origin, provider model, non-secret wire settings, credential
reference, and weight. Dispatch is selected exclusively by the stable protocol
adapter. Provider products, logos, form fields, default origins, and discovery
behavior remain in the control-plane
[Provider catalog](./router-native-access-control-provider-catalog) and are absent
from the inference data plane. Managed mode takes currency from Namespace;
standalone requires `billing_currency` when any Model is priced.

## Retry and timeout semantics

`max_retries` is additional attempts after the first and is bounded from zero to
five. It applies only to Router's fixed safe-retry predicate: protocol/transport must
prove that inference and billable processing never began, although bytes may have
reached the peer. Such an attempt is `known_zero`; failure classes are not user
configuration. `request_timeout` is the total
non-streaming invocation deadline including retries. `stream_timeout` is the total
streaming lifetime, including retries before the first client-visible byte. Durations
are 1 second through 24 hours. Defaults are zero retries and 300 seconds for each
timeout; every read returns configured and effective values.

The public gateway strips caller-supplied transport-control, destination, identity,
and credential headers. Envoy forwards every post-ExtProc inference request through
one stable internal Router backend-invoker cluster; it does not own per-Model routes,
credentials, timeouts, or retries. The BackendInvoker pins the Model and backend
revision, verifies that its compiled wire format is installed in the Codec Registry, journals the
bounded dispatch and attempt plan with the configured durability acknowledgement,
resolves the ProviderCredential only in process through its credential adapter, and
then performs the upstream call. Primary, Looper, fusion, workflow, multimodal, and
future adapter dispatches all use this same interface rather than calling an endpoint
directly.

Managed and standalone startup inject the same mandatory dispatch-capability runtime
into each immutable Router generation. The public edge decodes the selected client
format into the neutral request IR; ExtProc routes and applies plugins to semantic
views of that IR, then hands BackendInvoker the neutral request, client format, pinned
logical Model revision, and a body-bound capability. It never resolves a physical
host, provider model ID, provider credential, product dialect, retry, or fallback.
BackendInvoker resolves the pinned connection and credential, encodes the request
with that connection's installed codec, decodes the upstream response or stream back
to the neutral IR, and returns terminal evidence for the shared usage finalizer before
the client codec renders the response. Internal Looper calls use the same typed
dispatch plan and authorization boundary. Missing generation, capability, codec, or
immutable Model identity fails closed instead of activating an unmanaged direct call.

The terminal hand-off follows the deployment topology. Standalone mode uses one
bounded, expiring process-local store. Managed mode writes the bounded neutral
terminal to Valkey with `SET NX`, keyed by a digest of the immutable namespace,
publication, admission, request, dispatch, and Model identities. The owning ExtProc
replica consumes it exactly once with one atomic get-and-delete operation. This
short-lived rendezvous stores no response body, credential, provider error cause, or
source envelope; PostgreSQL usage events remain the durable accounting record. A
missing, duplicated, malformed, expired, or unavailable terminal fails closed as
unknown usage. Replica placement therefore cannot change settlement behavior, and
the public gateway requires no sticky-session policy.

Modality classification is routing evidence only. Image-bearing and other
multimodal requests keep that evidence through decision evaluation, then invoke the
selected logical Model through the same BackendInvoker boundary. ExtProc has no
separate omni, autoregressive, diffusion, endpoint-scan, or composite HTTP client.
An unsupported composite therefore fails at validation or adapter resolution rather
than falling back to a hidden direct call.

Internal retrieval dependencies are a separate boundary. A RAG vector/search
adapter may retrieve documents, and the optional memory-rewrite system service may
produce a private search query. Neither is addressable as a public Model, participates
in Entrypoint assignment, chooses a physical inference backend, or inherits Model
retry/fallback semantics. Classifier and embedding providers have the same internal
status. These narrow adapters must remain explicitly inventoried and cannot be used
to add a second request-facing Provider runtime.

Because the Router owns each attempt, it can prove the fixed safe-retry predicate and
record `known_zero`, `known_actual`, or `unknown` evidence before another attempt.
The final inference-capable attempt is one UsageDispatch. A failure not provably
pre-inference, any timeout after send, or any response after client-visible bytes is
never retried and becomes unknown usage when authoritative usage is absent. The
invoker returns a typed terminal record to the request finalizer; no caller or Envoy
header can claim attempt evidence.

External RPM is charged once. Transport-capacity proofs multiply a Model call by
`max_retries + 1`; billable-dispatch bounds do not because every earlier retry is
proven pre-inference. Snapshot validation derives a finite whole-admission deadline
from Recipe control flow, Model timeouts/retries, bounded loops, and parallel critical
paths. The dispatch journal pins that deadline and heartbeats cannot extend it.

## Priority fallback between Models

Model `max_retries` repeats the same immutable Model revision. Cross-Model fallback
is instead an optional value on one Entrypoint decision assignment:

```yaml
assignments:
  Complex:
    models:
      - {model: remote/primary, priority: 0, weight: "1"}
      - {model: local/secondary, priority: 1, weight: "1"}
    fallback:
      strategy: priority
      on: [unavailable, overloaded]
```

Lower numeric priority wins. The Recipe algorithm chooses only among eligible Models
in the active tier, using weights within that tier. A fallback-enabled decision must
have single-dispatch cardinality; this prevents a required fusion or workflow cohort
from being mistaken for a backup list. Tiers are contiguous from zero, bounded to 32,
and publication validates that every tier satisfies the decision's modality,
reasoning, tool, context, and protocol requirements.

The invoker exhausts the selected Model's safe same-Model retries before advancing.
`unavailable`, `overloaded`, and `timeout` are closed Router evidence classes, not
user-supplied status-code lists. A transition is allowed only before visible output
and when the adapter proves `known_zero`; an ambiguous timeout, partial stream,
unknown usage, or known billable work is terminal. Skipping a tier because every
Model is already unavailable does not create an attempt. Each real attempt and
transition is journaled and contributes to latency and diagnostics.

Envoy continues to provide connection management and endpoint health inside one
physical backend. It does not use an aggregate cluster, retry priority plugin, or
route retry to move between logical Models. Router owns that boundary so logical
identity, ProviderCredential selection, timeout budget, authorization, usage, and
cost remain deterministic. The whole request/stream deadline is shared across all
same-Model retries and fallback tiers; fallback never resets it.

Transport-cluster priority and retry load operate on physical upstream health. They
do not carry the Router's logical Model revision, pricing evidence, or admission
dispatch identity, so using them for Model fallback would create an unaccounted
second routing authority.

## Pricing and actual cost

Prices are plain non-negative decimal strings in the namespace's immutable ISO-4217
currency, with at most nine fractional digits and a maximum of 1,000,000 currency
units per million tokens. Exponent, NaN, infinity, overflow, and silent rounding are
rejected. Explicit `"0"` means free; null input/output means unpriced. Blank cache-read
or cache-write prices inherit input. All physical backends of one logical Model share
one price contract; a differently priced endpoint is a different Model.

Usage separates uncached input, cache-read input, cache-write input, and output
tokens. Rates compile to integer nano-currency units per million tokens. The shared
checked six-limb QuotaInteger accumulates `sum(tokens * pinned_rate)` across
dispatches. That integer is the exact cost numerator at a fixed `10^-15` currency
scale. A decimal limit compiles by exact multiplication by `10^15`; quota compares
numerator to numerator. Public decimals use the exact inverse scale with trailing-zero
normalization and no rounding, so many tiny requests cannot disappear. A differential cache price without
authoritative cache buckets, or nonzero tokens without a required rate, makes cost
unknown rather than zero. Historical events retain their Model/price revision and are
never recomputed after a price edit.

Valkey represents each unsigned cost numerator as a fixed six-limb base-10,000,000
integer. The settlement Function performs validated carry/add and high-to-low compare
without converting the value to a Lua number; every limb intermediate remains below
the exact-integer limit. Schema/publication validation rejects a Model, token bound,
or cost limit whose proven maximum exceeds this domain. PostgreSQL stores the same
canonical numerator as `numeric(42,0)`.

`cost` is a response-actual quota metric with the same crossing-request,
settlement, and unknown-fence semantics as token metrics. Publication rejects an
enforced cost rule whose reachable Models are unpriced or whose adapters cannot prove
the billing buckets required by differential cache prices; such a rule may be
shadow-only.

## Deliberate simplicity

There is no PriceBook, RetryPolicy, TimeoutPolicy, configurable retry-failure list,
backend-price override, or second transport API. Users edit one Model. The Dashboard
keeps these fields inside **Advanced settings**; the Router compiler, BackendInvoker,
immutable revisions, usage arithmetic, and consistency gates remain implementation details.
