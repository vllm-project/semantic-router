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

## One Model value, two embedded settings

A Model revision embeds two value objects; neither is a separately managed policy:

```yaml
execution:
  max_retries: 2
  request_timeout: 300s
  stream_timeout: 900s
pricing:
  input_cost_per_million_tokens: "0.50"
  output_cost_per_million_tokens: "1.50"
  cache_read_cost_per_million_tokens: "0.05"
  cache_write_cost_per_million_tokens: "0.625"
```

Managed CRUD/import/export and the standalone routing manifest use this same value
schema. A published immutable routing snapshot pins the complete Model revision.
Managed mode takes currency from Namespace. Standalone requires top-level
`billing_currency` whenever any price or cost-aware algorithm is configured; otherwise
its Models are unpriced. Both compile the same single-currency snapshot shape.

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

The public gateway strips caller-supplied transport-control headers. For the primary
upstream, ExtProc emits trusted Model-revision controls and Envoy owns only the safe
transport retries. One quorum-acknowledged dispatch intent records the bounded attempt
plan before forwarding, and Envoy attempt telemetry reconciles it. The final
inference-capable attempt is one UsageDispatch. A failure not provably pre-inference, any
timeout after send, or any response after client-visible bytes is never retried and
becomes unknown usage when authoritative usage is absent. Router-owned secondary
dispatches use the same rule.

External RPM is charged once. Transport-capacity proofs multiply a Model call by
`max_retries + 1`; billable-dispatch bounds do not because every earlier retry is
proven pre-inference. Snapshot validation derives a finite whole-admission deadline
from Recipe control flow, Model timeouts/retries, bounded loops, and parallel critical
paths. The dispatch journal pins that deadline and heartbeats cannot extend it.

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
keeps these fields inside **Advanced settings**; the Router compiler, Envoy controls,
immutable revisions, usage arithmetic, and consistency gates remain implementation details.
