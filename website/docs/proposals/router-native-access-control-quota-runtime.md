---
title: Global Quota Execution Appendix
description: Specifies rate-limit counter identity, Valkey projections, exact arithmetic, pending work, and reconciliation.
created: 2026-08-22
status: Proposal
---

> **Status:** Proposal appendix · **Created:** 2026-08-22

This appendix is normative for quota runtime behavior in
[Access Control and Quota Accounting](./router-native-access-control).
The [resource contract](./router-native-access-control-contracts) owns durable policy
and ledger schemas; the [deployment contract](./router-native-access-control-deployment)
owns durability, readiness, and disaster recovery.

## Rate-limit policy and counter ownership

A reusable policy is definition, not state. Counter identity is always:

```text
binding_id + rule_id
```

This prevents two keys that reuse `developer` from accidentally sharing quota. It
also makes intended sharing explicit:

| Binding subject | Counter semantics |
| --- | --- |
| Key | Only that key consumes the counter. |
| User | All keys resolving to that User binding share the counter. |
| Team | All keys resolving to that Team binding share the counter. |

```json
{
  "name": "developer",
  "rules": [
    {"metric":"requests","limit":"12","algorithm":"sliding_log","window":"60s","accounting":"request"},
    {"metric":"total_tokens","limit":"30000","algorithm":"sliding_log","window":"60s","accounting":"response_actual"},
    {"metric":"total_tokens","limit":"500000","algorithm":"calendar_window","period":"day","timezone":"UTC","accounting":"response_actual"}
  ]
}
```

An API-key create may select one existing policy or submit inline rules. Inline rules
atomically create an ordinary reusable RateLimitPolicy plus the key binding and return
its policy ID; it appears in normal Budget APIs, may be renamed/rebound, and is never
hidden, cascaded, or auto-deleted with the key. A policy can be deleted only with zero
bindings. Only a limit change on sliding-log, calendar-window, or concurrency keeps
rule ID and usage. Changing
metric, algorithm, window, timezone, accounting, refill, or GCRA semantics creates a
new rule ID. Publication carries state only when proven; otherwise it drains or uses
an audited reset. Reductions use a deny barrier and never clear counters.

Sliding windows are duration based rather than limited to minute/hour presets. The
same exact rule therefore represents an eight-hour API-key currency budget:

```json
{
  "metric": "cost",
  "limit": "20",
  "algorithm": "sliding_log",
  "window": "PT8H",
  "accounting": "response_actual",
  "enforcement": "enforce"
}
```

The rule charges pinned actual Model prices at settlement. The request that crosses
the limit completes and records its complete cost; the next admission is denied until
the exact rolling window releases sufficient settled debit. Key detail reads this
counter directly and Usage reads the ledger, so ingestion lag never creates false
remaining capacity.

All non-negative request/token/cost quantities use one `QuotaInteger`: a canonical
decimal of at most 42 digits, encoded in Valkey as six base-10,000,000 limbs. Functions
parse, carry, add, and compare limbs without Lua floating-point conversion; every
intermediate stays in the exact integer domain. PostgreSQL uses `numeric(42,0)`.
Publication proves limits plus maximum single-settlement debits fit this domain and
rejects an unsafe rule/snapshot. Cost counters store the exact per-million numerator
defined by the Model runtime contract.

OpenAPI rejects JSON numbers for quota quantities. Whole-unit meters carry `limit`,
`used`, and `remaining` as canonical integer strings. A cost rule accepts a canonical
currency-decimal `limit` with at most 15 fractional digits; the compiler multiplies
it exactly by the internal `10^15` currency scale and stores only the resulting
QuotaInteger in runtime state. Its wire shape is:

```json
{
  "metric": "cost",
  "limit": "5",
  "used": "2.5",
  "remaining": "2.5",
  "currency": "USD",
  "completeness": "complete",
  "knownDispatches": "18",
  "incompleteDispatches": "0",
  "capacityState": "available"
}
```

`limit`, `used`, and non-null `remaining` are canonical decimals with at most 15
fractional digits. `used` is the exact sum of known dispatch cost. `remaining` is an
always-present `string|null`: complete state returns the string and partial/unknown
returns null, so a client cannot show
false capacity. In complete state it is `max(limit - used, 0)`; `over_limit` plus
`used - limit` expresses the exact overage, never a negative remaining value. Partial means both known and incomplete dispatches; unknown means no
known dispatch and at least one incomplete dispatch; an empty meter is complete zero.
`capacityState` is `available|exhausted|over_limit|fenced|unknown`. Enforced incomplete
usage is fenced; shadow incomplete usage is unknown. The atomic live read also returns
reset and freshness. The API never exposes internal numerator, limb, or scale fields.

## Valkey runtime contract

PostgreSQL is desired state; Valkey is applied runtime state.

```text
access:credential:<kid> -> key_id, HMAC, pepper, subject/status/time references
access:delegation:<session-id> -> HMAC, session, principal, key/delegation epoch, context, expiry
access:delegation-epoch:<key-id> -> current epoch
access:provider-credential:<credential-id>:<version> -> provider, origin, wrapped secret, KEK, lifecycle
access:provider-credential:active:<credential-id> -> status, revision, active/retiring versions
management:revocations:active -> optional applied revocation generation
management:revocations:g:<generation> -> ready marker plus session/principal/source deny projection
routing:snapshot:<namespace-id>:<revision> -> digest, compiled Models/Recipes/Entrypoints
routing:active:<namespace-id> -> active revision/digest
routing:fleet-replicas:v1 -> bounded live Router process leases
routing:{partition}:replicas:<namespace-id> -> namespace-loaded process leases
routing:publication:<publication-id> -> staged | active | failed
routing:publication-required:<publication-id> -> current rollout member IDs
routing:publication-acks:<publication-id> -> warmed member IDs
access:policy:<key-id>:<revision> -> grants, quota bindings/rules, effective routing context
access:active:<key-id> -> current revision, optional pending publication
access:publication:<publication-id> -> staged | active | failed
access:deny:<resource-type>:<resource-id>
quota:{partition}:<binding-id>:<rule-id>:requests|tokens|cost|concurrency
quota:{partition}:unknown-fence:<fence-id> -> admission, reason, affected bindings
quota:{partition}:unknown-by-binding:<binding-id> -> active enforce-fence IDs
pending:{partition} -> admission deadlines
pending:{partition}:<admission-id> -> bindings, digest, heartbeat, deadline
pending:{partition}:<admission-id>:dispatches -> intents, route/rule/charge facts, state, evidence
settled:{partition}:<admission-id> -> canonical usage digest
usage-stream:{partition}
access:applied-revision:<namespace-id>
```

Management sessions, issuer evidence, exchange challenges, and logout tombstones stay
authoritative in PostgreSQL. When a runtime store is configured, only the rebuildable
Management revocation-barrier generation is projected to Valkey for globally
acknowledged deny checks; a Management-store-only deployment performs the same checks
against PostgreSQL. Neither form moves browser sessions into the inference-access
keyspace or makes Valkey a Management identity authority.

A 2-4 KiB compiled policy keeps 10,000 keys in tens of MiB before store overhead; the
hot path never joins PostgreSQL. Access runtime reads Valkey for every credential verification
and has no positive local authorization cache. A later revisioned L1 cache may use at
most one second TTL plus invalidation and must retain deny-barrier checks.

Every Router process using durable routing state renews a Redis-time fleet lease before it can report
ready or serve a routing generation. Publication snapshots the live fleet into the
namespace-local rollout proof, then waits for each member to discover the namespace,
validate and warm the candidate, acquire its namespace lease, and acknowledge the
exact publication digest. An empty fleet never makes a publication ready.
The activation transaction rechecks the namespace-local live set and acknowledgements,
closing the join race without a cross-slot transaction. A process that joins after
activation remains unready for that namespace until it has loaded the active generation.
Fleet membership is liveness only; it carries no tenant, credential, grant, or routing
authority and never appears in user-authored configuration.

A secret-bearing Management mutation has one additional delivery barrier. Before the
one-time API key or delegated credential leaves Management, one partition-local
operation compares the exact publication ID, runtime epoch, routing revision, and
routing digest observed by the credential reader with the coupled active gates. Using
store time, that same operation expires dead namespace leases and verifies that every
remaining live replica registered that exact generation. Membership change retries the
observation; a corrupt gate fails closed; an empty fleet is never success. A replica
that exits between candidate acknowledgement and local activation stops blocking only
after its lease expires, while a joining replica cannot register or serve a different
generation under the active gate.

`access:active:<key-id>` is the sole active policy pointer. A pending pointer is ignored
until its publication gate activates; CAS revisions prevent stale projectors. Pending
admissions never expire silently. Long requests heartbeat within their pinned maximum
lifetime, and one quorum-acknowledged Function journals each stable bounded dispatch
intent and maximum-charge facts before its backend path starts.

Normalization records known/unknown state and an evidence digest. `actual`
reconciliation requires canonical usage for every started intent; otherwise only
conservative debit or waiver is valid. Before admission, the Function drains a
bounded oldest-expiry batch. If the oldest remaining deadline is at or before Valkey
`TIME`, it returns `503` without consuming counters. A background reconciler installs
one stable fence on every affected enforce binding and persists a fence event. Shadow-
only unknown remains an incomplete ledger/health fact and never freezes admission.

Reconciliation is an idempotent saga with one ID. PostgreSQL records the immutable
charge plan while the fence is `reconciling`; one Valkey Function atomically applies
all deltas/marker and appends a correction item. The writer CAS-transitions the
settlement and appends correction UsageEvent/audit in one transaction. Only then does
a final idempotent Function remove the fence from every binding. Crashes retry the
same ID; partial progress never unfreezes a binding. A different plan conflicts.

Settlement-marker TTL covers maximum request lifetime, settlement retry, and Valkey
failover replay, not a monthly window. PostgreSQL `usage_settlements` independently
covers stream/pending replay. Stream retention, both deduplication horizons, and
recovery limits are coordinated but separately sized.

The first production runtime requires a highly available single-writer Valkey/Redis
and does not claim arbitrary Cluster atomicity. One namespace quota partition contains
all counters, pending work, fences, settlement, and stream keys; publication rejects
another partition. A future clustered runtime may map the partition to one hash tag.
Resharding requires a separately specified drain/epoch transition. All Functions use
Valkey server time for windows, leases, expiry, and reset values.

## Reproducible capacity evidence

The repository provides an opt-in capacity gate for the first production
runtime. Run it against a dedicated or otherwise isolated Valkey/Redis endpoint:

```bash
docker run --rm --name vllm-sr-access-capacity-valkey \
  -p 127.0.0.1:6379:6379 valkey/valkey:8-alpine \
  --save "" --appendonly no

# In another terminal:
ACCESS_CAPACITY_REDIS_URL=redis://127.0.0.1:6379/0 \
  make perf-access-capacity
```

The connection URL is read only from the environment and is never written to
the report. The gate uses a unique key prefix by default, refuses an explicit
prefix that already contains data, and removes its keys after the run. Set
thresholds through the `ACCESS_CAPACITY_*` Make variables when a deployment has
a reviewed service-level objective; the checked-in defaults are intentionally
broad, stable capacity guardrails rather than a hardware leaderboard.
Use a dedicated process when Redis operation counts are acceptance evidence:
key-prefix isolation protects data and memory accounting, while Redis
`commandstats` is process-wide.

The gate compiles and publishes 10,000 independent API-key credentials, access
policies, model-visibility grants, and quota bindings with the production
publication implementation. Multiple independent production access-runtime
instances then authenticate, authorize, admit, journal, and settle concurrent
requests against one real quota store. A production usage-stream consumer group
observes and acknowledges every settlement. Finally, one runtime client is lost,
the same request is rerouted, a replacement joins, and the shared request limit is
proved to remain exact.

Each run writes machine-readable `report.json` and a concise `summary.md` under
`.agent-harness/access-capacity/<run-id>/`. Both identify their failover scope as
`router_replica` and report:

- projection compile/publish duration, throughput, Redis operations, key count,
  bytes per API key, and sampled policy-isolation violations;
- authentication, admission, and settlement p50/p95/p99/max latency, event
  throughput, Redis operations per event, and incremental bytes per event;
- produced, observed, and acknowledged usage, consumer pending/lag, and usage
  observation latency; and
- failed-client rejection, reroute and replacement success, post-limit denial,
  and global-counter consistency.

This focused gate is not Router/Envoy HTTP end-to-end coverage and does not test
Valkey server failover. Those remain separate deployment acceptance scenarios so
test fixtures cannot be mistaken for data-plane or high-availability evidence.
