# Router Quality, Security, Resource-Safety, and API-Boundary Audit

## Document status

This reference records the repository-wide audit requested by
[#2375](https://github.com/vllm-project/semantic-router/issues/2375). It is a
maintainer and contributor index for understanding the identified risk classes,
their owning source surfaces, and the evidence required to close them.

- **Audited base:** `62e5eef7b5be2ad840776a9bfce5a15f3853d356`
- **Audit period:** 2026-07-12 through 2026-07-13
- **Roadmap parent:**
  [#2287](https://github.com/vllm-project/semantic-router/issues/2287)
- **Security parent:**
  [#2357](https://github.com/vllm-project/semantic-router/issues/2357)
- **Execution plan:**
  [PL0038](../agent/plans/pl-0038-router-hardening-audit.md)
- **Status:** findings and terminal ownership are inventoried; individual rows
  remain open until their stated evidence is merged and green.

The review combined source-level control-flow and data-flow analysis, route and
wire-type inventories, lifecycle ownership analysis, dependency scans,
baseline tests, existing-issue and technical-debt deduplication, and a focused
proof-of-fix for the Looper internal-request boundary. This document describes
defensive invariants and ownership boundaries. It intentionally omits reusable
attack requests, credentials, private infrastructure, and unnecessary payload
details.

## Terminal ownership vocabulary

The labels below describe finding lifecycle throughout the register. The
terminal catalog names the concrete issue, pull request, or debt record instead
of repeating a separate status column.

| Status | Meaning |
| --- | --- |
| Proof change in audit PR | A focused implementation is included with the audit work, but is not closed until all listed tests, AMD regression, and pull-request CI pass. |
| Existing tracker | An open issue already owns the behavior and must receive the current evidence and acceptance boundary. |
| Reopened exact tracker | A closed issue was an exact match, review confirmed the behavior remained present, and the tracker was reopened as the terminal owner. |
| Open child issue | A bounded GitHub issue owns implementation and acceptance evidence. Issue creation establishes ownership; it does not fix the defect. |
| Durable debt | An indexed technical-debt record owns the structural gap while implementation proceeds through bounded issues and pull requests. |
| Accepted risk | Valid only with a reviewed threat model, compensating controls, named owner, and revisit or expiry date. No finding in this audit is implicitly accepted. |

Finding severity and closure status are independent. Creating an issue does not
fix a defect, and passing one local test does not close a deployment-visible
risk.

## Executive conclusion

The audit identified two critical access-control gaps, a broad group of
high-priority security, durability, boundedness, and contract failures, and a
smaller set of medium architecture and performance debts. The recurring design
failure is a weak representation crossing into a privileged subsystem without
one authenticated, versioned, bounded, tenant-owned, or lifecycle-owned seam.

The audit does not recommend a router rewrite. The safe closure strategy is a
series of bounded changes: authenticate mandatory request boundaries; protect
management and Replay access; establish trusted tenant identity; introduce
runtime-generation ownership; bound asynchronous actors and native handles;
converge configuration contracts; contain dashboard network, browser, file,
job, and provisioning boundaries; and make the relevant gates mandatory.

The accompanying Looper work addresses
[#1443](https://github.com/vllm-project/semantic-router/issues/1443) as a
focused proof. It adds runtime-owned authentication for recursive router calls,
fail-closed handling of reserved metadata, no-leak boundaries, reload
continuity, and a deployment-secret mode for cross-replica reentry. It must not
be described as complete until focused tests, CPU gates, AMD positive and
negative regression, and pull-request CI are all green.

Issue #2375 is complete only when every raw finding maps to a merged pull request, an
open issue, indexed debt, or a reviewed accepted-risk record, and every
high-priority class has concrete ownership and acceptance evidence. No finding
may remain only in this prose document.

## Severity and confidence model

| Level | Meaning in this audit | Representative impact |
| --- | --- | --- |
| Critical | Reachable confidentiality or control-plane integrity loss without an adequate application-layer authorization boundary | Management API and detailed Replay access |
| High / P1 | Policy bypass, tenant-boundary failure, credential or content leakage, data loss, process crash, unbounded production resource use, or a broken acceptance gate | Looper trust, identity ownership, writer drain, runtime reload ownership, ML containment, strict config admission |
| Medium / P2 | Important hardening, interoperability, maintainability, or performance risk with narrower reachability or failure conditions | External RAG bounds, native lock granularity, in-memory search, hotspot extraction |
| Low | Operator-controlled or narrowly scoped correctness and hardening defect | CLI runtime-state filename normalization |

Confidence describes the strength of the observed code-level failure model,
not production frequency. The source-path findings below are high confidence.
Dependency exposure that depends on an exact deployed patch version remains
medium confidence until builds and images are reproducibly pinned.

## Coverage matrix

| Area | Reviewed surfaces | Consolidated result |
| --- | --- | --- |
| Configuration and validation | Loader, canonical v0.3 models, validators, projections, knowledge bases, decision trees, structured payloads | Missing supported-version gate, warning-only unknown fields, repeated manifest I/O, and cross-surface rule-tree drift |
| Extproc and headers | Header/body stage ordering, skip/Looper/Replay/Responses paths, redaction, routing credential mutation | Internal-request trust defect, branch-dependent credential handling, content logging, and raw-header identity drift |
| Router runtime and services | Construction, registry publication, config refresh, reload swap, shutdown | No complete ownership generation; classification config/classifier pairs can race; retired resources are abandoned |
| Selection and model selection | Registry/factory, Elo, RL, GMT, RouterDC, adapters, native handles | Eager native allocation, missing close ownership, unbounded adaptive state, and invalid-duration panics |
| Model runtime | Task executor and native model initialization, queues, reload, unload | Go task execution is structurally bounded; native generations and schedulers are not |
| Looper | Client I/O, fan-out, workflow state, internal reentry, reload interaction | Internal-auth proof plus body, work, state-store, and client-lifecycle follow-ups |
| Memory and learning | Milvus and Valkey stores, access tracking, cache wrapper, outcome ingestion | Unauthenticated outcome provenance, unbounded detached tracking work, unclear client ownership, and tenant identity gaps |
| Vector store | Runtime constructor, file store, pipeline, manager/registry, memory backend | Active jobs cannot cancel, working sets and statuses are unbounded, and metadata/backend changes are not transactional |
| Cache | Backend interface, cancellation, result metadata, reclaim lifecycle | Missing cancellation and cross-request shared similarity; hybrid reclaim lifecycle is a positive control |
| Replay stores | API dispatch, recorder, memory/Redis/Postgres/Qdrant/Milvus, retention, listing, close | Missing authorization and tenant minimization; asynchronous acknowledgment, drain, TTL, total-size, and pagination gaps |
| API and header contracts | OpenAI, Anthropic, Responses, Files, Vector Stores, Models, identity and diagnostics headers | Official SDK use is mixed with weak unions and duplicate models; ownership and versioning are missing at several seams |
| CLI | Pydantic contract, migration/version behavior, runtime paths, generated Envoy | Input strictness differs from Go; one low-risk path gap; edge stripping is defense in depth only |
| Dashboard | Authentication/RBAC, router proxy, browser credential transport, outbound fetches, config mutation, ML pipeline, OpenClaw | Egress, bearer transport, fail-open route authorization, file/job/image boundaries, config transaction gaps, and a WebSocket disconnect/fan-out lifecycle race |
| Kubernetes and operator | Dynamic CRDs, converter, operator CRD/webhook/controllers, canonical generation | Raw preserved routing can bypass strict core admission; legacy/canonical ownership and boolean semantics drift |
| Native bindings | Candle, ONNX, ML, NLP, OpenVINO Go/Rust lifecycle seams | Candle/ONNX/NLP lifecycle and concurrency gaps; lower ML wrappers are sound; ONNX Go tests do not compile |
| E2E, testing, performance | Acceptance floors, native compile lanes, numeric comparison, affected-profile selection, local image/serve orchestration | Report-only zero floors, missing ONNX compile coverage, non-gating numeric comparison, and local feature-gate false-green paths |
| Dependencies and supply chain | Go, npm, Cargo, frozen Python lock, toolchains | Reachable direct advisories and no reproducible multi-ecosystem vulnerability gate with expiring exceptions |
| Fleet and training | Large CLI/training orchestrators, artifact and data I/O, existing debt | Keep outside the proof PR; use existing Fleet debt and a focused training provenance/lifecycle follow-up |

## Consolidated risk register

The package IDs below consolidate related observations so that one underlying
defect is not counted several times. Raw IDs remain available in the mapping
later in this document.

### Critical security boundaries

#### SEC-01 — Management authorization and learning provenance

The management listener registers non-health read and mutation routes without
one deny-by-default authentication and authorization middleware. The same
surface can expose live configuration and mutate operational state. Learning
outcomes also accept caller-supplied authority and target claims without
binding the update to an authenticated principal and tenant-owned routing
event.

Owner surfaces include
[`pkg/apiserver`](../../src/semantic-router/pkg/apiserver),
[`pkg/extproc/router_learning_runtime.go`](../../src/semantic-router/pkg/extproc/router_learning_runtime.go),
the management listener manifests, and local CLI port publication.

Required invariant: every non-health route has an explicit method, permission,
sensitivity, and audit policy; the listener is protected by default; config
views are typed and redacted; and learning source, event, model, idempotency,
and abuse controls are derived from trusted server state. Existing
[#1452](https://github.com/vllm-project/semantic-router/issues/1452) owns the
feedback component. The management boundary is owned by
[#2463](https://github.com/vllm-project/semantic-router/issues/2463).

#### SEC-02 — Replay confidentiality and tenant isolation

Replay list and detail operations do not establish an authenticated principal
or tenant before reading records, and the current detail behavior exposes more
captured content than a summary caller needs. Backend keys and queries do not
consistently include trusted tenant ownership.

Owner surfaces include
[`pkg/extproc/router_replay_api.go`](../../src/semantic-router/pkg/extproc/router_replay_api.go)
and [`pkg/routerreplay`](../../src/semantic-router/pkg/routerreplay).

Required invariant: authenticated auditor/operator scopes, tenant in the
backend query predicate, summary-only default responses, a separate audited
detail privilege, capture-time minimization, and bounded retention. The exact
historical tracker
[#1146](https://github.com/vllm-project/semantic-router/issues/1146) was
reopened during this audit and remains the terminal owner;
[#2157](https://github.com/vllm-project/semantic-router/issues/2157) owns
summaries and pagination, and
[#2364](https://github.com/vllm-project/semantic-router/issues/2364) owns shared
resource isolation.

### High-priority request, identity, and diagnostics boundaries

#### SEC-03 — Authenticated internal Looper envelope

The previous design inferred privileged internal execution from ordinary HTTP
metadata. The proof change makes that decision depend on a runtime-owned
credential, validates the entire reserved metadata envelope before the skip or
body path, projects only validated values into typed request context, and
removes reserved metadata before upstream dispatch.

The credential is generated for a same-process default and retained across
config reload. A load-balanced, multi-replica reentry topology must inject one
shared 256-bit secret through `VLLM_SR_LOOPER_SHARED_SECRET`; it must remain in
the deployment secret store rather than router YAML. The configured Looper
endpoint is a privileged operator-controlled endpoint because it receives
request content, provider credentials, and the internal credential. See the
[security hardening guide](security-hardening.md) for the deployment contract.

The canonical Kubernetes CLI path creates one release-scoped immutable Secret
through standard input, generates a shared Looper key when the operator did not
supply one, and reuses only that key across later generations. Every retained
Helm revision protects its referenced Secret from garbage collection, and
credential-bearing revisions use non-overlapping `Recreate` rollouts. Cleanup
fails closed when current or rollback references cannot be proved.

The privileged HTTP client uses a private pooled direct transport, ignores
ambient proxy variables, and refuses redirects. The shared value remains a
generation-lifetime bearer credential: TLS and network policy are mandatory,
and threat models with hostile east-west observers require request-bound
authentication or mTLS under the broader
[#2336](https://github.com/vllm-project/semantic-router/issues/2336) hardening
track.

Owner surfaces include
[`pkg/looper`](../../src/semantic-router/pkg/looper),
[`pkg/extproc`](../../src/semantic-router/pkg/extproc),
[`pkg/headers`](../../src/semantic-router/pkg/headers), and affected E2E
profiles. Terminal owner:
[#1443](https://github.com/vllm-project/semantic-router/issues/1443), cross-linked
to [#2357](https://github.com/vllm-project/semantic-router/issues/2357).

#### SEC-04 — Caller and provider credential separation

Caller authentication and provider credentials are represented through the
same outbound header. On paths where no provider credential resolves, the
caller credential can be preserved; early or alternate branches can bypass
normal cleanup.

Required invariant: authenticate the caller into typed identity, strip its
credential at the mandatory ingress prelude, and inject only a selected
provider credential. Forwarding caller authorization must be a per-backend,
explicit, default-off contract that applies consistently to skip, Looper,
Replay, fast-response, and error paths. Terminal owner:
[#2286](https://github.com/vllm-project/semantic-router/issues/2286).

#### SEC-05 — Trusted identity and state ownership

Several consumers reread configurable identity headers directly, use
case-sensitive map access, hard-code a default header, or fall back to another
untrusted representation. Responses objects and previous-response chains are
also keyed by opaque ID without owner in the storage contract.

Required invariant: one ingress-owned `TrustedIdentity` value carries
principal, tenant, groups, and provenance to authz, rate limits, cache, memory,
Replay, Responses, learning, and audit. Tenant or owner must be part of backend
keys and predicates, not an after-load filter. ID generation must fail closed
on entropy failure. Terminal owners:
[#1445](https://github.com/vllm-project/semantic-router/issues/1445),
[#2362](https://github.com/vllm-project/semantic-router/issues/2362), and
[#2364](https://github.com/vllm-project/semantic-router/issues/2364).

#### SEC-06 — Content-free diagnostics

Header redaction is centrally implemented, but several logging paths can still
emit request, response, query, tool, or provider content, including fields in
processing-response bodies that the header redactor does not remove.

Required invariant: ordinary INFO, DEBUG, and TRACE output contains safe
metadata such as request identifiers, byte or token counts, model, latency, and
approved hashes—not user content. Canary tests must cover bodies, immediate
responses, provider errors, queries, and tool fields. Any exceptional content
diagnostics require a short-lived, access-controlled mode and separate
retention policy. The closed credential-header issue
[#2132](https://github.com/vllm-project/semantic-router/issues/2132) is not a
substitute. The content policy and canary work is owned by
[#2464](https://github.com/vllm-project/semantic-router/issues/2464).

### Dashboard and control-plane security boundaries

#### NET-01 — Dashboard outbound fetch policy

OpenWeb, FetchRaw, and remote setup import each perform server-side URL access
without one shared policy that controls resolution, dial targets, redirects,
proxies, response expansion, and deadlines. Some operations are available to a
read role.

Required invariant: one bounded outbound-fetch service validates every
resolution, dial, and redirect against an administrator policy, pins the
validated address, makes proxy/internal-network behavior explicit, and applies
hard request and response budgets. The exact historical tracker
[#1388](https://github.com/vllm-project/semantic-router/issues/1388) was
reopened during this audit and remains the terminal owner for the broadened
outbound-fetch boundary.

#### WEB-01 — Browser session credential transport

The browser UI copies a reusable bearer credential into script-visible storage
and protected URLs even though the backend already issues an HttpOnly session
cookie.

Required invariant: cookie-only normal browser sessions, no reusable
credential in URLs or script storage, CSRF protection for unsafe methods, and
one-time audience/path-scoped tickets only for a transport that cannot use the
cookie. Terminal owner:
[#2465](https://github.com/vllm-project/semantic-router/issues/2465).

#### WEB-02 — Fail-closed route authorization

Dashboard route registration and permission lookup are separate. Unknown
protected routes can inherit a read permission, and advertised Replay and
feedback permissions are not the actual gate for their proxied operations.
Middleware also authorizes at request entry; outside the auth-store account
mutations fixed by this proof, a slow in-flight settings, router-deploy, or MCP
write can retain that snapshot after its session is revoked.

Required invariant: register route, method, permission, audit action, and
sensitivity together; deny unknown protected routes; revalidate a live session
and current permission immediately before every privileged side effect (or in
the same transaction where possible); bound request admission; and make
startup or a contract test fail when metadata is missing. Terminal owner:
[#2466](https://github.com/vllm-project/semantic-router/issues/2466).

#### WEB-03 — Password lifecycle and password-manager integration

The dashboard previously had no self-service password-change endpoint, did not
revoke older sessions after an administrator reset, and let its SPA fallback
return `200 OK` for password-manager probes under `/.well-known/`. Login also
lacked an account-scoped throttle and performed distinguishable account lookup
and password work. Those gaps made a browser's compromised-credential warning
hard to resolve safely; disabling autofill would only hide a useful warning.

The audit proof adds a browser-compatible change-password route and form,
session replacement with prior-session revocation, bounded and uniform login
verification, strict non-cacheable auth requests, versioned long-password
hashing with legacy migration, and a password policy aligned with NIST SP
800-63B-4: a 15-character single-factor minimum, Unicode NFC normalization,
no composition rules, a sufficiently extensible common, compromised, and
contextual blocklist with separate exact-value and case/separator-variant
matching, and full-password verification. It also redirects
`/.well-known/change-password` and returns real `404` responses for unknown
well-known probes as recommended by Chrome's password-manager guidance.

The final review also closes credential-state races and startup ambiguity. A
successful login compares and optionally migrates the verified hash, updates
login state, and inserts its server-side session in one compare-and-swap
transaction; a concurrent self-change, administrator reset, disable, or delete
therefore cannot mint a post-revocation session or restore an older hash. JWTs
must use exact HS256, carry expiration and a non-empty server-owned session ID,
and configured signing keys reject weak length, control/whitespace, and obvious
placeholder values. Invalid signing-key or blocklist configuration fails
dashboard startup. Auth database and SQLite sidecar files are private by
default and existing permissive files are tightened at open. Deactivation and
session revocation also share one transaction, so reactivation cannot revive a
token issued before the account was disabled. Administrator role/status
updates compare against the state the handler authorized, and the store checks
the last-active-`users.manage` invariant transactionally across role-derived
and direct permissions, so stale or mutually destructive mutations cannot
restore a disabled account or remove every account manager. A monotonic auth
generation also rejects login or password-change work that spans an
active-to-inactive-to-active ABA cycle. Account create, reset, patch, and delete
transactions revalidate the acting session and live `users.manage` permission
at their commit boundary. This account-store guarantee does not waive the
broader WEB-02 control-plane reauthorization work.

The browser uses stable username/current/new fields with one new-password
input, removes the form after success, and identifies stale unauthorized
responses with a credential-free session revision. It never disables the
password manager or copies the bearer value into the unauthorized DOM event.

Required invariant: every password establishment path shares the policy;
verification work and failed attempts are bounded per account and globally;
private peers are not collapsed into caller-controlled forwarded identities;
unknown, inactive, and wrong-password login failures are externally uniform;
changing or resetting a password revokes all
prior sessions atomically; a self-change cannot keep the same password; request
and response bodies are small, strict, valid Unicode, non-cacheable, and
content-free; explicitly configured empty denylist corpora fail startup; and
browser fields retain stable `username`, `current-password`, and `new-password`
semantics. Production ingress must still enforce a trusted client-source rate
limit; process-local admission is a bounded backstop, not a distributed denial-
of-service boundary. This proof is owned by the audit pull request. The
separate script-visible bearer, query-token, and CSRF boundary remains WEB-01 /
[#2465](https://github.com/vllm-project/semantic-router/issues/2465).
Credential-lifecycle audit write failures are now content-free and
observable, while transactionally durable event/outbox delivery remains owned
by [#2482](https://github.com/vllm-project/semantic-router/issues/2482).

#### ML-01 / ML-02 — ML sidecar, upload, job, and artifact containment

The ML sidecar accepts filesystem paths and lacks one authenticated transport
boundary. The shipped containers do not share a consistent path namespace.
Dashboard uploads, job identifiers, output directories, concurrency, cleanup,
and returned artifact paths also lack a complete containment and resource
contract.

Required invariant: opaque server-owned upload, job, and artifact handles; one
shared root mounted at the same path; symlink-aware containment; loopback or
authenticated workload transport; hard body, file, queue, and concurrency
budgets; random private per-job directories; contained downloads; cancellation
and retention cleanup. The shared deployment and resource contract is owned by
[#2467](https://github.com/vllm-project/semantic-router/issues/2467).

#### OC-01 — OpenClaw provisioning trust

OpenClaw provisioning combines caller-selected image input with external skill
identifiers used in host workspace operations.

Required invariant: catalog-owned skill IDs, symlink-aware path containment,
server-owned asset copying, administrator-controlled image allowlists or digest
and provenance policy, and a non-root minimal container profile. Terminal
owner: [#2468](https://github.com/vllm-project/semantic-router/issues/2468).

### Runtime ownership, durability, and boundedness

#### RT-01 — Runtime generation and classification snapshot ownership

Router construction has no complete rollback stack, the router closes only a
subset of owned resources, reload swaps and retires objects without request
leases, and process shutdown drains only selected children. Classification
refresh writes config and classifier state under a lock while request paths can
observe them independently, and retired classifiers abandon native and MCP
resources.

Required invariant: one immutable `RuntimeGeneration` owns all closeable
children and bounded actors. Construction rolls back in exact reverse order;
requests lease one generation; reload stops admission and drains with a
deadline; shutdown gracefully stops servers before children. Classification
publishes one immutable config/classifier snapshot and retires it through the
same leases. Durable owner:
[TD045](../agent/tech-debt/td-045-runtime-generation-ownership-gap.md), with a
bounded implementation track in
[#2470](https://github.com/vllm-project/semantic-router/issues/2470) and
evidence added to
[#2396](https://github.com/vllm-project/semantic-router/issues/2396).

#### RT-02 — Looper workflow state and I/O budgets

Workflow state is constructed per request. Redis clients therefore lack a
stable owner, while the memory backend cannot resume across independent HTTP
turns. Looper HTTP reads buffer complete JSON or SSE bodies and candidate
fan-out bounds active calls but not waiting work or retained bytes.

Required invariant: one generation-owned state service with consistent
TTL/resume/take semantics; stable client lifecycle; per-call and aggregate byte
and token budgets; bounded worker submission; context-aware admission; capped
errors; and incremental streaming backpressure. Existing
[#1456](https://github.com/vllm-project/semantic-router/issues/1456) and
[#2336](https://github.com/vllm-project/semantic-router/issues/2336) own breadth
and Looper hardening. Workflow-state ownership is tracked by
[#2471](https://github.com/vllm-project/semantic-router/issues/2471).

#### RT-03 — Replay actor, durability, retention, and pagination

Persistent Replay stores can acknowledge asynchronous work before durable
success, expose unsafe close/admission ordering, use background contexts, apply
TTL inconsistently, and scan or materialize more retained history than a page
requires. Raw tool fields can escape the intended record-size budget.

Required invariant: a bounded `open -> closing -> closed` actor that stops
admission, drains, reports failures, closes its client, and honors runtime
cancellation and timeouts; backend-conformant TTL; a total serialized-record
budget; and backend-native cursor pagination. The lifecycle work is owned by
[#2472](https://github.com/vllm-project/semantic-router/issues/2472) and
coordinates with
[#1146](https://github.com/vllm-project/semantic-router/issues/1146) and
[#2157](https://github.com/vllm-project/semantic-router/issues/2157).

#### RT-04 — Cache cancellation and per-result similarity

Cache operations discard request/runtime context, and similarity is published
through cache-global mutable state that can be overwritten or inherited by
another request. One constructor failure also leaves a client open.

Required invariant: context on every backend method, cancellation reaching
storage and embedding work, and a lookup result value containing its own
similarity. Misses and errors must return an explicit absent or zero score.
Terminal owner: [#2473](https://github.com/vllm-project/semantic-router/issues/2473).

#### RT-05 — Vector ingestion lifecycle and consistency

Active vector ingestion cannot be cancelled, a file expands into a full
in-memory text/chunk/vector workset, statuses grow without a complete retention
policy, and registry/file/backend mutations can leave partial state.

Required invariant: a root-derived context and bounded `Stop(ctx)`, streaming
bounded batches, enforced file/worker/queue/status budgets, constructor cleanup,
and saga or reconciliation semantics for multi-store mutation. Terminal owner:
[#2474](https://github.com/vllm-project/semantic-router/issues/2474).

#### RT-06 — Selection ownership and adaptive cardinality

The selection registry eagerly constructs algorithms and native handles it may
never use, does not own their close lifecycle, and can replace handles without
draining readers. Adaptive selectors retain unbounded user, session, query, and
graph state; one affinity table is written but not consumed.

Required invariant: instantiate only configured algorithms, use leased and
idempotent handle generations, and apply tenant quotas plus TTL/LRU and
referentially safe graph eviction. Native ownership belongs with
[#2396](https://github.com/vllm-project/semantic-router/issues/2396); cardinality
evidence belongs with
[#2222](https://github.com/vllm-project/semantic-router/issues/2222). These
trackers own the high-priority lifecycle and boundedness work.

#### RT-07 — Memory access tracking and client ownership

Successful memory retrieval can start detached, long-lived tracking work per
request. Some counters use read-modify-write, and a dedicated Milvus client is
treated as borrowed so no owner closes it.

Required invariant: one bounded and coalescing store-owned tracking actor,
atomic backend updates, explicit drop/exactness policy, cancel and drain on
shutdown, and owned-versus-borrowed client semantics. Existing
[#2339](https://github.com/vllm-project/semantic-router/issues/2339) owns Router
Memory stabilization, including the actor and client-ownership acceptance
amendments from this audit.

#### RT-08 — Native model generation and handle lifecycle

Candle initialization can consume one-time state on failure, later report
success without a usable model, ignore a changed path, accept unbounded batch
work, and expose a shutdown that does not unload the process-owned model.
ONNX and NLP registries hold broad locks over inference and lack complete
path-aware close semantics. One exported Candle allocation path loses the
metadata needed to free the allocation.

Required invariant: synchronized `{path, state, error, handle}` generations,
truthful retry and path-mismatch semantics, bounded cancellable schedulers,
join and unload, short-held registry locks, and idempotent Go wrappers with
constructor rollback. Native lifecycle remains mapped to
[#2396](https://github.com/vllm-project/semantic-router/issues/2396) and
[TD042](../agent/tech-debt/td-042-ffi-embedding-structure-debt.md); the mandatory
ONNX compile contract is [#2477](https://github.com/vllm-project/semantic-router/issues/2477).

#### RT-09 — In-memory search and timer lifecycle

In-memory search paths hold collection-wide locks across expensive O(N) work
and sort all candidates for small top-K queries. Several autosave or population
intervals accept zero or negative durations that can panic ticker construction
or silently stop work; one close path is not concurrently idempotent.

Required invariant: immutable or copy-on-write snapshots, incremental indexes,
context checks, bounded top-K algorithms, positive duration validation at
schema and constructor boundaries, and synchronized idempotent lifecycle
state. Duration validation is owned by
[#2475](https://github.com/vllm-project/semantic-router/issues/2475). Search
optimization remains measured backend-specific work under TD045 rather than an
unscoped rewrite.

#### RT-10 — OpenClaw collaboration client lifecycle

The race runtime deterministically observes `WSClient.close()` mutating or
draining a client channel while the room collaboration event bus concurrently
sends to it. This violates the explicit WebSocket-disconnect/SSE-continuity
acceptance that originally closed #1521 and leaves delivery integrity and
send/close panic risk under real disconnect timing.

Required invariant: one owner controls send admission and channel close;
disconnect is idempotent; fan-out cannot send after close admission begins;
backpressure and client removal are bounded; and WebSocket teardown cannot
interrupt SSE delivery or leak goroutines. The exact historical tracker
[#1521](https://github.com/vllm-project/semantic-router/issues/1521) was
reopened with the focused race evidence and is the terminal owner.

### API, configuration, dependency, and gate contracts

#### API-01 — Typed OpenAI-shaped adapters

The main Chat Completions and Anthropic paths substantially use official SDK
types, but Responses, Files, Vector Stores, and Models include local duplicate
models and weak unions. Unsupported structured tool choices can degrade to a
zero-value behavior, and documented differential compatibility tests are
absent.

Required invariant: official SDK types where coverage is sufficient; otherwise
a narrow versioned adapter with strict unsupported-union rejection,
differential JSON fixtures, complete field round trips, stable client errors,
and path-segment escaping. Terminal owner:
[#2358](https://github.com/vllm-project/semantic-router/issues/2358).

#### API-02 — External RAG typed substitution and response bounds

External RAG request templates substitute user text into configured bodies
without a typed JSON boundary, and successful responses do not have a hard byte
limit. The client already validates configured headers, has a timeout, closes
bodies, and caps error text.

Required invariant: typed-node substitution or correctly marshaled values,
final document validation, and an exact successful-response byte limit. This is
a medium-priority focused API boundary and should not be folded into dashboard
SSRF work. Terminal owner:
[#2478](https://github.com/vllm-project/semantic-router/issues/2478).

#### CFG-01 — Transactional configuration mutation

Dashboard configuration writers use different locking and rollback paths,
share fixed temporary-file conventions, and can restore stale state or leave
disk and runtime on different generations.

Required invariant: one mutation service owns canonical path locking,
cross-process coordination, generation/ETag CAS, strict validation, same-dir
temporary files, fsync and rename, runtime apply, backup identity, rollback,
request limits, and audit. Terminal owner:
[#2326](https://github.com/vllm-project/semantic-router/issues/2326), with
structural ownership in
[TD046](../agent/tech-debt/td-046-control-plane-contract-convergence-gap.md).

#### CFG-02 — Versioned schema and boolean-rule convergence

Canonical config is recognized by shape rather than an enforced supported
version. Unknown fields are warnings in one path and ignored or preserved in
others. Rule-tree shape and `NOT` behavior differ across config, CLI, DSL,
dynamic CRDs, operator, and runtime. The operator can emit preserved raw
routing data without applying the exact core semantic admission and can accept
canonical and legacy owners for the same field.

Required invariant: reject unsupported versions and unknown core fields before
interpretation; one recursive rule validator owns operators, shape, depth, and
cardinality; exact operator output passes the core validator before rollout;
and canonical/legacy dual ownership is an error. Existing owners are
[#2122](https://github.com/vllm-project/semantic-router/issues/2122) and
[#2355](https://github.com/vllm-project/semantic-router/issues/2355). Strict
version and unknown-field convergence is tracked by
[#2469](https://github.com/vllm-project/semantic-router/issues/2469), with
structural ownership in
[TD046](../agent/tech-debt/td-046-control-plane-contract-convergence-gap.md).

#### CLI-01 — Runtime state path containment

One CLI runtime-state filename is derived from operator-controlled identity
without one canonical containment and atomic private-write boundary. The
finding is narrow, but runtime state must never escape its owned root or become
partially visible to another local process.

Required invariant: normalize the identity to a bounded filename, verify the
resolved path remains under the state root, and publish private state through
an atomic same-directory write. Terminal owner:
[#2479](https://github.com/vllm-project/semantic-router/issues/2479).

#### PERF-02 — Immutable validation snapshot

Projection and knowledge-base validation repeatedly scan and decode the same
manifests within one document validation, increasing cost and allowing one
validation pass to observe different file states.

Required invariant: one immutable validation context indexes each knowledge
base and reads each referenced manifest once, with file identity in diagnostics
and reload decisions. This medium-priority architecture gap remains indexed by
[TD046](../agent/tech-debt/td-046-control-plane-contract-convergence-gap.md).

#### ARCH-01 — Control-plane hotspot boundaries

Several config, extproc, CLI, dashboard, operator, and simulator files remain
large orchestrators that mix transport, mutation, lifecycle, and presentation
responsibilities. They are debt, not precedent: touched hotspots must not grow
in responsibility, and new behavior should enter through narrow services,
adapters, strategies, or generation owners.

Required invariant: extraction-first changes, one dominant responsibility per
module, typed seams at subsystem boundaries, and structure-ratchet evidence.
The durable owners are the existing indexed records, especially
[TD006](../agent/tech-debt/td-006-structural-rule-exceptions.md),
[TD020](../agent/tech-debt/td-020-classification-subsystem-boundary-collapse.md),
and
[TD046](../agent/tech-debt/td-046-control-plane-contract-convergence-gap.md).

#### DEP-01 — Reproducible dependency and toolchain vulnerability gate

Point-in-time scans found a reachable direct Go dependency advisory, committed
Rust and Python lock findings, frontend runtime advisories, website build-time
advisories, and floating patch-level toolchains. The repository lacks one
required, lock-aware policy with reviewed expiry for exceptions.

Required invariant: patched and consistently pinned dependencies/toolchains;
per-ecosystem scans against committed resolution artifacts; separate deployed
runtime and build-only classification; and machine-readable exceptions with
owner, reason, exposure, and expiry. Terminal owner:
[#2476](https://github.com/vllm-project/semantic-router/issues/2476).

#### TEST-01 — ONNX model-free compile contract

The ONNX Go tests target stale APIs and do not compile, while no mandatory
provider-independent CI lane catches the drift.

Required invariant: `go test ./...` compiles and passes without model assets,
with model/provider receipts selected separately and tied to native capability
contracts. Terminal owner:
[#2477](https://github.com/vllm-project/semantic-router/issues/2477), linked to
[#2396](https://github.com/vllm-project/semantic-router/issues/2396).

#### TEST-02 — E2E acceptance floors

Several cases categorized as acceptance can report success with zero successful
observations. Terminal owner:
[#2379](https://github.com/vllm-project/semantic-router/issues/2379). Acceptance,
benchmark, and report-only roles must be explicit, and deterministic acceptance
must have defensible non-zero floors.

#### TEST-03 — Local feature-gate failure integrity

The canonical local image recipe grouped build commands in shell conditionals
without fail-fast semantics. A router or dashboard image build could fail and
the recipe could print a success line and continue with a stale image. The
agent serve target also resolved a global `vllm-sr` executable instead of the
repository-managed virtual environment, so build and smoke could exercise
different CLI installations.

Required invariant: every image-build failure terminates the recipe before a
success message or later stage, and build, serve, stop, and smoke use the same
repo-owned CLI environment. Missing or failed stop tooling must not report a
successful cleanup. This proof is owned by the audit pull request and requires
an actual feature-gate rerun after the failure-path fix.

#### PERF-01 — Numeric performance comparison

The repository's baseline comparison does not currently act as a reliable
numeric regression gate. Terminal owner:
[#2455](https://github.com/vllm-project/semantic-router/issues/2455), related to
the broader hardware matrix in
[#1510](https://github.com/vllm-project/semantic-router/issues/1510).

## API and weak-type inventory

Weak typing is acceptable only at a named extension seam whose consumer owns
strict decoding, bounds, and versioning.

| Contract surface | Current shape | Disposition | Required guard |
| --- | --- | --- | --- |
| OpenAI Chat Completions hot path | Official SDK types | **Keep** | Focused tests and SDK compatibility fixtures |
| Anthropic inbound and provider response | Official Anthropic/OpenAI SDK types | **Keep** | Differential fixtures |
| Anthropic custom outbound mirror | Narrow workaround for required-field SDK marshalling | **Keep behind adapter** | Documented justification, differential and round-trip tests |
| Responses request/items/tool choice/text format | Local structs with weak union fields | **Replace where SDK coverage exists; otherwise versioned strict adapter** | Reject unsupported unions, stable 4xx errors, differential corpus |
| OpenAI Files and Vector Stores | Duplicate wire structs and hand-built resource paths | **Replace or narrow adapter** | SDK parity and path-segment escaping tests |
| Models list | Duplicate core models plus intentional UI extensions | **Move to one shared composition type** | Router/API/dashboard round trip |
| Canonical router config | Public v0.3 shape with optional or unvalidated version and warning-only extras | **Move behind explicit supported-version schema** | Strict parser and cross-language golden corpus |
| Python CLI `global` and `setup` | Broad dictionaries and extra-ignore models | **Replace with typed or schema-validated models** | Forbid extras outside named extensions; prove Go parity |
| Operator raw routing and algorithm JSON | Preserved unknown JSON | **Keep only as an evolution seam behind strict core validation** | Validate the exact generated canonical document before apply |
| Plugin and RAG provider configuration | Named structured payload extensions | **Keep** | Consumer-owned strict decode, size/depth limits, versioning |
| Trusted identity | Repeated raw-header lookup and hard-coded defaults | **Replace with typed `TrustedIdentity`** | Provenance and tenant/owner propagation |
| Outcome source and target claims | Caller-supplied strings and metadata | **Replace with server-derived typed provenance** | Event ownership, idempotency, model/decision match |
| Replay and Responses storage keys | Opaque IDs without owner in several operations | **Move behind tenant-aware store contracts** | Tenant included in backend query key |
| Dashboard route authorization | Separate prefix resolver with fallback permission | **Replace with typed route registration metadata** | Exhaustive route/method contract test |
| ML pipeline path API | Caller and sidecar filesystem paths | **Replace with opaque artifact/job handles** | One owned root, containment, immutable per-job manifest |
| OpenClaw generated maps | Internal `map[string]any` builder | **Keep internally** | Validate emitted output against the owned schema before write |

## Resource lifecycle inventory

| Lifecycle class | Audited status | Required ownership or closure |
| --- | --- | --- |
| Context and cancellation | Missing from cache, active vector jobs, Replay recorder/store work, memory tracking, and parts of native scheduling | RT-03, RT-04, RT-05, RT-07, RT-08 |
| Streaming and backpressure | HTTP bodies are generally closed, but Looper buffers complete JSON/SSE and unbounded error text | RT-02; retain body-close positives while adding byte and flow-control budgets |
| Response-body closing | Verified in Looper, Router-R1, model-selection benchmark, Llama Stack, external RAG, and inspected provider paths | Positive control; retain focused/static checks |
| Goroutine and channel ownership | Per-request memory tracking, eager Looper fan-out, Replay workers, OpenClaw disconnect/fan-out, classifier construction, and native queues lack one bounded generation owner | RT-01, RT-02, RT-03, RT-07, RT-08, RT-10 |
| Timers and tickers | Some storage lifecycle is sound; non-positive configured intervals can panic or silently stop workers | RT-09; validate at schema and constructor boundaries |
| Locks and snapshots | Classification pairs race; native registries and in-memory search hold broad locks; cache/vector result state crosses requests | RT-01, RT-04, RT-08, RT-09 |
| Native handles | Lower ML wrappers close correctly; upper selection and Candle/ONNX/NLP generations do not consistently own, swap, or unload | RT-06 and RT-08 |
| Backend clients | Redis workflow clients, cache constructor failure, memory Milvus ownership, rate-limit/storage clients lack a complete owner | RT-01, RT-02, RT-04, RT-07 |
| Constructor rollback | Router, classifier, and vector construction can discard already-created closeable children | RT-01 and RT-05 |
| Reload retirement | Pointer swap lacks request or stream leases; eager close would create use-after-close | RT-01 |
| Process shutdown | Only selected hooks are registered; shutdown is not a synchronized server-to-child drain | RT-01 and RT-03 |
| Retention and cardinality | Replay, adaptive selectors, vector statuses, workflow files, and raw tool traces need hard budgets | RT-02, RT-03, RT-05, RT-06 |
| Multi-store consistency | Vector metadata/backend/files and dashboard disk/runtime writes lack transaction or saga semantics | RT-05 and CFG-01 |

## Positive controls to retain

The audit also verified behavior that should not be reopened as a defect without
new evidence:

- The clean audited base passed `make agent-validate` and
  `make agent-scorecard`.
- `go vet ./...` passed for router, dashboard backend, operator, E2E, Candle,
  ML, NLP, and OpenVINO Go modules. ONNX test compilation is tracked separately.
- Default Envoy templates strip known internal Looper and identity headers as
  defense in depth. Runtime authentication remains mandatory because not every
  deployment uses the same edge filter.
- Dashboard authentication initialization fails closed. The backend already
  provides HttpOnly, SameSite=Lax session cookies and marks them Secure under
  HTTPS.
- Credential-like response headers are centrally redacted and have canary
  coverage. SEC-06 concerns body and content fields outside that policy.
- Default semantic-cache personalized flows are skipped or user-scoped. SEC-05
  concerns custom trusted-identity consistency and ownership.
- External RAG validates and sanitizes configured headers, has a timeout,
  closes bodies, and bounds error text. API-02 concerns typed body substitution
  and successful-response limits.
- Management JSON and upload helpers include several body caps, cleanup,
  validation, atomic-write logic, and rollback-name constraints. These are not
  caller authorization.
- Replay memory storage is a bounded deep-cloning ring, and Replay IDs use
  secure randomness and fail closed.
- `modelruntime.Execute` derives cancellation, bounds active parallelism, and
  drains one outcome per task.
- Hybrid cache reclaim and `FileEloStorage` have explicit stop, done, final-save,
  and join lifecycles.
- Postgres Replay row iteration closes rows and joins close errors.
- Lower ML native wrappers use locks and idempotent `Close` correctly.
- Vector pipeline start/stop serializes generations and rejects queued work on
  stop. Active-work cancellation remains RT-05.
- Dynamic Kubernetes reconcile already runs core parsing and model-reference
  validators. The residual gap is boolean shape and published semantics.
- Operator backend sorting and split builder files are deterministic positives.
- DSL unary `NOT` compilation is internally consistent; divergence occurs in
  other input surfaces.
- The Wizmap production dependency audit reported no finding at audit time.

## Terminal closure package catalog

The symbolic package ID remains stable even if issue decomposition changes.
Every terminal-record row below points to an applied issue or indexed debt
record. The underlying defect remains open until that record's evidence merges.

| Package | Priority / class | Raw findings | Terminal record | Minimum closure evidence |
| --- | --- | --- | --- | --- |
| SEC-01 Management RBAC and outcome provenance | Critical / security | SA-01, SA-05 | #2463 and #1452 | Deny-by-default route matrix, protected listener, redacted config, source/event ownership tests |
| SEC-02 Replay confidentiality | Critical / security | SA-02 | #1146, coordinated with #2157 and #2364 | Anonymous and cross-tenant denial, summary default, detail privilege, backend tenant predicates |
| SEC-03 Looper internal authenticity | High / proof change | SA-03, XR-SEC-001 | #1443 plus audit PR | Runtime credential, reload continuity, versioned rollback-safe Kubernetes Secret, shared-secret multi-replica contract, generic rejection, no upstream/log/config leak, focused and AMD E2E |
| SEC-04 Credential separation | High / security | SA-04 | #2286 | Default stripping on every branch, explicit forwarding opt-in, canary tests |
| SEC-05 Trusted identity and state ownership | High / security and API | SA-06, SA-09 | #1445, #2362, #2364 | Custom-name/casing matrix and two-tenant store/cache/memory/Replay/learning tests |
| SEC-06 Content-free diagnostics | High / security | SA-07 | #2464 | INFO/DEBUG/TRACE content canaries and approved helper/static check |
| NET-01 Dashboard outbound fetch | High / security | CC-01, XR-SEC-002 | #1388 | Address, redirect, resolution, proxy, expansion, and limit suite through one service |
| WEB-01 Browser session transport | High / security | CC-02 | #2465 | Cookie-only flows, no reusable URL/storage token, scoped tickets, CSRF tests |
| WEB-02 Route-bound authorization | High / security and API | CC-03 | #2466 | Exhaustive route/method mapping, commit-boundary live-session reauthorization for privileged writes, and independent Replay/feedback role tests |
| WEB-03 Password lifecycle | High / security and interoperability | CC-14 | Audit PR and #2482 | Shared NIST-aligned policy, CAS login/session replacement, bounded and uniform verification, private auth storage, fail-closed startup, Chrome form and well-known-path contract tests; transactional audit delivery remains #2482 |
| ML-01 Sidecar and artifact boundary | High / security and resource | CC-04, XR-SEC-003 | #2467 | Shared-root integration, containment, authenticated transport, mount/network contract |
| ML-02 Job and upload isolation | High / resource and security | CC-05, XR-SEC-003 | #2467 | Body budgets, random isolated jobs, output containment, concurrency/cancel/cleanup tests |
| OC-01 OpenClaw provisioning | High / supply chain | CC-06 | #2468 | Catalog-only skills, path containment, image provenance, reduced privilege |
| API-01 OpenAI adapter parity | Medium / API | SA-10 | #2358 | SDK differential/round-trip fixtures, strict unions, escaped IDs |
| API-02 External RAG bounds | Medium / API and security | SA-08 | #2478 | Typed substitution, exact byte limits, malformed/oversized fixtures |
| CFG-01 Transactional config writer | High / integrity | CC-07 | #2326 and TD046 | CAS, cross-process lock, fsync/rename, rollback fault injection, race tests |
| CFG-02 Version/schema/rule convergence | High / contract | CC-08, CC-09, CC-10 | #2469, #2122, #2355, and TD046 | Shared corpus across Go, CLI, DSL, dashboard, dynamic CRD, operator; strict pre-rollout admission |
| CLI-01 Runtime state path | Low / CLI | CC-11 | #2479 | Normalization, containment, atomic private writes |
| PERF-02 Validation snapshot | Medium / performance | CC-12 | TD046 | One manifest read per KB and scaling benchmark |
| ARCH-01 Control-plane hotspots | Medium / architecture | CC-13 | Existing indexed debt, especially TD006, TD020, TD046 | Extraction-first boundaries and structure ownership |
| RT-01 Runtime generation and classifier snapshot | High / resource | RR-01, RR-02 | #2470, TD045, and #2396 | Constructor fault matrix, stream/reload race, exact cleanup, bounded graceful shutdown |
| RT-02 Looper workflow and I/O budgets | High / resource and performance | RR-03, RR-04 | #2471, #1456, and #2336 | Cross-request resume, stable clients/goroutines, byte limits, streaming/cancellation fixtures |
| RT-03 Replay actor and retention | High / data integrity and resource | RR-05, RR-06 | #2472, coordinated with #1146 and #2157 | Admission-close-drain barriers, durable acknowledgment, TTL conformance, cursor pagination, total-size cap |
| RT-04 Cache cancellation and result value | High / resource and correctness | RR-07 | #2473 | Per-request scores, miss reset, cancellation propagation, constructor cleanup |
| RT-05 Vector lifecycle and consistency | High / resource and integrity | RR-08, RR-09 | #2474 | Stop deadline, bounded batching/status, side-effect fault matrix, restart reconciliation |
| RT-06 Selection ownership and cardinality | High / resource | RR-10, RR-11 | #2222 and #2396 | No unused allocations, concurrent lease/load/close, bounded soak and persistence |
| RT-07 Memory tracking actor | High / resource | RR-12 | #2339 | Bounded slow-backend soak, atomic/coalesced updates, owned/borrowed exact-close tests |
| RT-08 Native generation and lifecycle | High and medium / native | RR-13, RR-14, RR-15 | #2396, #2477, and TD042 | Init failure/retry/path tests, bounded queue/unload, allocator instrumentation, classify/reload/close sanitizers |
| RT-09 Search and timer lifecycle | High and medium / performance/resource | RR-16, RR-17 | #2475 and TD045 | Positive-duration validation, concurrent close race, 100k search profiles |
| RT-10 OpenClaw collaboration lifecycle | High / correctness and resource | RR-18 | Reopened #1521 | Single-owner send/close, idempotent disconnect, focused repeated race, full handler race, SSE continuity and reclamation |
| DEP-01 Vulnerability gate | High / supply chain | XR-DEP-001 | #2476 | Patched dependencies, lock/toolchain pins, ecosystem scans, expiring exceptions |
| TEST-01 ONNX compile lane | High / testing/native | XR-TEST-001 | #2477 and #2396 | Model-free `go test` plus selected runtime receipts |
| TEST-02 E2E acceptance floors | High / testing | XR-TEST-002 | #2379 | Role inventory and deterministic non-zero acceptance contracts |
| TEST-03 Local feature-gate integrity | High / testing | XR-TEST-003 | Audit PR | Forced build/stop-failure propagation, repo-owned CLI resolution, and successful post-fix feature smoke |
| PERF-01 Numeric regression gate | High / performance/testing | XR-PERF-001 | #2455 | Repaired compare path and required PR-visible numeric result |

## Raw finding-to-terminal mapping

This compact ledger is the completeness check for the four audit lanes.

| Lane | Complete mapping |
| --- | --- |
| Security and API | SA-01→SEC-01; SA-02→SEC-02; SA-03→SEC-03; SA-04→SEC-04; SA-05→SEC-01; SA-06→SEC-05; SA-07→SEC-06; SA-08→API-02; SA-09→SEC-05; SA-10→API-01 |
| Resource and runtime | RR-01→RT-01; RR-02→RT-01; RR-03→RT-02; RR-04→RT-02; RR-05→RT-03; RR-06→RT-03; RR-07→RT-04; RR-08→RT-05; RR-09→RT-05; RR-10→RT-06; RR-11→RT-06; RR-12→RT-07; RR-13→RT-08; RR-14→RT-08; RR-15→RT-08; RR-16→RT-09; RR-17→RT-09; RR-18→RT-10 |
| Control plane and contract | CC-01→NET-01; CC-02→WEB-01; CC-03→WEB-02; CC-04→ML-01; CC-05→ML-02; CC-06→OC-01; CC-07→CFG-01; CC-08→CFG-02; CC-09→CFG-02; CC-10→CFG-02; CC-11→CLI-01; CC-12→PERF-02; CC-13→ARCH-01; CC-14→WEB-03 |
| Cross-repository | XR-SEC-001→SEC-03; XR-SEC-002→NET-01; XR-SEC-003→ML-01/ML-02; XR-DEP-001→DEP-01; XR-TEST-001→TEST-01; XR-TEST-002→TEST-02; XR-TEST-003→TEST-03; XR-PERF-001→PERF-01 |

Publication and closure check: every raw finding now maps to an applied issue
or indexed debt record with priority metadata. Attach the final proof pull
request and green CPU, AMD, and CI receipts before closing #2375. A parent
comment is not terminal ownership by itself.

## Recommended remediation waves

1. **Immediate proof and containment:** finish SEC-03; execute the already
   assigned SEC-01 and SEC-02 terminal records; apply SEC-04
   branch-independent credential hygiene.
2. **Tenant and security plane:** SEC-05, SEC-06, NET-01, WEB-01, WEB-02,
   WEB-03, ML-01, ML-02, and OC-01.
3. **Runtime ownership:** RT-01, RT-03, RT-04, RT-05, RT-07, and RT-10
   establish safe close and drain semantics before more backends are added.
4. **Contract convergence:** CFG-01, CFG-02, API-01, API-02, operator admission,
   and the shared contract corpus.
5. **Bounded algorithms and native lifecycle:** RT-02, RT-06, RT-08, RT-09,
   and PERF-02.
6. **Permanent gates:** DEP-01, TEST-01, TEST-02, TEST-03, PERF-01, and
   structural debt checks.

## Looper proof-of-fix boundary

The #1443 proof is complete only if all of these defensive properties hold:

- A 256-bit credential belongs to server runtime lifecycle, not serialized
  router configuration or a general runtime registry.
- The same effective credential survives config reload. Multi-replica reentry
  uses an explicit deployment-secret value shared by replicas; malformed or
  empty configured values fail startup.
- The client snapshots configured headers, ignores all reserved internal names,
  applies runtime-owned metadata last, bypasses ambient HTTP proxies, and does
  not replay the privileged request across redirects.
- Extproc validates before skip or privileged body processing, rejects missing,
  invalid, partial, duplicate, unknown, or malformed reserved metadata with a
  generic response, and projects only validated values to dedicated context.
- Reserved metadata is absent from generic headers, trace extraction, config
  export, logs, Replay, and provider traffic.
- The iteration contract remains compatible with internal self-verification
  calls: zero or absent external metadata does not become a malformed positive
  iteration.
- Tests cover entropy failure, same-secret and cross-secret validation,
  configured-header precedence, duplicates and casing, reload continuity,
  standalone and shared-secret topologies, no-leak canaries, public-listener
  negative behavior, and genuine Looper reentry.

The proof does not close the separate Looper byte, fan-out, workflow state, or
decision/model membership work in RT-02.

## Gate recommendations

| Check | Repo-native placement | Findings protected |
| --- | --- | --- |
| Harness, schema, and structure validation | `make agent-validate`; extend structure rules only for durable boundaries | ARCH-01, CFG-02 |
| Changed-surface lint | `make agent-lint CHANGED_FILES="..."` | Every proof and follow-up PR |
| Core compile and unit gate | `make agent-ci-gate CHANGED_FILES="..."` | Proof PR and affected packages |
| Local PR reproduction | `make agent-pr-gate` | Baseline PR requirements |
| Feature/profile routing | `make agent-feature-gate ENV=<target> CHANGED_FILES="..."` and `make agent-e2e-affected CHANGED_FILES="..."`, where target is CPU or AMD | Deployment-visible security and runtime changes |
| Looper authenticity E2E | Public-listener missing/wrong-secret rejection plus a real internal reentry profile | SEC-03 |
| Route authorization inventory | Table-driven startup/CI contract enumerating every route, method, permission, and sensitivity | SEC-01, SEC-02, WEB-02 |
| Cross-language contract corpus | One corpus exercised by Go, Python CLI, DSL, dashboard, dynamic CRDs, and operator | CFG-02, API-01 |
| Race suites | Targeted `go test -race` for refresh/reload, Replay close/write, cache concurrency, vector stop, OpenClaw disconnect/fan-out, and config mutation | RT-01, RT-03, RT-04, RT-05, RT-10, CFG-01 |
| Fault and barrier tests | Constructor rollback, login versus password-reset CAS, config crash points, async writer drain, and multi-store compensation | WEB-03, RT-01, RT-03, RT-05, CFG-01 |
| Fuzz and adversarial units | Header ambiguity, JSON unions/templates, URL resolution/redirect, path/symlink containment, multipart limits | SEC-03, NET-01, API-01, API-02, ML-01, ML-02, OC-01 |
| Resource soak | Repeated reload, shutdown, and overload with goroutine, FD, connection, RSS, GPU-memory, queue, and acknowledged-write accounting | RT-01 through RT-09 |
| Native ABI/compile lane | Model-free Go/Rust compile and unit tests plus selected CPU/AMD receipts | RT-08, TEST-01, #2396 |
| Vulnerability gate | `govulncheck`, production npm audit, Cargo audit, and frozen-lock Python audit with owner/reason/expiry suppressions | DEP-01 |
| Acceptance threshold lint | Reject zero-only success floors in acceptance-tagged E2E; allow explicit report/benchmark role | TEST-02 |
| Local build/serve failure integrity | Forced image-build and stop-tool failure plus repo-owned CLI resolution in the canonical feature workflow | TEST-03 |
| Numeric performance comparison | Repair and require the #2455 compare result on affected benchmark paths | PERF-01 |
| Sensitive-content canaries | Logger, Replay, response, and provider capture tests; optional structural lint for direct content logging | SEC-04, SEC-06 |

For each follow-up, start with the smallest targeted test, then run
`agent-validate`, `agent-lint`, `agent-ci-gate`, the task-matrix-selected
feature and E2E gates, and finally `agent-pr-gate` and pull-request CI. Fix and
rerun failures to the applicable completion boundary.

## Validation receipts and limitations

The following are audit-time receipts, not a claim that all remediation is
green:

- Baseline `make agent-validate` and `make agent-scorecard` passed.
- The audit branch subsequently passed `make agent-validate`, changed-surface
  `make agent-lint CHANGED_FILES="..."`, and
  `make agent-ci-gate CHANGED_FILES="..."`.
- `go vet ./...` passed in router, dashboard backend, operator, E2E, Candle,
  ML, NLP, and OpenVINO modules.
- Focused config, DSL, Kubernetes, operator, auth, router, Replay, Responses,
  and Anthropic tests passed as recorded during the audit.
- Replay race tests passed, but they do not yet exercise the required
  close/admission barriers.
- Dashboard authentication and router race suites passed after the password
  lifecycle proof. The full handler race suite deterministically fails the
  OpenClaw WebSocket disconnect/fan-out contract; the focused failure is RT-10
  and was attached to reopened #1521 rather than waived.
- ML Rust tests passed. The NLP Rust crate currently contains no unit tests.
- ONNX Go tests fail to compile against current APIs. This is TEST-01, not an
  environment waiver.
- Native-linked Looper and extproc suites passed after the normal native build,
  including focused authentication/reentry integration tests and repeated race
  runs. Broader feature, affected-E2E, platform, and final PR receipts remain
  separate completion evidence.
- The complete dashboard backend suite passed with the repository-owned Python
  environment, including handlers that invoke the configuration templates. The
  ML pipeline package has no Go tests, which is itself part of the coverage
  finding.
- The locally available static analyzer could not analyze the Go 1.25 modules;
  it is not listed as a successful receipt.
- Dependency results are point-in-time. Unpinned Python requirements do not
  prove a reproducible deployed set; committed locks and built images are the
  authoritative targets.
- CPU feature/dev/smoke, affected E2E, the local PR gate, AMD regression, and
  final pull-request CI remain pending until their final post-review receipts
  are attached. Public evidence must contain commands, outcomes, and safe
  artifact references, never credentials or private host details.

## Definition of done for #2375

Issue #2375 can close only when all of the following are true:

1. This audit covers every row in the coverage matrix and links the owning
   source or contract surface without unnecessary exploit detail.
2. The raw-to-terminal mapping has no missing ID. Every package points to a
   merged pull request, open issue, indexed debt entry, or reviewed accepted-risk
   record.
3. Every critical and high-priority class has severity, owner surface,
   acceptance tests, repo-native validation, and milestone or priority metadata.
4. The #1443 proof passes hot-reload continuity, cross-replica compatibility,
   no-leak guarantees, focused unit/race tests, a deterministic public-listener
   negative E2E, and a genuine internal Looper positive regression.
5. The API/weak-type and lifecycle inventories remain explicit about
   keep/replace/adapter and ownership/cancellation dispositions.
6. Positive controls and ruled-out concerns remain recorded so follow-ups do
   not duplicate sound behavior.
7. Every recommended static or dynamic check is implemented in an existing
   harness gate or assigned to a numbered follow-up issue.
8. CPU validation, affected native tests, AMD end-to-end regression, and final
   pull-request CI are green, with failed runs and successful reruns preserved
   in the non-public evidence ledger.
9. The parent receives a concise closure comment linking this audit, the proof
   pull request, the child-issue table, a non-sensitive AMD evidence summary,
   and the final CI result.

Until all nine conditions hold, this document is a closure plan and evidence
index—not a declaration that the repository-wide risks are fixed.
