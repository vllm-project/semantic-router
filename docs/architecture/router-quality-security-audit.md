# Router Quality, Security, Resource-Safety, and API-Boundary Audit

## Document status

This reference records the repository-wide audit requested by
[#2375](https://github.com/vllm-project/semantic-router/issues/2375). It is a
maintainer and contributor index for understanding the identified risk classes,
their owning source surfaces, and the evidence required to close them.

- **Initial audited base:** `63bf27bf303a2ad78b88bb4be27e437af28652d0`
- **Final integration base:** `fe0b37d09ae0c113f8f7668656d837f2473b436a`
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

The audit identified four critical access-control, network-exposure, or
credential-exposure
gaps, a broad group of
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
| CLI | Pydantic contract, migration/version behavior, runtime paths, host publication, generated Envoy, local stores | Input strictness differs from Go; one low-risk path gap; internal ports were wildcard-published; local stores share weak credential/network boundaries; edge stripping is defense in depth only |
| Dashboard | Authentication/RBAC, router proxy, browser credential transport, public demo access, outbound fetches, config mutation, ML pipeline, OpenClaw | Shared credential, official bearer transport, and the WebSocket disconnect/fan-out race are closed by the proof; egress, fail-open route authorization, embedded-origin isolation, file/job/image boundaries, cross-process config transactions, and graceful WebSocket delivery/reclamation remain terminally owned |
| Kubernetes and operator | Dynamic CRDs, converter, operator CRD/webhook/controllers, canonical generation | Raw preserved routing can bypass strict core admission; legacy/canonical ownership and boolean semantics drift |
| Native bindings | Candle, ONNX, ML, NLP, OpenVINO Go/Rust lifecycle seams | Candle/ONNX/NLP lifecycle and concurrency gaps; lower ML wrappers are sound; ONNX compile drift, silent input truncation, and image-boundary divergence entered the proof loop |
| E2E, testing, performance | Acceptance floors, native compile lanes, numeric comparison, affected-profile selection, local image/serve orchestration | Report-only zero floors, missing ONNX compile coverage, non-gating numeric comparison, a baseline-profile short circuit that hid affected profiles, and local feature-gate false-green paths |
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

The canonical Kubernetes CLI path serializes each release's deploy and teardown
mutations with a renewable, compare-and-swap Kubernetes Lease. It reuses the
currently referenced immutable Secret only when the complete encoded key set is
unchanged; any credential change creates a fresh release-scoped immutable
generation through standard input. The first generation gets a shared Looper
key when the operator did not supply one, and later generations carry only that
key unless it is explicitly rotated. Every retained Helm revision protects its
referenced Secret from garbage collection, and credential-bearing revisions
use non-overlapping `Recreate` rollouts.

Teardown stamps every observed generation with a bounded 15-minute no-delete
deadline before uninstall. It releases the Lease while waiting, so an emergency
redeploy remains available, then reacquires the Lease, reloads lifecycle state,
and rechecks live references before each deletion. A later deploy can safely
adopt an exact quarantined generation. If any release reappears during the wait,
the entire pre-uninstall batch is deferred so both its live workload and retained
Helm revision history remain valid; only a still-absent release enters the final
per-Secret reference check. Creation and quarantine timestamps use a strict RFC
3339 grammar, future-skew is bounded, and cleanup fails closed on malformed
metadata or whenever current/rollback references cannot be proved.

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

The final-main delta review found the analogous routing-provenance failure at
the edge: several checked-in ExtProc examples could continue after processor
failure while route matches consumed the caller-settable `x-selected-model`
header. The proof change makes every configurable checked-in ExtProc path
explicitly fail closed and documents Agentgateway's fail-closed default. A
deployment that deliberately chooses fail-open still needs a proven
pre-routing strip-and-server-rebuild boundary; a route-level header modifier
runs too late to establish that provenance.

The same review found a provider-side confused-deputy boundary in the new
remote embedding surface: a configurable environment-variable name could turn
an endpoint into an oracle for unrelated process secrets. The proof accepts
only `VLLM_SR_EMBEDDING_API_KEY`, requires HTTPS when it is selected, rejects
userinfo/query/fragment ambiguity without echoing the URL, refuses redirects
and ambient proxies, bounds retries and response allocation, and never returns
provider bodies. Dashboard `config.write` alone no longer reaches any handler
that synchronously applies runtime configuration; `config.deploy` is the
explicit provider-credential use/disclosure authority and is required for
setup activation and every config apply path. Generic provider
`backend_refs[].api_key_env` remains a broader typed-binding problem owned by
the terminal issue below rather than being misrepresented as closed here.

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

The final-main documentation delta also contained copyable examples that
described full prompt/response capture as an ordinary learning workflow. The
proof change replaces those defaults with bounded content-free metadata and
cost records and confines payload capture to an exceptional, access-controlled,
short-lived incident mode. This closes the documentation-specific delta; the
runtime-wide canary inventory remains open below.

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

The browser UI copied a reusable bearer credential into script-visible storage
and protected iframe, EventSource, and WebSocket URLs even though the backend
already issued an HttpOnly session cookie. Embedded reverse proxies also
forwarded the dashboard bearer, session cookie, query token, and token-bearing
Referer to upstream services and allowed those services to overwrite the
dashboard cookie namespace. Their response CORS policy reflected arbitrary
origins with credentials, so a hostile same-site sibling could read protected
HTTP responses even though `SameSite=Lax` was set.

The final authentication review additionally found that the OpenClaw room
WebSocket upgrader accepted every `Origin`. `SameSite=Lax` does not prevent a
same-site sibling origin from carrying the dashboard host cookie, so this was a
cross-origin WebSocket hijacking boundary rather than a CORS preference. The
proof adds one strict same-origin validator to authentication middleware for
every protected WebSocket upgrade and reuses it at the room upgrader. It
canonicalizes HTTP(S) scheme, authority, default ports, IPv6, TLS, and one
well-formed forwarded scheme; it rejects missing, `null`, malformed, repeated,
path-bearing, userinfo, sibling-host, scheme-mismatched, and ambiguous proxy
origins before the protected handler runs.

The proof now makes the maintained browser client cookie-only: it purges the
legacy storage key, never injects the dashboard bearer or rewrites browser
URLs, restores sessions through `/api/auth/me`, and uses a credential-free
revision only to reject stale `401` events. Maintained login, bootstrap, and
password-change requests explicitly negotiate a cookie-only response whose
JSON omits the JWT; cookie-only is also the default, while metadata-free
non-browser clients explicitly request bearer mode. `/api/auth/me` replaces the exact validated legacy JavaScript
cookie with HttpOnly/SameSite/Secure attributes and the JWT's remaining expiry
without minting a session. A shared canonical-origin seam
protects WebSockets, credentialed CORS, and unsafe cookie-authenticated
HTTP methods; bearer-authenticated non-browser clients retain their explicit
API contract. Reverse proxies strip every dashboard credential and Referer,
preserve an explicitly opted-in Router API credential only when dashboard
bearer authentication was not used, filter dashboard `Set-Cookie`, and bound
raw WebSocket handshakes by time and bytes. Dashboard responses use
`Referrer-Policy: no-referrer`. Embedded responses cannot widen Service Worker
scope; every independent upstream CSP policy, including comma-combined policy
lists, is rewritten to `frame-ancestors 'self'` and receives
`worker-src 'none'` as defense in depth. Cookie logout, including a request
where `SameSite` withholds the cookie, requires same-origin browser evidence;
metadata-free bearer API logout remains compatible. That CSP does not replace
origin isolation. The raw WebSocket path constructs one escaped origin-form
request target, rejects decoded request-line and header controls before dialing
or hijacking, regenerates one canonical `Connection`/`Upgrade` handshake pair,
strips connection-nominated and framing headers, and forbids
static `Host` or handshake-key overrides. HTTPS upstreams use verified system
roots, SNI/hostname validation, TLS 1.2 or newer, context-aware dialing, and
IPv6-safe authority construction.

Required invariant: cookie-only normal browser sessions, no reusable
credential in URLs or script storage, CSRF protection for unsafe methods, and
one-time audience/path-scoped tickets only for a transport that cannot use the
cookie. Every cookie-authenticated WebSocket must be strict same-origin at the
shared auth boundary; non-browser origin omission needs an explicit reviewed
policy rather than a silent exemption. The audit pull request owns the official
cookie-only client, cross-origin HTTP/WebSocket, and proxy credential-isolation
proofs. Same-origin embedded upstream HTML still needs a separately hosted or
otherwise capability-isolated origin so a compromised embedded application
cannot act with the ambient dashboard cookie; that remaining boundary is owned
by [#2465](https://github.com/vllm-project/semantic-router/issues/2465).

#### WEB-02 — Fail-closed route authorization

Dashboard route registration and permission lookup are separate. Unknown
protected routes can inherit a read permission, and advertised Replay and
feedback permissions are not the actual gate for their proxied operations.
Middleware also authorizes at request entry; outside the auth-store account
mutations fixed by this proof, a slow in-flight settings, router-deploy, or MCP
write can retain that snapshot after its session is revoked.

The final browser review found one narrower fail-open path in the public-route
classifier: login, logout, bootstrap, and setup-state exemptions used string
prefixes, so an unknown suffixed `/api/` path could bypass authentication and
fall through to another handler or proxy. The proof makes every public auth and
setup exemption exact (with only the deliberately registered trailing slash),
and suffix regression tests cover both normal auth and the unavailable-auth
guard. The broader method/permission registry remains open below.

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
Every response reached through authentication middleware, including success,
authorization failure, handler error, and streaming paths, is non-cacheable;
public fingerprinted assets retain their separate cache policy. Session
issuance transactionally caps each user at 16 live sessions and 64 recent rows,
revoking the oldest live entries before pruning bounded inactive history.

The local container startup path treats dashboard authentication as a
dashboard-only capability. JWT and bootstrap-password values are inherited by
the dashboard container process without entering Docker argv or debug logs,
are removed from Router, Envoy, support-service, and Kubernetes environments,
and flow through one immutable preflight plan. A configured password blocklist
must resolve to an existing regular file before any runtime mutation and is
mounted read-only at a fixed container path.

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
maintained browser's script-visible bearer, query-token, and CSRF boundary
is closed under WEB-01; embedded-origin isolation remains with
[#2465](https://github.com/vllm-project/semantic-router/issues/2465).
Credential-lifecycle audit write failures are now content-free and
observable, while transactionally durable event/outbox delivery remains owned
by [#2482](https://github.com/vllm-project/semantic-router/issues/2482).

#### WEB-04 — Public and shared demo credentials

The final-main refresh exposed a reusable Playground username/password pair in
the public root README. A credential published in a public repository is
compromised by definition; browser breached-credential warnings are expected,
and deleting the current text or Git history cannot make the old value safe
again. A shared administrator password also provides no individual identity,
revocation, or attributable audit boundary.

The proof removes the credential block and adds a canonical validation scan of
every Git-visible Markdown, MDX, and HTML source document. The scanner detects
nearby reusable identity/password pairs in assignments, inline fields,
Markdown tables, and HTML tables or definition lists; it reports locations but
never values. Its maximum value length matches the dashboard's 1024-code-point
password limit; reserved email domains and credential-looking prefixes receive
no heuristic exemption, while only exact or syntactically explicit placeholders
are ignored. Pairing uses bounded line windows rather than a document-wide
Cartesian product. The gate intentionally excludes derived output and test
fixtures and does not claim to scan Git history, binaries, generated artifacts,
lone secrets, or arbitrary configuration formats. Copyable Milvus examples now
require shell-provided values rather than literal `YOUR_*` credentials. The
deployed value was rotated outside Git, its existing sessions were revoked, and
the live auth database and backup permissions were tightened. No replacement
secret belongs in this document, a pull request, CI output, or an issue. A
public demonstration that needs anonymous access must use a deliberately
capability-limited guest mode; authenticated access must use per-user or
short-lived, audience-scoped credentials.

Required invariant: no reusable secret is committed or published; any exposed
value is immediately rotated and every derived session is revoked; recovery
material remains in an approved secret manager; and public demo access cannot
carry administrator or general control-plane authority. The repository and
live-rotation proof are owned by the audit pull request.

#### NET-02 — Local container host-publication boundary

The split-runtime CLI passed two-field Docker publish specifications for every
service. Docker interprets `host-port:container-port` as a wildcard host bind,
so dashboard, router control/extproc/metrics, model, data-store, and
observability sockets were reachable beyond loopback. The same behavior made
an Envoy listener configured for `127.0.0.1` public at the host boundary. A TLS
reverse proxy therefore did not establish a mandatory browser-authentication
edge, and unauthenticated internal services could be reached directly. This is
raw finding CC-15.

The proof makes internal, control-plane, data, and observability publications
loopback-only by default, retains container-network service discovery, and
requires a strictly validated explicit override before they can bind another
host address. Envoy is different: its host publication follows the configured
listener address, so a wildcard remains possible only when the listener itself
requests one, while the split Envoy process listens on the container network
rather than accidentally binding only container loopback. One immutable plan
validates the internal bind address, every listener address, every fixed stack
port, and every offset-derived host port before old containers, networks, files,
or support services are changed. It also rejects fixed-service/listener host
bind overlap, duplicate split-listener container ports, and collision with the
fixed Envoy admin listener; distinct concrete host addresses remain valid.
Command-contract tests cover defaults, offsets, opt-in, invalid values,
collisions, restart preservation, and loopback versus wildcard listeners. A
non-public live receipt also applies a boot-persistent public-interface deny
boundary without recording private infrastructure here.

Required invariant: internal services are never host-wildcard-published by
default; a loopback listener stays loopback after container publication;
explicit exposure is validated and observable; the intended gateway remains
healthy; and direct public access cannot bypass the dashboard TLS edge.
Repository defaults and deployment proof are owned by the audit pull request;
SEC-01 remains defense in depth for router management authorization.

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

The final-main image-input review additionally found that one Candle FFI path
collapsed malformed client images and internal model failures into the same
status, the API treated every encode failure as caller-caused, and compressed
images had no decoded-dimension or allocation budget. The proof change adds a
typed invalid-input status and Go sentinel, keeps internal failures as generic
`5xx` responses, and bounds source dimensions, pixel area, and decoder
allocation before model lookup. Classify and eval request parsing also rejects
malformed allowlisted base64 images instead of returning a placeholder success.
Direct native image input accepts only the declared JPEG/PNG capability and
rejects decoded GIF/WebP even when the caller bypasses the HTTP parser.

The same admission review bounds embeddings and similarity input counts,
per-string bytes and code points, aggregate request bytes, and raw JSON Unicode;
it rejects empty, NUL-containing, invalid UTF-8, and unpaired-surrogate input.
One small process admission gate is process-global across embeddings,
similarity, classify/eval/combined work, and ExtProc signal plus local-selection
evaluation and fails overload immediately with a non-cacheable `503` plus
`Retry-After`. Classify/eval acquire before base64 or pixel decode; text work
also acquires whenever the effective runtime backend is local Candle/OpenVINO,
including `EMBEDDING_BACKEND_OVERRIDE` over a remote-configured provider, while
pure remote/placeholder text bypasses the scarce native budget. ExtProc uses
the same singleton and fixed overload response. Its OpenAI/Anthropic fast
extractors preserve inline-image presence and invalid state, so empty,
malformed, or unsupported inline images return a fixed non-cacheable `400`
instead of disappearing into a default-model route; valid remote/file inputs
and text-only requests retain their existing contract.

Runtime preparation and classification execution now consume the same resolved
`RuntimePlan`, including backend, canonical model type, and local model path.
That override-first contract covers embedding rules, reask, complexity query
and preload, contrastive preference, contrastive jailbreak, and knowledge-base
query and preload. The same construction-time plan is stored on ExtProc and
propagated to tool databases, request-time tool filtering, cached tool
registries, model selection, admission, runtime probes, BERT feature gates,
and the model-download manifest. No request path re-reads environment
overrides into a second effective plan. Remote plans validate their endpoint
capability before publication and skip local assets; remote-to-local plans
avoid remote clients and download only the selected local embedding model.
Semantic cache, memory, and non-LlamaStack vector-store implementations still
call Candle directly, so they retain a separately derived, required union of
their configured local assets instead of being accidentally disabled by a
remote or OpenVINO classification plan. Shared consumer models initialize once;
explicit local consumer models without a configured path fail during runtime
construction. Llama Stack remains the owner of its own vector-store embedder.
`EMBEDDING_MODEL_TYPE_OVERRIDE` is the preferred explicit model selector; the
deployed `EMBEDDING_MODEL_OVERRIDE` Helm/E2E contract remains a
lower-precedence compatibility alias. Switching a remote-configured provider
to a local backend requires either an explicit supported model with its
configured path or exactly one configured local model path. Missing,
ambiguous, unsupported, or unusable remote plans fail construction before
inference.
OpenVINO accepts one canonical text-embedding plan per classifier generation,
initializes it once, and rejects conflicting family plans before a second model
is built. An external, non-contrastive preference classifier does not initialize
an embedding runtime it cannot use.

Knowledge-base preload stages every exemplar and publishes the result only
after all embeddings are non-empty and successful. A partial failure publishes
nothing, reports aggregate counts without provider or exemplar detail, and can
be retried. Model-backed text failures propagate as a typed signal-evaluation
error rather than becoming an empty/default route; API and ExtProc return fixed,
non-cacheable client-safe failures. Standalone API-server auto-discovery also
returns its classifier build or initialization error instead of replacing a
configured failure with a placeholder `200` service.

Remote OpenAI-compatible embedding responses are read through a
batch-and-dimension-derived byte budget with a 32 MiB absolute ceiling before
JSON allocation. Content-Length and streamed overflow, trailing JSON values,
cardinality or dimension drift, non-finite numbers, and values outside the
float32 range fail closed without echoing provider response bodies. This keeps
a compromised or misconfigured provider from turning a routing probe into an
unbounded allocation or poisoning downstream similarity indexes with infinity.
Provider construction also enforces the same bounded HTTP(S), timeout, retry,
and dimension contract in Go, the CLI schema, the dashboard structured editor,
and the generated operator CRD. Retry delay saturates without integer overflow;
response indexes distinguish an omitted value from an explicit zero; bounded
error bodies are drained when safe so keep-alive remains usable, but are never
returned as public error text.

Tokenizers disable truncation and padding once at model initialization, assert
that contract at inference, and reuse one encoding/tensor construction for
mmBERT and multimodal batches instead of cloning or tokenizing twice per
request. A typed native token-limit status crosses Rust and Go and becomes a
fixed content-free `413 EMBEDDING_INPUT_TOO_LARGE` response on all embedding,
similarity, and batch-classification routes. Those input/error/resource
contracts are closed by the audit pull request. They are construction-time and
within-generation guarantees, not a config-refresh lease: atomic runtime/config
generation swap and retirement remain with RT-01 and
[#2470](https://github.com/vllm-project/semantic-router/issues/2470), while
native handle ownership and unload remain with the owners below.

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

The race runtime initially observed `WSClient.close()` closing a client channel
while the room collaboration event bus concurrently sent to it. The proof now
routes every producer through one non-blocking `enqueue` seam and serializes
enqueue admission with idempotent close, unregister, socket close, and channel
close under one ownership lock. A deterministic close/fan-out barrier, repeated
race runs, the full handler race suite, survivor-WebSocket delivery, and SSE
continuity all pass; send-after-close panic and the data race are closed here.

Required invariant: one owner controls send admission and channel close;
disconnect is idempotent; fan-out cannot send after close admission begins;
backpressure and client removal are bounded; and WebSocket teardown cannot
interrupt SSE delivery or leak goroutines. Graceful close-frame ordering,
explicit delivery/backpressure semantics for a full client buffer, and empty
room-map reclamation remain under the exact historical tracker
[#1521](https://github.com/vllm-project/semantic-router/issues/1521); they do not
invalidate the merged single-owner race proof.

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

The final handler regression also found that synchronous Docker status and log
commands could hang a configuration request indefinitely, treat daemon failure
as container absence, report success for a stopped Envoy without applying its
configuration, and let setup activation persist after runtime failure. The
proof adds context and output limits to status, logs, lifecycle, Python sync,
and OpenClaw runtime commands; keeps stdout/stderr status semantics distinct;
fails mutation on unknown or present-but-not-running managed containers; and
restores the previous setup/config bytes on apply failure. This is a bounded
request-path proof, not a replacement for the cross-process transaction owner.

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

The final-main documentation and website delta is narrower: the OpenCode guide
listed removed selection algorithms and the Chinese homepage retained a stale
positioning message after the English source changed. The audit pull request
aligns the guide with the actual algorithm registry and adds an executable
English-to-Chinese hero-copy contract. Broader multi-surface schema convergence
remains owned above.

#### CLI-01 — Runtime state path containment

One CLI runtime-state filename is derived from operator-controlled identity
without one canonical containment and atomic private-write boundary. The
finding is narrow, but runtime state must never escape its owned root or become
partially visible to another local process.

Required invariant: normalize the identity to a bounded filename, verify the
resolved path remains under the state root, and publish private state through
an atomic same-directory write. Terminal owner:
[#2479](https://github.com/vllm-project/semantic-router/issues/2479).

#### CLI-02 — Local storage credential and network lifecycle

The split runtime creates Postgres with one repository-known password and
starts Redis without authentication, then injects matching defaults into local
store configuration. Loopback host publication prevents a public socket but
does not authenticate east-west callers: auxiliary or user-selected workloads
must not gain storage authority merely by joining the application network.
This is raw finding CC-16.

Required invariant: independent CSPRNG credentials per stack; atomic private
secret state; authenticated Postgres and Redis clients; no secret in command,
log, report, or public config output; explicit migration and rotation; and a
data network that unrelated workloads do not join. Terminal owner:
[#2485](https://github.com/vllm-project/semantic-router/issues/2485).

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
advisories, and floating patch-level toolchains. The audit proof patches the
Dashboard's resolved runtime dependency advisories, upgrades its build chain,
pins Dashboard and pre-commit builders and CI to the supported Node 24 LTS
line, and makes the Dashboard workflow reject moderate-or-higher deployed
frontend findings. The repository still lacks one required multi-ecosystem,
lock-aware policy with reviewed expiry for build-only or unavoidable
exceptions.

Required invariant: patched and consistently pinned dependencies/toolchains;
per-ecosystem scans against committed resolution artifacts; separate deployed
runtime and build-only classification; and machine-readable exceptions with
owner, reason, exposure, and expiry. Terminal owner:
[#2476](https://github.com/vllm-project/semantic-router/issues/2476).

#### TEST-01 — ONNX model-free compile contract

The ONNX Go tests targeted stale APIs and did not compile, while no mandatory
provider-independent CI lane caught the drift. The audit proof repairs the Go
surface and model-free suite, adds native typed-error tests, and makes each GPU
and ExtProc image copy the complete binding package rather than a hand-picked
file list.

Required invariant: `go test ./...` compiles and passes without model assets,
with model/provider receipts selected separately and tied to native capability
contracts. The proof is part of the audit pull request; permanent mandatory CI
ownership remains with
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
different CLI installations. Deployment review additionally found workflow-
driven integration teardown paths that still target fixed default container
names or ports even when the surrounding run uses another stack identity.

Required invariant: every image-build failure terminates the recipe before a
success message or later stage, and build, serve, stop, and smoke use the same
repo-owned CLI environment. Missing or failed stop tooling must not report a
successful cleanup. This proof is owned by the audit pull request and requires
an actual feature-gate rerun after the failure-path fix. Stack-aware,
ownership-checked integration teardown is terminally owned by
[#2487](https://github.com/vllm-project/semantic-router/issues/2487); until that
lands, shared-host validation must use unique resources, avoid the unsafe
targets, and verify pre/post state.

#### TEST-04 — Affected-profile selection integrity

The Kubernetes integration workflow previously returned only its baseline
profile set whenever a core path changed. That early return discarded
independently affected security and multimodal profiles, so broad Router,
APIServer, native FFI, or image-input changes could make CI exercise less of
the repository than a narrower change. Separately, the canonical map classified
`e2e/config/**` as a full-CI trigger while the workflow's common-E2E filter did
not consume it, so an isolated shared-config change selected no profile at all.

Required invariant: the CI matrix is a stable de-duplicated union of mandatory
baseline profiles and every standard profile selected by the canonical path
map; profile names and booleans are strictly validated; manual-only profiles
remain outside automatic CI by explicit policy; canonical common-E2E triggers
and workflow filters are validated as the same set; and selector tests cover
simultaneous core, E2E-common, ML, multimodal, APIServer, native, and image-path
changes. This proof is owned by the audit pull request.

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
- The Dashboard and Wizmap deployed dependency audits reported no finding after
  the audit-time Dashboard lock refresh; build-only findings remain classified
  under DEP-01 rather than being shipped in either runtime image.

## Final-main delta review

The branch was rebased onto the audited base above and that newly arrived diff
was reviewed as a separate lane. These rows are proof changes in the audit pull
request, not evidence that the broader packages named in the last column are
complete. They remain conditional on the same CPU, AMD, and pull-request gates
as the Looper proof.

| ID | Final-main finding | Proof change in this pull request | Broader owner that remains open |
| --- | --- | --- | --- |
| MD-01 | Checked-in Envoy and AI Gateway ExtProc examples could be implicit or explicit fail-open while route selection consumed `x-selected-model` | Explicit fail-closed settings; strip every client-selected-model casing/duplicate before context; force trusted reroutes to clear route cache; generated-config, parsed-manifest, unit, and deployed forgery/outage tests | SEC-04 / #2286 for branch-independent caller-header and provider-credential separation |
| MD-02 | Image FFI and API paths conflated malformed input with internal failures, exposed internal failure text, and did not bound decoded image resources | Typed Rust statuses and Go sentinels, fixed client-safe `4xx`/`5xx`, JPEG/PNG capability gate, decoded dimension/pixel/allocation budgets, native and API tests | RT-08 / #2396 and TD042 for generation, unload, and FFI module ownership |
| MD-03 | Classify and eval accepted shape-valid but malformed base64 image parts and could return a placeholder `200 OK` | Validate every selected allowlisted data URI, return a typed content-free `400 INVALID_IMAGE`, and cover intent, eval, and embeddings through route and deployed-API tests | RT-08 for the remaining native lifecycle rather than this closed input contract |
| MD-04 | English homepage positioning changed without the Chinese message following it | Align the Chinese translation and fail website lint when the paired message IDs diverge | CFG-02 / TD046 for broader cross-surface contract convergence |
| MD-05 | Copyable blog examples named rejected algorithms, normalized full-content capture, and recommended unsafe ExtProc fail-open behavior | Align algorithm names with the registries, make ordinary diagnostics content-free and bounded, constrain exceptional capture, and document fail-closed outage semantics | SEC-06 / #2464 for runtime-wide content canaries; CFG-02 for the broader algorithm/schema corpus |
| MD-06 | The public README published one reusable Playground credential pair | Remove the pair, add a repo-wide Git-visible source-document credential-pair gate, rotate the deployed credential out of band, revoke existing sessions, and tighten auth-file permissions | WEB-04; any future anonymous demo access needs a capability-limited guest design rather than another shared secret |
| MD-07 | Split-runtime Docker publications exposed internal services on wildcard host interfaces, ignored loopback Envoy listener intent, and deferred address/port collisions until after runtime mutation | Bind internal publications to loopback by default, make the validated override explicit, separate host intent from container-network binding, reject host/listener/admin collisions in immutable preflight, add command contracts, and retain a non-public live containment receipt | NET-02; SEC-01 / #2463 remains the management-authorization defense in depth |
| MD-08 | Core-path CI selection returned baseline profiles early and silently discarded independently affected standard profiles; canonical `e2e/config/**` full-CI changes selected nothing | Replace workflow shell branching with one tested strict selector that returns the stable baseline union affected set; connect and validate common-E2E triggers; expand canonical ownership for embedding APIServer, image, native FFI, and selected-model orchestration while preserving manual-only policy | TEST-04; TEST-02 / #2379 still owns acceptance-floor semantics within selected profiles |
| MD-09 | Final auth review found cacheable protected responses, unbounded per-user session rows, and local dashboard secrets flowing through a broad runtime environment | Apply protected-response no-store, bounded transactional session retention, and an immutable dashboard-only env/mount plan whose secrets avoid argv and non-dashboard services | WEB-03; WEB-01 / #2465 owns remaining embedded-origin isolation and #2482 owns durable audit delivery |
| MD-10 | Embedding surfaces admitted oversized batches/text and malformed raw Unicode, cloned tokenizers at request time, retokenized two batch paths, and could not distinguish native token-limit failures | Add shared bounded validation/admission, strict raw-Unicode checks, initialization-time tokenizer controls, single-encode tensor reuse, and typed native-to-HTTP `413` propagation | RT-08 / #2396 and TD042 still own native generation, scheduling, unload, and structural lifecycle work |
| MD-11 | Cookie-authenticated OpenClaw WebSockets accepted every browser origin, including hostile same-site sibling origins | Enforce strict scheme/authority origin equality for every protected WebSocket in shared auth middleware and at the room upgrader; reject missing, ambiguous, and malformed origin/proxy metadata with focused and upgrade tests | WEB-01; #2465 owns remaining embedded-origin isolation |
| MD-12 | The official browser stored/replayed and received a dashboard JWT in script-visible state, embedded proxies leaked it through headers, cookies, query and Referer, upstreams could overwrite the session cookie or widen Service Worker scope, credentialed CORS reflected same-site siblings, logout/CSP policy-list edges were not fail closed, and the raw WebSocket proxy serialized request-target/header data without a complete control-byte, hop-by-hop, TLS, IPv6-authority, or target-base-path boundary | Make the maintained client negotiate cookie-only JSON and purge legacy storage; reissue legacy cookies as HttpOnly without minting a session; enforce one canonical HTTP/WebSocket origin and unsafe-method/logout CSRF boundary; strip dashboard credentials at every proxy branch while retaining explicit auth-disabled Router API forwarding; filter session `Set-Cookie`/Service Worker scope; preserve every CSP enforcement policy with self-only framing and no workers; add no-referrer/bounded upgrade defenses; build one escaped/control-free upgrade request, regenerate canonical hop-by-hop headers, forbid framing/host/key overrides, use verified TLS 1.2+ context-aware IPv6-safe dialing, and join stripped HTTP/WebSocket paths against the same upstream base; cover unit, source-contract, real-upgrade, browser, normal and race paths | WEB-01 proof in this PR; #2465 owns separate-origin/capability isolation for same-origin embedded upstream HTML and any scoped OpenClaw browser ticket |
| MD-13 | Process admission covered embeddings/similarity but classify/eval/combined and ExtProc native paths could run outside the limit; capability checks and execution could disagree under a local override; text backend failures could become default routing; tools, selection, runtime probes, BERT feature gates, and model downloads could derive a second backend/model choice; invalid image-only parts could disappear; partial knowledge-base preload could publish inconsistent state; standalone classifier startup could hide a configured build failure behind a placeholder | Resolve one override-first `RuntimePlan` and propagate it across embedding, reask, complexity, contrastive preference/jailbreak, knowledge-base query/preload, ExtProc tools and selection, admission, model runtime, and model download; validate remote endpoint capability before publication; make local-to-remote use the selected provider and remote-to-local avoid every remote client and unused asset; fail closed on missing or ambiguous remote-to-local model selection and conflicting OpenVINO generation plans; keep the legacy model-override alias below the preferred explicit selector; use one process admission singleton across API and ExtProc; propagate typed text failures and fixed non-cacheable `400`/`500`/`503`; preserve invalid image presence; make KB preload atomic/retryable; fail standalone configured startup; skip unused embedding initialization for external preference; prove both override directions and provider/native execution parity with HTTP call counters | RT-08 / #2396 and TD042 still own native generation, unload, scheduling, and structure; RT-01/#2470 owns config-refresh generation leasing and retirement |
| MD-14 | Dashboard Docker/status/log and OpenClaw subprocesses could hang or allocate unbounded output; daemon failure masqueraded as absence; stopped Envoy and setup failure could be reported successful | Add deadline, wait, cancellation and aggregate output budgets; separate status stdout/stderr; distinguish absent/unknown/present-stopped; fail and restore config on runtime uncertainty; isolate unit tests from host Docker | CFG-01 proof in this PR; #2326 and TD046 still own cross-process CAS/fsync/generation transactions |
| MD-15 | Public dashboard auth/setup route exemptions used broad prefixes, so unknown suffixed API paths could bypass authentication and fall through to another route or proxy; runtime-applying config writes required only draft-write authority even though endpoint selection can disclose provider credentials | Replace prefix exemptions with exact registered paths and test unknown suffixes through both active-auth and unavailable-auth guards; require explicit `config.deploy` authority for setup activation and every synchronous runtime-config apply while keeping `config.write` draft-only | WEB-02 proof in this PR; #2466 still owns the exhaustive route/method/permission registry and commit-boundary reauthorization |
| MD-16 | OpenClaw room fan-out could send while disconnect closed the same client channel, producing a deterministic data race and possible send-on-closed panic | Funnel every producer through one non-blocking enqueue/close ownership lock; make unregister/close idempotent; add deterministic close/fan-out barriers, survivor-WS/SSE continuity, repeated race, full-handler race, and vet receipts | Audit PR closes the race; #1521 retains graceful close-frame, delivery/backpressure, and room-map reclamation QoS |
| MD-17 | ONNX silently truncated true tokenizer output, reported whitespace counts as token lengths, silently substituted mmBERT for unsupported embedding models and layers, allowed a lower early-exit artifact to masquerade as the full model, trusted malformed model-output and direct audio/image tensor shapes across the C ABI, and diverged from Candle public result types, options, and sentinels; ONNX multimodal output could accept arbitrary-rank or unpooled tensors; Candle auto similarity could route pair or batch inputs into different embedding spaces or choose from an average rather than the longest input; Candle and ONNX accepted embedded-NUL strings that C truncated to another request, while Candle MLP JSON retained the same gap; native image URL helpers had backend-dependent SSRF policy and treated broad IPv6 space as public; non-CGO Candle contracts, builders, workflows, and official workarounds copied or selected incomplete surfaces | Reject embedded NUL before request-controlled Go-to-C text/JSON allocation; disable tokenizer truncation/padding and return true tokenizer counts; normalize ONNX similarity `auto` honestly while failing closed on unsupported embedding models and unproven layers; require a provable full-layer primary session and accept only a single pooled multimodal vector at the configured dimension; validate every output batch/sequence/hidden dimension and storage cardinality before row access; propagate model/layer/dimension/quality/latency options through public pair and batch APIs; choose one available, context-compatible Candle model from the longest pair/batch input and use it for the entire comparison; restore shared native/mock public types, current embedding/image/similarity validation, error sentinels, metadata, and no-CGO compilation; validate tensor dimensions, multiplication, addressability, encoded bytes, geometry, format, and decode allocation in Go and defensive Rust seams; apply credential-free HTTPS, no-redirect, no-proxy, public-DNS and pinned-dial policy to both backends without echoing URL paths; admit ordinary native IPv6 only inside `2000::/3` after explicit IANA special-use/tunnel/documentation denial and reapply public-IPv4 policy to NAT64; preserve typed `-3`, owned probability copies, and idempotent frees; repair model-free Go/Rust and real-consumer coverage; move complete package copies behind dependency/native cache layers; keep historical docs compatible while current docs use the canonical CPU/AMD/NVIDIA flow; default Git TLS verification on; and trigger consuming image workflows for direct build inputs, Docker ignore files, and draft-to-ready transitions | Audit PR closes the bounded input, error, tensor-memory, model-output, layer-truth, same-space similarity, options, focused mock validation, ownership, compatibility, SSRF, compile, image-copy, docs-example, TLS-default, cache-order, and CI-dependency contracts; #2491 owns the broader non-CGO classification/state decision and mandatory full-suite contract; RT-08/#2396 retains generation/lifecycle ownership and TEST-01/#2477 retains the permanent mandatory ONNX provider lane |
| MD-18 | A remote/OpenVINO `RuntimePlan` suppressed model initialization for semantic cache, memory, and non-LlamaStack vector stores even though their execution paths still called Candle; remote-provider success and error responses had no complete allocation/conversion or keep-alive lifecycle budget; timeout/retry values and exponential delay could exceed safe bounds; an explicit response `index: 0` was conflated with an omitted index; endpoint config could select unrelated process secrets, silently discard URL query/fragment data, follow redirects, or inherit an ambient proxy | Derive one required union of the local-only consumers' assets independently from the classification/tools plan, deduplicate shared models, fail construction on missing paths, and keep Llama Stack provider-owned; cap remote success/error bytes before JSON allocation, reject trailing data and float32-invalid values, never echo provider bodies or URLs, drain bounded error bodies for safe reuse, preserve omitted-versus-zero indexes, use saturating bounded backoff, accept only the dedicated embedding credential, require authenticated endpoints to use HTTPS, reject userinfo/query/fragment, redirects, and implicit proxies, and enforce the same timeout/retry/dimension/key contract across Go, CLI, dashboard, operator API, generated CRD, and runtime env wiring | RT-01/#2470 still owns refresh-generation leasing; RT-08/#2396 owns eventual provider-aware consumer migration and native generation lifecycle; SEC-04/#2286 owns generic provider credential bindings |
| MD-19 | Public native controls could narrow negative or oversized dimensions, layers, limits, counts, and top-k values through `C.int`; public-only image URL policy recognized only the well-known NAT64 prefix, could miss RFC 6052 network-specific translation, and did not explicitly deny the complete IANA special-purpose IPv6 set | Validate every public integer and priority before native dispatch with CGO0 parity; admit ordinary IPv6 only in `2000::/3` after explicit special-use denial; discover active PREF64 values through bounded RFC 7050 lookups, decode all six RFC 6052 layouts plus the well-known prefix, reapply public IPv4 policy, reject ambiguity/failure and RFC 8215 local-use space, briefly cache results, and preserve DNS pinning/no-proxy/no-redirect | #2491 retains broader non-CGO behavioral fidelity; RT-08/#2396 retains native handle lifecycle and mandatory provider coverage |
| MD-20 | Mixed text-plus-image embedding requests accepted ordinary text-model controls and returned same-dimension vectors from unrelated text and multimodal spaces, making a structurally valid response semantically unsafe for cross-modal retrieval | Require the backend to declare one shared multimodal text/image capability and dimension, reject explicit text models, layers, and routing priorities, route mixed text through `MultiModalEncodeText`, validate both modalities and dimensions, and label both results with the same multimodal model; preserve the single-modality contracts and cover fail-closed admission, typed errors, and both native build tags | RT-08/#2396 retains model generation and lifecycle ownership; TEST-01/#2477 retains mandatory ONNX provider/runtime receipts |
| MD-21 | ONNX returned an unloaded mmBERT placeholder before initialization while Candle returned no inventory; the model-info API counted the placeholder, making total/ready summaries backend-dependent and reporting a model the API contract described as loaded | Normalize native inventory at the API seam, exclude every `IsLoaded=false` row, derive backend, supported dimensions, Matryoshka, and target-layer metadata from the active backend capability contract, and run the router APIServer normal/race/vet suite through the ONNX module replacement in the canonical ONNX CI lane | RT-08/#2396 retains model generation and lifecycle ownership; TEST-01/#2477 retains mandatory ONNX runtime/provider coverage |
| MD-22 | The same-origin WizMap rendered knowledge-base records through raw HTML after only two substring checks, while URL query parameters could select attacker-hosted metadata and point data; ordinary event-handler markup therefore crossed from untrusted records into the authenticated Dashboard origin | Render records only through Svelte text interpolation with fixed highlight elements; treat query terms literally; require a complete same-origin HTTP(S) metadata/data pair; reject fragments, credentials, cross-origin values, and redirects; add hostile-markup, metacharacter, URL-policy, source-contract, build, and canonical Dashboard regressions | Audit PR closes the direct injection and external-data path; WEB-01/#2465 retains separate-origin capability isolation for any future embedded active content |
| MD-23 | Candle unified and multimodal startup raced to publish incompatible contents into one process-global `OnceLock`; an earlier or failed model request could be reported as success, Gemma failure could permanently publish an empty factory, ordinary local plans did not require their selected path, and auto similarity estimated context from whitespace rather than each real tokenizer | Serialize validated model publication, keep multimodal state independent, publish only after every requested model and exact path is present, leave failed construction retryable, require the selected Candle/OpenVINO path, initialize only the plan model plus explicit local-consumer union, and route single/pair/batch auto requests from exact untruncated per-model tokenization while preserving one embedding space and typed `-3` overflow | RT-08/#2396 retains model-generation retirement/unload; TEST-01/#2477 retains mandatory real-model provider coverage beyond the model-free concurrency, retry, CJK/subword, normal/race/vet, and AMD proof here |
| MD-24 | Reopened Dashboard review found MCP stdio command execution and secret-bearing DTOs under overbroad authority; built-in OpenClaw mutation reachable through generic tools; same-name host container lifecycle control; worker-token/host-path serialization; same-origin active worker UI with ambient session authority; unbounded live authorization, room history, and automation goroutines; permissive room JSON and cross-window messages; and outbound fetch paths with DNS-rebinding, redirect, proxy, decompression, spoofed-source, and content-log gaps | Split OpenClaw read/use/manage authority and revalidate live mutations; disable production stdio MCP and same-origin worker UI; redact secret DTOs while preserving omitted secret updates; add an internal MCP process capability; bind container and volume lifecycle to instance labels plus immutable IDs; keep private registry fields out of API JSON; require digest/network/resource production controls; cap live watchers, room rows/history, and automation; make HTTP/WS JSON strict and bounded; verify exact postMessage origin/source; and route every maintained fetch through one TLS/public-DNS/pinned/no-proxy/per-hop/bounded client with metadata-only logs | Audit PR closes the concrete authority, serialization, name-capture, admission, and maintained outbound paths. WEB-02/#2466 retains exhaustive future-route registration; OC-01/#2468 retains non-root worker/socket/network provenance; WEB-01/#2465 retains separate-origin capability delivery; RT-10/#1521 retains distributed/archival QoS. |
| MD-25 | Final control-plane review found loose or unbounded JSON, YAML, multipart, subprocess, and response handling across Builder, config rollback, topology, Evaluation, and ML; backup names and file roots admitted traversal, symlink, or permission hazards; Builder and internal classifier/topology clients could inherit proxies, follow redirects, or reflect upstream detail; Evaluation and ML lacked complete work, stream, upload, and artifact ownership; and every training job pointed at one mutable output directory | Canonicalize and cap every caller-controlled field before logs or side effects; use strict single-document decoders and explicit body/file/work limits; make config and result storage private, symlink-safe, atomic, and content-free on failure; use direct TLS 1.2+ clients with no redirects or ambient proxies; bound Evaluation runs, SSE registries, subprocess output, and private result paths; bound ML jobs, requests, files, sidecar streams, and downloads; isolate uploads and copy each successful training artifact through an already-open verified descriptor into a private job-owned snapshot | Audit PR closes the concrete input, egress, admission, and per-job isolation failures. #1388 retains policy for operator-selected Evaluation endpoints; #2467 retains authenticated/shared sidecar transport plus aggregate ML retention; #2326 and TD046 retain cross-process config transactions; TD048 retains aggregate Evaluation/ML job recovery, pagination, quota, and garbage collection. |
| MD-26 | Final adversarial review found fixed Dashboard reverse proxies inheriting `HTTP_PROXY`, unbounded Jaeger/OpenClaw/Replay response buffering, malformed Replay JSON failing open to a low-privilege caller, and the pull-request image job logging into GHCR with package-write authority before running PR-controlled code | Give every fixed reverse proxy one direct pooled TLS 1.2+ transport with a model-compatible bounded header wait; cap transformed response bodies with `limit+1`, close every body, return a generic `502`, and fail Replay redaction closed; make the PR image job read-only, disable checkout credential persistence, and remove registry login while preserving authenticated login only in non-PR publication jobs | Audit PR closes the concrete credential-egress, OOM, redaction, and CI-token paths. SEC-06/#2464 retains future-module content canaries; WEB-02/#2466 retains exhaustive route registration; DEP-01/#2476 retains the broader reproducible multi-ecosystem supply-chain gate. |

## Terminal closure package catalog

The symbolic package ID remains stable even if issue decomposition changes.
Every terminal-record row below points to an applied issue or indexed debt
record. The underlying defect remains open until that record's evidence merges.

| Package | Priority / class | Raw findings | Terminal record | Minimum closure evidence |
| --- | --- | --- | --- | --- |
| SEC-01 Management RBAC and outcome provenance | Critical / security | SA-01, SA-05 | #2463 and #1452 | Deny-by-default route matrix, protected listener, redacted config, source/event ownership tests |
| SEC-02 Replay confidentiality | Critical / security | SA-02 | #1146, coordinated with #2157 and #2364 | Anonymous and cross-tenant denial, summary default, detail privilege, backend tenant predicates |
| SEC-03 Looper internal authenticity | High / proof change | SA-03, XR-SEC-001 | #1443 plus audit PR | Runtime credential, reload continuity, versioned rollback-safe Kubernetes Secret, shared-secret multi-replica contract, generic rejection, no upstream/log/config leak, focused and AMD E2E |
| SEC-04 Credential separation | High / security | SA-04, MD-01, MD-12, MD-26 | Audit PR for MD-01/MD-12/MD-26; #2286 for the broader routing boundary | Default stripping on every branch, explicit forwarding opt-in, direct internal transports, fail-closed edge defaults, canary tests |
| SEC-05 Trusted identity and state ownership | High / security and API | SA-06, SA-09 | #1445, #2362, #2364 | Custom-name/casing matrix and two-tenant store/cache/memory/Replay/learning tests |
| SEC-06 Content-free diagnostics | High / security | SA-07, MD-05, MD-24, MD-25, MD-26 | Audit PR for maintained ExtProc and Dashboard runtime coverage; #2464 retains future-module enforcement | INFO/DEBUG/TRACE content canaries, bounded transformed bodies, and approved helper/static check |
| NET-01 Dashboard outbound fetch | High / security | CC-01, XR-SEC-002, MD-24, MD-25, MD-26 | Audit PR; #1388 retains operator-selected endpoint policy and any future product-policy expansion | Address, redirect, resolution, proxy, expansion, cancellation, rebinding, credential, and limit suite through owned clients |
| NET-02 Local host-publication boundary | Critical / network and control-plane security | CC-15, MD-07 | Audit PR plus non-public deployment receipt | Loopback-safe internal defaults, listener-address fidelity, validated explicit opt-in, command contracts, external negative and intended-gateway positive probes |
| WEB-01 Browser session transport | High / security | CC-02, MD-11, MD-12 | Audit PR for cookie/proxy/origin proof; #2465 for embedded-origin isolation | Strict same-origin HTTP/WebSocket boundaries, cookie-only flows, no reusable URL/storage token, scoped tickets, CSRF and compromised-upstream tests |
| WEB-02 Route-bound authorization | High / security and API | CC-03, MD-15, MD-24, MD-25, MD-26 | Audit PR for exact exemptions, live auth/OpenClaw boundaries, strict control-plane admission, and fail-closed Replay redaction; #2466 for the exhaustive registry | Exhaustive route/method mapping, commit-boundary live-session reauthorization for privileged writes, bounded inputs/responses, and independent Replay/feedback role tests |
| WEB-03 Password lifecycle | High / security and interoperability | CC-14, MD-09 | Audit PR and #2482 | Shared NIST-aligned policy, CAS login/session replacement, bounded verification and session retention, private auth storage, dashboard-only local secret transport, protected-response no-store, fail-closed startup, Chrome form and well-known-path tests; transactional audit delivery remains #2482 |
| WEB-04 Public/shared demo credentials | Critical / credential exposure | MD-06 | Audit PR plus non-public live-rotation receipt | No published reusable secret, old login rejected, sessions revoked, private recovery material, capability-limited demo access |
| ML-01 Sidecar and artifact boundary | High / security and resource | CC-04, XR-SEC-003, MD-25 | Audit PR for direct bounded transport and contained immutable job snapshots; #2467 for authenticated/shared sidecar transport and aggregate retention | Shared-root integration, containment, authenticated transport, immutable job identity, mount/network contract |
| ML-02 Job and upload isolation | High / resource and security | CC-05, XR-SEC-003, MD-25 | Audit PR for bounded private uploads, admission, SSE, and downloads; #2467 and TD048 for cancellation, recovery, aggregate quota, and cleanup | Body/work budgets, random isolated jobs, output containment, concurrency/cancel/recovery/cleanup tests |
| OC-01 OpenClaw provisioning | High / supply chain and host control | CC-06, MD-24 | Audit PR for catalog/path/digest/ownership/resource proof; #2468 for non-root/socket/network provenance | Catalog-only skills, symlink/path containment, image provenance, instance-owned container/volume lifecycle, bounded runtime resources, reduced privilege |
| API-01 OpenAI adapter parity | Medium / API | SA-10 | #2358 | SDK differential/round-trip fixtures, strict unions, escaped IDs |
| API-02 External RAG bounds | Medium / API and security | SA-08 | #2478 | Typed substitution, exact byte limits, malformed/oversized fixtures |
| CFG-01 Transactional config writer | High / integrity | CC-07, MD-14, MD-25 | Audit PR for bounded subprocess/apply rollback plus private atomic file proof; #2326 and TD046 for full transaction ownership | CAS, cross-process lock, symlink-safe private storage, fsync/rename, rollback fault injection, race tests |
| CFG-02 Version/schema/rule convergence | High / contract | CC-08, CC-09, CC-10, MD-04, MD-05 | Audit PR for the website/docs delta; #2469, #2122, #2355, and TD046 for convergence | Shared corpus across Go, CLI, DSL, dashboard, dynamic CRD, operator; strict pre-rollout admission |
| CLI-01 Runtime state path | Low / CLI | CC-11 | #2479 | Normalization, containment, atomic private writes |
| CLI-02 Local storage secret lifecycle | High / security and CLI | CC-16 | #2485 | Per-stack CSPRNG credentials, private persistence, authenticated stores, workload-network isolation, migration and rotation receipts |
| PERF-02 Validation snapshot | Medium / performance | CC-12 | TD046 | One manifest read per KB and scaling benchmark |
| ARCH-01 Control-plane hotspots | Medium / architecture | CC-13 | Existing indexed debt, especially TD006, TD020, TD046 | Extraction-first boundaries and structure ownership |
| RT-01 Runtime generation and classifier snapshot | High / resource | RR-01, RR-02, MD-18 | #2470, TD045, and #2396 | Constructor fault matrix, stream/reload race, exact cleanup, bounded graceful shutdown |
| RT-02 Looper workflow and I/O budgets | High / resource and performance | RR-03, RR-04 | #2471, #1456, and #2336 | Cross-request resume, stable clients/goroutines, byte limits, streaming/cancellation fixtures |
| RT-03 Replay actor and retention | High / data integrity and resource | RR-05, RR-06 | #2472, coordinated with #1146 and #2157 | Admission-close-drain barriers, durable acknowledgment, TTL conformance, cursor pagination, total-size cap |
| RT-04 Cache cancellation and result value | High / resource and correctness | RR-07 | #2473 | Per-request scores, miss reset, cancellation propagation, constructor cleanup |
| RT-05 Vector lifecycle and consistency | High / resource and integrity | RR-08, RR-09 | #2474 | Stop deadline, bounded batching/status, side-effect fault matrix, restart reconciliation |
| RT-06 Selection ownership and cardinality | High / resource | RR-10, RR-11 | #2222 and #2396 | No unused allocations, concurrent lease/load/close, bounded soak and persistence |
| RT-07 Memory tracking actor | High / resource | RR-12 | #2339 | Bounded slow-backend soak, atomic/coalesced updates, owned/borrowed exact-close tests |
| RT-08 Native generation and lifecycle | High and medium / native | RR-13, RR-14, RR-15, MD-02, MD-03, MD-10, MD-13, MD-17, MD-18, MD-19, MD-20, MD-21 | Audit PR for image/text input, cross-surface admission, same-space routing, loaded-only inventory, and error/resource contracts; #2396, #2477, and TD042 for lifecycle/structure | Init failure/retry/path tests, bounded queue/unload, allocator instrumentation, classify/reload/close sanitizers |
| RT-09 Search and timer lifecycle | High and medium / performance/resource | RR-16, RR-17 | #2475 and TD045 | Positive-duration validation, concurrent close race, 100k search profiles |
| RT-10 OpenClaw collaboration lifecycle | High and medium / correctness and resource | RR-18, MD-16, MD-24 | Audit PR for single-owner send/close plus bounded room/automation proof; reopened #1521 for distributed archival/QoS/reclamation | Idempotent disconnect, focused repeated race, full handler race, SSE continuity, close frame, bounded messages/history/workers/backpressure and reclamation |
| DEP-01 Vulnerability gate | High / supply chain | XR-DEP-001, MD-26 | Audit PR for least-privilege PR image jobs; #2476 for the broader dependency gate | No write credential before untrusted code, patched dependencies, lock/toolchain pins, ecosystem scans, expiring exceptions |
| TEST-01 ONNX compile lane | High / testing/native | XR-TEST-001, MD-17, MD-19, MD-20, MD-21 | Audit PR, #2477, and #2396 | Model-free binding and router-APIServer Go normal/race/vet, Rust typed-status tests, exact GPU image build, plus selected runtime receipts |
| TEST-02 E2E acceptance floors | High / testing | XR-TEST-002 | #2379 | Role inventory and deterministic non-zero acceptance contracts |
| TEST-03 Local feature-gate integrity | High / testing | XR-TEST-003 | Audit PR and #2487 | Forced build/stop-failure propagation, repo-owned CLI resolution, successful post-fix feature smoke, and ownership-safe isolated teardown |
| TEST-04 Affected-profile selection integrity | High / testing | XR-TEST-004, MD-08 | Audit PR | Strict baseline-union-affected selector, canonical/common-trigger consistency, complete security-path ownership, simultaneous-path tests, and selected-profile CI receipt |
| PERF-01 Numeric regression gate | High / performance/testing | XR-PERF-001 | #2455 | Repaired compare path and required PR-visible numeric result |

## Raw finding-to-terminal mapping

This compact ledger is the completeness check for the four audit lanes.

| Lane | Complete mapping |
| --- | --- |
| Security and API | SA-01→SEC-01; SA-02→SEC-02; SA-03→SEC-03; SA-04→SEC-04; SA-05→SEC-01; SA-06→SEC-05; SA-07→SEC-06; SA-08→API-02; SA-09→SEC-05; SA-10→API-01 |
| Resource and runtime | RR-01→RT-01; RR-02→RT-01; RR-03→RT-02; RR-04→RT-02; RR-05→RT-03; RR-06→RT-03; RR-07→RT-04; RR-08→RT-05; RR-09→RT-05; RR-10→RT-06; RR-11→RT-06; RR-12→RT-07; RR-13→RT-08; RR-14→RT-08; RR-15→RT-08; RR-16→RT-09; RR-17→RT-09; RR-18→RT-10 |
| Control plane and contract | CC-01→NET-01; CC-02→WEB-01; CC-03→WEB-02; CC-04→ML-01; CC-05→ML-02; CC-06→OC-01; CC-07→CFG-01; CC-08→CFG-02; CC-09→CFG-02; CC-10→CFG-02; CC-11→CLI-01; CC-12→PERF-02; CC-13→ARCH-01; CC-14→WEB-03; CC-15→NET-02; CC-16→CLI-02 |
| Cross-repository | XR-SEC-001→SEC-03; XR-SEC-002→NET-01; XR-SEC-003→ML-01/ML-02; XR-DEP-001→DEP-01; XR-TEST-001→TEST-01; XR-TEST-002→TEST-02; XR-TEST-003→TEST-03; XR-TEST-004→TEST-04; XR-PERF-001→PERF-01 |
| Final-main delta | MD-01→SEC-04; MD-02→RT-08; MD-03→RT-08; MD-04→CFG-02; MD-05→SEC-06/CFG-02; MD-06→WEB-04; MD-07→NET-02; MD-08→TEST-04; MD-09→WEB-03; MD-10→RT-08; MD-11→WEB-01; MD-12→SEC-04/WEB-01; MD-13→RT-08/RT-01; MD-14→CFG-01; MD-15→WEB-02; MD-16→RT-10; MD-17→TEST-01/RT-08; MD-18→RT-01/RT-08; MD-19→TEST-01/RT-08; MD-20→TEST-01/RT-08; MD-21→TEST-01/RT-08; MD-22→WEB-01; MD-23→RT-08/TEST-01; MD-24→SEC-06/NET-01/WEB-01/WEB-02/OC-01/RT-10 |

Publication and closure check: every raw finding now maps to an applied issue
or indexed debt record with priority metadata. Attach the final proof pull
request and green CPU, AMD, and CI receipts before closing #2375. A parent
comment is not terminal ownership by itself.

## Recommended remediation waves

1. **Immediate proof and containment:** finish SEC-03; execute the already
   assigned SEC-01 and SEC-02 terminal records; apply SEC-04
   branch-independent credential hygiene.
2. **Tenant and security plane:** SEC-05, SEC-06, NET-01, NET-02, WEB-01, WEB-02,
   WEB-03, WEB-04, ML-01, ML-02, and OC-01.
3. **Runtime ownership:** RT-01, RT-03, RT-04, RT-05, RT-07, and RT-10
   establish safe close and drain semantics before more backends are added.
4. **Contract convergence:** CFG-01, CFG-02, CLI-02, API-01, API-02, operator admission,
   and the shared contract corpus.
5. **Bounded algorithms and native lifecycle:** RT-02, RT-06, RT-08, RT-09,
   and PERF-02.
6. **Permanent gates:** DEP-01, TEST-01, TEST-02, TEST-03, TEST-04, PERF-01, and
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
| Baseline-union-affected profile selection | Strict tested selector plus canonical path-to-profile ownership | TEST-04 |
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
  lifecycle proof. The full handler race suite then exposed the OpenClaw
  WebSocket disconnect/fan-out contract; after the single-owner fix, focused
  repeated race and full handler race runs pass. Remaining graceful-delivery
  QoS is retained under RT-10/#1521 rather than waived.
- ML Rust tests passed. The NLP Rust crate currently contains no unit tests.
- ONNX model-free Go normal/race/vet, no-CGO and real-consumer compilation, and
  65 Rust library tests pass after repairing API/probability ownership, typed
  context-limit propagation, true token metadata, `auto` compatibility,
  unsupported-model/layer fallback, full-layer provenance, model-output shape
  admission, strict pooled multimodal output, complete similarity options,
  embedded-NUL admission, URL/tensor/image budgets, and build-package copies.
  Candle's focused embedded-NUL, SSRF, direct tensor, pair/batch same-model,
  metadata, mock input-contract compile, and MLP JSON normal/race/vet plus
  defensive Rust dimension tests and router-consumer suites also pass. Canonical
  Docker/workflow/docs contracts cover complete binding copies,
  dependency-cache ordering, safe mirror arguments, TLS-on defaults, direct
  build inputs, draft-to-ready events, and historical-document compatibility.
  Exact ROCm image/runtime evidence remains part of the final AMD receipt rather
  than being inferred from local compilation. The complete non-CGO behavioral
  suite still exposes pre-existing synthetic classification and state drift;
  [#2491](https://github.com/vllm-project/semantic-router/issues/2491) owns the
  decision to make that build contract-faithful or explicitly fail closed.
- Native-linked Looper and extproc suites passed after the normal native build,
  including focused authentication/reentry integration tests and repeated race
  runs. Broader feature, affected-E2E, platform, and final PR receipts remain
  separate completion evidence.
- The final-main delta passed focused CPU Rust image tests, Go binding and
  classify/API normal plus race tests, generated/checked-in ExtProc
  fail-closed contract tests, Operator tests, E2E compilation, website lint,
  and a Chinese static-site build. The deployed malformed-image testcase is
  wired into the multimodal profile; its live cluster and AMD receipts remain
  part of the final platform run rather than being inferred from compilation.
- The previously published demo credential was rotated outside the repository:
  the old value is rejected, its active sessions were revoked, a new value was
  verified and immediately logged out, and auth database artifacts were made
  private. Replacement material is intentionally absent from public evidence.
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
8. No reusable demo credential remains in public artifacts; the exposed value
   is rejected, its sessions are revoked, and replacement material remains in
   an approved private handoff path.
9. Internal container services are loopback-published by default, explicit
   exposure is validated, the intended gateway remains reachable, and a live
   external negative probe cannot bypass the TLS edge.
10. CPU validation, affected native tests, AMD end-to-end regression, and final
   pull-request CI are green, with failed runs and successful reruns preserved
   in the non-public evidence ledger.
11. The parent receives a concise closure comment linking this audit, the proof
   pull request, the child-issue table, a non-sensitive AMD evidence summary,
   and the final CI result.

Until all eleven conditions hold, this document is a closure plan and evidence
index—not a declaration that the repository-wide risks are fixed.
