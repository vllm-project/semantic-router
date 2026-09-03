# PL-0042: Session-Scoped Sticky Tool-Set Selection

## Goal

Implement an opt-in, bounded session tool-set policy around the existing
decision-scoped `tool_selection` plugin: a trusted session retains the exact
order of previously selected tools, pins tools observed in assistant tool
calls, and appends a bounded number of newly relevant tools per turn. Every
request re-resolves current tool definitions and reruns current
authorization, policy, availability, model-capability, and wire-capability
checks before any stored identity is emitted.

## Scope

Four phases, each its own reviewable commit series:

1. State/config/trust contract (config types, `pkg/sessiontools` state model,
   fingerprint helpers, in-memory store, session provenance/trust resolver —
   sticky stays disabled, no runtime manager constructed).
2. Deterministic local sticky selection (refactor add/filter to a shared
   finalizer, local-store-backed reuse/growth/pin/rehydrate).
3. Current-turn authorization/capability/invalidation (consumes issue #2361's
   eligibility contract or lands the minimal seam; full invalidation matrix).
4. Redis CAS store, restart/reload/partial-deployment recovery, maintained
   E2E.

## Non-Goals

Per blueprint section 12 — executing tools; caching or replacing final
execution-gateway authorization; storing full tool definitions or sensitive
request/response content; unbounded catalog growth; forcing provider
prompt-cache hits; changing Router Learning or session-aware model-selection
semantics; extending legacy selector-local stores; accepting derived or
anonymous session identity for sticky behavior.

## Exit Criteria

Issue #3347 may be closed only when:

- configuration is disabled by default and bounded in every representation;
- stored state is identity-only and hard-partitioned by recipe and
  authenticated principal/session;
- repeated sequential turns retain an identical ordered prefix;
- growth and pinning obey the fixed bound and deterministic merge rules;
- current authorization/availability/capability/schema/catalog are
  re-evaluated on every use;
- local and Redis concurrency semantics are documented and enforced by
  tests;
- restart, reload, expiry, missing store, corruption, and partial deployment
  fail safely;
- receipts are bounded and content-minimized;
- maintained E2E covers the full completion signal using actual
  provider-bound tool arrays;
- the final repo-reported fast, feature, smoke, and affected E2E gates pass;
- any deliberately deferred architecture gap is recorded in the indexed
  tech-debt register, not only in PR text.

This is committed here verbatim, not only referenced, because the original
design material it comes from was authored as local working notes outside
the tracked repository — a reviewer checking out this branch would not see
it. Kept in sync by hand if the issue's completion signal changes.

## Task List

### Phase 1 — State/config/trust contract

- [x] `TASK-01` `StickyToolSelectionConfig` on `ToolSelectionPluginConfig`
      (`config/tool_selection_plugin.go`), `Effective*` helpers, bounds
      validation (`config/tool_selection_plugin_validation.go`), unit tests.
      Also required (discovered via the package's own reflection-based
      coverage gate, `TestReferenceConfigCoversSupportedRoutingSurfaces`): a
      disabled `sticky:` block on the `tool_selection`/`add`-mode sample in
      `config/config.yaml` — every `ToolSelectionPluginConfig` field must
      appear somewhere in the reference config or that test fails.
      Public docs per `pkg/config/AGENTS.md`'s "Config-contract changes must
      update the relevant public docs" rule: updated
      `website/docs/tutorials/plugin/tool-selection.md` (new subsection) and
      added `config/fragments/plugin/tool-selection/sticky-add-from-database.yaml`.
      Evaluated and deliberately did *not* touch `config/README.md` or
      `website/docs/proposals/unified-config-contract-v0-3.md` for this
      task: both are purely structural/architectural docs with no
      per-plugin or per-sub-field content anywhere (neither mentions
      `tool_selection` at all), and precedent confirms this — neither the
      PR that introduced the entire `tool_selection` plugin (`e0e7df56`,
      touched only the tutorial page) nor the PR that added its
      `advanced_filtering` sub-config (`5f14781c`, touched none of the
      three) updated them either. `website/docs/installation/configuration.md`'s
      plugin table already lists `tool_selection` (one terse row per
      plugin, linking to its guide, same as every other row) and needs no
      change for a sub-field addition within an already-listed plugin.
- [x] `TASK-02` `global.stores.tool_sessions` store config:
      `config/tool_session_store.go` (`ToolSessionStoreConfig`,
      `ToolSessionRedisConfig`, `Effective*` helpers, backend/TTL/cardinality
      bounds), `config/tool_session_store_validation.go` (`Validate()`:
      backend enum, local-forbids-redis / redis-requires-address, all five
      numeric bounds including the `max_sessions_per_identity <=
      max_sessions` cross-field bound). All five numeric fields are `*int`
      from the start (0 is out of every one of their valid ranges and must
      be rejected, not defaulted) — applying TASK-01's unset-vs-explicit-zero
      lesson up front instead of needing a follow-up fix this time. Wired
      through `canonical_global.go` (`CanonicalStoreGlobal.ToolSessions`,
      `applyCanonicalGlobal`) and `canonical_export.go` (`exportGlobal`,
      plus a real deep-clone `cloneToolSessionStoreConfig` — this struct has
      five `*int` fields and a nested `*Redis`, so the existing
      `cloneVectorStoreConfig`'s shallow `*cfg` copy would have aliased
      pointers across the export boundary; a dedicated test
      (`TestToolSessionStoreCanonicalRoundTrip`) mutates the clone and
      asserts the original is untouched). `RouterConfig.ToolSessions` added
      in `config.go`. Reference-config coverage in
      `reference_config_global_test.go` plus a `tool_sessions:` sample in
      `config/config.yaml`.

      **Review round 2, both landed together because the first makes the
      second load-bearing, not optional:**
      1. `validateGlobalToolSessionsContracts` added to
         `globalConfigContractValidators` in `validator.go`, so an invalid
         `global.stores.tool_sessions` now fails at config admission
         (`ParseYAMLBytes`), not only at a later direct `.Validate()` call.
         Correcting what this entry said before landing this: it is *not*
         matching `VectorStoreConfig.Validate()`'s precedent —
         `VectorStoreConfig` has no entry in `globalConfigContractValidators`
         at all (confirmed by grep) and is called only from
         `routerruntime/vectorstore_runtime.go` at store-construction time.
         This instead matches the more common shape most other entries in
         that list already use (e.g. `validateAdvancedToolFilteringConfig`)
         — admission-time, fail-fast. `VectorStoreConfig` is the outlier,
         not the pattern.
      2. The `config/config.yaml` sample changed from `backend: local` with
         a present-but-inert `redis:` block to a self-consistent
         `backend: redis` (matching `response_api`'s real
         `store_backend: redis` sample, not `vector_store`'s
         multiple-simultaneous-backends shape — `local` has no
         backend-specific sub-block of its own needing separate coverage,
         so unlike `vector_store` there was never a field-coverage reason
         for the old shape). This wasn't just style: once (1) landed,
         `TestReferenceConfigUsesStrictCanonicalSchema` — which calls
         `ParseYAMLBytes` on the real `config/config.yaml` — would have
         started failing against the old sample. Verified by running the
         full `pkg/config` suite, not just the new/targeted tests, after
         both changes landed together.
      3. Hardened `validateBackendRedisContract` (whitespace-only
         `redis.address` now rejected via `strings.TrimSpace`) and added a
         `redis.database < 0` rejection.

      Net effect on the TASK-03/04 note below: since admission now
      guarantees any parsed `RouterConfig.ToolSessions` is already valid,
      TASK-03/04 constructing the sessiontools store does *not* need its own
      `Validate()` call the way `vectorstore_runtime.go` does for
      `VectorStoreConfig` — that construction-time call exists there
      specifically because nothing validates it earlier.

      **CLI/dashboard mirrors deliberately not added**, contrary to the
      original task wording: empirically confirmed (round-trip test against
      the live `UserConfig` Pydantic model) that `global` is an untyped
      `Optional[Dict[str, Any]]` passthrough in `src/vllm-sr/cli/models.py`
      (`UserConfig.global_`), unlike `routing.decisions[].plugins[].configuration`
      which is strictly typed per-plugin (that's why `StickyToolSelectionConfig`
      needed a Python mirror in TASK-01 — Pydantic silently drops unknown
      fields there, but not under an untyped dict). No `StoresConfig` class
      exists anywhere in `models.py`. `dslSchemas.ts` has no `global.*`
      representation at all — confirmed by grep, matching the task's own
      hedge ("if global stores are registered there"). Both left untouched;
      `global.stores.tool_sessions` already round-trips byte-for-byte
      through the CLI without any change.
- [x] `TASK-03` `pkg/sessiontools` package: `state.go` (`State`/`ToolState`,
      `Validate(maxTools, maxStateBytes)`, `Clone()`), `store.go` (`Store`
      interface, `VersionedState`, `QuotaKey`, sentinel errors), and
      `store_memory.go` (`MemoryStore`: 64-way sharded per-key locking for
      the hot reuse path, sliding TTL on Load and CompareAndSwap, exact
      per-identity eviction + sampled approximate-LRU global eviction —
      matching `sessiontelemetry/router_memory.go`'s own established
      sampled-eviction philosophy rather than a full O(n) scan on every
      admission). Fingerprint helpers landed in `pkg/tools/fingerprint.go`
      per the task (`ToolDefinitionFingerprint`, `ToolCatalogFingerprint`,
      `ToolPolicyFingerprint`, `ToolCapabilityFingerprint`), all canonical
      JSON + SHA-256, with a `canonicalizeJSON` unmarshal-then-remarshal
      helper so a tool's raw `InputSchema` bytes hash identically
      regardless of source key order/whitespace.

      **Two deliberate deviations from the literal task spec, both
      documented inline in the code, not just here:**
      1. `Store.Load` returns `(VersionedState, error)`, not
         `(VersionedState, bool, error)` as originally sketched (from the
         blueprint itself, section 7.1, not introduced by this task's
         wording). The sketched third `bool` would carry zero information
         beyond `VersionedState.Found` — two independent signals for one
         fact invites them to disagree. Fixing this now, before every later
         task builds against the interface, is far cheaper than after.
      2. `evictForCapacityLocked` (global `max_sessions` eviction) samples
         up to `memoryStoreEvictionSampleSize` (32) entries rather than
         scanning the full store — the task's own wording ("LRU eviction")
         together with `max_sessions`' ceiling of 100000
         (`config.ToolSessionStoreMaxMaxSessions`) would mean an O(100000)
         scan on every admission at full capacity. Per-identity eviction
         stays exact (that bucket is bounded by
         `max_sessions_per_identity` itself, typically far smaller).
         Matches this codebase's own precedent
         (`sessiontelemetry/router_memory.go`'s `evictOldestLocked`,
         explicitly documented there as approximate/sampled for the same
         reason), not an invented shortcut.

      **Verification, and its real limits, stated plainly:** `go build`/
      `go vet` clean on both packages; `go test ./pkg/sessiontools/... -race`
      — full pass, no races, exercising concurrent CAS across both distinct
      and identical keys. `pkg/tools`'s new fingerprint tests (24 cases)
      verified in isolation via `-run` with `CGO_ENABLED=0` — the package's
      *existing* Ginkgo suite needs the real candle native backend and
      cannot run either way in this sandbox (missing `.so` with cgo on;
      `BeforeSuite` fails outright with cgo off, unrelated to this task's
      files, which build and vet clean under both). One real bug the
      cgo-disabled run caught: a test-helper contamination in
      `TestToolDefinitionFingerprint_NameWhitespaceNormalized` (the
      fixture's `Description` field embedded the raw, untrimmed name,
      entangling two variables the test meant to isolate) — fixed in the
      test, not the fingerprint code, which was already correct. Whole-module
      `go build -buildvcs=false ./...` sweep: only the same pre-existing
      cgo/wasm failures as every prior task, nothing new.
- [x] `TASK-04` Session provenance + trusted-key resolver:
      `SessionProvenance` closed enum + `AuthenticatedPrincipal` field added
      to `RequestContext` (`request_context.go`); stamped at every
      `SessionID` assignment site in `session_transition.go`
      (`populateSessionTransitionFields`, `populatePinnedSessionFromHeaders`,
      `populateSemanticSessionIDIfNeeded`) and `AuthenticatedPrincipal`
      populated from `authHeaderUserID(ctx)` at the top of both entry
      points (not just one — `populatePinnedSessionFromHeaders` runs on the
      fast-extract path before full parsing, and a header-pinned trusted
      session reaching decision evaluation needs the principal available
      that early too, not only in the slower path). New
      `extproc/sticky_tool_identity.go`: `ResolvedStickyIdentity`,
      `ResolveStickyToolIdentity(ctx, recipeName, policyFingerprint)`
      implementing the trust rules (non-empty principal;
      `SessionProvenanceResponseAPI`/`Header` only; non-empty recipe) and
      opaque HMAC-SHA256 storage/quota key construction. Secret requirement
      wired into `buildOpenAIRouterFromConfig` (the task referenced this as
      `buildRouterFromConfig`; verified the actual name before editing) via
      a new `validateStickyToolSelectionSecret`, unconditional — not gated
      on `ManagementAPI.RemoteExposure` like its neighbor
      `validateResponseCacheScopeSecret`, since an unkeyed sticky
      deployment has no acceptable degraded mode the way response-cache
      scoping does (see the function's own doc comment for why).

      **One deliberate deviation, documented inline:**
      `stickyToolIdentitySecret` reads `USER_SCOPE_NAMESPACE_SECRET`
      directly via `os.Getenv`, rather than going through
      `cache.UserScopeNamespace` (the existing exported primitive the
      blueprint says to reuse). Checked `pkg/cache/cache.go` first:
      `UserScopeNamespace` silently falls back to an *unkeyed* SHA-256 when
      the secret is absent — the fallback the trust rules explicitly must
      reject — and `pkg/cache` exposes no accessor for the raw secret
      value, only that function and the presence check
      (`UserScopeSecretConfigured`, which this code does still call and
      reuse directly). Reusing the *primitive* (same env var, same
      HMAC-SHA256 construction) while implementing it locally, rather than
      calling through a function whose behavior doesn't fit, is not "a
      second weaker hash implementation" — it is the same one, used more
      strictly.

      **Verification, same limits as every prior extproc-touching task
      this session:** `go build`/`go vet` clean (including all new/edited
      test files, which vet fully type-checks). `go test` cannot link for
      this package in this sandbox either with cgo (missing
      `libcandle_semantic_router.so`) or without it (`valkey-glide`
      requires cgo unconditionally — confirmed again, not just assumed,
      matching what TASK-01 through TASK-03 already found for this same
      package). Whole-module `go build -buildvcs=false ./...`: only the
      same pre-existing cgo/wasm failures as every other task.
- [x] `TASK-05` Narrowed by earlier tasks — three of the original four items
      were already done: CLI mirror and dashboard schema for
      `StickyToolSelectionConfig` landed in TASK-01 (`src/vllm-sr/cli/models.py`,
      `dslSchemas.ts`); TASK-02 established, with evidence, that
      `ToolSessionStoreConfig` needs neither (`global.*` is an untyped
      passthrough in the CLI, and `dslSchemas.ts` has no `global.*`
      representation at all); the fragment
      (`config/fragments/plugin/tool-selection/sticky-add-from-database.yaml`)
      also landed in TASK-01. Closed the remaining item: a "Sticky tool-set
      session state" row added to
      `tools/agent/docs/architecture/state-taxonomy-and-inventory.md`'s
      "Current Inventory" table, plus a
      `global.stores.tool_sessions.backend = local` entry under "Default
      Memory-Backed Surfaces To Treat As High Risk". Corrected one factual
      detail while writing it: the row's backend value is `local`
      (`config.ToolSessionStoreBackendLocal`), not `memory` — verified
      against the actual constant before writing the row, rather than
      copying the term other memory-backed rows in that same table happen
      to use for their own (differently-named) backends.
- [ ] `TASK-06` Phase exit: `make agent-ci-gate CHANGED_FILES="<phase-1 files>"`.

### Phase 2 — Deterministic local sticky selection

- [ ] `TASK-07` Refactor `runSemanticToolSelection` /
      `runToolSelectionPluginAdd` / `runToolSelectionPluginFilter` to return
      `toolSelectionResult`; one shared finalizer.
- [ ] `TASK-08` `extproc/sticky_tool_selection.go` (new): catalog snapshot,
      deterministic merge (blueprint 5.3), rehydration, `applySelectedTools`
      + `Generation` bump only on change, receipt emission.
- [ ] `TASK-09` Fix `ToolsDatabase` nondeterminism found while touching this
      surface (load-order/sort, tie-break by name, duplicate-name rejection).

### Phase 3 — Authorization/capability/invalidation

- [ ] `TASK-10` Consume #2361's `ToolEligibilityEvaluator` or land the minimal
      seam; wire model/target-codec capability checks; implement every row of
      the invalidation matrix (blueprint 5.4).
- [ ] `TASK-11` Content-minimized Router Replay receipt.

### Phase 4 — Redis CAS + recovery + E2E

- [ ] `TASK-12` `store_redis.go`: versioned CAS envelope, bounded retry,
      quota/index maintenance, generation-owned lease slot.
- [ ] `TASK-13` `e2e/testcases/sticky_tool_selection_e2e.go` (new) + profile
      wiring; domain-registry ownership update.

## Next Action

TASK-06: run the Phase 1 exit gate,
`make agent-ci-gate CHANGED_FILES="<every phase-1 file>"`. This is the
last item before Phase 1 is complete — TASK-01 through TASK-05 are all
done.

Once TASK-06 passes, Phase 2 begins with TASK-07: refactor
`runSemanticToolSelection` / `runToolSelectionPluginAdd` /
`runToolSelectionPluginFilter` to return `toolSelectionResult` and share
one finalizer. That finalizer is the first real caller of
`ResolveStickyToolIdentity` (TASK-04) and `sessiontools.NewMemoryStore`
(TASK-03's constructor, not yet invoked anywhere) — construct the store
once at router-generation build time, gated on at least one normalized
recipe decision enabling `tool_selection.sticky` (construct no store at
all otherwise, per this plan's Operating Rules), and call
`ResolveStickyToolIdentity` per request to get the `QuotaKey`/`StorageKey`
before touching the store.

TASK-01 through TASK-05 are all done and verified. Verification depth
varies by what this sandbox can actually execute for each package:
`pkg/config` (TASK-01/02) and `pkg/sessiontools`/`pkg/tools` (TASK-03) got
real `go test` runs, `pkg/sessiontools` additionally race-clean via
`go test ./pkg/sessiontools/... -race`; `pkg/extproc` (TASK-04, and the
extproc-touching pieces of earlier tasks) has never been able to link a
test binary in this sandbox — confirmed again for TASK-04, not just
assumed — so `go build`/`go vet` (which do fully type-check test files) are
its verification ceiling here; CI or a full local dev environment needs to
actually run `pkg/extproc`'s tests before merge. Deviations from each
task's literal spec are documented inline in their respective task entries
above and in the touched code's own comments, not only here.

## Operating Rules

- Sticky stays `enabled: false` and constructs no runtime manager until an
  explicit later task turns it on behind tests.
- Never persist full `llmprotocol.Tool` values, schemas, arguments, prompts,
  authorization decisions, credentials, or raw session/principal identifiers
  — identities and bounded fingerprints only.
- Client-controlled session values are only safe inside the authenticated
  principal's hard namespace; a header alone is never authentication.
- Run `make agent-report ENV=cpu CHANGED_FILES="..."` before each phase's
  gate to confirm the applicable local rules and tests haven't shifted.
- `go build`/`go vet` for touched packages after every task; this sandbox has
  no Go toolchain installed directly — verified via `golang:1.25` in Docker
  with the repo mounted (native Rust/cgo bindings under
  `candle-binding`/`ml-binding`/`nlp-binding` and `valkey-glide` are not
  buildable in that container, so `extproc` package *tests* cannot execute
  there — `go build`/`go vet` still fully type-check it).

## Related Docs

- Issue: <https://github.com/vllm-project/semantic-router/issues/3347>
- Four-phase collaborator breakdown: <https://github.com/vllm-project/semantic-router/issues/3347#issuecomment-5512922218>
- Related capability/authorization contract: <https://github.com/vllm-project/semantic-router/issues/2361>

The original blueprint and reconnaissance documents this plan was built from
are local working notes, not part of the tracked repository, and are not
linked here for that reason — this plan's Goal/Scope/Non-Goals/Exit Criteria
sections are the durable, self-contained record of that material.
