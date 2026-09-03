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
- [ ] `TASK-03` `pkg/sessiontools` package: `state.go` (envelope + `ToolState`,
      decode validation, cloning), fingerprint helpers in `pkg/tools`
      (definition/catalog/policy/capability fingerprints, canonical JSON +
      SHA-256), in-memory `store_memory.go` (bounded LRU/idle-expiry).
- [ ] `TASK-04` Session provenance + trusted-key resolver:
      `extproc/sticky_tool_identity.go` (new), provenance fields on
      `RequestContext`, populated at every `SessionID` assignment site in
      `extproc/session_transition.go`. Opaque HMAC-derived key via the
      `USER_SCOPE_NAMESPACE_SECRET` primitive (extend/reuse, do not add a
      second weaker hash) — construction must require the secret and reject
      the unkeyed fallback when any sticky decision is enabled.
- [ ] `TASK-05` CLI mirror (`src/vllm-sr/cli/`) and dashboard schema
      (`dashboard/frontend/src/lib/dslSchemas.ts`) for both new config blocks;
      state-inventory row in
      `tools/agent/docs/architecture/state-taxonomy-and-inventory.md`; a
      disabled-by-default fragment
      (`config/fragments/plugin/tool-selection/sticky-add-from-database.yaml`).
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

TASK-03: `pkg/sessiontools` package — `state.go` (envelope + `ToolState`,
decode validation, cloning), fingerprint helpers in `pkg/tools`
(definition/catalog/policy/capability fingerprints, canonical JSON +
SHA-256), in-memory `store_memory.go` (bounded LRU/idle-expiry). Update from
TASK-02's review round: `ToolSessionStoreConfig.Validate()` is now wired
into `globalConfigContractValidators`, so admission already guarantees any
`RouterConfig.ToolSessions` reaching this task is valid — TASK-03 should
consume the `Effective*` helpers when constructing the in-memory store, but
does *not* need its own `Validate()` call the way `vectorstore_runtime.go`
does for `VectorStoreConfig` (that exists specifically because nothing
validates `VectorStoreConfig` earlier).

TASK-01 and TASK-02 are both done and verified (build/vet/test green; full
`pkg/config` suite, not just targeted tests, per the lesson from TASK-01's
own review round). TASK-01's deviation from the blueprint's literal Go
contract, and TASK-02's CLI/dashboard scope decision, are both noted inline
in their respective task entries above — see PR description once opened.

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
