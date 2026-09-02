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

Matches blueprint section 13 (Definition of done) verbatim — see
`.ai/issue3347/2_blueprint_gpt_5.md#13-definition-of-done`. Not restated here
to avoid drift between two copies; that file is the durable design source.

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
- [ ] `TASK-02` `global.stores.tool_sessions` store config: new
      `config/tool_session_store.go` (backend/TTL/cardinality/Redis
      sub-config, defaults, validation), wired through
      `canonical_global.go` / `canonical_export.go` / the canonical loader,
      reference-config coverage in `reference_config_global_test.go`.
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

TASK-02: add `config/tool_session_store.go` (store schema, defaults, bounds)
and wire `global.stores.tool_sessions` through the canonical layer. TASK-01 is
done and verified (build/vet/test green); its deviation from the blueprint's
literal Go contract is noted inline in `tool_selection_plugin.go` — see PR
description once opened.

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
- Blueprint: `.ai/issue3347/2_blueprint_gpt_5.md`
- Reconnaissance: `.ai/issue3347/1_handoff_gemini_38_flash.md`
