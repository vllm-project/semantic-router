# TD001 Config Contract Convergence Execution Plan

This plan turns [TD001](../tech-debt/TD001-config-surface-fragmentation.md) into a resumable workstream. The target is to retire config-shape drift across router runtime, Python CLI, dashboard config editing, and Kubernetes/operator translation without relying on parallel handwritten schemas forever.

## Goal

- Retire TD001 by converging the repository on one canonical, versioned router authoring contract.
- Make new config features land once in the canonical contract and then fan out through thin adapters, not parallel schema rewrites.

## Scope

- Canonical router authoring contract and versioning strategy
- Router compile/runtime boundary
- Python CLI schema, parser, validator, and merger alignment
- Dashboard config DTOs, backend config APIs, and editor flows
- DSL and Kubernetes/operator translation paths
- Cross-surface contract tests, migration tooling, and cutover sequencing

Out of scope for this workstream:

- broader local-vs-Kubernetes portability work already tracked by `TD002`
- environment lifecycle unification already tracked by `TD004`
- unrelated dashboard enterprise-console work tracked by `TD005`
- hotspot cleanup beyond the extractions needed to land the new seams safely

## Exit Criteria

- One canonical, versioned config contract exists for the common authoring path.
- Router runtime loads that contract directly or through one explicit compiler path.
- CLI, dashboard, DSL, and operator paths consume shared/generated contract types or thin adapters instead of owning independent handwritten schemas.
- Adding a config feature no longer requires parallel structural edits across Go router config, Python CLI schema, dashboard config types, and CRD fields for the common path.
- Cross-surface contract tests prove that one representative config can round-trip through CLI, router, dashboard save/load, and Kubernetes translation without silent field loss.
- Legacy or hybrid config compatibility is isolated behind an explicit migration layer instead of remaining the normal editing path.

## Design Direction

- Recommended architecture: introduce a new canonical authoring contract and keep `RouterConfig` as the runtime-oriented compiled form during migration.
- Do not make the current flat Go `RouterConfig` the long-term authoring source of truth. It is the runtime contract, already carries legacy flattening, and is coupled to router helper methods and validation paths.
- Do not make the current Python `UserConfig` the long-term source of truth as-is. It is closer to the desired user-facing shape, but it is Python-owned, incomplete, and currently relies on `extra = "allow"` plus merger passthrough for unsupported fields.
- Treat Kubernetes deployment knobs and router authoring config as separate seams. Platform-native backend discovery and service references can stay adapter-owned, but router behavior config should not be re-modeled as a second partial schema.
- Collapse dashboard dual-format editing into a single canonical DTO path. Legacy/hybrid detection should live only in migration and read-compatibility code, not in the steady-state editor.

## Task List

- [x] `C001` Produce a field inventory and divergence matrix across `RouterConfig`, CLI `UserConfig`, dashboard config DTOs, DSL emission, and `SemanticRouterSpec`.
  - Done when the repo has a durable mapping of canonical fields, adapter-only fields, legacy-only fields, and missing fields.
- [x] `C002` Record the canonical contract decision in an ADR.
  - Done when the repo explicitly chooses the canonical contract location, versioning model, and adapter ownership rules.
- [x] `C003` Introduce a canonical config package/module and one explicit compile boundary into router runtime config.
  - Done when new code can target the canonical contract without editing router runtime structs directly.
- [ ] `C004` Migrate Python CLI schema, parser, validator, and merger onto the canonical contract.
  - Done when handwritten `translate_*` and passthrough paths stop being the main way new config features move into router config.
- [ ] `C005` Replace dashboard dual-format config editing with canonical DTOs and backend validation.
  - Done when dashboard config save/load no longer depends on format detection or generic deep-merge semantics for the steady-state path.
- [ ] `C006` Rework DSL and CRD/operator translation to compile from the canonical contract instead of shuffling partially overlapping maps.
  - Done when CRD emission and operator reconciliation do not need to mirror independent handwritten config subsets for common router behavior fields.
- [ ] `C007` Add cross-surface contract tests and goldens.
  - Done when representative features are tested across CLI parse/validate, router compile/load, dashboard save/load, and Kubernetes translation.
- [ ] `C008` Ship migration tooling and versioned compatibility handling.
  - Done when legacy config can be upgraded deliberately and canonical config becomes the default write path.
- [ ] `C009` Remove temporary compatibility layers and close TD001.
  - Done when the debt item's exit criteria are met and the debt register can be updated in the same change.

## Initial Divergence Inventory

This section freezes the first loop-level inventory for `L001`. It is intentionally smaller than the final `C001` matrix: the goal here is to pin down the main contract seams, mismatch classes, and evidence before the ADR and vertical-slice work begin.

### Surface Snapshot

| Surface | Current primary shape | Evidence | Divergence class |
| --- | --- | --- | --- |
| Router runtime | Flat `RouterConfig` with embedded inline structs for infra and routing concerns | [src/semantic-router/pkg/config/config.go](../../src/semantic-router/pkg/config/config.go), [src/semantic-router/pkg/config/loader.go](../../src/semantic-router/pkg/config/loader.go), [src/semantic-router/pkg/config/validator.go](../../src/semantic-router/pkg/config/validator.go) | Runtime form also acts as the persisted config contract, so authoring and execution concerns are collapsed together. |
| Python CLI | Nested `UserConfig` with `providers` / `signals` / `decisions`, plus permissive top-level passthrough | [src/vllm-sr/cli/models.py](../../src/vllm-sr/cli/models.py), [src/vllm-sr/cli/parser.py](../../src/vllm-sr/cli/parser.py), [src/vllm-sr/cli/validator.py](../../src/vllm-sr/cli/validator.py), [src/vllm-sr/cli/merger.py](../../src/vllm-sr/cli/merger.py) | Typed authoring shape is only partial; unsupported fields bypass the schema and re-enter through merger passthrough. |
| Dashboard | Frontend owns duplicated TS config interfaces and supports both python-cli and legacy-flat payloads; backend reads and writes generic maps | [dashboard/frontend/src/types/config.ts](../../dashboard/frontend/src/types/config.ts), [dashboard/frontend/src/pages/configPageSupport.ts](../../dashboard/frontend/src/pages/configPageSupport.ts), [dashboard/backend/handlers/config.go](../../dashboard/backend/handlers/config.go) | UI editing contract is duplicated locally and accepts hybrid states instead of targeting one canonical write format. |
| DSL / YAML emission | DSL compiles to `RouterConfig`, then emitter denormalizes or rewraps maps into user YAML or CRD envelopes | [src/semantic-router/pkg/dsl/emitter_yaml.go](../../src/semantic-router/pkg/dsl/emitter_yaml.go), [src/semantic-router/pkg/dsl/dsl_test.go](../../src/semantic-router/pkg/dsl/dsl_test.go) | Translation is map-shuffling over the runtime struct rather than compilation from a distinct authoring contract. |
| K8s / Operator | `SemanticRouterSpec` splits router behavior across `spec.config` and `spec.vllmEndpoints`, while controllers synthesize additional config sections | [deploy/operator/api/v1alpha1/semanticrouter_types.go](../../deploy/operator/api/v1alpha1/semanticrouter_types.go), [deploy/operator/controllers/semanticrouter_controller.go](../../deploy/operator/controllers/semanticrouter_controller.go) | Platform contract is partial and adapter-owned fields are mixed with router authoring fields. |

### Mismatch Classes

- `M001` Shape ownership mismatch
  - `RouterConfig` is flat and runtime-oriented, but CLI authoring is nested under `providers`, `signals`, and `decisions`.
  - Dashboard keeps its own copies of that nested shape in both `src/types/config.ts` and `configPageSupport.ts`.
  - Operator `ConfigSpec` models only a subset of router behavior fields and pushes model/backend data into a separate `vllmEndpoints` surface.
- `M002` Hidden compatibility and passthrough
  - CLI `UserConfig` uses `extra = "allow"` for unsupported top-level fields, and merger copies `model_extra` back into merged router config.
  - Dashboard detects `python-cli`, `legacy`, and hybrid flattened states instead of enforcing one steady-state write contract.
  - `EmitCRD` preserves signal-rule keys that `ConfigSpec` does not explicitly model by moving extra fields into `spec.config`.
- `M003` Validation divergence
  - Router validation runs after YAML unmarshal of `RouterConfig`; in Kubernetes mode it skips decision/model validation during initial parse.
  - CLI validates twice: once on `UserConfig`, then again on the merged flat router dictionary with a different required-field set.
  - Dashboard validates only after JSON-to-map merge and YAML rewrite through the router parser, not through a dedicated dashboard/CLI contract validator.
  - CRD validation relies on kubebuilder JSON-schema annotations over `ConfigSpec`, which encode some constraints differently again.
- `M004` Write-path divergence
  - CLI write path is schema parse -> merge defaults -> translate nested fields back to flat router config.
  - Dashboard write path is generic deep-merge into the existing file, which means stale legacy keys and hybrid payloads can remain resident.
  - Operator reconciliation synthesizes `vllm_endpoints`, `model_config`, and even a `providers` block with placeholder endpoints, so the stored config is partially generated rather than purely authored.
  - DSL user-YAML emission denormalizes from `RouterConfig`, making DSL depend on the runtime form instead of an explicit authoring IR.
- `M005` Test coverage split
  - CLI tests cover local parse/merge validity and template behavior.
  - Dashboard handler tests cover HTTP update and merge semantics.
  - DSL tests cover DSL -> `RouterConfig` -> YAML round trips.
  - Operator tests cover `ConfigSpec` serialization and controller behavior.
  - There is still no canonical cross-surface golden proving one config concept survives CLI, dashboard, router, DSL, and operator paths without field loss or semantic drift.

### Field Buckets To Drive `C001`

- Canonical-candidate authoring fields
  - `version`, `listeners`, `providers`, `signals`, `decisions`, reasoning metadata, and most user-facing router behavior config.
- Adapter-owned fields
  - Kubernetes service discovery references, deployment/service/resources/autoscaling, and other platform-runtime knobs that should not define router authoring shape.
- Runtime-compiled fields
  - Flat `vllm_endpoints`, `model_config`, `categories`, `*_rules`, and similar denormalized router runtime structures.
- Legacy/hybrid compatibility fields
  - dashboard-recognized flat signal sections, root-level `model_config` / `vllm_endpoints`, merger passthrough blocks, and CRD extra-field preservation paths.
- Missing-from-shared-schema problem areas
  - any top-level field that still relies on CLI `model_extra`, dashboard `[key: string]: unknown`, or CRD map preservation instead of explicit shared typing.

### C001 Field Matrix

| Concern | Canonical authoring target | Router runtime / compiled form | CLI / dashboard authoring surface | Operator / platform surface | Bucket |
| --- | --- | --- | --- | --- | --- |
| Listeners | `listeners[]` | `APIServer.Listeners` | CLI `UserConfig.listeners`; dashboard root `listeners[]` | Kubernetes service ports live in `ServiceSpec`, not `ConfigSpec` | canonical plus adapter-owned platform split |
| Model endpoints | `providers.models[].endpoints[]` | `vllm_endpoints[]` plus `model_config[*].preferred_endpoints` | CLI `Providers.models[].endpoints`; dashboard `ProviderModel.endpoints` | `spec.vllmEndpoints[]`; controller synthesizes runtime `vllm_endpoints` and placeholder provider endpoints | canonical compiled into runtime, operator adapter-owned |
| Model metadata | `providers.models[].reasoning_family`, pricing, access key, param size, api format, capabilities | `model_config[*]` | CLI `Model`; dashboard `ProviderModel` / `ModelConfigEntry` duplicates | not modeled directly in `ConfigSpec`; controller rebuilds from endpoint specs and model config | canonical compiled into runtime; partial operator coverage |
| Default model | `providers.default_model` | root `default_model` | CLI nested under `providers`; dashboard supports both nested `providers.default_model` and root `default_model` | `ConfigSpec.default_model` plus controller-generated provider defaults | canonical plus legacy/hybrid duplication |
| Reasoning families | `providers.reasoning_families` | root `reasoning_families` | CLI nested in `providers`; dashboard duplicates in both nested providers and root unified config | `ConfigSpec.reasoning_families` | canonical plus root-level legacy/runtime duplication |
| Default reasoning effort | `providers.default_reasoning_effort` | root `default_reasoning_effort` | CLI nested in `providers`; dashboard duplicates nested and root forms | `ConfigSpec.default_reasoning_effort` | canonical plus root-level legacy/runtime duplication |
| Representative signal | `signals.keywords[]` | `keyword_rules[]` | CLI `Signals.keywords`; dashboard supports both nested `signals.keywords` and flat `keyword_rules` | not explicitly typed in `ConfigSpec`; CRD emission preserves as extra field | canonical compiled into runtime; missing shared operator schema |
| Representative decision | `decisions[]` | `decisions[]` over compiled rule names and runtime model refs | CLI `Decision`; dashboard duplicates `Decision` and `DecisionFormState` | `ConfigSpec.decisions[]` | mostly shared semantics, still handwritten per surface |
| Router-only compiled categories/domain view | not first-class in canonical first slice; derived from authoring signals/decisions | `categories[]` | dashboard legacy support reads `categories[]`; CLI may auto-generate during merge when domains are absent | may appear inside `spec.config` payloads and converted YAML | runtime-compiled plus legacy compatibility |
| Platform deployment knobs | none in canonical authoring contract | none in router behavior config | none in CLI/dashboard authoring path except deployment UX | `ServiceSpec`, `Resources`, `Autoscaling`, `Persistence`, `Ingress`, etc. | adapter-owned |
| Legacy flat top-level runtime keys | none | `vllm_endpoints`, `model_config`, `keyword_rules`, `categories`, other `*_rules` | dashboard unified config and backend merge path still accept these as editable/root keys | controllers and CRD emitters still generate or preserve them | legacy/hybrid compatibility |
| Unmodeled top-level passthrough | should become explicit canonical fields or explicit adapter-only fields | merged back into runtime dict unchanged | CLI `model_extra`, dashboard `[key: string]: unknown` | CRD extra-field preservation in `spec.config` | missing from shared schema |

### L003 Readiness Check

- The first vertical slice is now well-bounded:
  - providers/models/endpoints
  - `default_model`
  - `reasoning_families`
  - `default_reasoning_effort`
  - one direct-mapping signal (`signals.keywords` -> `keyword_rules`)
  - one decision path over that signal
- `L003` should not start by editing dashboard or operator flows directly.
- The correct next seam is `C003`: introduce a canonical config module plus a compile boundary for this first slice, then let `L003` wire one path through it.
- Conclusion: `C001` is complete and `L003` is now implemented for the first slice through a dedicated authoring seam.

### C003 First Slice Landed

- The repo now has a dedicated canonical authoring seam at [src/semantic-router/pkg/config/authoring](../../src/semantic-router/pkg/config/authoring).
- The first slice is versioned and strict:
  - `version`
  - `listeners`
  - `providers.models[].endpoints[]`
  - `providers.default_model`
  - `providers.reasoning_families`
  - `providers.default_reasoning_effort`
  - `signals.keywords[]`
  - `decisions[]` over the compiled keyword signal and provider models
- The seam compiles authoring config into runtime `RouterConfig` instead of extending the runtime structs with a second authoring shape.
- Unknown fields now fail fast at this seam instead of silently surviving via passthrough or map merge behavior.
- Current proof artifacts:
  - compile boundary: [src/semantic-router/pkg/config/authoring/compile.go](../../src/semantic-router/pkg/config/authoring/compile.go)
  - first-slice contract types: [src/semantic-router/pkg/config/authoring/types.go](../../src/semantic-router/pkg/config/authoring/types.go)
  - focused tests: [src/semantic-router/pkg/config/authoring/compile_test.go](../../src/semantic-router/pkg/config/authoring/compile_test.go)

### Existing Evidence and Coverage

- Shared first-slice fixtures: [config/testing/td001-first-slice-authoring.yaml](../../config/testing/td001-first-slice-authoring.yaml), [config/testing/td001-first-slice-runtime.yaml](../../config/testing/td001-first-slice-runtime.yaml)
- Authoring seam compilation and nested YAML shape coverage against the shared first-slice fixtures: [src/semantic-router/pkg/config/authoring/compile_test.go](../../src/semantic-router/pkg/config/authoring/compile_test.go)
- CLI parse/validate/merge contract coverage against the shared first-slice fixtures: [src/vllm-sr/tests/test_td001_contract_matrix.py](../../src/vllm-sr/tests/test_td001_contract_matrix.py)
- CLI coverage: [src/vllm-sr/tests/test_config_template.py](../../src/vllm-sr/tests/test_config_template.py)
- Dashboard handler coverage: [dashboard/backend/handlers/config_test.go](../../dashboard/backend/handlers/config_test.go)
- DSL round-trip coverage: [src/semantic-router/pkg/dsl/dsl_test.go](../../src/semantic-router/pkg/dsl/dsl_test.go)
- Operator schema coverage: [deploy/operator/api/v1alpha1/semanticrouter_types_test.go](../../deploy/operator/api/v1alpha1/semanticrouter_types_test.go)

### L004 Contract Matrix Landed

- The first slice now has shared fixture-backed contract tests instead of independent inline examples.
- Router-side proof:
  - canonical authoring YAML compiles into a shared expected runtime slice
  - the compiled runtime slice emits back into the shared nested user-YAML shape for `listeners`, `signals`, `providers`, and `decisions`
- CLI-side proof:
  - the same authoring fixture parses and validates through `UserConfig`
  - merger now carries the first-slice listener/runtime fields needed by the shared runtime fixture:
    - `listeners`
    - `preferred_endpoints`
    - model metadata used by the first slice (`description`, `capabilities`, `quality_score`)
    - `None` leaf fields are no longer serialized into decision rule trees
- `L004` is therefore complete for the initial TD001 slice, but this does not close `C004` or `C007`.

- Current gap statement
  - The repo now has one shared first-slice contract suite across router authoring compile, CLI merge, and DSL/user-YAML emission, but dashboard save/load and operator/CRD translation are still outside that shared fixture matrix.
  - `version` is still authoring-only metadata, so a full canonical round-trip cannot be claimed until the migration path decides where version persistence lives outside the runtime `RouterConfig`.
  - CLI still writes the steady-state router file by translating into the legacy flat runtime dict, so `C004` remains open even though the first slice is now aligned and covered.

## Current Loop

- [x] `L001` Freeze the initial divergence inventory for the TD001 surfaces listed in the debt register.
  - Focus on duplicated field definitions, type mismatches, validation mismatches, and hidden passthrough behavior.
- [x] `L002` Write the ADR that chooses the canonical contract and adapter boundaries.
  - Include explicit treatment for router runtime config, dashboard DTOs, and Kubernetes-native backend references.
- [x] `L003` Land one narrow vertical slice through the new seam.
  - Recommended first slice: providers/default model/reasoning families plus one signal and one decision path.
- [x] `L004` Add a contract round-trip test matrix for that slice before expanding to more fields.
  - The first slice should prove the seam before bulk migration starts.

## Decision Log

- Start with inventory and contract ownership, not bulk field moves.
- `2026-03-08`: `L001` completed with an initial divergence inventory that separates canonical-candidate authoring fields, adapter-owned fields, runtime-compiled fields, and legacy/hybrid compatibility seams.
- `2026-03-08`: `L002` completed by accepting ADR 0003, which chooses a versioned canonical authoring contract and keeps `RouterConfig` as the runtime-compiled form during migration.
- `2026-03-08`: `C001` completed with a durable field matrix for the first vertical slice and explicit classification of canonical, adapter-owned, runtime-compiled, legacy, and missing-schema fields.
- `2026-03-08`: `C003` and `L003` completed by landing `pkg/config/authoring`, a strict versioned authoring slice that compiles listeners/providers/keyword-signals/decisions into runtime `RouterConfig` without extending the runtime contract itself.
- `2026-03-08`: `L004` completed by landing shared first-slice fixtures plus router/CLI contract tests, and by tightening the CLI merger so the slice now preserves listeners, endpoint preferences, key model metadata, and clean decision rule trees.
- `2026-03-08`: `C004` advanced by replacing arbitrary CLI top-level passthrough with an explicit compatibility allowlist shared by parser and merger; unknown top-level blocks now fail during CLI parse instead of silently flowing through `model_extra`.
- `2026-03-08`: `C004` advanced again by moving `prompt_guard` off raw CLI passthrough onto a typed parser/merger seam, shrinking the remaining `model_extra` compatibility surface to blocks that still lack explicit CLI ownership.
- `2026-03-08`: `C004` advanced again by moving `tools` off raw CLI passthrough onto the same typed compat seam, including explicit support for `advanced_filtering`, so two stable router-owned top-level blocks now bypass `model_extra`.
- `2026-03-08`: `C004` advanced again by moving `observability` off raw CLI passthrough onto the typed compat seam, including nested metrics/tracing validation so router-owned telemetry config now bypasses `model_extra` as well.
- `2026-03-09`: `C004` advanced again by moving `looper` off raw CLI passthrough onto the typed compat seam, preserving the existing transitional `enabled` field while explicitly validating looper runtime knobs like `model_endpoints`, retries, and gRPC sizing.
- `2026-03-09`: `C004` advanced again by moving `router_replay` off raw CLI passthrough onto the typed compat seam, including explicit validation for backend selection and backend-specific redis/postgres/milvus storage settings.
- `2026-03-09`: `C004` advanced again by moving `response_api` off raw CLI passthrough onto the typed compat seam, including explicit validation for redis and milvus backend settings used by the response storage path.
- `2026-03-09`: `C004` advanced again by moving `authz` and `ratelimit` off raw CLI passthrough onto the typed compat seam, so credential-resolution and rate-limit provider blocks now parse through explicit provider/rule schemas instead of `model_extra`.
- `2026-03-09`: `C004` advanced again by moving `semantic_cache` off raw CLI passthrough onto the typed compat seam, including explicit nested redis and milvus backend shapes so cache backend config no longer relies on untyped `model_extra`.
- `2026-03-09`: `C004` advanced again by moving stable runtime router option keys (`auto_model_name`, `clear_route_cache`, streaming knobs, and config-list visibility) off raw top-level passthrough onto an explicit typed compat seam, reducing `model_extra` usage for non-block runtime toggles as well.
- `2026-03-09`: `C004` advanced again by moving `bert_model` and `feedback_detector` off raw CLI passthrough onto explicit typed compat schemas, shrinking `model_extra` for inline model settings that are already stable in router runtime config.
- `2026-03-09`: `C004` advanced again by restoring steady-state CLI merge support for nested `signals.modality` and `signals.role_bindings`, and by aligning authz signal validation with runtime semantics so `type: "authz"` conditions reference emitted `role` values instead of binding names.
- `2026-03-09`: `C004` advanced again by moving `hallucination_mitigation` and `modality_detector` off raw CLI passthrough onto explicit typed compat schemas, so modality detection and hallucination inline-model settings now bypass `model_extra` as well.
- `2026-03-09`: `C004` advanced again by moving `classifier` off raw CLI passthrough onto the typed compat seam, including `category_model`, `mcp_category_model`, `pii_model`, and `preference_model` support so one of the most common inline-model blocks no longer depends on the legacy passthrough path.
- `2026-03-09`: `C004` advanced again by moving `image_gen_backends` and `provider_profiles` off raw CLI passthrough onto explicit typed compat map schemas, so backend-model reference maps used by modality/image generation and external provider routing now bypass `model_extra` too.
- `2026-03-09`: `C004` advanced again by moving `config_source`, `mom_registry`, `strategy`, and `vector_store` off raw CLI passthrough onto explicit typed compat seams, shrinking the remaining legacy top-level surface to keys that overlap more directly with canonical provider/signal/runtime compilation paths.
- `2026-03-09`: While evaluating the remaining raw top-level keys, `api` was intentionally left on the legacy path because the steady-state YAML/dashboard shape (`batch_classification.max_batch_size`, `concurrency_threshold`, `max_concurrency`) does not currently match the narrow Go `APIConfig` struct in one place, so this block now sits in the explicit “cross-surface drift” bucket rather than the “safe typed-compat extraction” bucket.
- `2026-03-09`: `C004` advanced again by reconciling that `api` drift: Go runtime `APIConfig` now explicitly models the same batch-classification limit fields already present in operator/dashboard/YAML surfaces, and the CLI parser/merger now moves `api` through an explicit typed compat seam instead of raw passthrough.
- `2026-03-09`: `C004` advanced again by moving legacy root provider defaults (`default_model`, `default_reasoning_effort`, `reasoning_families`) off raw CLI passthrough onto an explicit typed seam, leaving the remaining raw top-level list dominated by compiled runtime keys plus the still-separate `model_selection` subsystem.
- `2026-03-09`: `C004` advanced again by teaching `parse_user_config()` to normalize the shared first-slice legacy runtime keys (`keyword_rules`, `model_config`, `vllm_endpoints`, and the root provider-default aliases) into the nested `UserConfig` shape before validation, and by extending the shared contract matrix so the runtime fixture now round-trips through the same dual-read authoring seam instead of `model_extra`.
- `2026-03-09`: `C004` advanced again by expanding that parser normalization seam across the remaining signal/runtime aliases (`categories`, `embedding_rules`, `fact_check_rules`, `user_feedback_rules`, `preference_rules`, `language_rules`, `context_rules`, `complexity_rules`, `modality_rules`, `role_bindings`, `jailbreak`, and `pii`), so valid legacy signal blocks now dual-read through nested `signals.*` instead of the top-level passthrough path.
- `2026-03-09`: `C004` advanced again by shrinking the merger passthrough allowlist to the still-separate `model_selection` subsystem; normalized legacy signal/provider-runtime keys must now either enter through the parser normalization seam or fail, instead of silently surviving as top-level extras.
- `2026-03-09`: `C004` advanced again by moving `model_selection` off the last named raw passthrough path onto an explicit typed compat schema that matches the dashboard/runtime shape (`enabled`, `method`, `elo/router_dc/automix/hybrid/ml`), so the merger now rejects unexpected top-level extras entirely.
- `2026-03-09`: `C004` is still intentionally left open after that seam reduction because the embedded CLI `router-defaults.yaml` reference still carries an older `model_selection` shape than the runtime/dashboard path, and the steady-state CLI write path still compiles through handwritten translation into the legacy flat router dict.
- Use dual-read, single-write migration wherever compatibility is needed.
- Keep `RouterConfig` runtime-oriented until the canonical contract and compiler path are stable.
- Prefer generated/shared bindings where practical; if generation is deferred, keep adapters in dedicated seam modules rather than scattering translation logic back into hotspots.
- Any unavoidable platform-specific divergence must be named, adapter-owned, and covered by tests instead of hidden in dashboard helpers or merger passthroughs.

## Follow-up Debt / ADR Links

- [../tech-debt/TD001-config-surface-fragmentation.md](../tech-debt/TD001-config-surface-fragmentation.md)
- [../adr/0003-td001-canonical-config-contract.md](../adr/0003-td001-canonical-config-contract.md)
