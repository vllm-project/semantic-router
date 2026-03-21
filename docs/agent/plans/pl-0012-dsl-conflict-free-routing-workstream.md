# DSL Conflict-Free Routing Workstream Execution Plan

## Goal

- Turn [`spec/dsl.md`](../../../spec/dsl.md) into the canonical resumable workstream for conflict-free and confidence-aware routing behavior.
- Close the gap between the DSL surface that already parses and validates `SIGNAL_GROUP`, `TEST`, and `TIER`, and the runtime surfaces that must enforce those semantics.
- Retire this workstream only when validator, compiler, runtime signal evaluation, decision selection, and tests all agree on the same routing behavior.

## Scope

- `spec/dsl.md`
- `src/semantic-router/pkg/dsl/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/decision/**`
- `src/semantic-router/pkg/config/**`
- `e2e/**` and other affected test surfaces needed for behavior-visible routing changes
- nearest local `AGENTS.md` for touched hotspot trees, especially `src/semantic-router/pkg/config/AGENTS.md`

## Exit Criteria

- The repository has one durable execution record for this spec-backed workstream and it points to the real spec file path in the repo.
- Validator behavior for domain overlap, mutual exclusion guidance, `SIGNAL_GROUP`, `TEST`, and `TIER` is covered by targeted tests and stays aligned with the runtime contract.
- Runtime signal evaluation enforces grouped exclusivity where `signal_groups` declare it, or any remaining unsupported behavior is promoted into indexed technical debt in the same change.
- Decision selection preserves categorical precedence across tiers and uses score-aware selection within a tier where the spec requires it.
- All applicable harness gates for the touched files pass, including required E2E or integration coverage for behavior-visible routing changes.

## Task List

- [x] `W001` Bootstrap the indexed execution plan, reconcile the spec path mismatch, and lock the first implementation loop to repo-native harness inputs.
- [x] `W002` Audit which `spec/dsl.md` features are already implemented in DSL/parser/compiler/validator code versus which runtime behaviors are still missing.
- [x] `W003` Implement runtime `signal_groups` enforcement for grouped domain and embedding signals, with focused classification and decision tests.
- [x] `W004` Implement tier-aware decision selection so cross-tier precedence stays categorical and within-tier conflicts are resolved by score rather than raw priority.
- [x] `W005` Close the remaining authoring/runtime gap for grouped signals, including any missing validator checks, runtime plumbing, docs, and targeted behavior coverage.
- [x] `W006` Run the full applicable harness validation ladder for the final changed-file set, update debt if any durable architecture gap remains, and mark the workstream complete.
- [x] `W007` Sync `routing.signals.signal_groups` into the canonical config contract, exhaustive reference config, and cross-surface schema/docs coverage, then rerun the applicable feature gate.
- [x] `W008` Implement runtime-backed `TEST` block validation for actionable `spec/dsl.md` coverage, plus any DSL AST/editor contract updates needed so `TEST` and related routing-only constructs round-trip cleanly across tooling.
- [x] `W009` Close the remaining actionable `spec/dsl.md` signal-group gaps by enforcing `SIGNAL_GROUP.default` as a runtime fallback contract and adding native centroid-similarity validation for `softmax_exclusive` embedding groups.
- [x] `W010` Re-audit the newly requested `spec/dsl.md` expansion scope: `DECISION_TREE` / `IF ELSE`, remaining stubbed core APIs, and maintained router/DSL example assets including `deploy/recipes/balance.yaml`.
- [x] `W011` Implement minimal `DECISION_TREE` / `IF ELSE` DSL support with tests, compiling it into the existing router decision model without a second runtime.
- [x] `W012` Implement the remaining core classification/config API handlers and tests so the published API surface no longer advertises stubbed config/classification endpoints.
- [x] `W013` Add maintained router-config and DSL-config examples for the new capabilities, and update `deploy/recipes/balance.yaml` so the shipped recipe visibly exercises the new routing surface.
- [x] `W014` Run the full applicable harness ladder for the expanded spec surface, then close all implementation-complete tasks once their shared feature gate passes.
- [ ] `W015` Replace invalid `balance.yaml` demo domains with repo-supported routing domains, and add early domain-value validation across config plus DSL compiler/validator surfaces.
- [ ] `W016` Rework `deploy/recipes/balance.yaml` to use repo-native learned-signal + heuristic-signal + `SIGNAL_GROUP` routing patterns that better match `.augment/clawrouter.md` without inventing a second runtime.
- [ ] `W017` Close the PR CI gaps for the new branch, including alternate Go modfile drift and any affected tests/gates, then rerun the applicable harness ladder until the branch is green.

## Current Loop

- Date: 2026-03-21
- Current task: `W015` in progress
- Changed files:
  - `deploy/recipes/balance.yaml`
  - `docs/agent/plans/pl-0012-dsl-conflict-free-routing-workstream.md`
  - `src/semantic-router/go.onnx.mod`
  - `src/semantic-router/pkg/config/validator.go`
  - `src/semantic-router/pkg/dsl/compiler.go`
  - `src/semantic-router/pkg/dsl/validator.go`
  - `src/semantic-router/pkg/dsl/validator_conflicts.go`
  - `src/semantic-router/pkg/dsl/dsl_test.go`
  - `src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go`
  - `src/semantic-router/pkg/config/config_test.go`
- Commands run:
- Failure observed:
  - `deploy/recipes/balance.yaml` currently declares `balance_demo_compact`, `balance_demo_deep`, and `balance_demo_default`, which are not part of the repo-supported 14 routing domains documented in `e2e/profiles/ml-model-selection/README.md`.
  - PR `#1620` currently fails broad CI because alternate build paths use `src/semantic-router/go.onnx.mod`, which did not pick up the new DSL parser dependency `github.com/alecthomas/participle/v2`.
- Fix applied:
  - none yet; this loop is still at the context-and-root-cause stage
- Current result:
  - The smallest actionable subtask is `W015`: centralize the allowed routing-domain contract and reject invalid domain values before runtime in config plus DSL flows.
  - Candidate files for this subtask are `deploy/recipes/balance.yaml`, `src/semantic-router/go.onnx.mod`, `src/semantic-router/pkg/config/validator.go`, `src/semantic-router/pkg/dsl/compiler.go`, `src/semantic-router/pkg/dsl/validator.go`, `src/semantic-router/pkg/dsl/validator_conflicts.go`, `src/semantic-router/pkg/dsl/dsl_test.go`, `src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go`, and `src/semantic-router/pkg/config/config_test.go`.
- Next action:
  - Run `make agent-report ENV=cpu CHANGED_FILES="deploy/recipes/balance.yaml,src/semantic-router/go.onnx.mod,src/semantic-router/pkg/config/validator.go,src/semantic-router/pkg/dsl/compiler.go,src/semantic-router/pkg/dsl/validator.go,src/semantic-router/pkg/dsl/validator_conflicts.go,src/semantic-router/pkg/dsl/dsl_test.go,src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go,src/semantic-router/pkg/config/config_test.go"` and then implement the smallest domain-contract fix first.

## Decision Log

- 2026-03-21: treat [`spec/dsl.md`](../../../spec/dsl.md), not `spec/spec.md`, as the canonical spec input because the latter path does not exist in the repository.
- 2026-03-21: keep this as an execution plan rather than an ADR because the repository is still executing implementation loops rather than recording a settled architecture decision.
- 2026-03-21: start by auditing the already-landed DSL/compiler/validator support before editing runtime code, because duplicating partially implemented spec work would create avoidable churn and risk.
- 2026-03-21: the audit confirmed that the remaining implementation gap is primarily runtime behavior, not DSL syntax or validator scaffolding. The next loop should start from classification and decision runtime seams rather than reopening parser/compiler files without evidence.
- 2026-03-21: implement `W003` as grouped pruning over already co-firing matched signals instead of a broader threshold/default rewrite. That keeps the extraction change narrow, preserves existing unmatched behavior, and still removes silent multi-match conflicts before the decision engine.
- 2026-03-21: `e2e/pkg/cluster/kind.go` deliberately reuses an existing Kind cluster when it already exists, so feature-gate reruns must clean `semantic-router-e2e` first when prior attempts may have left durable cluster drift.
- 2026-03-21: fix the `W003` OOM at the request-time signal evaluation seam instead of in the E2E values file alone; the profile already carried concurrency knobs, but the runtime config surface was not parsing or enforcing them.
- 2026-03-21: keep the load gate state outside `Classifier` with a lazy sidecar map so the hotspot orchestrator does not grow and the extraction stays within the local structure rules.
- 2026-03-21: implement `W004` with tier-first ordering in `src/semantic-router/pkg/decision/engine.go`; `Priority` remains only a same-tier tiebreaker so lower-tier categorical precedence cannot be accidentally overridden by a higher raw priority.
- 2026-03-21: keep `W005` narrow by rejecting impossible or unsupported `SIGNAL_GROUP` authoring patterns in the validator now, and promote the still-missing `SIGNAL_GROUP.default` runtime coverage contract to `TD035` instead of broadening runtime behavior without a dedicated migration loop.
- 2026-03-21: when the `W005` feature-gate build failed against external registries but direct `docker pull` succeeded, treat the event as transient infrastructure noise and rerun the same gate rather than weakening validation or changing runtime code without evidence.
- 2026-03-21: full changed-set validation must stay green for harness-owned files too; excluding `.venv-agent` from `shellcheck` and extracting helpers from `e2e/pkg/cluster/kind.go` was the smallest compliant way to keep the canonical lint gate passing.
- 2026-03-21: repeated local E2E and feature-gate loops must use `E2E_USE_WORKSPACE_MODELS=true` so workspace models are reused across reruns and the validation loop does not spend time redownloading identical model assets.
- 2026-03-21: the unsynced config surface was not `global.services.api.batch_classification`; those knobs were already present in `config/config.yaml`. The real omission was `routing.signals.signal_groups`, which canonical `routing.signals` and its schema mirrors were still dropping.
- 2026-03-21: syncing `signal_groups` requires the full config-platform path together: Go canonical import/export, the exhaustive reference config, CLI schema, dashboard config typing, and the public canonical-config docs. Patching `config/config.yaml` alone would keep the field invisible on round-trip.
- 2026-03-21: `W007` is externally blocked at the feature-gate stage until Docker is reachable again. The active context is `desktop-linux`, but `/Users/bitliu/.docker/run/docker.sock` is missing, so `make vllm-sr-test-integration` cannot build the CLI integration image.
- 2026-03-21: `spec/dsl.md` is not a flat "implement everything literally" checklist. Its own action plan explicitly excludes `E1` and `E2` and treats `E3a` as retraining work, so the actionable product gap to close in-repo is `M4` (`TEST` blocks) plus any adjacent tooling-contract holes, not a speculative DSL redesign.
- 2026-03-21: implement `TEST` blocks through a narrow runtime-backed validation seam that reuses classification signal evaluation and decision selection rather than inventing a second router pipeline inside the validator.
- 2026-03-21: `W008` moved native routing validation into `src/semantic-router/cmd/dsl` rather than `pkg/dsl` so dashboard/backend and other shared DSL consumers keep a clean package boundary without pulling classification runtime dependencies.
- 2026-03-21: the DSL package already sits behind explicit relaxed structure rules for `parser.go`, `decompiler.go`, and `dsl_test.go`; adding matching file-scoped exclusions to `.golangci.agent.yml` is the smallest repo-native way to keep changed-file Go lint focused on new regressions instead of re-reporting the same hotspot debt on every DSL repair branch.
- 2026-03-21: `agent-feature-gate` for both `W007` and `W008` is a real external blocker, not a code failure. It reaches `make vllm-sr-dev` and then fails because the configured Docker socket `unix:///Users/bitliu/.docker/run/docker.sock` is unavailable.
- 2026-03-21: the remaining actionable `spec/dsl.md` gap after `W008` is not another parser/compiler feature. It is the incomplete `SIGNAL_GROUP` contract: runtime fallback still ignores `default`, and centroid-similarity warnings for `softmax_exclusive` groups still need a native validation seam that can use the embedding model without polluting shared DSL packages.
- 2026-03-21: a full-spec audit after the `W009` code landed confirms there is no additional repository-local implementation gap in `spec/dsl.md`; the only unimplemented items left in the document are the spec's own intentionally out-of-scope `E1`, `E2`, and `E3a` tracks.
- 2026-03-21: `make agent-ci-gate` now passes for the `W009` changed-file set, and the only remaining blocker for `W007`/`W008`/`W009` closure is the unchanged Docker daemon failure at `unix:///Users/bitliu/.docker/run/docker.sock` during `make vllm-sr-dev`.
- 2026-03-21: the user explicitly expanded the required scope beyond the original action-plan cut line in `spec/dsl.md`; this workstream now needs a minimal repo-native implementation of `DECISION_TREE` / `IF ELSE`, real config/classification HTTP handlers, and maintained router/DSL example assets rather than treating those areas as permanently out of scope.
- 2026-03-21: the narrowest viable path for `DECISION_TREE` is to treat it as DSL authoring sugar that lowers into the existing flat `config.Decision` list. That avoids inventing a second router runtime, keeps the decision engine unchanged, and lets the same config/API surfaces continue to operate on canonical router config.
- 2026-03-21: `GET/PUT /config/classification` must merge through canonical config export/import rather than marshaling `RouterConfig` directly, otherwise partial routing payloads drop decisions and other existing routing state on write-back.
- 2026-03-21: the new conflict-free routing examples should be maintained as a paired DSL source and compiled YAML artifact under test, because the runtime/API surface is canonical config while the authoring surface now includes `DECISION_TREE`.
- 2026-03-21: `DECISION_TREE` is implemented as authoring sugar only. `DecompileRouting()` still reconstructs flat `ROUTE` blocks from `config.Decision`, so lossless tree round-trip remains a durable architecture gap and is now tracked in `TD036` instead of growing a parallel runtime model inside this loop.
- 2026-03-21: new debt entries are part of the canonical harness surface, not just the `docs/agent/tech-debt/` directory. `tools/agent/repo-manifest.yaml` must list them under both `docs` and `doc_governance.canonical_docs` or `make agent-validate` will fail.
- 2026-03-21: the Docker socket issue was environmental, not a code regression. After restoring Docker Desktop and reusing workspace models, the exact same feature-gate command passed without further product changes.
- 2026-03-21: `W007` through `W014` close together on the shared feature gate. Once `vllm-sr-test-integration`, local smoke, and the `ai-gateway` E2E profile were green on the full changed-file set, the remaining implementation-complete tasks and workstream exit criteria were all satisfied.

## Follow-up Debt / ADR Links

- [TD006 Structural Rule Target Still Exceeds Reality in Key Legacy Hotspots](../tech-debt/td-006-structural-rule-target-vs-legacy-hotspots.md)
- [TD015 Weak Typing Still Leaks Through Dashboard Editor Models and DSL Serialization Helpers](../tech-debt/td-015-weakly-typed-config-and-dsl-contracts.md)
- [TD020 Classification Subsystem Boundaries Have Collapsed Into Hotspot Orchestrators](../tech-debt/td-020-classification-subsystem-boundary-collapse.md)
- [TD035 SIGNAL_GROUP Default Coverage Contract Is Still Declarative Only](../tech-debt/td-035-signal-group-default-coverage-contract-gap.md)
- [TD036 Decision Tree Authoring Cannot Round-Trip Through Runtime Config](../tech-debt/td-036-decision-tree-authoring-roundtrip-gap.md)
