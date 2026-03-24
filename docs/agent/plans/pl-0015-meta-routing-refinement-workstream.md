# Meta-Routing Refinement Workstream Execution Plan

## Goal

- Introduce an optional request-phase meta-routing seam that wraps the current `signal -> projection -> decision -> model selection` pipeline with assess-and-refine control flow.
- Keep the existing routing layers intact while adding deterministic reliability assessment, targeted subgraph recomputation, and durable pass-level feedback records.
- Close this workstream only when the repository exposes one coherent `routing.meta` contract across runtime, config, DSL, CLI, dashboard, maintained assets, docs, and validation surfaces.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `src/semantic-router/pkg/extproc/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/decision/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/dsl/**`
- `src/vllm-sr/cli/**`
- `dashboard/frontend/src/**`
- `config/**`
- `website/docs/**`
- offline calibration or replay surfaces under the repository training stack
- targeted tests, integration coverage, and any affected E2E scenarios needed for behavior-visible routing changes
- nearest local `AGENTS.md` files for touched hotspot trees, especially `src/semantic-router/pkg/extproc/`, `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/classification/`, `src/vllm-sr/cli/`, and `dashboard/frontend/src/`

## Exit Criteria

- The repository has one durable architecture record for the meta-routing boundary and one indexed execution plan that can be resumed without chat context.
- The canonical routing contract exposes an optional `routing.meta` section with only mode, maximum pass budget, trigger policy, and allowed refinement actions.
- Router runtime owns a request-phase meta-routing seam with repo-native collaborators for base-pass execution, assessment, planning, refinement, policy lookup, and feedback persistence.
- The runtime emits stable pass-level artifacts named `RoutingTrace`, `PassTrace`, `MetaAssessment`, `RefinementPlan`, and `FeedbackRecord`.
- `observe` and `shadow` modes never change the final route. `active` mode changes routing only through declared refinement actions and bounded pass budgets.
- V1 refinement remains selective subgraph recomputation. Signal, projection, decision, and model-selection layers stay narrow and independently testable.
- Maintained config assets, DSL round-trips, CLI typed models, dashboard config editing support, and public docs all describe the same `routing.meta` contract.
- Offline replay or calibration inputs can consume `FeedbackRecord` without changing the runtime seam again.
- Applicable repo-native validation gates cover schema, runtime, transport surfaces, docs-contracts, and behavior-visible routing outcomes.

## Target Design

### Ownership Boundary

- Meta-routing is owned by a new request-phase orchestrator seam in `pkg/extproc`.
- Signal packages continue to extract facts.
- Projection packages continue to derive routing state from signal results.
- Decision packages continue to evaluate boolean rules.
- Selection packages continue to choose among decision candidate models.
- The orchestrator may re-run selected work, but those lower layers do not own retry logic or feedback policy.

### Runtime Flow

- `BasePassRunner` executes the existing one-pass routing flow.
- `MetaAssessor` scores the reliability of the base-pass intermediate state.
- `RefinementPlanner` converts trigger output plus configured allowed actions into a bounded `RefinementPlan`.
- `RefinementExecutor` re-runs only the affected signal families and dependent projection, decision, and selection work.
- `FeedbackSink` emits one durable `FeedbackRecord` per request in `observe`, `shadow`, and `active` modes.

The fixed flow is:

- base pass
- meta assessment
- optional targeted refinement
- final decision and model selection
- feedback emission

### Public Contract

The `routing.meta` contract stays intentionally narrow:

- `mode`
  - `observe`
  - `shadow`
  - `active`
- `max_passes`
- `trigger_policy`
- `allowed_actions`

Backward compatibility rules:

- `routing.meta` is optional
- absence means disabled one-pass routing
- any rollout-specific behavior is gated by `mode`, not by adding parallel booleans elsewhere in the config

### Stable Runtime Artifacts

- `RoutingTrace`
  - request-level summary across all passes
- `PassTrace`
  - one record per routing pass with inputs, derived state, outputs, timing, and trigger observations
  - pass-quality fields are fixed for this workstream:
    - `signal_dominance`
    - `avg_signal_confidence`
    - `decision_margin`
    - `projection_boundary_min_distance`
    - `fragile`
- `MetaAssessment`
  - deterministic assessment output for whether refinement is needed and why
  - field shape is fixed to:
    - `needs_refine`
    - `triggers`
    - `root_causes`
    - `trace_quality`
- `RefinementPlan`
  - bounded list of allowed refinement actions plus affected signal families and derived recomputation targets
- `FeedbackRecord`
  - durable per-request record joining request features, pass traces, final route and model, plugin outcomes, and downstream weak labels
  - envelope is fixed to:
    - `observation`
    - `action`
    - `outcome`

### V1 Trigger Policy

The deterministic v1 trigger set is:

- low decision margin
- projection score near mapping boundary
- conflicting partition evidence
- required family missing
- required family low-confidence
- disagreement between cheap and expensive signal families

The v1 root-cause taxonomy carried by `MetaAssessment` is:

- `missing_required_family`
- `low_confidence_family`
- `projection_boundary_pressure`
- `decision_overlap`
- `partition_conflict`
- `family_disagreement`
- `compression_loss_risk`

### V1 Refinement Actions

The allowed v1 refinement actions are:

- re-run selected high-cost semantic signal families
- re-evaluate with less lossy input preparation
- recompute affected projections
- rerun decision and model selection

Explicit non-goals for v1:

- no token-level regenerate loop
- no free-form agent planning
- no direct use of meta outputs inside decision-rule expressions

### Trace, Telemetry, and Feedback

Pass-aware request context and telemetry should record:

- pass count
- trigger names
- refined families
- overturned decision flag
- decision-margin deltas
- projection-boundary deltas
- latency deltas
- final outcome summary

`FeedbackRecord` should join:

- request and routing features
- pass traces
- final decision and selected model
- plugin outcomes
- downstream weak labels such as fallback, block, cache hit or miss, user feedback, and response-side safety or fact-check outcomes

`FeedbackRecord` should keep the data model stable for later replay:

- `observation`
  - request identifiers, request model, route-relevant features, and pass traces
- `action`
  - planned refinement actions, executed actions, and pass-budget usage
- `outcome`
  - final route and model, cache and block outcomes, and downstream weak labels

### Offline Lane

The offline lane should start with heuristics plus calibration, not learned runtime dependence.

The initial consumers of `FeedbackRecord` are:

- trigger-policy calibration
- refinement-action calibration
- projection and decision calibration
- replay evaluation for overturn rate, latency overhead, and weak-label alignment

## Task List

- [x] `MRR001` Create the indexed ADR and execution plan, register both inventories, and lock the meta-routing boundary before implementation.
- [x] `MRR002` Define `routing.meta` in Go config, canonical import/export, routing-only loaders, validators, reference config coverage, and maintained config assets.
- [x] `MRR003` Define the stable runtime artifacts and extend request context plus telemetry with pass-level trace state.
- [x] `MRR004` Add a repo-native request-phase meta-routing orchestrator seam in `pkg/extproc` with narrow collaborators and no behavior change when disabled.
- [x] `MRR005` Implement deterministic meta assessment for the v1 trigger set and run it in `observe` mode first.
- [x] `MRR006` Implement bounded refinement planning and targeted subgraph recomputation for `shadow` and `active` modes without pushing retry logic into signal, projection, or decision packages.
- [x] `MRR007` Expose `routing.meta` across DSL parser, AST, compiler, decompiler, JSON export, and validator surfaces.
- [x] `MRR008` Expose `routing.meta` across Python CLI typed models, config validators, and any related transport or schema mirrors.
- [x] `MRR009` Add dashboard typing and config editing support for `routing.meta` without growing existing page hotspots in responsibility.
- [x] `MRR010` Persist `FeedbackRecord`, add initial offline feature extraction and replay surfaces, and keep the learned-policy seam interface-stable.
- [x] `MRR011` Update maintained docs and tutorials so runtime behavior, YAML, DSL, CLI, and dashboard guidance all tell the same story.
- [x] `MRR012` Run the applicable validation ladder for the changed-file set, add focused runtime and contract coverage, and promote any unresolved durable gap into indexed technical debt.
- [x] `MRR013` Evaluate whether the post-v1 learned-policy loop needs a follow-on ADR or debt item before enabling non-deterministic policy providers.

## Current Loop

- Date: 2026-03-24
- Current task: `MRR012` completed
- Branch: `main`
- Planned loop order:
  - `L1` lock the boundary and execution record
  - `L2` land config and canonical contract support
  - `L3` land runtime trace spine and request-phase seam
  - `L4` land deterministic assess-and-plan behavior
  - `L5` land targeted refinement execution
  - `L6` land DSL, CLI, and dashboard contract coverage
  - `L7` land maintained docs, examples, offline replay inputs, and learned-policy follow-on debt
  - `L8` run the validation ladder and close the workstream
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/README.md`, `docs/agent/plans/README.md`, local `AGENTS.md` files, and the relevant harness skills and playbooks
  - `make agent-report ENV=cpu CHANGED_FILES="src/semantic-router/pkg/extproc/request_context.go,src/semantic-router/pkg/extproc/req_filter_classification.go,src/semantic-router/pkg/classification,src/semantic-router/pkg/decision,src/semantic-router/pkg/config,src/semantic-router/pkg/dsl,src/vllm-sr/cli,dashboard/frontend/src/pages,config"`
  - broad `codebase-retrieval` across runtime, config, DSL, CLI, dashboard, docs, and nearby tests for the meta-routing workstream
  - targeted source reads for the current request routing pipeline, classification and projection seams, decision ranking, canonical config surfaces, DSL routing contract, CLI models, dashboard config types, and reference-config tests
  - `go test ./pkg/classification -run 'TestRefreshSignalFamiliesRecomputesProjections|TestRefreshSignalFamiliesReplacesPIIMetadata|TestApplyProjectionsAddsDerivedOutputsAndScores|TestSignalGroupSoftmaxExclusiveChoosesSingleDomainWinner'`
  - `go test ./pkg/extproc -run 'TestChooseMetaRoutingFinalPassHonorsMode|TestChooseMetaRoutingFinalPassFallsBackWhenRefinementHasNoDecision|TestEmitMetaRoutingFeedbackPersistsOnce|TestRecordMetaRoutingBasePassCapturesAssessmentAndPlan|TestRecordMetaRoutingBasePassNoopWhenDisabled'`
  - focused source reads for pass-quality inputs and trigger observability under `pkg/extproc`, `pkg/decision`, and `pkg/classification`
  - targeted validation for `pkg/config`, `pkg/dsl`, Python CLI schema, dashboard checks, and `make agent-validate`
  - `go test ./pkg/decision ./pkg/classification ./pkg/extproc`
  - `go test ./pkg/extproc -run 'TestHandleMetaRoutingFeedback|TestCollectMetaRoutingFeedbackRecordsReturnsParsedPayloads'`
  - `python3 -m pytest src/vllm-sr/tests/test_meta_routing_validation.py src/vllm-sr/tests/test_config_contract.py`
  - `make dashboard-check`
  - `make agent-lint CHANGED_FILES="src/semantic-router/pkg/extproc/request_context.go src/semantic-router/pkg/extproc/req_filter_classification.go src/semantic-router/pkg/classification/classifier_projections.go src/semantic-router/pkg/classification/classifier_signal_context.go src/semantic-router/pkg/classification/classifier_signal_eval.go src/semantic-router/pkg/classification/classifier_signal_groups.go src/semantic-router/pkg/classification/classifier_signal_subset.go src/semantic-router/pkg/decision/engine.go src/semantic-router/pkg/config/meta_routing_config.go src/semantic-router/pkg/config/validator_meta.go src/semantic-router/pkg/dsl/ast.go src/semantic-router/pkg/dsl/compiler.go src/semantic-router/pkg/dsl/parser.go src/semantic-router/pkg/dsl/routing_contract.go src/vllm-sr/cli/models.py src/vllm-sr/cli/validator.py src/semantic-router/pkg/extproc/meta_routing_runtime.go src/semantic-router/pkg/extproc/meta_routing_trace.go src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_helpers.go src/semantic-router/pkg/extproc/meta_routing_feedback_store.go src/semantic-router/pkg/extproc/meta_routing_types.go tools/agent/scripts/meta_routing_feedback_features.py tools/agent/scripts/meta_routing_feedback_report.py docs/agent/adr/adr-0003-meta-routing-refinement-boundary.md docs/agent/plans/pl-0015-meta-routing-refinement-workstream.md docs/agent/tech-debt/td-037-meta-routing-learned-policy-provider-follow-on.md config/README.md website/docs/installation/configuration.md website/docs/tutorials/projection/overview.md"`
  - `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/request_context.go src/semantic-router/pkg/extproc/req_filter_classification.go src/semantic-router/pkg/classification/classifier_projections.go src/semantic-router/pkg/classification/classifier_signal_context.go src/semantic-router/pkg/classification/classifier_signal_eval.go src/semantic-router/pkg/classification/classifier_signal_groups.go src/semantic-router/pkg/classification/classifier_signal_subset.go src/semantic-router/pkg/decision/engine.go src/semantic-router/pkg/config/meta_routing_config.go src/semantic-router/pkg/config/validator_meta.go src/semantic-router/pkg/dsl/ast.go src/semantic-router/pkg/dsl/compiler.go src/semantic-router/pkg/dsl/parser.go src/semantic-router/pkg/dsl/routing_contract.go src/vllm-sr/cli/models.py src/vllm-sr/cli/validator.py src/semantic-router/pkg/extproc/meta_routing_runtime.go src/semantic-router/pkg/extproc/meta_routing_trace.go src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_helpers.go src/semantic-router/pkg/extproc/meta_routing_feedback_store.go src/semantic-router/pkg/extproc/meta_routing_types.go tools/agent/scripts/meta_routing_feedback_features.py tools/agent/scripts/meta_routing_feedback_report.py docs/agent/adr/adr-0003-meta-routing-refinement-boundary.md docs/agent/plans/pl-0015-meta-routing-refinement-workstream.md docs/agent/tech-debt/td-037-meta-routing-learned-policy-provider-follow-on.md config/README.md website/docs/installation/configuration.md website/docs/tutorials/projection/overview.md"`
  - `make agent-feature-gate ENV=cpu CHANGED_FILES="src/semantic-router/pkg/extproc/request_context.go src/semantic-router/pkg/extproc/req_filter_classification.go src/semantic-router/pkg/classification/classifier_projections.go src/semantic-router/pkg/classification/classifier_signal_context.go src/semantic-router/pkg/classification/classifier_signal_eval.go src/semantic-router/pkg/classification/classifier_signal_groups.go src/semantic-router/pkg/classification/classifier_signal_subset.go src/semantic-router/pkg/decision/engine.go src/semantic-router/pkg/config/meta_routing_config.go src/semantic-router/pkg/config/validator_meta.go src/semantic-router/pkg/dsl/ast.go src/semantic-router/pkg/dsl/compiler.go src/semantic-router/pkg/dsl/parser.go src/semantic-router/pkg/dsl/routing_contract.go src/vllm-sr/cli/models.py src/vllm-sr/cli/validator.py src/semantic-router/pkg/extproc/meta_routing_runtime.go src/semantic-router/pkg/extproc/meta_routing_trace.go src/semantic-router/pkg/extproc/meta_routing_feedback.go src/semantic-router/pkg/extproc/meta_routing_feedback_helpers.go src/semantic-router/pkg/extproc/meta_routing_feedback_store.go src/semantic-router/pkg/extproc/meta_routing_types.go tools/agent/scripts/meta_routing_feedback_features.py tools/agent/scripts/meta_routing_feedback_report.py docs/agent/adr/adr-0003-meta-routing-refinement-boundary.md docs/agent/plans/pl-0015-meta-routing-refinement-workstream.md docs/agent/tech-debt/td-037-meta-routing-learned-policy-provider-follow-on.md config/README.md website/docs/installation/configuration.md website/docs/tutorials/projection/overview.md"`
  - Completed on 2026-03-24: the end-to-end validation ladder passed, including dashboard checks, affected Go and Python tests, `make agent-lint`, `make agent-ci-gate`, and `make agent-feature-gate ENV=cpu` with local serve and smoke succeeding. The learned-policy follow-on later landed under `ADR 0005` and `pl-0017`, so this refinement workstream remains closed.

## Decision Log

- 2026-03-23: Meta-routing will be introduced as a request-phase orchestration seam, not as a new responsibility inside the decision engine or classifier hotspots.
- 2026-03-23: The public contract will stay intentionally narrow at `routing.meta` with only mode, pass budget, trigger policy, and allowed actions. Feedback sinks and runtime-internal knobs should not leak into the public schema.
- 2026-03-23: V1 refinement is selective subgraph recomputation. The repository will not introduce token-level regenerate loops or free-form agent control into the extproc path in this workstream.
- 2026-03-23: Pass-level tracing and durable `FeedbackRecord` output are part of the implementation boundary, not optional observability cleanup, because later calibration and learned policies depend on them.
- 2026-03-23: `observe` and `shadow` must share the same assessment and planning seams as `active`; they differ only in whether planned refinement is executed and whether the final route may change.
- 2026-03-23: Meta outputs remain orchestration-only in v1 to avoid circular rule graphs where decision rules directly consume refinement-state artifacts.
- 2026-03-23: Search-R2-inspired pass-quality semantics are part of the durable schema. `PassTrace`, `MetaAssessment`, and `FeedbackRecord` should not need another structural migration to support later calibration or learned-policy work.
- 2026-03-23: The v1 root-cause taxonomy is adapted to routing runtime rather than token search. `compression_loss_risk` is retained as an orchestration root cause even though refinement remains bounded to declared actions.
- 2026-03-23: The first runtime increment lands as observe-only assess-and-plan behavior on the existing base pass. The next increment should split the extproc seam into explicit collaborators and execute bounded refinement plans without changing the current lower-layer ownership.

## Follow-up Debt / ADR Links

- [ADR 0003: Introduce Meta-Routing as a Request-Phase Orchestration Seam](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [pl-0017-meta-routing-learned-policy-enablement.md](pl-0017-meta-routing-learned-policy-enablement.md)
