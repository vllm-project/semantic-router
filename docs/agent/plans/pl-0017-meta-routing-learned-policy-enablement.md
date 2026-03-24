# Meta-Routing Learned Policy Enablement Workstream

## Goal

- Close the learned-policy follow-on gap left after the deterministic meta-routing v1 rollout.
- Add one durable learned-policy contract spanning governance, runtime loading, artifact validation, and offline script consumption without widening the public `routing.meta` schema.
- Keep learned-policy behavior behind the existing extproc meta-routing seam so signal, projection, decision, and model-selection ownership stays intact.

## Scope

- `docs/agent/adr/**`
- `docs/agent/plans/**`
- `docs/agent/tech-debt/**`
- `src/semantic-router/pkg/extproc/**`
- `tools/agent/scripts/**`
- targeted runtime and script tests for artifact loading, policy gating, and provider attribution
- nearest local rules for `src/semantic-router/pkg/extproc/`

## Exit Criteria

- One indexed ADR defines the learned meta-routing policy artifact contract, rollout semantics, and runtime loading boundary.
- Runtime owns a `PolicyProvider` seam with a deterministic default and an artifact-backed provider behind the existing meta-routing orchestrator.
- The artifact contract is versioned, includes evaluation thresholds, and blocks runtime enablement when acceptance criteria are not met.
- `RoutingTrace` and `FeedbackRecord` retain provider identity so replay and calibration outputs can be attributed to the exact provider lineage.
- Offline scripts validate or summarize the same artifact envelope rather than inventing a second learned-policy format.
- Focused runtime and script validation pass, and the prior TD037 debt entry can be retired.

## Task List

- [x] `MLP001` Add a learned-policy ADR and execution plan for the post-v1 meta-routing contract.
- [x] `MLP002` Introduce a runtime `PolicyProvider` seam with deterministic-default behavior preserved.
- [x] `MLP003` Define and validate a versioned learned-policy artifact envelope plus runtime startup gate.
- [x] `MLP004` Extend traces and offline scripts to consume provider identity and the shared artifact contract.
- [x] `MLP005` Run focused validation, retire TD037 if the contract gap is closed, and update governance inventories.

## Current Loop

- Date: 2026-03-24
- Current task: `MLP005` completed
- Branch: `main`
- Planned loop order:
  - `L1` lock the learned-policy contract in ADR/plan form
  - `L2` land the runtime `PolicyProvider` seam and artifact loader
  - `L3` land script-side artifact validation/summarization
  - `L4` run focused validation and retire TD037 if the contract gap is materially closed
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/README.md`, `docs/agent/governance.md`, `docs/agent/plans/README.md`, `docs/agent/tech-debt/README.md`, and `src/semantic-router/pkg/extproc/AGENTS.md`
  - `make agent-report ENV=cpu CHANGED_FILES="src/semantic-router/pkg/extproc/meta_routing_policy_provider.go src/semantic-router/pkg/extproc/meta_routing_policy_loader.go src/semantic-router/pkg/extproc/meta_routing_runtime.go src/semantic-router/pkg/extproc/meta_routing_trace.go tools/agent/scripts/meta_routing_policy_validate.py docs/agent/adr/adr-0005-meta-routing-learned-policy-contract.md docs/agent/plans/pl-0017-meta-routing-learned-policy-enablement.md docs/agent/tech-debt/td-037-meta-routing-learned-policy-provider-follow-on.md"`
  - shell-based source reads for the existing meta-routing runtime seam, trace helpers, config contract, feedback scripts, debt entry, and governance indexes after codebase retrieval failed twice
  - `python3 -m compileall tools/agent/scripts/meta_routing_policy_support.py tools/agent/scripts/meta_routing_policy_validate.py tools/agent/scripts/meta_routing_feedback_support.py tools/agent/scripts/meta_routing_feedback_report.py tools/agent/scripts/test_meta_routing_policy_support.py`
  - `python3 -m pytest tools/agent/scripts/test_meta_routing_policy_support.py`
- `go test ./pkg/extproc -run 'TestRecordMetaRoutingBasePassCapturesAssessmentAndPlan|TestCreateMetaRoutingPolicyProviderLoadsArtifact|TestValidateMetaRoutingPolicyArtifactRejectsUnacceptedArtifact|TestArtifactMetaRoutingPolicyProviderOverridesTriggerPolicy'`
  - `go test ./pkg/extproc -run 'TestRecordMetaRoutingBasePassCapturesAssessmentAndPlan|TestCreateMetaRoutingPolicyProviderLoadsArtifact|TestValidateMetaRoutingPolicyArtifactRejectsUnacceptedArtifact|TestArtifactMetaRoutingPolicyProviderOverridesTriggerPolicy|TestEmitMetaRoutingFeedbackPersistsOnce|TestCollectMetaRoutingFeedbackRecordsReturnsParsedPayloads'`
  - `make agent-validate`
  - `make agent-lint CHANGED_FILES="src/semantic-router/pkg/extproc/meta_routing_policy_artifact.go,src/semantic-router/pkg/extproc/meta_routing_policy_provider.go,src/semantic-router/pkg/extproc/meta_routing_policy_loader.go,src/semantic-router/pkg/extproc/meta_routing_policy_loader_test.go,src/semantic-router/pkg/extproc/meta_routing_runtime.go,src/semantic-router/pkg/extproc/meta_routing_trace.go,src/semantic-router/pkg/extproc/meta_routing_trace_test.go,src/semantic-router/pkg/extproc/meta_routing_types.go,src/semantic-router/pkg/extproc/meta_routing_feedback.go,src/semantic-router/pkg/extproc/router.go,src/semantic-router/pkg/extproc/router_build.go,tools/agent/scripts/meta_routing_policy_support.py,tools/agent/scripts/meta_routing_policy_validate.py,tools/agent/scripts/meta_routing_feedback_support.py,tools/agent/scripts/meta_routing_feedback_report.py,docs/agent/adr/README.md,docs/agent/adr/adr-0005-meta-routing-learned-policy-contract.md,docs/agent/plans/README.md,docs/agent/plans/pl-0017-meta-routing-learned-policy-enablement.md,docs/agent/tech-debt/README.md,docs/agent/tech-debt/td-037-meta-routing-learned-policy-provider-follow-on.md,tools/agent/repo-manifest.yaml"`
  - `make agent-ci-gate CHANGED_FILES="src/semantic-router/pkg/extproc/meta_routing_policy_artifact.go,src/semantic-router/pkg/extproc/meta_routing_policy_provider.go,src/semantic-router/pkg/extproc/meta_routing_policy_loader.go,src/semantic-router/pkg/extproc/meta_routing_policy_loader_test.go,src/semantic-router/pkg/extproc/meta_routing_runtime.go,src/semantic-router/pkg/extproc/meta_routing_trace.go,src/semantic-router/pkg/extproc/meta_routing_trace_test.go,src/semantic-router/pkg/extproc/meta_routing_types.go,src/semantic-router/pkg/extproc/meta_routing_feedback.go,src/semantic-router/pkg/extproc/router.go,src/semantic-router/pkg/extproc/router_build.go,tools/agent/scripts/meta_routing_policy_support.py,tools/agent/scripts/meta_routing_policy_validate.py,tools/agent/scripts/meta_routing_feedback_support.py,tools/agent/scripts/meta_routing_feedback_report.py,docs/agent/adr/README.md,docs/agent/adr/adr-0005-meta-routing-learned-policy-contract.md,docs/agent/plans/README.md,docs/agent/plans/pl-0017-meta-routing-learned-policy-enablement.md,tools/agent/repo-manifest.yaml"` (blocked by unrelated `TestMaintainedBalanceWarningBudgetStaysBelowCeiling` in `src/semantic-router/pkg/dsl/maintained_asset_roundtrip_test.go`)
  - Completed on 2026-03-24: focused runtime and script validation passed, TD037 was retired, and governance inventories were updated. A broader `make agent-ci-gate` rerun still exposed the unrelated maintained-balance DSL warning-budget failure tracked under the separate `pl-0012` workstream, so that blocker does not keep this learned-policy contract plan open.

## Decision Log

- 2026-03-24: Learned-policy enablement stays behind the existing meta-routing seam and does not widen the public `routing.meta` schema.
- 2026-03-24: The first learned-policy artifact is an overlay contract for trigger and action policy, not an unconstrained inference-time planner.
- 2026-03-24: Runtime enablement must fail closed when artifact acceptance thresholds are not met; a silent fallback after loading an invalid artifact would weaken rollout safety.
- 2026-03-24: Provider identity belongs in `RoutingTrace`/`FeedbackRecord` so replay, calibration, and future dashboard surfaces can attribute behavior to exact policy lineage.

## Follow-up Debt / ADR Links

- [ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [pl-0015-meta-routing-refinement-workstream.md](pl-0015-meta-routing-refinement-workstream.md)
