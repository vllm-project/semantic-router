# TD037: Meta-Routing Learned Policy Providers Still Need a Durable Enablement Contract

## Status

Closed

## Scope

`src/semantic-router/pkg/extproc/**`, `src/semantic-router/pkg/classification/**`, offline calibration scripts under `tools/agent/scripts/**`, and any future runtime or dashboard surface that enables non-deterministic meta-routing policies

## Summary

The repository now has the durable learned-policy enablement contract that was missing after the deterministic meta-routing v1 rollout. Runtime owns a first-class `PolicyProvider` seam behind the existing request-phase orchestrator, deterministic behavior remains the default, and artifact-backed providers load through one versioned internal contract with rollout acceptance gates. `RoutingTrace`, `FeedbackRecord`, and offline scripts now carry or consume provider identity plus the shared artifact envelope, so calibrated or learned trigger and action policy can be promoted without inventing a second contract surface. The original gap is retired.

## Evidence

- [docs/agent/adr/adr-0003-meta-routing-refinement-boundary.md](../adr/adr-0003-meta-routing-refinement-boundary.md)
- [docs/agent/adr/adr-0005-meta-routing-learned-policy-contract.md](../adr/adr-0005-meta-routing-learned-policy-contract.md)
- [docs/agent/plans/pl-0015-meta-routing-refinement-workstream.md](../plans/pl-0015-meta-routing-refinement-workstream.md)
- [docs/agent/plans/pl-0017-meta-routing-learned-policy-enablement.md](../plans/pl-0017-meta-routing-learned-policy-enablement.md)
- [src/semantic-router/pkg/extproc/meta_routing_runtime.go](../../../src/semantic-router/pkg/extproc/meta_routing_runtime.go)
- [src/semantic-router/pkg/extproc/meta_routing_policy_provider.go](../../../src/semantic-router/pkg/extproc/meta_routing_policy_provider.go)
- [src/semantic-router/pkg/extproc/meta_routing_policy_loader.go](../../../src/semantic-router/pkg/extproc/meta_routing_policy_loader.go)
- [src/semantic-router/pkg/extproc/meta_routing_policy_artifact.go](../../../src/semantic-router/pkg/extproc/meta_routing_policy_artifact.go)
- [src/semantic-router/pkg/extproc/meta_routing_trace.go](../../../src/semantic-router/pkg/extproc/meta_routing_trace.go)
- [src/semantic-router/pkg/extproc/meta_routing_feedback.go](../../../src/semantic-router/pkg/extproc/meta_routing_feedback.go)
- [tools/agent/scripts/meta_routing_policy_support.py](../../../tools/agent/scripts/meta_routing_policy_support.py)
- [tools/agent/scripts/meta_routing_policy_validate.py](../../../tools/agent/scripts/meta_routing_policy_validate.py)
- [tools/agent/scripts/meta_routing_feedback_features.py](../../../tools/agent/scripts/meta_routing_feedback_features.py)
- [tools/agent/scripts/meta_routing_feedback_report.py](../../../tools/agent/scripts/meta_routing_feedback_report.py)

## Why It Matters

- Learned meta-routing policies can change route selection behavior even when the public `routing.meta` config shape stays constant, so rollout safety still depends on one durable artifact and loading contract rather than ad hoc scripts.
- The deterministic seam was designed so later learned policies could plug in without reshaping runtime artifacts. Retiring this debt confirms that those policies now have an explicit, seam-respecting enablement path.

## Desired End State

- The repo has one explicit contract for learned meta-routing policies covering artifact format, versioning, acceptance criteria, rollout mode semantics, and runtime loading boundaries.
- Runtime policy loading stays behind the existing meta-routing seam and does not leak learned-policy behavior into signal, projection, or decision packages.
- Offline replay and calibration jobs can prove readiness for learned providers with durable thresholds before `active` mode uses non-deterministic policy outputs.

## Exit Criteria

- Satisfied on 2026-03-24: [ADR 0005](../adr/adr-0005-meta-routing-learned-policy-contract.md) defines the learned `PolicyProvider` contract, artifact envelope, and rollout semantics behind the existing seam.
- Satisfied on 2026-03-24: runtime and offline scripts consume the same learned-policy artifact contract instead of inventing parallel formats.
- Satisfied on 2026-03-24: runtime startup blocks learned-policy enablement when artifact acceptance thresholds are not met.

## Resolution

- `pkg/extproc` now owns a `PolicyProvider` seam with a deterministic default plus an artifact-backed provider that overlays `trigger_policy` and `allowed_actions` without widening public `routing.meta`.
- The shared artifact contract is versioned as `meta-routing-policy/v1alpha1`, validated against `feedback_record_flattened/v1`, and loaded through `VLLM_SR_META_ROUTING_POLICY_PATH` with fail-closed rollout checks.
- `RoutingTrace` and `FeedbackRecord` now retain provider identity so replay, calibration, and future UI surfaces can attribute outcomes to exact policy lineage.
- Offline tooling validates and summarizes the same artifact envelope through `meta_routing_policy_support.py` and `meta_routing_policy_validate.py`.
