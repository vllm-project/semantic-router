# ADR 0005: Define a Learned Meta-Routing Policy Artifact Contract Behind the Existing Seam

## Status

Accepted

## Context

The repository now has a deterministic meta-routing seam with bounded refinement, pass-level traces, durable `FeedbackRecord` artifacts, offline feature extraction, and a dashboard/operator workflow for `observe`, `shadow`, and `active` modes. That closes the v1 request-phase runtime, but it still leaves one structural gap:

- the runtime has no first-class `PolicyProvider` seam
- deterministic trigger assessment and refinement planning are still hard-coded inside the extproc runtime
- offline calibration scripts and any future learned policy output have no shared artifact envelope
- rollout safety for a learned trigger/action policy would otherwise depend on ad hoc chat context, one-off scripts, or an undocumented file format

The public `routing.meta` contract must remain intentionally narrow. Learned-policy enablement cannot widen that user-facing YAML shape just to thread one artifact path or internal evaluation knob through the router. The follow-on contract therefore needs to live behind the existing meta-routing seam rather than becoming a second public config surface.

## Decision

The repository will introduce one internal learned-policy contract behind the existing meta-routing seam:

1. Runtime ownership
   - `pkg/extproc` owns a `PolicyProvider` seam for meta-routing assessment and planning.
   - The default provider remains deterministic and preserves current behavior when no artifact is supplied.
   - Signal, projection, decision, and model-selection packages remain unchanged owners; learned-policy logic must not bypass them.

2. Artifact contract
   - Learned or calibrated meta-routing policies are represented as one JSON artifact envelope with:
     - artifact version
     - provider identity and provider version
     - feature-schema identity
     - rollout acceptance thresholds
     - evaluation summary proving those thresholds were met
     - policy payload containing the trigger-policy and/or allowed-action overlay
   - The initial artifact version is `meta-routing-policy/v1alpha1`.
   - The initial feature schema reference is `feedback_record_flattened/v1`.

3. Runtime loading boundary
   - Runtime loading stays internal to the router and does not widen the public `routing.meta` schema.
   - The first loading seam is an internal environment-variable path, `VLLM_SR_META_ROUTING_POLICY_PATH`.
   - If the artifact is absent, the router uses the deterministic provider.
   - If the artifact is present but fails contract validation, router startup fails rather than silently changing routing behavior.

4. Rollout gating
   - A learned or calibrated artifact must carry explicit acceptance evidence.
   - Runtime refuses to load an artifact unless:
     - artifact version is supported
     - feature schema is supported
     - provider identity is present
     - evaluation is marked accepted
     - replay-count and quality thresholds in the rollout section are satisfied by the evaluation section
   - This gate is the first repo-native safety boundary for learned-policy enablement.

5. Scope of the first provider implementation
   - The first artifact-backed provider applies calibrated or learned overlays to:
     - `trigger_policy`
     - `allowed_actions`
   - It does not introduce arbitrary code execution, model inference inside the extproc request path, or any free-form planner.
   - This keeps learned-policy enablement compatible with the v1 guarantee that refinement remains selective subgraph recomputation.

6. Observability
   - `RoutingTrace` and `FeedbackRecord` include policy-provider identity so offline analysis can attribute outcomes to the exact provider and artifact lineage.
   - Offline scripts consume the same artifact envelope to validate or summarize artifacts instead of inventing a second policy format.

## Consequences

- The repository now has one durable contract for calibrated or learned meta-routing policies without widening the public YAML/DSL surface.
- Deterministic behavior remains the default, so the current runtime and dashboard user flow stay backward-compatible.
- Learned-policy rollout is safer because runtime enablement is blocked unless the artifact itself proves replay/calibration readiness.
- Future dashboard or CLI controls can point to the same artifact contract instead of defining a parallel schema.
- The first learned-policy implementation is intentionally conservative: it calibrates or replaces trigger/action policy through a bounded overlay, not an unconstrained inference-time planner.
