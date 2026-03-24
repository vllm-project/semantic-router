---
sidebar_position: 1
---

# Meta Routing

## Overview

`routing.meta` adds a bounded request-phase assess-and-refine loop on top of the normal routing pipeline:

1. extract signals
2. compute projections
3. evaluate decisions
4. select a model
5. assess whether that pass looks brittle
6. optionally run one targeted refinement pass before finalizing the route

It does not replace signals, projections, or decisions. It wraps them.

The public contract stays intentionally small:

- `mode`
- `max_passes`
- `trigger_policy`
- `allowed_actions`

The three operating modes are:

- `observe`: assess and record, but never change the final route
- `shadow`: plan and execute a refinement pass, but still keep the base route
- `active`: allow the refined result to become final when the bounded plan produces a better outcome

Use this page as the entrypoint, then continue with [Design](./design), [Problems](./problems), and [Usage](./usage).

## Key Advantages

- Adds reliability control without pushing retry logic into signal, projection, or decision packages.
- Makes routing brittleness visible through pass traces and feedback records instead of hiding it behind one final route.
- Supports staged rollout from `observe` to `shadow` to `active`.
- Keeps refinement bounded to declared actions such as rerunning signal families or disabling lossy compression.

## What Problem Does It Solve?

One-pass routing is fast, but it can be brittle around ambiguous boundaries:

- a route wins by a very small decision margin
- a projection score lands near a mapping threshold
- a required signal family is missing or low-confidence
- competing signal families disagree
- compressed input drops detail that the router needed

Without a request-phase meta layer, the router can only expose the final matched route. It cannot cleanly answer:

- was the route confident or fragile?
- which trigger suggested that the pass was unreliable?
- what bounded refinement action would have helped?
- did a second pass change the decision or just confirm it?

`routing.meta` makes that reliability story explicit.

## When to Use

Use meta routing when:

- you already have a layered routing graph and want safer handling of ambiguous traffic
- expensive semantic families should only rerun when there is a clear trigger
- you need rollout evidence before allowing a refinement pass to alter production routing
- operators need request-level traces for route overturns, trigger patterns, and latency overhead

Skip it when:

- the routing graph is intentionally simple and one-pass decisions are already easy to reason about
- all traffic should always run the same expensive refinement path, regardless of confidence
- you do not need pass-level observability or bounded remediation

## Configuration

```yaml
routing:
  meta:
    mode: observe
    max_passes: 2
    trigger_policy:
      decision_margin_below: 0.18
      projection_boundary_within: 0.08
      partition_conflict: true
      required_families:
        - type: embedding
          min_confidence: 0.74
      family_disagreements:
        - cheap: keyword
          expensive: embedding
    allowed_actions:
      - type: disable_compression
      - type: rerun_signal_families
        signal_families: [embedding, fact_check, preference]
```

Read the field-level behavior in [Usage](./usage), and read the orchestration boundary in [Design](./design).
