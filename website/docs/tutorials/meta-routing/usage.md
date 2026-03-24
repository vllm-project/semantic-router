---
sidebar_position: 4
---

# Using Meta Routing

## Overview

The normal operator flow is:

1. author `routing.meta`
2. run in `observe`
3. inspect traces and feedback
4. move to `shadow`
5. move to `active` only after the bounded refinement behavior is understood

This page focuses on the practical rollout path.

## Key Advantages

- Gives operators a staged rollout path instead of forcing immediate decision-changing behavior.
- Reuses the same config surface across YAML, DSL, and dashboard config editing.
- Produces one request record that can be inspected in the dashboard or replay tooling.
- Keeps learned-policy experiments behind the same runtime seam.

## What Problem Does It Solve?

Without an explicit usage pattern, a feature like meta routing often becomes “internal only”: hard to configure, hard to observe, and hard to promote safely.

This workflow makes it concrete:

- `observe` proves whether the triggers are useful
- `shadow` proves whether the planned refinement actions are sensible
- `active` proves whether bounded refinement improves live routing outcomes enough to justify the latency cost

## When to Use

Use this rollout sequence when:

- you are enabling meta routing for the first time
- you are tightening trigger thresholds
- you are changing allowed refinement actions
- you are introducing an internal calibrated or learned policy artifact

## Configuration

Start with a small config and a narrow action budget:

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
    allowed_actions:
      - type: disable_compression
      - type: rerun_signal_families
        signal_families: [embedding, fact_check]
```

Then promote deliberately:

1. `observe`
   - confirm that triggers and root causes look believable
   - confirm that latency overhead is just trace collection
2. `shadow`
   - confirm that refinement plans and second-pass traces make sense
   - compare base versus refined results without changing production routing
3. `active`
   - allow the refined pass to win only after the earlier phases are stable

## Dashboard Workflow

The dashboard surfaces the same feature directly:

- `Config -> Meta Routing` for authoring `routing.meta`
- `/meta-routing` for request lists, aggregates, and one-record inspection

The detail inspector is the fastest way to verify:

- base pass versus refined pass
- triggers and root causes
- planned and executed actions
- route overturns
- latency deltas

## API and Feedback

Router-owned feedback records are available through the meta-routing feedback APIs and power the dashboard view.

Those records join:

- pass traces
- final decision and model
- executed refinement actions
- weak outcome labels such as fallback, block, or replay metadata

That is the main evidence source for deciding whether a rollout should move forward.

## Advanced Provider Experiments

The public YAML contract does not change when internal calibrated or learned policy artifacts are used.

Advanced internal experiments keep the same `routing.meta` config and load a policy artifact behind the seam, so operators can compare deterministic versus artifact-backed behavior without inventing a second public config story.
