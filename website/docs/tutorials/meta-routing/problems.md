---
sidebar_position: 3
---

# Meta Routing Problems

## Overview

Meta routing exists because one-pass route selection can be correct, but still fragile.

This page focuses on the concrete failure modes that `routing.meta` is designed to surface and, when configured, mitigate.

## Key Advantages

- Turns routing brittleness into named triggers and root causes instead of vague “bad route” anecdotes.
- Helps operators separate true route ambiguity from detector absence, low confidence, or compression loss.
- Allows targeted remediation instead of rerunning every expensive signal family for every request.
- Produces comparable traces for base versus refined passes.

## What Problem Does It Solve?

The current v1 trigger set is built around a few recurring failure modes:

### Low decision margin

Two routes may be close enough that the winner is not trustworthy. This usually appears when the router has multiple plausible candidates but very little evidence separating them.

### Projection boundary pressure

A weighted score may fall close to a threshold band such as `balance_medium` versus `balance_reasoning`. That means a tiny change in evidence could flip the decision-visible output.

### Partition conflict

Competing domain or embedding members may both look plausible before one partition winner is selected. The route might still resolve, but the pass is less stable than a clean winner.

### Missing or low-confidence required families

Some traffic depends heavily on signal families such as embeddings, fact-check, or preference detectors. If those families are absent or weak, the route may still match, but it is doing so on thinner evidence than intended.

### Family disagreement

Cheap heuristics and more expensive semantic families may disagree. That does not always mean the base route is wrong, but it is a strong signal that one more bounded pass may be useful.

### Compression loss risk

If the first pass used compressed request text and already looked fragile, the missing detail may be the reason.

## When to Use

Use these triggers when you see routing patterns like:

- route overturns cluster near the same projection thresholds
- one family is often missing on the traffic that later falls back or gets corrected
- operators want to know whether rerunning embeddings or disabling compression actually helps
- you need a replayable explanation for why a request was refined

## Configuration

```yaml
routing:
  meta:
    mode: observe
    max_passes: 2
    trigger_policy:
      decision_margin_below: 0.15
      projection_boundary_within: 0.05
      partition_conflict: true
      required_families:
        - type: embedding
          min_matches: 1
          min_confidence: 0.72
        - type: fact_check
          min_confidence: 0.68
      family_disagreements:
        - cheap: keyword
          expensive: embedding
        - cheap: structure
          expensive: preference
```

This example does not force any refinement by itself. It only defines which brittle-pass signals the router should watch for.

## Root-Cause Vocabulary

The current runtime records root causes such as:

- `decision_overlap`
- `projection_boundary_pressure`
- `partition_conflict`
- `missing_required_family`
- `low_confidence_family`
- `family_disagreement`
- `compression_loss_risk`

That vocabulary matters because dashboard summaries, replay jobs, and future learned-policy promotion all reuse it.
