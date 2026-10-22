# GMT Router

## Overview

`gmtrouter` is an experimental personalized selector. It uses versioned model
intelligence for cold start, then blends that base score with preferences
learned from a user's prior model feedback.

## What Problem Does It Solve?

Global quality rankings cannot reflect that different users prefer different
models. GMT Router starts safely from static evidence, then learns a bounded
per-user preference from feedback.

## When to Use

Use GMT Router when requests carry a stable user identity, feedback is
available, and personalization is more useful than a single global ranking.
Prefer a stateless selector when those inputs are absent.

## Selection behavior

Before a user reaches `min_interactions_for_personalization`, candidates are
ranked by their available intelligence score. Afterward, the current selector
uses:

```text
score = 0.3 * intelligence + 0.7 * user preference
```

A candidate with no learned preference receives its intelligence score with a
small penalty. Missing intelligence uses the selector's neutral cold-start
fallback and remains distinct from a measured score of zero.

When a candidate declares `reasoning_effort`, GMT Router reads only that exact
evidence bucket. It does not borrow measurements from another effort. Coverage
is never multiplied into intelligence: higher coverage breaks a tie only when
the final GMT scores and intelligence scores are equal. Selection diagnostics
report the chosen score and coverage, or mark intelligence unavailable.

## Configuration

```yaml
algorithm:
  type: gmtrouter
  gmtrouter:
    enable_personalization: true
    min_interactions_for_personalization: 3
    max_interactions_per_user: 100
    history_sample_size: 5
    embedding_dimension: 768
    num_gnn_layers: 2
    attention_heads: 8
    storage_path: state/gmtrouter.json
```

Feedback updates bounded per-user interaction history and model preferences.
When `storage_path` is configured, that state is persisted. Query, response,
and model-description embeddings contribute only when an embedding function is
available.

## Limitations

- GMT Router is experimental; validate personalization on representative traffic.
- Requests without a user identity share the `anonymous` preference state.
- Learned preferences are model-level, not reasoning-effort-specific.
- Static coverage does not describe or weight learned feedback.
