# Embedding Signal

## Overview

`embedding` matches requests by semantic similarity to representative examples. It maps to `config/signal/embedding/` and is declared under `routing.signals.embeddings`.

This family is learned: it depends on the semantic embedding assets in `global.model_catalog.embeddings`.

## Key Advantages

- Handles paraphrases better than plain keyword rules.
- Lets teams tune routing with example phrases instead of retraining a classifier.
- Works well for support intents, product flows, and semantic FAQ routing.
- Provides a smooth step up from purely lexical signals.

## What Problem Does It Solve?

Keyword routing misses semantically similar prompts that use different wording. Full domain classification can also be too coarse when the route depends on a narrow intent.

`embedding` solves that by matching new prompts against example candidates in embedding space.

## When to Use

Use `embedding` when:

- phrasing varies but intent stays stable
- you want semantic routing without introducing a full custom classifier
- examples are easier to maintain than domain labels
- support or workflow intents need better recall than keywords can provide

## Configuration

Source fragment family: `config/signal/embedding/`

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        aggregation_method: max
        candidates:
          - how to configure the system
          - installation guide
          - troubleshooting steps
          - error message explanation
          - setup instructions
      - name: account_management
        threshold: 0.72
        aggregation_method: max
        candidates:
          - password reset
          - account settings
          - profile update
          - subscription management
          - billing information
```

Tune the threshold and candidate list together; that matters more than adding many low-quality examples.

Ranked fallback behavior is tuned separately under the router-owned embedding catalog:

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          enable_soft_matching: true
          top_k: 1
          min_score_threshold: 0.5
          prototype_scoring:
            enabled: true
            cluster_similarity_threshold: 0.9
            max_prototypes: 8
            best_weight: 0.75
            top_m: 2
            margin_threshold: 0.05
```

`prototype_scoring` compresses each embedding rule's candidate bank into a smaller set of representative prototypes, then scores the rule from those prototypes instead of relying on one flat candidate list forever.

The router now scores every embedding rule first and only then applies `top_k` as an emission limit. The default is `1`, so only the strongest embedding signal is returned unless you explicitly raise the limit. Set `top_k: 0` if you need the legacy "return every matched embedding rule" behavior.
