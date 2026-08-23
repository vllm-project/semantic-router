# Knowledge Base Signal

## Overview

`kb` binds routing signals to the output of a named knowledge base instance.
Define these bindings under `routing.signals.kb`.

Use it for an embedding-backed knowledge base that is loaded at Router startup
and reused across several routes.

## Key Advantages

- Reuses one exemplar set across several routes.
- Keeps labels, groups, and numeric metrics explicit instead of relying on magic runtime names.
- Supports both winner-style and threshold-style signal bindings.
- Lets projections consume continuous knowledge base metrics without turning signals into a scripting surface.

## What Problem Does It Solve?

Some routing policies depend on a curated exemplar set rather than a single
keyword or embedding candidate list. One knowledge base can classify requests
into privacy, safety, emotion, or preference labels, while the routing config
exposes only the labels or groups that decisions need.

`kb` keeps that split explicit:

- `global.model_catalog.kbs[]` owns the reusable knowledge base package
- `routing.signals.kb[]` binds specific labels or groups into named routing signals
- `routing.projections` can consume knowledge base metrics such as `best_score`, `best_matched_score`, or configured group margins

## When to Use

Use `kb` when:

- requests must be classified against a curated exemplar set
- one startup-loaded knowledge base result should feed several routes
- you want stable route-level groups without duplicating exemplars
- you need explicit bindings instead of implicit signal names

## Configuration

```yaml
global:
  model_catalog:
    kbs:
      - name: privacy_kb
        source:
          path: knowledge_bases/privacy/
          manifest: labels.json
        threshold: 0.55
        prototype_scoring:
          enabled: true
          cluster_similarity_threshold: 0.9
          max_prototypes: 8
          best_weight: 0.75
          top_m: 2
          margin_threshold: 0.05
        label_thresholds:
          prompt_injection: 0.7
        groups:
          privacy_policy: [proprietary_code, internal_document, pii]
          security_containment: [prompt_injection, credential_exfiltration]
          private: [proprietary_code, internal_document, pii, prompt_injection, credential_exfiltration]
          public: [generic_coding, general_knowledge]
        metrics:
          - name: private_vs_public
            type: group_margin
            positive_group: private
            negative_group: public

document:
  signals:
    kb:
      - name: privacy_policy
        kb: privacy_kb
        target:
          kind: group
          value: privacy_policy
        match: best
      - name: proprietary_code
        kb: privacy_kb
        target:
          kind: label
          value: proprietary_code
        match: threshold
```

Keep knowledge base names stable because `kb` signals bind to those names directly.

When `prototype_scoring` is enabled, the KB builds per-label prototype banks from the label exemplars. Runtime classification then scores labels from those label-owned prototypes instead of letting one raw exemplar dominate the whole label forever.

## Match Semantics

`routing.signals.kb[]` supports:

- `target.kind: label` or `group`
- `match: best` or `threshold`

Meaning:

- `label + best`: match only when the label is the knowledge base's best label
- `label + threshold`: match when the label score clears its effective threshold
- `group + best`: match only when the group is the knowledge base's best group
- `group + threshold`: match when any member label clears its threshold

## Projection Metrics

Knowledge base signals are boolean routing inputs. Numeric outputs stay in `routing.projections`.

For example:

```yaml
document:
  projections:
    scores:
      - name: privacy_bias
        method: weighted_sum
        inputs:
          - type: kb_metric
            kb: privacy_kb
            metric: private_vs_public
            value_source: score
            weight: 1.0
```

Named knowledge base metrics are declared under `global.model_catalog.kbs[].metrics[]`. Built-in metrics `best_score` and `best_matched_score` are always available.

## Dependencies and Limitations

- The knowledge-base package is loaded from
  `global.model_catalog.kbs[].source`. Keep its manifest and files versioned
  together.
- Request text is embedded through the shared semantic embedding runtime. A
  remote embedding provider therefore receives the text.
- Labels, groups, thresholds, and embedding model form one calibrated unit;
  re-evaluate them together when any part changes.
- See the knowledge-base signal example
  [`config/fragments/signal/kb/privacy.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/kb/privacy.yaml)
  and the complete KB declaration in
  [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml).
