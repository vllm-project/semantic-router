# Knowledge Base Signal

## Overview

`kb` binds routing signals to the output of a named knowledge base instance. It maps to `config/signal/kb/` and is declared under `routing.signals.kb`.

This signal family is for maintained embedding-backed knowledge bases that are loaded at router startup and then reused across several routes.

## Key Advantages

- Reuses one maintained exemplar package across several routes.
- Keeps labels, groups, and numeric metrics explicit instead of relying on magic runtime names.
- Supports both winner-style and threshold-style signal bindings.
- Lets projections consume continuous knowledge base metrics without turning signals into a scripting surface.

## What Problem Does It Solve?

Some routing policies depend on a curated exemplar set rather than a single keyword or embedding candidate list. You may want one startup-loaded knowledge base to classify requests into privacy, safety, emotion, or preference labels, then expose only the specific label or group bindings that the routing graph should see.

`kb` keeps that split explicit:

- `global.model_catalog.kbs[]` owns the reusable knowledge base package
- `routing.signals.kb[]` binds specific labels or groups into named routing signals
- `routing.projections` can consume knowledge base metrics such as `best_score`, `best_matched_score`, or configured group margins

## When to Use

Use `kb` when:

- requests must be classified against a maintained exemplar package
- one startup-loaded knowledge base result should feed several routes
- you want stable route-level groups without duplicating exemplars
- you need explicit bindings instead of implicit signal names

## Configuration

Source fragment family: `config/signal/kb/`

```yaml
global:
  model_catalog:
    kbs:
      - name: privacy_knowledge_base
        source:
          path: kb/privacy/
          manifest: labels.json
        threshold: 0.55
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

routing:
  signals:
    kb:
      - name: privacy_policy
        kb: privacy_knowledge_base
        target:
          kind: group
          value: privacy_policy
        match: best
      - name: proprietary_code
        kb: privacy_knowledge_base
        target:
          kind: label
          value: proprietary_code
        match: threshold
```

Keep knowledge base names stable because `kb` signals bind to those names directly.

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
routing:
  projections:
    scores:
      - name: privacy_bias
        method: weighted_sum
        inputs:
          - type: kb_metric
            kb: privacy_knowledge_base
            metric: private_vs_public
            value_source: score
            weight: 1.0
```

Named knowledge base metrics are declared under `global.model_catalog.kbs[].metrics[]`. Built-in metrics `best_score` and `best_matched_score` are always available.
