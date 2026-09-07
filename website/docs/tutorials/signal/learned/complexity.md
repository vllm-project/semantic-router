# Complexity Signal

## Overview

`complexity` estimates whether a request is `easy`, `medium`, or `hard` by
comparing it with configured example sets. It is independent of topic: two
requests in the same domain can still need different model tiers.

## Key Advantages

- Separates estimated difficulty from topic classification.
- Reuses one easy/medium/hard policy across multiple decisions.
- Tunes routing with examples instead of a custom classifier schema.

## What Problem Does It Solve?

Domain routing alone cannot distinguish a short factual request from a
multi-step analysis request. Complexity supplies a reusable difficulty signal
so decisions can escalate only the traffic that needs it.

## When to Use

Use complexity when model cost or reasoning mode should vary with estimated
task difficulty. Do not treat it as a correctness or safety guarantee; use
domain-specific evaluation and safety signals for those concerns.

## Configuration

```yaml
routing:
  signals:
    complexity:
      - name: needs_reasoning
        threshold: 0.10
        description: Escalate multi-step reasoning or synthesis-heavy prompts.
        hard:
          candidates:
            - solve this step by step
            - compare multiple tradeoffs
            - analyze the root cause
        easy:
          candidates:
            - answer briefly
            - quick summary
            - simple rewrite
```

`threshold` is a margin, not a similarity cutoff. The router scores the request
against the `hard` and the `easy` candidate banks, then compares the two:

```text
signal = hard_bank_score - easy_bank_score

signal >  threshold  -> hard
signal < -threshold  -> easy
otherwise            -> medium
```

Because the two banks can assign similar baseline scores, subtracting their
scores may produce a margin much smaller than either individual score. In one
calibration run using the candidate banks above, the largest observed absolute
margin across eight prompts was `0.197`; none reached the `hard` or `easy` band
with a threshold of `0.75`.

Treat `0.10` as a starting point rather than a universal default. Measure the
margin on representative traffic and tune the threshold for the configured
candidate banks and embedding model.

A rule emits a suffixed name. Decisions must reference
`<rule>:easy`, `<rule>:medium`, or `<rule>:hard`:

```yaml
routing:
  decisions:
    - name: escalate-hard-prompts
      description: Route hard prompts to the reasoning model.
      priority: 150
      rules:
        operator: AND
        conditions:
          - type: complexity
            name: needs_reasoning:hard
      modelRefs:
        - model: reasoning-model
          use_reasoning: true
```

For optional prototype-bank tuning, configure the family-level module once:

```yaml
global:
  model_catalog:
    modules:
      complexity:
        prototype_scoring:
          enabled: true
          max_prototypes: 8
          top_m: 2
```

## Dependencies and Limitations

- Complexity uses the configured semantic embedding runtime. A remote embedding
  provider receives the request text used for classification.
- Candidate phrases and thresholds must be calibrated together against labeled
  traffic. Re-evaluate them whenever the embedding model changes.
- Ambiguous prompts can land in the `medium` band; always define a route or
  fallback for every band you rely on.
- See a complete example:
  [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml).
