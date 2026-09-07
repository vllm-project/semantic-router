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

### Local and remote scoring

With no `backend`, complexity keeps its existing local behaviour: the `hard`
and `easy` candidate lists are embedded once at startup, and each request is
scored by how much more it resembles the hard examples than the easy ones. That
difference is a signed margin centred on zero, which is why `threshold` is
symmetric - above `+threshold` is hard, below `-threshold` is easy, and the band
between them is medium.

A remote scorer uses the shared backend block, declared beside
`prototype_scoring` rather than on a rule. `routing.signals` is replaced
wholesale by each routing recipe, so a backend declared on a rule would
disappear under any recipe that did not repeat it, leaving a signal that still
ran but had quietly reverted to local. Its `model` is an explicit name from
`global.model_catalog.external[]`, and that entry needs
`model_role: classification`.

Complexity reads two contracts, and the one you choose depends on what your
model returns. Because there are two, `contract` cannot be defaulted and must
be stated.

#### A model that returns a score: `score.v1`

A regression model - a query-difficulty scorer, say - returns one number, and
the router turns it into a verdict using each rule's boundaries. One request
means one call no matter how many rules read it.

The score arrives in the model's own units, not as a margin, so `threshold`
does not apply. State the two boundaries instead, and the pair you use says
which way difficulty runs:

```yaml
global:
  model_catalog:
    external:
      - name: difficulty-scorer
        model_role: classification
        llm_endpoint:
          address: difficulty-scorer.default.svc
          port: 8080
        llm_model_name: query-difficulty-v1
    modules:
      complexity:
        backend:
          protocol: http_classify
          contract: score.v1
          model: difficulty-scorer
          deadline_ms: 5000

routing:
  signals:
    complexity:
      - name: needs_reasoning
        hard_above: 0.85          # a higher score is harder
        easy_below: 0.60
      - name: extreme
        hard_above: 0.95          # same score, stricter boundaries
        easy_below: 0.30
```

For a model whose score falls as difficulty rises - one predicting the chance
of a correct answer, for instance - use `hard_below` with `easy_above`
instead. Encoding the direction in the field names means there is no separate
direction setting to keep in sync, and an overlapping band cannot be written
by accident. Those two fields require a `score.v1` backend: the local margin is
hard-minus-easy, so a higher value is harder by construction, and to invert it
locally you swap the candidate lists.

Want finer grading than three verdicts? Write more rules over the same score,
each with its own boundaries. The verdict vocabulary stays `hard|easy|medium`
because decisions match `<rule>:<verdict>`, and rules remain distinguishable
through their own `composer` conditions.

`score.v1` reports no confidence. A score just short of `hard_above` is the
least certain position rather than a strong one, so no confidence is derived
from it, and any decision gated on such a rule ranks on the engine's structural
default instead of a reported score. The router warns at startup when this
applies.

#### A model that returns the verdict: `label_distribution.v1`

A three-class model returns `hard`, `easy` and `medium` directly. No boundaries
are consulted - the winning label *is* the verdict - and its probability is a
real confidence, so confidence-based ranking is unaffected. The labels are the
fixed verdict vocabulary and are not configured:

```yaml
    modules:
      complexity:
        backend:
          protocol: http_classify
          contract: label_distribution.v1
          model: difficulty-classifier
          deadline_ms: 5000
```

Either way, the `hard` and `easy` candidate lists are never read once a backend
supplies the score, and the router says so at startup rather than leaving you
to edit examples that have no effect.


## Dependencies and Limitations

- Complexity uses the configured semantic embedding runtime. A remote embedding
  provider receives the request text used for classification.
- Candidate phrases and thresholds must be calibrated together against labeled
  traffic. Re-evaluate them whenever the embedding model changes.
- Ambiguous prompts can land in the `medium` band; always define a route or
  fallback for every band you rely on.
- See a complete example:
  [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml).
