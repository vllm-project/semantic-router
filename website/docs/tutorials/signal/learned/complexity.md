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

The two rules above are not alternatives: the scorer is called once, and each
rule reads the same score through its own boundaries, so one request yields
one verdict per rule.

| Score | `needs_reasoning` (0.85 / 0.60) | `extreme` (0.95 / 0.30) |
| ----- | ------------------------------- | ----------------------- |
| 0.20  | easy                            | easy                    |
| 0.50  | easy                            | medium                  |
| 0.70  | medium                          | medium                  |
| 0.90  | **hard**                        | medium                  |
| 0.99  | hard                            | **hard**                |

That gives decisions a ladder to match on - `extreme:hard` for the very top,
`needs_reasoning:hard` for anything past 0.85, `needs_reasoning:easy` for the
bottom. A score of 0.99 satisfies both `hard` conditions at once, so the
decision for `extreme:hard` needs the higher `priority`, or the looser rule
takes the request.

Give every rung an explicit `priority`, and give the stricter rung the higher
number. `priority` may be omitted, and two decisions that both match at the
same priority are separated by confidence - which `score.v1` never reports, so
the comparison falls through to the decisions' names in alphabetical order.
Renaming a decision would then change which model a request reaches, and
nothing reports that it happened - #3658 tracks making the comparison that
settled a request observable.

`score.v1` reports no confidence. A score just short of `hard_above` is the
least certain position rather than a strong one, so no confidence is derived
from it, and any decision gated on such a rule ranks on the engine's structural
default instead of a reported score. The router warns at startup when this
applies.

The endpoint may answer in either shape: the HuggingFace text-classification
array with exactly one entry, `[{"label": "difficulty", "score": 0.73}]`, or a
bare object, `{"score": 0.73}`. A missing or null score, more than one entry,
or a non-finite value is an error, never a zero - zero is a real score at the
easy end of a `[0,1]` range, and a fault must not route as one.

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

#### Watching a remote scorer

A network dependency fails in ways a local model does not, so the remote paths
report what they do:

- `llm_remote_connector_requests_total{operation, outcome}` and
  `llm_remote_connector_request_duration_seconds{operation}` count and time
  every call through the shared connector. `outcome` is `success` or the error
  kind (`transport`, `status`, `authorization`, `request`, `response`), and
  `llm_remote_connector_retries_total` counts retried attempts.
- `llm_complexity_verdict_total{rule, verdict, source}` counts verdicts, where
  `source` is `local`, `remote_score` or `remote_labels` - the same label the
  failure counter carries, so the two can be read together. The shape of this
  distribution is the check for a mismatched scale: a `[1,10]` scorer behind
  `hard_above: 0.85` shows up as `hard` taking every request for that rule,
  which no single log line would reveal.
- `llm_complexity_evaluation_failures_total{source}` counts evaluations that
  produced no verdict.

When the scorer cannot be reached, the request still routes. The complexity
signal is absent, every complexity rule is marked
`complexity_evaluation_failed` in the request's signal errors, and
`llm_complexity_evaluation_failures_total` counts it.

By default a decision whose condition reads a failed complexity rule treats it
as not matched, so the request falls through to whatever matches next. To
decide deliberately instead, set `on_unknown` on that decision's root `rules`
node:

```yaml
decisions:
  - name: deep-reasoning
    priority: 100
    rules:
      on_unknown: match       # no_match (default) | match | fail_request
      operator: AND
      conditions:
        - type: complexity
          name: needs_reasoning:hard
```

`match` sends a request you could not grade to the stronger model, which is
usually the safer default; `fail_request` refuses it outright. `on_unknown`
belongs on the root `rules` node and applies to the whole decision - the
per-condition `on_error` field is accepted only on `classifier` conditions,
so putting it on a `complexity` condition is rejected at config load.

The scorer's number is published as `complexity:<rule>:score`, in the model's
own units, where local scoring publishes `complexity:<rule>:margin` and its
text/image components. These reach replay records and observability; a
decision condition matches on the verdict (`<rule>:<verdict>`), not on the
number, so a numeric predicate over `:score` is not a routing mechanism.

## Dependencies and Limitations

- Complexity uses the configured semantic embedding runtime. A remote embedding
  provider receives the request text used for classification.
- Candidate phrases and thresholds must be calibrated together against labeled
  traffic. Re-evaluate them whenever the embedding model changes.
- Ambiguous prompts can land in the `medium` band; always define a route or
  fallback for every band you rely on.
- See a complete example:
  [`config/fragments/signal/complexity/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/complexity/escalation.yaml).
