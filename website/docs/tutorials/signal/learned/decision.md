# Decision Signal

## Overview

`decision` asks a decision model a typed question about the request and turns
the answer into a routing fact. The question is a System One question: a
Choice among named options, a Noul (a yes/no probability) or a Score on an
ordered scale. The model runs in the built-in model runtime through a
`model_runtime` deployment, which the Router starts and supervises, or
attaches to.

All decision questions of one request that target the same deployment travel
in one call, so the runtime answers them together.

## Key Advantages

- asks open, per-route questions in plain language, without training a classifier per label set
- returns calibrated probabilities and expected levels that decisions can threshold
- fails open: a late or failed answer leaves the signal unknown, never the request

## What Problem Does It Solve?

Fixed-label classifiers answer only the questions they were trained for. A
decision model answers a new question as soon as it is written in the
configuration, and the Router can route on the answer the same way it routes
on any other signal.

## When to Use

Use a decision signal for routing questions that need judgment about the whole
request ("does this need multi-step reasoning?", "which kind of work is
this?", "how difficult is it?"). Prefer heuristic signals for structural facts
(token counts, keywords, modalities) and specialized learned signals (PII,
jailbreak, domain) where they exist.

## Configuration

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        device: auto          # auto, cpu, cuda:N or rocm:N
        profile: exact        # exact (default) or an opt-in faster profile

routing:
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7            # on P(true); 0.5 when omitted
        timeout_ms: 1000
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: math
              description: Mathematics or quantitative reasoning
            - key: chat
              description: Anything else
      - name: difficulty
        deployment: decision-kai
        question:
          type: score
          instructions: How difficult is this request?
          levels: [Trivial, Moderate, Hard]
        predicate:
          gte: 1.5            # on the expected level (0..2 here); required for score

  decisions:
    - name: hard-code
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: decision
            name: request_kind
            label: code
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: large-coder
```

| Question type | Matches when | Published values |
| --- | --- | --- |
| `noul` | P(true) satisfies `predicate` (default `gte: 0.5`) | `decision:<name>` = P(true) |
| `score` | the expected level satisfies `predicate` | `decision:<name>` = expected level |
| `choice` | a condition's `label` is the arg-max option, and its probability satisfies `predicate` when one is set | `decision:<name>:<key>` = P(key), `decision:<name>` = P(chosen) |

A condition may add its own `predicate`, which reads the published value; for
a Choice condition with a `label`, it reads that option's probability.

When the runtime is starting, overloaded or too slow, the signal is unknown.
Set `rules.on_unknown` on the decision, or `on_error: match | no_match` on the
condition, to choose what an unknown answer means. Matched decision signals
are reported in the `x-vsr-matched-decision-model` response header.
