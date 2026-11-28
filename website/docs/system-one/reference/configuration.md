---
title: System One configuration reference
sidebar_label: Configuration
sidebar_position: 3
description: The configuration fields that name System One deployments, rules, the match block and algorithms, and the codes the configuration load reports.
---

# System One configuration reference

These are the fields that decide which deployment answers a question, how a
rule asks it, and how judges and algorithms use the answer. The guides explain
when to use them; this page lists them.

## Where the deployment comes from

| Field | Meaning |
| --- | --- |
| `global.model_catalog.system.decision_model` | `{deployment: <name>}`: answers every question nothing else binds. Default `{deployment: primary}` |
| `global.model_catalog.bindings.<type>` | The deployment of one application signal type's rules |
| `recipes[].model_bindings.<type>` | The same, for one recipe |
| `global.model_catalog.deployments.<name>` | A deployment: `provider: model_runtime` (local or attached) or `provider: systemone` (a remote API) |
| `global.model_catalog.deployments.<name>.fallbacks` | Ordered backup deployments for questions from `recipes[].signals.systemone` |
| `global.model_catalog.signal_timeout_ms` | The deadline of all of a request's signals |
| `global.model_catalog.on_signal_overload` | `{action, wait_ms, retry_after_seconds}`: with `action: retry`, calls without deployment fallbacks that an overloaded decision model refuses are asked again for up to `wait_ms`. A request that rules it still could not read would refuse then gets an error the client retries. See [when the decision model is overloaded](../deploy/performance.md#when-the-decision-model-is-overloaded) |

A `systemone` deployment takes `endpoint`, `model`, `auth` (`api_key_env` or
`api_key_file`, or `credential: delegated`; `header`, `prefix`), `headers`,
`timeout_ms`, `max_retries`, `tls` and `capabilities` (`question_types`,
`presets`, `modalities`).

### Back up a decision deployment {#decision-deployment-fallbacks}

For example, let a local CPU model answer custom System One signals if the
default remote decision deployment is unavailable:

```yaml
global:
  model_catalog:
    system:
      decision_model: {deployment: hosted}
    deployments:
      hosted:
        provider: systemone
        schema: vllm-sr/5.0
        endpoint: https://decisions.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        timeout_ms: 250
        fallbacks: [local-cpu]
      local-cpu:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        device: cpu
```

Failover runs after a transport failure, an attempt timeout, HTTP 429 or 5xx,
or an unavailable runtime. A rejection, a malformed response, or an answered
call (including a non-match) does not trigger failover. Backups are tried in list
order before any of their own backups, with no deployment visited twice.
Every target must support the rule's question and input; invalid references and
cycles prevent the configuration from loading.

The rule's `timeout_ms`, when set, caps the whole chain (the shortest when
rules share a call). The stage's deadline still applies. Each target has its
own `timeout_ms` (1000 ms when unset), bounded by the time left. An opted-in
signal tries the next target rather than waiting for capacity on a 429. If the
chain cannot answer, the rule's normal error policy still applies.

Each target receives the rule's input with that deployment's own schema,
credentials and limits. Configure backups only where that input may be sent.
The rule's thresholds do not change: check that the backup's scores work with
the configured predicate. The list applies only during request- or
response-stage evaluation of custom signals. Application signals, standalone
plugin reads, selectors, judges, and public inference and generation backends
keep their configured targets.

For each usable answer from a failover-enabled signal, the Router records the
deployment key that answered, including when the primary answered or the
answer did not match. OpenTelemetry's `routing.signal.deployments` attribute
is a JSON map from `systemone:<rule>` to deployment key, on the signal evaluation
span for request-stage answers and the request span for response-stage answers.
Router Replay stores request-stage answers in `route_diagnostics.signal_deployments`
and response-stage answers in the corresponding outcome's `metadata.deployment`.
If no deployment answered a rule, it has no deployment entry.

`sr_systemone_signal_failover_attempts_total` counts deployment attempts with
`deployment`, `stage` (`request` or `response`), `role` (`primary` or `backup`),
and `outcome` labels. For example, a primary returning HTTP 503 increments
`outcome="unavailable"`, and a backup returning a response increments its own
`outcome="ok"`. The counter records logical attempts, not HTTP retries or
individual answers within a batch; a successful call can still omit a rule's answer.

## Signals

A `systemone` rule, under `recipes[].signals.systemone`:

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which conditions and projections use |
| `question` | `type`, `instructions`, and `choices`, `levels` or `labels`; `threshold` and `head` for set and span |
| `labels` | The options or labels that match the rule |
| `ignore_labels` | Set and span: labels that never count |
| `predicate` | The bound the answer must meet: `gt`, `gte`, `lt`, `lte` |
| `deployment` | The deployment that answers |
| `stage` | `request` (default) or `response` |
| `input` | `history`, `previous_turns` (0 to 32), `tool_results`, `request` (`false` leaves the current user turn out), `answer`, `context`, `image`, `audio`, `video`; `max_state_bytes` caps each named part's bytes (0, the default, keeps every part whole) |
| `coverage` | `truncate` (default) or `full` |
| `on_unscanned` | With `coverage: full`: `match`, `no_match`, `unknown` (default) or `best_effort`; on `jailbreak` and `pii` rules, `best_effort` is the default |
| `timeout_ms` | Up to 60,000 |

An application rule takes `name`, `description`, `deployment`, `timeout_ms` and
the fields its type allows:

| Type | Further fields |
| --- | --- |
| `domains` | `labels` (default `[name]`), `predicate` |
| `fact_check`, `user_feedbacks` | `labels`, `predicate` |
| `jailbreak` | `predicate`, `stage`, `input.history`, `on_unknown` |
| `safety` | `labels` (harm categories), `predicate`, `stage` |
| `pii` | `predicate`, `ignore_labels`, `input.history`, `input.tool_results`, `input.request`, `on_unknown` |
| `hallucination`, `refusal` | `predicate` |
| `turn_relation` | `labels`, `input.previous_turns` (default 4) |
| `complexity` | `labels` |
| `modality` | `labels`, `predicate` |

Any other field is refused at load.

## The `match` block

```yaml title="The two stages of a match block"
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}    # optional; method: embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}   # optional
```

Each feature writes the block where it finds similar items:
`response_cache.semantic.match`, `context.sources[].match`,
`context.compress.match` and `tools.select.match`.

| Field | Meaning |
| --- | --- |
| `recall.method` | `embedding` (default; the store's embedding deployment) or `bm25` (the lexical index) |
| `recall.top_k`, `recall.min_similarity` | How many candidates recall returns (required when they come from an index), and the least similarity: a cosine for `embedding`, a BM25 score for `bm25` |
| `judge.preset` | `relevance`, `same_request` or `tool_rank` |
| `judge.question` | A custom question instead of a preset: `choice` (the candidates are its options), `noul` or `score` |
| `judge.deployment` | The deployment that answers; without one, the decision model |
| `judge.top_k`, `judge.min_score`, `judge.min_ratio` | Which judged candidates are kept: the best `top_k`, those at least `min_score`, those at least `min_ratio` of the best |
| `judge.max_candidates` | The most candidates the judge reads: 255 (also the most) for a choice judge, 64 by default and at most 4096 for a per-candidate judge; below 1 or above the most fails to load |
| `judge.on_error` | `skip` (default; keep the recall result, or the candidates as they came without recall) or `fail` (the feature's match fails) |
| `judge.timeout_ms` | The judge call's deadline; else the deployment's `timeout_ms`, else 1000 ms |

A block needs at least one stage (`match_empty` otherwise).

## Algorithms

| Field | Meaning |
| --- | --- |
| `algorithm.systemone` | The selector: `instructions`, `candidates` (model to description), `deployment`, `timeout_ms` |
| `algorithm.escalate.accept.systemone` | `{gte, deployment, timeout_ms}`: accept an answer whose `answer_check` probability is at least `gte` |
| `algorithm.escalate.accept.refusal` | `false`: accept an answer the `refusal` question does not flag |
| `algorithm.escalate.stages` | System One entrypoints: the decision models to ask in turn, each with native acceptance rules |

## Load-time codes

| Code | Kind | Means |
| --- | --- | --- |
| `span_required` | error | A span-needing rule or plugin resolves to a deployment without spans |
| `input_modality_unsupported` | error | A rule reads a medium its deployment does not read |
| `match_empty` | error | A `match` block has neither `recall` nor `judge` |
| `uncalibrated_preset` | warning | A preset is answered with the neutral threshold 0.5 |
| `renamed_signal_type` | warning | `signals.decision` or a `type: decision` condition, which load as `systemone` |
| `renamed_algorithm_type` | warning | `algorithm.type: decision`, which loads as `systemone` |
| `removed_signal_type`, `removed_field`, ... | error | A removed form; see the [migration guide](../../installation/migrate-to-v0-5.md) |
