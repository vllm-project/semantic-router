---
title: Performance and cost
sidebar_label: Performance and cost
sidebar_position: 2
description: What System One costs a request, and the batching, caps, budgets and placement choices that bound it.
---

# Performance and cost

A System One question costs a forward pass of a small model, not a generation
by a chat model: there is no output to decode and no text to parse. What a
request costs is the number of calls it makes, the size of what each reads, and
where the model runs.

:::tip
Most requests make **one** call: the request stage's. A rule that nothing reads
is not asked, and a stage with no question makes no call.
:::

## Calls per request

| Stage | Calls | Asked when |
| --- | --- | --- |
| Request signals | One per deployment, with every question about every state in it | A decision, projection or request plugin reads a rule |
| `systemone` selector | One | The matched decision uses the selector |
| Judges (`same_request`, `relevance`, `tool_need` + `tool_rank`) | One per feature, with one entry per judged candidate | The feature runs and has candidates |
| `masking` | One, for the text the request stage did not already answer | The decision masks |
| Response signals | One per deployment | A `response_guard` rule reads them |
| `escalate` checks | One per judged answer | `accept.systemone` or `accept.refusal` |

## Batching

- **Questions share a call.** Every question a stage asks one deployment
  travels together, whatever signal or rule it comes from. The model runtime
  batches calls from concurrent requests too
  ([engine arguments](../../model-runtime/deploy/engine-args.md#batching)).
- **Questions share an answer.** Rules of one preset on one deployment about
  one state share one question.
- **State-once models** (d3-turbo, d3-ultra) read a state once and answer
  every question from that reading, so the tenth question about a request costs
  little more than the first. Decision 2.0 computes the shared context of a
  multi-question request once (`enable_shared_context`).
- **Answers are reused.** The Router keeps the last 4,096 answers per local
  model (`VLLM_SRUN_RESULT_CACHE` changes the number; 0 turns it off), and a
  remote API's answers for up to 10 minutes, until the configuration changes.

## Caps

- **Judges** read at most `judge.max_candidates` candidates: 64 by default and
  at most 4096 for a per-candidate judge, at most 255 (the default) for a
  choice judge. Overflow
  is truncated by recall rank, or in listed order without recall, never failed;
  `vsr_match_judge_truncated_total` counts it.
- **Recall** bounds what a judge sees: a cache judge confirms its top 3
  recalled hits, a RAG judge its top 20 documents.
- **Tools** above 255 after the policy are truncated, or narrowed first by
  `recall: {method: bm25}`.

## Budgets and deadlines

| Setting | Bounds |
| --- | --- |
| A rule's or judge's `timeout_ms` | Its answer, up to 60,000 ms; else the deployment's, else 1,000 ms |
| A deployment's `timeout_ms` | One attempt of a call to an attached or remote deployment |
| `global.model_catalog.signal_timeout_ms` | All of a request's signals; 0 (the default) derives it from the request's own deadline |
| An executor's `budget` | `escalate`'s deadline and calls, its `answer_check` calls included in the deadline |

:::note
A late answer leaves its questions unknown, and the failure policy of each rule
or feature decides. A deadline ends the Router's wait, but does not stop a
forward pass already running.
:::

## Long inputs

A long user turn, such as a pasted document, a log or a long tool output, is
read in one of two ways, by the Router itself, so a local and a remote decision
model behave the same.

**Routing questions read a bounded view.** These are the rules that do not
read their whole input: `domains`, `complexity`, `modality`, `fact_check`,
`turn_relation`, `user_feedbacks` and `systemone` rules with
`coverage: truncate`.

- A turn longer than the deployment's `routing_view` keeps its head and its
  tail around a marker, `[... 41000 tokens omitted ...]`.
  The head takes a quarter of the view and the tail the rest, because the
  request's instruction usually closes the turn.
- A turn within the view is sent byte for byte as it came.

**Guard rules read the whole input, in windows.** These are `jailbreak`,
`safety` and `pii`, at the request and at the response stage.

- An input longer than one `scan_window` is split into windows that overlap by
  `scan_window.overlap`, so text across a cut is read whole by one of them.
- The windows are read tail first, then the head, then the middle in order:
  one batch to a local model, and at most `max_concurrency` calls at a time to a
  remote API.
- Any window that finds an attack, a harm or personal data flags the input.
  `pii` spans keep their places in the input, so masking replaces every one.
- One input reads at most 64 windows; the rest counts as unread.

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        routing_view: {max_tokens: 8192, head_share: 0.25}
        scan_window: {tokens: 8192, overlap: 512}
      hosted-decision:
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: example-decision-model
        timeout_ms: 6000
        routing_view: {max_tokens: 4096}
        scan_window: {tokens: 4096, overlap: 256}
```

| Setting | Default | Bounds |
| --- | --- | --- |
| `routing_view.max_tokens` | 8,192; 4,096 for `provider: systemone` | 256 to 1,048,576 |
| `routing_view.head_share` | 0.25 | 0 to 1 |
| `scan_window.tokens` | 8,192; 4,096 for `provider: systemone` | 256 to 1,048,576 |
| `scan_window.overlap` | 512; 256 for `provider: systemone` | 0 to less than half a window |

**Views and windows are sized in bytes.** A token of `max_tokens` or
`scan_window.tokens` counts as four bytes, about what a token of prose takes,
so the 8,192 default is a 32 KB view. The model's validated length (its card's
`validated_input_tokens`, else `max_input_tokens`) less 1,024 tokens for the
question caps both at one byte a token, because a byte-level tokenizer gives no
byte more than one token. Dense text such as JSON or base64 takes far fewer
bytes a token than prose, so a view of it holds up to about three times as many
tokens, but no setting makes a view or a window overrun the model: with a
32,768-token validated length, neither ever holds more than 31,744 bytes.

**Size them from the deadline.** A call should end well before its deadline,
which is the rule's or the deployment's `timeout_ms` for a remote API, and the
signals' deadline for a local model (45 s for a served request by default,
`global.model_catalog.signal_timeout_ms`). If an API reads R tokens a second and
takes T seconds to answer the smallest call, a view of up to
`(timeout - T) × R` tokens answers in time: an API that reads 2,000 tokens a
second and takes 2 s for the smallest call reads about 8,000 tokens in a 6 s
timeout, a `max_tokens` of 8,000 for prose, or about 2,700 if turns are often
JSON or base64. A scan's windows share the deadline: a local model reads them
in one batch, and a remote API `max_concurrency` at a time. Windows that do not
answer in time count as unread, and with `best_effort` the rule decides on the
windows read. Lower `scan_window.tokens` for a slow API, and raise
`max_concurrency` or the deadline for inputs you expect to scan in full.

**Coverage and failures.** When the deadline ends a scan before every window is
read, `on_unscanned` decides: `best_effort` (the default) decides on the windows
read, `match` matches (strict), `no_match` does not, and `unknown` leaves the rule
to `on_unknown`. When no window is read at all (the model is down, or no
window answered in time), a `jailbreak` or `pii` rule follows `on_unknown`,
which matches by default (fail closed), and a `safety` rule is unknown, which
its decision's `rules.on_unknown` decides.

Coverage is reported for every guard rule that read its input in windows:

- `vsr_signal_scan_coverage_total{signal, result}` counts each guard rule's read
  as `full`, `partial` or `none`;
- the routing preview (`scan_coverage`) and the Router Replay record
  (`scan_coverage`) give `full`, `partial n/m` or `none` per rule;
- `x-vsr-trace` carries it as `scan_coverage` for a caller allowed the debug
  header.

| Model | Validated length per question | Maximum |
| --- | --- | --- |
| vela3-fast | 32,768 tokens; its own 8,192-token windows for router and span questions | 32,768 |
| vela3-lite, -mini, -flash | 262,144 tokens; 32,768 per span target | 262,144 |
| vela3-turbo, d3-turbo | 32,768 tokens | 262,144 with long context |
| d3-ultra | 131,072 tokens | 262,144 with long context |

:::warning
A whole-input scan of a long request on a CPU with few cores can take longer
than the signal deadline. Measure your input lengths and concurrency, then give
the deployment dedicated CPUs or a GPU.
:::

## Placement

- **vela3-fast on a CPU** is the default. The loaded deployment holds about
  7 GB; give the Router 12 GiB.
- **A GPU** runs the larger Vela 3.0 and Decision sizes, and serves many more
  requests per second at the same latency.
- **A remote API** adds a network round trip to each call, and its models have
  no calibrated thresholds.
- **One deployment for everything** loads one model. Spreading types over
  several deployments makes one call per deployment per stage.

[Choose a model, size and hardware](../../model-runtime/get-started/choose-a-model.md)
has the measured latency per size, and the
[model runtime records](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)
the full measurements.
