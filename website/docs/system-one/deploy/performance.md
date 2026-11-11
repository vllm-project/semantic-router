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

- **Judges** read at most `judge.max_candidates` candidates: 64 by default for
  a per-candidate judge, at most 255 (the default) for a choice judge. Overflow
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

A rule reads what fits the model's input and drops the rest
(`coverage: truncate`), unless it reads all of it (`coverage: full`), in windows
when the input is longer than one forward. `jailbreak`, `safety`, `pii` and
`hallucination` always read their whole input; text the model could not read
counts against the request (a jailbreak or personal data), so padding cannot
carry content past them.

| Model | One decision | Windows |
| --- | --- | --- |
| vela3-fast | 32,768 tokens | 8,192-token windows for router and span questions |
| vela3-lite, -mini, -flash, -turbo | Up to 262,144 tokens | 32,768 tokens per span target |
| d3-turbo | 32,768 tokens per question; 262,144 with long context | No spans |
| d3-ultra | 131,072 tokens per question | No spans |

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
