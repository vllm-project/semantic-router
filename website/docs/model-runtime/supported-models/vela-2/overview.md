---
title: Vela 2.0
sidebar_label: Vela 2.0
description: The previous Router models, from 0.3B on a CPU to 9B on a GPU, which answer set and span questions and have calibrated thresholds.
---

# Vela 2.0 {#vela-20-family}

Vela 2.0 is the previous family of Router models. Besides `choice`, `noul` and
`score`, it answers questions that pick several labels (`set`) or mark spans of
text (`span`), and the Router routes on both. Every size has calibrated
thresholds for the built-in questions.

```bash title="Restore the previous default on a CPU"
vllm-sr serve vllm-sr/Vela-2.0-0.3B --platform cpu
```

## Sizes

<!-- BEGIN GENERATED MODEL TABLE -->

| Model | Parameters | Answers | Spans | Reads | Device memory |
| --- | --- | --- | --- | --- | --- |
| [Vela-2.0-0.3B](./vela-2-0-3b) | 0.309B | choice, noul, score, set, span | router | text | 4 GiB |
| [Vela-2.0-0.8B](./vela-2-0-8b) | 0.756B | choice, noul, score, set, span | router, broad | text | 4 GiB |
| [Vela-2.0-4B](./vela-2-4b) | 4.21B | choice, noul, score, set, span | router, broad | text | 16 GiB |
| [Vela-2.0-9B](./vela-2-9b) | 7.95B | choice, noul, score, set, span | router, broad | text | 24 GiB |

<!-- END GENERATED MODEL TABLE -->

- **On a CPU, run the 0.3B**, a ModernBERT encoder that reads up to 8,192
  tokens. The 0.8B is a decoder: on a CPU a request takes seconds.
- **On a GPU**, the larger sizes read inputs of up to 16,384 tokens: the 0.8B
  costs the least of them, and the 4B and 9B are the most accurate.
- **Span heads:** every size has the router span head, which answers the
  [`pii`](../../../tutorials/signal/application/pii.md) and
  [`hallucination`](../../../tutorials/signal/application/hallucination.md)
  signals with its own ready-made questions; the decoders add the broad head.

[Choose a model](../../get-started/choose-a-model.md#choose-a-size) has each
size's accuracy, latency and calibrated thresholds.

## Vela 2.0 0.3B, the previous default {#vela-20}

Naming `vllm-sr/Vela-2.0-0.3B` as the `primary` deployment's artifact restores
it, and the calibrated thresholds follow it (jailbreak 0.75, domain 0.28, PII
0.01, fact check 0.93, user feedback 0.37).

When it became the default, the maintainers chose the 0.3B although it missed
two goals they had set for it
([#4639](https://github.com/vllm-project/semantic-router/issues/4639)): level or
better accuracy on every signal, and level or better latency on a CPU. Measured
through the Router on the
[router signal suite](https://huggingface.co/datasets/vllm-sr/router-signal-suite)
([A/B record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-router-signals.md)):

- **Ahead:** prompt guard (held-out AUC +0.026; on the E2E attack fixtures it
  blocks all six attacks, Vela 1.0 Guard five) and safety (+0.052 held-out, and
  ahead on every set). The signals share one model.
- **Level:** PII and hallucination on held-out and fresh files.
- **Behind, most:** modality (held-out AUC −0.180; the 0.3B misses most requests
  that ask for a new image) and user feedback (accuracy −0.038 held-out, −0.178
  fresh).
- **Behind:** domain (accuracy −0.037 held-out, −0.088 fresh) and fact check
  (held-out AUC −0.101).
- **CPU time:** each request carries the questions, their options and the 17
  PII labels (at least 560 tokens) through the 307M-parameter model, where each
  Vela 1.0 model reads only the request. On 12 CPU cores, for the five request
  signals of the
  [latency record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/router-latency-cpu.md),
  the median request takes about 4.9 times as long:

| Router on 12 CPU cores | p50 | p95 | Requests per second | At concurrency 16 |
| --- | ---: | ---: | ---: | ---: |
| Vela 1.0 specialists (since removed) | 16 ms | 58 ms | 38.9 | 51.8 |
| Vela 2.0 0.3B (the previous default) | 79 ms | 100 ms | 11.9 | 12.8 |

[#4668](https://github.com/vllm-project/semantic-router/issues/4668) works on the
CPU latency. On a GPU, the 0.3B answers the same questions in about 7 ms at the
median on one AMD Instinct MI325X
([measurements](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/vela2-performance.md#against-the-vela-10-path)).

:::note
The Vela 1.0 specialist classifiers answer no System One question, so a binding
cannot name them; every application signal asks a decision model.
:::
