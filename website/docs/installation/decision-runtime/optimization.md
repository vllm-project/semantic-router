---
title: Tune performance
description: Measure Decision Runtime and adjust request capacity for your workload.
---

Start with the [default capacity settings](./parameters.md#request-capacity):
eight active requests, 32 waiting requests, and at most eight compatible
decision rows in one model forward. The runtime can combine rows from different
requests, so traffic shape matters as much as the model you choose.

## Choose the right request shape

Use [`POST /v1/systemone`](./api.md#one-state-three-question-types) for one
state with one or more questions. If several states share the same questions,
use [`POST /v1/decision/batches`](./api.md#many-states-one-question-set) to
send them together. The batch endpoint reduces repeated HTTP and question
payload work; it returns an answer for every state and question in caller
order. It does not change what the questions mean.

## Decide what to change

| What you observe | First check | Next step |
| --- | --- | --- |
| `529` responses during short bursts | `queued` and `max_queue` in `GET /api/status` | Honor `Retry-After`; a larger `--max-queue` can let a short burst wait. |
| Latency rises as the queue grows | Queue depth and p95 latency | Reduce sustained load or add capacity. Increasing the queue only makes requests wait longer. |
| Throughput stays low with little queueing | Model execution time and physical batch sizes in `GET /metrics` | Test a small `--max-concurrency` change on the same model and request mix. |

`--max-batch` is a separate control: it limits rows per model forward, not
incoming requests. Its default is eight for every model. A larger value on
ROCm needs a supported kernel profile and a check of answers, memory use, and
performance on the selected model and device. The
[optional Sol graph](./parameters.md#experimental-sol-graph) is another
separate setting; test it against ordinary Sol execution at batch size eight.

## Measure a change

Keep the model revision, image, device, and request examples fixed. Record
answers and input-token counts, then compare decisions per second, p50/p95
latency, error rate, queue depth, and observed physical batch sizes. Change
one setting at a time and repeat the same traffic pattern. Measure
single-state/many-question traffic separately from multiple states with shared
questions; batching can reduce HTTP work even when model execution is
unchanged.

For repeatable paired runs, use the
[Decision benchmark harness](https://github.com/vllm-project/semantic-router/tree/main/bench/decision_runtime).
Keep results for each model separately so a gain on one model does not conceal
a slowdown on another. Throughput measurements do not establish answer quality;
compare the actual answers as well.
