---
title: Measure and optimize
description: Tune Decision Runtime without hiding semantic changes or per-model regressions.
---

Decision models score many small, independent question rows against one
resident model. The runtime prepares rows once, batches compatible rows across
requests, and returns them in caller order. This helps when a single state has
many questions or a [batch request](./api.md#several-states-shared-questions)
shares questions across states; it does not make every model or workload
equally fast.

## Tune capacity

Start with the model profile's B8 physical batch, 8 admitted requests, and 32
waiting requests. Increase `--max-concurrency` only after measuring the target
model and workload. A deeper `--max-queue` absorbs bursts but does not raise
GPU throughput; clients should honor `529` and `Retry-After` rather than retry
in a tight loop. `/api/status` reports the live limits, while `/metrics`
exposes queueing, preparation, physical-batch sizes, and model execution.

`--max-batch` controls rows in one forward, not requests. A value above B8
requires kernel-profile coverage plus numerical, memory, and throughput
qualification on that model and device. The optional
[Sol ROCm B8 graph](./parameters.md#experimental-sol-graph) is a separate,
off-by-default variant; compare its capture, replay, and fallback counters to
the eager path before adopting it.

## Compare changes fairly

Keep model revision, artifact content ID, image, device, dtype, and request
shapes fixed while changing one runtime setting. First check answer and
input-token parity, then measure decisions per second, p50/p95 latency, error
rate, queue depth, and observed physical forwards at each concurrency. Compare
one-state/many-question and multi-state/shared-question traffic separately:
the batch API removes repeated HTTP work, so its speedup is not solely a GPU
kernel improvement. Choice and Score confidence use Decision's type-aware
formulas; they should not be compared numerically to a historical preview's
maximum-probability statistic.

For reproducible paired runs, use the
[benchmark harness](https://github.com/vllm-project/semantic-router/tree/main/bench/decision_runtime).
The protected release rule checks all six models and each declared workload
cell for no throughput regression; a gain in one model cannot hide a loss in
another. Keep every attempt, including borderline or failed runs. These
measurements do not replace model-quality evaluation.
