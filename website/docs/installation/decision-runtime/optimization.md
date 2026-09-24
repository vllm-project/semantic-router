---
title: Measure and optimize
description: Tune Decision Runtime without hiding semantic changes or per-model regressions.
---

Start with a working, ready eager instance and record its model ID, exact
revision, image digest or local validation image ID, artifact content ID,
device, dtype, and `/api/status` scheduler limits. Keep those fixed when
comparing one setting at a time. A faster response on one model or shape does
not establish a faster runtime for all six models.

1. Verify answer and input-token parity before timing. Choice and Score
   confidence use Decision's type-aware formulas; they are not numerically
   interchangeable with an old preview's maximum-probability statistic.
2. Measure both one-state/many-question and multi-state/shared-question
   workflows. The batch endpoint reduces repeated HTTP work, so multi-state
   throughput changes must not be attributed solely to a GPU kernel.
3. Record each shape and client concurrency separately. Observe `529` and
   `Retry-After`, queue length, active rows, physical batch sizes, and
   preparation time along with decisions per second and latency.
4. Rerun a complete paired measurement when a result is borderline or noisy.
   Retain every attempt; do not select favorable rounds or omit a losing cell.

The repository's `bench/decision_runtime/README.md` describes the paired
synthetic workflow harness. Its fixed-arrival mode is exploratory and is not
the protected release gate. The current protected policy requires at least
1.00× old-arm throughput in **each** of 54 predeclared model/shape/concurrency
cells, including c1, c8, and c32; an average gain cannot compensate for a
loss. Latency remains diagnostic. These are same-snapshot HTTP workflow
measurements, not task-accuracy scores or a universal speedup claim.

Begin with the profile's B8 physical batch. A larger `--max-batch` requires
verified kernel-profile coverage and model-backed numerical, memory, and
performance qualification; a deeper queue is not a substitute. The optional
[Sol ROCm B8 graph](./parameters.md#experimental-sol-graph) is off by default
and should be evaluated as a separate runtime variant with capture, replay,
and fallback telemetry. Preserve the eager result as the comparator.
