# Vela 2.0 0.3B ARM64 packed-linear record

This record covers the ARM64 `float32-packed` part of issue #4668. It does
not cover adaptive intra-op threading (#5165), schema compaction, ONNX or
int8. The packed copy remains subject to the Vela registry's existing CPU
consent; this change only lets ARM64 hosts use the already supported path.

## Method

- **Date and revisions:** 2026-10-22. Baseline `879121d17`; candidate
  `9128bf4d8`. Vela package `vllm-sr/Vela-2.0-0.3B` at
  `a3209a50dc3ebd7e3b7520440d8fba666000f4c4`.
- **Host:** AWS Linux on ARM64, with cores 0-11 pinned by `taskset`,
  `OMP_NUM_THREADS=12`, `OMP_WAIT_POLICY=PASSIVE` and device-backend
  autoload disabled. Both sides ran in one process on the same host.
- **Accuracy:** `tools/vela2_parity.py --reduced float32-packed` compared the
  packed merged path with the runtime's exact path on deterministic generated
  requests (seed 4668). Decisions are Choice label, Score arg-max, Noul above
  0.5, Set selection and span set.
- **Latency:** `tools/vela2_bench.py` alternated `runtime-shared` and
  `copy:float32-packed` for five rounds, with 20 measured requests and five
  warm-ups per side per round, 32 request tokens, one caller and tracing.
- **Provenance:** the candidate evidence archive has SHA-256
  `2385a25453b955acd240c2ebebf010a2ce7605987c571f56f772a4bc3907e36f`.
  Machine-specific paths and process-control commands from the issue's
  attached probes were not run. Their fixed cpuset, paired measurement and
  decision-invariance methodology were retained.

## Baseline before the change

On the same host, the released engine and exact native runtime had no decision
changes, rendering mismatches or span differences over eight generated
requests. Their maximum answer difference was 0.258, above the CPU parity bar
of 1e-4, so that pre-existing strict comparison failed. This ARM64 change does
not claim to resolve that discrepancy.

The paired exact-path baseline over five rounds was:

| Metric | Reference baseline | Runtime difference, 95% interval |
| --- | ---: | ---: |
| p50 | 387.504 ms | -6.686 [-7.164, -6.208] ms |
| p95 | 394.968 ms | -7.036 [-8.113, -5.959] ms |
| Mean | 388.066 ms | -6.500 [-6.946, -6.054] ms |
| Throughput | 2.577 requests/s | +0.044 [+0.041, +0.047] requests/s |

All runtime kernels in that baseline reported `torch-reference`.

## Packed-copy accuracy

The complete eight-request run created a 441,188,352-byte packed copy. It had
no errors or changed decisions, and every measured answer type agreed with the
exact path:

| Answer type | Same / questions | Rate |
| --- | ---: | ---: |
| Choice | 9 / 9 | 100% |
| Noul | 7 / 7 | 100% |
| Score | 6 / 6 | 100% |
| Set | 4 / 4 | 100% |
| Span | 10 / 10 | 100% |

The maximum answer difference was `9.84e-7`, the maximum span difference was
`3.58e-7`, and the 99% per-answer-type floor passed.

A larger run was stopped after 129 of 300 comparisons to limit instance time.
Those 129 completed comparisons had no decision changes and a maximum absolute
difference of `4.99e-6`. Because the run was interrupted (exit 130), it is
supporting evidence rather than a completed floor result.

## Paired latency result

The packed side was faster in every round:

| Metric | Shared baseline | Packed difference, 95% interval |
| --- | ---: | ---: |
| p50 | 380.438 ms | -8.542 [-9.060, -8.024] ms |
| p95 | 386.970 ms | -9.016 [-10.072, -7.960] ms |
| Mean | 380.744 ms | -8.626 [-9.175, -8.077] ms |
| Throughput | 2.626 requests/s | +0.061 [+0.057, +0.065] requests/s |

Across 100 forwards per side, the shared path took 377.31 ms per forward
(635.89 us/token) and the packed path took 368.68 ms (621.36 us/token). This
is a 2.25% median-latency reduction and a 2.32% throughput increase for this
cell, at the cost of about 421 MiB for the packed weights.

The result supports enabling the existing packed path on this ARM64 stack. It
is incremental progress on #4668 and does not meet or claim the issue's full
Vela 1.0-relative acceptance target.
