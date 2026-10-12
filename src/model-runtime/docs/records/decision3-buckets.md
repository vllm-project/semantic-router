# Decision 3.0 length buckets

Decision 3.0 graphs are captured per exact padded shape, so real traffic, where
almost every prompt has its own length, keeps meeting shapes it has not
captured and runs them eagerly. Under the approximate profiles (`batching`,
`max_speed`) a text forward whose rows times padded width fit the graph limit
(`MAX_GRAPH_TOKENS`, 4,096) now pads its width up to a multiple of 32 tokens,
on the left like every d3 row. Prompts of nearby lengths then share one graph.
The exact profile, image and video requests and larger forwards are unchanged.

- **Date:** 2026-10-13.
- **Device:** one AMD Instinct MI300X (gfx942) in the router's ROCm image
  `vllm-sr-rocm:latest` built 2026-10-11 (PyTorch 2.12.0, Triton 3.7.0,
  FLA 0.5.2), pinned kernel choices.
- **Model:** `vllm-sr/d3-flash` at the built-in revision `521525ef`.
- **Code:** the Decision 3.0 graphs branch against this branch.

## Why 32 tokens

Graph replay time grows linearly with the padded length and shows no steps at
tile boundaries: one row takes 10.7 ms at 64 tokens and 27.3 ms at 544, three
rows 15.1 and 58.7 ms. Every padded token costs about as much as a real one,
so the bucket has to stay small. With every bucket already captured, 188
three-question oasst1 prompts ran at a median of 24.4 ms with 32-token
buckets, 27.7 ms with 64 and 31.5 ms with 128, against 38.4 ms for exact
shapes. Rows are not padded: rounding three rows up to four added a third of
the work.

## Throughput and latency

`vllm-srun serve --profile batching` over HTTP, 240 two-question requests of
new lengths (5 to 250 words), each concurrency after a 16-request warm-up. The
second pass sends 240 other prompts of the same length distribution to the
same server.

| Concurrency | Exact | Batching | Batching with buckets |
| --- | --- | --- | --- |
| 1, first pass | 20.0 rps, p50 41.8 ms, p95 74.7 ms | 20.3 rps, p50 43.6 ms, p95 76.6 ms | 33.2 rps, p50 29.6 ms, p95 36.5 ms |
| 1, second pass | 29.7 rps, p50 30.8 ms, p95 69.2 ms | 22.4 rps, p50 43.0 ms, p95 76.8 ms | 34.7 rps, p50 29.1 ms, p95 36.5 ms |
| 4, first pass | 24.0 rps | 40.6 rps | 41.2 rps |
| 4, second pass | 33.4 rps | 38.6 rps | 43.6 rps |
| 16, first pass | 39.3 rps | 41.7 rps | 43.4 rps |
| 16, second pass | 40.1 rps | 45.1 rps | 44.3 rps |

Buckets help single requests most, because one request's forward is small and
host-bound until it replays a graph. Batches at concurrency 16 are mostly
above the graph limit or GPU-bound, so they gain nothing, and they are left
unpadded.

## Answers

Against the exact profile on the same prompts, buckets changed 0 to 2 of 480
decisions per run (largest probability change 0.016), the same range as
batching without buckets (0 to 2 of 480, 0.019). In process, on 600 oasst1
train prompts with three questions each, 32-token left padding changed 5 of
1,800 decisions (domain 1, reasoning 2, difficulty 2; largest probability
change 0.013). Right padding was not better (4 of 1,800 at 32 tokens, 8 of
1,800 unbucketed), so rows stay left-padded as the released runtime pads them.
