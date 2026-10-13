# Decision 3.0 length buckets

Decision 3.0 graphs are captured per exact padded shape, so real traffic, where
almost every prompt has its own length, keeps meeting shapes it has not
captured and runs them eagerly. Under the approximate profiles (`batching`,
`max_speed`) a text forward now pads its width up to a multiple of 32 tokens,
on the left like every d3 row, when the padded shape replays a captured graph
or the graph cache can still capture it: within the graph limit
(`MAX_GRAPH_TOKENS`, 4,096), not failed before, and with room left in the cache
(`Graphs.admits`). Prompts of nearby lengths then share one graph. Once the
cache is full, only shapes it already holds are padded, so no forward pays for
padding it cannot replay. The exact profile, media passes and larger forwards
are unchanged.

- **Date:** 2026-10-13 (bucket size), 2026-10-14 (serving).
- **Device:** one AMD Instinct MI300X (gfx942) in the router's ROCm image
  `vllm-sr-rocm:latest` built 2026-10-11 (PyTorch 2.12.0, Triton 3.7.0,
  FLA 0.5.2), pinned kernel choices.
- **Model:** `vllm-sr/d3-flash` at the built-in revision `521525ef`.
- **Code:** main with the cross-request batching of #5006, the Decision 3.0
  graphs branch, and this branch.

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
second set sends 240 other prompts of the same length distribution to the same
server. Every server was started fresh. Main ran twice, and the two runs show
the run-to-run spread at concurrency 16.

| Concurrency | Main, run 1 | Main, run 2 | Graphs | Graphs and buckets | Buckets with #5009 |
| --- | --- | --- | --- | --- | --- |
| 1, first set | 24.3 rps | 25.2 rps | 21.6 rps | 33.6 rps | 33.5 rps |
| 1, second set | 24.2 rps | 25.1 rps | 23.7 rps | 34.9 rps | 34.9 rps |
| 4, first set | 42.9 rps | 44.5 rps | 39.9 rps | 40.8 rps | 40.9 rps |
| 4, second set | 45.0 rps | 45.8 rps | 42.8 rps | 44.5 rps | 44.6 rps |
| 16, first set | 46.7 rps | 40.8 rps | 37.9 rps | 41.1 rps | 40.1 rps |
| 16, second set | 45.9 rps | 45.5 rps | 47.3 rps | 43.7 rps | 42.8 rps |

One request at a time went from a median of 39 to 41 ms to 29 ms, and from a
p95 of 44 to 47 ms to 36 ms. Buckets help single requests most, because one request's forward is small and
host-bound until it replays a graph, and graphs alone do not help there: new
lengths keep meeting shapes not captured yet. At concurrency 4 the stack stays
1 to 8% below main, and at 16 it is within main's own spread on the first set
and 4 to 5% below it on the second. Coalesced forwards meet many shapes and
pay for their captures, which likely explains it, though the capture time was
not measured separately. With the graph budget fix of #5009 every number stays
within 1 rps.

## Answers

Against the exact profile on the same prompts, buckets changed 0 to 1 of 480
decisions per run (largest probability change 0.015), the same range as main's
batching (0 to 2 of 480, 0.015). In process, on 600 oasst1
train prompts with three questions each, 32-token left padding changed 5 of
1,800 decisions (domain 1, reasoning 2, difficulty 2; largest probability
change 0.013). Right padding was not better (4 of 1,800 at 32 tokens, 8 of
1,800 unbucketed), so rows stay left-padded as the released runtime pads them.
