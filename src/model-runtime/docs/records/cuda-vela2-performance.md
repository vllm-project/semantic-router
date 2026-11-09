# Vela 2.0 performance on NVIDIA H200

Native CUDA vs each package's Transformers reference, ten interleaved
rounds, concurrency 1. Difference is runtime minus reference, 95%
t-interval. Machine readable:
[`cuda-vela2-performance.json`](cuda-vela2-performance.json).

- **Date:** 2026-10-20.
- **Hardware:** NVIDIA H200.
- **Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` plus the pod
  venv. Not the published `vllm-sr-cuda` image.
- **Head:** `ec307ad6ed081dd220b5dbb3571a1f9987d06914`.
- **Device:** `cuda:0`. Packages: 0.3B `a3209a50`, 0.8B and 4B as pinned in
  `registry/tables/vela2.py`.
- **Workloads:** 0.3B @ 32 tokens (40 requests/side/round), 0.8B @ 512, 4B @
  2048. These are fresh ten-round runs, not pooled with earlier batches.

| Workload | p50 (ms) [runtime − ref, 95% CI] | p95 | Mean | Throughput |
| --- | --- | --- | --- | --- |
| 0.3B @ 32 | **Worse** +0.508 [+0.088, +0.928] vs 21.959 | Better | Level | Level |
| 0.8B @ 512 | Better −6.100 [−6.337, −5.863] vs 79.966 | Better | Better | Better |
| 4B @ 2048 | Better −28.862 [−29.268, −28.456] vs 400.486 | Better | Level | Level |

**Known regression on this H200.** The 0.3B @ 32-token p50 interval is
wholly worse (~2.3% slower). Ten rounds are the repeat-policy cap; this
record keeps that H200 result rather than replacing it with another
opportunistic rerun. p95 on that workload improves. The no-regression
standard judges each row on its own hardware, so a Tesla T4 Level interval
does not clear this H200 row.

## Tesla T4 follow-up (2026-10-21)

Five interleaved rounds, concurrency 1, 40 requests/side/round. Difference is
runtime minus reference, 95% t-interval. Same package pin as the H200
follow-up (`a3209a50`). This is a new card, not pooled with the ten-round
H200 row.

- **Date:** 2026-10-21.
- **Hardware:** NVIDIA Tesla T4.
- **Image:** Google Colab GPU runtime, torch 2.11.0+cu130, CUDA 13.0.
- **Head:** `a1c137bcc9821766f9c0a440cc2912a21c671f3d`.
- **Device:** `cuda:0`. Package: Vela-2.0-0.3B `a3209a50`.

| Workload | p50 (ms) [runtime − ref, 95% CI] | p95 | Mean | Throughput |
| --- | --- | --- | --- | --- |
| 0.3B @ 32 | Level −0.158 [−0.447, +0.131] vs 57.288 | Level | Level | Level |

Five rounds meet the no-regression minimum on this T4. p50 is **Level**
(the interval includes 0). This is an additional card, not a replacement
gate for the H200 known regression above.

## Earlier 4×4 coverage (separate stack)

Older saved files cover **all four models at 32, 128, 512, and 2048
tokens**, including 9B. Head and image for those files are not labelled;
do not pool them with the ten-round follow-ups above. Runtime minus
reference, 5 rounds, concurrency 1:

| Model | Tokens | Rounds | p50 runtime−ref (ms) [95% CI] | Verdict |
| --- | ---: | ---: | --- | --- |
| 0.3B | 32 | 5 | −2.008 [−54.457, +50.441] vs 61.858 | Level |
| 0.3B | 128 | 5 | −10.370 [−52.731, +31.991] vs 63.342 | Level |
| 0.3B | 512 | 5 | −2.638 [−3.566, −1.710] vs 28.352 | Better |
| 0.3B | 2048 | 5 | −15.730 [−18.801, −12.659] vs 85.954 | Better |
| 0.8B | 32 | 5 | −14.032 [−30.154, +2.090] vs 380.872 | Level |
| 0.8B | 128 | 5 | −23.090 [−39.902, −6.278] vs 377.078 | Better |
| 0.8B | 512 | 5 | −35.880 [−71.701, −0.059] vs 417.400 | Better |
| 0.8B | 2048 | 5 | −34.450 [−74.950, +6.050] vs 1088.666 | Level |
| 4B | 32 | 5 | −153.134 [−230.206, −76.062] vs 557.336 | Better |
| 4B | 128 | 5 | −99.042 [−167.808, −30.276] vs 416.014 | Better |
| 4B | 512 | 5 | −19.820 [−24.945, −14.695] vs 399.472 | Better |
| 4B | 2048 | 5 | −79.560 [−135.296, −23.824] vs 741.146 | Better |
| 9B | 32 | 5 | −39.024 [−107.946, +29.898] vs 239.008 | Level |
| 9B | 128 | 5 | −25.596 [−29.527, −21.665] vs 207.822 | Better |
| 9B | 512 | 5 | −62.594 [−95.852, −29.336] vs 486.268 | Better |
| 9B | 2048 | 5 | −110.792 [−136.075, −85.509] vs 893.604 | Better |
