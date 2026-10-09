# Embedders, rerankers and Omni: performance

The runtime against the legacy bindings it replaces, on the same inputs and
hardware (design section 18), and the CPU engine decision between ONNX
Runtime and PyTorch. Every row against legacy follows the no-regression
standard: at least five interleaved rounds on the same cgroup-confined cores
(on ROCm, the same GPU and host cores), with 95 % intervals of runtime minus
legacy. A row passes when its interval is at or better than legacy; a row
whose interval straddles zero is level.

- **Date:** 2026-10-05; Omni on the native engine
  ([#4619](https://github.com/vllm-project/semantic-router/issues/4619)) on
  2026-10-06.
- **Runtime:** the CPU rows against legacy (Omni, the encoders and the
  native-model probe) at `e0e0e2850`, in one run; native Omni at `2afe0f878`. The ROCm rows run the
  router image's own runtime (`a580be6b9`); this branch's commits since
  change ONNX Runtime, Omni, request-option parsing and the heap freeze
  after a runtime loads, none of which `tools/embed_bench.py` or the native
  engine on the GPU runs. The engine-level CPU rows and the engine decision
  are from 2026-10-04 at `7c2c6e21b`.
- **CPU:** node B vCPUs 112–127 (16 vCPUs of an AMD EPYC 9575F KVM guest, one
  NUMA node), PyTorch 2.10.0 (oneDNN, MKL) and ONNX Runtime 1.30.0 in the
  runtime (the router image's pins), ONNX Runtime 1.22.0 in the legacy
  bindings. Every process of both sides runs in one cgroup cpuset of those 16
  vCPUs (`systemd-run --scope -p AllowedCPUs=112-127`; the effective cpuset is
  logged). A `taskset` mask is not enough: the legacy ONNX Runtime sessions
  pin their threads to CPUs they read from the host, which a mask does not
  stop; inside the cpuset that pinning fails and the threads stay on the 16
  vCPUs. The guest's 1-minute load was 5–19 of 160 during the run (other
  workloads, on other vCPUs).
- **ROCm:** one AMD Instinct MI325X (gfx942, node B GPU2) for both sides, with
  host vCPUs 80–87 in docker cgroup cpusets. The runtime side runs in the
  router's ROCm image, `Dockerfile.extproc` at `a580be6b9` (`ACCELERATOR=rocm`): vLLM's ROCm
  PyTorch 2.12.0+git6bbd260 (AOTriton 0.13.50, ROCm 7.2.3) and Triton 3.7.0.
  The legacy side runs in the legacy router image: ONNX Runtime 1.22.1 ROCm
  with the CK flash-attention operator library. Host load 2–11 of 160.
  `router` ran these rounds for this record (2026-10-05, 23:40–23:45).
- **Legacy paths:** the router's native facade at `61aa7eb2d` (candle on the
  CPU for Vela Embedding, Reranker and Qwen3-Embedding, the router's default;
  ONNX Runtime on the prepared bundle for Omni), and the legacy ONNX Runtime
  execution as the bindings ran it: one sequence per session run, pairs one
  after another (`tools/embed_legacy_baseline.py`; on ROCm the AMD recipe's
  `onnx/model_fa.onnx` with `ck_flash_attention`).
- **Runtime side:** `Runtime.call` as the HTTP server calls it, with the
  result cache off: one event loop running on its own thread, every input's
  request and encoded size built before timing (the server reads the size off
  the request). Engine-level scenarios run `LoadedModel.run` on token rows
  (`tools/embed_bench.py`).

## CPU: the A/B method

`tools/embed_legacy.py ab`: the legacy facade's test binary serves calls in
one process, the runtime in another, on the same 16 vCPUs. Per input, one
legacy and one runtime call alternate, each 50 ms after the previous call, so
neither side's ONNX Runtime threads still spin on the cores when the other
side's call starts. The order flips every round. A throughput run adds 10 s
closed-loop windows of 4 callers per side after each round's pairs, in the
same rotation. On the legacy side the callers are goroutines; on the runtime
side they are concurrent requests on the server's event loop, as the server
serves concurrent connections. Latency intervals resample the call pairs
(paired bootstrap, p50 and p95); throughput intervals resample the rounds
(median req/s of each side's windows). The process serves every model of the
run: Nano and Mini for Omni, the three encoders for the encoder rows.

## CPU: encoders, legacy router path vs `Runtime.call`

Six interleaved rounds over 31 corpus texts (short queries to documents
of a few thousand tokens) or 4 rerank sets of 8 documents.

**Sequential pairs.**

| Job | Pairs | Legacy p50 / p95 ms | Runtime p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 186 | 38.02 / 3,518 | 10.48 / 211.5 | -27.54 [-31.85, -21.87] better | -3,307 [-5,952, -1,456] better |
| Embedding (256, layer 22) | 186 | 38.68 / 3,527 | 10.42 / 216.4 | -28.27 [-32.71, -21.71] better | -3,310 [-5,996, -1,383] better |
| Embedding (768, layer 11) | 186 | 19.35 / 1,729 | 6.10 / 109.1 | -13.25 [-17.35, -11.41] better | -1,620 [-2,861, -665.4] better |
| Reranker (22, 768), 8 documents | 24 | 306.4 / 491.5 | 36.68 / 62.66 | -269.7 [-273.5, -267.5] better | -428.8 [-432.1, -422.5] better |
| Reranker (6, 256), 8 documents | 24 | 81.84 / 130.1 | 12.37 / 19.86 | -69.47 [-71.46, -68.09] better | -110.2 [-111.7, -108.4] better |
| Qwen3-Embedding | 186 | 85.14 / 10,183 | 26.10 / 663.7 | -59.04 [-79.89, -49.43] better | -9,519 [-17,996, -3,501] better |

**4 callers.**

| Job | Rounds | Legacy req/s | Runtime req/s | Δ req/s [95% CI] |
| --- | --- | --- | --- | --- |
| Embedding (768, layer 22) | 6 | 5.01 | 27.71 | +22.70 [+22.12, +24.84] better |
| Embedding (256, layer 22) | 6 | 5.04 | 27.62 | +22.59 [+21.57, +25.75] better |
| Embedding (768, layer 11) | 6 | 12.90 | 50.97 | +38.08 [+36.91, +39.79] better |
| Reranker (22, 768), 8 documents | 6 | 7.94 | 25.37 | +17.43 [+17.14, +17.75] better |
| Reranker (6, 256), 8 documents | 6 | 30.20 | 81.09 | +50.90 [+48.54, +52.39] better |
| Qwen3-Embedding | 6 | 2.18 | 8.35 | +6.16 [+5.81, +6.28] better |

The runtime runs the encoder's linear layers through oneDNN's pre-packed
FP32 kernels (the `exact` path's kernel variant on x86 CPUs), packs a rerank
set's pairs into one forward (the legacy scorer ran them one by one), and
runs long documents packed with block-local attention instead of candle's
dense padded path.

## CPU: Omni on the native engine (#4619)

Since #4619 the runtime serves Omni's published weights on the native engine
(design section 8.5). Two A/Bs by the method above, ten interleaved rounds
each, with the runtime side at `2afe0f878` and the inputs of the section below:

- **Against the legacy facade,** its ONNX Runtime 1.22 adapter on the prepared
  bundles, as the section below measured the ONNX Runtime path.
- **Against the ONNX Runtime path this replaces:** staging `91d369ff2`'s own
  runtime serving the prepared bundles on its `onnxruntime` engine, with that
  tree's defaults (`embed_legacy.py ab --baseline-engine onnxruntime
  --baseline-runtime <staging>/src/model-runtime`).
- **Cores and memory:** node B vCPUs 64–79, on NUMA node 0, in one cgroup
  cpuset, with every process's memory bound to node 0 (`numactl --membind=0`).
  On vCPUs 112–127 (NUMA node 1, with 25 GB free beside 585 GB of page cache)
  30–50 % of a runtime's memory, its transparent huge pages first, landed on
  node 0, and Mini audio took 515 or 685 ms depending on the process while
  every other cell held still. Bound to node 0, six fresh processes took
  493–509 ms. The baselines run under the same binding.
- **The native process** runs NumPy's OpenBLAS on one thread, the package's
  default since #4619: OpenBLAS's idle threads spin after each call and took
  the cores of the CPU device's OpenMP team, slowing Omni Mini's CLAP tower
  after its audio features seven times (91 → 14 ms).

**Against the legacy facade, 1 caller (latency run).**

| Job | Pairs | Legacy p50 / p95 ms | Native p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 5.42 / 8.52 | 4.73 / 6.72 | -0.69 [-0.94, -0.39] better | -1.80 [-3.05, -1.42] better |
| Nano image | 30 | 109.3 / 150.0 | 86.30 / 88.54 | -23.00 [-23.52, -22.43] better | -61.42 [-98.93, -22.21] better |
| Nano audio | 40 | 218.2 / 352.7 | 45.29 / 73.03 | -172.9 [-179.6, -119.7] better | -279.7 [-320.5, -260.1] better |
| Mini text | 260 | 28.21 / 74.15 | 21.31 / 48.36 | -6.90 [-8.03, -4.54] better | -25.79 [-31.54, -22.81] better |
| Mini image | 30 | 290.1 / 373.3 | 237.2 / 248.6 | -52.88 [-54.52, -51.58] better | -124.7 [-136.3, -45.45] better |
| Mini audio | 50 | 518.9 / 731.6 | 447.7 / 479.7 | -71.18 [-120.7, -53.48] better | -251.9 [-261.3, -212.8] better |

**Against the legacy facade, sequential pairs of the 4-caller run.**

| Job | Pairs | Legacy p50 / p95 ms | Native p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 5.09 / 8.40 | 4.68 / 6.82 | -0.41 [-0.63, -0.21] better | -1.58 [-2.03, -1.06] better |
| Nano image | 30 | 109.2 / 156.5 | 88.49 / 89.29 | -20.66 [-21.79, -19.98] better | -67.18 [-72.56, -1.71] better |
| Nano audio | 40 | 162.4 / 352.7 | 44.38 / 72.12 | -118.1 [-167.9, -115.1] better | -280.6 [-283.9, -266.8] better |
| Mini text | 260 | 28.18 / 74.44 | 21.82 / 49.29 | -6.36 [-7.57, -4.55] better | -25.15 [-30.31, -21.69] better |
| Mini image | 30 | 293.9 / 370.2 | 239.6 / 242.0 | -54.26 [-55.77, -51.98] better | -128.2 [-140.4, -55.29] better |
| Mini audio | 50 | 525.2 / 736.9 | 445.6 / 474.0 | -79.64 [-129.5, -54.39] better | -262.9 [-283.7, -220.2] better |

**Against the legacy facade, 4 callers.**

| Job | Rounds | Legacy req/s | Native req/s | Δ req/s [95% CI] |
| --- | --- | --- | --- | --- |
| Nano text | 10 | 233.2 | 257.0 | +23.74 [+16.69, +29.34] better |
| Nano image | 10 | 9.24 | 11.27 | +2.03 [+1.96, +2.10] better |
| Nano audio | 10 | 5.01 | 19.61 | +14.60 [+14.54, +14.68] better |
| Mini text | 10 | 27.62 | 39.81 | +12.19 [+11.75, +12.32] better |
| Mini image | 10 | 3.41 | 4.19 | +0.78 [+0.76, +0.80] better |
| Mini audio | 10 | 1.83 | 2.23 | +0.40 [+0.39, +0.42] better |

**Against the ONNX Runtime path, 1 caller (latency run).**

| Job | Pairs | ONNX Runtime p50 / p95 ms | Native p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 4.90 / 7.54 | 4.72 / 6.62 | -0.18 [-0.40, +0.11] level | -0.93 [-1.33, -0.54] better |
| Nano image | 30 | 108.9 / 111.0 | 87.67 / 89.53 | -21.21 [-21.87, -20.68] better | -21.45 [-22.73, -20.51] better |
| Nano audio | 40 | 122.2 / 178.8 | 45.88 / 71.23 | -76.35 [-84.43, -67.66] better | -107.6 [-116.4, -96.42] better |
| Mini text | 260 | 27.17 / 69.09 | 22.05 / 49.64 | -5.12 [-5.43, -3.54] better | -19.45 [-24.20, -18.45] better |
| Mini image | 30 | 292.9 / 297.8 | 239.3 / 252.0 | -53.66 [-54.89, -52.14] better | -45.84 [-59.58, -24.21] better |
| Mini audio | 50 | 481.2 / 543.5 | 444.8 / 471.7 | -36.39 [-48.89, -32.72] better | -71.82 [-73.66, -45.77] better |

**Against the ONNX Runtime path, sequential pairs of the 4-caller run.**

| Job | Pairs | ONNX Runtime p50 / p95 ms | Native p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 5.07 / 8.41 | 4.75 / 6.99 | -0.32 [-0.53, +0.09] level | -1.43 [-1.81, -1.05] better |
| Nano image | 30 | 114.4 / 148.3 | 88.26 / 93.00 | -26.11 [-38.82, -22.42] better | -55.29 [-59.61, -33.44] better |
| Nano audio | 40 | 120.0 / 181.1 | 49.32 / 69.41 | -70.63 [-90.53, -65.49] better | -111.7 [-115.6, -100.6] better |
| Mini text | 260 | 27.16 / 69.77 | 22.29 / 50.07 | -4.87 [-5.28, -3.43] better | -19.71 [-24.91, -18.40] better |
| Mini image | 30 | 293.7 / 298.0 | 240.0 / 242.4 | -53.67 [-55.42, -52.29] better | -55.61 [-56.94, -54.33] better |
| Mini audio | 50 | 485.7 / 540.9 | 507.0 / 534.4 | +21.28 [-0.09, +32.70] level | -6.51 [-18.20, -2.40] better |

**Against the ONNX Runtime path, 4 callers.**

| Job | Rounds | ONNX Runtime req/s | Native req/s | Δ req/s [95% CI] |
| --- | --- | --- | --- | --- |
| Nano text | 10 | 249.2 | 259.7 | +10.45 [+6.33, +18.02] better |
| Nano image | 10 | 8.97 | 11.32 | +2.35 [+2.31, +2.38] better |
| Nano audio | 10 | 6.67 | 19.88 | +13.21 [+13.07, +13.34] better |
| Mini text | 10 | 27.63 | 40.03 | +12.40 [+12.11, +12.57] better |
| Mini image | 10 | 3.40 | 4.17 | +0.77 [+0.74, +0.78] better |
| Mini audio | 10 | 1.97 | 1.99 | +0.02 [-0.04, +0.08] level |

**Verdicts.** No cell is worse against either baseline. Against legacy every
cell is better, including the one the ONNX Runtime path left open below
(Nano text p95 between load windows: 6.82 against 8.40 ms). Against the ONNX
Runtime path every cell is better or level. One level cell has a worse
point: Mini audio's p50 between load windows (+21.3 ms, its interval reaching
-0.09), where the same native process answered in 445 ms in the latency run.
Its 4-caller rate (+0.02 req/s) and Nano text's two p50s are level with a
better point. The node's 1-minute load was 18–31
during the run (other workstreams' jobs on other vCPUs, untimed profiles on
vCPUs 32–47 of the same NUMA node among them).

## CPU: Omni on the ONNX Runtime bundle, the legacy facade's adapter vs `Runtime.call`

Before #4619 (runtime at `e0e0e2850`, node B vCPUs 112–127).

The corpus's 26 short texts (3–104 tokens), the bundle's 3 golden images
(encoded bytes) and its 4 (Nano) or 5 (Mini) golden audio clips. Ten
interleaved rounds for 1-caller latency, then ten rounds of the throughput
run, whose sequential pairs give a second latency sample taken between load
windows.

**1 caller (latency run).**

| Job | Pairs | Legacy p50 / p95 ms | Runtime p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 4.49 / 7.06 | 4.62 / 7.06 | +0.12 [-0.06, +0.28] level | +0.01 [-0.16, +0.41] level |
| Nano image | 30 | 107.7 / 150.1 | 107.8 / 109.3 | +0.04 [-0.62, +0.64] level | -40.78 [-82.86, +3.06] level |
| Nano audio | 40 | 162.2 / 355.9 | 121.2 / 170.8 | -40.98 [-98.28, -23.51] better | -185.2 [-258.0, -162.5] better |
| Mini text | 260 | 27.43 / 73.98 | 26.09 / 66.74 | -1.35 [-3.41, -0.91] better | -7.24 [-9.01, -3.25] better |
| Mini image | 30 | 287.8 / 294.1 | 285.2 / 295.3 | -2.63 [-4.41, -1.06] better | +1.25 [-8.00, +2.48] level |
| Mini audio | 50 | 535.7 / 704.8 | 479.2 / 532.8 | -56.48 [-88.29, -28.60] better | -172.1 [-205.3, -162.8] better |

**Sequential pairs of the 4-caller run.**

| Job | Pairs | Legacy p50 / p95 ms | Runtime p50 / p95 ms | p50 Δ ms [95% CI] | p95 Δ ms [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Nano text | 260 | 4.50 / 7.00 | 4.55 / 7.45 | +0.05 [-0.09, +0.20] level | +0.45 [+0.21, +0.64] **worse** |
| Nano image | 30 | 108.3 / 114.7 | 109.2 / 141.8 | +0.94 [-0.86, +4.42] level | +27.12 [-43.37, +33.00] level |
| Nano audio | 40 | 165.5 / 336.2 | 120.4 / 175.2 | -45.10 [-96.15, -27.34] better | -161.1 [-185.9, -155.2] better |
| Mini text | 260 | 27.32 / 70.81 | 26.38 / 67.09 | -0.93 [-1.55, +0.29] level | -3.72 [-5.90, -0.19] better |
| Mini image | 30 | 289.7 / 294.1 | 289.4 / 302.4 | -0.24 [-1.61, +1.11] level | +8.32 [-11.28, +10.23] level |
| Mini audio | 50 | 509.5 / 701.9 | 480.7 / 533.4 | -28.76 [-84.46, -16.27] better | -168.5 [-188.6, -155.5] better |

**4 callers.**

| Job | Rounds | Legacy req/s | Runtime req/s | Δ req/s [95% CI] |
| --- | --- | --- | --- | --- |
| Nano text | 10 | 235.5 | 256.9 | +21.35 [+16.56, +25.00] better |
| Nano image | 10 | 9.31 | 9.20 | -0.11 [-0.26, +0.01] level |
| Nano audio | 10 | 5.02 | 6.72 | +1.70 [+1.66, +1.78] better |
| Mini text | 10 | 27.73 | 28.10 | +0.37 [+0.14, +0.59] better |
| Mini image | 10 | 3.47 | 3.47 | -0.00 [-0.03, +0.02] level |
| Mini audio | 10 | 1.85 | 1.99 | +0.14 [+0.08, +0.17] better |

**Verdicts.** Every cell is level or better but one. Cells whose interval
straddles zero with a worse point stay level at the ten-round cap: Nano
text's 1-caller p50 and p95, Nano image's p50s, its p95 between load
windows and its 4-caller rate, and Mini image's p95s.

**Open (P0-1): Nano text p95 in the 4-caller run's sequential pairs, 7.45
against 7.00 ms, +0.45 ms [+0.21, +0.64] at 10 rounds.** The
cause is the five longest texts. Legacy's text graph on 16 threads has a
sweet spot at 64 tokens: 5.4–5.6 ms, faster than its own 45–57-token texts
at 5.9–6.3 ms, and a bare ONNX Runtime run on 16 threads shows the same. The
runtime's 12 threads move the sweet spot to about 48 tokens. Over the 20
rounds of both runs, the runtime wins the 45–57-token texts in 94 of 100
pairs (medians −0.35 to −1.25 ms) and loses the 64–104-token ones (medians
+0.02 to +0.70 ms), and those set the p95.

The cell stays open in this PR, which lists it. No setting measured so far
closes it without a cost elsewhere. The 12-thread pool with ONNX Runtime's
own spin had the fastest runtime side in the decision runs below (p50
4.36 ms and p95 6.87 ms, against 4.43 and 7.06 ms with the 10 ms spin, in
the same run) and the highest rate. But that spin runs about 40 ms after
every call and slows the next graph or model on the same cores (the probe
below), which is why the spin is bounded. Those runs also had only 3
rounds, and the 10 ms spin read level there (p95 −0.38 [−0.84, +0.41])
before its 10-round run read worse. Two Nano text sessions, 16 threads for
texts of 60 tokens or more, collapsed the 4-caller rate. The engine at
`e0e0e2850` also reads every graph's spin setting for its receipt, so an
unbounded pool would need an engine change before a 10-round A/B could
measure it.

**A re-check that was not quiet.** Ten rounds of the Nano text row alone,
same cores, tool and pool, from 00:16 to 00:20 on 2026-10-06. Another
workstream's jobs were running on the same NUMA node then (CPU parity on
vCPUs 132–143, a GPU job's host threads on 128–131), at a 1-minute load of
13–23. In the 4-caller run's sequential pairs, legacy went from 4.50 to
5.09 ms at p50 against the final run, and the runtime from 4.55 to 7.97 ms:

| Nano text, re-check | Legacy | Runtime | Δ [95% CI] |
| --- | --- | --- | --- |
| p50 ms | 5.09 | 7.97 | +2.88 [+2.54, +3.19] **worse** |
| p95 ms | 7.84 | 11.57 | +3.73 [+3.41, +4.12] **worse** |
| 4 callers, req/s | 217.8 | 162.6 | −55.2 [−59.6, −50.7] **worse** |

So with those neighbours the runtime's pool lost far more than legacy's
did. The 16 cores themselves were idle when checked afterwards, and the
cause is not found. It stays open with the cell above.

Both sides run the same graphs, so every difference is in what surrounds
them and in how the graphs' thread pools are sized:

- **Text.** Nano's text graph runs on 12 of the 16 threads with a 10 ms idle
  spin; every other graph runs on all 16. The runtime runs up to four text
  inputs at once (`CONCURRENT_INPUTS`), so 12 threads leave a core to each
  concurrent caller. On 16 threads, 15 workers and four callers oversubscribe
  the cores; on 8, the texts of 64–104 tokens, which set the p95, ran slower
  than legacy's. With the engine's 2 ms spin some of the 12 threads sleep
  inside a 2–8 ms run: bare runs of the graph on 16 threads took 0.6 ms more
  under the 2 ms bound than under ONNX Runtime's own spin, although both
  pools sleep in the 50 ms between runs. The decision runs, against legacy on
  the same cores (3–4 rounds each, so indicative; `0877e5ce7` with the pool
  swapped in):

  | Nano text graph pool | p50 Δ ms | p95 Δ ms | 4 callers, Δ req/s |
  | --- | --- | --- | --- |
  | 8 threads, 2 ms spin (`0877e5ce7`) | −0.12 [−0.74, +0.26] | −0.83 [−1.23, +0.26] | +16.0 [+8.5, +16.3] |
  | 16 threads, ONNX Runtime's spin | +0.48 [+0.14, +0.62] | +0.51 [+0.14, +0.77] | −7.5 [−8.5, −3.6] |
  | 12 threads, ONNX Runtime's spin | −0.25 [−0.66, +0.00] | −0.11 [−0.44, +0.37] | +25.9 [+21.5, +29.3] |
  | 12 threads, 2 ms spin | +0.27 [−0.27, +0.36] | +0.15 [−0.20, +0.49] | +21.3 [+12.4, +27.8] |
  | **12 threads, 10 ms spin** | −0.10 [−0.31, +0.18] | −0.38 [−0.84, +0.41] | +24.4 [+15.4, +28.1] |

  ONNX Runtime's own spin (about 40 ms after a run) is as fast but slows the
  next graph or model on the same cores (see the pool section), so the spin
  stays bounded. The rest of a short text's time is the hand-off from the
  event loop to the model's worker and back: Omni already skips the hand-off
  to the CPU device thread (`LoadedModel.device_thread`, 1.4 ms per request),
  and running a lone short text on the event loop (`inline_cost`, measured at
  `1be83343f`) did not close the row and was reverted.
- **Images.** A batch's images run one after another, beside its text and
  audio inputs. One image's graph keeps all 16 cores busy, and concurrent
  images on one 16-thread pool only contend for them. Against legacy at 4
  callers (3 rounds each, so indicative; `0877e5ce7` with each variant
  swapped in):

  | A batch's images | Nano image Δ req/s | Mini image Δ req/s |
  | --- | --- | --- |
  | Up to four at once (`0877e5ce7`) | −0.14 [−0.18, −0.07] | −0.04 [−0.05, −0.01] |
  | Up to four at once, heap frozen | −0.07 [−0.08, −0.07] | −0.01 [−0.04, +0.02] |
  | Up to four at once, ONNX Runtime's spin | −0.12 [−0.21, −0.08] | −0.02 [−0.15, +0.16] |
  | Up to four at once, 10 ms spin | +0.05 [−0.04, +0.14] | −0.01 [−0.08, −0.01] |
  | Up to four at once, one shared pool | −0.03 [−0.05, +0.00] | −0.03 [−0.10, −0.01] |
  | **One at a time** | +0.00 [−0.05, +0.15] | +0.03 [+0.03, +0.04] |

  The vision graphs take the same time on ONNX Runtime 1.22 and 1.30 (Nano,
  bare runs alternating in one window: level, [−0.77, +0.79] ms), so the
  runtime keeps 1.30. Pillow decodes an image, then its three bands resize
  and normalize on three threads (bit-identical to one resize of the whole
  image, `9075c9959`), through a per-channel table of the processor's
  float64 and float32 steps.
- **Audio** keeps its concurrency: the runtime's NumPy features build each
  resampler kernel once and skip Whisper frames that hold only padding, and
  four inputs at once serve a third more Nano clips per second than legacy.
- **The heap after load.** Loading leaves the frameworks', tokenizers' and
  models' objects behind, and a full garbage collection walked all of them:
  83–131 ms, once in every diagnostic run of two to three minutes, stalling
  the request that triggered it. Every loading pass of the runtime now ends
  with `gc.collect()` and `gc.freeze()` (`freeze_heap`); after it no
  collection took more than 2 ms.

## CPU: ONNX Runtime pools in a process that serves several CPU models

The policy (`engines/onnxruntime/providers.py`):

- **Every CPU session has its own intra-op pool,** sized to the configured
  threads (else the CPUs the process may run on; ONNX Runtime's own default
  counts the host's CPUs, not the cpuset) and capped per graph by the
  family (`ModelSpec.graph_threads`: Omni Nano's text graph 12).
- **Idle threads spin 2 ms before they sleep,** or as long as the family sets
  per graph (`ModelSpec.graph_spin_us`: Omni Nano's text graph 10 ms), and
  at most 1 ms whenever another engine's models serve the process's CPU too
  (`EngineOptions.cpu_neighbors`, which the runtime fills before anything
  loads). The model receipt reports each graph's threads and spin.

Why per-session pools: a process-wide shared pool can't be sized per graph,
and graphs differ (Nano's text graph wants 12 threads, its image graph all
16: 109 ms on 16, 163 ms on 8). Why a bounded spin: ONNX Runtime's threads
spin about 40 ms after a run, on the cores the next graph or model needs.
Omni's audio graph after its CLAP windows took 124 ms instead of 43 with
per-session pools that spin unbounded, and pools that stop spinning the
moment a run returns cost Omni 13–24 % of its 4-caller throughput.

**A native model right after an ONNX Runtime call (the probe).** One process
serves Vela Embedding on `onnxruntime` and Vela Domain on `native`, 16
threads: Domain's p50 alone and when called right after an embedding call, at
16 / 64 / 256 tokens (40 calls each, 50 ms apart).

| Pool | Domain alone ms | Domain right after an embedding ms |
| --- | --- | --- |
| Shared, spinning | 12.5 / 22.5 / 47.6 | 38.0 / 61.5 / 87.4 |
| Own, ONNX Runtime's spin | 12.6 / 21.4 / 46.0 | 26.1 / 57.2 / 83.7 |
| Own, 2 ms spin | 10.7 / 19.8 / 42.2 | 12.5 / 21.7 / 42.0 |
| **Own, 1 ms spin (beside another engine), `e0e0e2850`** | 9.7 / 16.9 / 34.6 | 10.2 / 16.4 / 34.5 |

The first three rows are from `470c751de` and `bb6600d2f` with each policy
swapped in, the last from the final run (medians of three rounds' p50s).
With 1 ms pools the native model keeps its alone speed. That placement (an
ONNX Runtime model and a native one in one CPU process) is opt-in: on the CPU
the router's default is one model per process, and the `process:` key groups
models.

## CPU: legacy ONNX Runtime execution vs both runtime engines

Synthetic token rows (`tools/embed_corpus.py`), three rounds on the same 16
vCPUs (legacy, runtime `onnxruntime`, runtime `native` in turn each round),
p50 / p95 ms, medians of the rounds; throughput in items per second. Runtime
at `7c2c6e21b` (2026-10-04); this section sets the engine choice below, not
the gate against the legacy router path above.

| Vela Embedding | Legacy ORT | Runtime `onnxruntime` | Runtime `native` |
| --- | --- | --- | --- |
| one text, 16 tokens | 11.9 / 12.5 | 12.1 / 13.0 | 7.2 / 7.5 |
| one text, 64 tokens | 20.1 / 21.7 | 21.2 / 21.3 | 14.4 / 14.6 |
| one text, 256 tokens | 53.8 / 55.7 | 55.6 / 56.2 | 33.4 / 34.2 |
| one text, 1,024 tokens | 198.3 / 204.3 | 201.4 / 207.8 | 116.5 / 124.3 |
| 32 texts of 16–256 tokens | 1,137 / 1,162 (28.0/s) | 957 / 966 (33.6/s) | 528 / 541 (60.6/s) |

| Vela Reranker | Legacy ORT | Runtime `onnxruntime` | Runtime `native` |
| --- | --- | --- | --- |
| query + 10 documents | 377.0 / 380.3 (26.5/s) | 313.6 / 317.8 (32.0/s) | 182.2 / 185.0 (55.0/s) |
| query + 50 documents | 1,828 / 1,850 (27.3/s) | 1,345 / 1,399 (36.9/s) | 914 / 961 (54.8/s) |

Qwen3-Embedding has no ONNX graph in the pinned package, so it runs native
only: p50 21.4 / 42.7 / 100.5 / 1,190 ms for one text of 16 / 64 / 256 /
1,024 tokens and 5.6 s for 32 texts (5.7/s); the router-path table above
compares it with legacy candle.

## ROCm: legacy ONNX Runtime + CK flash attention vs the native engine

Engine-level scenarios, six interleaved rounds (the order rotated each round)
on one MI325X (node B GPU2) and the same 8 host vCPUs for both sides (80–87,
docker cgroup cpusets), 100 timed runs per scenario. The runtime side runs in
the router's shipped ROCm image (`a580be6b9`: vLLM's ROCm PyTorch
2.12.0+git6bbd260 with AOTriton 0.13.50, Triton 3.7.0), with the native engine
as it serves (gfx942 fused rotary, encoder graphs, block-local attention from
2,048 tokens). The legacy side is ONNX Runtime's ROCm EP with CK flash
attention in the legacy router image (`61aa7eb2d`). The node's 1-minute load
stayed between 2 and 11. p50 (p95) ms are medians of the rounds; Δ is runtime
− legacy over the paired rounds, with a 95% t interval.

| Vela Embedding | Legacy | Runtime | p50 Δ [95% CI] | p95 Δ [95% CI] |
| --- | --- | --- | --- | --- |
| one text, 16 tokens | 4.01 (4.14) | 1.53 (1.55) | −2.49 [−2.53, −2.45] | −2.59 [−2.64, −2.54] |
| one text, 64 tokens | 3.95 (4.12) | 1.74 (1.75) | −2.21 [−2.26, −2.17] | −2.35 [−2.44, −2.27] |
| one text, 256 tokens | 4.55 (4.64) | 2.63 (2.64) | −1.92 [−1.97, −1.87] | −2.00 [−2.04, −1.95] |
| one text, 1,024 tokens | 6.94 (7.00) | 6.46 (6.63) | −0.47 [−0.54, −0.41] | −0.39 [−0.53, −0.25] |
| 32 texts of 16–256 tokens | 136.7 (138.4) | 26.0 (26.9) | −110.7 [−111.6, −109.7] | −111.0 [−112.3, −109.8] |

| Vela Reranker | Legacy | Runtime | p50 Δ [95% CI] | p95 Δ [95% CI] |
| --- | --- | --- | --- | --- |
| query + 10 documents | 41.8 (42.3) | 9.13 (9.45) | −32.6 [−32.9, −32.3] | −32.7 [−33.2, −32.3] |
| query + 50 documents | 209.5 (213.8) | 34.0 (34.4) | −176.0 [−177.6, −174.4] | −175.1 [−185.9, −164.2] |

Every interval lies wholly on the runtime's side. Throughput, items per second
(median of the rounds): 32 texts 234 → 1,226; query + 10 documents 239 →
1,090; query + 50 documents 238 → 1,471.

Short rows replay a captured HIP graph per length bucket; multi-row batches
replay one only when the bucket pads little (else they run packed); a rerank
request is one packed forward of all its pairs. The 1,024-token row was the
last one behind: the fused rotary kernel (`rotary_half`, bit-exact) took it
from 7.5 to 6.5 ms, and block-local attention, which costs more than dense
masked attention below 2,048 tokens on this GPU, now starts there.
Qwen3-Embedding has no legacy GPU path (candle has no ROCm); on the MI325X it
takes 14–16 ms for one text of up to 256 tokens (launch-bound: its decoder
path runs without graphs), 26.5 ms at 1,024 tokens and 104 ms for 32 texts
(2026-10-04, `5a73fc17e`). The ROCm golden answers of all three models hold
in this image in every value (parity record).

## ONNX Runtime vs PyTorch on the CPU, per model

- **Vela Embedding:** PyTorch. The native engine is 1.4–1.7× faster than
  ONNX Runtime (the runtime's engine or the legacy execution) on single texts
  and 1.8× faster than the runtime's ONNX Runtime engine on 32 texts. Before
  the `exact` path took oneDNN's pre-packed FP32 linears (`5a73fc17e`), ONNX
  Runtime was 7–30 % faster than native on single texts.
- **Vela Reranker:** PyTorch: 1.7× faster than the ONNX Runtime engine with
  10 documents and 1.5× with 50. Native also serves all twenty pair scorers
  (four exits × five dimensions) from one forward; the graph engine runs one
  graph per scorer.
- **Qwen3-Embedding:** PyTorch only (the pinned package ships no ONNX graph).
- **Omni:** PyTorch since #4619 (the native section above); a prepared bundle
  still runs on `engine: onnxruntime`.

Decision: `auto` keeps native first for the three `task_heads` models, and no
`BuiltinModel.engines` entry sets a CPU preference. `engine: onnxruntime`
stays available for a package that ships graphs: it matches the legacy ONNX
Runtime execution on single texts (within 5 %) and beats it on batches, but
it serves only the exits it has graphs for. Torch's OpenMP threads spin after
a native forward and used to slow an ONNX Runtime run that followed in the
same process (16 tokens: 13.0 → 19.8 ms). `GOMP_SPINCOUNT=10000`, set before
torch loads, removes that and leaves native unchanged. `vllm-srun serve` now
sets it only for a process that serves an ONNX Runtime model on the CPU
(design §12). Other processes keep libgomp's default, on which native CPU
forwards run faster (`decision1-performance.md`).

## CPU: ONNX Runtime beside native models, per-process spin count

For #4611, at `8b92b620f`: `vllm-srun serve` serves Vela Embedding on
`engine: onnxruntime` (the package's graphs) and Vela Domain on the native
engine, over Unix sockets, with the result cache off. Node D, 16 cores with
memory bound to NUMA node 0, and 10 interleaved rounds of fresh processes.
"After" lets `serve` choose each process's spin count; "before" sets 10,000
in every process (the old import default). Per round, length and pattern
there are 30 samples. Each model is timed alone (50 ms after the last call)
and right after the other model's call.

- **One process for both models** (cores 16–31, 16 threads): `serve` picks
  10,000 there too, so both conditions run the same spin.
  - 23 of the 24 cells (p50 and p95, four patterns, 16 / 64 / 256 tokens) are
    level.
  - One reads worse: ONNX Runtime right after a native call at 256 tokens,
    p50 +0.51 ms [+0.01, +1.01] on 60.2 ms. With the same spin on both sides
    it is the one false positive the 24 intervals lead one to expect.
- **Two processes on the same 16 cores** (48–63, 8 threads each, as the
  router splits a CPU budget): the native process now runs libgomp's default.
  - 22 of 24 cells are level.
  - Two are worse, both ONNX Runtime right after a native forward in the
    other process, at 16 tokens: p50 +0.40 ms [+0.02, +0.77] on 15.2 ms
    (+2.6%) and p95 +0.78 ms [+0.16, +1.40] on 16.6 ms. The native process's
    threads now spin longer on the cores the next ONNX Runtime run takes.
  - At 64 and 256 tokens these patterns are level: p50 +0.48 [−0.37, +1.33]
    and −0.14 [−1.17, +0.90].
- **Where that applies:** no built-in model runs on ONNX Runtime by default
  since #4619, and the router's images ship none. A deployment that runs an
  ONNX Runtime model in its own process beside native CPU processes on shared
  cores keeps the old timing by serving the models in one process, or by
  setting `GOMP_SPINCOUNT=10000`, which `serve` keeps.
