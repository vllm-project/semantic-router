# ROCm golden answers and cross-process repeatability (MI325X)

> **Phase 1 record** (Decision 2.0, [#4481](https://github.com/vllm-project/semantic-router/pull/4481) and its follow-ups). The Phases 2–4 records are the `decision1-*`, `vela1-*`, `vela2-*`, `embed-*`, `stores-*`, `router-latency-cpu*`, `router-latency-rocm*`, `rocm-router-image` and `removal-footprint*` files.

Readiness on ROCm now checks reference answers for all six Decision 2.0
models, and every process on gfx942 answers the Qwen3.5 sizes bit for bit
like the released runtime. A shared autotune cache alone does not guarantee
that, so the runtime pins the released runtime's kernel choices.

- **Date:** 2026-10-03.
- **Device:** AMD Instinct MI325X (gfx942), GPU0 and GPU1 of one node, one
  process per container.
- **Image:** the packages' release image: PyTorch 2.12.0 (ROCm), Triton 3.7.1,
  FLA 0.5.2, causal-conv1d 1.7.0.
- **Packages:** the current Hub revisions, read from the Hugging Face cache:
  Kai `cd49ea38`, Eos `3594047d`, Sol `64235bef`, Nox `25e8f67d`, Lux `78bf3c03`
  and Vega `7aec49ae`.
- **Commits:** the cross-process runs at `cfecb2941` (no pinning) and
  `678ae6fe5` (pinning); golden answers at `678ae6fe5`.

## The current Hub revisions are the built-in models

The six current `main`s are runtime-only revisions: only the packages' bundled
Python changed, which this runtime never imports.

- Each one's model identity equals the built-in table's `model_sha256`; only
  the manifest digest differs.
- The runtime serves them with `--revision <40-hex>`. It verifies them against
  their own manifests, and the golden references and kernel choices apply by
  identity.
- On one GPU with one autotune cache, the pinned revisions and the current
  `main`s give bit-identical golden answers (Kai, Eos, Sol, Nox and Lux
  compared).

The table therefore keeps its pins. Moving the default revision to the current
`main`s is a separate change (docs, examples and tests pin the old revisions).

## Cross-process repeatability

Each run is a separate process. It loads the model as `vllm-srun serve`
does (verification, the golden check, `--autotune-cache`) and answers the four
scored panels (10,653 prompts, one request each, `exact`). It is then compared
with a reference run prompt by prompt (`tools/cross_process.py`). Identical
means every answer's JSON is byte-identical.

### Eos-0.8B

| Run | Autotune cache (entries before → after) | Against the reference |
| --- | --- | --- |
| Reference: cold, GPU0 | shared, empty → 11 | — |
| Warm, GPU0 | shared, 11 → 11 | 10,653 / 10,653 identical |
| Warm, GPU1 | shared, 11 → 11 | 10,653 / 10,653 identical |
| Cold, GPU1, run beside the reference | its own, empty → 11 | 10,653 / 10,653 identical |
| No `--autotune-cache` | container-local | 10,653 / 10,653 identical |
| **Two cold processes started together, GPU0 and GPU1** | **one shared, empty → 11** | **GPU0: identical. GPU1: 77 / 10,653 identical, 47 decisions changed, max \|Δp\| 0.056, median 0.0002** |
| The release's cache (a copy) | release, 11 → 11 | 10,653 / 10,653 identical |
| Pinned choices, cold | empty → 0 | 10,653 / 10,653 identical to the release's cache |
| **Pinned, two cold processes started together** | **one shared, empty → 0** | **both 10,653 / 10,653 identical to the release's cache** |

### Sol-2B

| Run | Autotune cache (entries before → after) | Against the reference |
| --- | --- | --- |
| Reference: the release's cache (a copy), GPU1 | release, 11 → 11 | — |
| **Two cold processes started together, GPU0 and GPU1** | **one shared, empty → 11** | **GPU0: 75 / 10,653 identical, 35 decisions changed, max \|Δp\| 0.045, median 0.0002. GPU1: 10,652 / 10,653 identical (one css15 prompt, max \|Δp\| 0.0005)** |
| Warm on that cache afterwards, GPU0 | shared, 11 → 11 | 10,652 / 10,653, the same as the GPU1 process (identical to it on 10,653 / 10,653) |
| **Pinned, two cold processes started together** | **one shared, empty → 0** | **both 10,653 / 10,653 identical** |

### What this shows

- **A warm shared cache repeats exactly.** Every later process reads every
  choice from the cache, on either GPU. It repeats the process that wrote the
  cache, not the release: Sol's shared cache differs from the release's on one
  long prompt.
- **A cold cache does not.** FLA times each kernel's configurations at the
  first call of each tuning key. Two processes that tune at the same time can
  choose differently, and one of them then answers almost every prompt
  differently by rounding. The drift exceeds the golden tolerance (0.02), and
  decisions flip. A fleet that starts several replicas at once on one empty
  volume, or without a shared volume, is exposed to it.
- **The release's choices are their own.** The release caches differ from
  fresh processes and from each other, even for the same tuning keys: Eos and
  Sol (16 heads), and Nox and Lux (32 heads), chose differently for
  `chunk_fwd_kernel_o`, `fwd_h`, `kkt_solve` or `recompute_w_u`. Many of these
  differences are neutral, but which ones matter is a property of the kernels'
  layouts, not something a cache can promise.

## Pinned kernel choices

`registry/kernel_choices.json` holds, per Qwen3.5 model and device class
(`rocm:gfx942`), the configuration of each autotuned FLA forward kernel for
each tuning key its released runtime saw:

- `chunk_fwd_kernel_o`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, `recompute_w_u_fwd_kernel`
  (keyed by head counts and widths: one entry per model);
- `chunk_local_cumsum_scalar_kernel` (keyed by the batch's rows) and
  `l2norm_fwd_kernel` (keyed by the number of 65,536-row blocks).

They come from each model's release autotune cache
(`tools/kernel_choices.py`): the cache its release ran with and that every
later runtime-only revision's parity runs copied forward. The choices are
identical across the release, hotfix and no-eviction revisions.

The runtime resolves them as FLA's `FLA_CACHE_MODE=full` resolves the same
entries written as config files. A recorded key matches exactly. Any other
key, for example a batch with more rows than the release saw, takes the first
entry, in key-hash order, that differs from it only in numbers, else the
kernel's first entry. No configuration depends on timing, so a pinned process
tunes nothing (0 autotune entries) and loads faster (Eos 22 s instead of 38 s
cold). Choices apply only when the installed FLA is the recorded version
(0.5.2); otherwise FLA keeps its per-process tuning and the model's golden
result is `unverified`. Kai (dense) has no autotuned kernels. Since review
P1-4 the choices are per model, not per process: each model's device work
runs in its own choice scope, so models sharing a GPU process keep their own
(the measurements behind that are in `decision1-parity.md`).

`--autotune-cache` still persists Triton's compiled kernels and the tuning of
any model without pinned choices.

## Golden answers on ROCm

`registry/golden_answers.json` now carries a `rocm` reference for every
built-in model, recorded with `tools/golden_answers.py` with the pinned kernel
choices, at `678ae6fe5`.

| Model | Current `main` (GPU0) | Pinned revision (GPU1) | Readiness |
| --- | --- | --- | --- |
| Kai-0.6B | `cd49ea38` | `881bee41` | matched 3 / 3 |
| Eos-0.8B | `3594047d` | `ad0aa724` | matched 3 / 3 |
| Sol-2B | `64235bef` | `4b75b521` | matched 3 / 3 |
| Nox-4B | `25e8f67d` | `ce1bdc9d` | matched 3 / 3 |
| Lux-9B | `78bf3c03` | `214ffa43` | matched 3 / 3 |
| Vega-27B | `7aec49ae` | `9b067a95` (with its pinned base) | matched 3 / 3 |

- Each process started with an empty autotune cache. The two revisions answer
  bit-identically on the two GPUs.
- Readiness was checked again at `4b8b4dc01` against the committed references,
  in twelve fresh processes with the GPUs swapped: every one matched 3 / 3, bit
  for bit.
- The first recording, with a fresh process's tuning, differed by up to 0.0021
  (Sol, Nox and Vega). That is within the 0.02 tolerance. The references are
  now the pinned, released numerics, so a pinned process matches them bit for
  bit.
- ROCm references differ from the CPU ones by up to about 0.01 (BF16 autocast
  on GPUs against FP32 on CPU), hence one reference per device class.
- Other ROCm architectures have no pinned choices. They tune per process, use
  the fused-kernel-free path, and are checked against the same references at
  the 0.02 tolerance; this record does not cover them.

## Graph-size cap under Index-like traffic

Graphs are captured only up to `MAX_GRAPH_TOKENS` padded tokens (4,096), and
at most 512 of them. Index-like traffic forms many shapes that rarely repeat,
so the question is what the cap does once the cache is full
(`tools/graph_cap_bench.py`; Eos at `4b8b4dc01`, Kai and the repeat at
`2fce4fec1`, the same bench).

- **Traffic:** 8,943 requests in stream order, on the exact profile, each
  asking 1–8 questions (mean 2.6) about one css15 or hs1-dev state, which
  runs from a few dozen tokens to the input limit. That is 22,875 questions.
- **Runs:** Eos-0.8B and Kai-0.6B, one process per cap, the caps split
  between GPU0 and GPU1.

Eos-0.8B:

| Cap (padded tokens) | Requests/s | p50 / p95 / p99 (ms) | Graphs captured | Replays | Eager | Refused (cache full) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (no graphs) | 49.7 | 15.9 / 33.3 / 75.6 | 0 | 0 | 8,939 | 0 |
| 1,024 | 63.0 | 7.7 / 49.8 / 76.1 | 345 | 5,540 | 3,399 | 0 |
| 2,048 | 60.9 | 7.8 / 55.4 / 77.6 | 512, full by request 8,000 | 6,565 | 2,374 | 230 |
| **4,096 (default)** | **61.4** | **7.7 / 53.7 / 80.8** | **512, full by request 7,000** | **6,626** | **2,313** | **663** |
| 8,192 | 60.3 | 7.7 / 55.2 / 86.3 | 512, full by request 7,000 | 6,639 | 2,300 | 800 |
| 16,384 | 61.0 | 7.7 / 53.7 / 84.6 | 512, full by request 7,000 | 6,636 | 2,303 | 842 |

Kai-0.6B:

| Cap (padded tokens) | Requests/s | p50 / p95 / p99 (ms) | Graphs captured | Replays | Eager | Refused (cache full) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (no graphs) | 63.8 | 10.4 / 34.6 / 89.9 | 0 | 0 | 8,928 | 0 |
| 1,024 | 75.1 | 6.2 / 36.8 / 89.3 | 347 | 5,640 | 3,288 | 0 |
| **4,096 (default)** | **71.8** | **6.2 / 43.0 / 91.6** | **512, full by request 7,000** | **6,758** | **2,170** | **554** |
| 16,384 | 71.8 | 6.2 / 42.9 / 96.2 | 512, full by request 7,000 | 6,761 | 2,167 | 737 |

A paired repeat at `2fce4fec1` ran 1,024 and 4,096 at the same time on the two
GPUs, swapped against the first round:

| Model | Cap 1,024: requests/s, p95 / p99 (ms) | Cap 4,096: requests/s, p95 / p99 (ms) | 1,024 against 4,096 (first round) |
| --- | --- | --- | --- |
| Kai-0.6B | 74.3, 37.4 / 89.4 | 72.1, 43.1 / 91.0 | +3.1% (+4.6%) |
| Eos-0.8B | 62.1, 51.2 / 76.5 | 61.6, 53.4 / 81.1 | +0.8% (+2.6%) |

- **Answers:** every cap answers bit-identically (one digest over every
  answer). Replays include each capture's own forward, so replays plus eager
  forwards is the number of forwards.
- **Graphs pay** 13–27% on this traffic, mostly through the many small
  requests (p50 halves on Eos).
- **A full cache is harmless.** Caps from 2,048 up fill the 512 graphs by
  about request 7,000 and then refuse captures. Throughput does not drop
  afterwards, because the frequent shapes were captured first.
- **A smaller cap is slightly better here.** At 1,024 the cache never fills,
  and the forwards that would capture large, rarely repeated shapes run
  eagerly instead. That is 1–5% more requests/s and a lower p95 on both
  models, in both rounds. Above 4,096 nothing changes but the tail.
- **Decision:** the default stays at 4,096. The gain is small and specific to
  traffic whose large shapes do not repeat. Where they do repeat, a graph
  between 1,024 and 4,096 tokens saves about 5–6% per replay (performance
  record: Kai 5% at 2,700 tokens, Eos 6% at 2,800), which a 1,024 cap would
  give up. The cap is a constant (`MAX_GRAPH_TOKENS`); making it a serving
  option is the way to tune it per deployment.

## Batch-shape buckets under concurrency

`batching` merges concurrent requests' questions into padded batches whose
shapes (rows, padded length) depend on the traffic, so a graph is replayed
only when a later batch has the same shape. Buckets pad each batch up to a
coarser shape, rows to a power of two and length to a multiple of 64, so
shapes repeat more often; the padding is extra compute
(`tools/bucket_bench.py`, at `4b8b4dc01` and `2fce4fec1`).

- **Traffic:** 4,106 single-question prompts (typed-final, mlx-diag,
  public231) that never repeat. The first 512 warm up and the remaining 3,594
  are timed, in waves of C concurrent requests through the scheduler.
- **Runs:** Eos-0.8B, one process per run. "Graphable only" buckets a batch
  only when the bucketed shape stays within the 4,096-token graph cap.

| C | `batching` (req/s) | Buckets | Buckets, graphable only | Forwards replayed: `batching` → buckets |
| --- | --- | --- | --- | --- |
| 4 | 260.4 | 277.3 (+6.5%) | 270.0 (+3.7%) | 847 / 899 → 871 / 899 |
| 16 | 378.3 | 360.9 (−4.6%) | 379.0 (±0) | 1 / 225 → 4 / 225 |
| 64 | 420.1 | 407.0 (−3.1%) | — | 0 / 60 → 0 / 60 |

A forward that captures a graph runs on it, so it counts as replayed.

The exact profile does 154.6 requests/s at C = 16 (one request per forward).

- **Accuracy:** every variant changes as many decisions against the exact
  path as `batching` does, 6–11 of 3,594 requests, with median |Δp| 4·10⁻⁵.
  That is the batching profile's known rounding noise; buckets add none.
- **Where buckets help:** at C = 4, batches are small enough to be graphed, and
  buckets turn 25 captures and 52 eager forwards into 6 and 28.
- **Where they cost:** from C = 16, almost every batch is above the graph cap.
  It runs eagerly either way, so bucketing only adds padding, unless it is
  limited to graphable batches.
- **Decision:** no change. The gain is at most a few percent, at low
  concurrency only. `batching` at low concurrency already replays most
  forwards, because its row count is the concurrency.

## Reproduce

```bash
python3 tools/golden_answers.py vllm-sr/Decision-2.0-Eos-0.8B --revision REV --device rocm:0 \
  --autotune-cache DIR
python3 tools/kernel_choices.py vllm-sr/Decision-2.0-Eos-0.8B --autotune-cache RELEASE_CACHE \
  --device-class rocm:gfx942 --fla 0.5.2 --triton 3.7.1
python3 tools/cross_process.py answer vllm-sr/Decision-2.0-Eos-0.8B --revision REV --device rocm:0 \
  --autotune-cache DIR --panel NAME:PROMPTS.jsonl:COUNT ... --answers A.jsonl --receipt A.json
python3 tools/cross_process.py compare REFERENCE.jsonl A.jsonl B.jsonl --output compare.json
python3 tools/graph_cap_bench.py --package PACKAGE_DIR --cap 4096 --output cap.json \
  --stream css15:CSS15.prompts.jsonl:6547 --stream hs1-dev:HS1_DEV.prompts.jsonl:2396
python3 tools/bucket_bench.py --package PACKAGE_DIR --mode buckets --concurrency 4 [--graphable-only] \
  --prompts typed-final:TYPED_FINAL.prompts.jsonl:1600 --prompts mlx-diag:MLX_DIAG.prompts.jsonl:2275 \
  --prompts public231:PUBLIC231.prompts.jsonl:231 --output buckets.json --answers buckets.jsonl
```
