# Vela 1.0 text models: performance against the legacy router path

On the same inputs and hardware, the runtime answers every Vela 1.0 text
model faster than the router path it replaces, per request (p50 and p95) and
under load, on CPU and on ROCm. Answers are the ones `vela1-parity.md`
records; raw results are in `vela1-performance.json`.

- **Date:** 2026-10-05. **Runtime commits**, unless a row names another:
  - CPU: `efb5ec4d7`, the staging head `b604edeab` with `vela1`'s A/B tool
    commit merged. On it, `exact` lets rows of every length share a batch on
    models that pack rows.
  - ROCm: `35ff3a8a5`, which adds the fixes for a shared GPU process and
    leaves the CPU code as it was, in the router's ROCm image stack (Python
    3.12.14, PyTorch 2.12.0+rocm7.2, HIP 7.2.53211, Triton 3.7.0, fla-core
    0.5.2), as its user (uid 65532).
  - The router's ROCm image as built at `be7366c49` has these pins plus
    `causal-conv1d`, which these models don't run, and Python 3.12.15. It
    answers byte for byte alike (`vela1-parity.md`), so the ROCm rows hold for
    it.
  - The interleaved ROCm A/B table is re-timed in the router's shipped ROCm
    image (`a580be6b9`: vLLM's ROCm PyTorch 2.12.0+git6bbd260 with AOTriton
    0.13.50); the other ROCm rows are stack B's.
  - The final head is `35ff3a8a5` with these records on top, so every
    row holds for it.
- **Legacy side:** the router's native facade at `61aa7eb2d`
  (`tools/legacy_parity.py`). On CPU that is the CPU recipe (candle). On
  ROCm it is the AMD recipe (`config/recipes/vela-amd`: ONNX Runtime
  MIGraphX / ROCm EP).
- **Hardware:**
  - CPU: AMD EPYC 9575F, node B vCPUs 96–111 for both sides, in one cgroup
    cpuset (its effective cpuset is logged), with the node's 1-minute load
    logged every 30 s;
  - ROCm: one AMD Instinct MI325X with 8 pinned host vCPUs.
- **Runtime side:** `Runtime.call` as the HTTP server calls it: one event
  loop, the encoded body size passed so small requests plan inline, result
  cache off, `exact` unless a row says otherwise.

## CPU, interleaved with the legacy facade

A shared node's load moves faster than a run lasts, so `tools/legacy_parity.py
ab` runs both sides on the same 16 vCPUs at once, in one cgroup cpuset, with
threads capped at 16: the systemd scope `vela1-ab-efb5ec4d7-021909`
(`systemd-run --scope -p AllowedCPUs=96-111`), whose effective cpuset read
96-111.

- **Per input:** the facade's test binary, in serve mode, answers one
  input, then the runtime answers the same input. The order flips each
  round, over 5 rounds of all 547 / 64 inputs.
- **Under load:** 20 s windows of 4 closed-loop callers then rotate over the
  legacy facade, the runtime's `exact` profile and its `batching` profile
  (5 rounds, the order shifting by one each round; medians).

`exact` batches the requests queued together, because the CPU forward is
batch-invariant (next section), so its answers stay those of a request
alone. The forward packs rows without padding, so rows of every length share
those batches. The node's 1-minute load stayed at or below 84 (median 46) on
160 vCPUs during the run.

The intervals are 95%:

- **Latency:** each ratio is legacy / runtime, so above 1 means the runtime
  is faster. The p50 and p95 ratios and the median per-input speedup
  bootstrap the input pairs (2,000 resamples).
- **Throughput:** each rate is runtime / legacy calls per second, so above 1
  means the runtime serves more. It is a geometric mean with a t-interval
  over the 5 rounds, each round pairing the sides' windows.

A row passes when its whole interval is at or above 1.

| Job | Pairs | p50 legacy → runtime (ms) [legacy / runtime, 95% CI] | p95 legacy → runtime (ms) [legacy / runtime, 95% CI] | Median per-input speedup (95% CI) | 4 callers, calls/s: legacy → `exact` | `exact` / legacy rate (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 2,735 | 37.86 → 12.50 [3.00, 3.08] | 215.42 → 83.61 [2.31, 2.99] | 3.08× [3.05, 3.11] | 40.6 → 78.8 | 1.91× [1.80, 2.02] |
| guard | 2,725 | 37.88 → 12.57 [2.97, 3.05] | 209.66 → 33.27 [5.96, 6.55] | 3.14× [3.11, 3.17] | 18.8 → 48.8 | 2.58× [2.43, 2.74] |
| safety | 2,735 | 37.52 → 12.21 [3.04, 3.10] | 214.23 → 34.29 [6.02, 6.77] | 3.15× [3.11, 3.18] | 41.3 → 79.1 | 1.89× [1.76, 2.03] |
| shield | 2,735 | 38.21 → 12.53 [3.02, 3.08] | 220.52 → 35.18 [5.96, 6.87] | 3.16× [3.13, 3.20] | 41.2 → 75.7 | 1.86× [1.77, 1.96] |
| factcheck | 2,735 | 37.62 → 12.54 [2.97, 3.04] | 215.13 → 34.00 [5.93, 6.91] | 3.10× [3.06, 3.12] | 40.9 → 77.5 | 1.90× [1.82, 1.99] |
| feedback | 2,735 | 37.72 → 12.49 [2.99, 3.05] | 212.20 → 34.76 [5.93, 6.76] | 3.11× [3.08, 3.15] | 40.6 → 74.5 | 1.88× [1.77, 2.01] |
| modality | 2,735 | 40.16 → 13.07 [3.03, 3.12] | 229.33 → 41.58 [5.22, 5.92] | 3.17× [3.14, 3.21] | 40.6 → 73.7 | 1.83× [1.72, 1.93] |
| hazard | 2,735 | 36.97 → 12.39 [2.95, 3.02] | 215.54 → 34.76 [5.88, 6.77] | 3.10× [3.07, 3.14] | 14.3 → 32.0 | 2.26× [2.06, 2.50] |
| pii | 2,735 | 37.37 → 12.50 [2.95, 3.03] | 211.39 → 34.45 [5.80, 6.87] | 3.10× [3.06, 3.12] | 18.7 → 36.6 | 1.90× [1.62, 2.22] |
| pii_truncate | 2,735 | 93.28 → 13.60 [6.78, 6.99] | 284.17 → 57.07 [4.22, 5.76] | 6.94× [6.92, 6.97] | 27.1 → 79.8 | 2.93× [2.84, 3.01] |
| halu | 320 | 1414.38 → 148.30 [8.91, 10.06] | 41,279.77 → 3,149.38 [8.28, 13.95] | 9.54× [9.28, 9.77] | 1.6 → 3.2 | 2.06× [1.85, 2.29] |

Under load, `exact` serves 1.8–2.9× legacy's calls per second.

- **Windowed jobs (Guard, Hazard, PII) and Halu:** they scan long prompts
  in full. The scheduler (design section 9) answers each job as soon as its
  own batches have run and runs short work first, so a short call no longer
  waits behind a long windowed one (Guard 2.6×, Hazard 2.2×, PII 2.0×, Halu
  2.0× legacy).
- **Short prompts:** the scheduler plans the jobs it takes between two
  forwards on their own. With 4 closed-loop callers that is usually one new
  request, so concurrent short calls rarely share a forward (74–80
  calls/s). `batching` waits up to 2 ms for company and serves
  108–114 calls/s on the same prompts. At `93a3492c0`, before
  this scheduler, `exact` merged everything queued (102–110/s on
  them) but left PII and Hazard at 1.1× and 1.5× legacy.

`batching` is opt-in; the router deploys `exact`. It waits up to 2 ms for
concurrent requests and fills each forward up to 65,536 tokens, where
`exact` caps a shared batch at 512. That pays on short prompts. On windowed
jobs, a short call then shares its forward with a long scan's windows:

| Job | 4 callers, calls/s: legacy → `batching` | `batching` / legacy rate (95% CI) | `batching` / `exact` |
| --- | --- | --- | --- |
| domain | 40.6 → 109.7 | 2.73× [2.62, 2.84] | 1.39× |
| guard | 18.8 → 53.8 | 2.79× [2.57, 3.03] | 1.10× |
| safety | 41.3 → 113.7 | 2.70× [2.58, 2.83] | 1.44× |
| shield | 41.2 → 112.0 | 2.69× [2.55, 2.85] | 1.48× |
| factcheck | 40.9 → 111.2 | 2.66× [2.49, 2.85] | 1.44× |
| feedback | 40.6 → 108.3 | 2.69× [2.54, 2.85] | 1.45× |
| modality | 40.6 → 108.2 | 2.65× [2.40, 2.91] | 1.47× |
| hazard | 14.3 → 18.6 | 1.36× [1.21, 1.54] | 0.58× |
| pii | 18.7 → 19.9 | 1.09× [0.99, 1.20] | 0.54× |
| pii_truncate | 27.1 → 110.3 | 4.00× [3.65, 4.39] | 1.38× |
| halu | 1.6 → 2.6 | 1.64× [1.56, 1.73] | 0.81× |

Every other `batching` row's interval lies above legacy; PII's at 1.09× [0.99,
1.20] (`exact`, the default: 1.9×) is level with it, its interval reaching 1.

## ROCm

One MI325X, the same inputs. The AMD recipe compiles one fixed
8,192-token ONNX Runtime session per model, so every legacy request is an
8K forward (Hazard's operating point compiles 2,048-token windows).

The two sides take turns on the same GPU (node B GPU1) and the
same 8 host vCPUs, 72–79, over 5 rounds, with the
order flipping each round. Each process runs in its container's own cgroup
cpuset (`docker run --cpuset-cpus`), and a container started this way on
node B reads `cpuset.cpus.effective` 72–79.
The runtime side ran in the router's shipped ROCm image (`a580be6b9`: vLLM's
ROCm PyTorch 2.12.0+git6bbd260 with AOTriton 0.13.50, Python 3.12.15), with its
own runtime package, as uid 65532; the parity runs above are stack B's.
The node's 1-minute load stayed at or below 25 (median 6) on 160 vCPUs during
the rounds.

- In each round, a fresh legacy process and a fresh runtime process
  answer every AMD-recipe input once, so the runtime's numbers include
  its graph captures.
- Each then serves one 20 s window of 4 callers per model.
- The two documents over 8,192 tokens, which both sides reject, are left
  out.

The intervals are computed as on CPU.

| Job | Pairs | p50 legacy → runtime (ms) [legacy / runtime, 95% CI] | p95 legacy → runtime (ms) [legacy / runtime, 95% CI] | Median per-input speedup (95% CI) | 4 callers, calls/s: legacy → `exact` | `exact` / legacy rate (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| domain | 2,725 | 153.27 → 1.89 [81.03, 81.34] | 154.87 → 3.28 [31.05, 49.07] | 81.15× [80.97, 81.30] | 6.5 → 436.7 | 67.73× [66.80, 68.68] |
| guard (ROCm EP) | 2,725 | 243.05 → 1.87 [129.46, 129.97] | 253.82 → 3.29 [52.28, 80.42] | 130.15× [129.91, 130.39] | 4.1 → 430.8 | 105.66× [103.11, 108.28] |
| safety | 2,725 | 158.99 → 1.81 [87.53, 87.88] | 166.02 → 3.19 [33.79, 53.69] | 87.74× [87.58, 87.87] | 6.2 → 448.1 | 56.78× [28.42, 113.42] |
| factcheck | 2,725 | 160.88 → 1.83 [87.87, 88.21] | 167.06 → 8.25 [13.14, 31.28] | 87.82× [87.65, 88.01] | 6.2 → 443.0 | 81.66× [49.08, 135.87] |
| feedback | 2,725 | 160.00 → 1.89 [84.28, 84.60] | 165.91 → 3.34 [33.60, 51.54] | 83.95× [83.80, 84.12] | 6.2 → 432.2 | 70.11× [69.23, 71.00] |
| modality | 2,725 | 159.12 → 1.83 [86.60, 86.93] | 168.96 → 3.27 [34.23, 53.87] | 86.95× [86.83, 87.07] | 6.2 → 445.6 | 70.54× [66.09, 75.29] |
| hazard | 2,735 | 13.28 → 1.85 [7.16, 7.20] | 13.55 → 4.77 [2.69, 4.27] | 7.17× [7.15, 7.18] | 70.6 → 350.4 | 4.98× [4.93, 5.03] |
| pii | 2,725 | 132.97 → 1.95 [68.14, 68.43] | 133.58 → 3.53 [27.20, 39.76] | 68.14× [68.01, 68.28] | 7.5 → 417.5 | 55.66× [55.36, 55.95] |

Shield is outside the AMD recipe. Its row comes from the parity run alone,
and is timing only: its package's ONNX graph answers differently from its
checkpoint (`vela1-parity.md`).

| Job | p50 legacy → runtime (ms) | p95 legacy → runtime (ms) | 4 callers, calls/s |
| --- | --- | --- | --- |
| shield (legacy ORT MIGraphX) | 151.22 → 1.80 | 157.22 → 3.44 | 6.5 → 431.9 |

Some deployments move from the router's CPU defaults to a GPU. On the same
inputs, against legacy candle on CPU:

- Halu: p50 1,424 → 7.7 ms and p95 38.7 s → 56 ms;
- sequence models: p50 33–40 → 1.8–1.9 ms, and 4-caller throughput
  35–41 → 431–478 calls/s.

Profiles on ROCm (AMD-recipe inputs, 4 callers):

| Profile | Domain p50 / p95 (ms) | Calls/s, sequence models | Values |
| --- | --- | --- | --- |
| `exact` (`35ff3a8a5`, the router image's stack) | 1.88 / 3.51 | 413–439 | the parity record's |
| `batching` (`35ff3a8a5`, the router image's stack) | 3.98 / 5.56 | 601–612 | identical to `exact` one request at a time |
| `max_speed`, BF16 copy (`973842d9e`, release image, records only) | 6.69 / 11.00 | 437–468 | fails the 99% floor for PII and Halu |

`batching` waits up to 2 ms for concurrent requests, so a lone request pays
the window, and under load it serves about 40% more. The BF16 copy adds a
cast per linear to a launch-bound forward and is slower still, so the
family consents to none (`vela1-parity.md`).

## Where the time goes, and what the runtime does about it

Every number below is Vela 307M in FP32.

- **One forward for every head.** A request's inputs, their windows, every
  head that reads them and every task of a bundle become one packed
  forward. Identical token sequences are computed once, each head reads its
  rows of the shared hidden states, and the result cache keys rows by
  content.
- **Packed rows, attention in grids of similar length.** Embeddings, norms,
  projections and MLPs run on the real tokens only, and attention scatters
  rows into grids. Rows whose grid would pad more than 25% of its tokens
  (past 1,024) attend in separate grids, because attention costs
  rows × width². One grid against length groups, 16 EPYC vCPUs / MI325X:

  | Batch (tokens) | CPU | MI325X |
  | --- | --- | --- |
  | 2,000 + 7 × 20 | 4,154 → 708 ms | 60.7 → 12.9 ms |
  | 8,192 + 7 × 16 | 22.5 → 3.5 s | 287 → 51 ms |
  | 4,096 + 31 × 24 | 32.8 → 1.6 s | 464 → 25 ms |
  | 512, 300, 64, 32, 16, 16, 10, 8 | 622 → 210 ms | 18.3 → 10.6 ms |
  | 16 × 128 (uniform) | 499 → 493 ms | 8.9 → 8.8 ms |

- **Local layers in query blocks for long rows.** Two thirds of the layers
  attend within 64 positions. Long rows run them in 128-query blocks over a
  256-key span through the fused 4-D SDPA kernels, from 1,024 tokens on CPU
  and from 2,048 on MI325X. On MI325X a 2,048-token row goes 17.7 → 11.7 ms
  and 8,192 tokens 91 → 50 ms. Below 2,048 dense masked SDPA is faster
  there: 1,024 tokens take 7.5 ms dense against 11.0 blocked.
- **GPU: a graph per length bucket.** A short forward is launch overhead,
  so batches pad to a (rows, width) bucket and replay a HIP graph captured
  on the bucket's second use: 8 tokens 3.96 → 2.28 ms, 128 tokens
  4.17 → 2.98 ms, 16 × 64 tokens 6.02 → 4.86 ms. A bucket replays only while
  it pads at most 25% of the real tokens or is launch-bound (≤ 1,024
  tokens); other batches run packed. The fused gfx942 rotary kernel
  (`embed`) runs inside.
- **GPU: one model's device work at a time.** Models capture their bucket
  graphs while serving, and a capture fails when another model launches on
  the device meanwhile. So the models of a process run their device work one
  at a time per GPU (design section 9). That also pays: five Vela encoders
  on one MI325X, called concurrently with graphs off, serve 208
  calls/s against 92 with overlapping launches, and 458
  with graphs (5 rotated rounds, medians).
- **CPU: oneDNN's packed FP32 linears.** GEMMs are 80% of a short forward,
  and `F.linear` (MKL on EPYC) streams the weights at ~37 GB/s. oneDNN with
  weights reordered once is 2.5–4.4× faster per GEMM (768 → 2,304: M = 10
  71 → 23 µs, M = 512 1,098 → 426 µs), and gives a row the same result in
  any batch. A whole one-row forward, measured in steady state (one backbone
  per process, no page faults, as in a serving process), is 1.7–2.3× faster
  at every length: 10 tokens 15.6 → 6.7 ms, 64 29.5 → 14.6, 128 42.8 → 21.9,
  256 65.6 → 34.5, 512 112 → 62, 1,024 206 → 116, 2,048 413 → 250.
- **CPU: `exact` batches concurrent requests without changing an answer.**
  Each row's forward is bit-identical alone or inside any batch, given:
  - packed linears;
  - GeGLU and the score sigmoid on aligned rows (`rowwise`);
  - unmasked grids for unpadded rows;
  - one rotary table per grid width;
  - grids of one length only.

  The family probes this on the host at load before it lets `exact`
  batch, so a PyTorch build where a kernel is not invariant keeps one
  request per forward.
- **CPU: one device thread.** PyTorch's OpenMP keeps a thread team per
  calling thread. Two teams on 16 vCPUs stop libgomp's spin-waiting, so
  every parallel region pays a wake-up. With one thread doing all CPU device
  work, a short forward goes 25 → 13 ms.
- **`Runtime.call` overhead:** 0.58 ms p50 on a 16 ms Domain call: planning
  0.22 ms (tokenization 0.11 of it), queue hops 0.08, the response 0.29.

## Reproduce

```bash
python3 tools/legacy_parity.py build-legacy --recipe cpu --tree <legacy tree> \
  --cache <hf cache> --flat <dir> --out legacy-cpu.test      # in the bindings' userland
sudo systemd-run --scope -p AllowedCPUs=96-111 --uid=$(id -u) --gid=$(id -g) \
  python3 tools/legacy_parity.py ab --binary legacy-cpu.test \
  --tree <legacy tree> --cache <hf cache> --threads 16 --rounds 5 --concurrency 4 \
  --seconds 20 --profile batching --out ab.json
# a stopped run, or more rounds: the same command with --rounds N --resume ab.raw.json

# ROCm, each round on one GPU and the same host cores, the order flipping each round:
python3 tools/legacy_parity.py legacy --recipe amd --inputs legacy-amd.jobs.json \
  --repeats 1 --concurrency 4 --out legacy-rN.jsonl ...       # in the legacy ROCm image
python3 tools/legacy_parity.py runtime --recipe amd --inputs legacy-amd.jobs.json \
  --device rocm:0 --repeats 1 --concurrency 4 --out runtime-rN.jsonl ...   # in the router's ROCm image
```
