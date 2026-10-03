# Runtime C (inference optimization) — state

Owner: inference-optimization worker 885d85cc (`track=runtime-c`), started 2026-10-03 11:35 UTC+8 after the
user stopped all training (COORDINATION 11:00 / 11:12). Budget 60 GPU-h. The Index submission worker (f38ee089) has
first call on every GPU. Times are UTC unless marked.

**Retargeted (COORDINATION 11:27 UTC+8):** all inference optimization lands in the built-in model runtime
`vllm_sr_runtime` (`src/model-runtime/`) on the lead's branch `xunzhuo/model-runtime-decision2` (lead 54e49843,
PR #4481). I own `families/decision2` (numerics), `engines/native`, `accel/*`, `profiles/*` and the GPU parity /
perf records. **New HF runtime revisions are paused**; the ROCm hotfix owner (490b6f72) is the temporary sole
writer of Nox / Lux / Vega. No repo was written by this worker.

## Current handoff

- **Branches:**
  - runtime C (HF runtime): `xunzhuo/decision-2-runtime-c` (`dcd15f5ce` runtime; worktree
    `vllm-sr-dev2-runtime-c`). Ready but paused; it would ship only if HF revisions resume (`rc.sh` needs a
    frozen-cache `triton.json` per key).
  - model runtime port: `xunzhuo/model-runtime-decision2-inference` (worktree `vllm-sr-mr-inference`). It
    merges the lead's PR branch at `203113687` (merge `61e989a2f`).
- **Model runtime commits after the port:**
  - `f21224e0d`: `max_speed` (shared-context trees plus batching); the bench times shared_context.
  - `012f76a0a`: GPU test above 2 GiB of q projection.
  - `61e989a2f`: merge of the PR branch; the backbones are mine, plus the lead's named constants.
  - `18c51bdaa`: ruff clean under the repo config.
  - `df51b0d5f`: graphs only up to 4,096 padded tokens; throughput after two warm passes.
- **PR branch** `xunzhuo/model-runtime-decision2` (PR #4481; the lead merged the port as `58c8550f6`) is at
  `48e12c040`. My commits on it:
  - `f705e2d10`: graph cap of 4,096 padded tokens;
  - `99208b6d3`: never destroy captured graphs;
  - `48e12c040`: `docs/records/` (parity, perf, profiles).
  - Local branch `mr-work` in `vllm-sr-mr-inference`. The old branch
    `xunzhuo/model-runtime-decision2-inference` is superseded.
- **Running:** nothing. **Leases:** all released (nodes A, B, F).
- **Next:**
  1. Released runtime: a runtime-only revision for all six that stops graph eviction, on the coordinator's go
     (HF revisions paused).
  2. Pin FLA autotune choices per device class (frozen cache or fixed configs), so the Qwen3.5 exact path is
     reproducible across processes; then golden answers per device class.
  3. Batching: shape bucketing for graph hits under real concurrency. Measure the Index on more sizes; Eos leans
     negative with its CI containing 0.
  4. `max_speed` approximate kernels (merged projections) with their own accuracy record.
  5. Optional vLLM investigation.

## Results so far

- **HF runtime C parity** (four panels, 10,653 prompts, Transformers 5.17 and 5.18): all six identical, 0.0 drift.
- **Native engine vs the released runtime** (same panels): fast and eager paths identical for all six sizes.
  - The graph cap was re-checked on Kai and Eos (identical).
  - GPU tests 5 / 5; CPU suite 96 passed, 6 GPU skips.
- **Native shared_context vs the released switch** (207 requests, every one shared): identical for all six sizes.
- **Batching** (16 concurrent requests):
  - panels: Kai typed-final −0.35 pt [−0.65, −0.05]; every other panel and size has a CI containing 0;
  - Kai Index (released-runtime harness, exact sample 900 / 900 equal to the base): the paired bootstrap CI
    contains 0.
- **Eos Index, profiles against exact in one process** (native engine with the fix, all 120,226 requests):
  - batching changes 999 / 282,335 decisions (0.35%) and max_speed 1,007 (0.36%); no status changes;
  - the noise floor (each question alone) is 0.30–0.41% on the Index sample;
  - the bootstraps are in `private-ix/eos-paired-f/boot-*.json`.
- **Autotune:** FLA's gated-delta kernels autotune per process, so the exact path of the Qwen3.5 sizes is
  bit-identical only under the same autotune choices. Two processes matched the stored Eos base on 667 and
  834 / 900 requests at rounding level. Parity runs reuse the release autotune cache.
- **Graph eviction crash:** large graphs evicted from the shared pool between large eager batches gave GPU memory
  access faults: 5 / 6 Eos shards; reproduced at group 155. Without eviction there were 0 / 6 crashes over the
  full Index. Evicting small graphs (8-graph cache, 1,700+ evictions) did not crash. Fix `a16d41f5c`: stop
  capturing at the limit. The released runtime has the same eviction code; not reproduced there.
- **Throughput:** batching scales on the GPU (Eos 1.35 ms per row at 64 rows; 710 req/s at 64 concurrent with
  pre-rendered requests). Under independent clients most batch shapes are new, so graphs were rarely replayed and
  captures cost several forwards each. Hence the 4,096-token graph cap, which is bit-identical because graphs and
  eager runs compute the same.
- **`attn_prep` 2 GiB compile failure:** root cause found; the native engine launches q and k separately
  (verified above 2 GiB); the same failure hit an Eos Index row (posted 13:30 UTC+8).

## GPU-hours (approximate)

| Job | Node / GPU | GPU-h |
| --- | --- | --- |
| profiles, benches, GPU tests | B 0–3 | 1.6 |
| repro / 2 GiB tests | C 1, F 3 | 0.2 |
| runtime C parity chains | A 0–6 | 6.0 |
| batching bench / fidelity / Index runs (HF harness) | B 0–3 | 2.5 |
| native parity (eager + fast), shared verification | A 2–6 | 3.5 |
| perf lanes (released + native, two rounds) | F 2–4 | 2.0 |
| Eos Index, native batching / max_speed / exact (four rounds incl. the crash hunt) | B 0–7 | 4.5 |
| GPU tests, parity re-checks, row-scaling and contention probes | A 2–6, F 2 | 1.0 |
| perf rerun (clean lane) | F 2–4 | 1.0 |
| **Total** | | **≈ 23** |
