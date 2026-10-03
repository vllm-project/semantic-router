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
- **Running:**
  - node F GPU2 / 4 / 3: `bin/perf_chain.sh` with `NATIVE_ONLY=1`, native perf at `df51b0d5f`
    (Kai / Eos / Sol, Nox / Lux, Vega), into `perf/<tier>-2026...T061*`;
  - node B GPU0–2: `ixnative.py --profile exact` for Eos, the same-image exact baseline
    (`results/ixn16-Eos-exact-s*`);
  - node B CPU: `eos_score.sh`, merge and bootstrap of the native batching / max_speed Eos runs against the
    stored base (`private-ix/eos-native-*`).
- **Leases:** node F GPU2–4 and node B GPU0–2 (`owner.runtime-c`); node A all released.
- **Next:**
  1. Merge the exact baseline; bootstrap batching and max_speed against it; write `docs/records/profiles.md`.
  2. Finish the perf record from the `df51b0d5f` lanes.
  3. Commit the records; post "ready" for the lead's merge.
  4. Report to the coordinator.

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
- **Eos Index, base mismatch:** the stored base differs from today's exact path at rounding level. The native
  exact sample is 667 / 900 identical, single-question requests included (median drift 0.0025, max 0.02),
  probably a different gated-delta kernel environment. Profiles are therefore compared against a same-image
  exact baseline.
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
| Eos Index, native batching / max_speed / exact | B 0–3, 6, 7 | 1.5 |
| GPU tests, row-scaling and contention probes | A 2, F 2 | 0.3 |
| **Total so far** | | **≈ 18** |
