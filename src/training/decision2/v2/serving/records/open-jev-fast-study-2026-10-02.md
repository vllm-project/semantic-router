# open-jev-fast study: what transfers to the Decision 2.0 runtime on MI325X (2026-10-02)

Worker 2d541b40; branch `xunzhuo/decision-2-ojf-study`; state [`ojf-study-state.md`](ojf-study-state.md).
Subject: [`lyuyiqi/open-jev-fast`](https://github.com/lyuyiqi/open-jev-fast) at `c52b8bb` (MIT), a faster
inference backend for Open-Jev-27B (a LoRA on Qwen3.8-27B) on one NVIDIA B300: 103.7 → 17.3 ms per example
request through a LoRA merge, CUDA graphs, a per-request prefix tree, cuBLASLt tuning and hand-written CUDA / PTX
kernels.

Ground rules kept: the third-party repository was read, not run; no third-party code, weights or data entered this
repository; prototypes are private scratch re-implementations against Transformers / PEFT / PyTorch; no released
package was changed. Private eval-panel (Index) structure, timings and throughput stay in the private report.
Times are UTC.

## Result

- **Adopt (bit-identical, measured):** one HIP graph per exact padded shape through `torch.cuda.graphs`, with the
  per-layer-type attention masks built from host-known lengths (no GPU→CPU syncs), plus two exact kernel trims
  (one BF16 cast per shared Linear input; RMSNorm's `1 + w` computed once) and, for the 27B, a leaner unmerged LoRA
  path (BF16 factors, the power-of-two scale folded into B, one input cast). Latency panel p50: 0.8B 23.9 → 9.5 ms,
  4B 26.8 → 19.7 ms (graph alone), 27B 93.4 → 75.2 ms; every latency-panel answer bit-identical to the released runtime (scored panels: 0.8B and 4B
  bit-identical; 27B not completed, study paused).
- **Do not adopt under the current parity rule (0 answer changes):** the LoRA merge into the BF16 base (27B: 4 of 400
  decisions changed, max |dp| 0.36 — the rank-64 delta does not survive BF16 rounding of the merged weights),
  Open-Jev's fused RMSNorm (0.8B: 3 of 400 changed, max |dp| 0.011) and the prefix-state handoff for multi-question
  requests (decision changes on a private sample; slower at small sizes). Removing the mask syncs alone gains nothing.
- **Not portable / not applicable:** the PTX Gated DeltaNet and tree-attention kernels (NVIDIA-only), the
  per-candidate prefix tree (our runtime already scores all candidates of a question in one row), split-K partial
  sums reduced inside RMSNorm (cuBLASLt-specific), server tweaks (our evals are in-process).

## 1. Catalogue

Open-Jev's numbers are its own (1x B300, 27B, example request of 3 questions / 7 candidate prompts / 539 tokens).
"Ours" is the shipped `decision2/qwen.py` runtime (Transformers 5.17 Qwen3.5 modules, FLA 0.5.2 + causal-conv1d,
BF16 autocast with BF16-resident Linear weights, FP32 residual stream, norms and head).

| Optimization (Open-Jev) | Their gain / evidence | Ours today | Portable to MI325X / our tiers | Our measurement / expectation | Effort |
| --- | --- | --- | --- | --- | --- |
| Merge the LoRA into the BF16 base | 109.9 → 100.1 ms; max dp .005; 1 near-tie flip on their 231-task benchmark | 27B LoRA unmerged, FP32 factors cast to BF16 on every call | yes; 27B only | 27B −27 ms but **4 / 400 decisions changed, max dp .36**: rejected. A bit-identical lean unmerged path keeps −14 ms | low |
| Fused RMSNorm (`F.rms_norm`, cached `1 + w`) | 100.1 → 80.6 ms; max dp .003 | HF FP32 RMSNorm (~7 kernels) | yes; all | 0.8B −0.9 ms after graphs but **3 / 400 changed**: rejected under the 0-change rule. The exact part (cached `1 + w`) is kept | low |
| Remove 2 CPU syncs in the mask code | 80.6 → 78.6 ms; same math | 2–4 syncs per request | yes (host-built per-layer-type mask dict) | 0 gain alone; required for graphs; bit-identical | low |
| Whole-model CUDA graph per exact shape | 78.6 → 58.0 ms; same math | none | yes: `torch.cuda.graphs` captures FLA Triton, hipBLASLt and causal-conv1d on ROCm | 0.8B 2.2x, 4B 1.4x, 27B 1.04x; bit-identical | low–med |
| causal-conv1d + `torch.compile` | 42.5 → 32.4 ms | causal-conv1d already used; no compile | inductor works on ROCm, but fusion / FMA contraction changes rounding; thousands of shapes | not run: same class as the fused RMSNorm (changes answers) | med |
| Six fused CUDA kernels; 9 → 4 GEMMs per layer; ~5,000 → 945 kernels per request | phase 2 32.4 → 20.1 ms with the items below | ~1,650 (0.8B) to ~7,900 (27B) kernels per request | CUDA C++; needs a Triton / HIP rewrite; changes rounding | after graphs every tier is bound by per-kernel GPU cost, so this is the next big lever — but it changes answers | high |
| Prefix tree (root / question / candidate) | 574 → 277 rows on the example; max dp .019 on a 10.7k-token request | candidates already share one row per question; a request's shared context is recomputed once per question | yes (HF 5.17 cache continuation) | state-handoff prototype: decision changes on a private multi-question sample; slower except on long shared contexts | med–high |
| cuBLASLt per-shape algorithm timing | part of phase 2 | hipBLASLt heuristic | PyTorch TunableOp (hipBLASLt + rocBLAS) | see §3 | low |
| Split-K with FP32 partials reduced in the RMSNorm kernel | part of phase 2 | — | cuBLASLt-specific | not portable as is | high |
| Graph cache per (rows, candidates, path) bucket, LRU 512 | part of phase 2 | — | yes | exact-shape keys keep answers bit-identical; bucketing changes GEMM shapes | low |
| PTX Gated DeltaNet kernel | 74 → 36.5 µs per layer; rel L2 1–2e-3 vs FLA | FLA Triton | NVIDIA-only (`mma.sync`, `ldmatrix`, `cp.async`) | not portable | very high |
| PTX tree-attention kernel | 21.4 → 12.7 µs per layer | SDPA | NVIDIA-only; needs a tree | not applicable | very high |
| LUTs, grouped q/k, no per-layer fills, buffer reuse | ~1–2 ms | — | ideas portable via Triton | minor | med |
| Server: TCP_NODELAY, buffered log, batched tokenizer | ~1–2 ms | in-process | n/a for in-process eval | encode 0.6–1.4 ms per request | low |
| FP8 GEMMs (theirs: measured, rejected) | GEMM 14.6 → 7.8 ms; changes numerics | — | MI325X has FP8 | rejected (numerics) | — |

## 2. Profiling (one MI325X; latency panel = the first 400 typed-final prompts, ~346 tokens, single requests)

| Tier | p50 ms | GPU kernels per request | GPU kernel time | GPU-busy | Bound by |
| --- | --- | --- | --- | --- | --- |
| 0.8B | 22.3–23.9 | ~1,650 | 10.4 ms | ~47% | CPU kernel launches (like Open-Jev with FLA: ~5,000 kernels, 51 of 103 ms busy) |
| 4B | 26.8 | ~2,180 | 19.7 ms | ~73% | mixed |
| 27B | 90.2–94.3 | ~7,900 (~3,600 from the unmerged LoRA) | 91.7 ms | ~97% | GPU: base GEMMs ~38 ms plus thousands of tiny kernels |

- Host stages per request: encode 0.6–1.4 ms, collate + H2D ~0.3 ms, readback + answer ~0.1 ms; 4 syncs.
- Eager CPU cost: ~10 ms of `hipLaunchKernel` per request at 0.8B; FLA's Python wrapper ~0.35 ms per gated-delta
  layer.
- GPU cost: at 0.8B the FP32 norm pieces and ~200 BF16 casts outweigh the GEMMs (~1.5 ms) and FLA (~2 ms); at 27B,
  2,016 BF16 casts take 11 ms and the LoRA side path another ~10 ms.
- With graphs every tier becomes GPU-bound; the floor is the per-kernel GPU cost of many small kernels, which is
  why Open-Jev's phase 2 fused kernels instead of only capturing them.

## 3. Prototypes and fidelity

Harness (private scratch): loads each released package through its own `decision2` runtime (as `runtime_bench.py`
and the Index engine do) and answers the same requests with the released `system_one` (reference) and with a copy
of it whose forward is swapped, in one process, so weights, image, kernels and the frozen autotune cache are shared.
Fidelity uses the release parity definitions (answer category changes; max |difference| of any reported number).
One leased MI325X per run, scored images (`f83b1d10` for 0.8B / 4B, `dbe5f32b` for 27B), `--network none`, only the
leased GPU's render node.

**Latency panel** (first 400 typed-final prompts, single requests, one untimed pass, then timed; ms):

| Tier | Released p50 / p95 | Stack | Stack p50 / p95 | Kernels per request | Answers vs released |
| --- | --- | --- | --- | --- | --- |
| 0.8B | 22.3–23.9 / 22.5–25.0 | host masks only (sync removal) | 22.6 / 23.3 | 1,643 | 400 / 400 bit-identical |
| 0.8B | | HIP graph per exact shape + host masks | 10.1–10.7 / 11.9–12.5 | 1,645 (20 launch calls) | 400 / 400 bit-identical |
| 0.8B | | graph + exact trims | **9.5 / 11.4** | 1,494 | 400 / 400 bit-identical |
| 0.8B | | + Open-Jev fused RMSNorm | 8.6 / 10.4 | 1,189 | **3 / 400 changed**, max dp 0.011 |
| 4B | 26.8 / 30.2 | host masks only | 27.3 / 28.1 | 2,172 | 400 / 400 bit-identical |
| 4B | | graph + host masks | **19.7 / 22.4** | 2,174 | 400 / 400 bit-identical |
| 27B | 90.2–94.3 / 95.5–98.5 | host masks only | 94.9 / 96.4 | 7,895 | 400 / 400 bit-identical |
| 27B | | graph + host masks | 88.4–88.6 / 94.1–94.4 | 7,897 | 400 / 400 bit-identical |
| 27B | | lean unmerged LoRA (eager) | 79.9 / 85.1 | — | 400 / 400 bit-identical |
| 27B | | lean LoRA + trims + graph | **75.2 / 80.6** | 5,384 | 400 / 400 bit-identical |
| 27B | | LoRA merged into BF16 (+ graph) | 66.7 / 71.7 | 4,297 | **4 / 400 changed**, max dp 0.36 |
| 27B | | TunableOp on the released path / on the stack | 92.3 / 72.0 (p50) | — | **3 / 400 changed**, max dp 0.11 |

**Scored panels** (all four, 10,653 prompts / 11,053 answers, one pass, captures included):

| Tier | Stack | typed-final | css15 | public231 | mlx-diag | Max drift | Graphs captured |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B | graph + trims | 0 / 2,000 changed | 0 / 6,547 | 0 / 231 | 0 / 2,275 | 0.0 (every prompt bit-identical) | 525 (73 ms each) |
| 4B | graph + trims | 0 / 2,000 | 0 / 6,547 | 0 / 231 | 0 / 2,275 | 0.0 (every prompt bit-identical) | 525 (155 ms each) |
| 27B | lean LoRA + trims + graph | not completed (paused) | | | | | |

Why the LoRA merge fails: the 27B adapter is rank 64 on every Linear of a BF16 base; rounding `W + dW` to BF16
discards much of `dW` element by element (its size is comparable to the BF16 spacing of `W`), so the merge changes
the model, not just the rounding order. The lean path instead keeps the unmerged arithmetic exactly: PEFT under
autocast launches 9 kernels per adapted Linear (input cast for the base GEMM; input, A and B casts for the side path;
three GEMMs; scale multiply; add); holding A and B in BF16 (the values autocast multiplies with), folding the scale
2 into B (exact for a power of two) and casting the input once leaves 5 kernels with identical products.

## 4. Quality side

- **Cheaper evaluation loops: real.** The bit-identical stack makes private-panel runs measurably cheaper (largest
  at the small tiers, smaller at 27B where long and multi-question requests are GPU-bound); a frozen autotune
  cache for that panel's shapes saves a few percent more and makes shards reproducible. That buys more candidate
  runs per GPU-day under the Index-first rule; it does not raise any score by itself.
- **Latency headroom: real but not a frontier lever.** With the stack the 4B answers in about the time the 0.8B takes
  today; the Index frontier is parameters vs score, so a larger model at equal latency does not move it.
- **Training throughput: not shown.** Training micro-batches are already length-sorted inside 4,096-row buckets
  under a token budget (`v2/dec/batching.py`), so padding waste is small; sharing a state's prefix across its
  questions during training would change batch composition and needs backward through the state handoff; graphs
  do not apply. Not worth pursuing without first measuring the shared-context share of the training mixtures.
- **What does not help:** the per-candidate tree (we already score all candidates in one row), FP8, the LoRA merge,
  the PTX kernels, server tweaks, and sync removal on its own.

## 5. Recommendations (ranked by measured gain at no answer change)

1. **Adopt now (next runtime-only revision, standard parity rollout): HIP graphs per exact padded shape with
   host-built masks, plus the exact kernel trims.** Measured bit-identical on the latency panel at all three tiers
   and on all four scored panels at 0.8B and 4B (11,053 / 11,053 answers identical each). Latency p50 0.8B −60%, 4B −27%; eval
   GPU time on the private panel falls roughly a fifth at the small tiers (numbers private). Risks: graph memory
   (bounded by an LRU cap and an eager fallback above 64k padded tokens), capture time per new shape (0.07–0.35 s,
   capture on the second sighting), coupling to the image's Transformers mask semantics (keep a `graphs=False`
   switch and the parity gate). Effort: low–medium (~150 lines in `decision2/qwen.py` + tests).
2. **Adopt now with it (27B): the lean unmerged LoRA path.** Bit-identical by construction and measured; 27B p50
   −11% alone, −19.5% with graphs and the trims. Asserts vanilla, bias-free LoRA with a power-of-two scale and
   eval-mode dropout, else falls back to PEFT. Effort: low (~40 lines).
3. **Eval-side, now: a pre-warmed, frozen Triton autotune cache for the private eval panel's shapes.** Each shard
   currently compiles and autotunes shapes missing from the scored-run cache on first use (a few percent of every
   run, numbers private) and may pick different configurations per shard; freezing one cache removes both.
4. **Not now: TunableOp.** 27B: −1% on the released path, −4% on the stack, and 3 of 400 decisions changed
   (max |dp| 0.11). Revisit only together with item 5.
5. **Later, only with a parity-policy decision: numerics-changing fusion** (fused norms, fused element-wise
   kernels, `torch.compile`, Open-Jev's phase-2 style kernels in Triton). After graphs every tier is bound by
   per-kernel GPU cost, so this is the next big latency lever, but even the smallest step (fused RMSNorm) changed
   3 of 400 near-tie decisions. It needs either a tolerance rule or a one-time re-baselining of the scored
   predictions.
6. **Later, strategic: batch-invariant GEMM / attention / norm kernels.** Prefix sharing, packing, micro-batch
   splits and cross-request batching all fail exact parity today only because hipBLASLt picks different kernels for
   different row counts. One re-baseline onto batch-invariant kernels would make all of them exact afterwards.
7. **Reject:** the LoRA merge into BF16 (changes ~1% of decisions), the PTX kernels and tree attention
   (NVIDIA-only / not applicable), the per-candidate tree, split-K-in-RMSNorm, server tweaks, sync removal alone.

_Paused 2026-10-02 04:13 UTC by user decision before the 27B scored-panel pass and the private-panel runs finished;
see the state log for what remains and how to resume._
