# Decoder track Milestone 2 results (0.8B / 2B / 4B), 2026-09-28

Development readouts (SELECT, CAL, typed DEV1600, CSS pilot1430, A7 AHO) are
never release scores. JevArena v3 and public JevBench 231 are **post-key
same-panel** comparisons (node A, the eval track's frozen runner). Proxy
P = 100·√(T·H); |ΔP| < 4 is a tie (eval track calibration). Intervals are
10,000-draw paired bootstraps over items/groups (no seed variance).

Records: [prereg](dec-m2-prereg-2026-09-28.md) `143cfc214`;
[amendment 1](dec-m2-amendment-1-2026-09-28.md) `f316bad37`;
[amendment 2](dec-m2-amendment-2-2026-09-28.md) `181ec58a0`;
[amendment 3](dec-m2-amendment-3-2026-09-28.md) `fc6d1e508`;
locks [X4R](dec-m2-x4r-formal-lock-2026-09-28.md) `0ac081347`,
[batch 1](dec-m2-batch1-formal-lock-2026-09-28.md) `6dbc18141`;
[0.8B candidate](dec-m2-0p8b-candidate-2026-09-28.md) `f02b73966`.
Code: checks `aac6959ad`–`56525ff81`, trainer/builder/driver `15ef238ef`,
`163d40dab`, diagnostics `230e7ddb1`, `efc099914`, `1fbd917c0`, soup/calibration
`fc6d1e508`; shared change `0a399c1d9` (calibration loader, see below).

## Correctness checks (coordinator decision 2)

| Check | Outcome | Milestone 1 arms |
| --- | --- | --- |
| (a) padded vs unpadded micro-batches | **No bug.** Real Eos / Qwen3.5-0.8B-Base / Sol / Nox, image kernels: gradient cosine ≥ 0.99999 (FP32), ≥ 0.9985 (BF16); zero decision flips; BF16 gaps equal BF16's own rounding on unpadded rows and do not grow with padding (62 → 29,918 pad tokens). Tiny real Qwen3.5 hybrid regression test: padded = unpadded to 1e-5. The probe's pre-set absolute thresholds were tighter than TF32/BF16 kernel numerics (it prints FAIL; disclosed). | valid |
| (b) Score-only overfit | **PASS**: 288/288 (32 rows per level count L = 2..10, every level predicted). The same checkpoint answers level 0 on 399/400 typed-DEV Score items → the 0.8B DEV Score collapse is a transfer limit, not a defect. | valid |
| (c) runtime | **Fixed.** Every GPU entry point requires FLA gated-delta + causal-conv1d kernels and a shared persisted Triton autotune cache; preflight requires exact cross-process reload (all M2 arms: drift ≤ 9e-8, 700/700). Node B's M1 image `ce895822…` lacked causal-conv1d (reference convolution; node B now `dbe5f32b…`, identical versions to node A); M1 C1/E1 had no persisted cache (≤ 5/700 cross-process flips). | flagged runtime-noncompliant, not invalid |

**Data check after the data track published positional-key files (20:15,
revision `d8eae3e4…`, `m3/pk1/A0s`):** the decoder's locally renumbered A0s-r
(6,547 rows) equals pk1 A0s minus the 752 excluded-family rows exactly
(identical id, input hash and label for all 6,547 rows; no other difference).
Every M2 mixture therefore used the canonical renumbered A0s.

## Same-limit 1.0 controls (post-key, node A, 8,192 tokens, shared 2.0 renderer)

| 1.0 model | v3 | vs adopted native run | T | H | Choice / Noul / Score | public231 |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| Nox 1.0 | 55.689 | −0.78 [−1.34, −0.07] (56.470) | .5975 | .5190 | 549 / 627 / 180 (native 552 / 653 / 178) | 173 |
| Sol 1.0 | 45.544 | −0.04 [−0.60, +0.83] (45.580) | .4228 | .4906 | 372 / 438 / 153 | 161 |
| Eos 1.0 | 42.361 | −0.19 [−0.55, +0.55] (42.547) | .3931 | .4565 | 318 / 406 / 121 | 142 |

The 2.0 renderer costs Nox 1.0 itself 0.78 v3 points (26 typed-FINAL Noul
answers); M1's E1-X2 was +0.30 over this control (its "exception stack −37"
vs the native run was mostly the renderer).

## Arms (development vs post-key, labelled)

Development = node B, M2 image, same-image 1.0 control; post-key = node A,
8,192 tokens unless stated, vs the adopted 1.0 run (and the same-limit control).

### 4B (start Nox 1.0 `@cde2a68d`)

| Arm | Dev T / H / P | Dev C / N / S | Post-key v3 [Δ vs Nox1 56.470, 95% CI] | Post-key T / H | Post-key C / N / S | public231 |
| --- | --- | --- | --- | --- | --- | ---: |
| Nox 1.0 (same image) | .6663 / .4131 / 52.46 | 460 / 228 / 378 | 56.470 (native); 55.689 same-limit | .6144 / .5190 | 552 / 653 / 178 | 173 |
| X4C-s1 control (A0s-r + own-Lux KL) | .6638 / .4547 / 54.94 | 471 / 224 / 367 | dev only | | | |
| X4C-s2 | .6606 / .4488 / 54.45 | 474 / 225 / 358 (Score floor fail) | dev only | | | |
| X4R-s1 retention (+ A2 + stage4v2 replay) | .6725 / .4319 / 53.90 | 457 / 233 / 386 | **54.413** [−2.06; −4.45, +2.04] | .6050 / .4894 | 537 / 646 / 185 | 175 |
| X4R-s2 | .6725 / .4437 / 54.62 | 484 / 222 / 370 | **53.705** [−2.76; −5.78, +1.59] | .6013 / .4797 | 544 / 641 / 177 | 179 |
| X4K-s1 trust region (A0s-r + own-Nox KL) | .6669 / .4353 / 53.88 | 468 / 225 / 374 | **55.362** [−1.11; −3.75, +1.63] | .5981 / .5124 | 560 / 617 / 180 | 173 |
| X4K-s2 | .6588 / .4240 / 52.85 | 460 / 223 / 371 | **54.116** [−2.35; −4.83, +0.88] | .5825 / .5027 | 539 / 614 / 179 | 175 |
| E4F-h-s1 Nox full FT, half mixture (71.2M tokens) | .5744 / .3641 / 45.73 | 522 / 214 / 183 | dev only (−6.73 [−10.03, −3.49] vs Nox; screen fail) | | | |
| B4F-h-s1 Qwen3.5-4B-Base full FT, half mixture | .6369 / .3542 / 47.50 | 632 / 214 / 173 | dev only (−4.97 vs Nox; screen fail) | | | |

Reference: M1 E1-X2 55.993 (T .5844, H .5365; A0-based, FLUTE-inflated H).
**4B: HOLD** — no M2 recipe beats Nox 1.0; retention replay restores typed
reasoning and public231 but loses CSS15 transfer; the own-Nox trust region
stays at Nox 1.0 (+0.0 T, −.007 H vs the same-limit control).

### 2B (start Sol 1.0 `@ce0c018a`) — development only

| Arm | T / H / P | C / N / S | Screen |
| --- | --- | --- | --- |
| Sol 1.0 (same image) | .5869 / .3191 / 43.27 | 410 / 218 / 311 | — |
| S4C-s1 | .5594 / .3370 / 43.42 | 376 / 225 / 294 | fail (Choice −4.3 points) |
| S4C-s2 | .5931 / .3351 / 44.58 | 425 / 224 / 300 | pass |
| S4R-s1 | .5875 / .3260 / 43.76 | 418 / 226 / 296 | fail (Score −3.75) |
| S4R-s2 | .5756 / .3466 / 44.67 | 408 / 228 / 285 | fail (Score −6.5) |

**2B: HOLD** (no recipe passes on both seeds; every |ΔP| < 1.5).

### 0.8B (full fine-tuning on the full mixture, 138.4M tokens)

| Arm | Dev T / H / P | Dev C / N / S | Post-key v3 [Δ vs Eos1 42.547, 95% CI] | Post-key T / H | Post-key C / N / S | public231 |
| --- | --- | --- | --- | --- | --- | ---: |
| Eos 1.0 (same image) | .4956 / .1887 / 30.58 | 510 / 198 / 85 (all level 0) | 42.547 (native); 42.361 same-limit 8K/16K | .3925 / .4612 | 315 / 410 / 120 | 142 |
| E8F-s1 (Eos start) | .5563 / .2558 / 37.72 | 486 / 236 / 168 | **49.777** [+7.23; +1.83, +13.68] (8K) | .5713 / .4337 | 510 / 613 / 103 | 150 |
| E8F-s2 | .5619 / .2768 / 39.44 | 523 / 193 / 183 | **47.360** [+4.81; +0.72, +9.59] (8K) | .4931 / .4548 | 454 / 476 / 117 | 157 |
| E8F-s3 | .4063 / .2839 / 33.96 | 374 / 191 / 85 | 41.084 [−1.46; −4.28, +4.99] (8K) | .4644 / .3635 | 409 / 530 / 76 | 149 |
| **E8F soup (artifact; CAL698; 16K)** | .6131 / .2705 / **40.73** | 610 / 212 / 159 | **50.236** [+7.69; **+3.65, +13.32**] | .5734 / .4401 | 529 / 611 / 107 | **156** |
| B8F-s1 (official Qwen3.5-0.8B-Base start) | .6838 / .2592 / 42.10 | 723 / 221 / 150 | 43.084 [+0.54; −3.53, +5.04] (8K) | .4122 / .4503 | 323 / 476 / 85 | 157 |
| B8F-s2 | .4925 / .2437 / 34.65 | 421 / 204 / 163 | 44.000 [+1.45; −1.26, +6.91] (8K) | .5034 / .3846 | 450 / 536 / 92 | 150 |

**0.8B: the E8F soup meets the first-release threshold** (record
[`23df7f5c6`](dec-m2-0p8b-release-candidate-2026-09-28.md)). Seed evidence:
two of three E8F seeds beat Eos 1.0 on their own (s1 +7.23, s2 +4.81 with
lower bounds > 0); s3's human transfer collapsed (H .3635) and it lands at
−1.46; the seed mean on v3 is 46.07 (+3.5); the soup, chosen on development
data by the preregistered rule, is above every seed (50.236). The
official-Base start led on the development proxy (seed 1) and did not beat
Eos 1.0 on v3 with either seed (+0.54, +1.45; both intervals cross 0); it is
also seed-unstable (typed DEV Choice 723 vs 421). Soup `mlx-diag`
(development diagnostic): type-macro 65.2 vs Eos 1.0 66.5, English 67.1 vs
67.9, non-English Choice / Noul / Score 68.7 / 54.2 / 71.9 vs 68.5 / 59.2 /
71.0 (PAWS-X Noul −5.0, disclosed), weakest language ko 55.0 vs 56.0.

## GPU-hours (wall-clock × GPUs from every job receipt; CPU builds excluded)

| Use | GPU-h |
| --- | ---: |
| Correctness checks (padding probes 0.23, Score overfit 0.62, throughput 0.05) | 0.91 |
| 4B arms (X4R 1.25, X4C 0.61, X4K 0.61, E4F-h 2.05, B4F-h 2.04; incl. preflights and readouts) | 6.56 |
| 2B arms (S4R 0.70, S4C 0.40) | 1.10 |
| 0.8B arms (E8F × 3 3.63, soup + CAL698 fits 0.10, B8F × 2 2.45) | 6.18 |
| Controls / labels / diagnostics (node B 1.0 readouts, own-Nox labels, A7g diagnostic) | 0.19 |
| Formal post-key collections (1.0 controls 8K/16K, X4R × 2, X4K × 2, E8F × 3 + soup, B8F × 2, smokes, mlx-diag) | 1.66 |
| **Total** (node A GPU5 2.16; node B GPU0–4 14.35) | **16.51** |

## Findings for other tracks

1. **Node B image `ce895822…` (tagged `decision20-train-fast:host2` /
   `decision20-lux-runtime` there) has no causal-conv1d kernel**: Transformers
   falls back to the reference convolution (warning in the logs). The eval
   track's node B collections (Lux 1.0 frozen, AutoJev-27B, Eikos-27B,
   Jebadiah-27B) used it, so the node A vs node B answer differences
   attributed to "below the software stack" coincide with a software
   difference. `dbe5f32b…` (`decision20-train-fast:latest` on node B) has the
   same versions as node A's `f83b1d10…`, including the kernel.
2. **Shared 2.0 renderer vs native 1.0 runtime:** Nox 1.0 loses 0.78 v3
   points (typed FINAL Noul −26) when read through the 2.0 renderer at 8,192
   tokens; Sol/Eos are unchanged. Decoder-track 4B candidates are compared with
   the adopted native Nox run, which is 0.78 stricter than a same-renderer control.
3. **Development-proxy misses (again):** at 0.8B the official-Base start led
   Eos continuation by +4.4 P and lost by 6.7 on v3; at 4B the retention arm
   raised CSS-pilot H but lowered CSS15 H. Only a formal run separates these.
4. **Full fine-tuning on the full A7 + v1 mixture is the first recipe that
   moves a decoder tier on v3** (Eos 1.0 → +7.2). The 1.0 corpora (A7) plus
   the v1 arms at volume, not objective/readout factors, drive it — the same
   conclusion as the 9B study (TRAIN breadth/volume is the binding limit).
5. **Padded micro-batches are sound in this trainer** (a), and token-budget
   batching runs 3.5–4.7× faster at 2B/4B than one-row micro-batches; the
   0.8B DEV Score collapse is a transfer limit, not a defect (b).
6. **A0s rows of the two excluded A7h families:** A0 was sampled from the
   same natural pool, so 752 A0s rows are Cosmos QA / SQuAD 2.0 answerability;
   the decoder track excluded them everywhere for consistency. Research &
   data: decide whether renumbered A0s versions should drop them too.
7. **Full fine-tuning erodes a strong 1.0 start:** the same recipe on half
   the mixture takes Nox 1.0 from 52.46 to 45.73 on development (DEV Score
   378 → 183) while it lifts the weak Eos 1.0. Every model trained on this
   mixture pulls 3-level DEV Score toward level 0 although the mixture's
   3-level Score labels are balanced (2,860 / 2,630 / 2,793).

## Operational incidents (recorded, none affects a result)

- Node B's M1 image lacked causal-conv1d (found by the new runtime gate; the
  first three node B padding probes stopped at the gate and were rerun on
  `dbe5f32b…`); node B's Qwen3.5-0.8B-Base snapshot lacked weights (completed,
  hash-matched node A); node A's Sol 1.0 snapshot was partial (completed).
- Two mixture builds failed on a partial v1 download (HF CLI treated extra
  include patterns as filenames); rebuilt after a full `v2/*` download,
  byte-identical on both nodes.
- Same-limit controls: attempt 1 failed (`infer_1p0` lacked `--max-items`),
  attempt 2 failed for Sol/Eos (partial snapshot; Eos ships no
  `temperature.json` → fallback to 1.0 added); one relaunch step ran twice
  and the operator then killed the only live chain — the in-flight Nox
  collection survived, Sol/Eos were relaunched (logged in
  `formal/m2/OPERATIONS.log`).
- Node B → workstation copies run at ≈0.5 MB/s; full checkpoints move via the
  private HF staging repo (amendment 2).

## Decisions for the coordinator

1. **0.8B release:** the E8F soup qualifies (v3 50.236, +7.69 [+3.65, +13.32]
   vs Eos 1.0; public231 156 vs 142). Start release engineering on
   `llm-semantic-router/dev2-dec-staging@16c0929a` `m2/E8F-soup/` (profile
   `qwen-full`, 16,384 tokens). Disclosures: typed Score −13, CSS15 H −.021,
   `mlx-diag` Noul −5.0, strong seed dependence (seed 3 alone is −1.46).
   The release pipeline must accept `frozen_checkpoint` calibration reports
   (shared change `0a399c1d9`).
2. **4B and 2B stay on 1.0** for now (all M2 recipes HOLD). Decide whether the
   4B gate compares with the adopted native Nox run (56.470) or the
   same-renderer control (55.689): the renderer alone costs 0.78.
3. **Cross-track:** node B image `ce895822…` lacks causal-conv1d (eval
   track's node B runs used it); the A0s Cosmos QA / SQuAD 2.0 rows question
   for the data track.

## Next step (proposed Milestone 3)

- 0.8B: release support for the E8F soup (one node A GPU for verification);
  a 16K mlx-diag-aware card; then the data-v2 full-M recipe on the soup's
  start with ≥ 3 seeds to address human transfer and multilingual Noul.
- 2B: E8F's recipe at 2B (Sol 1.0 is mid-strength: full fine-tuning on the
  full mixture, ≥ 2 seeds, trust-region soft targets from own Sol to limit
  erosion) — the 0.8B result suggests the lever is data volume with full
  fine-tuning, and the 4B result says a strong start needs a trust region.
- 4B: Nox continuation with full fine-tuning *plus* own-Nox soft replay (the
  9B trust-region result) on data v2 full-M + A7 replay, two seeds; the
  renderer-matched control decision from item 2.
