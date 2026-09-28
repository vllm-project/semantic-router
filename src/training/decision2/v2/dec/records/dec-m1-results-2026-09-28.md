# Decoder track Milestone 1 results (0.8B / 2B / 4B), 2026-09-28

Development readouts (SELECT, CAL, typed DEV1600, CSS pilot1430, arm held-out
slices) are never release scores. JevArena v3 and JevBench public 231 numbers are
**post-key same-panel** comparisons. Proxy = `100·sqrt(T·H)`, T = typed-DEV
family-macro accuracy, H = CSS-pilot task-median macro-F1. Intervals are
10,000-draw paired bootstraps over typed generator groups and CSS items; they
do **not** include training-seed variance (see σ_seed).

## Records

C1 [prereg](dec08-c1-factor-screen-prereg-2026-09-28.md) (`1a5241981`),
[amendment 1](dec08-c1-amendment-1-2026-09-28.md) (`d8b3f9137`, matrix v1, N0),
[amendment 2](dec08-c1-amendment-2-2026-09-28.md) (`268568ad8`, A2/A3 stops, A2b/A3b);
C2 [prereg](dec08-c2-template-s-prereg-2026-09-28.md) (`053438443`);
E1 [prereg](dec-t2-e1-prereg-2026-09-28.md) (`03172ca28`) and
[seed amendment](dec-t2-e1-amendment-1-seeds-2026-09-28.md) (`05e9d49e8`);
X2 [formal lock](dec-t2-e1-x2-formal-lock-2026-09-28.md) (`f13c03006`) and
[same-node amendment](dec-t2-e1-x2-formal-lock-amendment-1-2026-09-28.md) (`b91181d0e`).
Code: `v2/dec/` at `11f02b8d1` → `cea7cd7ca` (C1 arms), `edfb5f15c` (A3b),
`a4c6f0000` (E1, post-runs), `053438443` (C2), `8e5f22cd8` (formal collection).

## Start points (verified)

All nine candidates were checked by HF API and hash-verified downloads, and the
0.8B/2B/4B ones loaded through the shared loaders: Eos 1.0 / Qwen3.5-0.8B
Base / general 753,446,208 decision parameters (text 752,393,024); Sol 1.0 /
Qwen3.5-2B 1,883,930,944; Nox 1.0 / Qwen3.5-4B 4,208,383,488. Official Qwen3.5
checkpoints are generative VLMs (a decision head is attached untrained); 1.0
packages ship a trained typed candidate head. Current Hub heads of the 1.0 repos
differ from earlier pinned revisions only in docs/metadata (weights identical).
All Apache-2.0.

## 0.8B: C1 factor screen (own Eos 1.0, rights-clean v2, 466 updates, node A GPU5)

| Arm | T | H | Proxy | Choice / Noul / Score | Δ vs A0 [item CI] | SELECT@BEST | CAL Brier |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| Eos 1.0 (same runtime) | .4956 | .1887 | 30.58 | 510 / 198 / 85 | — | — | — |
| A0 control | .5081 | .2176 | **33.25** | 531 / 197 / 85 | +2.67 vs Eos [+0.63, +4.93] | 577 | .1175 |
| N0 (A0, data-order seed 2) | .4613 | .2228 | 32.06 | 454 / 199 / 85 | −1.19 [−3.30, +0.23] | 573 | .1215 |
| A1 type-balanced loss | .5069 | .2054 | 32.27 | 531 / 195 / 85 | −0.99 [−2.16, +0.28] | 608 | .1030 |
| A2b own-Lux soft targets | .5006 | .2147 | 32.78 | 519 / 197 / 85 | −0.47 [−1.81, +0.76] | 571 | .1189 |
| A3b ordinal Score residual | .5000 | .2145 | 32.75 | 520 / 195 / 85 | −0.50 [−2.00, +0.57] | 573 | .1182 |
| A4 layer-mix readout | .4938 | .2121 | 32.36 | 509 / 196 / 85 | −0.89 [−2.20, −0.19] | 569 | .1190 |

σ_seed (from A0/N0): proxy 0.84, T 0.033, H 0.004. All four factors are
**negative** (Δ < +1.0; none exceeds 2σ_seed). Every arm, including a 4.8×
Score loss weight, a query-predicted ordinal Score prior and a teacher that is
87% correct on TRAIN Score rows, still answers all 400 DEV Score items with
level 0. Best arm A0 (33.25) is below the pre-registered formal threshold
(33.83): **no 0.8B formal candidate**. Stopped at preflight: A2 (two-sided
teacher-fidelity tolerance, teacher not degraded) and A3 (first window had no
Score rows); same-configuration versions A2b/A3b completed.

## 0.8B: C2 matrix wave-1 data factors (template S, node B)

| Arm | ρ | Rows / updates | Proxy | Δ vs control [item CI] | Choice / Noul / Score | AHO (arm) vs control |
| --- | ---: | --- | ---: | --- | --- | --- |
| C1 control | 2.10M | 11,167 / 698 | 32.94 | — | 528 / 192 / 85 | A6g 166/494, A2 281/551 |
| D6g Score all levels (gen.) | 2.10M | 11,679 / 730 | 31.11 | −1.83 [−3.28, +0.91] | 494 / 198 / 85 | 295/494 |
| D2 verifiable families | 2.10M | 12,561 / 786 | 32.03 | −0.91 [−2.72, +0.67] | 490 / 193 / 85 | 329/551 |
| C2 control | 1.44M | 10,009 / 626 | 32.45 | — | 486 / 194 / 85 | A6h 321/646, A1 451/540 |
| D6h Score (human ordinal) | 1.44M | 13,077 / 818 | 31.92 | −0.53 [−2.04, +1.34] | 542 / 200 / 85 | 429/646 |
| D1 cross-domain human | 1.44M | 11,727 / 733 | 32.42 | −0.03 [−1.35, +1.79] | 467 / 199 / 85 | 497/540 |

Every data arm learns its own held-out family (+46 to +129 AHO items) but none
moves typed DEV T, CSS-pilot H or DEV Score beyond seed noise at 0.8B. Under
matrix v1.1 these are negative on T1b; the 0.8B development panel is at floor
on Noul (chance) and Score (constant level), and its Choice swings ±40 items
with the data order, so it cannot resolve these factors.

## 2B / 4B: E1 early arms (C1 configuration, node B)

| Arm | T | H | Proxy | Choice / Noul / Score | Δ [item CI] |
| --- | ---: | ---: | ---: | --- | --- |
| Sol 1.0 (same runtime) | .5919 | .3194 | 43.48 | 414 / 221 / 312 | — |
| S0 control | .5931 | .3503 | 45.58 | 420 / 231 / 298 | +2.11 vs Sol [−0.46, +4.52] |
| S2 + own-Lux KL | .5913 | .3399 | 44.83 | 414 / 231 / 301 | −0.76 vs S0 [−1.86, +0.99] |
| Nox 1.0 (same runtime) | .6681 | .4123 | 52.48 | 462 / 229 / 378 | — |
| X0 control | .6613 | .4585 | 55.06 | 464 / 228 / 366 | +2.58 vs Nox [+0.34, +4.62] |
| X2 + own-Lux KL | .6713 | .4690 | **56.11** | 486 / 228 / 360 | +1.05 vs X0 [−0.36, +2.51]; +3.63 vs Nox [+1.48, +5.61] |

X2 cleared the 4B formal rule and was locked for one post-key run.

Second data-order seed (E1 amendment 1): X0s2 **55.39**, X2s2 **56.35**
(BEST 466 for both). σ_seed(4B) = 0.23 (vs 0.84 at 0.8B). X2 − X0 = +1.05,
X2s2 − X0s2 = +0.97, mean +1.01 > 2σ with both seeds positive: the own-Lux
soft-target factor is **confirmed on the 4B development proxy** (not at 2B:
−0.76; not at 0.8B: −0.47). The formal result below shows that this
development gain does not carry to v3.

## Formal post-key same-panel result (E1-X2, 4B) — HOLD

Collected once on node A GPU5 with the eval track's frozen runner (code
`8e5f22cd8`, image `f83b1d10…`, persisted autotune cache), inference identity
`de41a699…` as locked; seal `034192be…`. Compared with the adopted Nox 1.0 run
(`m1-adopt/nox1`) on the same node:

| Post-key same-panel | E1-X2 | Nox 1.0 | Δ |
| --- | ---: | ---: | --- |
| **JevArena v3** | **55.993** | **56.470** | −0.48, paired 95% CI [−3.09, +1.78] |
| T (typed FINAL family macro) | .5844 | .6144 | −.030 |
| H (CSS15 task-median macro-F1) | .5365 | .5190 | +.017 (without FLUTE: .5305 vs .5081) |
| Choice / Noul / Score (of 800/800/400) | 551 / 616 / 168 | 552 / 653 / 178 | −1 / **−37** / −10 |
| constraint competition / evidence join / exception stack / resource ledger | 151 / 800 / **216** / 168 | 152 / 800 / 253 / 178 | exception stack −37 |
| **Public JevBench 231** (easy/standard/hard) | **174** (48/67/59) | 173 (48/66/59) | +1 |
| Typed Brier / ECE10 | .232 / .140 | .205 / .092 | worse |
| CSS15 invalid (over the 8,192-token budget, all `tropes`) | 18 | 4 | +14 |

The pre-stated reading (v3 gain ≥ +3.0 with lower bound > 0 and public231 ≥
173) is not met: **HOLD**. The +3.63 development gain became −0.48 on v3, the
same direction as earlier own-Sol and official-4B arms: rights-clean v2
continuation improves human transfer (11 of 15 CSS15 tasks up) and loses
typed FINAL reasoning, concentrated in exception-stack Noul. A node B
cross-node repeat of the same package differs on 56 of 8,778 answers (0.64%);
it is not scored. Formal collection GPU time 0.149 GPU-hours (node A) plus
0.158 (node B repeat) and 0.066 of smokes.

## GPU-hours (wall-clock × GPUs, from every job receipt)

| Allocation | Occupancy | Container-hours | Main use |
| --- | ---: | ---: | --- |
| node A GPU5 | 3.95 | 18.95 | C1 (7 arms incl. N0, preflights, Lux teacher labels 0.08, Eos/Lux readouts), X2 formal 0.15 + smoke |
| node B GPU0 | 1.82 | 3.30 | Sol control/readouts, E1 S0/S2, C2 C1/D6g, X2 cross-node repeat 0.16 |
| node B GPU1 | 2.76 | 3.55 | Nox control/readouts, E1 X0, C2 D2/C2, X0s2 |
| node B GPU2 | 2.82 | 3.66 | 2B/4B runtime probes, E1 X2, C2 D6h/D1, X2s2 |
| **Total** | **11.35** | **29.46** | CPU-only builds, loader checks and scoring excluded |

## Operational findings for all tracks

- Concurrent jobs on one GPU trigger repeated Triton autotuning on each new
  sequence shape (single 16-row updates stalled for up to 335 s); a shared
  persistent autotune cache (`DEC_TRITON_CACHE`) or one job per GPU restores
  throughput (C2 with a shared cache ran ~2× faster per GPU than C1).
- Cross-process runs of the same model differ only through autotune choices
  (≤ .03 probability, ≤ 5/700 SELECT flips); in-process parity is exact.
