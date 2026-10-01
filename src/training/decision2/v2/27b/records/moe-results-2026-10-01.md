# 27B MoE milestone (MoE-1): results — FINAL

Preregistration `moe-prereg-2026-10-01.md` (signed `bc82b771e`; amendments 1–4). Gate `moe-gate-2026-10-01.md`.
Development readouts are never release scores; post-key numbers are "post-key same-panel". Nothing was uploaded to
Hugging Face. Decision Index values are private (label "independent provisional 0.2.1 reproduction"); this record
holds only the procedure, row accounting, hashes and GPU-hours of that run.

## Verdict

**No successor and no win over AutoJev-27B.** The Gemma-4-26B-A4B-it soup (`MOE-Git-soup`, rank 64, 25.31B loaded /
3.90B active) scores post-key v3 **68.75** (T .812, H .582) against DEV2.0-27B = A20r's 72.36:
**−3.61 [−6.54, −1.18]**. Human transfer is level (−.002 [−.048, +.036]); the whole loss is typed FINAL (−.084
[−.108, −.062]), mostly Noul (.784 vs .921), and mlx-diag Choice + Noul is also below (−.036 [−.052, −.020]).
Successor items 1 and 4 fail; items 2, 3, 5, 6 and 7 pass. vs AutoJev-27B: −3.39 [−5.60, +0.79].
At the A20r recipe the dense 27B base is the better base for this panel; the MoE base keeps human transfer, not
typed accuracy. DEV2.0-27B stays A20r. The private Decision Index run of the frozen package (IX1 harness, parity
gate bit-identical, dual scoring passed) does not clear the 27B-class frontier bar either, so no release hand-off
is issued (values private).

## Summary

- **Screen:** Gemma-4-26B-A4B-it won; both Qwen3.5-35B-A3B cells were stopped by the proxy rule (8.5 points below the
  dense matched reference, mostly typed-DEV Score and the CSS pilot). Stage B = one Gemma soup (two seeds).
- **Stage B:** the soup passed the development gates (P_dev 75.26, HT-DEV v2 TIE) with T = 1 (CAL698 rejected) and
  failed the formal successor items 1 and 4 (above).
- **Latency:** at batch 1 the Gemma MoE decision is slower than the dense 27B (soup 114.6 vs 83.6 ms p50) although
  ≈ 15% of its parameters are active per token.
- **Gate:** Gemma-4-26B-A4B(-it) and Qwen3.5-35B-A3B(-Base) are Apache-2.0 (Gemma 4's licence link is the verbatim
  Apache License 2.0), ungated, pinned and hash-verified on both nodes; the pinned ROCm image supports both; LoRA fits
  one MI325X.
- **Experts kernel:** eager experts are infeasible for training (≈ 49 s per update Gemma, ≈ 110 s Qwen-MoE);
  `grouped_mm` is as close to an FP32 reference as eager (hidden-state error ratio 1.05 / 0.85 at the median) and is
  pinned for training and inference (amendment 1, which replaced a mis-specified random-head parity rule before any
  training).
- **Cost:** at micro-batch 1 the MoE bases are no cheaper to train than the dense 27B: Gemma 9.9 s per update
  (A20r ≈ 10), Qwen-MoE 15.4–16.9 s. Per-token active compute (≈ 3.8B / 2.9B + head) does not translate into
  training speed on this path.
- **Trainer provenance finding (27B track, no result changes):** the 27B launcher's `python -m` from `/code` shadows
  the vendored BEST368 tar, so A20r and the other 27B arms trained with the repository trainer (their provenance
  hashes equal the mirror's files). The MoE cells use the same byte-frozen trainer through a wrapper (amendment 2).

## Stage P0 (node A GPU3 / GPU5)

| | Gemma-4-26B-A4B-it | Qwen3.5-35B-A3B |
| --- | --- | --- |
| Text parameters (loaded text decoder) | 25,233,141,760 | 34,152,051,328 |
| Active per token (text decoder; + LM head for Qwen: 508.6M) | 3,822,530,560 | 2,946,429,568 |
| LoRA targets / adapter / head (rank 32) | 205 / 37,171,200 / 2,895,360 | 310 / ≈ 42.3M / 2,105,856 |
| Training s per row, eager → grouped_mm | 3.05 → 0.66 | 6.91 → 1.07 |
| FP32-reference hidden-state error, eager / grouped_mm (median, max) | .185 / .193, .338 / .359 | .234 / .198, .349 / .419 |
| Peak allocated (training / T = 1 inference at 32K) | ≈ 99 GiB / ≈ 101 GB | ≈ 131 GiB / pending |

## Stage A cells (seed 1 = 20260926; `a20`; A20r contract)

| Cell | Node / GPU | Admission | One-step (r2) | Reload | s per update | Checkpoint 892 | Screen readout |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| MOE-Git-s1 (Gemma-4-26B-A4B-it) | A / 3 | 56,969 rows, 26,128,868 tokens, max 4,708 (limit 4,736) | pass, 0.078 GPU-h | 0 / 32, 6e-8 | 9.9 | pending (≈ 19:25Z) | pending |
| MOE-Qit-s1 (Qwen3.5-35B-A3B) | A / 5 | 56,969 rows, 25,043,392 tokens (= A20r) | pass, 0.139 | 0 / 32, 6e-8 | 15.4 | pending | pending |
| MOE-Qpt-s1 (Qwen3.5-35B-A3B-Base) | B / 6 | same as Q-it | pass, 0.146 | 0 / 32, 6e-8 | 16.9 | pending | pending |

- The first one-step of each cell failed on an argparse flag (≈ 0.0004 GPU-h each, no model code ran; amendment 2).
- **Dense matched reference** (M4-A20r-s1 checkpoint 892, node B, T = 1, 32K, kernel path, cache `583241fb`):
  P_dev **78.22** (T_dev .9244, H_pilot .6618), H_dev2 **.5634** (vs the A20r soup: −.002 [−.015, +.011], TIE).
  Readout `/data/dev2/runs/27b-moe/readouts/A20r-s1-c892`.

## G1 projection (≈ 300 updates; reconstructed after the fact)

The previous worker stopped before recording G1. From each cell's `train-metrics.jsonl` (mean seconds of updates
1–300): G-it 9.59 s → 9.5 GPU-h per full attempt (cap 15.0), Q-it 15.33 s → 15.2 (cap 20.0), Q-pt 16.21 s → 16.0
(cap 20.0); the worst-case plan (a Qwen winner) stayed ≈ 46–51 of 60 GPU-h. **G1 passed: no budget stop was due**, so
no preregistered action was missed.

## Screen decision (automatic, 2026-09-30 21:58Z; node B `/data/dev2/runs/27b-moe/screen/SCREEN.json`)

All readouts at checkpoint 892, T = 1, 32K, on node B GPU7; HT-DEV v2 deltas are paired against the dense matched
reference's own predictions (`readouts/A20r-s1-c892`, SHA-256 `a809af48…`).

| Cell | P_dev | T_dev | H_pilot | H_dev2 (Δ vs reference, 95% CI) | Choice / Noul / Score | Proxy gap | Outcome |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- |
| Dense matched reference (A20r-s1) | 78.22 | .924 | .662 | .5634 | 1.000 / .698 / 1.000 | — | reference |
| MOE-Git-s1 (Gemma-4-26B-A4B-it) | 72.82 | .913 | .581 | .5670 (+.004 [−.016, +.022], TIE) | .999 / .743 / .910 | 5.40 | **continues; gets seed 2** |
| MOE-Qit-s1 (Qwen3.5-35B-A3B) | 69.69 | .854 | .569 | .5528 (−.011 [−.029, +.008], TIE) | 1.000 / .670 / .745 | 8.53 | stopped (rule 3) |
| MOE-Qpt-s1 (Qwen3.5-35B-A3B-Base) | 69.75 | .867 | .561 | .5739 (+.010 [−.008, +.028], TIE) | 1.000 / .735 / .733 | 8.46 | stopped (rule 3) |

- **Gemma-4-26B-A4B-it won the screen.** Both Qwen cells were stopped by **rule 3** (P_dev at least 8 below the best of
  the cells and the dense reference, 78.22). No cell tripped rule 1 (collapse: no type at chance, top answer shares
  ≤ .79, no invalid answers) or rule 2 (every HT-DEV v2 verdict TIE); rule 4 had nothing left to choose.
- **Where the Qwen cells lost:** typed-DEV Score (.745 / .733 vs 1.000) — they almost never answered the middle Score
  level (Q-it 5 of 400, Q-pt 0, gold 107) — and the CSS pilot (.569 / .561 vs .662). Gemma also lost most of the pilot
  gap (.581) but kept typed DEV (.913). On SELECT700 (the training-side validation) the order was reversed: Q-it .840,
  Q-pt .843, G-it .790 at the same checkpoint.
- **Timeline:** node A relayed Gemma's checkpoint 892 at 19:22Z (read 19:26–19:42Z) and Q-it's at 20:58Z (read
  21:02–21:30Z); Q-pt was read on node B 21:30–21:58Z. `SCREEN.json` landed 21:58:17Z; node B stopped Q-pt (update
  1,058) and node A stopped Q-it (update 1,136) by `STOP` file plus `docker stop` (exit 137), then started
  **MOE-Git-s2** on node A GPU4 at 21:59:37Z (one-step 0.073 GPU-h, reload pass; full attempt from 22:05:42Z).
- At the screen the MoE cells were a quarter-trained adapter with experts frozen (37–42M trainable parameters vs ≈ 233M
  for the dense rank-32 adapter); the dense seed was already within 0.8 of the final A20r soup (78.99).

## Stage B (Gemma-4-26B-A4B-it; amendment 4)

| Seed | Node / GPU | Preflights | Full attempt | BEST (SELECT700 family macro / Brier) | Last update |
| --- | --- | --- | --- | --- | --- |
| MOE-Git-s1 (20260926) | A / 3 | pass (r2) | 16:55:57Z → 03:10:21Z, **10.240 GPU-h**, exit 0 | `checkpoint-0002676` .8281 / .1188 | .8209 / .1211 |
| MOE-Git-s2 (20260928) | A / 4 | pass (one-step 0.073, reload 0 / 32) | 22:05:42Z → 08:28:00Z, **10.373 GPU-h**, exit 0 | `checkpoint-0002676` .8219 / .1212 | .8101 |

- Both seeds selected the same update (2,676 of 3,561). Neither seed had a nonfinite loss or a restart.
- The soup, its readout, CAL698 fit, adoption, development gates, package, formal run, gates, latency and mlx-diag
  ran unattended from mirror `eacb6b85c` (amendment 4). Node A relayed both BEST checkpoints at 08:32:53Z.

### Soup, development readout and gates (node B; development numbers, never release scores)

- **Soup `MOE-Git-soup`** (08:37:45–08:39:20Z, CPU): exact uniform rank concatenation of the two BEST checkpoints,
  rank 64 / α 128, head averaged; 205 projections, max relative error 4.5e-7; model `9165bed7…`.
- **T = 1 readout** (08:39:20–08:54:36Z, GPU7, 32K):

  | | P_dev | T_dev | H_pilot | Choice / Noul / Score (typed DEV) | H_dev2 (Δ vs A20r, 95% CI) |
  | --- | ---: | ---: | ---: | --- | --- |
  | A20r (M5 `A20r-ref`) | 78.99 | .916 | .681 | .995 / .675 / 1.000 | .5655 |
  | **MOE-Git-soup** | **75.26** | **.927** | **.611** | .999 / .790 / .920 | **.5728** (+.007 [−.011, +.026], TIE) |
  | MOE-Git-s1 at update 892 (screen) | 72.82 | .913 | .581 | .999 / .743 / .910 | .5670 |

  CSS pilot by task (soup vs A20r): discourse .611 vs .681, implicit hate .453 vs .514, SemEval stance .687 vs .718.
- **CAL698** (0.048 GPU-h): Choice .465, Noul .381, Score .481. **Not adopted** under the 23:15 rule: CSS-pilot
  ECE worsened (.043 → .128) although typed-DEV Brier (.0609 → .0596) / ECE (.054 → .038) and CSS-pilot Brier
  (.607 → .531) improved. The package binds **T = 1**.
- **Development gates** (`DEVGATES.json`): collapse pass, HT-DEV v2 not FLAG vs A20r, proxy 3.73 below A20r (limit
  8) → **finalist**. Reported only: M5's typed guard (T_dev ≥ A20r − .03 = .886) passes (.927).
- **Frozen package** (08:57:35Z, before any formal collection): `PACKAGE.json` `ffb11e1c…`; decision T = 1; base
  `google/gemma-4-26B-A4B-it@4d7ae498` (tree `b05e076d…`), `grouped_mm`, BOS prompt, 32,768 tokens, rank 64 /
  α 128; **25,310,379,550 loaded / 3,899,768,350 active** parameters.
- **Parameters (safetensors headers):** base text decoder 25,233,141,790 (vision tower not loaded); routed experts
  22,837,985,280, top-8 of 128 active. A rank-32 seed: 25,273,208,350 loaded / 3,862,597,150 active. The rank-64 soup
  adds 74,342,400 adapter + 2,895,360 head: **25,310,379,550 loaded / 3,899,768,350 active (≈ 15% of loaded)**, vs
  A20r's 26,096,775,168 (all active).
- **Latency reference (BF16 decision forward, M1's 64-prompt SELECT roster, node B GPU7):** A20r p50 / p95
  **83.6 / 88.8 ms** (FLA kernels, 52.2 GB resident); a rank-32 Gemma MoE adapter (path check on MOE-Git-s1 checkpoint
  892) **119.7 / 126.6 ms** (50.7 GB). At batch 1 on ≈ 140-token prompts the MoE is ≈ 43% slower per decision than
  the dense 27B on this stack, despite ≈ 15% of the active parameters (the soup's own number comes from the chain).

## Formal run (node B GPU7, post-key same-panel, 32,768 tokens, T = 1 package calibration)

- Smoke (8 items per panel) 08:57:35–09:02Z, then typed FINAL (1,600 prompts, 290 s), CSS15 (6,547, 1,038 s) and
  public 231 (231, 118 s); sealed `674fee4e…`; 0.404 GPU-h (+ smoke 0.069). mlx-diag collection 2,275 rows
  (09:29–09:35Z, 0.107 GPU-h), scored and paired on node A (09:38Z).
- **Scores:** v3 **68.75** = 100·√(T·H), T **.812** (2,000 typed questions: Choice .928, Noul .784, Score .800),
  H **.582**; public 231: 198 correct.

| Run | v3 | T | H | typed Choice / Noul / Score | public 231 |
| --- | ---: | ---: | ---: | --- | ---: |
| DEV2.0-27B = A20r (`M4-A20r-soup/formal`) | 72.36 | .896 | .584 | .941 / .921 / .860 | 203 |
| **MOE-Git-soup** | **68.75** | **.812** | **.582** | .928 / .784 / .800 | 198 |

- **Paired (5,000 draws), MOE-Git-soup − right:**

  | Right | Δ v3 [95% CI] | Δ T [95% CI] | Δ H [95% CI] |
  | --- | --- | --- | --- |
  | DEV2.0-27B (A20r) | **−3.61 [−6.54, −1.18]** | −.084 [−.108, −.062] | −.002 [−.048, +.036] |
  | AutoJev-27B (72.133) | −3.39 [−5.60, +0.79] | −.075 [−.098, −.053] | −.005 [−.038, +.056] |
  | Eikos-27B (69.290) | −0.54 [−3.17, +3.90] | −.006 [−.032, +.021] | −.005 [−.047, +.063] |
  | Jebadiah-27B (65.472) | +3.27 [+0.61, +6.10] | +.070 [+.042, +.098] | +.004 [−.037, +.050] |
  | F1 (`M3-A-soup`, 67.209) | +1.54 [−1.49, +3.81] | | |

- **mlx-diag** (node A, `mlx-paired` vs A20r's node A collection): card-eligible macro (Choice + Noul) .802 vs .838,
  **−.036 [−.052, −.020]** (R4 fail); per type Choice −.012, Noul −.060, Score −.016; type macro −.029 (reported).
- Typed DEV had shown the opposite (T_dev .927 vs .916, Noul .790 vs .675): the soup's Noul is ≈ .78–.79 on both
  panels while A20r's moves from .675 (DEV) to .921 (FINAL), so typed DEV did not predict the formal typed gap.
- **Where the typed loss sits** (typed FINAL by family, soup vs A20r; descriptive): `exception_stack` .618 vs .843
  (**−.225**, 400 questions), `resource_ledger` .800 vs .860 (−.060), `constraint_competition` .855 vs .882 (−.027;
  ECE .271 vs .038), `evidence_join` .975 vs 1.000 (−.025). One family carries most of the gap.

## Successor items 1–7 vs DEV2.0-27B (A20r) and "beats AutoJev" (`gates/VERDICTS-20261001T094035Z.json`, `9474a46e…`)

| Item | Rule | Result | Verdict |
| --- | --- | --- | --- |
| 1 | v3 paired lower bound > 0 | −3.61 [−6.54, −1.18] | **fail** |
| 2 | H not significantly below | −.002 [−.048, +.036] | pass |
| 3 | no type collapsed (`gates types`) | Choice / Noul / Score OK | pass |
| 4 | mlx-diag Choice + Noul not significantly below | −.036 [−.052, −.020] | **fail** |
| 5 | v3 ≥ 64.92; H not below AutoJev / Eikos / Jebadiah | 68.75; H CIs include 0 | pass |
| 6 | no overlap exposure (`a20`) | 0 groups, methods agree | pass |
| 7 | `gates public231` not REGRESSION | 198 vs 203, −5 [−14, +4], OK | pass |
| 8 | C1 post-key (eval custodian) | not requested (items 1–7 fail) | — |
| Beats AutoJev | v3 > 72.133, LB > 0, H not below | 68.75; −3.39 [−5.60, +0.79] | **fail** |

## Latency and parameters (same tools for the soup and A20r)

- **Decision forward, BF16-resident** (`latency.py`, M1's 64-prompt SELECT roster, ≈ 145 tokens, batch 1, node B
  GPU7): MOE-Git-soup p50 / p95 **114.6 / 120.5 ms** (50.7 GB resident) vs A20r **83.6 / 88.8 ms** (52.2 GB, FLA
  kernels): the MoE is ≈ 37% slower per decision at batch 1 with ≈ 15% of its parameters active.
- **Formal path, per prompt** (FP32-resident + BF16 autocast, `REPORT.json`): typed FINAL p50 / p95 128.4 / 133.7 ms
  vs A20r 118.5 / 145.1; CSS15 132.8 / 185.4 vs 103.8 / 232.6; public 231 133.3 / 281.7 vs 106.2 / 414.8. The MoE
  is slower at the median and faster in the long-prompt tail.
- **Parameters** (safetensors headers): 25,310,379,550 loaded (base text decoder 25,233,141,790 + rank-64 adapter
  74,342,400 + head 2,895,360), **3,899,768,350 active** per token (non-expert + top-8 of 128 experts); A20r
  26,096,775,168 (all active).

## Private Decision Index run (IX1 harness; values private)

- **Engine:** `v2/27b/moe/index_engine.py`, a kit engine over the frozen package through its formal run's native
  path (`training.model.infer.run_prompts` behind the package's checkpoint fingerprint and frozen calibration;
  FP32-resident, BF16 autocast); responses pass IX1's `check_response`. Launcher `moe-index.sh`; kit `87d4650b`;
  IX1's panel builder on node A (120,226 rows, run-ID digest `6455d7be…` = IX1's). The frozen package was offered
  by node B as soon as the chain froze it and staged on node A over the temporary link (SHA-256 list verified).
- **86-request parity gate: PASS** — the package's formal collector (`v2.27b.moe.collect`) vs the kit runner with
  the engine, separate processes, one frozen Triton cache: 86 / 86 `ok`, 419 questions, **max |Δp| 0.0** (0.065
  GPU-h).
- **Memory limit of the native path** (path check on synthetic rows): one MI325X fits 16 questions × 16K tokens
  (257K padded tokens) but not 20 × 16K or 32 × 16K, and the kit halts a shard on any device error. A scan of the
  panel under the package's encoder (inputs only): 149.0M tokens, 0 questions over 32,768 tokens, 20 requests of
  ≥ 196,608 padded tokens (ToolRet / BRIGHT 32-question requests). IX1's rule applies: those are taken out of their
  shards and rerun alone afterwards; a request that fails alone is a final error (counted wrong by both scorers).
- **Run** (node A GPU3 / 4 / 5, 09:03:45–11:37Z): three shards (40,010 / 40,302 / 39,894 requests, every one `ok`)
  then the 20 pre-split requests alone (18 `ok`; the two largest 32-question ToolRet requests, ≈ 488K and 751K
  padded tokens, ran out of memory alone and are final errors). Row accounting (IX1 merge): **120,224 `ok` + 2
  `error`**, 0 missing, 0 duplicates, 0 superseded; results `bbfc7a98…`. **Dual scoring: PASS** (port vs kit
  87d4650b, per benchmark ≤ 1.5e-4, headline ≤ 0.01). Public receipt: [`moe-index/receipt-MOE-Git-soup.json`](moe-index/receipt-MOE-Git-soup.json).
- **Outcome (private values):** the package does **not** clear the 27B-class frontier bar, so the hand-off is
  not issued (`moe-handoff-2026-10-01.md`). Headline, areas, per-benchmark gaps and the frontier comparison are
  in the private report (node A private dir and the local private folder), labelled "independent provisional
  0.2.1 reproduction".
- GPU-hours: full run 7.126 (shards 6.575, alone-reruns 0.551), parity gate 0.065, path checks 0.127.

## Licence of a derived release (Gemma terms; from the gate)

- `google/gemma-4-26B-A4B-it@4d7ae498` is **Apache-2.0**: the card says `license: apache-2.0` and its `license_link`
  (ai.google.dev/gemma/docs/gemma_4_license) redirects to the verbatim Apache License 2.0. The repository is ungated
  and ships no `NOTICE` file. Gemma 4 is **not** under the Gemma Terms of Use or the Gemma Prohibited Use Policy
  that cover Gemma 1–3, so no use restrictions pass through and no terms acceptance is needed.
- A derived release (LoRA adapter + head served on the unmodified base, or merged weights) may be Apache-2.0 and must:
  include the licence text (§4(a)); state prominently that the files were modified — here, an adapter and head
  trained on top of the named base (§4(b)); keep any upstream attribution notices (§4(c); none ship); and use
  "Gemma" / "Google" only to describe the origin (§6), so a product name such as DEV2.0-26B-A4B carries no
  trademark. The card names the exact base repository and revision as the direct weight origin.
- Data terms are A20r's (same `a20` mixture, SELECT700, CAL698), unchanged by the base. Rune-26B-A4B and Decider
  35B-A3B were never teachers or weight sources.

## Budget (GPU-h; cap 60)

- **Receipts at 09:45Z: 33.987 finished** — node A 26.151 (P0 0.199, failed launches 0.001, preflights 0.344,
  Q-it full 4.993, G-it s1 full 10.240, G-it s2 full 10.373), node B 7.836 (Q-pt preflights + full 5.148; screen,
  reference and smoke readouts 1.615; path checks 0.167 incl. the formal smoke check 0.069; Stage B 0.906:
  readout 0.253,   CAL698 0.048, formal smoke 0.069, formal 0.404, mlx-diag 0.107, latency 0.025) — **plus the private
  Index run 7.318** (full run 7.126, parity gate 0.065, path checks 0.127). **Milestone total 41.31 of 60 GPU-h.**
- Receipts at 03:15Z: 22.64 finished (+ MOE-Git-s1 full 10.240, path checks 0.069, A20r latency reference 0.030)
  **plus MOE-Git-s2 running (≈ 5.2) → ≈ 27.8.**
- Receipts at 02:30Z: 12.301 finished — probes 0.150, kernel checks 0.049, failed launches 0.001, preflights 0.513
  (G-it s1 0.094, s2 0.089; Q-it 0.161; Q-pt 0.168), Q-it full 4.993, Q-pt full 4.980, readouts 1.615 (dense reference
  0.248, Gemma / Qwen-pt path smokes 0.176, cell screens 0.260 + 0.469 + 0.462) — **plus the two running Gemma seeds
  (≈ 9.6 + 4.4) → ≈ 26.3.**
- At 17:20Z ≈ 1.3: probes 0.150, kernel checks 0.049, failed launches 0.001, preflights 0.49, dense reference
  readout ≈ 0.4, MoE path smokes ≈ 0.15, plus the three cells' running full attempts.
- Projection (amendment 1 plan with measured speeds): the screen decision lands ≈ 21:45Z with the three cells at
  ≈ 4.8 h each (≈ 14.4); a Gemma winner then needs ≈ 5 + 10 (seed 2) + ≈ 4 evaluation → ≈ 36 total; a Qwen winner
  ≈ 10 + 15.5 + 4 → ≈ 46 total.
