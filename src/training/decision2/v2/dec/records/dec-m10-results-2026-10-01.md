# Decoder Milestone 10 — results (4B readout × initialisation probe; 2026-10-01)

Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock
[`dec-m10-datalock-2026-10-01.md`](dec-m10-datalock-2026-10-01.md) (`6a8092977`); amendments
[1](dec-m10-amendment-1-2026-10-01.md) (`04088322b`, LT → LT2 at 8,448 tokens),
[2](dec-m10-amendment-2-2026-10-01.md) (`88aacb9fb`, two gate waves, shared C0 reference) and
[3](dec-m10-amendment-3-2026-10-01.md) (`1a5bec9d9`, NT2's formal refused by a finished runner's lease entry);
state [`m10-state.md`](m10-state.md); hand-offs [`dec-m10-handoff-2026-10-01.md`](dec-m10-handoff-2026-10-01.md).
Development readouts are never release scores; v3 / public 231 are post-key same-panel comparisons. Nothing was
uploaded; C1 was not opened; no Index row was read for selection or tuning.

## Bottom line

- **Two 4B finalists pass successor items 1–7 against DEV2.0-4B. The C1 candidate is `m10-4b-LH`.** It is a
  rank-128 LoRA on Qwen3.5-4B-Base with the candidate head, trained on the released 4B mixture at matched tokens:
  three seeds, merged and souped.

  | Finalist | v3 | vs DEV2.0-4B (63.15) | vs Decider 4B | vs Jet v6.2 | T / H | public 231 | mlx-diag (card-eligible) |
  | --- | ---: | --- | --- | --- | --- | ---: | --- |
  | **`m10-4b-LH`** | **67.34** | **+4.19 [+0.10, +9.88]** | **+5.46 [+0.28, +8.11]** | +6.97 [+0.98, +11.13] | .814 / .557 | 172 | **+.038 [+.026, +.050]** |
  | `m10-4b-NT2` | 65.25 | +2.10 [+0.49, +5.82] | +3.37 [−1.64, +6.92] | +4.87 [+0.81, +8.47] | .746 / .570 | 169 | +.002 [−.008, +.011] |

  The M6 choice rule takes the highest paired lower bound vs the best same-size peer (LH +0.28; NT2 −1.64), which
  gives LH. Item 8 (C1 post-key) and the release package are hand-offs, as is an IX1 Index run of the frozen
  candidate.
- **H1 (initialisation) is supported.** The released Nox lineage has lost knowledge and maths relative to its own
  base; every base-start arm recovers it.
  - Retention probe macro (MMLU validation + dev / ARC validation / GSM8K train hold-out): base .762, **DEV2.0-4B
    .710**, LH .770, LT2 .769, FB .787, NT2 .710.
  - GSM8K: DEV2.0-4B .515 vs base .637; LH .637, FB .673.
  - The Nox-start label-token arm (NT2) keeps C0's retention exactly. The loss belongs to the lineage, not the
    readout.
- **H2 (readout) is mixed.**
  - From the base (development only), the label-token readout adds nothing over the head: LT2 − LH retention −.001
    [−.013, +.010], HT-DEV v2 −.010 [−.025, +.006]. LT2 FLAGged vs C0 (−.024), so it was not formally tested.
  - On the Nox lineage, the N4XF recipe with the label-token readout (NT2) beats the same recipe with the head
    (DEV2.0-4B) formally: +2.10 [+0.49, +5.82], typed FINAL .746 vs .688, H level.
  - By the preregistered rule the readout verdict is "neutral" (both development CIs span 0); the NT2 result
    argues for a formal LT-type test from the base next.
  - The native `label_token` runtime path passes parity, so stock-vLLM generate mode can serve it.

## Design as run

All arms take one pass over the same 58,739-row TRAIN (`c385406e…`), with own-Lux KL 1.0 where the recipe has it,
N4XF batching and seeds 20260926 / 27 / 28.

| Arm | Start | Trained | Readout | Updates | Seeds' BEST (SELECT700) | Artifact |
| --- | --- | --- | --- | --- | --- | --- |
| C0 | — (DEV2.0-4B = N4XF soup `df602ca9…`) | — | head | — | — | — |
| LH | Qwen3.5-4B-Base `1001bb4d…` | LoRA r 128 / α 256 on 248 projections, LR 1e-4; fresh head (shared init), LR 1e-4 | head | 787 | 787 / 777 / 785 (.895 / .890 / .905) | merged FP32 soup |
| LT2 | Qwen3.5-4B-Base | the same LoRA, `--max-length 8448` | label-token | 791 / 774 / 783 | 791 / 677 / 783 (.888 / .909 / .890) | merged FP32 soup |
| FB | Qwen3.5-4B-Base | full FT, backbone 2.5e-6, head 1e-4 (shared init) | head | 787 | 689 / 680 / 785 (.897 / .906 / .913) | FP32 soup |
| NT2 | Nox 1.0 `cde2a68d…` | N4XF recipe (full, 5e-6), `--max-length 8448` | label-token | 791 / 774 / 783 | 692 / 677 / 783 (.898 / .892 / .891) | FP32 soup |

- Every preflight passed (12 seeds); cross-process drift ≤ 1e-7 on the seeded Triton cache. LT (8,192 tokens)
  stopped at its zero-step: 9 TRAIN rows exceed 8,192 tokens under the label prompt (amendment 1). The label prompt
  costs 3.3% more tokens (30.39M vs 29.40M) for the same rows.
- **Paths.**
  - Readouts on nodes E / F reproduce node B's stored readouts of DEV2.0-4B exactly (0 differing decisions, drift
    0.0, five panels), and node F reproduces node E exactly (seven panels).
  - The node-F formal path (`run_same_panel --isolate`, copies of node B's frozen masters) reproduces the stored
    bar run exactly (typed FINAL, CSS15, public 231).
  - LoRA merges agree with their adapters on 128 / 128 SELECT rows (drift ≈ .011, BF16 rounding).

## Development (16K, T = 1, paired against C0 on the M10 path)

| Point | Typed T (C / N / S) | H3 | HT-DEV v2 Δ vs C0 [95% CI] | Score5t check | Retention macro (MMLU / ARC / GSM8K) | Δ vs C0 [95% CI] | Gates |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Base (label-token, untrained) | .549 (477 / 192 / 210) | .448 | −.092 [−.116, −.069] FLAG | COLLAPSE | .762 (.711 / .939 / .637) | +.052 [+.035, +.067] | (ceiling) |
| **C0** = DEV2.0-4B | .704 (501 / 264 / 362) | .563 | — | clean (.39) | .710 (.662 / .954 / .515) | — | (reference) |
| **LH** | .868 (728 / 290 / 371) | .557 | −.014 [−.031, +.003] TIE | clean (.36) | .770 (.715 / .958 / .637) | **+.060 [+.047, +.072]** | **pass (wave 1)** |
| LT2 | .880 (696 / 325 / 387) | .528 | **−.024 [−.042, −.006] FLAG** | clean (.44) | .769 (.719 / .961 / .627) | +.058 [+.045, +.071] | fail (HT-DEV v2) |
| FB | .869 (741 / 307 / **342**) | .561 | −.011 [−.027, +.006] TIE | clean (.30) | .787 (.734 / .955 / .673) | +.077 [+.063, +.089] | fail (Score type floor 342 < 350) |
| **NT2** | .754 (547 / 278 / 382) | .543 | −.006 [−.021, +.008] TIE | clean (.46) | .710 (.681 / .958 / .493) | −.000 [−.011, +.011] | **pass (wave 2)** |

- Contrasts:
  - LT2 − LH: HT-DEV v2 −.010 [−.025, +.006], retention −.001 [−.013, +.010].
  - FB − LH: +.003 [−.010, +.015] and +.017 [+.008, +.027].
  - NT2 − LT2: +.018 [+.001, +.034] and −.059 [−.070, −.047].
  - Every trained arm is an HT-DEV v2 GAIN over the base (+.068 to +.081).
- `hs1-dev` (diagnostic): policy-packet accuracy C0 .609, LH .607, FB .603, LT2 .605.
- Rules: wave 1 (`m10/select/4b-finalists-w1.json`) `4b-LH`; wave 2 (`4b-finalists-w2.json`) `4b-NT2`, which takes
  the free second slot (amendment 2).

## Formal and successor (node F GPU3; image `dbe5f32b`; scored on node A)

| Item | `m10-4b-LH` | `m10-4b-NT2` |
| --- | --- | --- |
| Package (staging revision) / identity | `da0d982f…` / `5fa2a699…`; 4,208,383,488 parameters | `e607a58a…`; 4,205,751,296 parameters (no head) |
| Calibration (23:15 rule) | CAL698 16K adopted (T .621 / .479 / .209) | rejected (T = 1) |
| Typed FINAL C / N / S (bar 582 / 734 / 185) | 702 / 743 / 258 | — (T .746) |
| 1 v3 vs bar | **PASS** +4.19 [+0.10, +9.88] | **PASS** +2.10 [+0.49, +5.82] |
| 2 H vs bar | PASS [−.082, +.074] | PASS [−.035, +.056] |
| 3 types | PASS | PASS |
| 4 mlx-diag card-eligible | PASS +.0377 [+.0256, +.0498] | PASS +.0016 [−.0079, +.0113] |
| 5 tier gates (vs adopted Nox 1.0) | PASS +10.87 [+5.00, +14.67] | PASS +8.78 [+5.15, +11.69] |
| 6a exposure of new files | PASS (0 groups, `d42e69ca…`) | PASS |
| 6b reduced panels | PASS [+0.11, +9.75] | PASS [+0.34, +5.94] |
| 7 public 231 | PASS 172 vs 171 | PASS 169 vs 171 (p .75) |
| 8 C1 post-key | pending (custodian) | pending (not the C1 candidate) |

- LH's CSS15 tasks vs the bar (disclosure): down on ibc −.100, talklife −.050, conv_go_awry −.042, reddit_humor
  −.041, persuasion −.026, wiki_politeness −.023, tropes −.017, raop −.013; up on wiki_corpus +.101, flute +.048,
  media_ideology +.039, mrf +.026, indian_english_dialect +.020, emotion +.009, tempowic +.003. Public 231 hard 57
  vs 56.
- The receipts flag changed autotune entries in the formal caches (LH 13, NT2 13 + 201 new for the label prompts'
  shapes); every answer is deterministic given the persisted cache, which is staged on node A for item 8.
- Choice (`formal/m10/successor/4b-choice.md`): C1 candidate `m10-4b-LH`.

## Recipe recommendation (preregistered rule)

- **4B:** base-init with a rank-128 LoRA and the candidate head (LH). It passed every development gate and items
  1–7, and it is the C1 candidate. The readout verdict by the rule is "neutral". NT2's formal gain (+2.10 at the Nox
  start, same recipe) makes a formal test of the label-token readout from the base the cheapest next 4B lever.
- **2B / 0.8B / 9B (prior):** each tier should try the LH recipe on its own released mixture from its Qwen3.5 base,
  at matched tokens, screened by the same probes and HT-DEV v2. 9B has the most base knowledge to keep; 2B must
  watch its Score head (the M8s Score floor); 0.8B development gains have not carried to formal before.
- **IB1:** W+IB1 vs W+C on the LH recipe once a release-safe IB1 record lands (IB1-r2; COORDINATION 13:10).
- **FB:** the best retention arm (+.077) missed the development Score type floor by 8 items. An informational formal
  run is a coordinator decision; it is not a successor path under this preregistration.

## Incidents

- LT's zero-step stopped on prompt length (amendment 1); NT2's first formal step was refused by a finished runner's
  lease entry before any GPU work (amendment 3, M8's stale-entry clearing added).
- Two remote command blocks ran twice (an mlx pull / score, and a formal-select regeneration). Both steps are
  idempotent or refuse to overwrite; no result changed, and the second runs only logged.
- The early `m10_gpuh.py` also counted receipts copied from node B under `inputs/` (0.014 GPU-h per node); fixed in
  `528efb121` and subtracted below.

## GPU-hours: 13.24 of 120

| Item | GPU-h |
| --- | ---: |
| LH / FB training (3 seeds each, incl. preflights) | 2.984 / 2.654 |
| LT2 / NT2 training (3 seeds each, incl. preflights); LT's zero-step 0.008 | 3.074 / 2.739 |
| Readouts (C0 on both nodes, base, 4 soups; 7 panels each), merges, runtime parity | 1.330 |
| Formal (C0 parity, LH, NT2: CAL fits, smokes, collections, mlx-diag) | 0.450 |
