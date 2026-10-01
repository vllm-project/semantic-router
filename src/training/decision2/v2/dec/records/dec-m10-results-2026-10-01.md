# Decoder Milestone 10 — results (4B readout × initialisation probe; 2026-10-01)

Preregistration [`dec-m10-prereg-2026-10-01.md`](dec-m10-prereg-2026-10-01.md) (`2dea44d6f`); data lock
[`dec-m10-datalock-2026-10-01.md`](dec-m10-datalock-2026-10-01.md) (`6a8092977`); amendments
[1](dec-m10-amendment-1-2026-10-01.md) (`04088322b`, LT → LT2 at 8,448 tokens) and
[2](dec-m10-amendment-2-2026-10-01.md) (`88aacb9fb`, two gate waves, shared C0 reference); state
[`m10-state.md`](m10-state.md); hand-offs [`dec-m10-handoff-2026-10-01.md`](dec-m10-handoff-2026-10-01.md).
Development readouts are never release scores; v3 / public 231 are post-key same-panel comparisons. Nothing was
uploaded; C1 was not opened. **Interim:** NT2 (the optional second wave) is still training; its wave-2 gates and the
final GPU-hours follow in this record.

## Bottom line

- **A 4B successor candidate: `m10-4b-LH`, a rank-128 LoRA on Qwen3.5-4B-Base with the candidate head, trained on the
  released 4B mixture (matched tokens; uniform soup of three merged seeds).** Post-key v3 **67.34** vs DEV2.0-4B's
  63.15: **+4.19 [+0.10, +9.88]**. It passes successor items 1–7; item 8 (C1 post-key) and the release package are
  hand-offs.
  - vs Nox 1.0 (adopted) +10.87 [+5.00, +14.67]; **vs Decider 4B +5.46 [+0.28, +8.11]; vs Jet v6.2 +6.97 [+0.98,
    +11.13]** — the first 4B candidate significantly ahead of both peers.
  - T .814 vs .688 (typed FINAL C / N / S 702 / 743 / 258 vs 582 / 734 / 185); H .557 vs .580 (−.023, CI [−.082,
    +.074], not significantly below); public 231 172 vs 171; mlx-diag card-eligible **+.0377 [+.0256, +.0498]**.
- **H1 (initialisation) is supported.** Every base-start arm keeps the base's knowledge / maths; DEV2.0-4B does not.
  On the retention probes (MMLU validation + dev, ARC validation, a GSM8K train hold-out; Index-suite and TRAIN
  overlaps removed), the probe macro is base .762, **C0 .710**, LH .770, LT2 .769, FB .787: the released Nox lineage
  sits .052 [.035, .067] below its own base, and the base-start arms recover all of it (LH / LT2 level with the
  base, FB +.025 above it). GSM8K moves most: C0 .515 vs base .637, LH .637, FB .673.
- **H2 (readout) is not supported.** At the same start and adapter, the label-token readout adds nothing: LT2 − LH
  retention −.001 [−.013, +.010], HT-DEV v2 −.010 [−.025, +.006]; LT2 alone FLAGs vs C0 (−.024). The native
  `label_token` runtime path exists and passes parity, so a stock-vLLM generate-mode serving path is available, but
  it is not a quality lever at 4B.

## Design as run

Arms (one pass over the same 58,739-row TRAIN, `c385406e…`, own-Lux KL 1.0 where the recipe has it, N4XF batching,
seeds 20260926 / 27 / 28; 787 updates per seed):

| Arm | Start | Trained | Readout | Seeds' BEST (SELECT700) | Soup |
| --- | --- | --- | --- | --- | --- |
| C0 | DEV2.0-4B (N4XF soup, `df602ca9…`) | — | head | — | — |
| LH | Qwen3.5-4B-Base | LoRA r 128 / α 256, LR 1e-4; fresh head (shared init), head LR 1e-4 | head | 787 / 777 / 785 (.895 / .890 / .905) | merged FP32 soup |
| LT2 | Qwen3.5-4B-Base | the same LoRA, `--max-length 8448` (amendment 1); 791 / 774 / 783 updates (longer prompts) | label-token | 791 / 677 / 783 (.888 / .909 / .890) | merged FP32 soup |
| FB | Qwen3.5-4B-Base | full FT, backbone LR 2.5e-6, head 1e-4 (shared init) | head | 689 / 680 / 785 (.897 / .906 / .913) | FP32 soup |
| NT2 | Nox 1.0 | the N4XF recipe, `--max-length 8448` | label-token | training (wave 2) | — |

- Every preflight passed (nine seeds plus NT2's three); cross-process drift ≤ 1e-7 on the seeded Triton cache. LT
  (8,192 tokens) stopped at its zero-step: 9 TRAIN rows exceed 8,192 tokens under the label prompt (amendment 1).
- **Paths.** Readouts on node E / F reproduce node B's stored readouts of DEV2.0-4B exactly (C0: 0 differing
  decisions, drift 0.0, five panels), and node F reproduces node E exactly (seven panels). The formal path on node F
  (isolated runner, copies of node B's frozen masters) reproduces the stored bar run exactly (0 differing answers on
  typed FINAL, CSS15, public 231), so the bar stands.

## Development (16K, T = 1, paired against C0 on the M10 path)

| Point | Typed T (C / N / S) | H3 | HT-DEV v2 Δ vs C0 [95% CI] | Score5t check | Retention macro (MMLU / ARC / GSM8K) | Δ vs C0 [95% CI] | Gates |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| Base (label-token, untrained) | .549 (477 / 192 / 210) | .448 | −.092 [−.116, −.069] FLAG | COLLAPSE | .762 (.711 / .939 / .637) | +.052 [+.035, +.067] | (ceiling) |
| **C0** = DEV2.0-4B | .704 (501 / 264 / 362) | .563 | — | clean (.39) | .710 (.662 / .954 / .515) | — | (reference) |
| **LH** | .868 (728 / 290 / 371) | .557 | −.014 [−.031, +.003] TIE | clean (.36) | .770 (.715 / .958 / .637) | **+.060 [+.047, +.072]** | **pass → finalist** |
| LT2 | .880 (696 / 325 / 387) | .528 | **−.024 [−.042, −.006] FLAG** | clean (.44) | .769 (.719 / .961 / .627) | +.058 [+.045, +.071] | fail (HT-DEV v2) |
| FB | .869 (741 / 307 / **342**) | .561 | −.011 [−.027, +.006] TIE | clean (.30) | .787 (.734 / .955 / .673) | +.077 [+.063, +.089] | fail (Score type floor 342 < 350) |

- Contrasts: LT2 − LH HT-DEV v2 −.010 [−.025, +.006], retention −.001 [−.013, +.010]; FB − LH +.003 [−.010, +.015]
  and +.017 [+.008, +.027]; vs the base, every trained arm is an HT-DEV v2 GAIN (+.068 to +.081).
- `hs1-dev` (diagnostic): policy-packet accuracy C0 .609, LH .607, FB .603, LT2 .605.
- Wave-1 rules (`m10/select/4b-finalists-w1.json`): finalists `4b-LH`.

## Formal and successor (finalist `4b-LH`; node F, image `dbe5f32b`, isolated runner; scored on node A)

Package `m10-4b-LH` (staging revision `da0d982f…`; FP32 merged-LoRA soup, 4,208,383,488 parameters; CAL698 16K fit
adopted by the 23:15 rule). Collection 358 s on a copy of `formal/m5/cache-frozen` (`f6d0f920…`; 0 new and 13
changed cache entries, flagged by the receipt); mlx-diag on a copy of `cache-frozen-mlx` (`65d7d38f…`).

| Item | Verdict | Detail |
| --- | --- | --- |
| 1 v3 vs bar (DEV2.0-4B, T = 1 run) | **PASS** | 67.34 vs 63.15, +4.19 [+0.10, +9.88] |
| 2 H vs bar | PASS | −.023, [−.082, +.074] |
| 3 types | PASS | none collapsed (typed FINAL 702 / 743 / 258) |
| 4 mlx-diag card-eligible | PASS | +.0377 [+.0256, +.0498] vs the N4XF reference (full +.0319 [+.0215, +.0425]); overall .802; non-English Noul .763 |
| 5 tier gates | PASS | vs adopted Nox 1.0 +10.87 [+5.00, +14.67]; v3 ≥ 55.7; H vs Decider 4B / Jet v6.2 not below |
| 6a exposure of new training files | PASS | 0 groups (`exposure-m10-4b-base.json`, `d42e69ca…`; payload `2194716a…`) |
| 6b rules 1 and 5 without flagged items | PASS | bar [+0.11, +9.75]; 4 / 4 stored pairs reproduced |
| 7 public 231 (`gates public231`) | PASS | 172 vs 171 (easy 48, standard 67, hard 57 vs 56) |
| 8 C1 v1.2 post-key | **pending** | custodian run on a release-format package (hand-off) |

## Recipe recommendation (preregistered rule)

- **Initialisation: base-init**, which here means the LoRA (LH), the arm that passed every gate and items 1–7. FB
  (low-LR full FT) has the best retention but missed the typed Score floor in development, so it was not formally
  tested.
- **Readout: head** ("neutral" by the rule: LT2 − LH spans 0 on both HT-DEV v2 and retention; LT2 FLAGs vs C0).
- **Other tiers (prior):** 2B, 0.8B and 9B should each try the LH recipe on their own released mixture from their
  Qwen3.5 base (2B and 0.8B: `-Base`; 9B: its base). Expect the largest gain where the released lineage lost the most
  knowledge relative to its base. Watch each tier's recorded failure mode: the 2B Score head; development gains at
  0.8B that did not carry to formal; the 9B paraphrase yes-bias, here read on mlx-diag Noul.
- **IB1:** the IB1 arms (W+IB1 vs W+C) run on the LH recipe once a release-safe IB1 record lands (IB1-r2;
  COORDINATION 13:10).
