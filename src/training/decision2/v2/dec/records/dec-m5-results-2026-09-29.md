# Decoder Milestone 5 — results (4B multilingual-Noul successor study; 2026-09-29)

Preregistration [`dec-m5-prereg-2026-09-29.md`](dec-m5-prereg-2026-09-29.md) (`c62cc0853`); amendments
[1](dec-m5-amendment-1-2026-09-29.md) (`e574bbf56`), [2](dec-m5-amendment-2-2026-09-29.md) (`a131563c5`),
[3](dec-m5-amendment-3-2026-09-29.md) (`d0e7f39fc`, written after the N5N readout — disclosed); data lock
[`dec-m5-datalock-2026-09-29.md`](dec-m5-datalock-2026-09-29.md) (parts 1–3, `7132fc7b9`). Development readouts are
never release scores; v3 / public 231 are post-key same-panel comparisons; `mlx-diag` is a diagnostic panel.

## Bottom line

- **No successor.** The 4B release candidate N4XF stands. No arm was a preregistered finalist, and no diagnostic run beat
  N4XF on v3.
- **No arm fixes the multilingual-Noul loss at equal or better v3.** The best, N5BN (diagnostic run, not selected), is
  the multilingual-block recomposition plus own-Nox replay. It keeps v3 (−0.51 [−2.48, +1.47]) and human transfer
  (+0.005 [−0.028, +0.034]). It closes about a third of the non-English Noul gap: `mlx-diag` .752 vs .725, Nox 1.0 .800.
  - Predicted-yes .735 vs .762; gold-No recall .517 vs .463; ko .67 vs .63.
  - That is below the preregistered card-only bar (≥ .764), and non-English Choice slips −.011.
- **Each factor alone lost typed reasoning** and did not move Noul: replay only (N5N) −3.09 v3; recomposition only
  (N5B) −2.90.
- **The 2B port was not triggered** (no finalist, no development Noul fix).

## Arm table

All arms start from own Nox 1.0 with N4XF's recipe, seeds and budget. Rows outside the 14,298-row non-English
multilingual block are identical to N4XF. Artifacts are three-seed uniform soups (soup R ≥ seed-mean R in every arm).
Reference = the N4XF soup.

**DEVELOPMENT** (typed DEV, CSS pilot, MLX-DEV; node B, 8K; selection only):

| Arm | typed-DEV C/N/S | T_dev | P (R) | P_mean3 | P − N4XF [CI] | MLX-DEV Noul-ML | Noul-ML Δ vs N4XF [CI] | pred-yes | M_dev | Selection |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| N4XF soup (control) | 501 / 264 / 362 | .704 | 61.47 | — | — | .853 | — | .530 | .770 | — |
| **N5N** (Nox replay on block) | 492 / 273 / 378 | .714 | 61.81 | 62.69 | +0.34 [−1.18, +1.85] | .842 | −.011 [−.017, −.005] | .541 | .761 | dropped (Noul-ML) → diagnostic |
| **N5B** (recomposed block, Lux incl. H8) | 509 / 256 / 358 | .702 | 61.20 | 62.51 | −0.27 [−1.83, +1.26] | .845 | −.008 [−.015, −.002] | .535 | .774 | dropped (Noul-ML) → diagnostic |
| **N5BN** (recomposed + Nox replay) | 545 / 234 / 342 | .701 | 61.38 | 61.89 | −0.09 [−1.85, +1.65] | .840 | −.013 [−.021, −.006] | .543 | .766 | dropped (Noul-ML) → diagnostic |
| Nox 1.0 (context) | 460 / 228 / 378 | .666 | — | 55.33 | — | .788 | — | .526 | .691 | — |

Selection file `m5/select/select-final.json` (`a11e70d2…`): every arm is eligible and ties on P. Every arm is dropped
by the preregistered Noul-ML criterion, so there are **no finalists**. Amendment 3 gave each arm one
**diagnostic (not selected)** formal run.

**POST-KEY** (JevArena v3 typed FINAL + CSS15, public 231; node-B collection at 16K, node-A scoring; all three runs
are diagnostic, not selected):

| Arm | v3 | Δ vs N4XF ref [95% CI] | T (Δ [CI]) | H | ΔH human transfer [CI] | typed-FINAL C/N/S | public 231 (E/S/H) | Δ vs adopted Nox 1.0 [CI] |
| --- | ---: | --- | --- | ---: | --- | --- | --- | --- |
| N4XF reference | 63.151 | — | .688 | .580 | — | 582 / 734 / 185 | 171 (48/67/56) | +6.68 [+0.99, +9.64] |
| N5N | 60.064 | −3.09 [−4.19, −1.19] | .638 (−.050 [−.069, −.032]) | .565 | −.014 [−.030, +.016] | 552 / 684 / 183 | 174 (48/67/59) | +3.59 [−1.98, +7.19] |
| N5B | 60.248 | −2.90 [−5.36, −0.50] | .633 (−.056 [−.077, −.035]) | .574 | −.006 [−.048, +.029] | 540 / 669 / 203 | 172 (48/67/57) | +3.78 [−1.41, +6.81] |
| N5BN | 62.642 | −0.51 [−2.48, +1.47] | .672 (−.016 [−.038, +.006]) | .584 | +.005 [−.028, +.034] | 586 / 692 / 196 | 173 (48/67/58) | +6.17 [+0.45, +9.26] |

No type collapsed. The top-answer share is at most .55 for every type, and typed-FINAL types are ≥ 75% of Nox 1.0's.

**DIAGNOSTIC** `mlx-diag` (read after each v3 report was sealed; node B, same image and frozen cache as the reference):

| Arm | overall | non-en Choice | non-en Noul | non-en Score | pred-yes (non-en Noul) | gold-No recall | ko (Noul only) | Noul de / es / fr / ja / zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N4XF reference | .770 | .751 | .725 | .811 | .762 | .463 | .63 | .75 / .79 / .75 / .67 / .76 |
| N5N | .777 | .761 | .723 | .825 | .770 | .453 | .61 | .74 / .78 / .76 / .69 / .76 |
| N5B | .777 | .758 | .730 | .820 | .760 | .470 | .62 | .76 / .79 / .77 / .70 / .74 |
| N5BN | .776 | .741 | **.752** | .810 | **.735** | **.517** | **.67** | .77 / .81 / .78 / .71 / .77 |
| Nox 1.0 (node A, context) | .795 | .759 | .800 | .808 | .663 | .637 | .69 | .81 / .87 / .82 / .77 / .84 |

Outcome under the prereg's "Goal and bar": N5N negative, N5B negative, N5BN negative. N5BN's v3 is equal, but its
non-English Noul .752 is below .764 and its non-English Choice is −.011 against the −.01 limit. A diagnostic run
cannot make a card-only claim anyway.

## Findings

1. **The multilingual-Noul loss is a paraphrase-style yes-bias that answerability training cannot see.**
   - On held-out non-English answerability / relevance slices (MLX-DEV), N4XF has no yes-bias: predicted-yes .530 vs
     Nox 1.0 .526, gold .50. It beats Nox 1.0 on Noul-ML (.853 vs .788).
   - On PAWS-X pairs it says "yes" .762 of the time.
   - Every non-English human Noul row we train on is answerability / relevance style. So the in-distribution development
     signal and the diagnostic disagree by construction, and every M5 arm lost a little in-distribution Noul-ML.
2. **Replay alone does not restore Nox 1.0's paraphrase behaviour and costs typed reasoning.** N5N replays own-Nox on
   the block: non-English Noul .723 vs .725, v3 −3.09, typed T −.050.
   - The block is 18.75% of tokens; own-Lux targets there carried typed-reasoning signal.
   - The training-data diagnostic shows own-Lux is somewhat more yes-leaning than own-Nox on block Noul rows (.557 vs
     .516, gold .500). That is not the driver.
3. **Recomposition alone (N5B) also loses typed reasoning** (T −.056, v3 −2.90) with a small Noul change (.730).
4. **The combination (N5BN) holds v3 and moves every language's paraphrase Noul by +.01 to +.04.** That includes ko +.04
   and ja +.04. The yes-rate falls by .027 and gold-No recall rises by .054.
   - Why both single factors lose typed reasoning while the combination does not is unexplained.
   - The paired CIs resample items, not training runs. At 4B, soup-to-soup training variance is of the same order
     (M4: N4LX vs N4XF 2.8 v3), so the arm-to-arm v3 differences here are not attributable to treatment with confidence.
   - N5BN's Noul gain is consistent across all seven languages, but it is one soup.
5. **The development proxy did not separate the arms:** P 61.2–61.8 against a v3 spread of 60.1–62.6. This is the
   proxy-v2 caveat.

## Selection note (amendment 3)

The preregistered Noul-ML drop criterion was mis-specified. It measures the in-distribution answerability trade that
every M5 intervention makes by design (see the prereg's own caveat). Amendment 3 kept the preregistered outcome and
added one diagnostic formal run per arm dropped only by that criterion. It was written after N5N's readout and before
any N5B / N5BN readout. Diagnostic runs make no successor or card-only claim and cannot trigger the 2B port. The
coordinator may accept or reject this use; no decision rests on it.

## Formal-run provenance

- **Reference.** The node-B N4XF soup (hash list `cea4cc9a…`, 14 files, the node-A calibration `176a2f0e…`), collected
  first on a fresh cache on image `dbe5f32b`. It reproduces the node-A formal N4XF run exactly: v3 63.151, public 231
  171, **0 differing answers** on typed FINAL, CSS15 and public 231.
  - The cache was then frozen: `cache-frozen`, 1,654 files, manifest `f6d0f920…`. The mlx master is `cache-frozen-mlx`
    (`65d7d38f…`); six mlx entries were re-tuned there.
  - Node-B `mlx-diag` of the reference: .770 overall and .725 non-English Noul, vs node A's .727. The card-only
    comparisons use the same-node reference.
- **Diagnostic-run packages.** Each is staged by the M4 procedure without upload: 16K CAL698 fit, rejected by the 23:15 rule
  for all three (T = 1), and a per-file hash list used as the revision.
  - N5N `d9decc22…`, N5B `f410a6bf…`, N5BN `3e92da3f…` (15 files each).
  - Seals: reference `dc64166b…`, N5N `e3c02345…`, N5B `06b9c304…`, N5BN `9ee7dfb1…`.
  - Each diagnostic run used a copy of the frozen master. The flagged "changed cache entries" are `__grp__` index files
    whose absolute paths Triton rewrites when it loads a copy. All 13 `.autotune.json` results are identical to the
    master, so the kernels are identical.
- **Aggregates.** Node A `formal/m5/m5-results.{json,md}` (`1b869530…` / `33a7a1d9…`).

## GPU-hours

**≈ 10.2 GPU-h** (cap 15), all on node B GPU3–4:

- **Training side ≈ 9.36:** Nox labels 0.08; 9 seeds' preflights, full runs (49–53 min each) and postruns; three soups
  with readouts; MLX-DEV baselines.
- **Formal ≈ 0.87:** reference, three diagnostic runs, four `mlx-diag` collections, two smokes, three CAL fits.

Formal jobs that shared a GPU with training are counted in full on top of it. Node A GPU5 was not used.

## Failures, deviations, incidents

- **N5B preflight 3 failed** (1,054 added MTOP rows without a target from the prereg's short source list). Fixed by
  amendment 2 (the full M4 own-Lux source list, including RP-v2 waves) before any N5B training.
- **The MLX-DEV segment rule was infeasible as written:** template lines emptied six cells. Fixed by amendment 1 before
  any readout.
- **No preflight of any seed failed; no arm was rerun.**
- **The first reference launch was refused by the runner's idle-GPU check** before any collection; it was relaunched
  with the shared-lease flag.
- **Relay anomaly.** Two pulls (reference `mlx-diag`, N5B `mlx-diag`) found the node-A destination already present;
  the cause was not identified. Both copies were hash-checked before scoring. The reference copy was truncated (2,178
  of 2,275 lines) and was deleted and re-copied, then verified (`bc43b459…`). N5B's matched node B (`33f1f521…`).
- **Lease bookkeeping.** The prep script's idle marker twice briefly marked GPU4 idle during N5N-s2; it was restored
  at once.
- **Mid-milestone installs on node B.** The formal tooling installed the gold-free `mlx-diag` prompts on node B
  (hash-checked installer).

## Items for the coordinator

1. No successor: DEV2.0-4B (N4XF) stands. Its card disclosure of the `mlx-diag` loss stays as is; no 2B / 0.8B work
   follows from M5.
2. **Calibration.** N4XF's staged CAL698 16K temperatures fail the 23:15 rule on the CSS pilot (ECE .063 → .068, 0
   answer changes). Release engineering should confirm DEV2.0-4B ships T = 1.
3. **Formal gold files exist on node B** (`gold/typed-final`, `css15`, `public231`), against "gold lives on node A".
   This is for the eval track; M5 tooling never read them.
4. **Research & data.** The yes-bias is paraphrase-style (high-overlap non-paraphrases called "same"). No admitted
   non-English paraphrase / NLI Noul source exists: PAWS-X / XNLI / MASSIVE are excluded, and esnli-R is a C1
   registry source. A licence-clean, C1-independent multilingual paraphrase / NLI Noul arm is the direct lever and would
   also give a development slice that can see the failure.
5. **Teacher coverage.** Own-Lux XL-r2 coverage spans the RP-v2 waves, the XL waves and `h-w1`. A track composing
   targets for XL-r2 rows needs the full source list (amendment 2).
6. **Eval.** HT-DEV v1's model list has no N4XF; the eval track could add N4XF and the three M5 soups. The
   node-B-image mlx-diag differs from node A's by a few Noul answers (same-node pairing handles it).
7. **Follow-up if wanted.** N5BN is the only lead: a partial paraphrase-Noul fix at equal v3. A replicate soup (new
   seeds) plus a paraphrase-style development slice from an admitted source would test it before any card change.
