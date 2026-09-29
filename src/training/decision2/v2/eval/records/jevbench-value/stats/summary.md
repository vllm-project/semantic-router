# JevBench public 231: reliability and validity (stored results, CPU only)

Scope: the 231 public items (easy 48, standard 72, hard 111) as reproduced by `jev_arena/jevbench_public.py`,
across every stored `public231.score.json` (98 runs: 86 stored on node A, 12 copied from node B). Aggregates
only. Per-item data stays on node A. The drivers used are in `drivers/` (01 collect on node A, 02 analyze in
the CPU-only container with no network, 03 local file split). Seed 20260927. There are 5,000 stratified
bootstrap draws, with items resampled within tier, and 1,000 tier×type-stratified split halves.
Files: `runs-index.json` (run → model, totals, hashes), `reliability.json`, `validity.json`.

## 1. Matrix

- All 98 runs: `predictions_sha256` matches the predictions file, `point_argmax_disagreements` = 0 and
  `renormalized` = 0. Every report's public-231 count matches its score file.
- The 98 runs collapse to **66 distinct models**, each with a distinct correctness vector: 6 own Decision 1.0
  models, 6 released 2.0 models, 37 2.0 sibling or candidate arms and 17 peers. Every duplicate run is
  byte-identical, or at least vector-identical, to its canonical run (list in `runs-index.json`).
- Canonical runs: Lux1 = `eval/m1/d1-lux1-autotune-cache`, Nox1 = `eval/m1-adopt/nox1`, Kai1 = `eval/m1-adopt/kai1`
  (1K limit, 44 invalid), AutoJev, Eikos and Jebadiah = `eval/m4/nodeB-kernel/*`, the released 2.0 models are the
  scored soups (the `release/*` copies are identical), and F1 = node-B `27b/M3-A-soup/formal`.
- Invalid answers occur only for Kai1 and Lex1 (44 each, at the 1K limit), GLiNER2.5 (56), This-That (37) and
  Jebadiah (36). All other runs have strict_valid = 231.

One row per family (lineage: own1 = own Decision 1.0, rel2 = released 2.0, peer = external model). CI is the
95% stratified bootstrap interval of the public total. Residual is the public total minus the fit on v3
(public ≈ 60.7 + 2.01·v3, n = 29).

| Model | Size | Lineage | Public | E/S/H | 95% CI | v3 | T | H | Choice/Noul/Score | C1 | mlx | Residual |
| --- | --- | --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| Decision 1.0 Kai | 0.6B | own1 | 114 | 45/39/30 | [102, 127] | 35.94 | .362 | .357 | .346/.505/.245 | 17.77 | .454 | −18.9 |
| Decision 1.0 Lex | 0.6B | own1 | 113 | 42/42/29 | [100, 126] | 31.02 | .362 | .266 | .294/.502/.268 | 20.82 | .369 | −10.0 |
| Decision 1.0 Eos | 0.8B | own1 | 142 | 48/55/39 | [130, 154] | 42.55 | .393 | .461 | .394/.512/.300 | 37.94 | .665 | −4.2 |
| Decision 1.0 Sol | 2B | own1 | 161 | 48/66/47 | [150, 172] | 45.58 | .422 | .492 | .458/.551/.385 | | .710 | +8.8 |
| Decision 1.0 Nox | 4B | own1 | 173 | 48/66/59 | [161, 184] | 56.47 | .614 | .519 | .690/.816/.445 | | .795 | −1.1 |
| Decision 1.0 Lux | 9B | own1 | 183 | 48/67/68 | [172, 194] | 65.81 | .776 | .558 | .889/.880/.568 | | .828 | −9.9 |
| DEV2.0-0.6B | 0.6B | rel2 | 142 | 48/54/40 | [130, 154] | 43.54 | .395 | .480 | .319/.552/.333 | 33.21 | .615 | −6.2 |
| DEV2.0-0.8B | 0.8B | rel2 | 156 | 48/63/45 | [144, 168] | 50.24 | .573 | .440 | .661/.764/.268 | 40.24 | .652 | −5.6 |
| DEV2.0-2B | 2B | rel2 | 171 | 48/66/57 | [160, 182] | 53.44 | .543 | .525 | .556/.709/.438 | | .708 | +3.0 |
| DEV2.0-4B | 4B | rel2 | 171 | 48/67/56 | [160, 182] | 63.15 | .688 | .580 | .728/.917/.463 | | .770 | −16.5 |
| DEV2.0-8B | 9B | rel2 | 178 | 48/66/64 | [167, 189] | 67.74 | .811 | .566 | .920/.894/.615 | | .822 | −18.7 |
| DEV2.0-27B F1 | 27B | rel2 | 198 | 48/71/79 | [188, 208] | 67.21 | .787 | .574 | .785/.894/.792 | | .828 | +2.3 |
| GLiNER2.5-Decide | 0.6B | peer | 116 | 48/46/22 | [104, 128] | 42.52 | .410 | .441 | .391/.627/.220 | 22.82 | .520 | **−30.1** |
| Bosun 0.6B | 0.6B | peer | 133 | 47/50/36 | [121, 146] | 38.52 | .434 | .342 | .456/.676/.207 | 35.03 | .601 | −5.1 |
| Bosun 1.7B | 2B | peer | 151 | 46/62/43 | [139, 163] | 42.12 | .467 | .380 | .519/.616/.247 | | .721 | +5.7 |
| JPT-0.8B | 0.8B | peer | 171 | 47/60/64 | [159, 183] | 40.09 | .430 | .374 | .499/.578/.253 | 39.01 | .690 | **+29.8** |
| JPT-4B | 4B | peer | 203 | 48/68/87 | [193, 212] | 54.56 | .626 | .476 | .824/.721/.400 | | .786 | **+32.7** |
| JPT-9B | 9B | peer | 197 | 48/68/81 | [187, 207] | 60.99 | .749 | .497 | .894/.820/.568 | | .810 | +13.8 |
| Decider 2B | 2B | peer | 175 | 48/64/63 | [164, 187] | 49.50 | .583 | .420 | .681/.621/.333 | | .742 | +14.9 |
| Decider 4B | 4B | peer | 192 | 48/71/73 | [182, 202] | 61.88 | .689 | .555 | .938/.779/.285 | | .806 | +7.0 |
| Intern 0.8B | 0.8B | peer | 164 | 47/57/60 | [152, 176] | 43.53 | .496 | .382 | .619/.539/.220 | 34.58 | .583 | +15.9 |
| Kev 0.8B | 0.8B | peer | 147 | 48/58/41 | [135, 159] | 43.22 | .479 | .390 | .611/.509/.205 | 39.06 | .648 | −0.5 |
| This-That 1.2 | 2B | peer | 147 | 48/64/35 | [136, 158] | 46.11 | .525 | .405 | .661/.539/.198 | | .742 | −6.3 |
| Jet v6.2 | 4B | peer | 174 | 48/69/57 | [163, 185] | 60.38 | .678 | .537 | .900/.800/.312 | | .796 | −8.0 |
| Nimble v2 | 9B | peer | 185 | 48/68/69 | [175, 196] | 62.06 | .719 | .536 | .980/.818/.268 | | .763 | −0.3 |
| Hopper (G) | 4B | peer | 194 | 48/70/76 | [184, 204] | 58.40 | .644 | .530 | .880/.738/.340 | | .786 | +16.0 |
| AutoJev-27B | 27B | peer | 201 | 48/72/81 | [192, 210] | 72.13 | .887 | .587 | 1.00/.921/.705 | | .842 | −4.6 |
| Eikos-27B | 27B | peer | 212 | 48/72/92 | [204, 219] | 69.29 | .818 | .587 | .979/.736/.840 | | .819 | +12.2 |
| Jebadiah-27B | 27B | peer | 176 | 48/70/58 | [165, 187] | 65.47 | .742 | .578 | .771/.900/.625 | | .832 | −16.2 |

## 2. Reliability

**(a) Per-model 95% CI of the public total.** The median half-width is **±11 items** (range 7.5 to 13), and the
median bootstrap SE is 5.7 items.

**(b) Item statistics across the 66 distinct models.**

| Tier | Mean p | All right | All wrong | Items with 0 < p < 1 | Items with .1 ≤ p ≤ .9 | Median point-biserial | Negative point-biserial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| easy (48) | .995 | 37 | 0 | 11 | 0 | .31 | 0 |
| standard (72) | .882 | 5 | 0 | 67 | 19 | .31 | 1 |
| hard (111) | .512 | 4 | 2 | 105 | 86 | .37 | 9 |
| all (231) | .728 | 46 | 2 | 183 | 105 | .31 | 10 |

The easy tier is saturated: 58 of 66 models score 48/48, and only 8 score between 42 and 47. The effective
panel is about 105 items, 86 of them from the hard tier. Among the 29 family representatives, 126 items
fall between .1 and .9.

**(c) Split-half reliability and KR-20.**

| Pool | n models | Split-half Pearson | Split-half Spearman | Spearman–Brown | KR-20 | Between-model SD / bootstrap SE (items) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| All distinct | 66 | .93 | .91 | .96 | .96 | 21.7 / 5.7 |
| Family representatives (own 1.0, released 2.0, peers) | 29 | .95 | .94 | .97 | .97 | 27.1 / 5.7 |
| 0.6B pool | 13 | .87 | .74 | .93 | .91 | 14.8 / 6.1 |
| 0.8B pool | 11 | .59 | .61 | .74 | .70 | 8.3 / 6.0 |
| 4B pool | 19 | .83 | .22 | .91 | .88 | 8.9 / 5.7 |
| 9B pool | 9 | .68 | .39 | .81 | .75 | 5.7 / 5.6 |
| 27B pool | 9 | .83 | .65 | .91 | .89 | 9.5 / 5.0 |
| Own 2.0 arms, 0.6B | 9 | .21 | .29 | .35 | .21 | 3.9 / 6.1 |
| Own 2.0 arms, 0.8B | 7 | .27 | .31 | .42 | .30 | 4.4 / 5.9 |
| Own 2.0 arms, 4B | 14 | −.11 | −.22 | <0 | −.35 | 2.0 / 5.7 |
| Own 2.0 arms, 9B | 6 | .08 | .07 | .14 | −.06 | 2.1 / 5.6 |
| Own 2.0 arms, 27B | 6 | .16 | .20 | .28 | .13 | 3.1 / 5.0 |

**(d) Test–retest.** Same weights produce 0 flips in almost every pair: Lux1 across all 7 runs (d1, d2, r1,
node B r4, node-B kernel, same-renderer 16K, 8K), Kai repeat, Eikos old vs kernel, all release t1-derived
copies, C0 same-panel vs 8K, and Nox1 and Eos1 at 8K vs 16K. The exceptions are small:

- AutoJev node A vs node-B kernel: 1 flip (200 → 201).
- Jebadiah old vs kernel: 1 flip (177 → 176).
- Sol1 8K vs 16K: 1 flip (161 → 160).
- Kai1 1K vs 8K: 13 flips (114 → 127), all from the 44 items that are invalid at the 1K limit.

Run-to-run noise is at most 1 item. The ±11-item uncertainty comes from item sampling, not the runtime.

**(e) Paired minimum detectable difference.** The MDD is the smallest gap an exact McNemar test (two-sided
α = .05) would detect with 80% power, given the observed discordant count b + c. Medians are over all
same-size-tier pairs of distinct models.

| Size tier | Pairs | Median discordant | Median MDD, all pairs | MDD, non-sibling pairs | MDD, 2.0 sibling pairs | Pairs with p < .05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6B | 78 | 43.5 | 19 | 22 | 14 | 34 |
| 0.8B | 55 | 47 | 20 | 21 | 15 | 11 (0 of 21 sibling) |
| 2B | 10 | 39 | 18 | 18 | — | 6 |
| 4B | 171 | 14 | 11 | 17 | 9 | 50 (2 of 91 sibling) |
| 9B | 36 | 14.5 | 11 | 14.5 | 9 | 8 (0 sibling) |
| 27B | 36 | 18.5 | 12 | 13 | 11 | 16 (0 sibling) |

The typical within-tier MDD is **about 11 to 20 items**: 17 to 22 items between lineages and 9 to 15
between 2.0 siblings. The observed gaps between 2.0 siblings have a median of 2 items, so the panel cannot
rank sibling arms.

**(f) Focus pairs.** b counts items only the first model gets right, c counts items only the second gets right,
and Diff CI is the paired stratified bootstrap. Power 5 is the power to detect a true 5-item gap. N needed is
the panel size that gives 80% power for that gap at the observed discordance rate (exact McNemar).

| Pair | Totals | b | c | McNemar p | Diff 95% CI | MDD | Power 5 | N needed |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| DEV2.0-8B vs Lux1, same renderer or adopted (identical vectors) | 178 vs 183 | 2 | 7 | .18 | [−11, +1] | 8 | .37 | 629 |
| DEV2.0-4B vs Nox1, adopted or 16K (identical vectors) | 171 vs 173 | 5 | 7 | .77 | [−9, +5] | 9 | .27 | 809 |
| DEV2.0-4B vs Decider 4B | 171 vs 192 | 6 | 27 | .0003 | [−32, −10] | 17 | .11 | 2,433 |
| DEV2.0-4B vs JPT-4B | 171 vs 203 | 7 | 39 | <.0001 | [−44, −20] | 21 | .07 | 3,353 |
| DEV2.0-4B vs Jet v6.2 | 171 vs 174 | 15 | 18 | .73 | [−14, +8] | 17 | .11 | 2,433 |
| DEV2.0-4B vs Hopper (G) | 171 vs 194 | 9 | 32 | .0004 | [−35, −11] | 19 | .08 | 3,027 |
| F1 vs AutoJev-27B | 198 vs 201 | 6 | 9 | .61 | [−11, +5] | 11 | .21 | 1,102 |
| F1 vs Eikos-27B | 198 vs 212 | 2 | 16 | .0013 | [−22, −6] | 12 | .16 | 1,317 |
| F1 vs Jebadiah-27B | 198 vs 176 | 28 | 6 | .0002 | [+11, +33] | 18 | .08 | 2,477 |

The "178 vs 183" gap is not a detectable difference. Only 9 items differ, the test would need a true gap of
8 items for 80% power, and a real 5-item gap would be detected 37% of the time. Getting 80% power for a
5-item gap at this discordance rate would take about 630 items. The same applies to DEV2.0-4B vs Nox1 (171 vs 173).
The deficits against Decider 4B, JPT-4B, Hopper and Eikos are real on this panel.

## 3. Validity

Spearman ρ and Pearson r, with 95% bootstrap-over-models CIs, use the public-231 total. "Partial" controls
for log10 loaded parameters. "Within tier" is the pooled Pearson r after demeaning by size tier.

| Set | Criterion | n | ρ [CI] | r [CI] | Partial | Within tier |
| --- | --- | ---: | --- | --- | ---: | ---: |
| All distinct | v3 | 66 | .83 [.74, .89] | .84 [.76, .90] | .29 | .27 |
| | T / H | 66 | .83 / .75 | .80 / .77 | .25 / .18 | .21 / .21 |
| | Choice / Noul / Score | 66 | .83 / .69 / .73 | .80 / .70 / .71 | .41 / .06 / −.04 | .33 / −.10 / .10 |
| | log10 params | 66 | **.91** [.84, .95] | .88 | — | — |
| | mlx-diag | 51 | .89 [.80, .93] | .89 | .59 | .56 |
| | C1 | 10 | .77 [.19, .99] | .86 [.60, .97] | .58 | .63 |
| Family representatives (ii) | v3 | 29 | .85 [.67, .93] | .85 [.73, .93] | .35 | .19 |
| | T / H | 29 | .89 / .74 | .85 / .75 | .34 / .22 | .26 / .06 |
| | log10 params | 29 | .90 | .84 | — | — |
| | mlx-diag | 29 | .86 | .88 | .61 | .32 |
| (ii) without JPT and Hopper (iii) | v3 | 25 | **.93** [.81, .97] | .91 [.84, .96] | .54 | .54 |
| | log10 params | 25 | .92 | .88 | — | — |
| | mlx-diag | 25 | .89 | .89 | .59 | .32 |

By tier, over all distinct models:

- The standard tier tracks C1 best (ρ .93, r .95, partial .87) and mlx-diag (r .93), and it has the
  highest within-tier correlation with v3 (.51).
- The hard tier has no within-tier validity against v3 (r .06, partial .08), and its partial correlation
  with C1 is about 0 (−.09).
- The tier-macro score behaves like the total, slightly stronger.

Most of public-231's correlation with the Arena is shared model size. Within a size tier its agreement with
v3 is weak (r .19 to .27), and it is moderate only when JPT and Hopper are removed (.54).

## 4. Information beyond v3

The fit public ≈ v3 gives R² .70 over all 66 models (residual SD 12 items), .72 over the 29 family
representatives (SD 14.6) and .83 once JPT and Hopper are excluded (SD 11.5).

- **±2 SD outliers.** Over all 66: JPT-4B +33 (z 2.8), JPT-0.8B +28 (2.3), 27B C0 +26 (2.2) and GLiNER2.5 −32
  (−2.7). Over the 29 family representatives: JPT-4B (+2.2), JPT-0.8B (+2.0) and GLiNER (−2.1). Below 2 SD but
  notable: Hopper +16, Intern +16, Decider 2B +15, JPT-9B +14 and Eikos +12. On the negative side, DEV2.0-8B −19,
  DEV2.0-4B −17, Jebadiah −16 and Kai −19.
- **Component tracking.** Over the 29, T+H gives R² .73 with standardized β_T .71 and β_H .17. The partial
  correlation of public-231 with T given H is .61, and with H given T it is .18. The hard tier tracks T
  (.62 given H vs −.01 for H given T). The easy tier tracks only H (.59; this rests on the few models below 48, mainly Kai and Lex).
  The standard tier tracks both (.46 and .39). Public-231 is essentially a T-type (Choice-heavy) panel.
  Its partial correlation with Choice given log params is .41 to .57, versus about 0 for Noul and Score.
- **Relation to mlx-diag.** The public residual correlates with the mlx residual (mlx ~ v3): r .57 [.34, .75]
  over n 51 and .55 over n 29. Public-231 and mlx-diag share size-independent signal that v3 misses.
- **Relation to C1 (n = 10).** Public vs C1 gives r .86, versus .68 for v3. The partial correlation r(public,
  C1 | v3) is .77 (p .016). In leave-one-out prediction of C1, the RMSE is 6.76 from v3, 5.13 from public and
  4.84 from both (Q² .28, .58 and .63).
  - This gain rests on Kai, Lex and GLiNER, whose invalid answers hurt both panels. Without them (n 7), public vs
    C1 falls to r .36 [−.45, .88], the partial to .30 (p .56), and adding public makes the leave-one-out fit worse.
  - Within the 0.8B event, ρ(public, C1) is −.10: JPT-0.8B scores 171 and Intern 164, yet their C1 is 39.01 and 34.58.
  - Treat the C1 gain as unproven.

## Bottom line (statistics only)

- Public-231 is internally consistent across the full range of model sizes (KR-20 .96, run-to-run noise at
  most 1 item). The consistency comes from about 105 informative items, mostly in the hard tier.
- It does not resolve differences within a size tier. The MDD is 11 to 20 items against a CI half-width of ±11,
  and reliability among our own 2.0 arms is about 0.
- Its correlation with the Arena is largely a size effect (ρ .91 with log params), and it is Choice/T-weighted.
- JPT (all three sizes), Hopper, Intern and Decider score far above their v3 on this panel, and our 2.0 4B and
  8B releases score far below it. That pattern is the "good on Arena, poor on JevBench" gap.
- The 4B and 8B gaps to their own 1.0 bases (−2 and −5 items) are within noise.
