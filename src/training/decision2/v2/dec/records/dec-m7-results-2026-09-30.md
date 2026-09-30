# Decoder Milestone 7 — results (round-2 data for DEV2.0-4B and DEV2.0-2B; 2026-09-30)

Preregistration [`dec-m7-prereg-2026-09-30.md`](dec-m7-prereg-2026-09-30.md) (`58e41dd60`); data lock
[`dec-m7-datalock-2026-09-30.md`](dec-m7-datalock-2026-09-30.md) (`ea7540df4`). Development readouts are never
release scores; the formal runs are post-key same-panel comparisons. Nothing was uploaded; C1 was not opened.
Aggregate files (finalists, successor evaluations, HT-DEV v2 diagnostics) are in
[`dec-m7-results-2026-09-30/`](dec-m7-results-2026-09-30/).

## Bottom line

**No successor in either tier. DEV2.0-4B and DEV2.0-2B stand.** Each tier had one development finalist, and both
fail item 1 against the current revision. No finalist passed items 1–7, so there was no C1 candidate and nothing
went to the eval custodian (item 8 not requested). No release hand-off.

| Item | 4B `4b-N7C-b1_2` (½ N7C soup + ½ N4XF) | 2B `2b-S7H-b1` (S7H soup) |
| --- | --- | --- |
| Revision (per-file list) / calibration | `7018e83b…` / CAL698 16K adopted by the 23:15 rule | `54dbcae0…` / T = 1 (CAL698 rejected) |
| v3 / T / H | 61.200 / .656 / .571 | 50.673 / .555 / .463 |
| 1. v3 vs current revision | −1.95 [−3.03, +0.004] **FAIL** | −2.76 [−3.89, +1.90] **FAIL** |
| 2. H vs current revision | [−.024, +.023] not significantly below | [−.081, +.023] not significantly below |
| 3. Types (typed FINAL C / N / S) | OK; 558 / 718 / 173 vs 582 / 734 / 185 | OK; 433 / 616 / 158 vs 445 / 567 / 175 |
| 4. mlx-diag card-eligible | +.0057 [+.0003, +.0116] OK | +.0037 [−.0053, +.0128] OK |
| 5. Tier gates | vs adopted Nox 1.0 +4.73 [−0.07, +7.97] **FAIL**; v3 ≥ 55.7; H vs Decider 4B / Jet OK | vs Sol 1.0 16K +4.89 [+2.64, +10.26]; v3 ≥ 44.5; H vs Decider 2B OK |
| 6. Overlap | (a) 0 exposed groups; (b) rule 1 on reduced panels [−2.97, −0.02] **FAIL** | (a) 0; (b) [−3.79, +2.03] **FAIL** |
| 7. `gates public231` vs the bar | 177 vs 171, OK (p .07); hard 62 vs 56 | 173 vs 171, OK (p .79); hard 59 vs 57 |
| 8. C1 | not requested (fails items 1–7) | not requested |
| vs best same-size peer | Decider 4B −0.68 [−6.67, +2.52]; Jet v6.2 +0.83 [−4.23, +4.17] | Decider 2B +1.17 [−4.19, +5.40] |

**Current revisions.** Items 1–7 compare against the scored T = 1 runs of the released weights
(`runs/release/dev2-4b-t1-derived`, v3 63.151; `runs/release/dev2-2b-t1-derived`, 53.437). The current `main`
revisions DEV2.0-4B `fadbba4f` and DEV2.0-2B `a53cf66a` are BF16 storage copies of those weights with 0 answer
changes on typed FINAL, CSS15, public 231 and mlx-diag (release record `dev2-bf16-storage-2026-09-30.md`), so the
same runs are their bars.

Run directories (node A): `/data/dev2/runs/dec/formal/m7/m7-{4b-N7C-b1_2,2b-S7H-b1}` (+ `-mlx`, `.overlap`), successor
files in `formal/m7/successor/`. 4B collected on node B GPU3 (`dbe5f32b`, copy of `formal/m5/cache-frozen`
`f6d0f920…`, 0 new cache entries), relayed gold-free and scored on node A; mlx-diag on node B vs the node-B N4XF
reference. 2B collected and scored on node A GPU5 (`f83b1d10`, `HIP_FORCE_DEV_KERNARG=1`, copy of the S2T node-A cache
`abdfd687…`, DEV2.0-2B's C1 cache); mlx-diag vs `formal/m3/m3-S2T-soup-mlx`. Each had a CAL698 16K fit and a passing
8-item smoke first.

## Development lines (16K, paired against `I` on the same node, image and limit)

Gate (`ops/m7/m7_rules.py`, 9B rule module unchanged): type floors c_t ≥ c_t,I − 0.03·n_t, family floors
F − 0.10, CSS-pilot H3 ≥ H3_I; no typed-gain requirement; pick = largest passing β. All arms trained three seeds
(no cap block; 4B 1.35 GPU-h per seed, as preregistered).

| Line | β | T | H3 | Proxy | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| 4b `I` (N4XF) | | .704 | .5625 | 61.47 | |
| L-N7H (HS1 + long prose) | 1 / ½ / ⅓ | .749 / .741 / .731 | .5458 / .5513 / .5593 | 62.83 / 62.93 / 63.14 | H3 below `I` at every β |
| L-N7P (PN1-r2 ×3 + filler) | 1 / ½ / ⅓ | .713 / .712 / .711 | .5376 / .5520 / .5574 | 60.85 / 62.13 / 62.28 | H3 below at every β; β 1 Noul 241 < 252 |
| L-N7C (matched-token control) | 1 / ½ / ⅓ | .648 / .703 / .708 | .5634 / .5663 / .5643 | 59.86 / 62.91 / 62.46 | β 1 Choice 440 < 477, Score 338 < 350; **β ½ passes** |
| 2b `I` (S2T) | | .610 | .4278 | 47.14 | |
| L-S7H | 1 / ½ / ⅓ | .683 / .668 / .652 | .4421 / .4388 / .4378 | 49.48 / 49.42 / 48.69 | **β 1 passes** (all three pass) |
| L-S7P | 1 / ½ / ⅓ | .621 / .609 / .608 | .4320 / .4244 / .4251 | 47.72 / 46.31 / 46.24 | Score 172–208 < 245 (and H3 at ½, ⅓) |
| L-S7C | 1 / ½ / ⅓ | .614 / .618 / .621 | .4163 / .4181 / .4200 | 44.80 / 45.76 / 46.36 | Score 178–225 < 245; H3 below |

Finalists: 4B `4b-N7C-b1_2`, 2B `2b-S7H-b1`. The rule module numbers slots in fill order (H, P, C), so with the H
and P lines empty the C line's pick is "slot 1"; the order, and so every tie-break, is the prereg's.

## Diagnostics (read and reported, never selected on)

HT-DEV v2 is a diagnostic here (COORDINATION 04:10: the prereg predates it and no amendment adopted it before the
first development readout at 21:20Z). Every point was collected on the M7 readout path (same node, image, 16K,
T = 1); on it both references score the same as the eval track's own reference collections (N4XF and S2T:
Δ 0.0000, paired CI [0, 0]). `hs1-dev` and PN1 dev are read at β 1 against `I`. PN1 dev's es / fr / ar / ru / ko near rows and Russian name
swaps may carry wrong "no" labels (data amendment 1 §B).

| Point | HT-DEV v2 Δ vs `I` [95% CI], verdict | F1 adopt (ideal .50) | F3 false yes | F1 / F2 / F3 acc. | PN1 yes, PAWS-X six (gold .50) |
| --- | --- | ---: | ---: | --- | ---: |
| 4b `I` | H_dev2 .539 | .775 | .262 | .534 / .609 / .774 | .708 |
| 4b-N7H-b1 | −.011 [−.024, +.001] TIE | .630 | .000 | .644 / .862 / .991 | .759 |
| 4b-N7P-b1 | −.027 [−.039, −.015] **FLAG** | .800 | .250 | .523 / .628 / .769 | .517 |
| 4b-N7C-b1 | −.008 [−.022, +.006] TIE | .779 | .205 | .541 / .628 / .779 | .729 |
| 4b-N7C-b1_2 (finalist) | +.001 [−.009, +.011] TIE | | | | |
| 2b `I` | H_dev2 .459 | .780 | .625 | .466 / .513 / .552 | .925 |
| 2b-S7H-b1 (finalist) | +.002 [−.012, +.015] TIE | .663 | .012 | .597 / .785 / .982 | .926 |
| 2b-S7P-b1 | −.013 [−.026, +.001] TIE | .776 | .429 | .455 / .525 / .601 | .527 |
| 2b-S7C-b1 | −.002 [−.015, +.010] TIE | .745 | .589 | .475 / .529 / .558 | .900 |

- **F1 out of distribution:** the HS1 arms' quote-check gain is in-distribution only (dev-ood 4B .450 vs .456;
  2B .456 vs .441).
- **PN1 dev by family (P arms):** name swaps .465 (both tiers; gold .465), near misses .124 / .156 (gold 0), twins
  .617 / .641 (gold .576), hops .933 / .938 (gold 1.0).
- **mlx-diag (formal):** non-English Noul yes-rate 4B finalist .758 vs N4XF .762; 2B finalist .817.
- **Data effect at matched tokens (β 1, development):** H − C typed T +.101 (4B) / +.069 (2B), CSS-pilot H3 −.018 /
  +.026, HT-DEV v2 −.003 / +.004; P − C typed +.066 / +.007, H3 −.026 / +.016, HT-DEV v2 −.019 / −.011.

## Findings

1. **The round-2 blocks teach their skills and leave human transfer where it was.** HS1 removes the unmet-condition
   false yes (4B .262 → .000, 2B .625 → .012) and moves quote adoption toward .50; PN1-r2 brings the PN1-dev yes-rate
   to the gold rate (.52 / .53). HT-DEV v2 reads every H and C soup as a tie with the incumbent and the 4B P soup as
   a FLAG; the CSS pilot put every 4B H / P point below `I`.
2. **Development typed gains again failed to carry.** 2B S7H: typed DEV +.073 became typed FINAL +.012 (Noul +49,
   Choice −12, Score −17), and human transfer fell (H −.063, n.s.; CSS15 mrf −.074, reddit_humor −.054). The CSS pilot
   (+.014 H3) and HT-DEV v2 (tie) did not foresee the drop, which is within formal noise. 4B N7C β ½: typed DEV level
   with `I` became typed FINAL −.032 (every type down), human transfer level.
3. **More of the released 4B recipe (the matched-token control) is not a lever either.** At β ½ it was the only 4B
   point to pass development; formally it is −1.95 (upper bound +0.004; reduced panels significantly negative),
   though mlx-diag rose (+.0057, significant) and public 231 +6 (hard +6, p .07). This matches M6b's dose result.
4. **Public 231 hard moved only within noise** (2B S7H 59 vs 57; 4B N7C 62 vs 56, p .07). The 4B HS1 arm, aimed at
   Decider 4B's hard-skill lead, never reached formal (CSS-pilot guard).

## Tooling and incidents

- Continuation worker from 23:30Z (the first M7 worker was stopped by the platform; no job was lost). Chains,
  soups and line watchers ran unchanged from `ea7540df4` / `ed568b34d`.
- `5ae4cb9cc` (integration merged first): `ops/m7/m7-formal.sh` (formal wrapper: finalists from the watcher's
  select file, smoke → collection, 2B report and mlx-diag on node A; this track's finished runner lease entries moved
  aside before each step — M6b's leftovers on node B GPU3–4 were moved this way), `ops/m7/m7-htdev2.sh` +
  `m7_htdev2.py` (HT-DEV v2 diagnostic), `m7-relay.sh` (`htdev2-panel`, `htdev2`, `exposure`, `pkg`); 6 new tests.
  The HT-DEV v2 gold-free prompts (`90cd409a…`) were placed in both nodes' decoder panel directory; the gold stayed on
  node A.
- Incident (no GPU time, no data affected): the first HT-DEV v2 relay wrote an empty `weights.json` on node A,
  because an `ssh` in the hash argument read the piped file. The file was removed and the relay fixed (`6672f602c`).

## GPU-hours

| Item | GPU-h | Cap |
| --- | ---: | ---: |
| N7H / N7C / N7P (3 seeds each) | 4.029 / 4.052 / 4.090 | 4.5 each |
| S7H / S7C / S7P (3 seeds each) | 1.897 / 1.918 / 1.967 | 2.8 each |
| Lines, references, diagnostics (4B 1.332, 2B 0.708; HT-DEV v2 0.364 of it) | 2.040 | 2.5 |
| Formal (2 CAL fits, 2 smokes, 2 collections, 2 mlx-diag) | 0.272 | 2.5 |
| **M7 total** | **20.265** | 29.7 |
| M6b + M7 | 20.549 | 30 |

Per-job receipts: `m7/gpuh-node-{a,b}.json`, `m7/lines/<tier>/GPU-SECONDS.jsonl`, `formal/m7/GPU-SECONDS.jsonl`.
