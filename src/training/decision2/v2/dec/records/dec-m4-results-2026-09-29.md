# Decoder Milestone 4 — results (4B own-Lux line; 2026-09-29)

**Outcome: a 4B release candidate qualifies — N4XF soup, post-key v3 63.151 vs the adopted Nox 1.0 run
56.470 (+6.68 [+0.99, +9.64]).** Candidate record: [`dec-m4-4b-candidate-2026-09-29.md`](dec-m4-4b-candidate-2026-09-29.md).

Records: [prereg](dec-m4-prereg-2026-09-29.md) `0b8ebfc54`, [amendment 1](dec-m4-amendment-1-2026-09-29.md)
`63de7fe45`, [data lock](dec-m4-datalock-2026-09-29.md) `12af91294` / `42f26c846`,
[selection](dec-m4-selection-2026-09-29.md) `7edb1be89`, [incident](dec-m4-incident-hf-staging-2026-09-29.md)
`f0c78cc99`.

## Arm table

Development readouts are typed DEV + CSS pilot. Formal is post-key same-panel v3 (node A, 16K).

| Arm (3 seeds + soup) | Contrast | Dev R = P_mean3 | Dev T / H_mean3 | Dev typed C / N / S | Formal v3 [Δ vs adopted Nox1] |
| --- | --- | ---: | --- | --- | --- |
| **N4XF** XL r2 full subsample | vs N4XA: + v2 pools, H7 / H8 | 62.94 | .704 / .563 | 501 / 264 / 362 | **63.151 [+6.68; +0.99, +9.64] — qualifies** |
| N4LX cross-arm soup (N4LR + N4XF) | rule 5 | 63.40 | .730 / .551 | 516 / 273 / 379 | 60.367 [+3.90; −1.75, +6.96] — HOLD |
| N4LR M3 mixture (r2-clean), Lux on all rows | vs M3 N4LKr: Lux on retention rows | 63.07 | .729 / .546 | 518 / 288 / 360 | not a finalist (slot-2 tie-break) |
| N4LR2 same, KL 2.0 | vs N4LR: KL | 62.65 | .725 / .541 | 584 / 256 / 320 | not a finalist |
| N4LRQ A7q/k/s at 20% instead of v2-M | vs N4LR: multilingual Score | 62.63 | .709 / .553 | 530 / 238 / 367 | not a finalist |
| N4XA XL r2 A7-only subsample | vs N4LR: XL A7-only | 58.19 (median seed) | .604 / .561 | 536 / 226 / 204 | ineligible (Score floor; Noul one-sided) |
| *M3 N4LKr (reference)* | | 59.06 | .646 / .540 | 505 / 225 / 303 | 59.539 [+3.07; −4.90, +5.00] |
| *Nox 1.0* | | 55.33 | .666 / .460 | 460 / 228 / 378 | 56.470 |
| 2B probe S2LR (Sol 1.0, N4LR recipe) | vs S2T | 48.00 (S2T 51.09) | .525 / .439 | 397 / 244 / 199 | not run (R below S2T by 3.1) |

Formal details:

- **N4XF:** T .688, H .580. Typed-FINAL 582 / 734 / 185. public 231: 171. `mlx-diag` .770 (Nox 1.0 .795).
  - vs Decider 4B +1.27 [−5.71, +4.31]; vs Jet v6.2 +2.78 [−3.33, +5.94]; vs N4LKr +3.61 [+1.22, +12.13].
- **N4LX:** T .646, H .564. Typed-FINAL 565 / 694 / 175. public 231: 173. `mlx-diag` .773.
  - vs Decider 4B −1.51; vs Jet v6.2 −0.01.

## Findings

1. **The XL full recipe with own-Lux targets on every row is the 4B lever.**
   - N4XF's formal T (+.074) and H (+.061) both rise.
   - Its CSS15 gain is broad (11 of 15 tasks up), which is what moves the task-resampled lower bound above 0.
   - On development panels it tied the M3-line arms. The formal runner separated them, as proxy v2 predicted for
     close within-tier pairs.
2. **Own-Lux targets on Nox's own A7 retention rows raise typed DEV sharply** (N4LR vs N4LKr: dev T +.083). The
   typed-FINAL gain of the cross soup containing N4LR was smaller (+.032).
3. **KL 2.0 buys nothing over 1.0** (dev tie).
4. **A7q / A7k / A7s at 20%:** raise the pilot median and typed Score (367), lower typed Noul, and tie on R.
   Multilingual Score on `mlx-diag` stays level for N4XF (.811 vs .808). The `mlx-diag` loss is non-English Noul,
   which no M4 arm fixed.
5. **The XL A7-only control collapses typed-DEV Score at 4B** (206; levels mostly 0 / 2) while it gives the best
   pilot transfer. Adding the v2 pools and the H7 / H8 gap arms (N4XF) repairs typed DEV.
6. **Cross-arm soup N4LX was the best development artifact but only 60.37 formally.** Averaging the M3-line and
   XL seeds diluted N4XF's formal gains.
7. **2B probe:** the own-Lux line lowers typed DEV at 2B (as S2L did in M3); own-Sol S2T stays the 2B candidate.

## GPU-hours: 16.65 (budget ≈ 20)

- Node B: 16.32.
  - Five 4B arms at 2.85–2.90 each (15 seeds incl. preflights and postrun): 14.29.
  - Soups and readouts (six): 0.43.
  - Gap labels: 0.04.
  - 16K staging calibrations: 0.02.
  - 2B probe (three seeds + soup): 1.44.
- Node A GPU5 formal: 0.33 (two finalists: smoke + v3 / public 231 + `mlx-diag`).
- Median seed wall-clock 52 min (one GPU per seed).

## Failures and incidents

- **Staging-repo LFS cleanups rewrote history, and one removed unrequested weights.** See the incident record.
  - S2T was restored at `dev2-dec-staging@376c992c`.
  - The E8F soup's staging weights are gone; they are released in DEV2.0-0.8B.
  - A new tool with `rewrite_history=False` was used for the N4LX cleanup: 16.83 GB freed, nothing else lost, history
    intact.
- **The lock checker's first version sliced components in alphabetical order.** Found and re-run before data lock 2;
  the claims stood.
- **First formal chain wait condition** read a stale `status=running` line of a co-tenant lease entry. A
  `pkill -f` to replace it matched its own remote shell (the M3 pitfall). Relaunched with a last-status check;
  no GPU work was lost.
- **A `find` for rescreen receipts on node A listed file names under `/data/dev2/private/sealed/`** (names only;
  no content read). Recorded here; not repeated.
- **No preflight failed; no arm was rerun.**

## Items for the coordinator

1. **DEV2.0-4B release candidate: N4XF soup.**
   - Staged at `dev2-dec-staging@e8656221…`, `m4/N4XF-soup/`.
   - Scored run `/data/dev2/runs/dec/formal/m4/m4-N4XF-soup-nodeA`.
   - It is also +1.27 above Decider 4B, the tier leader (interval spans 0).
   - Release engineering applies the 23:15 calibration rule. C1 event 3 needs the XL r2 / H7-H8 / Lux-XL w3–w5
     independence recheck.
2. **The M3 mixture `m3-v2m-ret` contains 57 rows of r2's evaluation-panel-excluded groups**, 45 of them with
   CSS15 hits. It trained N4T / N4J / N4L / N4LKr and **the 2B candidate S2T**. This is a disclosure item for S2T;
   the eval track's rescreen-overlap check (`e4976c5d2`) can quantify it.
3. **Staging SHAs cited in M2 / M3 records no longer exist** (history rewritten by the cleanups). Use the node-B
   SHA-256 lists or the new S2T commit `376c992c`. Other tracks should check their LFS cleanup tools for the
   `rewrite_history=True` default.
4. **Proxy v2 vs the "not the pilot median" instruction** was resolved by amendment 1 (P_mean3, band 9). As proxy v2
   says, development panels did not separate the M4 finalists; the formal runner did.
