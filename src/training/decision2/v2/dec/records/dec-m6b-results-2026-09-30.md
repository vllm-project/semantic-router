# Decoder Milestone 6b — results (formal test of the N6D lead against DEV2.0-4B; 2026-09-30)

Preregistration [`dec-m6b-prereg-2026-09-30.md`](dec-m6b-prereg-2026-09-30.md) (`833ec0967`). Post-key same-panel
evidence; nothing was uploaded; C1 was not opened.

## Bottom line

**No successor. Both N6D finalists fail item 1** against DEV2.0-4B `452f1332` (v3 63.151). DEV2.0-4B stands.

| Item | Slot 1 `4b-N6D-b1` (N6D soup) | Slot 2 `4b-N6D-b2_3` (⅔ N6D + ⅓ N4XF) |
| --- | --- | --- |
| Revision (per-file list) / calibration | `25747627…` / CAL698 16K adopted by the 23:15 rule | `a726b09f…` / T = 1 |
| v3 / T / H | 60.484 / .6469 / .5655 | 62.263 / .6713 / .5775 |
| 1. v3 vs current revision | −2.67 [−4.59, +0.33] **FAIL** | −0.89 [−2.88, +1.16] **FAIL** |
| 2. H vs current revision | [−.044, +.033] not significantly below | [−.034, +.029] not significantly below |
| 3. Types (typed FINAL C / N / S; N4XF 582 / 734 / 185) | OK; 547 / 672 / 216 | OK; 574 / 691 / 209 |
| 4. mlx-diag | not collected (item 1 decides) | not collected |
| 5. Tier gates | vs adopted Nox 1.0 +4.01 [−0.63, +7.52] **FAIL** | +5.79 [+0.57, +8.94]; v3 ≥ 55.7; H vs peers OK |
| 6. Overlap | not run (item 1 decides) | not run |
| 7. `gates public231` vs the bar | 173 vs 171, OK (p .73); hard 58 vs 56 | 174 vs 171, OK (p .51); hard 59 vs 56 |
| 8. C1 | not requested (fails items 1–7) | not requested |
| vs Decider 4B | −1.40 [−6.69, +1.59] | +0.38 [−5.76, +2.86] |

Run directories: node A `/data/dev2/runs/dec/formal/m6b/m6-4b-N6D-{b1,b2_3}`; successor files in
`formal/m6b/successor/`. Collections on node B GPU3 / GPU4 (`dbe5f32b`, copies of `cache-frozen` `f6d0f920…`,
16,384 tokens), relayed gold-free with manifest checks and scored on node A.

## Findings

1. **The development typed gain reversed on typed FINAL.** Typed DEV had N6D ahead (T .729 vs .704; Choice +39,
   Noul +38); typed FINAL has it behind (T .647 vs .688; Choice −35, Noul −62, Score +31). The 2× XL dose trades
   typed Noul / Choice for Score on the formal families.
2. **The CSS-pilot guard was right here.** Formal human transfer is level (H .566 / .578 vs .580) and the typed
   axis decides the loss. The M6 rule removed both points for H3 and for the typed-DEV Score floor; the formal
   runs agree they are not successors.
3. **Public 231** moved up slightly on the hard tier (+2 / +3 items), within noise (the guard only catches large
   losses).
4. For Milestone 7: more of the same XL recipe does not help 4B; a 1× dose increase is at best neutral.

## Tooling

- `m6_successor.py` now gates item 7 on `v2.eval.gates public231` against the bar and reads item 8 from the eval
  custodian's C1 SUMMARY (`833ec0967`, 8 tests). Its first live use is above.
- Incident (no GPU time): the first launch stopped before any job because M6's finished runs had left their
  `owner.dec-formal` / `owner.m6-formal-smoke` lease entries on node B GPU3–4. Only those decoder entries were
  removed (copies kept under `m6b/logs/stale-leases/`); the chains were then launched once.

## GPU-hours

**0.284** (two CAL698 16K fits, two 8-item smokes, two formal collections; node B GPU3–4).
