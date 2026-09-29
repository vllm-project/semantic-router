# 9B Milestone 5 result: no finalist, so no successor; DEV2.0-9B stands

Run under the [preregistration](lux9b-m5-prereg-2026-09-29.md) (`567e7fd39`), with no amendments. Development readouts only:
no M5 candidate reached a formal post-key run, so no v3 number exists for any M5 artifact. The released
`llm-semantic-router/DEV2.0-9B@ae683196…` (K-a13; weights byte-identical to `53bac735`) stays the 9B model. Nothing was
uploaded.

**Verdict.** The preregistered α rule, anchored on the incumbent's own development readout, found no pick on any of the
three lines (KD, KG, UM5), so the finalist list is empty (`m5/rules/finalists.json` `f566bd56…`). The prereg says slots
are not refilled, so there were no formal runs and the successor rule was never reached. The family-equal A7 dose
**raises typed-DEV Score and lowers Noul rule_precedence**. Every point that keeps Noul above the incumbent's floor is the α
⅓ point, and none of those gains ≥ 0.01 typed-DEV T with H3 at or above the incumbent's.

## Development readouts

Typed DEV 1,600 (families AG / RP / SR / TT) plus CSS pilot 1,430, at 16K on the runner mirror `3277dec9d`. R is the fresh
re-read of the incumbent, which reproduces M4's K ⅓ readout exactly; Lux is the fresh Lux 1.0 reference.

| Artifact | T | Choice / Noul / Score (of 800 / 400 / 400) | H3 | H | P | α rule (vs R) |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| R = K-a13 (incumbent) | .9250 | 799 / 338 / 343 | .5622 | .5795 | 73.22 | anchor; floors C 775, N 326, S 331 |
| Lux 1.0 | .8763 | — | .5283 | .5724 | 70.82 | report only |
| KD-s1 / KD-s2 | .7956 / .7662 | — | .559 / .544 | .530 / .512 | 64.94 / 62.66 | seeds |
| KD soup (α 1) | .8506 | 792 / 216 / 353 | .5662 | .5538 | 68.63 | ✗ Noul floor; RP family floor |
| KD ⅓ | .9319 | 800 / 342 / 349 | .5589 | .5765 | 73.30 | ✗ H3 < R (G would be +.0069) |
| KD ½ | .8994 | 800 / 300 / 339 | .5651 | .5820 | 72.35 | ✗ Noul floor |
| KD ⅔ | .8781 | 799 / 253 / 353 | .5745 | .5710 | 70.81 | ✗ Noul floor; RP family floor |
| KG-s1 / KG-s2 | .8588 / .7812 | — | .551 / .534 | .523 / .500 | 66.98 / 62.50 | seeds |
| KG soup (α 1) | .8688 | 791 / 211 / 388 | .5483 | .5225 | 67.38 | ✗ Noul, H3, RP |
| KG ⅓ | .9163 | 800 / 327 / 339 | .5517 | .5769 | 72.71 | ✗ H3 < R |
| KG ½ | .9006 | 799 / 281 / 361 | .5546 | .5585 | 70.92 | ✗ Noul, H3, RP |
| KG ⅔ | .8881 | 799 / 239 / 383 | .5451 | .5395 | 69.22 | ✗ Noul, H3, RP |
| UM5 ⅓ | .9300 | 800 / 339 / 349 | .5651 | .5873 | 73.91 | ✓ eligible, G = +.005 |
| UM5 ½ | .9069 | 799 / 291 / 361 | .5616 | .5690 | 71.84 | ✗ Noul, H3, RP |
| UM5 ⅔ | .8869 | 799 / 250 / 370 | .5625 | .5698 | 71.09 | ✗ Noul, RP |
| UM5 (α 1) | .8775 | 798 / 221 / 385 | .5627 | .5540 | 69.72 | ✗ Noul, RP |

- **Seed rule:** both soups won (KD 68.63 vs seed mean 63.80; KG 67.38 vs 64.74). The outputs are `rules/seed-KD.json`
  `085aeb19…` and `seed-KG.json` `dfb64593…`.
- **α rule:**
  - KD has no eligible α (`alpha-KD.json` `7dcfbd0f…`), and neither has KG (`alpha-KG.json` `93644a88…`).
  - UM5's only eligible point is ⅓, with G* = .005 < .01, so it gives no pick (`alpha-UM5.json` `10ac47df…`).
  - Report-only: M4's Lux-anchored rule would have picked the ⅓ point on every line. That rule has no incumbent
    floors, so those picks would only have matched K-a13.

## Reading (development only; not selection)

- **The dose moves the wrong family.** The seed-2 soups lost Noul `rule_precedence` badly:
  - KD soup 216 and KG soup 211, against M4's K soup at 309 on the same x60 recipe, and Lux 1.0 and K-a13 at 272 and 338.
  - Score (`set_reconciliation`) went up: 353 / 388 vs 340.
  - So adding 24M family-equal A7 Stage tokens to the K recipe traded rule-precedence Noul for Score. Interpolating back
    toward Lux repairs Noul only at α ⅓, where the typed gain is gone.
  - The 27B result (+.099 typed from a family-equal A7 slice, measured without own-1.0 KL) does not transfer to the 9B
    K recipe at this dose.
  - The dose is Choice-heavy (17.7 / 4.3 / 1.9M tokens for Choice / Noul / Score), and its Noul families have the lowest
    own-Lux agreement (.82).
- **KD − KG** (KL on the dose rows):
  - KL on the dose kept human-transfer screening higher (soup H3 .566 vs .548; ⅓ point .559 vs .552) and Noul about equal.
  - Gold-only dose rows raised Score further (388 vs 353) and lowered the CSS pilot.
  - Neither variant clears the incumbent.
- **UM5 ⅓** (1/6 KD soup + 1/6 KG soup + 2/3 Lux) is the only artifact at or above R on every development screen. Its gain
  (+.005 T, +.003 H3, P +0.69) is inside readout noise, well short of the ≈ +.05 typed gain a successor needs (prereg
  power check).
- **For Milestone 6** (suggestions):
  - Keep the K recipe's Noul protected. A dose must be Noul-balanced, or its rule-precedence conflicts identified first.
  - The paraphrase-style Noul arm (PN1) and the JevBench hard-skill families (quoted-conclusion verification, long policy
    packets with amendments, "condition not met" negatives; 17:15 note) are the untested data levers for the 9B weak
    spots (non-English Noul, public 231 hard).
  - The mlx-diag card-part check (`lux9b.mlx_paired`) showed that α ½ artifacts risk a significant multilingual loss.

## Receipts, storage, resources

- **Data** (frozen in the prereg):
  - KD train `86a41afb…`; teachers KD `0f9e6ed2…` and KG `cdcd99c1…`.
  - Exposure `5639cfa3…`, groups [].
  - The dose is not typed-FINAL-generator material. Its families come from the Decision 1.0 curriculum generators, so
    the 16:40 rule holds.
- **Runs** (node A `/data/dev2/runs/9b/m5/`), all steps exit 0:
  - Preflights `pf-KD-s1-check` and `pf-KG-s1-check`: PASS.
  - Seeds: KD-s1 and KG-s1 took 09:22–12:56Z, BEST `checkpoint-0002287`. KD-s2 and KG-s2 ran 12:54/12:56 to about 16:25Z.
  - Soups and lines finished by 16:50Z, and UM5 by 17:09Z.
  - Readouts: `readout-refs`, `readout-seeds1`, `readout-lines`, `readout-um5` (`readout.json` each); rules in `rules/`.
- **Chains** (files in `lux9b/m5/chains/`; each uploaded and size + SHA-256 verified, then launched separately, with
  liveness checked by PID and container):
  - `m5-gpu6` `49a81cf3…` and `m5-gpu7` `48a18381…`;
  - `m5-um5-gpu6` `64c687b2…` and `m5-um5-gpu7` `bfe1689a…`.
- **Kept on node A for M6** (no HF upload): `m5/KD-soup-build/soup` (`SHA256SUMS` `e377ca2e…`, 16 files) and
  `m5/KG-soup-build/soup` (`1aeccdeb…`). The line points' `*-build/soup` directories are scratch.
- **GPU-hours:**
  - 14.88 by job receipts (64 GPU jobs: 4 seeds with in-arm SELECT / CAL / dev readouts, 2 preflight trios, 2 reference and
    12 soup / line readouts). About 15.3 by chain wall-clock × 2 GPUs.
  - Soups were CPU (0.55 h). The cap was 24 and the projection 15.5.
  - No formal runs. The GPU7 C1 lease stayed idle throughout, so there was no conflict.
- Leases node A GPU6–7: `track=9b-m5 status=idle` (17:11Z), free for reassignment.
