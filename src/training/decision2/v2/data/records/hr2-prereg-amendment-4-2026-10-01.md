# HR2 preregistration — amendment 4: HR2-r2, licence drops, construction-level filters and a fresh blind review (2026-10-01)

Committed after the round-1 error analysis and **before any HR2-r2 row is built and before any row is sampled
for review**. Assignment: coordinator task "HR2-r2" (2026-10-01), modelled on PN1-r2 (note 2026-09-30 02:25). The
round-1 result stands: HR2 `m5/hr2@afc3bc1e` is not release-safe (13 / 216 gold errors, 6.02%, exact 95% CI
[3.24, 10.07]%). §1–§3 of the prereg and amendments 1–3 are unchanged except where this amendment says so. There is
one attempt; there is no further iteration.

Tooling: `v2/data/hr2/r2.py` (analysis, boundary list), `build.py finalize` (new drop reasons), `review.py --round
hr2-r2`, runner `v2/data/hr2/node_a_r2.sh` (all at `874e60e46`). Error-analysis receipt (counts only):
[`hr2/r2/r1-error-analysis.public.json`](hr2/r2/r1-error-analysis.public.json); ids stay on node A.

## A. Error analysis of round 1 (CPU only)

| Family | Errors / 24 | Gold of the errors | Construction and agreement statistics | Filter that removes them |
| --- | ---: | --- | --- | --- |
| `vitc` | 4 | all "refutes" | All four are cross pairs of 2 × 2 revision cases: a claim written for one revision, paired with the other revision's sentence and labelled REFUTES. The edits do not change what the claim needs: a rewording (a spelled-out number, a near-synonym genre label, a character's surname for the first name) or a numeric update where both revisions satisfy the claim's "more than" comparison. Both reviewers said "supported", three times with high confidence from both. | L1 (licence) |
| `hs3_help` | 3 | levels 1, 2, 4 | Every error is two levels from the annotators' median; reviewers medium confidence on 22 of the 24 rows. Two errors have a 2–1 annotator split, one is unanimous. The causes differ: annotators rating against the annotation date, unanimous annotators stricter than the reviewers, annotators missing a factual error. All three are properties of absolute 5-level helpfulness ratings, not of a sub-construction. | C1 |
| `eth_just` | 2 | yes, no | One "desert" claim (R1 / R2 split at low confidence, R3 against gold), one "impartiality" claim (both reviewers medium). | none adopted |
| `eth_cs` | 1 | yes | Borderline "clearly morally wrong" scenario; the files carry no agreement statistics. | none adopted |
| `hs3_pref` | 1 | b | Overall margin 2, three annotators all ≥ 2 in the same direction; the disagreement is whether a specific answer that cannot be checked from the state beats a hedge. | none adopted |
| `prm_step` | 1 | yes | The +1 step directly before the rater's first −1 (phase 2, step 10 of a failing solution): arithmetically right, but it pursues a candidate that breaks the problem's constraints. 5 of the 12 reviewed yes rows were such boundary steps (1 error); the other 7 yes rows and 12 no rows had none. | C2 |
| `allegro` | 1 | level 3 | An organic 4-star rating on a review whose text reads negative (the text does not determine the rating; 5 of 24 rows had a low-confidence reviewer). | L2 (licence) |
| `eth_deon`, `eth_util` | 0 | | | |

Filters considered (round-1 errors removed / reviewed rows removed / HR2 TRAIN rows removed):

| Filter | Errors | Reviewed | TRAIN rows (native tokens) | Adopted |
| --- | ---: | ---: | ---: | --- |
| L1 drop VitaminC | 4 | 24 | 3,526 (0.49M) | yes (§B) |
| L2 drop Allegro | 1 | 24 | 3,125 (0.96M) | yes (§B) |
| C1 drop `hs3_help` | 3 | 24 | 4,019 (4.08M) | yes |
| C2 drop PRM800K boundary yes rows | 1 | 5 | 362 (0.15M), plus as many no rows by re-balancing | yes |
| `hs3_help` unanimous ratings only | 2 | 10 | 1,600 (1.67M) | no: the unanimous rows had 1 / 14 |
| `hs3_pref` margin 3 only | 1 | 12 | 4,751 (6.37M) | no: 67% of the family for one error that is not about the margin |
| `hs3_pref` no "slightly" annotator | 0 | 6 | 2,573 (3.48M) | no |
| `eth_just` without "desert" claims | 1 | 7 | 555 (0.06M) | no: one error in each template |

With L1, L2, C1 and C2, 9 of the 13 errors are removed. On the round-1 rows those filters keep, the error is 4 / 139 =
2.88% (exact [0.79, 7.20]); population-weighted 2.64%. This is an optimistic estimate of the families that remain
(they were chosen after looking), which is why the fresh review below decides.

## B. Licence decision: VitaminC and Allegro Reviews are dropped

The rule (coordinator): no NC, and no ShareAlike that is incompatible with Apache-2.0 model releases and data-card
attributions; if incompatible or uncertain, drop the source and record why. Neither source is NC. Both fail on
"uncertain":

- **ShareAlike against Apache-2.0 releases.** CC BY-SA 3.0 (§4(b)) and 4.0 (§3(b)) require Adapted Material to be
  shared under the same licence, a later version, or a BY-SA-compatible licence; Apache-2.0 is none of these (CC's
  compatible-licence list for BY-SA 4.0 holds only GPLv3 and the Free Art Licence 1.3). Whether trained weights are
  Adapted Material of their training text is unsettled law. The program's 2026-09-28 22:20 card policy (Apache-2.0
  label, CC BY-SA training data credited on the card) assumes they are not; under this task's rule, "unsettled" is
  "uncertain".
- **VitaminC** (`tals/vitaminc@be6febb7`): the pinned `LICENSE` makes the annotations available "under the license
  terms described on the applicable Wikipedia article pages, or, where Wikipedia license terms are unavailable, under
  the Creative Commons Attribution-ShareAlike License (version 3.0)", while the card metadata says `cc-by-sa-3.0`. The
  effective terms are each article's (Wikipedia's text licence changed over time, CC BY-SA 3.0 then 4.0, with GFDL
  dual licensing for much of the text), so no single licence can be credited on a card.
- **Allegro Reviews** (`allegro/klej-allegro-reviews@b9ca796d`): the pinned snapshot has no licence file, card or
  tag; CC BY-SA 4.0 is stated outside it (the GitHub README of `allegro/klejbenchmark-allegroreviews`, and the KLEJ
  website per the §1 source audit). The texts are marketplace users' reviews, and the release does not document the
  rights chain from the users. CC BY-SA 4.0 also covers EU sui generis database rights (§4(b)): a database in which
  the re-user holds such rights and that includes a substantial portion of the contents is Adapted Material. HR2
  carried 3,375 of the 9,577 train-split reviews (35%), so ShareAlike could reach the HR2 dataset as a whole, not only
  its Allegro rows.

Effect: VitaminC and Allegro leave HR2-r2 (TRAIN and DEV); `license-registry-hr2.json` lists only the remaining
sources (HelpSteer3 CC BY 4.0, ETHICS MIT, PRM800K MIT; none ShareAlike), so `v2.data.freeze` refuses any row of a
dropped source. This decision is stricter than the 22:20 card policy; the same question applies to the CC BY-SA sources
already in released mixtures, and is left to the coordinator.

## C. Construction-level filters

- **C1 — drop `hs3_help`.** Absolute 5-level helpfulness labels from free-text feedback openers depend on annotator
  calibration and on context outside the item (§A). The preference judgments of the same source (relative, strict
  agreement) are kept. HR2-r2 has no Score family.
- **C2 — PRM800K error boundary.** Drop a `prm_step` yes row if, in any record that produced it (same exclusions as
  §2.3), it is the last +1 step on the walked path before that path's first −1 step. Rows are listed by
  `r2.py boundary` from the candidates (TRAIN and DEV). No rows are dropped by a step-index rule; the per-bucket
  balance of §2.3 then removes the same number of no rows.
- Not adopted: the alternatives in §A's table. ETHICS has no agreement statistics to filter on, and its two error
  templates have one error each.

## D. HR2-r2 rows

`build.py finalize` on the HR2 candidates (`45b56010…` / `b0498c1f…`), so every r2 row was scanned as a candidate:
the round-1 drops (G4 families `indonli`, `kob_boolq`; 1,202 quarantined groups; 1,180 DEV groups near TRAIN; the 13
round-1 gold errors; the 28 leak-guard rows) plus L1, L2 (reason `licence`), C1 (`construction_family`) and C2
(`construction`), then the same re-balancing. DEV gets the same drops.

**Expected size** (simulated on the round-1 final files): TRAIN ≈ 16,303 rows (from 27,697), 15,226 groups, ≈ 10.9M
native tokens (from 16.75M); Choice 10,207 (`hs3_pref` 7,099, `eth_util` 3,108), Noul 6,096 (`prm_step` ≈ 2,838,
`eth_deon` 1,094, `eth_cs` 1,084, `eth_just` 1,080); DEV ≈ 961 rows. Polish leaves with Allegro; the other languages
are HelpSteer3's. TRAIN is below §2.6's 20–60k target: the shortfall is reported and not filled by relaxing a rule.
Round-2 gold errors are then removed (with re-balancing).

## E. Audits again (G1–G8, on the r2 files)

G1 names; G3 C1 source-term scan over the raw files and the r2 files; G2 overlap re-scan of r2 TRAIN and DEV against
PI-v4 (full and quarantining) and PI-hr2, and DEV against TRAIN, with the quarantine recheck — it must find no group
with a quarantining hit and no DEV group near TRAIN, and any group it does find is dropped whole before sampling
(reported); G4 per family on r2 TRAIN (a failing family is dropped before sampling); G5 balance, G6 guard, G7
isolation, G8 sizes on the final files; the leak guard on the r2 files (amendment 3's rule).

## F. Fresh blind review of HR2-r2

- **Frame:** r2 TRAIN after §E, minus every row and every group of the round-1 key.
- **Sample:** 48 rows per family × 6 families = **288 rows**, one per group, gold-stratified (24 / 24: Noul by
  answer, Choice by gold position), in `sha256("hr2-r2-review-v1:" + id)` order. Review ids `qNNN` in
  `sha256("hr2-r2-packet-v1:" + id)` order; two packets of 144 per reviewer order; R2 order
  `sha256("hr2-r2-review-r2-order-v1:" + rid)`, R3 order `sha256("hr2-r2-review-r3-order-v1:" + rid)`.
- **Reviewers:** fresh `claude-opus-5-5-max` subagents with the same instructions
  ([`dq/reviewer-hr2.md`](dq/reviewer-hr2.md)) and the same packet fields; R1 and R2 answer every item; a fresh R3
  answers the splits. Agreement and gold-error rules as in §4 and amendment 2.
- **Thresholds, unchanged as rules:**
  - **P1:** pooled error ≤ 5.0% and exact 95% upper bound ≤ 8.0%. For n = 288 that is **≤ 13 errors** (13 / 288 =
    4.51%, upper 7.60%; 14 gives 8.02%).
  - **P2:** population-weighted error ≤ 5.0% (weights: r2 TRAIN family counts; stratified bootstrap 10,000 reported).
  - **P3:** a family fails when its exact lower bound exceeds 5%, the rationale of round 1's 5 / 24: here **≥ 7 of
    48** (6 / 48 gives 4.73%, 7 / 48 gives 6.07%).
- **No fix rule.** Rows judged gold errors are dropped from r2 TRAIN in any case.
- **PASS** (P1, P2 and P3): `release_safe: true`. **FAIL**: HR2-r2 is published flagged `release_safe: false`, and
  nothing more is tried in this job.

## G. Publication

`hf_headroom.sh` first; upload as a **new revision** of the private dataset, replacing the contents of `m5/hr2/`
(the round-1 files stay at `afc3bc1e`); read-back SHA-256 check against `registry.json`, the remote folder must hold
exactly the registry's files; private before and after. Manifests, `status.json` and the data card are updated; the
record, gist 02 and the merge follow. Disclosed with the data: no Score family, the size shortfall, the changed DEV
slice, and the review's limitation (one model family, AI adjudication rather than human annotation).
