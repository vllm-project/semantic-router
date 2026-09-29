# Amendment 2 to the PN1 preregistration (research & data Milestone 4, 2026-09-29)

Amends [m4-pn1-prereg-2026-09-29.md](m4-pn1-prereg-2026-09-29.md) after
[amendment 1](m4-pn1-prereg-amendment-1-2026-09-29.md). Committed and pushed after the first judge pass and one
finalize run, and before the second judge pass.

## 1. What the first judge pass showed

- **Pass 1:** `Qwen/Qwen3.8-27B@1d4bf0f2…`, 50,220 items on node B GPU3 and GPU4, 0.133 GPU-h.
  - It judged 34,087 rows: per language × stratum × label, ⌈1.6 × planned units⌉ + 2 rows in seed-hash order
    (amendment 1, §2).
- **The judge leans strongly to "No" on the yes constructions** (disagreement before filtering):

  | Construction | Label disagreement per language | Median P(yes) |
  | --- | --- | --- |
  | `pn-hop` | 69–91% | 0.11 |
  | twin "same" | 41–87% | — |
  | name coordination (A CONJ B → B CONJ A) | 15–49% | 0.59 |

  - The fluency check fails 58–79% of twin edits and 31–89% of name swaps.
  - The no constructions agree with the judge: `pn-near` 0.3–4.5%, role swaps 0–5%, twin "different" 2–11%.
- **Consequence:** finalize on pass 1 alone (build `aaad8cd7a5f3`, commit `eb0d24271`) gave 5,212 of the
  20,000 TRAIN rows and the full dev slice (2,000). Yes rows bind in almost every stratum.
  - That build is superseded. It was never audited and is not a deliverable.

## 2. Change: a second judge pass over more of the same pool (§8, volumes)

- **Same judge, same rules.** The pass judges unjudged rows of the same candidate pool with the same model,
  revision, prompts, A/B order rule, single-token P(yes) and thresholds:
  - a yes row is kept at P ≥ .5 and a no row below .5;
  - an edited sentence needs fluency ≥ .5.
  - No family, stratum or balance rule changes.
- **Volume rule** (`pn1_build judgeset-extend`): per language × stratum × label, unjudged rows are added in
  seed-hash order until the kept rows plus the expected keeps reach 1.25 × the planned units (TRAIN + dev).
  - The expected keep of a row is the pass-1 keep rate of its language × family × label. It is pooled over the
    languages where fewer than 20 rows were judged.
  - Rows are ordered by need. The judge job keeps whole rows in that order within its budget.
- **Budget:** what judging has left (0.45 − 0.133 GPU-h). The cumulative stop stays at 1.05 GPU-h
  (amendment 1).
- **Then:** finalize runs once on the union of both passes, dev first as preregistered. Any shortfall after that
  is taken as it is and reported.

## 3. Not changed

- **The judge filter is not recalibrated.** P(yes) ≥ .5 stays the yes rule, although the judge gives exact
  coordination swaps a median P(yes) of .59.
  - A recalibration (another threshold or judge prompt) would be a new amendment for the track. It is not a
    builder decision.
