# IB1 preregistration — amendment 3: round 3 (2026-10-01)

IB1 round 2 (run `r2`, `m6/ib1/` revision `82bf70a7`, NOT release-safe) failed the fresh decisive review:

- 12 errors in 225 items;
- `wands` alone had 4 of 19;
- in 5 of the 12 errors, reviewers chose a class that round 2 kept as an option but no longer used as gold.

Round 3 changes exactly the four points below, then repeats the build, every audit and a fresh blind review. This
amendment is committed **before any round-3 row is built, sampled or reviewed**. The prereg (`1f6823fae`), amendment 1
(`a21ee5f8a`) and amendment 2 (`1464c28df`, with its addendum `602fc4e9c`) are otherwise unchanged. No family, source
or rule is added.

## A. Changes

1. **`sentfin` becomes a true two-option item.**
   - The options are exactly negative and positive, in that order (labels 0 and 1). Neutral is removed from the option
     set and from gold: rows with neutral gold are not converted.
   - The instructions and option descriptions are unchanged; the template id becomes `ib1_sentfin_2way_v1`.
   - Row ids and group ids depend only on the source record, so they are unchanged.
   - The family cap is unchanged (8,000 in total). With two labels, it allows 4,000 per label.
   - The `sentfin-neutral` construction of amendment 2 no longer applies.
2. **`wands` leaves IB1.** Its relevance boundary is ambiguous, and round 2 had 4 errors in 19.
3. **`csqa` leaves IB1.** CommonsenseQA is already in the released 2.0 mixtures, so it adds no coverage, and it had 2
   errors in round 2. Balanced COPA (`copa`) stays.
4. **SummEdits domain rule.**
   - A domain is dropped when it has 2 or more stage-R (decisive) review errors across rounds 1 and 2 combined. The
     family is dropped if every domain fails.
   - Stage-R errors by domain:

     | Domain | Round 1 | Round 2 | Total | Decision |
     |---|---:|---:|---:|---|
     | `shakespeare` | 3 | (not in round 2) | 3 | dropped |
     | `sales_email` | 0 | 2 | 2 | dropped |
     | `billsum` | 0 | 1 | 1 | kept |
     | `sales_call` | 0 | 0 | 0 | kept |

   - The round-1 stage-S screen had one `sumedit` disagreement, in `billsum`. Stage S was a family screen, not a
     decisive review, so it is not counted. That row stays dropped by id.
   - Constructions dropped in `finalize`: `sumedit-shakespeare` and the new `sumedit-sales_email`.
5. **Everything else is unchanged:** the remaining families, sources, converters, caps, balance rules, gates (G0–G8)
   and review thresholds.

## B. Expected size impact

These figures come from the round-2 final TRAIN file (31,923 rows) and the raw SEntFiN label counts (neutral 5,507,
positive 5,060, negative 3,804). They are before round-3 audits and review drops:

| Family | Round 2 | Round 3 (expected) |
|---|---:|---:|
| `wands` | 4,068 | 0 |
| `csqa` | 4,981 | 0 |
| `sumedit` | 576 | about 354 (`billsum` 128, `sales_call` 226) |
| `sentfin` | 4,778 | about 6,500–6,900: the negative class (about 3.7k after conflict resolution, all splits) binds, not the cap |
| Other 8 families | 17,520 | unchanged up to re-scan and review drops |
| **TRAIN** | **31,923** | **about 24,500** |

IB1 round 3 then has 10 families, and `sentfin` becomes its largest family (about 27%).

## C. Build, audits and review

- **Build and audits.**
  - Run `r3` rebuilds the candidates with the round-3 builder. Only `sentfin` rows differ from round 2.
  - Every audit is re-run as in amendment 2 §B: G0 with positive controls, G1, G2, the C1 names check, G4, G5, G7,
    G8 and the leak guard.
  - The finalized files are re-scanned until clean before sampling, and the final files are re-scanned after the
    review drops.
- **Carried drops by id:**
  - the 17 round-1 stage-S and stage-R ids;
  - the 12 round-2 stage-R error ids;
  - the 12 round-1 leak-guard ids.
- **Review sample.**
  - Stage R runs on the clean finalized TRAIN file, with `per_family = max(18, ceil(216 / F))`.
  - The amendment-2 addendum (raise the quota until the total reaches 216) applies from the first draw.
  - Excluded, together with their groups: every round-1 and round-2 sampled row, which means the round-1 stage-S and
    stage-R keys, the round-2 stage-R key and the unreviewed round-2 first draw.
  - Fresh salts: `ib1-r3-review-v1`, `ib1-r3-packet-v1`, `ib1-r3-review-r2-order-v1` and
    `ib1-r3-review-r3-order-v1`. Item ids use the prefix `u`.
- **Reviewers.**
  - Fresh reviewers (new subagents), blind to gold, source and family, with the round-1 instructions. Stage S is not
    repeated.
  - R1 and R2 answer every item; R3 answers the splits only.
  - Every reviewer runs in the foreground (`run_in_background: false`).
- **Thresholds unchanged:**
  - P1: error ≤ 5% and Clopper–Pearson upper bound ≤ 8%;
  - P2: population-weighted error ≤ 5%;
  - P3: no family with a Clopper–Pearson lower bound above 5%.
- **Review errors** are dropped from the final files.
- **Publication.** A new private revision of `m6/ib1/`, with `release_safe: true` only if P1–P3 and every audit pass.
