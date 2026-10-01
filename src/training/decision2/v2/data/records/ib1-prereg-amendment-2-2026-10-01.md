# IB1 preregistration — amendment 2: round 2 (2026-10-01)

IB1 round 1 (run `b2`, published as `m6/ib1/` flagged NOT release-safe) failed the stage-R review on P1 and P2. Nine
of its 13 errors came from three label constructions whose gold label sits on a judgement boundary. Round 2 drops
exactly those three constructions and repeats the build, every audit and a fresh blind review. This amendment is
committed **before any round-2 row is built, sampled or reviewed**. The prereg (`1f6823fae`) and amendment 1
(`a21ee5f8a`) are unchanged except where this amendment says so. No family, source or rule is added.

## A. Dropped constructions

| Name | Family | Rows dropped (TRAIN and DEV) |
|---|---|---|
| `sentfin-neutral` | `sentfin` | every row whose gold class is `neutral` |
| `sumedit-shakespeare` | `sumedit` | every row of the SummEdits `shakespeare` domain (both labels) |
| `wands-partial` | `wands` | every row whose gold class is `Partial` |

The drop applies in `finalize`, before balancing, with a `construction` reason in the removal report. Everything else
in these three families is unchanged.

- **Option sets stay as built.** `sentfin` rows keep their three options (negative, neutral, positive) and `wands` rows
  keep theirs (Exact, Partial, Irrelevant), so the row format and the instructions are unchanged. The dropped class is
  never gold in round 2. This creates a prior against the dropped class; the data card states it, and a track that
  needs that class must get it from other data.
- **Balancing (§3, G5).** For `sentfin` and `wands`, equal-label balancing runs over the two remaining classes. G5
  then requires each remaining class share to be within 1/2 ± 0.05 and the dropped class to have zero rows. For
  `sumedit` the per-domain yes / no equality is checked over the three remaining domains (billsum, sales_call,
  sales_email). Every G5 check also fails if any row of a dropped construction is present.

**Expected size impact**, from the round-1 final TRAIN (37,314 rows) before round-2 audits and review drops:

| Family | Round 1 | Round 2 (expected) |
|---|---|---|
| `sentfin` | 7,158 | about 4,772 (−2,386) |
| `sumedit` | 1,280 | about 608 (−672; Shakespeare was 52% of the family) |
| `wands` | 6,357 | about 4,238 (−2,119) |
| others | 22,519 | unchanged up to re-scan and review drops |
| **TRAIN** | **37,314** | **about 32,100 (−14%)** |

DEV loses the same constructions. The round-2 counts may be slightly higher if the dropped class was the binding one in
the round-1 balance.

## B. Build and audits

Run `r2` rebuilds the candidates from the same pinned raw files with the same builder, then re-runs every audit:

1. **G0** Index-row exclusion (rules E, N1, N2, G) on the candidates, with fresh positive controls (2,000 exact and
   2,000 perturbed suite rows).
2. **G1** and **G3** names checks (protected-name check and the C1 registry names check) on the raw files, the
   candidates and every final file.
3. **G2** held-out panel exclusion: overlap against PI-v4 (full and quarantining), PI-ib1 and the DEV-vs-TRAIN
   self-scan; quarantine lists as in round 1.
4. **Carried drops.** The round-1 drop lists are applied by row id (ids are deterministic):
   - the 17 stage-S and stage-R disagreement / error ids;
   - the 12 leak-guard ids.

   No other round-1 decision is carried.
5. **Re-scan of the round-2 files.** The overlap scan exempts boilerplate units, and that exemption depends on the
   size of the candidate set. So the finalized round-2 files (pass 1) are re-scanned: G0, the four overlap scans, the
   quarantine rule and the names checks. Every newly flagged group is dropped (pass 2). Pass 2 is then scanned again
   and must come back clean before any row is sampled. If it does not, the drop-and-rescan step repeats (at most 3
   times; otherwise round 2 stops).
6. **G4** shortcut audit per family on the pass-2 TRAIN file. A failing family leaves IB1, as in round 1.
7. **G5–G8, freeze and leak guard** on the final files, as in round 1. After the review drops, the final files are
   scanned once more. Any group flagged there is dropped without re-sampling, and the drop is reported.

## C. Fresh blind review

- **Stage S is not repeated.** The family set is the same as the one stage S passed in round 1. Its four
  disagreement ids stay dropped (B.4).
- **Stage R runs on the pass-2 TRAIN file** (the final round-2 TRAIN file before review drops). Excluded from the
  sample, together with their groups:
  - the round-1 stage-S rows;
  - the round-1 stage-R rows.
- **Sample size.** `per_family = max(18, ceil(216 / F))`, so at least 216 rows (18 per family with 12 families).
  - **Addendum (committed before any round-2 item was reviewed).** The first draw gave 214 items: once the round-1
    groups are excluded, `sumedit` has enough distinct groups for only 16. The draw was set aside unreviewed. If a
    draw falls below 216, the per-family quota is raised by one and the draw is repeated until the total reaches 216.
- **Fresh salts.** `ib1-r2-review-v1` (sample), `ib1-r2-packet-v1` (packets), `ib1-r2-review-r2-order-v1` and
  `ib1-r2-review-r3-order-v1`. Item ids use the prefix `t`.
- **Thresholds unchanged.**
  - P1: pooled error ≤ 5% and Clopper–Pearson 95% upper bound ≤ 8%.
  - P2: population-weighted error ≤ 5%.
  - P3: no family whose Clopper–Pearson lower bound is above 5%.
- **Reviewers.** Fresh reviewers (new subagents that did not take part in round 1), each blind to gold, source and
  family, with the round-1 instructions (`dq/reviewer-ib1.md`). R1 and R2 answer every item; R3 answers the R1 / R2
  splits only. Every reviewer runs in the foreground (`run_in_background: false`). Any deviation is recorded with its
  reason in the results record.
- **Review errors** (rows whose majority answer differs from gold) are dropped from the final files.

## D. Publication

The final files are uploaded as a new revision of the private dataset path `m6/ib1/`, after the headroom check, with a
read-back check. `status.json` sets `release_safe: true` only if the round-2 review passes P1–P3 and every audit
passes. Otherwise round 2 is published flagged NOT release-safe, as in round 1. The manifest, data card, records,
state and gist 02 are updated, and the branch is merged into the integration branch.
