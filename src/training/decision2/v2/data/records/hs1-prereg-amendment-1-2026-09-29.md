# HS1 preregistration — amendment 1 (2026-09-29)

Committed before the release build on node A, before any upload, and before any model output on HS1 material exists.
It amends [`hs1-prereg-2026-09-29.md`](hs1-prereg-2026-09-29.md) (`6e932c285`). A session interruption (account usage
limit, 19:04–20:45 UTC+8) fell between the preregistration and this amendment. No row was published and no model
was run in that gap.

## What the first local builds and audits showed

The builds and audits were CPU-only, run in a local scratch directory, and nothing was uploaded.

- **Generation.** The generators run, all rows validate, and every item's oracle equals its re-check.
- **Leaks found and fixed in the generators before this amendment:**
  - F1 `expense_total`: the Score band scale was correlated with the gold level (option-only .53 vs a .28 baseline).
  - F1 Choice: the gold's rank among ordered values was not uniform (`order_sla`, `expense_total`,
    `transit_connection`).
  - F2: "first matching provision" was right on .558 of Score rows.
  - F3 (before its repair): 71% of draws failed on rendering and re-parse errors.
- **Hash-drawn targets.** Noul and Score targets drawn from a seed hash left some kind cells at .45–.54.
- **Three audit rules in §5 contradict the preregistered design itself (§1).** A rule-conforming generator cannot
  pass them:
  - **F1 quote-only probe (A3b).** The quote states an answer, and §1/A3c fix "adopt the quote" at exactly .50 for
    Choice, Noul-direct and Score. A quote-only classifier can therefore reach .50, above the Choice / Score majority.
    The design concern is whether the quote text reveals *correctness*.
  - **F3 Choice (A3a/A3b).** The catch-all option ("all requirements are met" / "none of them") is gold in exactly
    one twin of every Choice world, which is 50% of F3 Choice rows by design.
  - **F3 Score balance (A2).** Level 0 is the negative twin, which is 50% of F3 Score rows by design.
- **Per-kind shortcut cells (A3a) are small**, at 60–136 rows per type for F2. On labels shuffled to carry no
  information, the plain rule (accuracy ≤ majority + .05) fails 12–16 of 48 such gates, so it measures noise at that
  size.
- **Tokens.** F3 states have the preregistered lengths (400–1,200 / 1,200–3,000 characters), but the token estimate
  in §2 (~1.8M) was wrong by about a factor of two. Characters ÷ 4 gives about 4.0M for F3, 4.0M for F2 and 4.7M for
  F1, about 12.7M TRAIN in total, against the ~8.2M target (±25%).

## Changes

1. **Balanced targets.** The build passes a per-(kind, interface) counter `index` to `make_group`. Noul targets
   alternate and Score levels cycle, so label balance is exact by construction (F1, F2). F3 was already balanced by
   its twins.
2. **Redraws and stops (clarifies §4).**
   - An oracle / re-check disagreement raises `RecheckMismatch` and stops the build; it is never redrawn.
   - A draw is redrawn from a fresh deterministic generator when it fails a design constraint: a rendering length, a
     decisive distractor, a re-parse ambiguity, a twin that cannot be formed, or a heuristic quota. Every emitted row
     has oracle = re-check.
3. **A3b, F1 quote-only probe.** Choice and Score are judged against max(majority, .50) + .05. Noul-direct and
   Noul-verify stay at majority + .05. Noul-verify is also probed on its own, which is the correctness question.
4. **A3a / A3b, F3 Choice.** The baseline is max(majority, catch-all gold rate) + .05.
5. **A2, F3 Score.** The 0.8–1.2× uniform rule applies to levels 1–3 of the positive twins. Level 0 = 50% is the
   design.
6. **A3a cells.**
   - The primary gate is the family-pooled cell per type, under the unchanged rule (accuracy ≤ majority + .05, with
     the baselines of changes 3–4).
   - Per-(family, kind) cells are diagnostics. They fail only if the Wilson 95% lower bound of accuracy exceeds the
     threshold.
7. **A3c, F2 Score.** §5 gave no Score limit for the F2 heuristics. Score uses the Choice limit (≤ .45).
8. **Chance floor.** Every shortcut gate's baseline is max(cross-validated majority, chance). Chance is the mean of
   1/K over the cell's rows. Exactly balanced labels (change 1) push the cross-validated majority below chance: for
   example .13 for 4-level Score in a 180-row cell.
9. **Gold positions.** When the rows of a group share one option set but have different golds (F3 twins, F2 cases),
   the build places the distinct golds at consecutive balanced positions, so A2's position balance holds for every
   row.
10. **Tokens.** Rows and state lengths stay as preregistered, and no row is cut (§2). The token total is reported
   from the freeze step's native Qwen3.5-0.8B counts and disclosed as above target.

## Not changed

- Designs, kinds and domains, and the sizes: 20,000 TRAIN rows and 2,400 dev rows.
- The seed-group isolation, the dev-id / dev-ood split and the `hs1-dev` panel.
- A1, A4, A5, A6 and A7.
- The validity rule and its budget (§7).
- No LLM text.
