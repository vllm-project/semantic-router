# Decoder Milestone 8 — results (DEV2.0-4B: cross-size distillation from DEV2.0-27B; 2026-09-30)

Preregistration [`dec-m8-prereg-2026-09-30.md`](dec-m8-prereg-2026-09-30.md) (`a629a6ce2`, before any GPU job); data
lock [`dec-m8-datalock-2026-09-30.md`](dec-m8-datalock-2026-09-30.md) (parts 1 / 2: `862c99b1e`, `4d8b40d48`).
Development readouts are never release scores; formal runs are post-key same-panel comparisons. Nothing was uploaded;
C1 was not opened. Aggregate files (finalists, successor evaluations, teacher and data manifests, early rules, hs1-dev)
are in [`dec-m8-results-2026-09-30/`](dec-m8-results-2026-09-30/).

## Bottom line

**No successor. DEV2.0-4B stands.** Distilling DEV2.0-27B (A20r) into the released 4B soup members did not beat the
released model; no finalist passed items 1–7, so there is no C1 candidate, no item-8 hand-off and no release hand-off.

| Finalist (slot) | v3 | vs DEV2.0-4B (T = 1 run, 63.151) | T / H | vs Decider 4B | vs Jet v6.2 | Items 1–7 |
| --- | ---: | --- | --- | --- | --- | --- |
| `4b-D1-a2_3` (1): ⅔ D1 soup + ⅓ DEV2.0-4B | 58.91 | −4.25 [−5.13, −0.33] | .651 / .533 | −2.98 [−7.32, +1.86] | −1.47 [−5.41, +3.59] | fail 1, 5, 6(b) |
| `4b-D2-a1_3` (2): ⅓ D2 soup + ⅔ DEV2.0-4B | 60.91 | −2.24 [−2.79, +0.71] | .681 / .545 | −0.97 [−6.17, +3.52] | +0.54 [−3.78, +4.86] | fail 1, 6(b) |
| `4b-C-a1` (3): C soup (the matched control) | 57.95 | −5.20 [−6.18, −0.02] | .669 / .502 | −3.93 [−7.60, +2.19] | −2.42 [−5.37, +4.05] | fail 1, 5, 6(b) |
| *DEV2.0-4B (bar)* | *63.15* | | *.688 / .580* | *+1.27* (M4) | *+2.78* (M4) | |

- Items passed by all three: 2 (H not significantly below: D1 [−.061, +.022], D2 [−.043, +.017], C [−.092, +.012]),
  3 (no type collapsed), 4 (card-eligible mlx-diag: D1 −.0027 [−.0089, +.0033], D2 +.0030 [−.0017, +.0077],
  C −.0005 [−.0076, +.0065]), 6(a) (0 exposed groups) and 7 (public 231 171 vs 171 for each, p 1). Item 5: D2 +4.44
  [+0.40, +9.26] vs the adopted Nox 1.0 (pass); D1 +2.44 [−1.23, +7.72] and C +1.48 [−1.44, +8.10] fail.
- All three ship T = 1 (the 23:15 rule rejected the CAL698 16K fit); formal node B `dbe5f32b`, copy of
  `formal/m5/cache-frozen`, 0 new cache entries; relayed gold-free and scored on node A.

## Development (16K, node B, paired against `4b-I` = the released weights)

| Line | α 1 | α ⅔ | α ⅓ | Pick |
| --- | --- | --- | --- | --- |
| L-D1 (A20r KL on every row) | T .701 (C / N / S 483 / 263 / 375); HT-DEV v2 −.027 [−.040, −.014] **FLAG** | T .710; −.016 TIE | T .714; −.009 TIE | α ⅔ |
| L-D2 (A20r KL on human-rated rows, gold on typed) | T .708 (500 / 272 / 360); −.034 [−.047, −.020] **FLAG** | T .717; −.021 **FLAG** | T .714; −.010 TIE | α ⅓ |
| L-C (N4XF recipe continued) | T .748 (562 / 277 / 358); −.017 [−.027, −.006] TIE | T .743; −.012 TIE | T .725; −.006 TIE | α 1 |
| `4b-I` | T .704 (501 / 264 / 362); H_dev2 .539 | | | |

- Every point passed the typed type / family floors, the Noul `rule_precedence` floor (all ≥ 263 vs the 260 floor)
  and the Score floor (Score5-typed-DEV check half: no flag anywhere; top share .37–.43 vs `4b-I` .39). No pick was
  dropped by the proxy rule (P 61.9 / 63.0 / 63.8). Early rule (member 1, SELECT700 vs C-m1 .8975): D1 .9000 and D2
  .9033, both continue. All nine members finished; no preflight failed; nothing was rerun.
- **hs1-dev (report only):** quote adoption C .793 / D1 .721 / D2 .693 (`4b-I` .775; ideal .50); false yes on unmet
  conditions .280 / .268 / .295 (`4b-I` .262).

## Findings

1. **A stronger teacher's soft targets lowered human transfer instead of raising it.** At matched tokens and α 1, HT-DEV
   v2 fell more under A20r targets than under the recipe's own-Lux targets (D1 − C −.011, D2 − C −.017), and most when
   the teacher acted only on the human-rated rows (D2). Formal human transfer of every finalist is below the bar
   (−.035 to −.078, none significant). This extends the 9B / 27B finding (06:10, 06:40 notes: teacher soft targets
   on human rows are not a lever) to a teacher with a far larger gap. A20r's CSS15 H (.584) is level with the student's
   (.580); its advantage is typed (T .896 vs .688) and C1, which the slice's soft targets did not transfer.
2. **The teacher is better on the slice rows, but the student did not absorb it as held-out skill.** On the rows both
   teachers cover, A20r matches gold .890 / .859 / .633 (C / N / S) against own-Lux's .834 / .798 / .511, and they
   disagree on about a seventh of the Choice / Noul rows and a third of the Score rows. KD kept typed DEV level with
   `4b-I` (the control inflated it by +.044), yet every finalist lost typed FINAL (T .651–.681 vs .688; Noul
   687–717 vs 734, Score 168–182 vs 185, Choice 586–609 vs 582).
3. **The continuation itself costs quality.** The matched control (one pass over 21% of the recipe at the recipe's peak
   LR) is the worst finalist (−5.20), repeating M6b / M7: more steps of the released recipe lose typed FINAL and
   human transfer at 4B. The distilled picks score higher than the control's pick (D2 α ⅓ −2.24, D1 α ⅔ −4.25), but
   they sit at smaller α, so the formal D − C contrast is confounded with dilution toward the released model.
4. **A20r does move the quoted-conclusion behaviour** (hs1-dev adoption .775 → .69–.72, toward .50) without moving
   public 231 (171 for every finalist).
5. **Screens behaved as designed.** HT-DEV v2 FLAGged the α 1 distilled soups; the TIE picks that went formal all
   lost formal H in the same direction. The A20r teacher runtime reproduced its scored run exactly (80 / 80, drift 0.0).

## Tooling and incidents

- New (`v2/dec/ops/m8/`, 12 tests in `v2/dec/tests/test_m8.py`): top-up data builder with a render-equality check of
  every label prompt, A20r label runner on the unchanged scored collector with a parity smoke, teacher converter,
  member chains with the early rule, line readouts with HT-DEV v2 and Score5-typed-DEV, node-A scoring and rules, M8
  formal wrapper (the M6 / M7 procedure), batched compressed relays (18 files in 20 s instead of 5.5 min) and a
  direct node-link helper (not needed: no C1 candidate). No shared module changed.
- The decoder worktree was missing locally and was re-created from the pushed branch (`e066e0505`); a `.venv-agent`
  link was added for the pre-commit hooks (untracked).
- One workstation command ran twice (the first copy detached and kept running): the node-B mlx-diag step and the
  relay refused the duplicate ("exists"), so nothing was collected twice; the C formal copy was verified afterwards
  by content manifest (`1d58e8b1…` on both nodes).

## GPU-hours: 3.90 of 24

| Item | GPU-h |
| --- | ---: |
| A20r labels (node A GPU0 / GPU1 shared, GPU5; 3 shards incl. load and smoke) | 0.748 |
| Members (node B GPU3 / GPU4): C 0.605, D1 0.619, D2 0.625 (member-1 preflights included) | 1.849 |
| Lines, references, hs1-dev (co-tenant readouts) | 0.822 |
| Formal (3 CAL fits, 3 smokes, 3 collections, 3 mlx-diag) | 0.481 |
| **Total** | **3.900** |

Run directories: node B `/data/dev2/runs/dec/m8/` (data, arms, soups, lines), `/data/dev2/runs/dec/formal/m8/`
(packages and formal runs); node A `/data/dev2/runs/dec/m8/label/` (A20r labels), `m8/lines/4b/diag/` (HT-DEV v2,
Score5-typed-DEV), `m8/select/`, `/data/dev2/runs/dec/formal/m8/` (scored runs, `successor/`). The A20r targets
(`m8/teacher-a20r/D1/teacher.jsonl`, every slice row, keyed by id and input hash) stay on node B for the 2B / 0.8B
worker.

## Next

- The 4B decoder should not spend more on teacher or recipe top-ups: M5–M8 found no lever among more recipe tokens,
  round-2 blocks or a much stronger own teacher. HR2 (new human-rated data, COORDINATION 11:30) is the remaining
  data lever; if it lands, a 4B test should add it as a block with a matched-token control, screened with HT-DEV v2.
- For the parallel 9B M8 and 2B / 0.8B M8-small distillation runs: expect the same pattern (HT-DEV v2 decline under
  A20r targets, largest with human-row-only KD); their α lines toward the incumbent are the protection.
