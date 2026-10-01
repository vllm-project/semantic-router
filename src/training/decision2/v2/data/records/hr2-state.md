# HR2 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-hr2`, branch `xunzhuo/decision-2-training-data-hr2`. Prereg
`records/hr2-prereg-2026-09-30.md` (`6090213b4`) with amendments 1–3; builder `v2/data/hr2/` (`build.py`,
`families.py`, `audit.py`, `review.py`, `node_a.sh`). Results `records/hr2-results-2026-10-01.md`; receipts
`records/hr2/`. Private data on node A under `/data/dev2/private/data/hr2/` (mode 700): `raw/` (pinned publisher
files) and the run directory `b2-c54b8d444cac/` (candidates, scans, review keys, final files, upload tree, logs).

**Status: round 1 DONE, flagged NOT release-safe.** Private `llm-semantic-router/decision-2.0-training-data@afc3bc1e1d6849058f6fafdfbc3dfe007d067400`,
`m5/hr2/`: TRAIN 27,697 rows (`0fd9b2db…`), DEV 1,615 (`697c3142…`). Blind review 13 / 216 = 6.02% (P1 / P2 fail).

**HR2-r2: DONE, flagged NOT release-safe.** Amendment 4 (`records/hr2-prereg-amendment-4-2026-10-01.md`,
`84a2fb714`): VitaminC and Allegro dropped (licence uncertain), `hs3_help` (C1) and PRM800K boundary yes rows (C2)
dropped; fresh review 288 rows: 13 errors, 4.51% [2.43, 7.60] (P1 pass), weighted 8.02% (P2 fail), `hs3_pref` 7 / 48
(P3 fail); no fix rule. Private `…@16ea6cf7b3c8229eb51ecb5782bab7e10d93aa01`, `m5/hr2/` (38 files, replaces round 1;
round 1 stays at `afc3bc1e`): TRAIN 16,095 rows (`820cb5a7…`, 10.61M tokens), DEV 954 (`dff8f4ea…`). Results
`records/hr2-r2-results-2026-10-01.md`; receipts `records/hr2/r2/`. Runner `v2/data/hr2/node_a_r2.sh`; run directory
`/data/dev2/private/data/hr2/b2-c54b8d444cac-r2/` (review keys, packets, answers and error ids stay there).

## Log (newest first)

- 2026-10-01 11:58 UTC+8 — **r2 published**: final TRAIN 16,095 / DEV 954 (G5 PASS, G7 PASS, leak 0; final-files
  recheck 0 / 0); `hf_headroom.sh` 47.98 GB free; upload `16ea6cf7` on `afc3bc1e` with `--delete '*'` under
  `m5/hr2/`: read-back 37 / 37 SHA-equal, remote folder = registry, other 750 files unchanged, private before and
  after. Records written (results, card, status, receipts). Local review copies to be removed after the merge.

- 2026-10-01 11:47 UTC+8 — **round-2 blind review FAILED** (288 rows; R1 / R2 four fresh subagents, R3 fresh on 3
  splits): 13 / 288 = 4.51% [2.43, 7.60] (P1 pass), population-weighted 8.02% [3.83, 12.82] (P2 fail), `hs3_pref`
  7 / 48 (P3 fail); `prm_step` 3, `eth_cs` 1, `eth_just` 1, `eth_util` 1, `eth_deon` 0. κ .986. No fix rule
  (amendment 4): HR2-r2 is published flagged `release_safe: false`; the 13 round-2 errors leave TRAIN. Next: final,
  final-audits, card, upload.
- 2026-10-01 11:26 UTC+8 — **r2 recheck converged** (runner `fd3cd9e40`): iterations dropped 163 + 7 DEV, 41 + 1
  DEV, 7 + 0 groups; iteration 4 recheck 0 / 0. Pass TRAIN 16,107 / DEV 954; G1 PASS, C1 terms 0, G4 no failure,
  leak guard 0. (A duplicated ssh invocation of the iteration loop was found and its loop shell stopped; the running
  `audits` finished normally; long node stages now run detached with `nohup`.) Next: round-2 sample.
- 2026-10-01 11:15 UTC+8 — **r2 pass + audits (iteration 1)** at `84a2fb714`: boundary list 364 TRAIN / 39 DEV
  candidate rows; pass TRAIN 16,329 / DEV 961. G1 PASS, C1 terms 0, G4 no family fails, leak guard 0. **The
  quarantine recheck found 163 groups (155 `hs3_pref`, 8 `prm_step`; all N-rule, mostly A7q AHO, Score5-DEV,
  DBv4) and 7 DEV groups near TRAIN.** Cause: `v2.data.overlap` exempts a matched unit as boilerplate when it occurs
  in ≥ 5 distinct candidate groups, so units exempt in round 1's larger candidate set are no longer exempt once
  VitaminC, Allegro and `hs3_help` leave. Amendment 4 §E applies (drop whole before sampling); because a drop can
  lower other units below 5 groups, pass + audits repeat (`next-iter`) until the recheck finds none. Nothing else
  changes; no row has been sampled.
- 2026-10-01 11:03 UTC+8 — **r2:** tooling `874e60e46` (tests 403 data + guard pass); error analysis on node A
  (`analyze`, 4 s): the r2 drops remove 9 of 13 round-1 errors; kept reviewed rows 4 / 139 = 2.88%; expected TRAIN
  ≈ 16,303 rows / 10.9M tokens, DEV ≈ 961. Licence evidence re-read (VitaminC LICENSE per-article terms; Allegro
  snapshot without licence). Amendment 4 written; committed before `pass`.

- 2026-10-01 01:18 UTC+8 — merged into `xunzhuo/decision-2-training` (`24b024a8b`, fast-forward); gist 02 entry
  added (read back identical). Local review copies removed (packets, answers and keys stay on node A under
  `review/`). No HR2 container left. **Worker done.**
- 2026-10-01 01:15 UTC+8 — uploaded (34 files, read-back SHA-256 equal, private before / after); results record
  and receipts written.
- 2026-10-01 01:08 UTC+8 — **blind review FAILED** (13 / 216 = 6.02%, CP95 upper 10.07%; weighted 6.50%; P1 / P2
  fail, P3 pass; F1 had nothing to drop) → published flagged not release-safe, not tuned further; 13 gold-error
  rows dropped. Final v1 (TRAIN 27,725) → the leak guard found 28 upstream rows (26 IPv4, 2 token-like) →
  amendment 3 (`571e80b38`) → final v2 TRAIN 27,697 / DEV 1,615; isolation and balance PASS; tree leak guard clean.
  Old outputs kept as `final-v1/`, `freeze-v1/`, `hf-v1/`, `hf-v2/`.
- 2026-10-01 00:36 UTC+8 — review sample (216 rows, 9 families) → R1 / R2 (two packets each) → 3 splits → R3.
  The first sampling run hit a U+2028 reader bug (nothing sampled; fixed `8f2d51db1` / `347d81720`).
- 2026-10-01 00:25 UTC+8 — scans + pass 1: quarantine 1,202 groups, 1,180 DEV groups near TRAIN; G4 dropped
  `indonli` (hypothesis-only .613 > .535) and `kob_boolq` (question-only .559 > .534); amendment 2 (review of the
  9 remaining families, 216 rows, ≤ 9 errors) committed before sampling (`91b60b04f`).
- 2026-10-01 00:18 UTC+8 — tooling `3bf56c149` mirrored; scans: G1 names PASS; C1 source terms 0 found; PI-hr2
  345 groups; PI-v4 quarantining 1,128 groups; full PI-v4 4,556 flagged (998 on quarantining roles); DEV self-scan
  1,286 of 3,143 DEV groups.
- 2026-10-01 00:05 UTC+8 — candidate build on node A (`b2-c54b8d444cac/cand/`): TRAIN 34,874 (`45b56010…`), DEV
  3,879 (`b0498c1f…`); Choice 12,499 / Noul 17,754 / Score 8,500.
- 2026-09-30 23:55 UTC+8 — continuation started (the first worker stopped ~12:01). `b1-25ec87a8db1e/` was empty (the
  first build stopped at its duplicate-id check). Kept the 3 uncommitted files (amendment 1 + builder changes);
  PRM800K agreement also covers alternative completions; builder registered in the guard test; tests added
  (`60a0b033d`); integration merged (`c54b8d444`).

## Next steps (for the coordinator)

1. Gist 02 entry and the merge into `xunzhuo/decision-2-training` for HR2-r2 (worker, in progress).
2. Coordinator decision: HR2-r2 (`16ea6cf7`) for experiments only (not release-safe; it supersedes round 1 for
   experiments), or a separately preregistered round. Cleanest candidates: ETHICS (6 / 288 errors over both rounds)
   and PRM800K (4 / 72); HelpSteer3 preferences are the open problem (8 / 72; all round-2 errors at medium or low
   reviewer confidence), for example a human adjudication of the disagreements or a margin rule tested on fresh rows.
3. Licence: amendment 4 applied "ShareAlike not established compatible with Apache-2.0 releases → drop"; the same
   question applies to CC BY-SA sources already in released mixtures (the 22:20 card policy credits them).
4. Before any C1-scored model trained on HR2 / HR2-r2: the custodian C1 content recheck (inside the rescan roots).
