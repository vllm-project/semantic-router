# HR2 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-hr2`, branch `xunzhuo/decision-2-training-data-hr2`. Prereg
`records/hr2-prereg-2026-09-30.md` (`6090213b4`), builder `25ec87a8d`, amendment 1
`records/hr2-prereg-amendment-1-2026-09-30.md`. Private data on node A under `/data/dev2/private/data/hr2/`
(mode 700): `raw/` (pinned publisher files), one run directory per build commit.

## Log (newest first)

- 2026-10-01 01:08 UTC+8 — **Blind review FAILED** (13 / 216 = 6.02%, CP95 upper 10.07%; weighted 6.50%; P1 / P2
  fail, P3 pass; F1 nothing to drop) → HR2 published flagged **not release-safe**, not tuned further; 13 gold-error
  rows dropped. Final v1 (TRAIN 27,725) → leak guard found 28 upstream rows (26 IPv4, 2 token-like) → amendment 3
  (`571e80b38`) → final v2: **TRAIN 27,697 (`0fd9b2db…`), DEV 1,615 (`697c3142…`)**, isolation / balance PASS,
  tree leak guard clean. Old outputs kept as `final-v1/`, `freeze-v1/`, `hf-v1/`. Next: card update, assemble, upload.

- 2026-10-01 00:25 UTC+8 — scans + pass 1 done. Full PI-v4: 4,556 groups flagged, 998 on quarantining roles;
  report-only hits disclosed (largest: HS3 vs A7q TRAIN 3,549 groups, HS3 vs v2 AHO H3 376, VitaminC vs H3 141).
  Quarantine lists: 1,202 groups (TRAIN + DEV) and 1,180 DEV-near-TRAIN groups. Pass 1: TRAIN 33,293 / DEV 2,101.
  **G4:** `indonli` FAIL (hypothesis-only .613 > .535) and `kob_boolq` FAIL (question-only .559 > .534) →
  dropped; 9 families pass. Amendment 2 (review of 9 families, 216 rows, ≤ 9 errors) committed before sampling.

- 2026-09-30 23:55 UTC+8 — continuation worker started (the first worker stopped silently ~12:01). Node A:
  `b1-25ec87a8db1e/` is empty (the first build stopped at its duplicate-id check; no candidate file exists).
  Kept the 3 uncommitted files (amendment 1 and its builder changes); PRM800K agreement now also records the
  alternative completions' ratings at every walked step (amendment 1 item 2 wording); builder registered in the
  eval-only guard test (G6); unit tests added.

- 2026-10-01 00:18 UTC+8 — tooling `3bf56c149` mirrored; `node_a.sh 3bf56c149… scans` running (run dir
  `b2-c54b8d444cac/`, logs in `logs/steps.log`). Done: G1 names PASS (0 hits); C1 source-term names over
  raw + candidates: 0 sources found (769,706 rows, 31 files); PI-hr2 inventory `f25c95bf…` (7 roles,
  11,552 rows); PI-hr2 scan 345 groups / 469 rows (HS3 338: Score5-DEV 259, HT-DEV v2 57, HT-DEV v1 27;
  N-gram hits, mostly 1–3 units); PI-v4 quarantining scan 1,128 groups / 1,520 rows (HS3: A7q AHO 689,
  Decision Bench v4 175, CSS15 156, v1 AHO a3 81; PRM800K: DBv4 11, public 231 3; VitaminC: v1 AHO a3 20);
  DEV-vs-TRAIN self-scan: 1,286 of 3,143 DEV groups near a TRAIN group (HS3 837, PRM 281, KoBEST 84).
  Waiting: full PI-v4 scan (report-only roles).
- 2026-10-01 00:15 UTC+8 — amendment 1 + fixes committed (`60a0b033d`), integration merged (`c54b8d444`,
  pushed, mirrored). **Candidate build done** on node A (`b2-c54b8d444cac/cand/`): TRAIN 34,874 rows /
  28,332 groups (`45b56010…`), DEV 3,879 / 3,143 (`b0498c1f…`); Choice 12,499, Noul 17,754, Score 8,500;
  0 group merges; build-level conflicting duplicates 2 (VitaminC). Audit tooling added (`hr2/audit.py`,
  `hr2/review.py`, `hr2/node_a.sh`); G1 names PASS locally (no hit in any registry / C1 / panel document).

## Next steps

1. ~~Commit amendment 1 + builder fixes, push, merge integration.~~ ~~Candidate build.~~
2. Commit the audit tooling, push, mirror; `node_a.sh <commit> scans` then `pass1` (as root on node A).
3. Audits G1–G8 (PI-v4 + PI-hr2 overlap, DEV self-scan, shortcut, balance, guard, isolation, tokens).
4. Finalize pass 1 → blind review (264 rows, R1 / R2 / R3) → finalize pass 2 with review drops.
5. `hf_headroom.sh`, private upload `m5/hr2/`, read-back hashes; results record, gist 02, merge.
