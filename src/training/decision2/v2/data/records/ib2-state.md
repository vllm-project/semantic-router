# IB2 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib2`, branch `xunzhuo/decision-2-training-data-ib2` (from
`origin/xunzhuo/decision-2-training` at `c6db2623d`). Prereg `records/ib2-prereg-2026-10-01.md` (`fff14438e`) with
amendment 1 (`0a2c8a2b9`) and amendment 2 (`fe3aa751f`); licence registry `records/license-registry-ib2.json`;
reviewer instructions `records/dq/reviewer-ib2.md`; tooling `v2/data/ib2/` (imports IB1's G0 matcher and review
scoring unchanged; runner `node_a.sh`, `IB2_RUN=c3`). Results `records/ib2-results-2026-10-01.md`; receipts and data
card `records/ib2/`. Private data on node A under `/data/dev2/private/data/ib2/` (mode 700): `raw/` (pinned publisher
files; `download.log`, `download.sh`) and the run directory **`c3/`** (candidates, scans, quarantine, rescans,
shortcut receipts, screen and review keys and answers, final files, upload tree, logs; superseded outputs kept as
`final-v1/`, `final-v2/`, `freeze-v1/`, `freeze-v2/`). `c1` (first build) and `c2` (the run behind amendment 2) are
kept as records and were never published. Local review packets and answers (private, mode 700):
`/home/xunliu/.cache/dev2-ib2-review/`.

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; IB1's worktree is not
touched.

**Status: DONE, RELEASE-SAFE.** Private
`llm-semantic-router/decision-2.0-training-data@c5dbdd0a88efe58059c6ece8ae2b181f9132619f`, `m6/ib2/`: TRAIN 24,518
rows (`ee137efa…`), DEV 1,272 (`ab009fb1…`), 5.42M native tokens, six families. Decisive blind review 5 / 216 =
2.31% (P1, P2, P3 pass).

## Log (newest first)

- 2026-10-01 15:20 UTC+8 — uploaded `m6/ib2/` (38 files, read-back SHA-256 equal, private before / after; revision
  `c5dbdd0a`, parent `82bf70a7` = IB1-r2's upload); receipts, data card and results record written. Remaining: merge
  into integration, gist 02.
- 2026-10-01 15:13 UTC+8 — final files: leak guard flagged 1 DEV row (IPv4-like string) → `final` re-run
  (`final-v1/`); the final-file re-scan flagged 1 group (6 TRAIN rows) → `final` re-run without re-sampling
  (`final-v2/`). G5, G7, leak guard PASS.
- 2026-10-01 15:05 UTC+8 — **stage R PASSED**: 5 / 216 = 2.31% [0.76, 5.32], weighted 1.41% [0.33, 2.81]; P3 pass
  (`ytspam` 3, `hover` 2); R1 / R2 0 splits (no R3). The 5 rows leave TRAIN.
- 2026-10-01 14:55 UTC+8 — stage R sample (216 rows, 36 per family; no screened row or group); the first sampler call
  failed on an IB1 round-2-only argument and was fixed in `2591cafb3` before anything was drawn.
- 2026-10-01 14:52 UTC+8 — stage S screen (one fresh reviewer, 60 rows): 0 disagreements; no family dropped.
- 2026-10-01 14:47 UTC+8 — run `c3` (builder `06dbdceca`, amendment 2): G0 2,977 groups (controls PASS), rescan 1 41
  groups, rescan 2 clean; **G4: `gsm2` passes; `fc_sel2`, `fc_args2` fail and are dropped**. Six families left.
- 2026-10-01 14:35 UTC+8 — amendment 2 pushed before any review sample: G4 on `c2` dropped `fc_sel`, `fc_args`,
  `gsm`, `qasc`, `arc`; `cnli` empty after G0 / G2.
- 2026-10-01 14:27 UTC+8 — scans on `c2`: G0 3,567 groups; positive controls PASS; G1 PASS; C1 names 0.
- 2026-10-01 14:23 UTC+8 — run `c2` (builder `feb0aab35`, amendment 1): TRAIN 52,946 / DEV 5,824 candidates.
- 2026-10-01 14:21 UTC+8 — amendment 1 pushed before any audit (Glaive slots removed; constructed `fc_rel`
  negatives).
- 2026-10-01 14:17 UTC+8 — run `c1` build (builder `12e8d8c97`): TRAIN 33,842 / DEV 3,721; superseded.
- 2026-10-01 13:57 UTC+8 — prereg pushed (`fff14438e`) before any row was converted.
- 2026-10-01 13:50 UTC+8 — source audit done; raw files on node A (HoVer dev / test deleted unread; ContractNLI
  `train.json` extracted only).
- 2026-10-01 13:30 UTC+8 — job started; worktree created.

## Next steps (for the coordinator)

1. Tracks may use `m6/ib2/` (release-safe) as one block against a matched-token control; the custodian C1 content
   recheck comes before any C1-scored model trained on IB2.
2. Uncovered families for a later round: function selection / arguments (a distractor design matched on frequency
   and form), knowledge MCQ (options without key signal), contracts, phishing email (a licensed legitimate-email
   corpus), sarcasm.
