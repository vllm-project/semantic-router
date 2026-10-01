# IB3 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib3`, branch `xunzhuo/decision-2-training-data-ib3` (from
`origin/xunzhuo/decision-2-training` at `518ffdcc7`). Prereg `records/ib3-prereg-2026-10-01.md` (`b30bcae5c`) with
amendment 1 (`bcaaa70f7`) and amendment 2 (`3be7cb06e`); licence registry `records/license-registry-ib3.json`;
reviewer instructions `records/dq/reviewer-ib3.md`; tooling `v2/data/ib3/` (runner `node_a.sh`, `IB3_RUN=d2`; downloads
`download.sh`). Results `records/ib3-results-2026-10-01.md`; receipts and data card `records/ib3/` (`audits/d1/` holds
the run-`d1` G4 receipts behind amendment 2). Private data on node A under `/data/dev2/private/data/ib3/` (mode 700):
`raw/` (pinned publisher files; `download.log`, `download.sh`), the run directory **`d2/`** (candidates, scans,
quarantine, rescans, shortcut receipts, screen and review keys and answers, final files, upload tree, logs) and `d1/`
(the run behind amendments 1 and 2; never published). Local review packets and answers (private, mode 700):
`/home/xunliu/.cache/dev2-ib3-review/`.

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0 + G0u); C1 registry datasets and the sealed directory are never read; Jev-derived
benchmark repositories are never sources; IB1 / IB2 files are not edited.

**Status: DONE, NOT RELEASE-SAFE.** Private
`llm-semantic-router/decision-2.0-training-data@c2401ab4e7aee01cdf76c98d066eaca6e9102355`, `m6/ib3/`: TRAIN 24,148
rows (`688b643e…`), DEV 1,762 (`e79d90d0…`), 4.65M native tokens, four families (`wpd`, `phiu`, `esci`, `mqa`).
Decisive blind review 29 / 216 = 13.43% (P1, P2, P3 fail); every audit passes.

## Log (newest first)

- 2026-10-02 00:00 UTC+8 — records, card and status; merge into `xunzhuo/decision-2-training` and gist 02 next.
- 2026-10-01 23:50 UTC+8 — uploaded `m6/ib3/` (38 files, read-back SHA-256 equal, private before / after; revision
  `c2401ab4`, parent `31b200a3`); `hf_headroom.sh` 47.49 GB free.
- 2026-10-01 23:47 UTC+8 — final: TRAIN 24,148 / DEV 1,762; final re-scan clean; leak guard 0; G5, G7 PASS.
- 2026-10-01 23:43 UTC+8 — **stage R FAILED**: 29 / 216 = 13.43% [9.18, 18.71], weighted 10.65%; `phiu` 12, `wpd` 8,
  `esci` 8, `mqa` 1; R1 / R2 2 splits (R3 on both). The 29 rows leave TRAIN.
- 2026-10-01 23:37 UTC+8 — stage S: 5 / 50 disagreements; `fdial2` 3 / 10 → dropped.
- 2026-10-01 23:34 UTC+8 — run `d2` (`3be7cb06e`): build 30,952 / 3,434; G0 628 groups (controls PASS), G0u 103
  (controls PASS), G2 145 + 2, re-scan 2 clean; G4 all five PASS on pass 2.
- 2026-10-01 23:27 UTC+8 — amendment 2 pushed before any screen (G4 on `d1` pass 3: `fdial`, `haluqa` FAIL; one
  passage-swap redesign `fdial2`).
- 2026-10-01 23:25 UTC+8 — amendment 1 pushed before any screen (G4 audit-copy fix; `maud` leaves; pass 3).
- 2026-10-01 23:10 UTC+8 — run `d1` (`c84136941`): build 32,508 / 3,426; scans, pass 1–2, re-scans 1–2.
- 2026-10-01 22:58 UTC+8 — tooling pushed (`c84136941`); prereg pushed (`b30bcae5c`) before any row was converted.
- 2026-10-01 22:40 UTC+8 — source audit: raw files on node A (MathQA archive deleted after extracting `train.json`
  only); aggregate counts only.
- 2026-10-01 22:25 UTC+8 — job started; worktree created.

## Next steps (for the coordinator)

1. Do not use `m6/ib3/` in a release candidate (NOT release-safe). `mqa` (1 / 54) is the clean family.
2. An IB3-r2 needs its own preregistered round and a fresh review: see the last section of the results record
   (maths-only block; phishing with visible-signal page evidence; stricter `esci`; claim-level grounding sources).
