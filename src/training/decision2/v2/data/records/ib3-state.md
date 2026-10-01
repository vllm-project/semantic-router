# IB3 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib3`, branch `xunzhuo/decision-2-training-data-ib3` (from
`origin/xunzhuo/decision-2-training` at `518ffdcc7`). Prereg `records/ib3-prereg-2026-10-01.md`; licence registry
`records/license-registry-ib3.json`; reviewer instructions `records/dq/reviewer-ib3.md`; tooling `v2/data/ib3/`
(downloads `download.sh`). Private data on node A under `/data/dev2/private/data/ib3/` (mode 700): `raw/` (pinned
publisher files; `download.log`, `download.sh`).

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0 + G0u); C1 registry datasets and the sealed directory are never read; Jev-derived
benchmark repositories are never sources; IB1 / IB2 files are not edited.

**Status: IN PROGRESS** — run `d1` at G4 (pass 3); screen and review next.

Commits: prereg `b30bcae5c`; tooling `c84136941` (build, scans, pass 1–2, re-scans 1–2 ran from this mirror);
amendment 1 `bcaaa70f7` (G4 audit-copy fix, `maud` leaves, pass 3 is the sample pass; pass 3, G4, re-scan 3 and every
later step run from this mirror). Run directory `/data/dev2/private/data/ib3/d1/` on node A.

## Log (newest first)

- 2026-10-01 23:25 UTC+8 — amendment 1 pushed before any screen sample; pass 3 + G4 + re-scan 3 started.
- 2026-10-01 23:16 UTC+8 — pass 1, re-scan 1 (42 groups, G2), pass 2, re-scan 2 (11 groups, G2). G4 on pass 2 audited
  only `wpd` / `phiu` (both PASS); the renamed audit copies of the other families were refused (input hash) → amendment 1.
  `maud` lost 100 of its contract groups to G2 (near-duplicates of Decision Bench v4 rows) → amendment 1 drops it.
- 2026-10-01 23:10 UTC+8 — scans: G0 754 groups (controls 2,000 / 2,000 both), G0u 103 groups (controls 500 / 500
  both), G1 PASS (preregistered lineage hits only), C1 names 0, PI-v4 / PI-ib3 quarantine 292 groups.
- 2026-10-01 23:05 UTC+8 — build `d1` (`c84136941`): TRAIN 32,508 / DEV 3,426 candidates.
- 2026-10-01 22:58 UTC+8 — tooling pushed (`c84136941`); prereg pushed (`b30bcae5c`) before any row was converted.
- 2026-10-01 22:40 UTC+8 — source audit: raw files on node A (MathQA archive deleted after extracting `train.json`
  only); aggregate counts only.
- 2026-10-01 22:25 UTC+8 — job started; worktree created.

## Next steps

1. With `IB3_SAMPLE_PASS=3`: screen (packets to `/home/xunliu/.cache/dev2-ib3-review/`, one fresh reviewer),
   screen-score, review (R1, R2; R3 on splits), score, final, re-scan final, leak, hf-assemble, `hf_headroom.sh`,
   hf-upload; results record, card, status; gist 02; merge into integration.
