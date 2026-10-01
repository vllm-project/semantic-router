# IB3 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib3`, branch `xunzhuo/decision-2-training-data-ib3` (from
`origin/xunzhuo/decision-2-training` at `518ffdcc7`). Prereg `records/ib3-prereg-2026-10-01.md`; licence registry
`records/license-registry-ib3.json`; reviewer instructions `records/dq/reviewer-ib3.md`; tooling `v2/data/ib3/`
(downloads `download.sh`). Private data on node A under `/data/dev2/private/data/ib3/` (mode 700): `raw/` (pinned
publisher files; `download.log`, `download.sh`).

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0 + G0u); C1 registry datasets and the sealed directory are never read; Jev-derived
benchmark repositories are never sources; IB1 / IB2 files are not edited.

**Status: IN PROGRESS** (prereg written; tooling next).

## Log (newest first)

- 2026-10-01 23:15 UTC+8 — prereg, licence registry and reviewer instructions written (before any row is converted).
- 2026-10-01 22:40 UTC+8 — source audit: raw files on node A (MathQA archive deleted after extracting `train.json`
  only); aggregate counts only.
- 2026-10-01 22:25 UTC+8 — job started; worktree created.

## Next steps

1. Push the prereg; write `v2/data/ib3/` (families, build, audit, index guard with G0u, review, runner).
2. Build on node A (`IB3_RUN=d1`), scans, pass 1, rescans, pass 2 + G4, screen, review, final, upload, records.
