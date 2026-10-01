# IB2 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib2`, branch `xunzhuo/decision-2-training-data-ib2` (from
`origin/xunzhuo/decision-2-training` at `c6db2623d`). Prereg `records/ib2-prereg-2026-10-01.md`; licence registry
`records/license-registry-ib2.json`; reviewer instructions `records/dq/reviewer-ib2.md`; tooling `v2/data/ib2/`
(imports IB1's G0 matcher and review code unchanged). Private data on node A under `/data/dev2/private/data/ib2/`
(mode 700): `raw/` (pinned publisher files; `download.log`, `download.sh`).

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; IB1's worktree is not
touched.

**Status: prereg written; build not started.**

## Log (newest first)

- 2026-10-01 14:50 UTC+8 — source audit done; raw files on node A (HoVer dev / test deleted unread; ContractNLI
  `train.json` extracted only). Not admitted: SpamAssassin (no licence), FEVER (per-article Wikipedia clause),
  Sarcasm Headlines (no licence), OpenBookQA (unknown), MMLU auxiliary train (RACE NC), xLAM (research-only note),
  ToolACE / API-Bank (ToolRet / Index sources), IBM claim_stance (HT-DEV v1 panel source), Nazario (no licensed
  negative class).
- 2026-10-01 13:30 UTC+8 — job started; worktree created.

## Next steps

1. Push the prereg; write `v2/data/ib2/` (families, build, audit, node runner) and tests; mirror to node A.
2. Build, scans (G0 + controls, G1, G2, G3), pass 1, rescan, pass 2, G4; stage S; stage R; final; leak; upload.
