# IB1 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib1`, branch `xunzhuo/decision-2-training-data-ib1` (from
`origin/xunzhuo/decision-2-training` at `3ee34c625`). Prereg `records/ib1-prereg-2026-10-01.md`; licence registry
`records/license-registry-ib1.json`; reviewer instructions `records/dq/reviewer-ib1.md`. Private data on node A under
`/data/dev2/private/data/ib1/` (mode 700): `raw/` (pinned publisher files; `download.log` holds every revision and
commit) and run directories. Index suite rows (read-only): `/data/dev2/private/eval/index021/suite-0.2/`.

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; HR2-r2's worktree is
not touched.

**Status: prereg written; build not started.**

## Log (newest first)

- 2026-10-01 11:55 UTC+8 — source audit done (16 sources admitted; HellaSwag repo DMCA-blocked, RAGTruth / VAST /
  FinEntity / Humicroedit / ANLI / ACORD excluded); raw files downloaded on the node A host; iSarcasmEval test files
  deleted unread; prereg, licence registry and reviewer instructions written.
- 2026-10-01 11:07 UTC+8 — job started; worktree created.

## Next steps

1. Commit and push the prereg (before any row is converted).
2. Builder `v2/data/ib1/` (families, Index guard, audits, review, node runner) with tests; mirror; build on node A.
3. Scans (G0 Index rows, G1, G2, G3), pass 1, G4; screen review (stage S); decisive review (stage R).
4. Final files, freeze, isolation, tokens, stats; HF upload `m6/ib1/`; results record; gist 02; merge.
