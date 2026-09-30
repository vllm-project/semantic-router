# HR2 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-hr2`, branch `xunzhuo/decision-2-training-data-hr2`. Prereg
`records/hr2-prereg-2026-09-30.md` (`6090213b4`), builder `25ec87a8d`, amendment 1
`records/hr2-prereg-amendment-1-2026-09-30.md`. Private data on node A under `/data/dev2/private/data/hr2/`
(mode 700): `raw/` (pinned publisher files), one run directory per build commit.

## Log (newest first)

- 2026-09-30 23:55 UTC+8 — continuation worker started (the first worker stopped silently ~12:01). Node A:
  `b1-25ec87a8db1e/` is empty (the first build stopped at its duplicate-id check; no candidate file exists).
  Kept the 3 uncommitted files (amendment 1 and its builder changes); PRM800K agreement now also records the
  alternative completions' ratings at every walked step (amendment 1 item 2 wording); builder registered in the
  eval-only guard test (G6); unit tests added.

## Next steps

1. Commit amendment 1 + builder fixes, push, merge integration.
2. Mirror the pushed commit to node A; `build` in `decision20-train-fast:host2` with `--network none`.
3. Audits G1–G8 (PI-v4 + PI-hr2 overlap, DEV self-scan, shortcut, balance, guard, isolation, tokens).
4. Finalize pass 1 → blind review (264 rows, R1 / R2 / R3) → finalize pass 2 with review drops.
5. `hf_headroom.sh`, private upload `m5/hr2/`, read-back hashes; results record, gist 02, merge.
