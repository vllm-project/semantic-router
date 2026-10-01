# IX1 follow-ups — state (keep current; newest first)

Assignment: coordinator follow-up to IX1, 2026-10-01 ~05:30Z. A: the shipped-runtime failure on two long ToolRet
requests on DEV2.0-27B (repro, root cause, runtime fix as a separate commit with tests, parity, release hand-off).
B: an Index diagnostic of the frozen M5-L128 soup (`95d61175…`; not releasable; informs 27B M6). C: a private note on
the release's CAL698 fit vs IX1's CAL700 fit. Worktree `vllm-sr-dev2-eval-index`; GPUs node C GPU1–7, node D GPU0–7
under `track=eval-ix1` leases (node C GPU0 never). Every result is an independent provisional 0.2.1 reproduction;
Index numbers, request content and answers stay in the private directories. This file holds only counts, hashes,
shapes and GPU-hours.

## Now

- 06:05Z — **A root-caused.** FLA 0.5.2's gated-delta forward computes some element offsets in 32-bit integers; a
  forward whose q / k / v tensors (padded tokens × value heads × 128) exceed 2^31 − 1 elements gives wrong rows,
  non-finite logits or a GPU page fault. DEV2.0-27B (48 value heads) passes the limit at 349,525 padded tokens; the
  two requests pad to 455,168 and 725,760. Fix (runtime forward token budget) written; tests pass on CPU.
  - Repro (private, node D `ixA/`): failing request subsets by question count; isolated kernel test on random
    tensors: rows whose offsets pass 2^31 differ, rows before are exact (48 heads) or BF16-rounding close (32 heads).
  - Other sizes: 0.8B, 2B, 4B, 9B (16,384-token limit) refuse the long question of the memory-fault request
    (`max_length_exceeded`) and answer the other one; 9B stays below its own limit (524,287) on that request.
  - C: the discrepancy is an IX1 bug, not the release's: 42 of the 290 CAL700 Noul rows list `true` before `false`,
    and the IX1 fit read their labels in the wrong order. Fixed in `calib.py` with a test; refits agree with the
    release's CAL698 Noul temperatures. Note in the private folders.
  - B: M5-L128 checkpoint copied to node D (7.0 GB, from node B).
  - Next: commit, mirror; synthetic regression on the unfixed and fixed runtime; both requests on the fixed runtime;
    parity; start B.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| A repro: shapes, growing subsets, other sizes, isolated kernel | node C GPU1–4, node D GPU2–3 | done |
| A fix in `v2/release/runtime/qwen.py` (separate commit) + CPU tests + GPU synthetic test | worktree | written |
| A verify: both requests, synthetic before / after, parity 0.8B / 9B / 27B, latency | nodes C / D | |
| A merge + release hand-off record | worktree | |
| B package (fixed runtime, M5 checkpoint), full panel, dual score, per-family vs A20r | node C / D | |
| C note | private folders | |
