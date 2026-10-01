# IX1 follow-ups — state (keep current; newest first)

Assignment: coordinator follow-up to IX1, 2026-10-01 ~05:30Z. A: the shipped-runtime failure on two long ToolRet
requests on DEV2.0-27B (repro, root cause, runtime fix as a separate commit with tests, parity, release hand-off).
B: an Index diagnostic of the frozen M5-L128 soup (`95d61175…`; not releasable; informs 27B M6). C: a private note on
the release's CAL698 fit vs IX1's CAL700 fit. Worktree `vllm-sr-dev2-eval-index`; GPUs node C GPU1–7, node D GPU0–7
under `track=eval-ix1` leases (node C GPU0 never). Every result is an independent provisional 0.2.1 reproduction;
Index numbers, request content and answers stay in the private directories. This file holds only counts, hashes,
shapes and GPU-hours.

## Now

- 07:50Z — C extra: the IX1 calibration study rerun with the fixed `calib.py` on all six sizes (old outputs kept as
  `*-v1-labelbug`; 12 rescorings, both scorers pass each). Results in the private C note only.
  - A20r baseline for B: its merged results with the two failed long requests replaced by the fixed-runtime
    reruns (`runs/DEV2.0-27B/merged-budget`): 120,226 / 120,226 ok, scorers pass.
  - Tool `v2/eval/ix1/family_delta.py` (`ddd977c54`; per-family difference of two compare files, private output).
  - B: M5-L128 shards about 45% (6,300–8,800 of ~15,000 rows each).
- 07:35Z — **A done.** 27B parity passes (0 changes on all four panels, drift 0). Benches old vs new, 400 / 400
  bit-identical each: p50 ms 0.8B 25.6 → 23.2, 9B 28.7 → 28.6, 27B 98.9 → 98.8; request peak unchanged. Hand-off
  record `v2/release/records/dev2-runtime-forward-budget-2026-10-01.md` (`5517093c1`). Merged into
  `xunzhuo/decision-2-training` at `fea2f016b` (merge `0ad342663`; `qwen.py` conflict with the label-token
  readout limited to the two new constructor arguments; runtime + IX1 tests pass on the merged tree).
  - A20r's two failed long requests rerun on the fixed 27B package (`DEV2.0-27B-budget` diagnostic entry,
    `b02395659`): 2 / 2 ok, private run `runs/DEV2.0-27B-budget/extra-toolret`.
  - B: M5-L128 8 shards running (≈ 75 rows / min / shard; expected end ≈ 10:00–10:30Z).
- 07:15Z — **Budget halved to 2^30 gated-delta elements** (`e876fbefc`; 27B 174,762 padded tokens, 9B 262,143,
  0.8B 524,287): forwards between about 2^30 and 2^31 elements also hung or crashed in the HIP runtime on 27B
  (cause not isolated; passed only with serialized kernels). Restaged 0.8B / 9B / 27B packages under `fix2/` (only
  `decision2/qwen.py` differs from `e13a40f8` / `b4f65fa8` / `4e89288d`).
  - synthetic request, 32 questions, long prompt 14,224 (twice) / 18,000 / 22,676 tokens: all pass (2 forwards
    each, 0 invalid, 32 / 32 equal to the question asked alone, max drift ≤ 0.015).
  - both real requests (twice each): valid, 32 / 32 equal to alone (max drift 0.0034 / 0.0022), peak 95.2 / 93.4 GiB.
  - parity (typed-final 1,600, css15 6,547, public231 231, mlx-diag 2,275): 0.8B and 9B pass (0 changes, 0 missing,
    0 input mismatches, max drift < 1e-15); 27B running (node D GPU2). Latency benches (runtime_bench, 400 typed-final
    after 400 warm-up, old vs new on one GPU) queued: 0.8B → 9B on GPU5, 27B on GPU2 after its parity.
  - B running: M5-L128 (`fix2/` package, manifest `c5a031ce…` on both nodes) 8 shards: node D GPU0, 1, 4, 6, 7, 3
    (shards 0–3, 6, 7), node C GPU6, 7 (shards 4, 5). Node C GPU1–5 are now leased by the 9B M9 track.
  - C note copied to node D `ixC/` (sha `6d322161…`).
  - The 06:55Z entry below was written at about 06:40Z.
- 06:55Z — Fix `8e6bdfc33` mirrored (`7b57c77c4`, then `12b27df7a` with the launcher's diagnostic packages).
  Verification so far (DEV2.0-27B `4e89288d` restaged with the fixed `qwen.py`; only that file differs):
  - synthetic request (generated text, 32 questions, long prompt 14,224 / 22,676 tokens): **unfixed runtime fails**
    (2 invalid answers, 9 of 32 differ from the question asked alone; the 22,676 one hits a GPU page fault);
    **fixed runtime at 22,676 passes** (2 forwards, 32 / 32 equal to alone, max drift 0.009).
  - real memory-fault request on the fixed runtime: valid, 32 / 32 equal to alone (max drift 0.0075).
  - **open:** at the 14,224-token shapes the fixed runtime's first sub-batch (24 rows, 341,376 padded tokens) ends
    in a host-side segfault in the HIP runtime (3 of 3 runs). With `AMD_SERIALIZE_KERNEL=3` or a synchronize after
    every module both pass. Running: smaller budgets (300,000 / 200,000) and the unfixed runtime at 24 rows.
  - C done: note in the private folders (local `private/ixC/`).
  - B waiting on A: the M5-L128 package is restaged (fixed runtime, checkpoint `95d61175…`, 27,497,508,864 loaded
    parameters); panel and frozen cache relaying to node C.
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
| A fix in `v2/release/runtime/qwen.py` (separate commits `8e6bdfc33`, `e876fbefc`) + CPU tests + GPU synthetic test | worktree | done |
| A verify: both requests, synthetic before / after, parity 0.8B / 9B / 27B, latency | node D | done |
| A merge + release hand-off record | worktree | done (`fea2f016b`, `dev2-runtime-forward-budget-2026-10-01.md`) |
| B package (fixed runtime, M5 checkpoint), full panel, dual score, per-family vs A20r | node C / D | running |
| C note | private folders | done |
