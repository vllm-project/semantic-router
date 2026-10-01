# IX1 — Decision Index 0.2.1 campaign — state (keep current; newest first; no scores)

Assignment: COORDINATION 2026-10-01 10:55 (IX1, eval). Nodes C / D, 15 GPUs (node C GPU1–7, never GPU0; node D
GPU0–7). Worktree `vllm-sr-dev2-eval-index`, branch `xunzhuo/decision-2-eval-index`. Records:
[prereg](ix1-prereg-2026-10-01.md) (`662b9343f`, tie wording fixed in `a9845b9b9` before any GPU job). Index values
are private (node `/data/dev2/private/eval/index021/ix1/`, local private folder); this file holds only steps,
counts, hashes and GPU-hours.

## Now

- 2026-10-01 04:25Z — **4B and 9B done, merged, dual-scored.** Both: 120,226 rows = 120,224 `ok` + 2 `unsupported`
  (`max_length_exceeded`; the external report's count), 0 errors; port vs kit `87d4650b` gate PASS (per-benchmark
  ≤ 5e-5, headline ≤ 0.002); **0 of 38 benchmarks differ from the external report by more than 1 skill point**
  (values private). GPU-h: 4B 2.17, 9B 2.71. 2B running on node C (then 0.8B, 0.6B by per-GPU chain); 27B on node D.
  - **27B device faults:** two requests (ToolRet, 32 questions, ≈ 18–19.5K tokens) aborted their processes with a
    GPU memory access fault (no-retry page fault) at 03:42Z, while four other 27B processes were loading; one
    resumed fine past its row (transient), the other faulted again on the same row on the same GPU (node D GPU3)
    with nothing else loading. That shard resumes without that one request (`skipped.json`); the request will be
    rerun alone on another GPU to tell a row defect from a GPU defect. One further 27B ToolRet request
    (≈ 15K tokens) returned `invalid_model_output` (a non-finite or malformed head output); it stays an error unless
    a resume answers it.
  - **Contamination audit (CPU, node C) done:** each model's training files from its release / gate records,
    hash-checked (0.6B six M6 files; 0.8B `d1dc33fc…`; 2B `13804ac6…`; 4B `c7d51219…`; 9B `a66131b1…`; 27B A20r
    `a20.train.jsonl` `4aa0dc96…`; plus the Decision 1.0 decoder corpora behind the 0.8B–9B parents). Planted control
    200 / 200. **Item duplicates (all question-specific text present in training): 0.6B 0, 0.8B 1, 2B 0, 4B 0, 9B 0,
    27B 0; 1.0 corpora 1.** Familiar-text matches (a premise, evidence passage or board position seen in training,
    without the hypothesis / claim) are reported separately (ANLI, HoVer, ChessBench).
- 03:45Z — **Full runs live.** 27B: 8 shards on node D GPU0–7 (`runs/DEV2.0-27B`), ≈ 1.4 rows/s per GPU,
  ETA ≈ 06:40Z. 4B: 7 shards on node C GPU1–7 (`runs/DEV2.0-4B`), ≈ 4–5 rows/s per GPU, ETA ≈ 04:35Z; then 9B, 2B,
  0.8B, 0.6B on node C. Kit runner + adapter at mirror `a898de429`, frozen caches from the parity runs.
  - Load profile, 27B: ≈ 98 GB host RAM per process for ≈ 30 s, then ≈ 6 GB; four concurrent loads left ≥ 770 GiB
    available. Launch fixes on the way: the shard-count variable was clobbered (4B shards 1–6 exited at start with
    no rows; relaunched with `--only`), and the stagger gate raced the container's start marker (27B shards 4–7
    stopped during loading and restarted); both are fixed in the launcher (`--only`, host-side `launched` marker).
    No row was run twice; the aborted shard directories are under `ix1/void/`.
- 03:40Z — **Parity gates PASS, all six packages:** 86 / 86 `ok`, identical choices, **max |Δp| = 0.0**
  (bit-identical; kit runner + adapter vs the package's own `system_one`, same GPU class and image, frozen Triton
  cache). No compatibility row was unsupported.
- 03:30Z — **Voided:** the first 27B parity attempt (launcher `a9845b9b9`) ran the transformers reference
  `chunk_gated_delta_rule` because the launcher's `PYTHONPATH` replaced the image's `/opt/decision-fla`. Fixed in
  `63f18bdd9` (kernel path kept; containers exit 97 without the kernels; parity fails on a fallback log line). The
  voided run is kept under `ix1/void/`; it is not used.
- 03:15Z — **Suite verify PASS on node C and node D:** kit `19ad28ec` `suite verify` (uncompressed rows
  `b2b56d6f…`, added `7429f3c9…`, exclusions `331df32d…`, all match) and the port's `verify-suite` (150,317
  scoreable, `upstream_021_row_ids_matched`, keep `ca4f8903…`). **Panel:** 38 Index benchmarks, **120,226 rows**
  (267,668 Choice + 14,700 Noul questions), run-ID digest `6455d7be…`; gold-free shards 8-way (node D) and 7-way
  (node C); compatibility sample `1356ceaf…` (86 rows, 44 benchmarks).
- 03:07Z — Container isolation check: each container gets only its GPU's render node (node C GPU0 is never
  visible); device count 1, expected PCI bus.
- 02:58Z — Packages downloaded at the pinned revisions on both nodes (`/data/dev2/models/ix1/<name>-<rev8>`); the
  pinned Qwen3.8-27B base copied node A → node D and → node C into `/data/dev2/hf-cache` (≈ 1 min each).

## Plan / checklist

- [x] Worktree, prereg
- [x] Suite verify C / D
- [x] Released-package engine adapter + tests (`publication/decision_index_release_engine.py`)
- [x] 86-request parity gate: 27B, 4B, 9B, 2B, 0.8B, 0.6B
- [ ] Full runs: 27B (D, running), 4B + 9B (C, done), then 2B (running), 0.8B, 0.6B
- [ ] Dual scoring + external comparison (4B, 9B done)
- [x] Contamination audit (CPU); override-adjusted values pending the last merges
- [ ] Calibration study
- [ ] Gap analysis + data-plan input (private)

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| Device smoke and kernel diagnostics | 0.01 |
| Voided 27B parity attempt | 0.06 |
| Parity gates (6 packages; reference + kit pass) | 0.30 |
| Aborted shard starts (load only) | 0.04 |
| Full run 4B (7 GPUs) | 2.17 |
| Full run 9B (7 GPUs) | 2.71 |
| Full runs 27B, 2B, 0.8B, 0.6B | running |
| **Total so far** | **≈ 5.29** |

## Hand-off notes

- Tools: `v2/eval/ix1/` (`launch.sh parity|run`, `panel`, `native_ref`, `parity`, `merge`). Packages
  `/data/dev2/models/ix1/`, run root `/data/dev2/private/eval/index021/ix1/` (mode 700): `panel-7`, `panel-8`,
  `parity/<model>`, `runs/<model>`, `logs/`.
- Leases: `owner` files with `track=eval-ix1` on node C GPU1–7 and node D GPU0–7 while jobs run.
- Launch detached with `nohup bash launch.sh … > log 2>&1 < /dev/null &` as a single command (a `cd … && nohup … &`
  list keeps the SSH session open).
