# IX1 — Decision Index 0.2.1 campaign — state (keep current; newest first; no scores)

Assignment: COORDINATION 2026-10-01 10:55 (IX1, eval). Nodes C / D, 15 GPUs (node C GPU1–7, never GPU0; node D
GPU0–7). Worktree `vllm-sr-dev2-eval-index`, branch `xunzhuo/decision-2-eval-index`. Records:
[prereg](ix1-prereg-2026-10-01.md) (`662b9343f`, tie wording fixed in `a9845b9b9` before any GPU job),
**[results](ix1-results-2026-10-01.md)**, receipts [`ix1/`](ix1/). Index values are private (node
`/data/dev2/private/eval/index021/ix1/`, the private eval-artifacts dataset `ix1/`, the local private folder); this
file holds only steps, counts, hashes and GPU-hours.

## Now

- 2026-10-01 05:35Z — **IX1 DONE.** All six packages run, merged and dual-scored; parity, scorer and external
  agreement gates pass; contamination audit and calibration study done; private report written
  (`ix1-report-2026-10-01.md` in the private places). No GPU job running; every IX1 lease on node C / D is marked
  released (node C GPU0's K8s lease untouched). Private artifacts uploaded to the private eval-artifacts dataset
  (`ix1/`, two commits).
- 05:05Z — 27B tail: shards 3 and 5 stopped once four GPUs were free; their 14,532 unanswered rows ran as five
  `extra-r*` reruns on GPUs 1, 4, 5, 6, 7; merge verified one final record per row.
- 04:45Z — 27B: shard 3's skipped request also faulted alone on GPU0 (so a request defect, not a GPU defect) and was
  recorded as a final error; shard 2's `invalid_model_output` request failed again on resume. CAL passes for all six
  packages done (node C GPU1, node D GPU2).
- 04:25Z — 4B and 9B done and dual-scored; contamination audit done (item duplicates: 0.8B 1, others 0).
- 03:45Z — Full runs live (27B node D; 4B, then 9B / 2B / 0.8B / 0.6B by per-GPU chains on node C). Launch fixes
  on the way: shard-count variable clobbered (4B shards 1–6 exited at start, no rows, relaunched); stagger-gate
  race (27B shards 4–7 stopped during loading, restarted); lease race between chained shards (busy check now
  waits for the GPU to drain). No row ran twice; aborted directories are under `ix1/void/`.
- 03:40Z — Parity gates PASS, all six packages (86 / 86 `ok`, max |Δp| = 0.0).
- 03:30Z — Voided: the first 27B parity attempt ran transformers' reference `chunk_gated_delta_rule` (the launcher
  had replaced the image's `/opt/decision-fla` on `PYTHONPATH`). Fixed: kernel path kept, containers exit 97
  without the kernels, parity fails on a fallback log line.
- 03:15Z — Suite verify PASS on node C and node D; panel 120,226 rows (run-ID digest `6455d7be…`).
- 03:07Z — Container isolation: each container gets only its GPU's render node (node C GPU0 never visible).
- 02:58Z — Packages downloaded at the pinned revisions on both nodes; the pinned Qwen3.8-27B base copied from
  node A to node C / D.

## Checklist

- [x] Worktree, prereg
- [x] Suite verify C / D
- [x] Released-package engine adapter + tests
- [x] 86-request parity gate: all six packages
- [x] Full runs: 27B, 4B, 9B, 2B, 0.8B, 0.6B
- [x] Dual scoring + external comparison
- [x] Contamination audit (CPU) + override-adjusted values (private)
- [x] Calibration study (CAL fits; Index and own-panel rescoring)
- [x] Gap analysis + data-plan input (private)

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| Device smoke and kernel diagnostics | 0.01 |
| Voided 27B parity attempt; aborted shard starts (load only) | 0.10 |
| Parity gates (6 packages; reference + kit pass) | 0.30 |
| Full run 0.6B / 0.8B / 2B / 4B / 9B (7 GPUs each) | 0.89 / 1.21 / 1.34 / 2.17 / 2.71 |
| Full run 27B (8 GPUs, incl. resumes, the tail reruns and the stopped intervals) | 9.82 |
| CAL passes (6 packages) | ≈ 0.15 |
| **Total** | **≈ 18.7** |

## Hand-off notes

- **Open items:** (1) refit the Noul temperature on CAL698 (file `19cc1a8c…`, logits `dacdbba3…`; not found on node
  A in the obvious places) and compare with the CAL700 fit; (2) report the 27B batched-ToolRet runtime defect to
  the 27B / runtime owners (run IDs in the private receipts); (3) confirm the licences flagged "verify" in the
  private data-plan table and run `v2/data/overlap.py` on any chosen source against the Index rows and the
  held-out inventories before use; (4) the temporary transfer keys on node A / B are the coordinator's to remove.
- **Where things are (node C / D, `/data/dev2/private/eval/index021/ix1/`, mode 700):** `panel-7`, `panel-8`,
  `parity/<model>`, `runs/<model>/{shard-*,extra-*,merged*}`, `calib/` (CAL requests, labels, per-model answers and
  fits), `audit/` (training copies, outputs), `void/` (aborted attempts), `logs/`, `ix1-report-2026-10-01.md`.
  Packages: `/data/dev2/models/ix1/`. Node A: `/data/dev2/private/eval/ix1-calib/` (own-panel calibration effects).
- **Tools:** `v2/eval/ix1/` (see the results record). Launch detached with a single `nohup bash … > log 2>&1 <
  /dev/null &` command; a `cd … && nohup … &` list keeps the SSH session open. Do not `pkill -f` a pattern that
  also appears in the same SSH command line.
