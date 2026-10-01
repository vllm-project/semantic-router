# IX1 — Decision Index 0.2.1 campaign — state (keep current; newest first; no scores)

Assignment: COORDINATION 2026-10-01 10:55 (IX1, eval). Nodes C / D, 15 GPUs (node C GPU1–7, never GPU0; node D
GPU0–7). Worktree `vllm-sr-dev2-eval-index`, branch `xunzhuo/decision-2-eval-index`. Records:
[prereg](ix1-prereg-2026-10-01.md). Index values are private (node `/data/dev2/private/eval/index021/ix1/`, local
private folder); this file holds only steps, counts, hashes and GPU-hours.

## Now

- 2026-10-01 03:15Z — Setup. Packages downloaded at the pinned revisions on both nodes
  (`/data/dev2/models/ix1/<name>-<rev8>`); the pinned Qwen3.8-27B base copied node A → node D (and → node C) into
  `/data/dev2/hf-cache`. No GPU job yet.

## Plan / checklist

- [x] Worktree, prereg
- [ ] Suite verify C / D
- [ ] Released-package engine adapter + tests
- [ ] 86-request parity gate: 27B, 4B, 9B, 2B, 0.8B, 0.6B
- [ ] Full runs: 27B (D), 4B + 9B (C), then 2B, 0.8B, 0.6B
- [ ] Dual scoring + external comparison
- [ ] Contamination audit (CPU)
- [ ] Calibration study
- [ ] Gap analysis + data-plan input (private)

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| **Total** | **0** |

## Hand-off notes

- Node helpers: packages `/data/dev2/models/ix1/`, run root `/data/dev2/private/eval/index021/ix1/` (mode 700).
