# Decoder M16 — state

Branch `xunzhuo/decision-2-training-dec-m16`, worktree `vllm-sr-dev2-dec-m16`. Prereg `9923e8b51`, ops `dfb86c08d`
(mirror on nodes A / B), data lock `5eb61705f`. M15 (nodes E / F) untouched.

## 2026-10-01 17:47Z

- Prep done on both nodes (17:40–17:44Z): all 8 release / arm pairs pass lineage; MLX-DEV panels equal M15's; M14's
  reference readouts staged; leases A GPU3–5 / B GPU2–4 `track=dec-m16`.
- Chains launched 17:47Z from `dfb86c08d`:
  - A GPU3 `08b-C0-a` MLX-DEV, then `08b-RAUP` a25 / a50 / a75; GPU4 `08b-RASD` ×3; GPU5 `08b-RA` ×3.
  - B GPU2 `2b-C0-b` MLX-DEV, `2b-RAUP` ×3, `2b-RASD` ×3; GPU3 `4b-LH-b` MLX-DEV, `4b-LHA10UP` ×3; GPU4 `2b-RA` ×3,
    `4b-LHA10SD` ×3.
- GPU-h so far: 0 (prep was CPU only).

## 2026-10-01 18:12Z

- Builds and readouts running cleanly (≈ 10 min per 0.8B / 2B point). Read: 0.8B a25 and a50 of all three lines;
  2B `2b-RA` / `2b-RAUP` a25, a50; 4B `4b-LHA10UP-a25`. Reference MLX-DEV reads done on both nodes (17:49–17:50Z).
  No failure.
- Node B formal masters copied into `formal/m16/masters` (manifests equal). `m16-fscore.sh` 2B bar fix (bar-t1's
  mlx-diag run is node A's `formal/m3/m3-S2T-soup-mlx`, the DEV2.0-2B weights; the release dir holds only its score)
  committed here before any formal data exists.
