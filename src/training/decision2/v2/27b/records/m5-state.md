# ~27B M5 state (resume file)

Updated: 2026-09-30 07:55 UTC+8 (M5 worker; step 1, HT-DEV v2 references)
Branch: `xunzhuo/decision-2-training-27b` (worktree `/home/xunliu/code/vllm-sr-dev2-27b`; merge-only into
`xunzhuo/decision-2-training`). Gist file: `06-decision-2-27b.md`. Assignment: COORDINATION 2026-09-30 07:20.

## Target and budget

- **Milestone 5 goal:** post-key v3 paired lower bound > 0 vs AutoJev-27B (72.133) with human transfer not below.
  A20r (72.36) is level: lower bound −1.60.
- **Budget:** 72 GPU-h cap (per-attempt caps in the preregistration). GPUs: node B GPU5–7, node A GPU2–4; node B
  GPU0–2 by lease when the F-b codec worker is not using them. Formal runs on node B, image `dbe5f32b`, DEV2.0-27B's
  scored cache `03b172f1…`. Nothing to HF.
- **GPU-h used:** 0 (step 1 starting).

## Infrastructure

- **Private node link (node B → node A):** a temporary ed25519 key on node B (`/data/dev2/tmp/27b-m5-xfer/`, mode
  700) is authorized on node A for rsync only: `from=<node B private address>,restrict,command="/usr/bin/rrsync
  /data/dev2/xfer/27b-m5"` (tagged `dev2-27b-m5-xfer-temp`; node A's previous `authorized_keys` backed up as
  `authorized_keys.bak.27b-m5-<UTC>`). Measured ≈ 850 MB/s (96 GB in 113 s). **Remove the key line and the node B
  key directory at milestone end.**
- **Staged on node A** (`/data/dev2/xfer/27b-m5/stage/`), per-file SHA-256 equal to node B:
  - `M4-A20r-soup/soup/checkpoint` (9 files, 1.8 GB) and `M4-A20r-soup/package/` (calibration `518e19cd…`);
  - `m4b/A1-soup/checkpoint` (F-b, 35 files, 96 GB FP32; hash list `84437b35…`) and `m4b/F-b/package/`
    (calibration `c225e10a…`).
- Code mirror on node A: `22ac0a893` (integration head at the start).

## Step 1: HT-DEV v2 references (node A GPU2–4)

- Driver `v2/27b/m5/m5-htdev2.sh`, specs `v2/27b/m5/htdev2-refs-nodeA.json`. Outputs
  `/data/dev2/runs/27b/m5/htdev2/<key>/` on node A. Keys: `dev2-27b-f1`, `m4-a20r-soup`, `m4b-f-b`, `autojev27`,
  `eikos27b`. Reference for M5 gates: `m4-a20r-soup`.

## Next

- Preregistration (≤ 3 finalists) before any training GPU job.
