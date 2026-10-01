# BF16-resident runtime rollout — state (keep current; newest first)

Assignment: coordinator note 2026-10-01 10:30 UTC+8 (b5f60b33): runtime-only revisions of the six released repos with
the shipped runtime's Linear weights kept BF16-resident, only with 0 answer changes on every scored prompt and on
mlx-diag; latency and memory old vs new. Worktree `/home/xunliu/code/vllm-sr-dev2-release` (re-created from the pushed
`xunzhuo/decision-2-training-release`), gist `07-decision-2-release.md`. GPUs: node A GPU0–1, node B GPU2–4 (shared
leases `owner.release-bf16r`). Started 2026-10-01 02:23Z.

## Now

- 02:40Z — Context read; plan fixed (below). Implementing the runtime change.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Runtime: BF16-exact backbone Linear weights BF16-resident on GPU (CPU unchanged), `bf16_resident=False` opt-out; tests | worktree | in progress |
| Specs + decisions (runtime-only path: superseded judgement carried forward, same gate profile) | worktree | |
| Bench old vs new (400 typed-final items, p50/p95, peak memory), same GPU, fresh frozen-cache copies | node A GPU0/1; 27B node B GPU2 | |
| Release per tier, in order 0.6B, 0.8B, 2B, 4B, 9B, 27B: `release.sh --upload --collect --already-collected`, full parity pre/post | node A GPU0/1 (0.6B–9B); node B GPU2 (27B) | |
| Record, gist 07, merge into `xunzhuo/decision-2-training` | worktree | |

## Facts (verified)

| Tier | Current `main` | Spec | runtime_source → | vendor_source (kept) | Parity inputs | Cache |
| --- | --- | --- | --- | --- | --- | --- |
| 0.6B | `476fe984` | `dev2-0p6b-card-c1pk.json` (successor profile) | new | `2f21790ba` (builder tree of 476fe984) | `06b/m8/formal/m8-s5-b05` (+`-mlx`) | none (Qwen3 dense); tolerance 0 |
| 0.8B | `bede7938` | `dev2-0p8b-bf16.json` | `33de83cea` → new | `33de83cea` | `release/inputs/dev2-0p8b-t1/derived` | `m2-E8F-soup-nodeA-triton` (all 4 panels) |
| 2B | `a53cf66a` | `dev2-2b-bf16.json` | `33de83cea` → new | `33de83cea` | `dev2-2b-t1/derived` | `m3-S2T-soup-nodeA-triton` (all 4) |
| 4B | `fadbba4f` | `dev2-4b-bf16.json` | `f8f52c695` → new | `f8f52c695` | `dev2-4b-t1/derived` | `m4-N4XF-soup-nodeA-triton`; mlx-diag: `m4-N4XF-soup-triton` (separate no-upload run) |
| 9B | `e51f9881` | `dev2-9b-card-c1.json` | `2926952c` → new | `3277dec9` | `dev2-8b-t1/derived` | `9b/formal-m4/triton-cache` (all 4) |
| 27B | `5323310` | `dev2-27b-a20r-release.json` (successor) | `c68de36a` → new | `ff660322` | `27b/M4-A20r-soup/formal/output`; mlx `27b/m4-mlx/M4-A20r-soup/output` | frozen `M4-A20r-soup/formal/triton-cache` `f474e2e9` (all 4) |

- 9B runs on node A: its scored image `f83b1d10` and inputs exist only there (node B has image `dbe5f32b` and the 27B
  inputs). 27B runs on node B; 3 evidence files are relayed (sha-checked) to the same paths.
