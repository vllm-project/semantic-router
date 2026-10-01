# BF16-resident runtime rollout — state (keep current; newest first)

Assignment: coordinator note 2026-10-01 10:30 UTC+8 (b5f60b33): runtime-only revisions of the six released repos with
the shipped runtime's Linear weights kept BF16-resident, only with 0 answer changes on every scored prompt and on
mlx-diag; latency and memory old vs new. Worktree `/home/xunliu/code/vllm-sr-dev2-release` (re-created from the pushed
`xunzhuo/decision-2-training-release`), gist `07-decision-2-release.md`. GPUs: node A GPU0–1, node B GPU2–4 (shared
leases `owner.release-bf16r`). Started 2026-10-01 02:23Z.

## Now

- 05:35Z — **DONE: all six released** as runtime-only revisions, 0 answer changes pre and post on every scored prompt
  and on mlx-diag, weights byte-identical, every post-check ok. 0.6B `def20a1c`, 0.8B `e13a40f8`, 2B `56950ec5`, 4B
  `4f560ae5`, 9B `b4f65fa8`, 27B `4e89288d`. Record
  [`dev2-bf16-resident-2026-10-01.md`](dev2-bf16-resident-2026-10-01.md); 2.684 GPU-h; storage 52.13 / 100 GB. All
  `owner.release-bf16r` lease entries removed. Remaining: integration merge and gist 07 (this commit), nothing running.
- 04:28Z — **0.6B, 0.8B, 2B, 4B released** (0 answer changes pre and post on all four panels; weights byte-identical;
  every post-check ok): 0.6B `def20a1c`, 0.8B `e13a40f8`, 2B `56950ec5`, 4B `4f560ae5` (4B mlx-diag in its own
  no-upload run, 0 changes). Receipts copied under `<key>/release/` (4B also `4b/mlx-parity/`).
  - Running: node A GPU0 9B release (`dev2-bf16r-9B-20261001T042603Z`, from 04:26Z); node B GPU2 27B release
    (`dev2-bf16r-27B-20261001T042616Z`, from 04:26Z; log `bf16r-relB-20261001T042616Z.log`).
  - Next: fetch 9B / 27B receipts (`ops/fetch_receipts.sh`), fill the record (`ops/tables.py`), gist 07, merge.
- 03:45Z — **Releases running** from mirror `45e097444` (final specs + decisions, bench receipts under
  `dev2-bf16-resident-2026-10-01/<key>/bench/`).
  - Image tests (mirror `79bff5da9`): unit 5/5 OK; GPU fixture 3/3 packages pass (byte-identical answers).
  - Clean benches (one at a time per node; timed second pass): every tier 400/400 bit-identical answers, drift 0.
    p50 ms 0.6B 19.2→16.6, 0.8B 22.7→21.6, 2B 24.1→23.5, 4B 28.5→28.1, 9B 33.8→27.2, 27B 122.9→93.3; peak GiB
    2.32→1.50, 2.91→2.01, 7.14→4.57, 15.83→9.25, 29.79→16.83, 97.56→52.07. Overlapping first benches superseded.
  - node A: `bf16r-relA-20261001T034035Z.log` (GPU0: 0.6B → 0.8B → 2B → 4B (after mlx-only) → 9B) and
    `bf16r-mlx4b-20261001T034035Z.log` (GPU1: 4B mlx-diag-only parity, no upload).
  - Next: start the 27B release on node B GPU2 when the 9B release starts (`rollout.sh 27B --release --gpu 2`).
- 03:20Z — Runtime commit `5dc962b00` (shared module, separate commit); tests `fe666a1ac`, `9a8d72b91`; tooling
  `91d36d213`, `79bff5da9` (mirrored on both nodes). Hub `main` of all six verified = the table below; storage 52.02 /
  100 GB.
  - Image tests (mirror `91d36d213`): GPU fixture byte-identical answers BF16-resident vs FP32-master for tiny Qwen3,
    Qwen3.5 and LoRA packages; failures were CPU-only image limits (fixed in `9a8d72b91`, re-running).
  - First benches (20 warm-up; superseded): 0.6B / 0.8B 400/400 bit-identical answers; p95 dominated by first-use
    shape spikes → bench now times a second pass (`79bff5da9`). 27B bench failed: node B blob store unmounted (fixed).
  - Running: node A GPU0 tests → 0.6B / 0.8B / 2B benches; GPU1 9B preview → 9B / 4B benches; node B GPU2 27B bench.
  - Next: `make_bf16r.py final` from the copied bench compares → commit, mirror → releases in order on node A
    (4B mlx-only on GPU1 first); 27B release on node B started when the 9B release starts (its upload comes ~35 min in).
- 02:40Z — Context read; plan fixed (below). Implementing the runtime change.

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Runtime: BF16-exact backbone Linear weights BF16-resident on GPU (CPU unchanged), `bf16_resident=False` opt-out; tests | worktree | done (`5dc962b00`; tests 5 / 5 unit, 3 / 3 GPU fixture) |
| Specs + decisions (runtime-only path: superseded judgement carried forward, same gate profile) | worktree | done (`45e097444`) |
| Bench old vs new (400 typed-final items, p50/p95, peak memory), same GPU, fresh frozen-cache copies | node A GPU1; 27B node B GPU2 | done (400 / 400 identical each) |
| Release per tier, in order 0.6B, 0.8B, 2B, 4B, 9B, 27B: `release.sh --upload --collect --already-collected`, full parity pre/post | node A GPU0 (0.6B–9B); node B GPU2 (27B) | done, all six |
| Record, gist 07, merge into `xunzhuo/decision-2-training` | worktree | done |

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
