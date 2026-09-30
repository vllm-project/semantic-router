# 27B MoE milestone (MoE-1): state (resume file)

Updated: 2026-10-01 00:35 UTC+8 (16:35Z). Worker: 27B MoE (branch `xunzhuo/decision-2-training-27b-moe`, worktree
`/home/xunliu/code/vllm-sr-dev2-27b-moe`, gist `06c-decision-2-27b-moe.md`). Assignment: COORDINATION 2026-09-30
23:50. **Continuation workers: read "Next steps" first.**

## Target, budget, GPUs

- Beat AutoJev-27B significantly (post-key v3 > 72.133, paired lower bound > 0, H not below); successor items 1–7 vs
  DEV2.0-27B = A20r (72.360); item 8 via the eval custodian.
- 60 GPU-h cap. Receipts: 0 (CPU only so far).
- GPUs (leases `track=27b-moe` written 15:49Z): node A GPU3, 4, 5; node B GPU6, 7. Never use node A GPU2 / node B
  GPU0, 1, 5 (27B M5 L128), node A GPU6–7 (9B M7), node A GPU0–1 / node B GPU2 (serving / eval), node B GPU3–4 (HR2).
- Platform rule: poll ≤ 30 min with one-line state updates; hand off through this file near 5 h.

## Done

- Licence and feasibility gate: PASS for all four bases (`moe-gate-2026-10-01.md`). Hub audit and base verification
  receipts: node A `/data/dev2/runs/27b-moe/logs/{hub-audit-1,verify-bases-nodeA}.json`; token admission
  `/data/dev2/runs/27b-moe/admission/`.
- Bases on node A `/data/dev2/models/moe/{gemma-4-26B-A4B-it,gemma-4-26B-A4B,Qwen3.5-35B-A3B,Qwen3.5-35B-A3B-Base}`
  (hash-verified). Node B download started 16:15Z (same script; verify with `verify_bases.py` before use).
- Code: shared-module MoE support (separate commit), MoE pipeline tar, drivers, collector, tests.
- Preregistration `moe-prereg-2026-10-01.md`.

## Running now

(nothing yet)

## Next steps (in order)

1. Commit, push, mirror to both nodes (`src/training/decision2/v2/common/mirror_to_node.sh --path src/training/decision2 node-a|node-b <sha>`).
2. Verify node B bases (`verify_bases.py` against `hub-audit-1.json`).
3. P0 probes: G-it on node A GPU3, Q-it on node A GPU5 (`moe-arm.sh a MOE-Git-probe 3 gemma-4-26B-A4B-it 20260926 <mirror> probe`).
   Pin the experts kernel per family (amendment 1), then admit/onestep/reload per cell, then full.
4. Stage A screen at checkpoint-0000892 (readouts on node B GPU7), rules in the prereg.

## Poll log (newest first)

- 16:35Z: gate done, prereg written; no GPU job yet.
