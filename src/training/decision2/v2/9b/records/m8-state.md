# 9B M8 state (resume file)

Updated: 2026-09-30 17:45 UTC+8 (09:45Z) — M8 worker (started 17:03 UTC+8). Branch
`xunzhuo/decision-2-training-9b-m8` (worktree `vllm-sr-dev2-9b-m8`; merge-only into
`xunzhuo/decision-2-training`). Gist file `05b-decision-2-9b-m8.md`.

- Prereg `records/lux9b-m8-prereg-2026-09-30.md` (frozen before any M8 GPU job).
- Code mirror on node A: `60cecd9995e818b2af0494d8edf3670a3a525853` (tooling; use it for every step).
- Control = M7's C line (read-only): `m7/C-m1..C-m5`, `m7/C5-a13/a12/a23` with readouts and `m7/screens/`;
  anchor R = `m7/ref-ka13`.

## Status (node A `/data/dev2/runs/9b/m8/`)

| Item | State |
| --- | --- |
| Prompt build (CPU) | done: `m8/data/m8-prompts/build` manifest `4777b12b…`; 12,443 rows / 6,198,202 tokens; S = 3,153 rows |
| Chains | not launched yet |
| GPU-hours | 0 |

## Next

1. Upload chains `lux9b/m8/chains/m8-g{2,3,4}.sh` (upload_chain.sh), launch each separately with
   `bash $L/launch.sh m8-gpuN /data/dev2/runs/9b/m8/chains/m8-gN.sh SIZE SHA 60cecd9995e818b2af0494d8edf3670a3a525853`
   where `L=/data/dev2/src/60cecd9995e818b2af0494d8edf3670a3a525853-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8`.
2. Liveness: `bash $L/alive.sh m8-gpu2` (PID + containers); step logs `m8/logs/m8-gpuN.log`.
3. When all three chains print `chain ... done`: `bash $L/rules.sh 60cecd99… rules-lines` → finalists; locks; `m8-post.sh`.
