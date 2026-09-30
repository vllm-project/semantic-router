# 9B M8 state (resume file)

Updated: 2026-09-30 17:55 UTC+8 (09:55Z) — M8 worker (started 17:03 UTC+8). Branch
`xunzhuo/decision-2-training-9b-m8` (worktree `vllm-sr-dev2-9b-m8`; merge-only into
`xunzhuo/decision-2-training`). Gist file `05b-decision-2-9b-m8.md` (created).

- Prereg `records/lux9b-m8-prereg-2026-09-30.md` (`0d5534325`, before any GPU job) + amendment 1
  `records/lux9b-m8-prereg-amendment-1-2026-09-30.md` (`2641b484e`, before any GPU job: node A GPU3–4 only after
  the coordinator's 17:15 reclaim of GPU2; KDX dropped; ≤ 2 finalists KD1, KD2).
- **Use mirror `2641b484ec0526a996aa49dbfca2b953e40c1f40` for every step** (node A, verified).
  `L=/data/dev2/src/2641b484ec0526a996aa49dbfca2b953e40c1f40-src_training_decision2/src/training/decision2/v2/9b/lux9b/m8`
- Control = M7's C line (read-only): `m7/C-m1..C-m5`, `m7/C5-a13/a12/a23` (readouts + `m7/screens/`); anchor
  R = `m7/ref-ka13`. M7's worker still owns GPU6–7 and M7's rules / formal.

## Status (node A `/data/dev2/runs/9b/m8/`)

| Item | State |
| --- | --- |
| Prompt build (CPU) | done: `data/m8-prompts/build` manifest `4777b12b…`; 12,443 rows / 6,198,202 tokens; S = 3,153 rows (`human.jsonl` `7efa4984…`) |
| Teacher parity | **PASS** (09:44–09:46Z, GPU3): 80 / 80 typed FINAL prompts, max abs Δp 0.0; cache source tree `f474e2e9…` (= the C1 spec's) |
| Chains | `m8-gpu3` (PID 3890649, launched 09:44Z; chain file `059d0458…`, 2,457 B) and `m8-gpu4` (PID 3891205, 09:44Z; `bd42fc9e…`, 2,036 B); both verified running with first log lines (chain rule) |
| GPU-hours | ≈ 0.1 at 09:50Z (`bash -c ". $L/lib.sh; m8_gpu_hours"`) |

## Next (a continuation worker)

Liveness / logs (never `pgrep -f`): `bash $L/alive.sh m8-gpu3`, `bash $L/alive.sh m8-gpu4`; step logs
`logs/m8-gpu{3,4}.log`; consoles `logs/*.console`; early rules `rules/early-D{1,2}.json`.

1. When both chains print `chain ... done` (or stop by an early rule): re-read the newest COORDINATION notes, then
   (CPU, node A) `cd /data/dev2/runs/9b/m8 && bash $L/rules.sh 2641b484ec0526a996aa49dbfca2b953e40c1f40 rules-lines`
   → `rules/rules-lines/{alpha-KD1,alpha-KD2,alpha-C5.report,finalists}.json` (priority KD1, KD2).
2. Per finalist NAME (e.g. `KD1-a12`): a lock record `records/lux9b-m8-formal-lock-NAME-2026-09-30.md` (checkpoint
   `m8/NAME-build/soup` model_sha256 from its `console.log`, calibration `m8/NAME-cal/calibration.json` SHA-256,
   rule-output hashes, `formal-m3/triton-cache` tree `af623300…`), committed and pushed **before** its formal run;
   then `SHA256SUMS` of `m8/NAME-build/soup` + `m8/NAME-cal` (M6 / M7 pattern).
3. Formal: upload `lux9b/m8/chains/m8-post.sh` (upload_chain.sh, size + SHA-256), then separately
   `bash $L/launch.sh m8-post-NAME /data/dev2/runs/9b/m8/chains/m8-post.sh SIZE SHA 2641b484ec0526a996aa49dbfca2b953e40c1f40 GPU NAME [--with-ref]`
   (GPU 3 or 4; exactly one finalist chain gets `--with-ref`).
4. Successor items 1–7 from `formal-m8/NAME.gates/successor.json` (and `NAME-16k-t1.gates/` if T = 1 ships). A
   passer of 1–7 → item-8 hand-off to the eval custodian (frozen package on node A + C1 spec; "Eval runners").
5. Result record `records/lux9b-m8-result-2026-09-30.md`, gist 05b entry, merge (`git merge
   origin/xunzhuo/decision-2-training`, tests, `git push origin HEAD:xunzhuo/decision-2-training`).

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m8/chains/<file>` (scp + size / SHA-256 check),
then separately `bash $L/launch.sh <chain> <remote file> <size> <sha> <mirror sha> [args]`.
