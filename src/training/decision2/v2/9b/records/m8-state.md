# 9B M8 state (resume file)

Updated: 2026-09-30 18:50 UTC+8 (10:50Z) — M8 worker (started 17:03 UTC+8). Branch
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
| Teacher shards + build | done 10:11Z: three shards exit 0 (12,443 / 12,443 valid, identity `2e074511…`, T = 1, 32K); build `data/m8-kd/build` manifest `8455dbb0…`: D1 teacher `cd531c3e…` (12,443 rows), D2 teacher `b7985a44…` (3,153 rows), train = C's `eb55dbb2…` |
| Teacher diagnostics (train rows; never selection) | accuracy vs gold A20r / own-Lux: all .803 / .736, Choice .899 / .853, Noul .846 / .791, Score .615 / .500; Noul yes-rate gold .498, A20r .546, own-Lux .526 |
| Member 1 | preflights PASS (D1, D2); recipe checks PASS (zero-step vs `m7/pf-C-m1-zero`, full run vs `m7/C-m1`: only the teacher differs) |
| **D2 early rule** | **STOP** (`rules/early-D2.json` `9f354342…`): `rule_precedence` 245 < C-m1 286 − 4 (typed T .752 vs .791; HT-DEV v2 .530 vs .515; hop .996 vs .992; SELECT .875 vs .859). No further D2 member, no KD2 line |
| **D1 early rule** | **CONTINUE** (`rules/early-D1.json` `5e3ecd26…`): T .834 vs .791; C / N / S 685 / 309 / 340 vs 678 / 286 / 302; RP 309 vs 286; HT-DEV v2 .526 vs .515; hop .996 vs .992; SELECT .873 vs .859 (report only: clean gold-no .735 vs .714, PAWS-X-6 .853 vs .843) |
| Incident (CPU, no GPU cost) | both early scripts wrote `screens/C-m1-e1/htdev2.json` in the same second; D1's crashed on "file exists" (10:38:43Z) and chain `m8-gpu3` ended. The deterministic CPU rule was then completed on the unchanged inputs (10:45Z). No training or readout was repeated |
| Chains now | `m8-gpu3b` (PID 3924769; file `f6a16fe5…`, 1,489 B): D1-m2, D1-m4, then the KD1 line; `m8-gpu4b` (PID 3925107; `a1e2a13b…`, 949 B): D1-m3, D1-m5 (GPU4 freed by D2's stop). Launched 10:46Z from mirror `2641b484e`; both verified running (containers up) |
| GPU-hours | ≈ 2.4 at 10:46Z |

## Next (a continuation worker)

Liveness / logs (never `pgrep -f`): `bash $L/alive.sh m8-gpu3`, `bash $L/alive.sh m8-gpu4`; step logs
`logs/m8-gpu{3,4}.log`; consoles `logs/*.console`; early rules `rules/early-D{1,2}.json`.

1. When `m8-gpu3b` prints `chain m8-gpu3b done` (KD1 line; `m8-gpu4b` ends after D1-m5): re-read the newest COORDINATION notes, then
   (CPU, node A) `cd /data/dev2/runs/9b/m8 && bash $L/rules.sh 2641b484ec0526a996aa49dbfca2b953e40c1f40 rules-lines`
   → `rules/rules-lines/{alpha-KD1,alpha-C5.report,finalists}.json` (KD2 has no line: D2 stopped early; at most one finalist).
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
