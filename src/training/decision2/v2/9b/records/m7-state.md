# 9B M7 state (resume file)

Updated: 2026-09-30 11:50 UTC+8 (03:50Z) — continuation worker (started 11:25 UTC+8). **Q stopped by its early
rule; only chain `m7-gpu7` (C) is running; nothing needs relaunching.** Branch `xunzhuo/decision-2-training-9b`
(merge-only into `xunzhuo/decision-2-training`; integration already holds `471efec85`).

- Prereg `records/lux9b-m7-prereg-2026-09-30.md` (`3bbdd4349`, before any GPU job) + amendment 1
  `records/lux9b-m7-prereg-amendment-1-2026-09-30.md` (`317a8614e`, before any Q GPU job).
- Mirrors on node A (created, verified): code `7168f0864…` (chains `m7-gpu6` / `m7-gpu7`), code `df6dcccc9…`
  (amendment-1 tooling: Q build, `early.sh ARM`, `rules.sh` with Q5; chain `m7-gpu6q`). Runtime / formal runner
  `3277dec9d` (M6's).
- **Use mirror `df6dcccc946799f9bc52d99bef099105dcf35986` for every remaining step** (rules, formal): it is a superset
  of `7168f0864` (only the Q additions differ).

## Status (node A `/data/dev2/runs/9b/m7/`)

| Item | State |
| --- | --- |
| Preflights P / C / Q (member 1) | P PASS, C PASS, Q PASS (all gates) |
| **P** (PN1-r2 ×2) | **STOPPED by its early rule** (`rules/early-P.json`): clean gold-no .047 vs C-m1 .714 (passes), SELECT .874 vs .859 (passes), **hop .949 vs .992 (−.042 < −.03)**. No further members, no line. Chain `m7-gpu6` ended |
| **C** (K-mix control) | chain `m7-gpu7` (PID 3826223): C-m1, C-m2 done; C-m3 training from 03:34Z, then C-m4..m5 (~16 min each), C5 line (~55 min), K5-a12 and Lux 1.0 screens. ETA ≈ 05:35Z |
| **Q** (PN1-r2 ×1, amendment 1) | **STOPPED by its early rule** (`rules/early-Q.json` `4df5d09b…`, 03:43Z): clean gold-no .052 vs C-m1 .714 (passes), SELECT .862 vs .859 (passes), **hop .945 vs .992 (−.047 < −.03)**; PAWS-X-6 yes .515, accuracy .966 (216 updates, 03:25–03:41Z). No further members, no line. Chain `m7-gpu6q` ended 03:43Z; GPU6 idle |
| R = K-a13 re-read | done: typed DEV answers identical to M6 (1,600 / 1,600); HT-DEV v2 on the M7 path = eval reference (Δ 0.0000, H_dev2 .5636, TIE); PN1 dev hop .987, clean gold-no .262, PAWS-X-6 yes .626, acc .867; MLX-DEV-9B Noul-ML .846, Choice-ML .921, Score-ML .547 |
| GPU-hours | 1.74 at 03:44Z (`m7_gpu_hours`); projection ≈ 5 of 24 (C line + at most one formal run) |

Data: P `e6a74ea9…`, C `eb55dbb2…`, Q `82496294…` (C rebuilt byte-identical by the Q build); exposure P / C / Q
`groups: []`; MLX-DEV-9B panel `d07fe654…` (6,147 rows). Failed CPU builds before the prereg (no output, no GPU) are
under `logs/failed-builds/`.

## Next (the continuation worker)

Liveness / logs (never `pgrep -f`):
`L=/data/dev2/src/df6dcccc946799f9bc52d99bef099105dcf35986-src_training_decision2/src/training/decision2/v2/9b/lux9b/m7`;
`bash $L/alive.sh m7-gpu7`, `bash $L/alive.sh m7-gpu6q`; step logs `logs/m7-gpu7.log`, `logs/m7-gpu6q.log`; consoles
`logs/*.console`. GPU-hours: `bash -c ". $L/lib.sh; m7_gpu_hours"`.

1. When both chains print `chain ... done`: re-read the newest COORDINATION notes, then (CPU, node A)
   `cd /data/dev2/runs/9b/m7 && bash $L/rules.sh df6dcccc946799f9bc52d99bef099105dcf35986 rules-lines`
   → `rules/rules-lines/{alpha-C5,finalists}.json` (P5 and Q5 have no readouts and are skipped; priority P5, Q5, C5).
   Q stopped early, so only C5 can yield a finalist.
2. Per finalist NAME (e.g. `Q5-a12`): a lock record `records/lux9b-m7-formal-lock-NAME-2026-09-30.md` (checkpoint
   `m7/NAME-build/soup` model_sha256 from its `console.log`, calibration `m7/NAME-cal/calibration.json` SHA-256,
   rule-output hashes, `formal-m3/triton-cache` tree `af623300…`), committed and pushed **before** its formal run;
   then `SHA256SUMS` of `m7/NAME-build/soup` + `m7/NAME-cal` (M6 pattern).
3. Formal: upload `lux9b/m7/chains/m7-post.sh` (upload_chain.sh, size + SHA-256), then in a separate call
   `bash $L/launch.sh m7-post-NAME /data/dev2/runs/9b/m7/chains/m7-post.sh SIZE SHA df6dcccc946799f9bc52d99bef099105dcf35986 GPU NAME`
   (GPU 6 or 7, whichever is idle). It runs formal.sh (runs under `/data/dev2/runs/9b/formal-m7/`), hs1-dev,
   ship_cal (23:15 rule) and derive_t1.
4. Successor items 1–7 from `formal-m7/NAME.gates/successor.json` (and `NAME-16k-t1.gates/` if T = 1 ships), as in
   the prereg; item 6 = the exposure receipts above. A passer of 1–7 → item-8 hand-off to the eval custodian (frozen
   package on node A + C1 spec, see COORDINATION "Eval runners"; Tatoeba content-recheck note for Q).
5. Result record `records/lux9b-m7-result-2026-09-30.md`, gist 05 entry, merge (`git merge origin/xunzhuo/decision-2-training`,
   tests, `git push origin HEAD:xunzhuo/decision-2-training`).

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m7/chains/<file>` (scp + size / SHA-256 check),
then separately `bash $L/launch.sh <chain> <remote file> <size> <sha> <mirror sha> [args]`.

## Notes

- The workstation's foreground shell stopped returning exit statuses at ~03:02Z; background commands still worked.
  Nothing on the nodes was affected.
- The P early readout shows the lever is very strong in family (PN1 dev accuracy .656 → .965 at member level). Q
  halves the dose; if Q also stops, the natural M8 design is a lower continuation LR or a hop-balanced PN1 block.
