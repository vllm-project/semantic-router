# Decoder M18 state (prereg dec-m18-prereg-2026-10-02.md)

Index values stay private (node private dirs and the program's private folder); this file records operations only.

## 2026-10-02T05:10Z

- Part A: all eight BF16 copies built and restaged (identity, loaded-count and calibration checks pass). Index pools
  running: node A GPU1 / 2 / 7 (panel-3, the four 0.8B candidates), node E GPU0-3 / 6-7 (panel-8, 2b-RAUP-a75,
  2b-UPRA, 2b-RAUP-a50), node F GPU4-5 (panel-8, 2b-UPRAa75). Every parity gate so far 86 / 86.
- `m18-cand.sh`: 2B candidates are built and copied on node E (`WORK=e`), 0.8B on node A; node-to-node copies start
  on node A / B (their key is authorized on C-F only); the pool is started detached.
- Part B ops (`ops/m18`): `m18-prep.sh` (data on node F, in the decoder image), `m18_swap.py` (M17's builder),
  `m18-chains.sh` / `m18-arm.sh` / `m18-launch.sh` (node F GPU4-5, after the node's Index pool), `m18-soup.sh`,
  `m18_gpuh.py`. IB3-r2 TRAIN fetched to node F from the dataset revision `1c8452da` (hash matches `9d92d92a…`).
- Next: build the data, commit the data lock, write READY-m18.json, launch the chains.

## 2026-10-02T05:50Z

- Part A 2B: all four runs scored on node C (scorer gates pass); bootstraps vs DEV2.0-2B running on node C.
  `2b-UPRAa75`: one `invalid_model_output` row, scored with `--allow-errors` (amendment 1).
- Part A 0.8B: `08b-RAUP-a75`, `08b-UPRAa75` scored, bootstraps vs the current Eos release (`M16-08b-RA-a75`) running;
  `08b-UPRA` shards running on node A; `08b-RAUP-a50` dropped (amendment 1; node A pool stopped before its parity).
- Hub: Eos `main` = `3de61185` (b49d1f36's `08b-RA-a75` release, 04:44Z); Sol `main` still `8ed41433`.
- Part B: data lock committed; node F chains started 05:32Z (`2b-RS17UP` seeds 1 / 2, then `2b-RAUPM`).
- `m18-cand.sh` hop: per-item staging directory (two concurrent relays to one parent collided on `.m18-part`; the
  copy itself was complete and verified by SHA-256 list).
- Amendment 1 arms (`2b-RAM`, `08b-RAM`) on node A GPU1-2 / 7: inputs copied node F -> node A (pulled by node A).

## 2026-10-02T06:25Z

- Part A complete: 2B four runs scored and bootstrapped vs DEV2.0-2B (all exit 0); 0.8B three runs scored
  (`08b-UPRA` with one `invalid_model_output` row counted wrong, as `2b-UPRAa75`); no 0.8B Part A candidate is
  above the current Eos release, so their bootstraps vs it are report-only.
- Part B node F: `2b-RS17UP` seeds DONE (preflights PASS), soup built 06:03Z; `2b-RAUPM` seeds running.
  Node A: `2b-RAM` seeds 1 / 2 and `08b-RAM` seed 1 running (preflights PASS); `08b-RAM` seed 2 follows on GPU1.
- Candidates `2b-RS17UP` (soup) and `2b-SWRA` (½ RS17UP soup + ½ `2b-RA`, built on node E, lineage check PASS):
  BF16 copies restaged (identity / loaded / calibration checks pass); IX1 pool on node E GPU0-3 / 6-7 since 06:08Z
  (node E leases of the finished M18 runs set to released first).
- Bootstraps of `2b-UPRA` / `2b-UPRAa75` vs the `IS-2b-RAUP` run (the likely next Sol release) running on node C.

## 2026-10-02T06:50Z

- Part B / amendment-1 soups built: `2b-RS17UP` (06:03Z), `2b-RAUPM` (06:37Z), `2b-RAM` (06:42Z); all four seeds of
  each 2B arm passed preflight and completed. `08b-RAM` seeds 1 / 2 running on node A.
- Index: `2b-RS17UP`, `2b-SWRA` scored on node C (scorer gates pass). `2b-RAUPM` running on node F GPU4-5; `2b-UPRAM`
  (½ RAM + ½ RAUPM, lineage check PASS) and `2b-RAM` on node E GPU0-3 / 6-7.
- Bootstraps of `2b-UPRA` / `2b-UPRAa75` vs the `IS-2b-RAUP` run: 95% lower bounds below 0 (not releasable over it).
- Amendment 2: wider uniform soups `2b-U5`, `2b-U4`, `08b-RRM-a75`.

## 2026-10-02T07:45Z

- Seeds: all 2B arms done (`2b-RS17UP`, `2b-RAUPM`, `2b-RAM`, two seeds each, preflights PASS); `08b-RAM` seed 1 done
  07:25Z, seed 2 running on node A GPU1.
- Index scored on node C (scorer gates pass): `2b-RS17UP`, `2b-SWRA`, `2b-RAUPM`, `2b-UPRAM`, `2b-RAM`. Running:
  `2b-U4` (node E), `2b-U5` (node A GPU2 / 7, panel-3), `2b-MLRAM` (node F GPU4-5). `2b-U3ML` built and restaged,
  queued for node E. Bootstraps of `2b-RAM` vs DEV2.0-2B, vs `IS-2b-RAUP` and vs `M16-2b-RASD-bf16` (the 2B point
  b49d1f36 is about to release) running on node C.
- Hub: Eos `3de61185`, Sol `8ed41433` (b49d1f36's 2B upload is held by the Vega-27B release on node A).

## 2026-10-02T07:45Z — PAUSED (coordinator interrupt 15:45 UTC+8: user pauses all 0.6B / 0.8B / 2B training)

Sol-2B and Eos-0.8B are not touched by this milestone (b49d1f36 releases `2b-RASDML`). Index values: private only
(node C `ix1/runs/M18-*/merged/compare.json`, `m18-boot-*.json`; `decision2-program/private/dec-m18/index-summary.txt`).

**Stopped (07:40Z):** node A chain `a1` (`08b-RAM` seed 2, mid full run: container `m18-m18-08b-RAM-s2-full`) and
the A pool (`2b-U5`, shards 1 / 2 running); node E pool (`2b-U4`, shards 4-7 running); node F pool (`2b-MLRAM`,
shards 0 / 1 running); node C bootstraps still running. No M18 process or container is left. Leases: owner files
of node A GPU1 / 2 / 7, node E GPU0 / 1 / 6 / 7, node F GPU4 / 5 moved to `owner.prev-m18-<UTC>`; all nine GPUs idle
(0% use, 0 GiB). M18 held no node C GPU.

**Candidates (verdicts; values private):**

| Candidate | State | Verdict |
| --- | --- | --- |
| `2b-RAUP-a75`, `2b-RAUP-a50`, `2b-UPRA`, `2b-UPRAa75` | scored, bootstrapped | above DEV2.0-2B; `UPRA` / `UPRAa75` not significant over `IS-2b-RAUP` |
| `2b-RS17UP`, `2b-SWRA`, `2b-RAUPM`, `2b-UPRAM` | scored | below the 2B point being released |
| `2b-RAM` (2b-RA recipe + IB3-r2 maths) | scored, bootstrapped | 95% lower bound > 0 vs DEV2.0-2B, `IS-2b-RAUP` and `M16-2b-RASD-bf16`; below the sweep's `IS-2b-RASDML` point estimate |
| `2b-U4`, `2b-U5`, `2b-MLRAM` | Index run stopped part-way | not measured |
| `2b-U3ML` | BF16 restaged on node E | not run |
| `08b-RAUP-a75`, `08b-UPRA`, `08b-UPRAa75` | scored (`08b-UPRA` with one error row counted wrong) | below the Eos release (bootstrap upper bounds < 0 for the two run) |
| `08b-RAUP-a50` | dropped (amendment 1) | — |
| `08b-RAM` | seed 1 done, seed 2 stopped | no soup; `08b-RAM-a75` / `08b-RRM-a75` not built |

Training used: 2B arms 6 seeds (≈ 2.2 GPU-h), `08b-RAM` ≈ 2.6 GPU-h; Index ≈ 19.9 GPU-h finished + ≈ 3 GPU-h in the
stopped runs. Total ≈ 28 GPU-h of the 40 cap.

**How to resume** (only on a new coordinator decision; mirror = the commit holding this entry, mirrored to the node):

1. GPUs: take leases again (track `dec-m18` for training, `eval-ix1` for Index), never node C GPU0, E GPU4-5, F GPU0-1.
2. Stopped Index runs (`2b-U4` node E, `2b-U5` node A, `2b-MLRAM` node F): their shard directories exist without an
   end, so `m18_ixpool.py` refuses them. Either resume the ended / killed shards with
   `v2/eval/ix1/launch.sh resume --src <mirror> --model <NAME> --gpus "<g>" --run <runs/NAME> --rows-dir <panel>
   --only "<k>"` (the kit runner keeps finished rows), or move `runs/<NAME>` and `parity/<NAME>` to `void/` and run
   `m18-cand.sh <sha> <NAME> pool <node> "<gpus>"` afresh. Then `relay` / `score` / `boot` with `m18-cand.sh`.
   `2b-U3ML`: `m18-cand.sh <sha> M18-2b-U3ML-bf16 pool e "0 1 2 3 6 7"` (package already restaged on node E).
3. `08b-RAM` seed 2: the chain marks a killed full run FAILED only when it exits normally; it was killed, so no
   marker exists. Move `m18/arms/full/m18-08b-RAM-s2*` aside (keep for the record), write the node A lease, then
   `M18_NODE=a m18-chains.sh launch <mirror> 1` after editing GPU1's items to `08b-RAM:2` (remove the chain lock
   `m18/chains/launch-a1.lock` first). Then `m18-soup.sh <mirror> 08b-RAM` on node A, and `m18-cand.sh` build /
   bf16 / restage / pool for `M18-08b-RRM-bf16` (build only), `M18-08b-RRM-a75-bf16` and `M18-08b-RAM-a75-bf16`.
4. Release rule unchanged (prereg "Release"): bootstrap lower bound > 0 vs the tier's then-current release run.
