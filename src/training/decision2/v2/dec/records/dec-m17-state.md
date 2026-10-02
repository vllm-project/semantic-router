# Decoder M17 — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17` (restart worker; the first M17 worker
stopped after creating the worktree, nothing had run). Prereg `da770d98a`, ops `1ce8b2220` (mirror on nodes E / F),
data lock `4f68f1eb0`.

## 2026-10-02 03:20Z

- **Training done** (all four preflights and full runs completed; no cap stop): `4b-LHS10SD` s1 / s2 DONE 02:59 /
  03:07Z (BEST checkpoint-0000896 / -0000669), `4b-LHS17SD` s1 / s2 DONE ≈03:05Z (BEST -0000853 / -0000977). Soups
  built 03:08Z (`4b-LHS17SD`, model `8995bd9d…`) and 03:10Z (`4b-LHS10SD`, `537553da…`).
- **Formal parity run `m17-4b-LH` exact** vs the stored LH run (0 / 0 / 0 differing answers on typed FINAL, CSS15,
  public 231; v3 67.345 reproduced); its mlx-diag collected 02:51Z and scored on node A. A first mlx-diag attempt was
  launched from a different mirror than the collection's and exited after 0.22 s at loading the adapter spec (no
  inference); it was moved to `formal/m17/void/mlx-4b-LH-20261002T024703Z/` and relaunched from the collection's
  mirror.
- **MLX-DEV2 reference:** LH's own IX1 package read on node F (5,215 predictions, Triton cache then frozen); its answers
  equal the eval track's LH read on all 5,215 items (card delta 0.0).
- **Index (amendment 1):** both BF16 copies staged on node F and shipped to node E (`4b-LHS10SD` → `d44a336a…`,
  `4b-LHS17SD` → `74ec8b2f…`; identity, loaded 4,208,383,488 and calibration none checked; per-file lists equal on F
  and E). Pool on node E GPU0–2 from 03:14Z: **both 86-request parity gates PASS** (86 / 86 ok, max |Δp| 0.0); panel-8
  shards running. Node E GPU3 is held by the ROCm kernel track (ed4a4d73, lease from 02:59Z) and is not used.
- Contamination audit of both M17 TRAIN files running on node C (CPU, from 02:44Z).
- Readouts: both arms' panels being read on F GPU6 / GPU2 (post chains); formal collections of both arms launched
  03:19Z on F GPU7 / GPU3 (references + item 3), select file regenerated with slot 0 unchanged (`d5e0c085…`).

## 2026-10-02 02:45Z

- **Amendment 1 (`367cdfa67`)**, written before any M17 soup or readout: the user's Index-first release rule
  (COORDINATION 09:55) applies to M17. Both arm soups are the frozen candidates, each measured once on the Index as its
  BF16 copy restaged onto LH's IX1 package (the 4B Index-first worker's method); development gates, formal items and
  MLX-DEV2 become references; C1 item 8 is not run; anchors only on a type-collapse-only failure.
- IX1 DIAGNOSTIC entries `DEV2.0-4B-LHS10SD-bf16` / `-LHS17SD-bf16` (`db00429a1`, a separate commit to
  `v2/eval/ix1/launch.sh`); the Index driver `ops/m17/m17-index.sh` + `m17_ixpool.py` (`99cbfb269`), mirrored on
  nodes A, C, E, F.
- Node E: kit `87d4650b` and panel-8 copied from node C through node A (per-file SHA-256 lists equal; kit HEAD
  checked). GPU0–3 still idle (Index inference only).
- Training: seeds at checkpoints 4 / 3 / 3 / 3 of 8 (02:36Z); the formal parity run `m17-4b-LH` (LH soup, co-tenant on
  F GPU7) passed its smoke at 02:29Z and is collecting. Its select file holds slot 0 only (listing `d5e0c085…` = M13's);
  the finalists' file is written later with the same slot 0.

## 2026-10-02 02:30Z

- All four preflights PASS (pre-warm `4b-LHS10SD` s1 02:04Z; the other three 02:09–02:10Z); full runs at update
  ≈ 190–290, ≈ 15–19 updates / min: seeds end ≈ 03:05–03:15Z.
- Post chains queued on the GPU flocks (02:16Z, mirror `4dfd484a1`): GPU6 = the reference MLX-DEV2 read (LH's own IX1
  package, fresh Triton cache, then frozen), then `4b-LHS10SD` (soup, 8 panels, old MLX-DEV, MLX-DEV2); GPU2 =
  `4b-LHS17SD` (the same, its MLX-DEV2 after the reference read).
- MLX-DEV2 inputs on node F: LH's IX1 package `DEV2.0-4B-13d42143` copied from node C through node A (tree digest
  `29bc37d4…` equal on C and F, 0 manifest mismatches); gold-free prompts `35747a26…` from node A. Gold stays on A.
- Ops for scoring, rules, anchors, formal and the successor (with the Index path's public items) at `a25cfb391`,
  mirrored on F and A.
- Incident (no effect): a local lint command with an empty tool path executed `m17-arm.sh` on the workstation; it
  stopped at its first `mkdir` (no `/data` there). Nothing ran on any node.

## 2026-10-02 02:00Z

- Data built on E and F (01:58Z), byte-identical: `4b-LHS10SD` 66,004 rows / T + 64 tokens (IB .100, 17.1% of English
  tokens swapped out), `4b-LHS17SD` 71,088 rows / T + 68 (IB .170, 29.0%); multilingual share .415 (LH .414);
  `sentfin` dropped from the IB pool (COORDINATION 01:05). `READY-m17.json` written on F.
- Leases node F GPU2, 3, 6, 7 `track=dec-m17`; chains launched 01:59:48Z from `1ce8b2220`: GPU6 `4b-LHS10SD` s1
  (pre-warm, zero-step running), GPU7 s2, GPU2 / GPU3 `4b-LHS17SD` s1 / s2 waiting for the pre-warm marker.
- Node F prep: Triton caches 4b-train / 4b-read copied from M15; reference readouts `4b-LH-f` and `4b-LHA10SD-m13`
  copied from M15 (8 panels each, plus their old MLX-DEV reads); old MLX-DEV 4B panel copied (report only).
- Node E: data build only; GPU0–3 untouched (Index runs only, if any). GPU-h so far: 0.
