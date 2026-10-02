# Decoder M17 — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17` (restart worker; the first M17 worker
stopped after creating the worktree, nothing had run). Prereg `da770d98a`, ops `1ce8b2220` (mirror on nodes E / F),
data lock `4f68f1eb0`.

## 2026-10-02 07:30Z — the sweep's 4B candidate is not a successor; stage-2 Index runs in flight

- **`IS-4b-LHA10SDML-bf16` (handed over by the Index sweep, coordinator 07:30Z): not a successor.** M15
  `4b-LHA10SDML` soup `1b515675…`, BF16 copy `991a8fb8…`. The sweep's cross bootstrap vs the release's run (node D
  `ix1/index-sweep/cross/IS-4b-LHA10SDML-bf16-vs-LHS17SD-bf16-full.json` `f296935f…`; base input = the
  `DEV2.0-4B-LHS17SD-bf16` results `bb726d5d…`) has a 95% lower bound **≤ 0**, so it fails the release gate; no
  integrity check is run for release. Its transfer-only part (`…-transfer.json` `a3e8ae26…`; without HoVer,
  When2Call, iSarcasmEval, GSM8K, BPoMP) is significantly positive vs the release (private values; the release's
  own gain is in-distribution). No formal v3 / mlx-diag run of it exists (M15 ran development only: MLX-DEV-M15
  above LH, retention below LH); its formal references are queued with stage 2's formal collections.
- Wave 2 (`4b-LHS17IB4`, `4b-LHS17IB4X`) training on node F GPU2 / 3 / 6 / 7 since 07:02Z.
- Wave-1 candidates, each staged as its BF16 release copy (receipt source = soup SHA-256) and restaged onto
  `DEV2.0-4B-13d42143` (identity checked, loaded 4,208,383,488, T = 1):

  | Candidate | FP32 soup → BF16 identity | Package manifest | Index run |
  | --- | --- | --- | --- |
  | `DEV2.0-4B-LHS17UP-bf16` | `b5617263…` → `2e754a80…` | `9ff9286b…` | node C GPU1–5, parity PASS 07:12Z, shards running |
  | `DEV2.0-4B-LHS23SD-bf16` | `5b0268b5…` → `3f70e632…` | `fb38ba05…` | node E fast lane, parity started 07:23Z |
  | `DEV2.0-4B-LHS17UP-m50-bf16` | `495aadad…` → `3135d10b…` | `53755d67…` | node E fast lane, parity started 07:23Z |
  | `DEV2.0-4B-LHS23SD-m50-bf16` | `57b71da8…` → `4c902a6e…` | `13dddccb…` | node E fast lane, parity started 07:23Z |

- Formal-path readouts (typed DEV, CSS pilot) of all four finished on node F 07:07–07:12Z (co-tenant).
- Contamination audit of the stage-2 TRAIN files (`audit2`, node C CPU) running since 06:12Z.
- C1 recheck r3 (IB4 phase 1): PASS (verdict `54bc9361…`), so IB4 arms may be released.

## 2026-10-02 06:15Z — stage 2 training (wave 1 running, wave 2 queued)

- 05:55Z coordinator: `4b-LHS17SD` is the released Nox-4B (`b285e7a1`, by 5e7b8132); M17 is now the 4B trainer and
  the single Nox-4B publisher. 5e7b8132's branch is merged here (`8af7f2d89`; `ix1/launch.sh` conflict resolved as
  the union of both entry lists) for its release ops (`release/records/dev2-4b-indexfirst-2026-10-02/ops/`,
  `dec/ops/4bif/`).
- Stage 2 prereg `5bedf6f6f`, amendment 1 `da9416b8a` (wave-1 data lock; arm (a) = S17 + IB4 p1 + IB3-r2, with and
  without `isarc2`).
- Node F wave 1 started 06:00:56Z (5 min after the note): GPU2 / 3 `4b-LHS17UP` s1 / s2, GPU6 / 7 `4b-LHS23SD`
  s1 / s2; all four preflights PASS 06:06Z. Wave 2 (`4b-LHS17IB4` GPU2 / 3, `4b-LHS17IB4X` GPU6 / 7) is queued on
  the GPU flocks; data built 06:05Z (`READY-m17s3.json`).
- Ops for the measurement path (this commit): `m17-index.sh` takes the stage-2 names, `M17_REF=S17` (bootstrap base =
  the current release's run `DEV2.0-4B-LHS17SD-bf16`), the eval fast lane node E GPU0–3 / 6–7, and `audit2-*` for
  the new TRAIN files; `m17-soup.sh` builds the stage-2 arms and the interpolation points `4b-<X>-m50` (uniform
  .5 / .5 with the `4b-LHS17SD` soup, the prereg's arm (d) rule); IX1 DIAGNOSTIC entries for the seven names.
- GPU-h stage 2 so far ≈ 2.6 (4 GPUs × 0.65 h). Nothing measured yet.

## 2026-10-02 04:35Z — candidates handed to the Nox-4B publisher

- Coordinator interrupt (12:25 UTC+8): M17 does not publish; 4B Index-first worker 5e7b8132 is the single Nox-4B
  publisher. Both BF16 candidates are handed over with their integrity evidence in
  [`dec-m17-handover-2026-10-02.md`](dec-m17-handover-2026-10-02.md). Every hash in it was re-read on the nodes at 04:30Z.
- Formal typed-FINAL (item 3) already exists for both arms (types OK / OK / OK); no new run was needed.
- Still running: the two CPU paired bootstraps vs `DEV2.0-4B-LH` on node C (detached; exit files pending). Their
  output is fetched to the private directory only.
- No M17 GPU lease or container is active. Next: stage-2 prereg (swap + typed upweight, + a75 interpolation).

## 2026-10-02 03:55Z — references complete; Index runs finishing

- **Development (node A, vs `4b-LH-f`; references under amendment 1; the arms rules ran once, 03:30Z):**

  | Arm | Typed C / N / S (LH 728 / 290 / 371) | HT-DEV v2 | Retention Δ | Transfer Δ (IB DEV Δ) | MLX-DEV2 card Δ [95% CI] | Gates failed |
  | --- | --- | --- | --- | --- | --- | --- |
  | `4b-LHS10SD` | 699 / – / 333 | −.010 [−.026, +.005] TIE | −.048 [−.076, −.024] | +.043 (+.080) | **−.023 [−.029, −.017]** | 1 (choice, Score floors), 4, 7 |
  | `4b-LHS17SD` | – | −.026 [−.041, −.010] **FLAG** | −.026 [−.052, +.000] | +.046 (+.082) | **−.026 [−.032, −.019]** | 3, 7 |

  M13 `4b-LHA10SD` re-scored here (report): HT −.003 TIE, retention −.022, transfer +.046. Contrasts: `4b-LHS10SD` vs
  `4b-LHA10SD` HT −.007 TIE, retention −.026 [−.052, −.003]; `4b-LHS17SD` vs `4b-LHS10SD` HT −.015 TIE.
  The prereg's anchor trigger fired (better arm `4b-LHS17SD`), but amendment 1 builds anchors only on a
  type-collapse-only failure: none built.
- **Formal (references; two bars agree, the parity run is exact):** `4b-LHS10SD` v3 64.252, −3.09 [−7.49, +0.93];
  `4b-LHS17SD` v3 62.090, **−5.25 [−11.26, −1.67]**. Types OK / OK / OK for both (the type-collapse integrity check
  passes). mlx-diag card-eligible −.0257 [−.0358, −.0161] / −.0279 [−.0395, −.0167]: **MLX-DEV2's flags confirmed** (as
  additive `4b-LHA10SD`, −.0287). Public 231 174 vs 172. Exposure 0 groups for both TRAIN files; overlap 5 / 5 pairs
  reproduced. Classic items 1–7 FAIL (items 1, 4, 6(b)); 02:05 Index-path items: `LHS10SD` 1'(a) / 6(b)' pass but
  item 4 fails, `LHS17SD` fails 1'(a) / 6(b)'.
- **Integrity checks so far:** IX1 parity 86 / 86 (both); Index contamination audit of both TRAIN files: 0 item rows
  (planted 200 / 200; 100 / 90 duplicate-class rows); no collapsed type.
- **Index runs:** `4b-LHS10SD` on node E GPU0–2 (shards 0–2 done exit 0; 3, 4, 6 running; lanes for 5, 7 after the
  pool was stopped at 03:33Z), `4b-LHS17SD` on node F GPU2, 3, 6, 7 (its parity gate copied from node E with its frozen
  cache, per-file lists equal; shards 0–3 running). Both end ≈04:10Z; then relay to node C, scoring, bootstrap.

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
