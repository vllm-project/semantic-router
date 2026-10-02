# 4B Index-first release — state (worker 5e7b8132; newest first)

Assignment: COORDINATION 2026-10-02 09:55 / 10:00 (user rule: INDEX-FIRST). Release gate: the frozen candidate's
private Jev Decision Index delta vs the current release significantly positive (paired bootstrap over rows within
benchmarks through the board weights, 2,000 replicates, 95% lower bound > 0). Integrity checks: exact package parity,
Hub `trust_remote_code` smoke (Transformers 5.17 / 5.18), a row-level Index contamination audit of the training file,
no decision type collapsed (formal typed panel item 3). References (recorded, not blocking): JevArena v3, human
transfer, mlx-diag, public 231, C1. Selection: the largest Index-gain lower bound. Branch
`xunzhuo/decision-2-training-4b-indexfirst`; ops `v2/dec/ops/4bif/`. **No Index value appears in this repository**;
values live in the node private directories and the local private folder only.

Reference: the released LH, `Decision-2.0-Nox-4B` `main` `54b084f9` (weights `6a555335…`, the same weights and
runtime as DEV2.0-4B `13d42143`); its IX1 run `DEV2.0-4B-LH` (node C, panel run IDs `6455d7be…`).

## 2026-10-02 03:10Z

### Candidates and the weights each Index run scores

Every new run scores exactly the weights a release would ship: the `v2.release.bf16_copy` of the frozen FP32 soup or
interpolation point, built on the Index node in a CPU-only container of image host2 (`f83b1d10`), then restaged into
the released LH package (`v2.eval.ix1.restage` onto DEV2.0-4B `13d42143`; runtime, T = 1 and every non-model file
unchanged; loaded 4,208,383,488).

| Candidate | Frozen FP32 identity | BF16 identity (Index name) | Restaged manifest | Formal item 3 |
| --- | --- | --- | --- | --- |
| M13 `4b-LHA10SD` | `255021e0…` | existing FP32 run (dec M15 part A, node D) | — | OK / OK / OK (M13) |
| M13 `4b-LHA10SD` as BF16 | `255021e0…` | `c6a73881…` (`DEV2.0-4B-LHA10SD-bf16`) | `6eb8314b…` | as above |
| M14 `4b-LHA10UP` | `9e80e765…` | `673cb169…` (`DEV2.0-4B-LHA10UP-bf16`) | `fa62787c…` | OK / OK / OK (this run) |
| M16 `4b-LHA10SD-a75` | `96d09b16…` | `e173cd02…` (`DEV2.0-4B-LHA10SD-a75-bf16`) | `9308567d…` | OK / OK / OK (M16) |
| M16 `4b-LHA10SD-a50` | `84f462f2…` | `47d95914…` (`DEV2.0-4B-LHA10SD-a50-bf16`) | `681bc271…` | OK / OK / OK (M16) |

- bf16-copy receipts: UP `65a6cbde…`, a75 `5e37a976…`, a50 `ff9c2d22…` (built on node C and on node D: identical
  receipt and manifest), SDB `c81b4d20…`. Each receipt's source fingerprint equals the frozen FP32 identity.
- M15 `4b-LHA10SDML` (optional) is not measured: the Index nodes are shared and the four runs above come first.

### Index runs (IX1 harness: image host2, kit `87d4650b`, 86-request parity gate, dual scoring)

- **M13 FP32 run imported** to node C (relayed node D → node B → node C; the workstation link is too slow): results
  `24660792…` = its receipt's. The paired bootstraps vs LH (full panel and transfer-only) are done (private).
  The Index-first gate (`gate.py` IF1) binds the Index run to exactly the shipped weights, so releasing M13 needs the
  BF16 run `DEV2.0-4B-LHA10SD-bf16`; it is queued.
- **a75:** parity gate PASS (86 / 86, node C GPU5, 02:55Z); full run started on node C (panel-8 shards).
- **UP, a50 (node D), SDB (node C):** queued in `ops/4bif/pool.py`.
- **GPU contention.** At 02:11–02:27Z the 27B M6 worker's M6-IB Index run took node C GPU1–4 and node D GPU4–7, and
  the Index sweep took node C GPU5–7, minutes before this worker's first parity gates; the harness refused those
  attempts (busy GPU; nothing ran) and they were moved to `void/`. `pool.py` now dispatches parity gates and single
  shards to GPUs of this allocation that are unleased, released, ours, or abandoned (the run ended 15 min ago and the
  GPU idle over three polls), never another job's active lease; it adopts its own running jobs on restart.

### Integrity checks so far

- **Contamination audit** (`ops/4bif/audit4b.sh`, node C CPU, `v2.eval.ix1.contamination`): every 4B candidate
  trained on M12's locked 4b-LHA10 TRAIN `d41cdd1a…` (72,847 rows; the released LH TRAIN `c385406e…` as its first
  58,739 rows plus 14,108 IB rows; M15's arm adds only copies of released rows). Against all 120,226 Index rows:
  **0 item rows**, 114 familiar-text rows, the same as the released LH TRAIN in the same run (0 / 114); planted
  control 200 / 200. The IB rows add no overlap.
- **Formal item 3 for M14 `4b-LHA10UP`** (`ops/4bif/formal4b.sh`; node B GPU3, image `dbe5f32b`, the 4B formal
  library's master cache `f6d0f920…`, T = 1 by the 23:15 rule; relayed gold-free node B → node C → node A, scored on
  node A): run `4bif-4b-LHA10UP`, seal `c6504bcc…`; **types Choice / Noul / Score OK / OK / OK**. References vs
  bar-lh (the released LH's stored run): v3 −1.24 [−4.00, +3.04], human transfer [−.026, +.081], public 231 178 vs
  172 (OK). Node B GPU3 released.

### GPU-hours so far

Parity a75 0.06; formal UP (smoke + collection) 0.12; a75 shards running. CPU only: bf16 copies, restages,
bootstraps, the audit.
