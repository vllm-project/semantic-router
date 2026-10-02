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

## 2026-10-02 04:40Z

- **Coordinator interrupt (12:25 UTC+8): release the highest-scoring candidate; never a lower one once a higher one
  is measured.** The comparison set is every finished BF16 Index run: M13 `DEV2.0-4B-LHA10SD-bf16` and M17
  `DEV2.0-4B-LHS10SD-bf16` / `DEV2.0-4B-LHS17SD-bf16` (both finished, scorers PASS; M17 hand-over
  [`dec-m17-handover-2026-10-02.md`](dec-m17-handover-2026-10-02.md) on `xunzhuo/decision-2-training-dec-m17`).
  a75 / a50 / UP are not waited for.
- Paired bootstraps vs `DEV2.0-4B-LH` (2,000 replicates, seed 20261002; full panel and transfer-only) running on CPU:
  SDB and LHS10SD on node C; LHS17SD on node D (node C at load > 160), on SHA-256-checked copies of the two
  `results.jsonl` files (`runs/4bif-bootcopy/`), mirror `3b8f2cf85` (`v2/eval/ix1` identical to `535a884db`).
- Integration merged (`055affd2a`, includes the collection-check fix `cd565a588`).

## 2026-10-02 03:40Z

- **Progressive-release directive (COORDINATION 11:35 UTC+8):** release M13 `4b-LHA10SD` now if its lower bound vs LH
  is > 0 and the integrity checks pass; LHA10UP / a75 / a50 follow as successors against that release. M13's FP32 run
  qualifies (private), so the release choice is M13's BF16 copy `DEV2.0-4B-LHA10SD-bf16` (`c6a73881…`), whose own
  Index run is the IF1 evidence (the gate binds the run to the shipped weights): **parity gate PASS** (86 / 86, node C
  GPU5, 03:32Z); shards running on node C (moved ahead of a75 in the node-C queue).
- a75: shards 0–1 done (exit 0); the rest wait behind SDB. Node D: UP, then a50, waiting for GPUs.
- `pool.py` now judges an abandoned eval-ix1 lease per GPU (the shards the run's launcher records placed on that GPU,
  ended at least 15 min ago); M6-IB's finished shards on node C GPU1–3 are taken over that way (the old owner files
  are kept as `owner.prev-4bif-<UTC>`).
- **M13 release inputs on node A** (`ops` of `v2/release/records/dev2-4b-indexfirst-2026-10-02/`): the BF16 checkpoint
  (13 files equal to the Index-run copy), the formal and mlx-diag runs' persisted autotune caches (manifests equal to
  the runs' cache-after: `d58dbddb…`, `e7b23b1e…`), the current revision's gate / decision copies, and the gates on
  the formal run `m13-4b-LHA10SD`: types OK / OK / OK; references vs the released LH v3 −0.16 [−2.84, +3.64], human
  transfer [−.036, +.062], public 231 174 vs 172 (OK), card-eligible mlx-diag −.0287 [−.0394, −.0184] (a material
  multilingual regression, disclosed; not a blocker under the rule); vs adopted Nox 1.0 +10.72 [+6.53, +13.66]; vs
  Decider 4B +5.30 [+0.41, +8.50].

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
