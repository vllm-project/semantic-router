# Decoder 0.8B fast track — formal lock for M12's `08b-RA` (2026-10-01)

Assignment: COORDINATION 2026-10-01 22:35 (user directive 22:20: fix the 0.8B regression vs Decision 1.0 Eos first).
Coordinator exception, recorded before any new result: M12's `08b-RA` gets **one formal attempt**; it failed only the
`attribute_gate` development family floor (277 vs 280). **Successor items 1–8 bind with no exception.** Written
2026-10-01 ≈14:50Z, before any fast-track GPU job. Development readouts are never release scores; v3 / public 231 are
post-key same-panel comparisons; Index results stay private. State: [`dec-08b-fast-state.md`](dec-08b-fast-state.md).

## Frozen inputs (node E, `runs/dec/formal/f08/inputs`, `cp -a` copies made read-only)

| Input | Source | SHA-256 |
| --- | --- | --- |
| `08b-RA` soup (tree manifest of `inputs/08b-RA-soup`; equal to M12's `soup/08b-RA/build/08b-RA-soup`) | uniform FP32 soup of `m12-08b-RA-s1/checkpoint-0004115` and `-s2/checkpoint-0004109` | `83926edfea269a3f1509c08c7f3d716b9002e8caf2992d293c3991dd3cbbc18b` |
| its backbone / head / `decision_config.json` | | `f6d9cc55…` / `3bbfa5cb…` / `863ebeae…` |
| its 16K typed DEV / CSS pilot readouts (M12, node E) | `m12/lines/08b-RA` | `7e152f50…` / `ac108d21…` |
| `08b-C0` (tree manifest of `inputs/08b-C0`) | DEV2.0-0.8B weights `bede7938…`, the weights of the current `main` `4afea305` (later revisions changed runtime and remote code only) | `26baab0177827efdf4f15f06f72d3281725c59d43da5cb322873c8af55b598bf` |
| its 16K typed DEV / CSS pilot readouts | M12's `08b-C0-e` | `6f12c527…` / `fe3e332a…` |
| `08b-RA` TRAIN (M12 data lock `d506595c2`) | `m12/data/08b/08b-RA/train.jsonl`, 309,225 rows | `12bd63d8b215877b6f471c12652e534e0f0be794c15bff74ad5ac50e98018bf3` |

## Path (M12 prereg "Formal, successor, hand-offs", unchanged)

- Node E **GPU6–7 only** (fast-track lease `track=dec-08bfast`; node E GPU0–5 never), image `dbe5f32b`,
  `run_same_panel --isolate`, the M6 formal library with `M6_SMALL_NODE=E`, `cp -a` copies of node B's frozen 0.8B
  masters in `formal/m11/masters` (`cache-frozen-08b` manifest `822e9f2d…`, `-mlx` `ba120cb4…`), 16,384 tokens.
  Outputs under `runs/dec/formal/f08` (prefix `f08`); wrapper `ops/f08/f08-formal.sh`.
- One chain, stopping at the first failure: `08b-C0` smoke (8 items) and collection, then `08b-RA` smoke and
  collection, so a path failure on C0 cannot spend RA's attempt. Staging per point: the CAL698 16K fit and the 23:15
  rule (adopt or T = 1). The release ships T = 1 (COORDINATION 13:40); a calibrated collection is retempered to T = 1
  for parity, which changes no answer. mlx-diag runs after node A has reported the v3 run.
- Scoring on node A (`ops/f08/f08-score.sh`): runs pulled gold-free over the transfer key; report, types, compares.

## Bars (fixed now)

- **`bar-t1`** = the stored formal run of DEV2.0-0.8B, `runs/release/dev2-0p8b-t1-derived` (v3 50.236), the bar the
  coordinator named.
- **`bar-e`** = `f08-08b-C0`, node E's collection of the same weights on this path (the M12 prereg's parity run). Its
  answers are compared with `bar-t1` and with node B's `m8s-ref-08b-I` (`C0-EXACT.json`; M8s found node B's path not
  answer-identical to node A's).
- **Items 1, 2, 4, 6(b) and 7 must pass against both bars.** If C0 reproduces `bar-t1` exactly, the two bars have the
  same answers. Item 4's references are node A's `m2-E8F-soup-nodeA-mlx` (m6-score's) and C0's node-E mlx-diag run.

## Successor items (`ops/m6/m6_successor.py` per bar; `ops/f08/f08_successor.py` combines)

1. v3 paired 95% lower bound > 0. 2. `axis_ci95.H.delta.high` ≥ 0. 3. No type collapsed (`gates types`).
4. mlx-diag card-eligible macro (Choice + Noul) paired upper bound ≥ 0. 5. Vs the adopted Eos 1.0 run
   (`eval/m1/r4-eos1`): lower bound > 0; no type collapsed.
6. (a) The 08b-RA TRAIN file's `overlap_effects exposure` receipt against the r2 payload `2194716a…` lists 0 groups
   (built on node A from the file streamed from node E, hash-checked). 08b-RA contains no incumbent weight (Eos 1.0
   retrained on the r2-clean recipe plus IB), so it inherits no exposure. (b) Rules 1 and 5 hold on the reduced
   panels (the 84 flagged items removed), against both bars.
7. JevBench public 231: `gates public231` not REGRESSION against each bar.
8. Only if 1–7 pass: the custodian's **C1 content recheck for IB1-r3 + IB2** first, then one `c1-postkey.sh` attempt
   against the 0.8B baseline, labelled post-key.

Then, only if 1–8 pass: the release (built on `4afea305`, exact package parity vs the formal predictions) and a
private Index run. If items 1–7 fail: record and stop; no further attempt on `08b-RA`. If DEV2.0-0.8B's `main` moves
(for example an M13 release) before item 8, this attempt stops and is recorded; a later line is judged against the
then-current release (COORDINATION 22:35).

## Budget and stop rules

About 1 GPU-h on node E GPU6–7 for the formal (CAL fits, smokes, collections, mlx-diag). A failed smoke, collection or
scoring step stops the point and is never rerun. Nothing is uploaded unless items 1–8 pass. No Index row is read for
selection.
