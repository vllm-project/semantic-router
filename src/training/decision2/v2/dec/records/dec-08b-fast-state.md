# Decoder 0.8B fast track — state (polled ≤ 30 min; newest first)

Assignment: COORDINATION 2026-10-01 22:35 — one formal attempt for M12's `08b-RA`, then items 1–7 vs DEV2.0-0.8B
(`4afea305`; bar = its stored formal run, v3 50.236); if they pass, the C1 content recheck for IB1-r3 + IB2, item 8,
the release on `4afea305` and a private Index run. Branch `xunzhuo/decision-2-training-dec-08bfast`. Lock record
[`dec-08bfast-formal-lock-2026-10-01.md`](dec-08bfast-formal-lock-2026-10-01.md).

## 2026-10-01 17:05Z — Index diagnostic closed (values private)

- Full run done 16:39Z: 120,226 rows (120,224 ok, 2 unsupported `max_length_exceeded`, 0 errors), 8 shards, results
  `9c28a430…`. Dual scoring (port vs kit `87d4650b`) **PASS**. Receipt
  [`dec-08bfast-index-diagnostic-receipt-2026-10-01.json`](dec-08bfast-index-diagnostic-receipt-2026-10-01.json).
- Contamination audit (CPU, node C, IX1 method, planted 200 / 200): the 08b-RA TRAIN file (`12bd63d8…`) has 18
  duplicate-class rows and 1 item row against the Index, vs 20 and 1 for the DEV2.0-0.8B TRAIN file in the same run.
- Comparison with DEV2.0-0.8B, Eos 1.0 and the 0.8B frontier entrant, and the family deltas, went to the private
  folder and to the coordinator only.
- GPU-h: parity 0.026, voided start 0.001, full run 1.234 → 1.26 on node D GPU5; lease released 16:39Z.

## 2026-10-01 15:25Z — follow-up: private Index diagnostic of the frozen `08b-RA` soup (not releasable)

- Coordinator follow-up: IX1 harness (`40c14b760`), 86-request parity gate, full 38-benchmark panel + HLE, dual
  scoring, family comparison with DEV2.0-0.8B's IX1 run. All values private; the repo gets counts and hashes only.
- Frozen soup copied node E → node D (tree `83926edf…` re-verified); restaged with `v2.eval.ix1.restage` onto the
  DEV2.0-0.8B `bede7938` package (runtime unchanged; backbone `f6d9cc55…`, head `3bbfa5cb…`; identity
  `f25beedd…` = the formal run's model hash). Diagnostic name `DEV2.0-0.8B-08bRA` in `ix1/launch.sh` (`7b9fff09a`).
- Node D GPU5 (no prior lease) leased `track=eval-ix1`. Parity gate **PASS** (86 / 86 ok). A first parity start
  failed before loading (identity set to the tree hash instead of the runtime fingerprint), voided.
- Full run: panel-8 shards 0–7 one after another on GPU5, started 15:22Z.

## 2026-10-01 15:10Z — closed: `08b-RA` fails items 1–7 (items 1, 4, 6(b)); stop

- `f08-08b-RA` (seal `50b65064…`, T = 1): v3 53.102, T .5491, H .5136, public 231 166. Item 1 vs `bar-t1` +2.866
  [−2.866, +4.579] and vs `bar-e` +2.839 [−2.920, +4.487] **FAIL**; item 4 mlx-diag card-eligible −.0136
  [−.0256, −.0024] **FAIL**; 6(b) **FAIL**; items 2, 3, 5, 6(a), 7 pass. Results
  [`dec-08bfast-results-2026-10-01.md`](dec-08bfast-results-2026-10-01.md).
- Per the coordinator's exception this was the only attempt. No C1 recheck, item 8, release or Index run follows.
  DEV2.0-0.8B `4afea305` stays.
- 0.21 GPU-h on node E GPU6–7. The leases were released at close-out and the node-E work stays under
  `runs/dec/formal/f08`.

## 2026-10-01 14:56Z — C0 parity run done; 08b-RA collecting

- Lock `a3ab9eda2` mirrored to nodes E and A; select file `formal/f08/select/08b-finalists.json` (`2f1123a2…`; list
  SHA-256 = the frozen trees). Chain `points-e6` launched 14:48Z on node E GPU6.
- `f08-08b-C0`: CAL698 16K rejected by the 23:15 rule (T = 1); smoke passed; collected 14:53Z (seal `756b9108…`).
  On node A: v3 50.263, T .5741, H .4401, public 231 155; **answer-identical to node B's `m8s-ref-08b-I`** (0 changes
  on typed FINAL / CSS15 / public 231) and **not to the stored bar** (9 / 20 / 1 changes; vs `bar-t1` +0.03
  [−0.34, +0.47]). So both bars bind, as the lock says. C0 mlx-diag collecting on GPU7.
- Item 6(a) input: the 08b-RA TRAIN receipt on node A (streamed, `12bd63d8…`, 309,225 rows) lists **0 groups**
  (`exposure-08b-RA.json` `f5432792…`, payload `2194716a…`).
- `08b-RA`: CAL fit 14:53Z, smoke / collection next on GPU6.

## 2026-10-01 14:50Z — inputs frozen, lock written

- Node E GPU6–7 leased (`track=dec-08bfast`, 14:30Z); node E GPU0–3 run M13 (untouched), GPU4–5 foreign.
- Frozen copies on node E under `runs/dec/formal/f08/inputs` (read-only): `08b-RA-soup` (tree `83926edf…`, equal
  to M12's soup), `08b-C0` (DEV2.0-0.8B `bede7938…`, tree `26baab01…`) and their 16K typed DEV / CSS pilot readouts.
- Code: `ops/f08/f08-formal.sh` (node E chain), `ops/f08/f08-score.sh` (node A, two bars), `ops/f08/f08_successor.py`;
  tests `v2/dec/tests/test_f08.py`.
- Next: mirror to nodes E and A, select file, C0 then RA formal on node E GPU6.
