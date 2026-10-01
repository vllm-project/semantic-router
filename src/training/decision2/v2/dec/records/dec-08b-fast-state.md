# Decoder 0.8B fast track — state (polled ≤ 30 min; newest first)

Assignment: COORDINATION 2026-10-01 22:35 — one formal attempt for M12's `08b-RA`, then items 1–7 vs DEV2.0-0.8B
(`4afea305`; bar = its stored formal run, v3 50.236); if they pass, the C1 content recheck for IB1-r3 + IB2, item 8,
the release on `4afea305` and a private Index run. Branch `xunzhuo/decision-2-training-dec-08bfast`. Lock record
[`dec-08bfast-formal-lock-2026-10-01.md`](dec-08bfast-formal-lock-2026-10-01.md).

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
