# Decoder 0.8B fast track — state (polled ≤ 30 min; newest first)

Assignment: COORDINATION 2026-10-01 22:35 — one formal attempt for M12's `08b-RA`, then items 1–7 vs DEV2.0-0.8B
(`4afea305`; bar = its stored formal run, v3 50.236); if they pass, the C1 content recheck for IB1-r3 + IB2, item 8,
the release on `4afea305` and a private Index run. Branch `xunzhuo/decision-2-training-dec-08bfast`. Lock record
[`dec-08bfast-formal-lock-2026-10-01.md`](dec-08bfast-formal-lock-2026-10-01.md).

## 2026-10-01 14:50Z — inputs frozen, lock written

- Node E GPU6–7 leased (`track=dec-08bfast`, 14:30Z); node E GPU0–3 run M13 (untouched), GPU4–5 foreign.
- Frozen copies on node E under `runs/dec/formal/f08/inputs` (read-only): `08b-RA-soup` (tree `83926edf…`, equal
  to M12's soup), `08b-C0` (DEV2.0-0.8B `bede7938…`, tree `26baab01…`) and their 16K typed DEV / CSS pilot readouts.
- Code: `ops/f08/f08-formal.sh` (node E chain), `ops/f08/f08-score.sh` (node A, two bars), `ops/f08/f08_successor.py`;
  tests `v2/dec/tests/test_f08.py`.
- Next: mirror to nodes E and A, select file, C0 then RA formal on node E GPU6.
