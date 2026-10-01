# Decoder M11 — state (polled ≤ 30 min)

Prereg `b8c3caf22`; ops `cef1a1b66`; data lock `7c3aed8c8`; amendment 1 (node F layout). Branch
`xunzhuo/decision-2-training-dec-m11`.

## 2026-10-01 07:15Z — launch

- Stage 1 chains launched from the mirror of the amendment-1 commit:
  - node F GPU2 / 3 / 6: `2b-LH` s1–s3 then `08b-LH` s1–s3 (GPU2 pre-warms); GPU7: post chains `2b-LH`, `08b-LH`;
  - node E GPU0 / 1 / 2: `2b-NT` s1–s3 then `08b-NT` s1–s3 (GPU0 pre-warms); GPU3: refs (C0 and BASE per tier), then
    post chains `2b-NT`, `08b-NT`.
- Node F GPU4–5 are held by a foreign workload (amendment 1); untouched.
- Stage 2 (4B LH + IB1) waits for a release-safe IB1-r3 on origin.

## 2026-10-01 07:17Z — poll

- Pre-warm preflights PASS: `2b-NT-s1` (E GPU0, 07:14:14Z), `2b-LH-s1` (F GPU2, 07:14:42Z); seeds 2–3 of both arms
  started on the markers at 07:15Z. All six 2B seeds train (≈2 s / update NT, ≈3.4 s / update LH; ≈877 updates).
- E GPU3: `2b-C0-e` readouts in progress (refs chain), then `2b-BASE-e`, `08b-C0-e`, `08b-BASE-e`.
- Node A: mirror `f45c2f481`; tier probe gold built (2B 2,989 items `0c0781ee…`; 0.8B = M10's 3,089 `5c674e35…`).
- Next: when the 2B seeds finish (≈08:10–08:30Z), the 0.8B seeds start (pre-warm on E GPU0 / F GPU2 first); post
  chains build soups and read; then `m11-score.sh pull / points / readout / rules` on node A per tier.
