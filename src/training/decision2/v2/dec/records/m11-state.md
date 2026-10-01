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
