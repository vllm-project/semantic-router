# IB4 state (data worker; counts only, no Index numbers)

- 12:20 +08 — started; worktree / branch `xunzhuo/decision-2-training-data-ib4` from `052450488`.
- 12:50 +08 — prereg `ib4-prereg-2026-10-02.md` and phase-1 code pushed (`ba7c08af0`); mirrored to node A; raw files
  pinned under `/data/dev2/private/data/ib4/raw`.
- 13:00 +08 — first build (counts only) showed the IB1 `sentfin` dedupe, per-name `fc_pick` cells and the Arabic column
  bug; amendment 1 before any audit. Next: rebuild, scans, pass 1 + G4, final, IX1, C1 recheck request, publish p1.
- 13:10 +08 — run a1 (`ae9eeeeb`): G0 / G0u controls pass, G1 / C1 names clean; G4 fails `sentfin3`, `fc_pick`,
  `w2c_act`, `smish`. Amendment 2: `smish`, `w2c_act` out of phase 1; `sentfin3` per-entity and `fc_pick` per-name
  balance; full re-audit in a fresh run.
