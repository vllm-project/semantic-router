# Decoder M17b (4B owner and the only Nox-4B publisher; M17 continuation) — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17`. M17's record:
[`dec-m17-state.md`](dec-m17-state.md). Release in force: `vllm-sr/Decision-2.0-Nox-4B@d55528d1` = `4b-SDMLxALL`
(public since 2026-10-03 01:17 UTC+8). Index values stay private (node private stores and
`decision2-program/private/m17b/`); this file has none. Times UTC.

## 2026-10-02 18:05Z — started; wave 6 preregistered

- Merged integration (`47dd06be7`, the central Hub visibility policy) and the arm-factory branch (its soup, staging
  and measurement tools, its IX1 entries and M10's `ixchain.sh`).
- Wave 6 prereg: [`dec-m17b-wave6-prereg-2026-10-03.md`](dec-m17b-wave6-prereg-2026-10-03.md). Candidates in order:
  `4b-AFxALL2`, `4b-AFxALL3`, `4b-AFxALL`, contingent `4b-AFxALL4`.
- Inventory:
  - M17's 21 soups and their Index runs (node F soups, node C runs and scores) are intact. The factory's built 4B
    soups: `4b-LHS17IB4-x5`, `4b-SDMLIB4-x5`, `4b-LHS17IB4ML`, `4b-LHS23IB4` (node F); `4b-LHS17IB4-lrh`,
    `4b-LHS17UP-s34` (node C). Batch-2 seeds `4b-LHS17IB4X` s3 / s4 and `4b-LHS17ML` s3 / s4 are DONE (node C,
    not merged). `4b-SDMLIB4-lrh` s1 / s2 failed at the node F return (the factory re-runs them).
  - The factory's node F soup pipeline (`soups-all.sh`: `4b-LHS17UP-x4`, `4b-AFxALL`, `4b-AFxALL2`) survived the
    return and waits only for `4b-LHS17UP-s34` and `4b-LHS17IB4-lrh`: both are being copied C → F (node A relay,
    SHA-256 lists compared).
  - `AF-4b-LHS17IB4ML-bf16`: shards 0–5 exit 0, shards 6–7 died at the return (not scored).
  - Per-case results for the paired bootstrap against `d55528d1`: the release's run `DEV2.0-4B-SDMLxALL-bf16` on
    node C (`ix1/runs`) and node F (`ix1/af/refs`, results hash pinned).
- Leases: node F GPU6–7 (`track=eval-ix1`, 4B owner Index runs).
