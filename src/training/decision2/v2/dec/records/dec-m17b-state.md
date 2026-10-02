# Decoder M17b (4B owner and the only Nox-4B publisher; M17 continuation) — state

Branch `xunzhuo/decision-2-training-dec-m17`, worktree `vllm-sr-dev2-dec-m17`. M17's record:
[`dec-m17-state.md`](dec-m17-state.md). Release in force: `vllm-sr/Decision-2.0-Nox-4B@d55528d1` = `4b-SDMLxALL`
(public since 2026-10-03 01:17 UTC+8). Index values stay private (node private stores and
`decision2-program/private/m17b/`); this file has none. Times UTC.

## 2026-10-02 20:05Z — wave 6a: no successor; wave 6b on node C

- **Not successors** (full-panel paired bootstrap vs `DEV2.0-4B-SDMLxALL-bf16`, 2,000 replicates; values private):
  `AF-4b-AFxALL3-bf16` (95% CI entirely below 0) and `AF-4b-AFxALL2-bf16` (point below 0, lower bound below 0).
  Both lose most on RAGTruth, then GPQA Diamond, When2Call and MuSR; RAGTruth follows `4b-LHS17UP`'s share of a soup.
- **Amendment 1** (`0e444cb1a`, written before any further result): `4b-XALLx` (`7ae47bc2…`, the release's seven
  arms at their most seeds) and `4b-XALLU2` (`58fa0585…`, UP listed twice); `4b-AFxALL4` dropped. Both on node C
  GPU1–4 (leased 19:30Z, idle since 19:07Z), parity PASS 19:44Z. Their formal runs are collected and scored (node F
  co-tenant, node A); mlx-diag collecting.
- `AF-4b-AFxALL-bf16` (node F GPU6–7) and the information points `AF-4b-LHS23IB4-bf16` (scored) and
  `AF-4b-LHS17IB4-lrh-bf16` (node A GPU5–7) follow the prereg.
- Merged `9d90afd10` (shared-context switch, COORDINATION 02:23): the wave-6 spec's `runtime_source` is this
  branch's mirror `b50e8650b` (phase A plus the opt-in switch); 40 release tests pass on node A (vendor source,
  shared ctx, bf16 copy, hub upload / collect).
- Fixes: `af-stage.sh` now finds an IX1 entry that is the first name of its list (`b3a917279`); the launcher's
  `AF-4b-XALLx` / `AF-4b-XALLU2` entries sit on the factory's 4B line (`247440f37`).
- GPU-h so far ≈ 14 (Index ≈ 11.5, formal and readouts ≈ 1.5, parity and merges ≈ 1).

## 2026-10-02 19:05Z — wave 6a on the Index (two lanes); release inputs staged

- **Soups built** (uniform FP32, `af-soup.sh` from this branch's mirror; node C merges on GPU5, SELECT agreement
  checked): `4b-LHS17IB4X-s34` `9c9e9355…`, `4b-LHS17ML-s34` `e4056234…` (node C, copied to F, lists equal); on node F
  `4b-LHS17UP-x4` `4422a353…`, `4b-AFxALL` `8f282287…` (12), `4b-AFxALL2` `70612d1c…` (10), `4b-LHS17IB4X-x4`
  `6edd6165…`, `4b-LHS17ML-x4` `8c69cd51…`, `4b-AFxALL3` `df95b286…` (10).
- **Index (reference `DEV2.0-4B-SDMLxALL-bf16`, results `54389c5a…` pinned on F and A):**
  - `AF-4b-AFxALL2-bf16` (BF16 `3cfe2816…`): node F GPU6–7, parity PASS 18:12Z, shards running.
  - `AF-4b-AFxALL3-bf16` (BF16 `06b3633f…`): node A GPU5–7 (02:00 assignment; panel-8 copied C → A, lists equal),
    parity PASS 18:22Z, last two shards running.
  - Queued: `AF-4b-AFxALL` on node F after AFxALL2; information point `AF-4b-LHS23IB4-bf16` on node A.
- **IF3:** `audit6` (node C CPU, one audit of the ten distinct member TRAIN files of every wave-6 candidate): 120,226
  Index rows, planted 200 / 200, **0 item rows in all ten**.
- **Formal path (R3, card reports) run ahead of the Index results**, since it costs about 15 GPU-minutes per
  candidate: `m17-4b-AFxALL2` and `m17-4b-AFxALL3` collected on node F (co-tenant, T = 1), scored on node A: types
  choice / Noul / Score OK for both; mlx-diag collected and scored.
- **Release ops** `v2/release/records/dev2-4b-w6-2026-10-03/ops/` (`ca2b8fe94`). The card's Index input takes each
  other tier's point from the released input that scored its current main (9B Lux `f3122c7c`, 27B Vega `5c85c127`).
  Already staged on node A for both candidates: current-revision receipts, the gate files, the BF16 checkpoints.
- GPU-h so far ≈ 4.5 (Index ≈ 3.5, formal and readouts ≈ 0.6, merges ≈ 0.1, parity ≈ 0.3).

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
