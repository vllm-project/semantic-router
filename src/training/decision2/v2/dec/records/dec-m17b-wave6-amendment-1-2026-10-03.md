# Decoder M17b wave 6, amendment 1: two one-factor variants of the released soup (2026-10-03 ≈19:30Z)

Written after `AF-4b-AFxALL2-bf16` and `AF-4b-AFxALL3-bf16` were scored (point estimates only; both below the
release's run, so neither can pass; values private) and before `AF-4b-AFxALL` or either candidate below has a result.
Prereg: [`dec-m17b-wave6-prereg-2026-10-03.md`](dec-m17b-wave6-prereg-2026-10-03.md).

## What the two results say (no values)

Both ten-member soups lose most on RAGTruth, then GPQA Diamond, When2Call and MuSR. In M17's runs RAGTruth was
high only for `4b-LHS17UP` (kept released rows weighted ×1.5) and fell as UP's share of a soup fell (UP alone, the
.5 point, `SDMLxALL` at 1/7, `SDMLxALL9` at 1/9, `SDMLxALL15` at 1/15); wave 6a cuts UP to 1/10. Adding the three new
arms and the extra seeds together therefore did not help. Two clean one-factor variants of `4b-SDMLxALL` (the
release) separate the two effects.

## Candidates (uniform FP32 averages; measured once each, after `4b-AFxALL`)

| Candidate | Members | One change vs `4b-SDMLxALL` |
| --- | --- | --- |
| `4b-XALLx` | M15 `4b-LHA10SDML`, `4b-LHS17SD`, `4b-LHS17UP-x4`, `4b-LHS17IB4-x5`, `4b-LHS17IB4X-x4`, `4b-SDMLIB4-x5`, `4b-LHS17ML-x4` | every arm at its most seeds (the arm weights stay 1/7) |
| `4b-XALLU2` | `4b-SDMLxALL`'s seven two-seed soups with `4b-LHS17UP` listed twice | UP's weight 2/8 instead of 1/7 |

- Order: `4b-XALLx` (node A lane), `4b-XALLU2` (the next free lane). Same measurement, gate and release rule as the
  prereg. Both reuse audited TRAIN files (audit6 covers them).
- **Dropped:** the contingent `4b-AFxALL4` (`4b-AFxALL3` plus `4b-SDMLIB4-lrh`); its ten-member base scored below
  the release and one more member would not change that.
- Information points stay last: `AF-4b-LHS23IB4-bf16` (running), then the two missing shards of
  `AF-4b-LHS17IB4ML-bf16`, then `4b-LHS17IB4-lrh`.
