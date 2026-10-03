# Decoder M17b wave 7 preregistration: the factory's UP / IB4-dose arms, read alone, then rule-based soups (2026-10-03 ≈20:30Z)

Written before any wave-7 arm has finished training and before any wave-7 Index result exists. Wave 6
([prereg](dec-m17b-wave6-prereg-2026-10-03.md), [amendment 1](dec-m17b-wave6-amendment-1-2026-10-03.md)) has
produced no successor so far; `4b-XALLU2` and `4b-AFxALL` are still on the Index.

## What wave 6 taught (no values)

- Every ten-member soup and the seven-arm soup with extra seeds (`4b-XALLx`) scored below the release. Most of each
  loss is RAGTruth: the release's RAGTruth reading falls as soon as more seeds of the same arms join, so part of it is
  a favourable seed draw that regresses with averaging. GPQA Diamond, When2Call and MuSR follow.
- Single arms stay at the level of the release's members (`4b-LHS23IB4` included).
- So more members of the same recipes do not lift the soup. New members must bring a skill that survives averaging.

## Arms (trained by the arm factory; amendments 5 and 8 on its branch)

| Arm | Change | Ready (UTC+8) |
| --- | --- | --- |
| `4b-SDMLIB4-UP`, `4b-LHS17IB4-UP` | UP weights (kept released rows ×1.5, IB rows ×1) on the audited TRAIN | ≈ 06:25 |
| `4b-LHS17ML-UP`, `4b-LHS17IB4X-UP` | the same | ≈ 07:35 / 07:45 |
| `4b-SDMLIB4W2` | IB4 p1 rows ×2 on `4b-SDMLIB4`'s TRAIN | batch 3 |
| `4b-SDMLIB4-lrh` | half LoRA / head LR | batch 3 (re-run) |

## Measurement and soups (rules fixed now)

1. Each finished arm's two-seed soup (both seeds' BEST checkpoints merged, as `af-soup.sh`) is read **once alone**:
   BF16 release copy, IX1, panel-8, paired bootstrap vs the release's run. These are information points, never
   released on their own (single arms sit below the release).
2. **Soup rule.** An arm *qualifies* if its single-arm Index point is at least the median single-arm point of the
   release's seven members (M17's runs).
   - `4b-W7xALL`: the release's seven two-seed soups plus every qualifying wave-7 arm (uniform, one arm one vote).
     Built only if at least two arms qualify.
   - `4b-W7UP`: built only if at least three UP arms qualify. The qualifying UP arms plus `4b-LHS17UP` and M15
     `4b-LHA10SDML`, uniform.
3. Every built soup is measured once. The release gate and rule are unchanged: the 95% lower bound vs `d55528d1` must
   be above 0, plus R3, IF3 and the 86-request parity, the fast-path post-checks and the purge. Any new TRAIN or
   weights file of a member is audited row-level before release (the UP and W2 variants reuse audited TRAIN files;
   only the weights files are new).
4. Order: arms in the order they finish; soups as soon as their members are measured.

## Budget

About 15 GPU-h of the 4B owner's 40 remain after wave 6. Six single-arm reads (≈ 13 GPU-h) plus up to two soups
(≈ 4.5) slightly exceed that, so the 4B owner asks the coordinator for +5 GPU-h when the first soup is due, or drops
the last arm's read.
