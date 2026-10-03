# Arm factory — amendment 12: the 4B owner continues the factory's 4B line (2026-10-03 ≈02:00Z)

The factory handed off at 00:20Z (`e41c091d0`, "continuation needed"), and no continuation has started. The 4B
owner (decoder M17b, 2d3664f4) runs the factory's 4B tools for the arms its wave-7 amendment 6 needs. The tools
are `af-chain.sh`, `af-arm.sh`, `af-soup.sh` and the node C data locks. The 9B backlog (amendment 11) is untouched.

## Arms (two seeds each, 20260926 / 20260927; locked, audited TRAIN files; one changed flag)

| Arm | Data lock | Change from M17's stage-2 recipe |
| --- | --- | --- |
| `4b-SDMLIB4-lrq` | `4b-SDMLIB4` | LoRA / head LR 2.5e-5 |
| `4b-LHS17IB4X-lrq` | `4b-LHS17IB4X` | LoRA / head LR 2.5e-5 |
| `4b-SDML-lrq` | `4b-SDML` | LoRA / head LR 2.5e-5 |
| `4b-LHS17IB4-lre` | `4b-LHS17IB4` | LoRA / head LR 1.25e-5 |

- **Node C training gate:** 44 → 56 GPU-h, to fit seven or eight more seeds. These hours count against the 4B
  owner's budget, not the factory's 130.
- **Rules (prereg) unchanged:**
  - seed cap 2.5 GPU-h;
  - preflights before each first seed of an arm;
  - a failed preflight stops its arm, with no rerun;
  - markers under `/data/dev2/runs/af/4b/status/`.
- **Two-seed soups** use `af-soup.sh` on node C. Merges run on a GPU the 4B owner holds.
