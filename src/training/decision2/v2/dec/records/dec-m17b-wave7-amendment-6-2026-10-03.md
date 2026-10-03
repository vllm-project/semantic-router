# Decoder M17b wave 7, amendment 6: the quarter-LR lever (2026-10-03 ≈02:00Z)

**Disclosed:** written after these reads (values private): `4b-LHS17IB4-lrq`, `4b-LRHxALL-L2`, `4b-LHS17ML-lrh` and
`4b-SDML-lrh`. `4b-LHS17IB4X-lrh` and `4b-LRH2` are unread. Earlier:
[amendment 5](dec-m17b-wave7-amendment-5-2026-10-03.md).

## What the reads say (no values)

- `4b-LHS17IB4-lrq` (quarter LR, two seeds) reads above the release `4b-LRHxALL` and above its own half-LR arm, but
  its gate lower bound is just below 0. The single arm gained the full → half → quarter steps in turn, each by less
  than the one before. It gains on PhishNChips, GPQA Diamond, RAGTruth, VAST and cfcolor, and gives back When2Call,
  WinoGrande, CRUXEval, ContractNLI and BFCL, where `4b-LRHxALL` is strong.
- `4b-LHS17ML-lrh` is significantly below the single-arm release, so it does not qualify. `4b-SDML-lrh` qualifies.
  `4b-LRHxALL-L2` reads slightly above `4b-LRHxALL`.

## Added candidates (gate unchanged: against `AF-4b-LRHxALL-bf16`'s run until a successor ships)

| Candidate | Members (FP32) | Built when |
| --- | --- | --- |
| `4b-LRQxLRHxALL` | ½ `4b-LHS17IB4-lrq` + ½ `4b-LRHxALL` | now (CPU, node C) |
| `4b-LRQxALL` (amended) | uniform over `4b-LHS17IB4-lrq`, `4b-SDMLIB4-lrq`, `4b-LHS17IB4X-lrq`, `4b-SDML-lrq` | all four pairs DONE |
| `4b-LHS17IB4-lre` | two seeds of `4b-LHS17IB4` at an eighth of the LR (1.25e-5), read alone | its pair DONE |

- **`4b-LRQxALL` is amended** (amendment 4 defined it before this read): `4b-SDML-lrq` replaces `4b-LHS17ML-lrq`,
  because `4b-LHS17ML-lrh` reads significantly below the single-arm release.
- **`4b-LHS17IB4-lre`** tests whether the trend continues one step further.
- **The arm factory handed off at 00:20Z with no successor yet**, so the 4B owner trains these seven seeds with the
  factory's own tools (`af-chain.sh` on node C, factory amendment 12): `4b-SDMLIB4-lrq`, `4b-LHS17IB4X-lrq`,
  `4b-SDML-lrq` and `4b-LHS17IB4-lre`, seeds 20260926 / 20260927 (only the first seed of `-lre` if the node gate
  binds).
  - Locked TRAIN files only, all audited (audit6), so IF3 holds.
  - These are the first arms the 4B owner trains; the GPU-hours count against the 4B budget.
- **Order:**
  - `4b-LRQxLRHxALL` goes first, on node C GPU2–4.
  - The seeds take the node C GPUs the 4B owner holds and any that are idle > 20 min with no claim.
  - The half-LR rule soups (`4b-LRHxQ`, `4b-LRHxTOP3`) come after the quarter-LR candidates, and only if budget
    remains.
