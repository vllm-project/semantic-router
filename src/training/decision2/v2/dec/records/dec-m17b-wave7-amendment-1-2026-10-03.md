# Decoder M17b wave 7, amendment 1: low-LR candidates (2026-10-03 ≈21:45Z)

Written after `4b-LHS17IB4-lrh` passed (wave 6 amendment 2) and while its release runs, before any result below
exists. Wave 7 prereg: [`dec-m17b-wave7-prereg-2026-10-03.md`](dec-m17b-wave7-prereg-2026-10-03.md).

The half-LR arm gains on the 4B deficits and gives back the strongholds (HoVer, WinoGrande, BANKING77, CLINC150), where
the full-LR cross-arm soup `4b-SDMLxALL` is strong. The arm factory now trains low-LR arms (COORDINATION 05:20).

## Candidates, in order (each measured once, BF16 release copy, IX1, panel-8)

| Candidate | Members (FP32) | Why |
| --- | --- | --- |
| `4b-LRHxXALL-m50` | ½ `4b-LHS17IB4-lrh` + ½ `4b-SDMLxALL` | the two profiles are complementary; CPU only, both soups on node F |
| `4b-SDMLIB4-lrh` | the factory's two seeds (batch 3, done) | the second half-LR arm, read alone |
| `4b-LRHxALL` | uniform over every half-LR arm whose two-seed soup is built by its turn (`4b-LHS17IB4-lrh`, `4b-SDMLIB4-lrh`, then the factory's `4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`, `4b-SDML-lrh` as they finish) | the cross-arm lever on low-LR arms |
| `4b-LHS17IB4-lrq` | the factory's quarter-LR seeds (one or two) | the LR trend |

- The gate is against the then-current release. If `4b-LHS17IB4-lrh` lands, that is `AF-4b-LHS17IB4-lrh-bf16`'s run
  (node A, results hash pinned when copied). Otherwise it is `d55528d1`'s.
- Every member TRAIN is audited (audit6: `4b-LHS17IB4`, `4b-SDMLIB4`, `4b-LHS17ML`, `4b-LHS17IB4X`,
  `4b-LHA10SDML`). Learning rate and seeds do not change the rows.
- The single-arm rule of the wave-7 prereg (members must hold up on their own) applies to the full-LR UP arms only;
  the factory has stopped those for the low-LR ones.
