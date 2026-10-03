# Decoder M17b wave 7, amendment 4: gate everything against `4b-LRHxALL`; a conditional quarter-LR cross-arm soup (2026-10-03 ≈00:45Z)

Written after `4b-LRHxALL` passed the gate (its release is running) and before any other wave-7 result after
`4b-SDMLIB4-lrh`'s. Earlier records: [prereg](dec-m17b-wave7-prereg-2026-10-03.md), amendments
[1](dec-m17b-wave7-amendment-1-2026-10-03.md), [2](dec-m17b-wave7-amendment-2-2026-10-03.md) and
[3](dec-m17b-wave7-amendment-3-2026-10-03.md).

## The gate for every remaining candidate

- The current release becomes `4b-LRHxALL` once its upload verifies. Every other wave-7 candidate, whether
  already on the Index or not, is then gated against `AF-4b-LRHxALL-bf16`'s run:
  - a paired bootstrap of its merged results minus that run's (results `852e9928…`, pinned on nodes B and C as
    `ix1/af/refs/AF-4b-LRHxALL-bf16`);
  - the same panel, `v2.eval.ix1.paired_boot`, 2,000 replicates and seed 20261002;
  - it qualifies only if the 95% lower bound is above 0.
- The bootstrap against `AF-4b-LHS17IB4-lrh-bf16` that the running chains compute is kept as information.
- This replaces amendment 2's sentence that candidates already on the Index "finish and are not released over it".
  The release rule itself (the gate against the then-current release) is unchanged.
- New chains use the new reference directly.

## Added candidate

| Candidate | Members (FP32) | Built when |
| --- | --- | --- |
| `4b-LRQxALL` | uniform over four quarter-LR arm soups (LoRA / head LR 2.5e-5): `4b-LHS17IB4-lrq`, `4b-SDMLIB4-lrq`, `4b-LHS17ML-lrq`, `4b-LHS17IB4X-lrq` | only if `4b-LHS17IB4-lrq`'s single-arm point is above `4b-LHS17IB4-lrh`'s single-arm point. Then the factory trains the three new quarter-LR arms, two seeds each |

- The condition tests whether the LR trend continues, comparing the same arm at both learning rates against the
  same reference run.
- IF3: the three arms reuse their audited TRAIN files.
