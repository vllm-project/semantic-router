# Decoder M17b wave 7, amendment 2: every low-LR arm read alone, and three rule-based low-LR soups (2026-10-03 ≈22:30Z)

Written by the third 4B owner (2d3664f4) at takeover, before any wave-7 result exists: `AF-4b-LRHxXALL-m50-bf16` is
still in its shards, and none of the candidates below is built. Earlier records: wave 7
[prereg](dec-m17b-wave7-prereg-2026-10-03.md), [amendment 1](dec-m17b-wave7-amendment-1-2026-10-03.md).

## Unchanged

- Gate: the full-panel paired bootstrap of the candidate minus the current release (`c60d3b5c`, the run
  `AF-4b-LHS17IB4-lrh-bf16`), 2,000 replicates, 95% lower bound above 0; then R3, IF3, the 86-request parity, the
  fast-path post-checks and the purge.
- Every candidate is measured once: BF16 release copy, IX1, panel-8, `ixchain.sh`.
- Members are two-seed arm soups as `af-soup.sh` builds them (both seeds' BEST checkpoints, uniform FP32). The arm
  factory builds the arm soups. The 4B owner builds every cross-arm soup.
- IF3: every member trains on a TRAIN file that audit6 covers (`4b-LHS17IB4`, `4b-SDMLIB4`, `4b-LHS17ML`,
  `4b-LHS17IB4X`, and for `4b-SDML-lrh` the `4b-LHA10SDML` file `fef6b036…`). Learning rate and seed do not change
  the rows.

## Added candidates

| Candidate | Members (FP32) | Built when |
| --- | --- | --- |
| `4b-LHS17ML-lrh`, `4b-LHS17IB4X-lrh`, `4b-SDML-lrh` | the factory's two seeds of each, read alone like `4b-SDMLIB4-lrh` | each pair DONE |
| `4b-LRHxALL-L2` | the members of `4b-LRHxALL` with `4b-LHS17IB4-lrh` listed twice | with `4b-LRHxALL` |
| `4b-LRHxQ` | `4b-LHS17IB4-lrh` plus every other half-LR arm that qualifies (below) | all five half-LR arms read |
| `4b-LRHxXALL-m75` | ¾ `4b-LHS17IB4-lrh` + ¼ `4b-SDMLxALL` | only if `4b-LRHxXALL-m50`'s point is above the release's |
| `4b-LRQxLRH` | ½ `4b-LHS17IB4-lrq` + ½ `4b-LHS17IB4-lrh` | only if `4b-LHS17IB4-lrq` qualifies |

- `4b-LRHxALL` is fixed now as amendment 1 defines it: uniform over the half-LR arm soups built when it is built.
  It is built as soon as the factory's `4b-SDMLIB4-lrh`, `4b-LHS17ML-lrh` and `4b-LHS17IB4X-lrh` soups exist,
  expected ≈ 23:00Z, so its members should be those three plus `4b-LHS17IB4-lrh`. `4b-SDML-lrh` ends later, and
  `4b-LRHxQ` covers it.
- **Qualifies:** an arm's single-arm paired bootstrap against the release's run has a 95% upper bound above 0, so it
  is not significantly below the release. `4b-LRHxQ` is built only if it has at least two members and its member
  set differs from `4b-LRHxALL`'s.
- `4b-LRHxALL-L2` carries the lesson that RAGTruth falls as members join: it keeps more of the release's own arm.

## Order and lanes (scheduling only)

1. Lanes: node B GPU4 / 6 (held), and node C GPU2–7 as the factory's seeds release them.
2. `4b-LRHxALL` runs in parallel with the single-arm reads, not strictly after `4b-SDMLIB4-lrh` as amendment 1's
   table lists it.
3. The rest run in this order: the single arms as their pairs finish, `4b-LRHxALL-L2`, `4b-LHS17IB4-lrq` (both
   seeds), then the conditional soups.
4. A candidate that passes is released at once. Later candidates are then gated against the new release's run,
   unless they are already on the Index; those finish and are not released over it.
5. The full-LR UP / W2 arms of the wave-7 prereg (`4b-SDMLIB4-UP`, `4b-LHS17IB4-UP`, `4b-SDMLIB4W2`) are read only
   after every low-LR candidate, and only if budget remains.

## Budget

The coordinator approved +30 GPU-h at the handoff, 70 in total. About 26 were used by the start of this amendment.
Ten reads cost about 24, and a release about 1.
