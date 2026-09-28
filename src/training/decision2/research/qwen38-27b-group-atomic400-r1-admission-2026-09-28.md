# Qwen3.8-27B group-atomic400 r1: independent CPU admission

**Result: HOLD before zero-step GPU preflight.** The new r1 candidate was
locked in signed preregistration commit `80cfcc853` before this independent
validation. Its seed 96, 2,560 IDs/input hashes, 160 nominal updates, teacher
mask, token exposure and overlap stop rules were not changed after observing
the result. This is neither a model score nor a revised pass for the failed
Score≥460 or pooled candidates. GPU-hours: **0**.

The exact mirrored admission code SHA-256 is
`2f2cf940daf0508cb0bb1de1ea70953d30a50498246a832a7ae1a26ff1fea484`.
It ran in pinned CPU image
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
The final mode-0600 private receipt SHA-256 is
`3f3da6e6ba616565a5ceb0cc7a4552d7c5aa563577598354b7aa29b7e88ae49d`.
It contains exact selected and protected pair IDs, which are omitted here.
The locked schedule digest remains
`0743da3f1ed18cc8eb824c41c736d92b23fb3d8a738e1b3401ff830229424be5`.
No protected target value or model prediction was consulted.

| Independently checked property | Result |
| --- | --- |
| Selected ID/full-input/native-token identity | 2,560/2,560; Choice 1,201, Noul 940, Score 419; 1,244,573 raw and 1,253,880 eight-token-rounded tokens |
| Whole source/group integrity | 1,932 selected groups; **zero** missing task type or row |
| Saved near-exclusion identities | 83 TRAIN rows/61 entire source groups; CAL 101 and SELECT 60 bounded pairs reproduced |
| Composite `(source, group_id)` and bare `group_id` collisions with SELECT/CAL | **Zero** under pinned partition metadata |
| Teacher artifact row/option identity and mask | 7,455/7,455 vectors; 971 masked, including 174 Score and 63 three-level Score |
| Declared rights/source ledger | Nine raw TRAIN source IDs mapped to seven pinned ledger entries; integrity passed, not an independent legal conclusion |
| Eight protected roles | All 20,263 gold-free input projections and file hashes verified |
| Exact raw/normalized complete input and state matches | **Zero** in all seven disjoint roles |
| Bounded near state evidence | **One SELECT pair and one CAL pair**; five other roles zero |
| Bounded near complete native input | SELECT **344**, CAL **262** pair IDs; five other roles zero |

The two missed near-state pairs are within their respective shared template
families, despite having different original source/group IDs. Their
normalized state edit ratios are approximately `.943` and `.972`, with word
set Jaccard about `.846` and `.825`. They are conservative collision signals,
not proof of identical answer labels; the r1 preregistration nevertheless
requires **zero** such pairs. The 83-row capacity exclusion used a
two-span/prefixed-state candidate index; the independent plain-state scan
found these two additional matches. This is exactly why the independent
validation was required. Removing their groups now would alter the locked
candidate, so r1 remains HOLD.

The 606 near-complete-input pairs are not silently treated as duplicates or
as harmless boilerplate. All are within the same named task family; none
shares an exact `(source, group_id)`. Among them, 135 have normalized state
edit ratio at least `.9`. The private receipt preserves every pair ID for
semantic/source review. The projected native evaluation roles preserve full
input digests but do not expose original source record IDs, so independent
original-corpus provenance remains unverified even where this bounded scan
finds zero lexical matches. These are separate blockers, not reasons to
relax the two-pair hard stop.

The first CPU launch failed because a read-only runtime lacked a writable
temporary directory (private receipt SHA-256
`8f9fa498b338fb252d9ac62f78c2c411c1a750ce35c6f91c5c9b9d545d279a28`).
The second reached an input-only projection error caused by the projection
helper omitting `split`; it did not read any protected target (receipt
`43ae7cc879415236ffe3263bfac6bdc41579c793455612dda0f3d2d78980dac7`).
The corrected run first exposed the HOLD (`637df2795403403cbc7b0ad2b2f4a1e0eb16861b6d42d4a1d84bd2d1958196eb`);
subsequent instrumentation added reproducible source-family similarity
aggregates and bare-group cross-split identity, without changing thresholds
or the candidate. All receipts remain private and immutable.

**Next decision:** no GPU preflight for r1. A distinct future r2 would need a
new predeclared selector that excludes the additional near-state source
groups before locking rows, then repeats the exact capacity, teacher, rights,
full-input and semantic/source audits. The old Score≥460 floor is still
mathematically infeasible under the current whole-group exclusions. Any new
candidate also needs actual dynamic-padding verification and the separately
versioned zero/one-step native parity gate before an optimizer can run.
