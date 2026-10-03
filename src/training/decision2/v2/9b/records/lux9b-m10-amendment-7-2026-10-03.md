# 9B M10 amendment 7: the continuation after KIB4-a40 (arm-factory hand-offs, cross-arm weightings; 2026-10-03)

Written 2026-10-02 ≈16:15Z (10-03 00:15 UTC+8) by the M10 continuation worker (Lux-9B publisher; COORDINATION
2026-10-02 23:50 UTC+8), after the KIB4-a40 release and before any arm-factory 9B soup or Index result exists and
before any point named here was built. Index values stay private (node private run directories and
`decision2-program/private/9b-m10/`); this branch carries verdicts, counts and hashes only.

## Reference and gate

- **Current release:** `Decision-2.0-Lux-9B` `main` `f3122c7c` = M10 KIB4-a40 (BF16 identity `0ece5faa…`). Its IX1
  run `M10-KIB4-a40-bf16` (node C, panel-7) is the base of every comparison from now on.
- **IF1:** the paired bootstrap of a candidate's IX1 run minus `M10-KIB4-a40-bf16` (`v2.eval.ix1.paired_boot`
  exactly as `m10/ixchain.sh` runs it: 2,000 replicates, seed 20261002, cases resampled within benchmarks through
  the board weights, full panel), 95% lower bound > 0. The transfer-only variant (without HoVer, When2Call,
  iSarcasmEval, GSM8K, BPoMP) is recorded privately.
  - Each candidate is measured once, on its BF16 release copy. An arm-factory candidate is not measured again: its
    factory run gets this bootstrap on the node that holds it (`m10/ix.sh gate`), next to the factory's own bootstrap
    vs K-a13IB.
  - The KIB4-a40 run's merged results are copied from node C to the measuring node with SHA-256 checks
    (`m10/ix.sh kref`).
- **Integrity checks (unchanged):** formal typed-FINAL with no collapsed decision type (`m10/formal.sh`), package
  parity, the Hub `trust_remote_code` smoke (Transformers 5.17 / 5.18), and the row-level Index contamination audit of
  every member's TRAIN. If a point contains K-a13IB's arm soup KIB, the audit is rerun with K-a13IB's TRAIN added to
  the audited sets, and its planted control must be complete.
- **Selection:** among the passers whose integrity checks are in, the highest measured Index (ties: the larger lower
  bound). A candidate measured after a release must beat the then-current release.

## Candidates

1. **Arm-factory hand-offs** (factory prereg `75da74ff6`, amendment 2 `b1b34a3a9`): `KF-a40`, `KF-a50`,
   `KIB4W2-a40`, `KIB4L2-a40`, `KFxKIB-a40`, in the factory's order and as far as it measures them.
2. **Cross-arm averages with the factory's points** (no training; built on node A once the inputs exist). F* is the
   factory 9B point with the highest measured Index.
   - `Y1` = the uniform FP32 average of [F*, KIB4-a40].
   - `Y2` = the uniform FP32 average of [F*, KIB4-a40, X5-a33].
   - Built when F*'s run is scored, and measured once each in that order.
3. **Cheap cross-arm weightings of the existing M10 soups** (no training; built on node B, measured on node B's idle
   M10 GPUs now, while the factory trains):
   - `X7-a40` = the uniform FP32 average of [KIB4-a40, KX-a40, KSW-a40], that is
     0.6 × Lux 1.0 + (0.4 / 3) × (KIB4 + KX + KSW): X5-a33's three arms at α = 2/5, KIB4's better measured α.
   - `X8-a40` = the uniform FP32 average of [KIB4-a40, KSW-a40], that is 0.6 × Lux 1.0 + 0.2 × (KIB4 + KSW).
   - They are measured in that order. KX-a40 moves from node A to node B with `ix.sh soupcopy` (SHA-256 lists
     equal).
4. **One more KIB4-family arm,** only if some measured point of 1–3 has a positive point delta vs KIB4-a40. It
   needs a further amendment (data, seeds and budget line) before any of its data or GPU jobs.

Each average of α-points equals the per-tensor average of their arms and Lux 1.0 (FP32 soups are linear). Points
from the factory are read in place, or hard-linked into `m10/soup/AF-<NAME>` with equal SHA-256 lists. They are
never modified.

## Formal path

- The formal run (`m10/formal.sh` on a node A GPU, lease `track=9b-m10`) starts speculatively for the leading
  candidate as soon as its soup exists. The first is `KF-a40`, the factory's first-measured point.
- After that, the formal run starts for any candidate whose IF1 lower bound is > 0.
- There are at most three formal runs. A formal result never replaces IF1.

## Budget

- **30 GPU-h** (COORDINATION 2026-10-02 23:50 UTC+8), on top of M10's 90. The costs are about 2.7 GPU-h per own
  Index run (the factory's runs are on its budget), about 2 per formal run and about 2 for the release.
  Bootstraps, soups and BF16 copies run on CPU.
- No new Index or formal run starts once the continuation total reaches 27 GPU-h.

## Release

The highest passer goes out through the M10 release ops lineage (`release_ka13ib.sh` → `dev2-9b-m10-2026-10-02/ops`),
re-based on the KIB4-a40 release:

- the spec derives from `specs/dev2-9b-m10-KIB4-a40.json`;
- the current revision is `f3122c7c`, with its sealed gate receipt and decision;
- the purge removes KIB4-a40's blobs, with its node A release package as the node copy.

The fast-path post-upload checks apply (COORDINATION 2026-10-02 23:20 UTC+8). Banner A, the repository stays private,
`main` is re-read before the upload, and nothing is ever published concurrently.
