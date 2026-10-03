# 9B M10 amendment 13: half-learning-rate wave 1 (factory half-LR points, a four-seed half-LR α ladder) (2026-10-03)

Written 2026-10-02 ≈23:55Z (10-03 07:55 UTC+8) by the M10 continuation worker (third continuation), before any
half-learning-rate 9B Index read exists. At writing, KIB4H s1 / s2 (amendment 12) were at ≈ 58% of their updates on
node B GPU5 / 7, and the arm factory's 9B half-LR seeds (`KIB4-lrh` s1 / s2, `KIB4-lrq` s1 on node A; `KIB4-lrhh`
s1 / s2 on node B) at ≈ 80%. No soup of any of them was built. Amendments 7–12 are unchanged, except as noted here.
The budget is COORDINATION 05:38's: 60 GPU-h for the continuation.

## Evidence (values private)

- **4B:** the factory's two-seed half-LR soup `4b-LHS17IB4-lrh` passed the Nox-4B gate and is the released Nox
  (COORDINATION 05:20 / 06:05; disclosed before amendment 12). Its gains were on the deficit benchmarks.
- **9B, the α and LR results of M10** (verdicts vs `M10-KIB4-a40-bf16`): KIB4-a33 is level with KIB4-a40, KIB4-a50 is
  significantly below it, and KIB4L2-a40 (backbone LR ×2) is far below it. Each of these points moves a fixed
  direction further from Lux 1.0 or less far. Read together, they put the best step from Lux 1.0 near the released
  point's, about 2/5 of the full-LR seed soup, with a steep loss beyond it.
- **Why the α ladder moves up at half LR.** An Adam-type update is proportional to the learning rate, so a half-LR
  seed ends about half as far from Lux 1.0 as a full-LR seed. Averaging more seeds shortens the step further, because
  the seeds' own components partly cancel. A half-LR soup at α = 2/5 therefore sits well short of the released
  point's step. Any half-LR gain must come from a better direction (the 4B result), and the step that shows it is
  likely above α = 2/5. Hence the four-seed half-LR soup is read at α = 3/5 and 4/5, beside the preregistered
  two-seed α = 2/5 points.

## Points (each an FP32 uniform soup by `v2.dec.soup`; a member listed k times carries weight k / n)

| Point | Members | α | Measured as |
| --- | --- | --- | --- |
| `KIB4H-a40` | amendment 12: `[KIB4H, KIB4H, Lux × 3]` (`post.sh KIB4H`) | 2/5 | `M10-KIB4H-a40-bf16` |
| `KIB4-lrhh-a40` | the arm factory's `[s1, s2, Lux × 3]` of `KIB4-lrhh` (KIB4's TRAIN and lock; backbone LR 5e-6 and head LR 5e-5, as KIB4H; seeds 21 / 22), linked with `xpts.sh link` (`M10_LINK_AS`) | 2/5 | `M10-KIB4-lrhh-a40-bf16` |
| `KIB4-lrh-a40` | the arm factory's `[s1, s2, Lux × 3]` of `KIB4-lrh` (KIB4's lock; backbone LR 5e-6, head LR 1e-4; seeds 15 / 16), linked likewise | 2/5 | `M10-KIB4-lrh-a40-bf16` |
| `HLR4` (not measured) | `[KIB4H, AF-KIB4-lrhh]`: the uniform soup of the four seeds' BEST checkpoints at the same half learning rates | 1 | — |
| `HLR4-a60` | `[HLR4 × 3, Lux × 2]` | 3/5 | `M10-HLR4-a60-bf16` |
| `HLR4-a80` | `[HLR4 × 4, Lux]` | 4/5 | `M10-HLR4-a80-bf16` |

- The Lux member is M10's pinned zero-step checkpoint (node B `m10-KUP-s1`), checked against
  `lux-zero-m9-KIB-s1.sha256` before each build, as `post.sh` does. `xpts.sh` gains the member `LUX` for this.
- Before a factory point is linked, its seeds' TRAIN and teacher hashes are checked equal to KIB4's lock
  (`2e72bcfd…` / `377f8878…`), so audit `m10c` (0 item rows) covers every point here.
- `KIB4H-a33` / `-a25` are built by the post chain and stay unmeasured (amendment 12).

## Measurement, gate and placement

- Each point is measured once on panel-7 with M10's IX1 chain and gated against `M10-KIB4-a40-bf16` (amendment 7's
  IF1: paired bootstrap, 2,000 replicates, 95% lower bound > 0). The gate bootstraps start as soon as the run is
  scored (`M10_GATE_WAIT=scored`), beside the chain's own bootstraps.
- **Node A, pool GPU5–7** (the factory's `KIB4-lrh` / `KIB4-lrq` seeds release them, ≈ 00:30Z): `KIB4-lrh-a40`,
  then `HLR4-a60` (staged on node C from node B, then copied to node A).
- **Node B, pool GPU2 / 3 / 5 / 7** (GPU2 / 3 when `KIB4-lrhh` ends, GPU5 / 7 when KIB4H ends): `KIB4-lrhh-a40`,
  then `HLR4-a80` and `KIB4H-a40`.
- One chain per node at a time places shards; a node's next chain starts once the previous one has placed all seven.

## Wave 2 (conditional; at most two more reads, chosen by these rules from wave 1's point deltas d vs KIB4-a40)

1. **If no wave-1 point passes** and the better of HLR4-a60 / HLR4-a80 has d > 0: one more step on the HLR4 ladder
   in its rising direction: `HLR4-a100` (`[HLR4]`) if d(a80) > d(a60), otherwise `HLR4-a50` (`[HLR4, Lux]`).
2. **If d(KIB4-lrh-a40) ≥ d(KIB4H-a40) − 0.15** and the best HLR4 point has d > 0: `LRX6-aB`, the uniform soup of
   all six half-LR seeds (`[HLR4 × 2, AF-KIB4-lrh]`, each seed 1/6) at the best HLR4 point's α B, mixed with Lux as
   above.
3. **Quarter LR** (`KIB4-lrq`, one factory seed) is not read in wave 2. At α ≤ 1 a one-seed quarter-LR point sits at
   about half the released point's step or less. A two-seed quarter-LR arm is asked of the arm factory only if a
   half-LR point passes or the HLR4 ladder still rises at α = 4/5.

## Release

- The release candidate is the highest measured passer once its formal typed-FINAL run (R3) and release inputs are
  ready. A wave-1 read finishing within about 60 minutes of that is awaited first. Later passers are gated against
  the new release.
- Release path and checks are amendment 7's (`dev2-9b-m10c-2026-10-03/ops`): IF1, R3, IF3 (audit `m10c`), 86-request
  package parity and `gate evaluate`, then the fast-path post-checks and the purge of `f3122c7c`'s weights
  (`rewrite_history=False`, node copy kept). The candidate's `make_m10c.py` entry is added when it is chosen.

## Budget

About 28.9 GPU-h were committed at writing, including KIB4H-a40's Index run. Wave 1 adds four Index runs (≈ 10.8),
for ≈ 39.7. Wave 2 adds at most two (≈ 5.4) and a release ≈ 3–5, for ≈ 50 of the 60 approved. Speculative formal
runs (≈ 0.25 GPU-h each) are counted as they run.

## Revision 1 (2026-10-03 ≈00:05Z, before any half-LR read or build)

COORDINATION 07:55: 27B #7 claimed node A GPU5–7 for M9 half-LR seeds when the factory's 9B seeds end. Node B GPU2 / 3
(the factory's `KIB4-lrhh` seeds) and GPU5 / 7 (KIB4H) remain for 9B, so the placement and the first read change:

- **All wave-1 reads run on node B, pool GPU2 / 3 / 5 / 7, one at a time in this order:** `KIB4-lrhh-a80`,
  `HLR4-a80`, `HLR4-a60`, `KIB4H-a40`. KIB4H-a40 is read last because the α argument above makes it the least likely
  passer. It is still read (amendment 12).
- **`KIB4-lrhh-a80` replaces `KIB4-lrhh-a40`:** `[AF-KIB4-lrhh × 4, LUX]`, the factory's two-seed half-LR arm soup at
  α = 4/5. It is the first point that can be built (≈ 00:50Z), and it tests the higher-α half-LR step an hour
  before HLR4 exists. `KIB4-lrhh-a40` is not read.
- **`KIB4-lrh-a40` moves to wave 2** as `KIB4-lrh-aB` (the factory's backbone-only half-LR pair at the best wave-1 α
  B, built on node A), read only on GPUs outside node B that are free by then. Wave-2 rule 2's condition becomes
  d(KIB4-lrh-aB) ≥ d(best HLR4 point) − 0.15.
- If node C GPUs are granted when the 4B owner's current reads end (≈ 03:45Z), the remaining wave-1 reads run there
  in parallel (packages copied B → C, SHA-256 lists equal).
- The budget is unchanged: four wave-1 reads.
