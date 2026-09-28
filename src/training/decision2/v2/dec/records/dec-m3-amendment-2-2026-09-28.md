# Decoder Milestone 3 — amendment 2 (matched own-Lux teacher arms; E8V re-queued)

Parents: [M3 preregistration](dec-m3-prereg-2026-09-28.md) (`5c8bbc569`),
[amendment 1](dec-m3-amendment-1-2026-09-28.md) (`271502d8f`). Written 2026-09-28
22:20 UTC+8. At this point the development readouts of S2T-s1 and S2T-s2 exist
and nothing else has been read. No AutoJev (J) arm, 4B arm or soup has a readout
yet. Nothing preregistered changes; this adds arms and one
selection rule for tiers with several qualifying artifacts.

## Why

The coordinator's 21:35 note requires a **matched own-Lux-target control for every
AutoJev-distilled candidate** (the M3 instruction allowed an own-teacher control,
which the T arms already give). Own Lux 1.0 targets exist for exactly the same
47,922 recipe rows AutoJev covers, so the Lux arm is an exact three-way match of
T and J. It is also a candidate in its own right: Lux 1.0 (post-key v3 65.8) is
a far stronger teacher than Nox 1.0 / Sol 1.0 and carries no provenance caveat.
Priority: item 4 (teachers) outranks item 5 (0.8B follow-up), so E8V moves behind
the Lux arms; no E8V job had started (its waiters were stopped at 22:17).

## Arms L (same recipe M3F, mixture `13804ac6…`, seeds 20260926 / 27 / 28)

| Arm | Start | Teacher (KL 0.5) | Cap |
| --- | --- | --- | ---: |
| N4L-s1/s2/s3 | Nox 1.0 `@cde2a68d` | own Lux 1.0 on the 47,922 recipe rows, own Nox on the 8,276 retention rows | 1.6 GPU-h each |
| S2L-s1/s2/s3 | Sol 1.0 `@ce0c018a` | own Lux 1.0 on the recipe rows, own Sol on the retention rows | 0.9 GPU-h each |

Lux targets, all from the private dataset and joined by id and input hash:

- pk1 canonical A0 file `m3/pk1/lux1/A0-train.canonical.jsonl` @ `d8eae3e4…`
  (`56627939…`), which covers the 6,547 A0s rows.
- RP-v2 wave 1 `m2/teachers/lux1/rp-v2/wave1.targets.jsonl` @ `7885baf6…`
  (`47049a6b…`), 12,686 rows.
- Wave 2 @ `6bd8eb4d…` (`2c6ab38d…`), 28,643 rows.
- Wave 4 @ `03b1e72d…` (`084adacd…`), 46 rows.

The teacher files are composed like the J files: `merge_labels --part <own-1.0>`,
then `--override` for each of the four Lux files, with `--override-subset`. Their
hashes are recorded at composition.

**Order (node B):**

| GPU | Queue |
| --- | --- |
| GPU3 | S2L-s1, then S2L-s3, then E8V-s1 |
| GPU4 | S2L-s2, then E8V-s2 |
| GPU0 | N4L-s1, then E8V-s3 |
| GPU1 | N4L-s2 |
| GPU2 | N4L-s3 |

Each queue starts after that GPU's preregistered chain has finished. E8V keeps its
amendment-1 rules. It runs only as long as the milestone has GPU time left.

## Rules

- Artifact, finalist and formal rules: as in the prereg (soup rule; P > P(1.0) − 4
  and ≥ 75% type floors; 16,384-token formal runs; comparator = the stricter of the
  adopted run and the 16K same-limit control).
- Measured 16K same-limit controls (node A, 21:58–22:14; selection-neutral 1.0 runs
  collected when GPU5 came back, before any M3 formal candidate existed): Nox 1.0
  **55.689** (= its 8K control) → the 4B comparator stays the adopted Nox1 run
  **56.470**; Sol 1.0 **45.781** (vs adopted 45.580, +0.20 [−0.54, +0.80]) → **the 2B
  comparator is the 16K control, 45.781**.
- **Several qualifying artifacts in one tier (fixed now, before any 2B/4B formal
  result):** the recommended candidate is the clean-teacher artifact (T or L) with
  the higher v3 point estimate; a J artifact is recommended instead only if its v3
  paired 95% lower bound against that clean artifact is > 0, and it then carries
  the AutoJev provenance disclosure. All qualifying artifacts are reported.
