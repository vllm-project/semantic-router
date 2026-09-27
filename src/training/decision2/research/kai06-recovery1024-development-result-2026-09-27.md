# Kai2 0.6B recovery arm: development result and stop

This is the single predeclared 1,024-row recovery arm from
`kai06-human-recovery1024-prereg-2026-09-27.md`. All results below are
**development diagnostics**, not independent JevArena v3 evidence. The
project's v3 labels had previously been accessed; this arm made no new
formal predictions or scores.

## Fixed arm and technical evidence

- Start: own-Kai2 clean-v2 native export manifest
  `03d90e449078bacd18390ad52c3ba3af7cff86bb507c2c2f9088f0260305d1ab`.
  The 700-row frozen SELECT source-checkpoint predictions, export predictions
  and training-configured/restored predictions had identical selected classes
  and **0.0 maximum numeric drift**. Gold-free zero-step receipt SHA-256:
  `542b488d71a8aecb73c81cff7f52e29303fdb397d4e899f697ba2501219157e0`.
- TRAIN SHA-256
  `58b94ac987b2fe134a392310a8d9d63f3d0e996a6016884f6dafadf3ae5449da`;
  SELECT SHA-256
  `429285511fe5737dabfc799f01e7715155ad0f913994d5d10f5fdb627f6b0976`.
  Native admission and prompt-only overlap checks passed as documented in
  `kai06-recovery1024-preflight-2026-09-27.md`.
- The run paused at step 1 for a numerical check and resumed from its exact
  checkpoint manifest `04a71786063fd2c7c57a8deb60187aa388cfe5d02201e33508775246a3f5f53d`.
  All 16 planned updates completed with finite losses and gradients. The
  native trainer selected step 16, the preregistered sole eligible trained
  checkpoint. Peak reserved accelerator memory was 11.13 GB. The final
  checkpoint manifest is
  `92b63d9a04b001ff2f6449381f29e566af60076d6c5f6bfbdca75680bd670068`;
  the 30-file native export manifest is
  `4d1d75e2c576fc2afe7db779eb6c7c8f3cae848348c065cd8727493567fb6997`.
  Completed run receipt SHA-256:
  `be891e8f74746693c6e8df2240e0efd9ea20f24602000ac6c51327d8a389ab37`.
- One isolated AMD GPU was used. Container wall time, including the initial
  zero-step checker failure and retry, step-1 pause, resumed training and both
  development inference passes, was approximately five minutes (about 0.08
  GPU-hours). No formal panel inference was performed for this candidate.

## Frozen development gates

| Gate | Predeclared minimum | Result | Decision |
| --- | ---: | ---: | --- |
| SELECT row hard accuracy | 0.72710 | 0.73857 | Pass |
| typed DEV Choice/Noul mean | 0.37875 | 0.34938 | **Fail** |
| typed DEV Score accuracy | 0.48000 | 0.31750 | **Fail** |
| CSS pilot median task macro-F1 | 0.17977 | 0.15274 | **Fail** |

typed DEV was 1,600/1,600 valid: Choice 0.23375, Noul 0.46500 and Score
0.31750. CSS pilot was 1,408/1,430 valid, with 22 complete-input overflows
counted as failures. Relative to the starting Kai2 control, Choice moved from
0.23000 to 0.23375, Noul from 0.48750 to 0.46500, Score from 0.52000 to
0.31750, and CSS pilot from 0.15728 to 0.15274. The SELECT increase therefore
does not establish task transfer. DEV prediction/score SHA-256 are
`fd2dba7be7dcf4167f04b31aae248481195a4e2db4a93616ee6516e037bcf9c3`
and `cdddba7059bf66c18b2745f2a83218a3fd73ed9163a34e599da819ec37381dc6`;
CSS pilot prediction/score SHA-256 are
`e3324036cc0a57eb40a90f3ce1940de0676af7eb7e55f073077e7edb68642c50`
and `ff1c74dd51cb5f2940d502ce9c4ba60d46586a67caca011db01168f43111159b`.

**Decision: HOLD and stop this arm.** No step-8 substitution, mixture change,
CAL fit or formal v3/public231 run is justified. The predeclared data repair
was completed before any training outcome; the later zero-step checker schema
repair did not change the candidate or evaluation. This arm also lacks a new
untouched source-disjoint human transfer result and exact-package weight
redistribution review. A future 0.6B hypothesis must use a new prospective
data and evaluation plan rather than select on these outcomes.
