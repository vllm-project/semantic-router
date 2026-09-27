# Official Qwen3 0.6B TRAIN-only external-teacher KL result

Decision: **HOLD; do not advance this arm to DEV, CSS, JevArena, JevBench or
release**. This is one completed, preregistered development comparison, not
formal or public evaluation. The signed prospective plan is
`autojev-kl06-student-prereg-2026-09-28.md` and unified gist revision
`3578611`. The student initialized from the pinned official Qwen3 0.6B Base,
never from a third-party decision checkpoint.

## Execution and integrity

The exact local code was mirrored to the authorized runtime and the private
TRAIN/teacher artifact, official Qwen source and SELECT/CAL partitions passed
their pinned hashes and CPU source/option/token checks. A single GPU was used
for zero-step, one-step/reload, and the sole full 466-step run. The runtime
image digest was
`sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.

- Zero step: all **700** native SELECT requests matched the archived control
  exactly: **0** category changes and **0** maximum probability drift. Private
  comparison receipt SHA-256
  `316295d6eaad64159bba31b8fe29d0481eaec10226d232c9e542d9b939def3d3`.
- One step: finite CE/Brier/KL and gradients; **9/16** original TRAIN examples
  in the first update carried eligible teacher distributions. Reloading the
  checkpoint reproduced the first **32** native SELECT predictions exactly:
  **0** category changes and **0** maximum probability drift.
- Full run: exactly **466** updates and **4,094,489** original TRAIN token
  exposures. All recorded losses and gradients were finite; **3,690** eligible
  teacher-vector exposures, equal to the frozen eligible roster, and no Score
  soft-target exposure. CAL remained untouched. Final and BEST are the same
  step **466**, selected by the existing family-macro rule.
- Lease from initial GPU reservation through full-run completion was
  **1,185 seconds = 0.329167 GPU-hours**, within the prospective **0.5** cap.
  The device and task container were verified idle/absent after completion;
  the reservation was released. No retry or schedule change occurred.

## Fixed SELECT comparison

| Arm | Correct / 700 | Family macro accuracy | Family macro Brier |
|---|---:|---:|---:|
| Archived hard-label control, same start/data/schedule | **562** | **0.772593** | **0.142634** |
| Teacher KL 0.05, BEST/final step 466 | 510 | 0.594544 | 0.240292 |
| Difference (KL minus control) | **-52** | **-0.178048** | **+0.097658** |

The prospective gate required at least 562 correct **and** family macro at
least 0.7725925925925926 at the fixed-rule BEST checkpoint. This arm fails
both. The Brier increase is an additional probability-quality regression.

| SELECT family | Control correct | KL correct | Change |
|---|---:|---:|---:|
| Human Choice (200) | 161 | 148 | -13 |
| Human Noul (200) | 175 | 179 | +4 |
| Pilot narrative (130) | 130 | 123 | -7 |
| Pilot open-world abstention (40) | 40 | 16 | -24 |
| Pilot string composition (40) | 24 | 7 | -17 |
| Score quantized median (90) | 32 | 37 | +5 |

All frozen milestone results are retained; they were not searched to change
the rule. `(step, correct, macro)`:
`(64,243,.321425)`, `(128,336,.391375)`, `(192,377,.468632)`,
`(256,425,.513376)`, `(320,485,.580570)`, `(384,486,.551959)`,
`(448,471,.522749)`, `(466,510,.594544)`.

The large open-world and string-composition losses are descriptive failure
slices. Their mechanism is not established by this single arm. Teacher
agreement on TRAIN and a modest global KL coefficient did not protect the
student's held-out native decisions. A future test should first use a
prospectively defined source-specific gradient/conflict and loss audit before
spending another full-control-matched GPU arm; this experiment provides no
support for replacing the official-Qwen hard-label control.

## Private artifact checksums

No raw prediction, label, teacher vector or private path is in this report.
Private outputs remain mode `0600` in the isolated experiment directory.

| Artifact | SHA-256 |
|---|---|
| Full-run provenance | `b4d9bcaf314e0c5be0317fa2e74d1a0840b44efefd9628732942648852fcdfc4` |
| Full-run completion receipt | `6ddda3f692c0b22d8ad9d5705b91fd9a067510b98e08ed4d999c3197aeea7d6c` |
| BEST pointer | `e96277b9e0b07b7ef8c8df4b2fcbb71b7bdb9e8b4b8ed58816d8381dc221e3b6` |
| Final SELECT metrics | `9d561e1f24e302940ca50f8d283cbdad17a9ccc87bfe168d8cdff24675b1ff27` |
| Final SELECT predictions, private | `e8c899614b9a9816fc6f017265aaa9516293a4b3075845670cd226f8f4c19d24` |
| Full train metrics | `375702de1046725c46ec27d2e52d703e0cb721da22f85882e6bc9d696af4b4ae` |
| Final checkpoint backbone weights, private | `b682016567b686c4ad33978cefe041fa65485954cff099d6a9e9e60cda7ccbcb` |
