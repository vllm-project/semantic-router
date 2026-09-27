# Lux 9B native Score probability: batch-shape diagnostic result

**Outcome: HOLD remains.** This was one gold-free diagnostic request, not a
JevArena evaluation or a new release-example reference. It used the published
own Lux 1.0 package, the same qualified image and native collector, and the
unchanged `severity` Score question. The only input change was removal of the
other two questions from that request. The pre-registered intervention and
interpretation are in `lux9b-release-example-batch-shape-prereg-2026-09-28.md`.

| Comparison | Maximum absolute Score probability or expected-score difference |
| --- | ---: |
| Prior exact three-question request vs published example | 0.021284254 |
| Single-question probe vs prior three-question request | 0.000953389 |
| Single-question probe vs published example | 0.020655471 |

The within-run batch-shape effect is below the pre-registered diagnostic
threshold of `0.005`. Dropping the other questions therefore does not explain
the `0.021284254` discrepancy in the actual three-question release example.
The one-question probe also remains above the frozen `0.020` example ceiling.
The actual example was **not** rerun as a substitute release check, and its
threshold was not relaxed. An earlier fresh-process rerun had already
reproduced the same actual-example probabilities exactly, excluding process
reuse as an explanation. Together these tests leave runtime provenance below
the recorded version/profile granularity as the next unresolved source of
numerical variation; they do not establish a particular kernel as the cause.

The one probe completed with a validated native runtime and one valid Score
answer. It used one isolated GPU for 40 wall seconds, **0.011111 GPU-hour**;
the GPU was released afterward. The exact prompt SHA-256 was
`75f9e3d606b1e229b28996cb136b50c758ba7e9a2a8dfc62546d0161ca25d889`;
prediction SHA-256 was
`7794593a003a11c5771d85018c90ecce3e4ad52809577e763156bd191ecd9e6e`.
The prior actual-example prediction SHA-256 was
`923536bbf4d0f0a971e62e4aca4a9ae74828817e0b1ebd8aeeddae5155111a1c`.
Raw prompts, predictions, logs and machine identifiers remain private.

**Next discriminating action, if Lux 9B control becomes necessary:** obtain
the original release-example per-op or kernel selection trace, or reproduce
the original build/runtime image to byte-identical provenance. Compare that
trace against the currently qualified run before proposing any prospective
numerical acceptance rule. No typed FINAL or CSS15 gold-free predictions were
started for this Lux control, and no paired v3 interval against the 2.0
candidate is available.
