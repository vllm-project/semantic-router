# Kev 4B TRAIN-only teacher signal: separate rounded-output screen

**Prospective v2, zero new teacher inferences at registration.** The first
96-group screen stopped at its distribution validator before producing an
aggregate result. This is a new screen with a disjoint TRAIN-group roster and
a publisher-compatible numeric rule; it does not reinterpret or retry the
failed v1 receipt. Kev remains an external teacher or peer, never a direct
Decision 2.0 weight initializer. A teacher signal result cannot establish any
2.0 student quality.

| Frozen component | v2 commitment |
| --- | --- |
| TRAIN | Rights-clean v2, 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| Teacher | `jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101`; release source `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63` and exact published native Checkpoint/API path |
| Roster | 96 independent TRAIN groups, 32 Choice/32 Noul/32 Score; exclude all v1 groups, then sort candidates by SHA-256 of `kev-v2`, type, original group and row ID. Freeze resulting roster digest from CPU dry-run before GPU reservation. |
| Code | [`kev_teacher_train_pilot_v2.py`](kev_teacher_train_pilot_v2.py) SHA-256 `f6d5d44d37114a901c94d12b6b1ee9de549778a5e443a7a7661df7e72ffb5ca9`; five synthetic contract tests passed before registration. |
| Numeric rule | Use the published four-decimal Choice/Score option probabilities. Require exact option keys, finite values in [0,1], positive sum and `abs(sum-1) ≤ number_of_options × 0.00005 + 1e-8`; normalize accepted values to sum one for scoring. Noul uses its native yes probability and exact complement. Tied argmax counts as wrong. Malformed output stops the run, context overflow counts invalid. |
| Report | Aggregate full-denominator validity, ties and TRAIN hard-label agreement per type; mean gold probability and half multiclass Brier. No source rows or row-level predictions enter a public record. |
| Resource | One exclusive GPU, ≤15 minutes total. Stop on unexpected exception; no altered model, roster, threshold, prompt or same-roster retry. |

Before GPU use, CPU dry-run must attest the new roster digest and disjointness
from v1. Revalidate exact downloaded weights and publisher source hashes,
official base cache, TRAIN and script bytes, runtime image and idle GPU
reservation. Mirror signed local code exactly. Keep any private receipt mode
0600; no protected evaluation answer is read.

The **future** three-type distillation eligibility screen remains all 96
native-valid and at least 22/32 Choice, 26/32 Noul, 20/32 Score hard-label
matches, with Score half Brier ≤0.25. These thresholds are frozen before the
v2 teacher readout; a failure authorizes no three-type student. A passed
screen only permits designing a separately preregistered, source-reviewed
student/control comparison. The observed Kev v3 peer result 59.226 and
public231 175 do not substitute for a TRAIN signal measurement, and neither
is a Decision 2.0 score.
