# MASSIVE v4 localized-option pilot: independent blind review

**Decision: BLOCK_FOR_TRAINING.** This preregistered TRAIN-only pilot has twelve
source groups and six localized question-option pairs per group (72 rows). I
judged every row and all twelve complete six-locale groups from the verified
blind packet, sealed the private verdict, and received the separate key only
after the sealed SHA-256 was independently confirmed. The candidate remains
`training_approved=false`. No training or FINAL evaluation was performed.

## Receipts and result

| Receipt | SHA-256 |
| --- | --- |
| Blind packet | `081e76098eeab5da5875cdf9ec1b9faf92f3895bea90775c46986cb1ca54cb29` |
| Sealed independent blind verdict | `bb5192496a2734b32daa1be53cb90bd9ae85195f51e4f97b10db03783dccbfb6` |
| Separate answer key | `276e8de84b9cb5fe9326c0432b8e74d9aca51878bcb6a3a8d73b7d46d2d950b6` |
| Private post-key comparison | `cc268431adbb9004291039bbf859644b2e785208f2082123c6f4b198c66885a7` |

Inferred action choices match the key in **72/72 rows** and all six rows of
**12/12 groups**. That agreement does not establish semantic fidelity. Under
the blind editorial rubric, **58/72** rows have a clear precise option fit;
**12** have only a closest option because the request and description differ in
scope or action, and **2** have an additional ambiguity. Full parallel meaning
is preserved in **51/72** rows, with **6** minor shifts and **15** material
drifts. Wording is natural in **56/72** rows, has minor issues in **8**, and is
awkward in **8**. The criteria overlap: **32/72** rows pass precise option fit,
full meaning preservation, and natural wording together.

**No complete six-locale group passes the preregistered gate.** Four groups
have material failures: one reverses an arithmetic operation, one changes the
currency pair in three locales, and two change the named contact involved in
an action. The other eight groups also fail the gate through option
scope, omitted-detail, ambiguity, or naturalness findings. These include
repeat or requested-time details lost in translation, uncertainty between
speech manner and volume, and wording that can imply a different device
action. A matching intent label therefore cannot make these parallel requests
interchangeable.

The pilot is a fixed, small feasibility sample, so these counts do not estimate
full-corpus prevalence. Per the preregistration, quarantine the complete
failed groups; do not relabel or refill this pilot. Any redesigned candidate
needs a newly frozen packet and independent blind six-locale review before a
larger audit. The v2/v3/v4 candidates remain unapproved for multilingual
training. No GPU optimizer step, held-out TEST, or FINAL access follows from
this review.
