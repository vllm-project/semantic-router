# Sol 2B policy-conflict v2: prospective CPU screen result

**HOLD.** V2 corrected the known v1 Noul question-mode/answer association,
but the same precommitted state-removed diagnostic exceeded the Choice
shortcut ceiling. No GPU training, model selection, calibration, formal
scoring, exact replacement roster, or HF upload followed. The failed
[v1 receipt](sol2b-policy-conflicts-v1-cpu-hold-2026-09-28.md) and generator
remain unchanged. The [v2 preregistration](sol2b-policy-conflicts-v2-prereg-2026-09-28.md)
was signed before generating this candidate.

V2 fixed the Noul question mode, target truth and subject side as independent
index bits before sampling structured facts. In all 512 groups, there are
exactly 16 observations for each `(family, mode, truth, subject side)` cell.
The program oracle and independent ordered-rule interpretation agree for all
generated cases. The full private packet contains 512 distinct structured
situations, 128 per domain family, each with native Choice, Noul and Score
rows. The 1,536 labels are Choice A/B `256/256`, Noul false/true `256/256`,
and Score tiers 0/1/2 `172/172/168`. The 48-group answer-blind review packet
has 12 groups per family and passes the *structural* answer-exclusion check;
it has not received the required independent semantic verdict.

The frozen 128-group diagnostic used four group-disjoint folds and a
bag-of-words classifier on the instruction and option descriptions only;
policy state was withheld. Each type had a fixed global majority baseline,
and the preregistered maximum was baseline +5 percentage points.

| Type | Fixed majority | State-removed accuracy | Difference | Gate |
| --- | ---: | ---: | ---: | --- |
| Choice | 50.00% | **57.81%** | **+7.81 points** | **Fail** |
| Noul | 50.00% | 45.31% | −4.69 points | Pass |
| Score | 34.38% | 34.38% | 0 points | Pass |

Noul's v1 leak no longer appears under this diagnostic, but the Choice
result means v2 still has a state-independent signal or finite-sample
instability too large for the fixed admission rule. The exact causal source
of that Choice result is unresolved; no name, order, seed or threshold was
changed after seeing it. A third candidate would need a separate prospective
registration and fresh diagnostic.

## Token and overlap preliminaries

The exact archived control TRAIN hash was verified. Native segmented
tokenization used the original Sol tokenizer and no truncation. V2 has
1,004,069 source tokens, **23.938%** of the 4,194,465-token control; the
Choice/Noul/Score token shares are **32.95% / 32.86% / 34.19%**, each within
the frozen ±5-point balance rule. Maximum native input length is 677 tokens.
The original 2,800 human-labeled and 516 Score rows remain protected by the
replacement-capacity calculation. There are 3,551 whole replaceable groups,
4,139 rows and 3,583,360 tokens. This is sufficient *capacity* for a
1,536-row source but is not proof of an exact 7,455-row, ±1%-token roster;
none was selected after the shortcut failure.

Preliminary exact input-hash, normalized full-input, normalized state,
row-ID and structured-fact-digest intersections are all zero against the
v1 candidate, original TRAIN7,455, SELECT700 and CAL700. These are only
preliminary checks; near/semantic overlap and gold-free formal/public prompt
checks have not been performed. Zero preliminary intersections do not admit
the source.

The first CPU audit invocation produced no report because the overlap checker
assumed every control `state` was a string; some control states are
structured objects. An audit-only serialization fix was signed and the
**same frozen candidate, seed, folds and gates** were rerun successfully.
There was no candidate search or changed generator. The first invocation's
empty report was not interpreted as a pass. Five focused v2 tests and the
repository `make check` passed before the complete audit.

Private candidate SHA-256:
`e3b4e3a14af34088ee2c3f2cf064ea30ff0271cb8fdb292da66b94e7fa0f5ead`.
Private blind packet SHA-256:
`7c8b0ff19843225e3a9c0d3fce5c9527e6ef4579362ce4fb706e4184f6812051`.
Complete private CPU audit SHA-256:
`46f83b64377170b8290b7f763f94c6c0501cdc112819650e89fdf2f8844ab4f5`.
No private raw rows, infrastructure locations or protected answers are in
this record.

Independent blind review, source rights, near/semantic and gold-free prompt
overlap, exact replacement roster and original Sol zero-step parity remain
unperformed. They are not waived by the token and schema passes. This v2
candidate must not be treated as an admitted training source.
