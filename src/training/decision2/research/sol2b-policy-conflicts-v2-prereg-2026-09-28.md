# Sol 2B policy-conflict v2: prospective source-correction screen

**Frozen before v2 generation or diagnostic. CPU data admission only.** The
[v1 source](sol2b-policy-conflicts-v1-cpu-hold-2026-09-28.md) remains HOLD:
its Noul state-removed shortcut scored 67.19% against a 50% fixed majority
baseline, above the registered +5-point ceiling. The v1 generator, packet,
hashes, and failed result must remain unchanged. This v2 is one correction
of a known sampling flaw, not a license to search seeds or thresholds.

## Precommitted change and causal hypothesis

The v1 code first chose a desired Noul truth value and then picked from
matching `(subject, question mode)` combinations. The `full` question mode
was false 165/238 times, while `eligible` was true 183/274 times, so the
question alone predicted the answer. The v2 code will instead set three
independent bits **before** sampling facts, for every family and group index
0–127:

- Choice correct-option position: `index & 1`.
- Noul truth: `(index >> 1) & 1`.
- Noul query mode (`full` or `eligible`): `(index >> 2) & 1`.
- Noul subject side (`left` or `right`): `(index >> 3) & 1`.
- Score tier: `index % 3`.

The existing executable policy oracle and decision-tier meanings remain
fixed. Draw a fresh structured two-candidate situation under the v2 seed
`decision2-sol2b-policy-conflicts-v2`; accept it only if its preselected
subject has the preselected Noul truth under the preselected mode, the two
candidate tiers differ, and one candidate has the preselected Score tier.
Render the same three native request types and option descriptions. Do not
choose the mode or subject after seeing the truth, and do not change the
questions or facts based on the answer. This creates exactly 32 observations
for every `(mode, truth)` combination per family and 16 for every
`(mode, truth, subject side)` combination per family. The deterministic
generation cap is 4,096 draws per group; any failure to fill a group stops
the source.

The causal prediction is that the Noul prompt/answer association will be
removed, while state reading is still necessary to answer. V2 may still
fail due to name, lexical, sampling or other shortcuts; only the fixed
diagnostic determines admission. Rewriting v2 after that result requires a
new prospective version, not a second candidate under this registration.

## Frozen outputs and gates

Generate 512 independent situations, 128 per eligibility, permission,
scheduling and inventory, each with complete native Choice, Noul and Score
rows (1,536 rows). Preserve the same policy oracle, ordered-rule cross-check,
native row contract, 48-group answer-blind packet (12/family), and all
[v1 admission gates](sol2b-formal-gap-next-arm-2026-09-28.md): balanced
Choice/Noul/Score token shares within ±5 points of one third; 20–25% of the
archived 4,194,465 native-token control as new-source exposure; 7,455 total
rows and total tokens within ±1%; retain every one of the 2,800 human-label
rows and 516 old Score rows, replacing only whole eligible nonhuman,
non-Score groups. Require 44/48 independently reviewed groups unambiguous
across all three outputs. The blind packet's structural validity is not a
substitute for independent review.

Generate exactly one group-disjoint, state-removed diagnostic with seed
`decision2-sol2b-policy-conflicts-v2-diagnostic`, 32 groups per family
(128 groups total). Use the already committed v1 four-fold group-disjoint
bag-of-words probe, which sees only instructions and option descriptions,
and compare each native type with its **fixed global majority baseline**.
Every type must score no more than 5 percentage points above that baseline.
Do not tune the generator, seed, folds, features or threshold after viewing
this result. A failure of any type places this v2 source on HOLD; no training,
calibration, formal scoring, exact replacement roster or HF upload follows.

Before any future GPU use, separately verify v2 oracle consistency; zero
exact/near cross-split groups and reviewed suspicious semantic overlaps
against original TRAIN/SELECT/CAL, v1, typed DEV/FINAL gold-free prompts,
CSS pilot/15-task gold-free prompts and public231; source rights; exact
group-atomic replacement at 7,455 slots and ±1% tokens; independent blind
review; and the original Sol zero-step/source parity plus one-update smoke.
Preliminary overlap and token-capacity checks may run on CPU even if the
shortcut fails, but cannot convert a failed source into an admitted one.
No formal answer labels or GPU inference are needed for this screen.
