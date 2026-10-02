# IB4 prereg amendment 4 (2026-10-02): `atom` narrowed after the author spot-check; publication bar fixed now

Committed and pushed **before the rebuilt run's first row is converted**. Run `p2` (`3578ff58`) passed every audit
(G0 / G0u with controls, G1, C1 names, overlap, G4, G5, G7, final rescan, leak guard, IX1 with the planted control
200 / 200). It stays on node A as `p2-a3`, **unpublished**.

**Why.** The author spot-check of 30 `atom` TRAIN rows found the cross-event "no" inferences too often plausible in the
generic dimensions:

- wants (xWant / oWant) and effects (xEffect / oEffect), e.g. "yells at X" as a negative for breaking someone's phone,
  or "stop and rest" for starting to move;
- an estimated 10–15% of labels wrong, well above the ≤ 5% bar the IB blind reviews used.

The intent (xIntent) and precondition (xNeed) negatives in the same sample were implausible, as intended.

**Change.**

1. `atom` uses only xIntent ("Why PersonX does this") and xNeed ("What PersonX needed to do beforehand").
2. An eligible inference has at least 3 words (was 2). Every other rule of amendment 3 stands.
3. The rebuilt family runs every audit from scratch in a fresh `p2`.

**Publication bar (fixed now, before any rebuilt row is read).** After the audits, the author reads the first 40 TRAIN
rows in `ib4-spot-v1` hash order. Phase 2 is published only if **at most 2 of 40** labels are clearly wrong (a "no"
inference that is a plausible intent or precondition of the event, or a "yes" inference that is not). Otherwise phase 2
publishes nothing and `atom` is reported as dropped. This is an author check, not a blind review, and is disclosed as
such.
