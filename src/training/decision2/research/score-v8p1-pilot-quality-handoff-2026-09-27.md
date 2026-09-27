# Score v8.1: CPU quality pilot and blind review handoff

**Status: PENDING_INDEPENDENT_BLIND_REVIEW.** No candidate has been trained from
these rows. The previous v8 candidate remains at HOLD_SHORTCUT and is not
silently reused. The v8.1 design was signed in `50c36d99d` before generation;
the generator and audit are signed in `66ccb0494`.

| Artifact or check | v8.1 result |
| --- | --- |
| TRAIN | 15 complete groups / 45 rows, 15 per Score level |
| SELECT | 10 complete groups / 30 rows, 10 per Score level |
| Mechanisms | Dated update, numeric limits, evidence sufficiency, scoped exception, long memo: five groups each across both roles |
| Rendered oracle | 75 / 75 matched structured oracle |
| Parent TRAIN/SELECT/CAL identities | Exact pinned SHA-256 values verified |
| Protected inputs | 28 gold-free role inventories, identity checked |
| Prior corpus | v7p TRAIN and gold-free blind SELECT checked; answer key unopened |
| Exact/near overlap findings | Zero flagged comparisons against parent, prior, protected, or cross-role prompts |
| Global marker shortcuts | None of the frozen inspected terms uniquely identified a level |
| v8 candidate reuse | Zero exact input hashes or group IDs shared with v8 |
| Long memo length | 2,203 TRAIN and 2,210 SELECT characters per item |

The private v8.1 manifest SHA-256 is
`a68fa38bc443532df3402b3ec38ab5fa08ec71980732b56b3d2b53566a8dbc54`;
the private audit receipt SHA-256 is
`6c9ba1918de7c1c07bc832a4b9508b9e0d66e5df806888d668ded1a51eb79762`.
The separate gold-free blind-packet SHA-256 values are
`0052e359b517e249b7ff5da96887b7fd6b62b7a72c7a60b6a6789931419aab8d`
(TRAIN) and
`8ce1e30879337a7470547f6ba641b78d3cf898918b642e96dda97635d44ac5cc`
(SELECT). Their source labels and joins remain in separate private files. The
TRAIN blind answer slots for levels 0/1/2 were, respectively, 4/7/4, 7/4/4,
and 4/4/7 across positions 1/2/3; SELECT slots were 4/2/4, 2/6/2, and
4/2/4. No level is bound to a fixed slot.

These mechanical results do not certify realistic wording, genuine source
necessity, semantic independence, or transfer. All cases are generated from
five compact template families with opaque IDs and synthetic day numbers; the
long memo includes repeated background prose. A separate reviewer must blind
solve all 75 prompts, document ambiguity and realism for each of the 25
groups, and seal findings before the keys are compared. A negative review
remains a HOLD. Even a positive review would only permit a new, larger data
preregistration and matched training-control proposal, not direct release or
formal-set evaluation. No GPU-hours were used for v8 or v8.1.
