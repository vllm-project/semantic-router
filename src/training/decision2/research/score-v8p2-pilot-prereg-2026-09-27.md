# Score v8.2: prospective source-necessity and document repair

**Status: design only, before v8.2 generation.** The v8.1 independent
rendered-text review matched all 75 answers but held the data for a target
one-source shortcut and repeated long-memo filler. Do not reuse its TRAIN or
SELECT items, keys, generated organizations, or selected examples as v8.2.
This pilot tests data quality only. It authorizes no GPU use, model training,
CAL access, or formal JevArena prediction.

Keep five mechanisms, exactly three TRAIN and two SELECT independent triplet
groups per mechanism: 15 TRAIN groups/45 rows and 10 SELECT groups/30 rows,
with one Score level 0, 1, and 2 per group. Use a new private 32-byte seed,
HMAC-separated by `decision2-score-v8.2-multimechanism-pilot/1`, role,
mechanism, and group. Publish its digest only. English is the sole pilot
language; records and organizations must be disjoint across roles. Freeze
source rows and separate gold-free blind packets before any reviewer sees
them. A different agent reviews the rendered packets; the author does not
self-review them or inspect model outputs.

## Repairs fixed before generation

1. **Evidence sufficiency.** Level 2 remains outage/outage for the named
   site and hours. Across each three-item group, level 0 puts `normal` in
   either dispatch or meter, and level 1 puts `unavailable` in the *other*
   source, with the remaining source reporting outage. Alternate this
   orientation across groups and roles. For every group, both the dispatch
   projection and meter projection must be non-injective across 0/1/2: neither
   target source alone may decode all three levels. Put source observations
   in separately attributed notes. A decoy record must never have a
   deterministic answer-complement relationship to the target. A group-level
   source-ablation audit, target/decoy placement check, and global cue count
   audit are mandatory; any failed condition holds the whole pilot.
2. **Long document.** Use five substantively different dossier formats,
   one per group, with distinct organizational setting, document order,
   section headings, and actual signed/unsigned authority evidence. Include
   the old schedule, controlling signed amendment, target delivery notice,
   conflicting unsigned draft, and relevant but noncontrolling other-record
   notices. Every section must help establish authority, scope, chronology,
   confirmation, or a plausible competing interpretation; do not pad with
   interchangeable prose. Minimum rendered state length is 1,500 characters.
   After normalizing record IDs and numbers, cross-group first-variant
   similarity must be below 0.85. This is a small synthetic document pilot,
   not proof of real-world long-context transfer.

The three unaffected mechanism rules remain dated updates, numeric limits,
and scoped exceptions, but their new source IDs and rendered rows are still
new v8.2 outputs under the new seed. The structured oracle is computed first;
an independent rendered-text parser must recover the same answer in all 75
rows. Label balance, answer positions, option wording, prompt length, target
placement, non-target status counts, and one-field projections are audited by
mechanism and level. An answerable pilot can still fail realism.

## Protected separation and review boundary

Audit the exact v8.2 rows against parent TRAIN/SELECT/CAL, v7p, v8, v8.1,
and all available gold-free protected DEV, formal, transfer, authored, and
public prompt inventories using exact input/group/source identity and bounded
near-text checks. Open no protected labels. Any exact or flagged near match,
rendered-oracle discrepancy, source-necessity failure, or deterministic
non-target shortcut is a whole-pilot HOLD. A bounded near-text pass does not
prove semantic independence.

Produce immutable `train` and `select` blind packets with opaque review IDs
and separate keys. A separate reviewer must solve all 75 rendered items and
judge all 25 complete groups for answerability, ambiguity, source necessity,
document realism, and shortcuts, sealing answers and findings before the keys
are opened. No partial group substitution, threshold relaxation, or reroll
after seeing review findings. Even a PASS only enables a new, larger
source-disjoint data and matched-control preregistration; it does not qualify
BEST368 or any 27B release model.
