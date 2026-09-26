# Score TRAIN v4: prospective design only

**Status: method preregistration; no v4 examples, packet, GPU step or score
exist.** V3 remains `BLOCK_FOR_TRAINING` after independent sealed review
identified accepted-core-count, adjacent-pair-count and one-signal shortcuts.
This design is fixed before any new v4 data are generated. V1–v3 and their
reviews stay immutable; do not relabel or silently reuse their packets.

## Candidate scale, isolation and unchanged controls

Target 81 independent groups per mechanism, each with one row per level
0/1/2: four equal mechanisms, 324 groups and 972 TRAIN-only rows before
quarantine. The multiple of three permits counterbalanced signal assignments.
Target 729 English and 243 Chinese rows, balanced within each level. Retain
the v3 equal-link/degree directed-route mechanism as a control, but use a new
seed, cases, source IDs and renderer. Keep the fixed rights-clean-v2 parent
TRAIN/SELECT/CAL partitions, explicit source rights, pinned tokenizer, 1,024
token cap, and group-complete exact/approximate quarantine against those
partitions plus **all** gold-free DEV and authored rosters frozen before the
build. Never open protected labels/proofs. Require at least 72 surviving groups
per family; any overlapping group is removed whole. Recheck same-level
cross-group near clones, contradictory labels, unambiguous cutoffs, and
independent oracles. No benchmark template or held-out answer is a source.

## Three mechanism repairs

**Obligation review:** Keep two required core controls and at least one
informational distractor. For each core control, the raw event-status
multiset, event count, timestamp multiset and control name are identical
across the three variants; only the association of status to timestamp
changes. A separate oracle sorts events by timestamp, keeps the latest
applicable event per core control, ignores informational events, then applies
rejection precedence, unresolved next and all accepted last. Construct the
two current core states by level as `(rejected, accepted)`, `(accepted,
unresolved)`, `(accepted, accepted)`, rotating which named control carries
each non-accepted state across groups. Thus all model-visible raw status and
record counts are constant, while neither single core control strictly ranks
the three labels. Randomize display order independently of timestamps and
verify no raw accepted/rejected/unresolved count, event count, scope count,
name, or first/last displayed record predicts level above chance within a
group. A derived *latest applicable status* may differ: computing it is the
intended decision operation, not a forbidden raw-count shortcut.

**Timely streak:** Change the explicit v4 rule to levels `0` for longest run
at most two days, `1` for exactly three, and `2` for at least four. Use days
1–12 with exactly six on-time days, six late days and exactly three on-time
runs in *every* variant. Require both endpoint days late and hold the late-run
length multiset at `(1,1,1,3)`, permuting its positions independently of level.
The on-time run-length multisets `(2,2,2)`,
`(3,2,1)` and `(4,1,1)` give the three levels while all variants have exactly
three adjacent on-time pairs: six on-time days minus three runs. Event count,
on-time count, late count, on-time-run count, adjacent on-time pair count and
transition count must be identical within each triplet. Shuffle record
display order and independently recompute the longest chronological run;
do not use an unordered-pair count as oracle. If a group cannot meet these
constraints, reject and reseed it rather than weakening a gate.

**Weighted points:** Retain fixed signal names/order, fixed name–weight
assignment, fixed marks multiset, cutoffs, rule/options and group context
across the three labels. Keep the v3 weight pools `(1,2,3,4,5)`,
`(1,2,2,4,5)`, `(1,2,3,3,5)` and mark pools `(0,1,2,3,4)`,
`(0,1,1,3,4)`, `(0,1,2,2,4)`, `(0,1,2,3,5)`, `(0,0,2,3,5)`.
As in v3, anchor one maximum mark to one maximum-weight signal, enumerate
attainable totals, and use the sorted unique totals at indices `floor(n/3)`
and `floor(2n/3)` as fixed lower/upper cutoffs. Enumerate mark pairings and
choose one per weighted
score tier only if **no individual signal position has three distinct marks
across the triplet** or a strictly increasing/decreasing mark sequence by
level. Keep the strongest individual product fixed; reject/reseed groups
that cannot meet all hard constraints. Counterbalance the remaining binary
mark changes across label and position/weight buckets over complete
three-group cycles; conditional mark histograms per position and weight
bucket must match across labels, with at most one count of discrepancy only
for an unavoidable remainder. A single signal cannot fully rank a triplet,
and no named/positioned signal should become a corpus-level label proxy.
It is mathematically impossible to keep every signal's own mark constant
within a triplet while changing a fixed-weight total; the hard invariant is
the *group marks multiset*, and per-signal label marginals are balanced over
groups. The independent oracle must recompute all products, sum and tiers.

## Frozen gates before any review or training

Run a preregistered shallow-feature audit for each mechanism: raw counts,
single field, field position, name, language, first/last displayed record,
max/sum statistics, and group-held-out decision stumps. The obligation raw
counts and streak event/adjacent counts must classify exactly 1/3 of rows
within complete triplets. Weighted single-signal three-way perfect-ranking
groups must be **zero**, versus 24/80 in v3; group-held-out single-signal
features must not exceed 40% three-way accuracy. Keep per-family reports,
not only an aggregate. A failed shortcut, answerability, rights, tokenizer
or overlap gate rejects the entire v4 version; register a new method before
changing it. Passing these limited features does not certify reasoning.

Only after mechanical QA may a fresh 12-complete-groups-per-family packet be
sealed with new HMAC-opaque row and group IDs and a new private salt. The
editor receives only gold-free packet and manifest, solves every row and
checks whole triplets, multilingual wording, order effects and any new
shortcut before sealing judgments. The author validates packet/review/seal
hashes and ordering *before* opening the private join; any material blind
finding keeps v4 blocked. Even a clean editorial verdict does not authorize
GPU training: first create a separately frozen, group-disjoint three-level
SELECT diagnostic and get the project lead's matched-control pilot decision.
