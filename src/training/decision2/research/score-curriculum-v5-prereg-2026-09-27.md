# Score TRAIN v5: prospective conditional weighted-score design

**Status: method only; zero v5 rows, no packet, no GPU or HF action.** This
registration follows the frozen v4 negative feasibility receipt. V4 is
`HOLD_NO_CORPUS` after 112/243 weighted rows were classifiable from one signal
against its maximum of 97/243. V1–v3 remain blocked. A future v5 failure will
be retained rather than repaired under this version number.

## Scope and source isolation

Keep four equally represented TRAIN-only mechanisms and target 81 independent
groups per mechanism, one related row at each ordinal level: 324 groups and
972 rows before whole-group quarantine. Target 729 English and 243 Chinese
rows, with each language balanced by level and mechanism. Carry forward the
v4 obligation, streak and equal-degree route *methods* as controls, but build
with a fresh seed, IDs, context and rendering. Re-run their independent
oracles and all shallow-feature gates; no v4 corpus or packet exists to copy.
Pin the rights-clean parent TRAIN/SELECT/CAL partitions, the maximum native
1,024-token input, tokenizer revision, source rights, and every gold-free
protected roster available before construction. Quarantine an entire source
group on exact or approximate context overlap. Never read protected labels,
proofs or FINAL. Require at least 72 admitted groups per family; report every
source, hash and exclusion.

## Weighted candidate-table intervention

Each weighted source group contains **one fixed table of three candidate
plans**, with five named signals per plan. The three candidate profiles use
the same group-wide signal names, name-to-weight assignment, mark multiset,
unweighted mark sum, strongest-weight mark and maximum individual product.
Keep the v3/v4 weight pools `(1,2,3,4,5)`, `(1,2,2,4,5)`,
`(1,2,3,3,5)` and mark pools `(0,1,2,3,4)`, `(0,1,1,3,4)`,
`(0,1,2,2,4)`, `(0,1,2,3,5)`, `(0,0,2,3,5)`. Only the pairing of marks to
non-anchor weights differs among plans. Each plan has a distinct weighted
total. Choose a triplet with the **smallest attainable total span** that
still meets the three distinct ordinal bands and the single-signal conditions
below; use a fixed seeded tie-break. Set group-wide integer cutoffs at the
middle and high selected totals, so selected totals fall in `< lower`,
`[lower, upper)`, and `>= upper`. Reject and reseed if this is impossible.

Show the complete three-plan table, its weights, marks, cutoffs and option
wording **identically in all three rows of the group**. Vary only which
candidate ID a separate request points to. The answer is the weighted band
of the selected candidate. Random opaque candidate IDs and candidate display
order are assigned independently per group. Exactly counterbalance the
mapping from target display position to level within English and Chinese:
for each language, every target position occurs equally often at levels
0/1/2. The table is deliberately repeated within a group to make raw
counts, single displayed card fields and table-level statistics invariant;
the intended operation is to bind the request to a candidate and compute
its weighted score. The different selected cutoffs are a **new v5 task rule**,
not a retroactive v4 change.

## Prospective feasibility and quality gates

Before emitting text, run a deterministic abstract pairing feasibility
search over the pinned pools and selection rule. It must report the number
of eligible triplets by pool and score span. No label, model or DEV metric
enters this search. If the search cannot supply the planned group quotas,
stop v5 without emitting a corpus; register a new method rather than relaxing
the criteria.

For every complete three-row group, enforce and independently verify:

1. The candidate table, weight/mark multisets, unweighted sum, cutoffs,
   strongest product, rule text and option descriptions are byte-identical.
   The requested candidate reference is the sole semantic edit.
2. An independent oracle resolves the reference, recomputes all five
   products and the sum, then applies the two inclusive cutoffs. All levels
   0/1/2 must occur, with no ambiguous boundary.
3. No one signal of the **selected** plan has three distinct marks or
   strictly ranks all three levels within a group. Keep a local feature
   inventory of single mark, weight, product, selected plan position and
   first/last displayed card. Flag a predeclared global one-field rule or
   group-heldout fitted one-field rule that solves a whole triplet; never
   fit a position-to-label map on that triplet's own gold. A fixed
   table-only feature must score exactly 1/3.
4. On group-heldout prediction over all weighted rows, the best one selected
   signal `(position, weight, mark)` lookup must be **at most 40%**. Also test
   one selected product, maximum selected product, selected unweighted sum,
   target display position and shallow decision stumps on each single field;
   each must be at most 40%. Fixed-table features and target position alone
   must score exactly chance under the balanced schedule. Report both overall
   and per-language outcomes. A failed gate blocks v5; do not retune within
   v5 after seeing it.

The algebra motivating this design is explicit. Fixed-weight additive scores
cannot have identical selected `(weight, mark)` marginals across all three
ordered levels: identical marginals imply equal mean totals. V5 therefore
does **not** claim that every selected signal is statistically independent of
the answer. It makes *unselected table information* exactly invariant, makes
the target reference marginal exactly balanced, chooses nearby distinct
totals to reduce the contribution of any selected signal, and retains the
strict empirical 40% selected-signal gate. Passing these checks will not by
itself prove deep reasoning or transfer.

After abstract feasibility, build a fresh candidate and run tokenizer, rights,
group-complete source isolation, protected prompt overlap and same-level
cross-group near-clone checks. Only a fully passing candidate may produce a
new HMAC-opaque gold-free packet, with a new private salt and private join.
An independent reviewer must seal a full-triplet solvability and shortcut
review before private post-key comparison. A clean blind review is necessary
but still does not authorize training; a separate group-disjoint SELECT
diagnostic and project-lead matched-control decision are required.
