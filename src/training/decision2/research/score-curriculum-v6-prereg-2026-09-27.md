# Score TRAIN v6: prospective two-source set reconciliation

**Status: method only. No v6 examples, corpus, packet, GPU run or model
result exists.** This protocol is written after the frozen v4 and v5
weighted-family failures and before any v6 construction. V1–v3 remain
blocked by blind editorial shortcuts; v4 and v5 retain their negative
feasibility receipts. No v4/v5 row is relabeled, repaired or reused.

## Reason for the change

The 27B development audit found a severe low-level bias on Score set
reconciliation. V4 and v5 attempted to add weighted arithmetic examples,
but their fixed group-held-out single-signal gate failed at 112/243 and
115/243, respectively, against a 97/243 limit. Exact independence of every
component is incompatible with a strictly ordered additive total, and the
weaker 40% threshold was empirically unstable under the two frozen
constructions. This does **not** prove all weighted-score data unsafe or the
threshold mathematically impossible. V6 changes the fourth mechanism to a
direct evidence-join task, instead of raising that same threshold after
seeing failed data. The purpose is a new transfer hypothesis, not rescue of
the old weighted candidate.

## Candidate and isolation

Target 81 independent source groups per family and one related row per
level 0, 1 and 2: four families, 324 groups and 972 TRAIN-only rows before
whole-group quarantine. Target 729 English and 243 Chinese rows, balanced
by mechanism and level. Keep fresh implementations of the v4 obligation,
streak and equal-degree route methods as controls, with new seed, entity
universe, context and rendering. Re-run their independent oracles and
mechanism-specific shallow-feature gates; no v4 corpus exists to copy.

The new family is **independent evidence intersection**. Each case defines
four named eligible claims. Two separately maintained sources each attest
exactly two distinct claims; a claim counts only when both sources attest
it. The ordinal answer is the number of jointly attested claims: 0, 1 or
2. Every complete triplet has one row per answer, with the same four-claim
universe, rule, source names, two-entry-per-source count, output vocabulary
and context. The source-pair relationship changes, not the number of
attestations. Across groups, rotate claim identities, which source lists a
given pair, source order and presentation order; no fixed claim or location
may encode a level. Author distinct realistic settings and at least four
evidence formats, with each sentence either defining an applicable fact or
disambiguating scope. The question must specify that only *current active*
attestations count, but do not introduce an unreviewed temporal exception in
the first candidate.

An independent oracle computes the two-source intersection from structured
facts, not from hand-entered target labels. For every row, enumerate
domain-valid alternative contents of **either** single source while holding
the other fixed and prove that the answer can change; each source is
necessary. For every complete triplet, a reader given only one of the two
sources must have at least two feasible answer levels under the unchanged
rule. A source-specific word, one field, one list length or display position
must not solve all three levels within any group. The casebook keeps the
full two-source witness and alternative-completion proof private.

## Frozen diagnostics before any model run

1. Test full-group counts, 0/1/2 balance, both source-swap witnesses,
   alternative completions, independent oracle agreement, explicit
   applicability and native answer format. A single failed group is
   quarantined whole; fewer than 72 admitted groups in any family blocks
   this version.
2. For the evidence-intersection family, report gold-free distributions of
   source lengths, IDs, source order and context styles. Report
   group-held-out accuracy of fixed-position, single-source and one-field
   heuristic baselines, but do not claim their exact independence from the
   label. A heuristic that solves every row of any complete triplet is a
   blocker. A best one-source or one-field classifier above two-thirds on
   all admitted rows is also a blocker; this cap is deliberately different
   from v4/v5 because one partial witness of a two-source join can carry
   legitimate information. The full two-source oracle must solve every
   row. Preserve the actual heuristic accuracies even when below the cap.
3. Pin the rights-clean parent TRAIN/SELECT/CAL, tokenizer revision,
   1,024-token native limit, source rights and all available gold-free DEV,
   authored and earlier curriculum rosters. Run group-complete exact and
   bounded near-overlap screens and same-level cross-group clone checks.
   Never open protected labels or FINAL. Do not export restricted raw text
   to public source or gist.
4. Only a fully passing mechanical candidate may produce a new salted,
   HMAC-opaque gold-free packet. A fresh independent reviewer must solve a
   complete-group sample across all four families, inspect both languages,
   and seal ambiguity, naturalness and shortcut findings before any private
   key comparison. A material finding blocks v6; repairs need v7 rather
   than an in-place revision.

Even a clean v6 review does not by itself authorize training. Use the
separately frozen, source-disjoint three-level Score SELECT protocol and a
run-specific matched-control plan before a GPU optimizer step. Report all
controls, including negative results, and do not use DEV/CSS pilot or public
leaderboard scores for checkpoint selection.
