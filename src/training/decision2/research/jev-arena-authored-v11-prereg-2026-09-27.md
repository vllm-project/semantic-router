# JevArena authored v11: prospective DEV editorial repair

Status: **PREREGISTERED DESIGN ONLY**. This note is frozen before any v11
source text, gold, reviewer packet or model run. v9 and v10 remain failed
development evidence. v11 is limited to at most 12 newly authored DEV
originals, four Choice, four Noul and four Score, with no training, FINAL
access, release score or Hugging Face publication.

## Failure addressed

v9 leaked an approval through its lead and had all-true Boolean labels. v10
balanced answers and passed literal source-deletion proofs, but a generic
structured-DATA envelope plus long provenance boilerplate made the text
artificial. Its unchanged introductions referred to omitted records, six
deletions left answer or status cues, and it included sources that did not
change the original answer. These are editorial failures even where an
automated world-completion check passes.

## Prospective source and wording method

1. Each case gets a short, **source-neutral task preamble** stating only the
   decision sought and the current rule. It must not assert a source count,
   roster, document presence, signature, authorization, outcome, or likely
   branch. The exact same preamble remains coherent after any source deletion.
   No source text may restate a fact certified only by another source.
2. Every model-visible source carries a distinct, necessary fact or bundle of
   facts. There are **no decorative or wrong-scope sources** in this pilot.
   Each source must independently change what can be proved: after removing
   it, at least two admissible completions consistent with all remaining
   evidence give different answers, one differing from the original. No
   omitted field may be reconstructed from the preamble or surviving prose.
   A mechanically necessary source that a blind reader can infer from context
   still fails editorial review.
3. Present evidence in varied natural formats: at least four formats across
   the pilot, including excerpts of correspondence, operational logs, meeting
   minutes, and compact tabular or form entries. Format variation must carry
   meaningful evidence, not header decoration. Source labels and item IDs are
   private-salted opaque strings. Source prose may explain provenance, but
   repeated generic disclaimers about what a source does *not* prove are
   disallowed.
4. Use twelve distinct real-world domains and twelve distinct semantic
   operations. Planned operation families are temporal rule precedence,
   interval compatibility, constrained set coverage, and minimax choice;
   exception hierarchy, dependency chain, quantified eligibility, and
   state transition; ordinal milestone grading, bounded weighted penalties,
   evidence consistency, and rate-to-rubric mapping. Do not transplant v9/v10
   characters, fact packs, or formulas. Balance positive/negative and low/high
   cases through the task logic, not by superficial label edits.
5. Of the 12 originals, target three compact cases, six medium cases, and
   three long cases. A long case should have roughly 500–900 useful words;
   lower length is acceptable if additional words would be filler. Every
   paragraph of a long source must affect interpretation, scope, temporal
   precedence, or evidence quality; author-side review removes repetitive
   restatement. Across cases, repeated eight-word prose sequences outside
   rule syntax must be zero, and paired source-text trigram overlap above
   0.15 requires manual rewrite or quarantine. A quality gate can reduce the
   accepted count below 12; it cannot trigger filler generation.

## Gold isolation, validation and stop rule

The local signed code is the source of truth; any execution to construct
packets happens on an authorized experiment host with CPU only. A single
private 256-bit salt is generated once, committed by hash, never searched for
favorable answers or positions. The predeclared 4/4/4 type quota, two true and
two false Boolean answers, at least three Score levels, and four distinct
Choice displayed positions are hard acceptance gates. Choice positions may
be balanced by a predeclared private schedule after semantic answers are fixed;
case order and IDs remain salted. Option wording itself must not signal gold.
If a quota or source necessity check fails, preserve that attempted revision
and its failure receipt. No rejected packet is overwritten.

The author-side program uses two independent operation oracles and validates
unique extraction spans or equivalent form cells against a separate private
fact map. For every original it writes a private gold/proof file; for every
source it writes a **gold-free** deletion variant and a private two-completion
witness. The author then inspects the full deletion packet for surviving
answer cues and document-reference contradictions before freeze. Hashes bind
the source commit, author specification, salt commitment, prompts, deletions,
private targets, proof traces, and reviewer manifest.

After the hashes are frozen, an independent reviewer sees original prompts
without targets or deletion variants, records answers and editorial concerns,
and seals that report. Only then does the reviewer see gold-free deletions,
record whether original answers remain inferable from natural prose, and seal
a second report. No target/proof access until both seals. A post-key operator
then checks seal and file hashes, time order, joins and aggregate agreement.

**Stop condition:** Any direct or plausible surviving answer/status cue,
source whose removal leaves the original answer provable, confusing evidence
format, repeated filler, ambiguous original, or failure of answer balance
blocks v11 entirely from the release-authored set. Do not repair that same
sealed packet after seeing blind judgments. Record the negative result and
prospectively design a later version if needed. Passing the 12-case pilot is
only permission to consider a larger independently authored release pool;
it is not a JevArena score or a claim of model improvement.
