# JevArena authored v12: prospective DEV editorial method

**Status: PREREGISTERED DESIGN ONLY.** This protocol is frozen before any v12
case prose, fact pack, target, reviewer packet or model run. v9, v10 and v11
retain their blocked verdicts. This is a small DEV editorial experiment, not
training material or a release benchmark. At most twelve new originals may be
authored, with at most four per Choice, Noul and Score; fewer pass if the
quality gates cannot be met. No FINAL access, GPU run or publication is
authorized by this protocol.

## Why the method changes

v11's originals were independently solvable, but four stage-source deletions
became ambiguous because the rule had not fixed the universe and order of
stages. A rate-rubric deletion left a conventional-grade hint. The repeated
whole-exhibit removal instructions and parent linkage gave reviewers
unnecessary metadata, while long cases still repeated caveats. Mechanical
two-completion sensitivity is insufficient to make a natural missing-source
question well posed.

## Stable rule and source design

1. Before writing any evidence, define a full decision contract: entity
   universe, named members/stages, their canonical order, precedence,
   missing-evidence treatment, tie rules, units, exact score bands and output
   vocabulary. The contract stays verbatim in every variant. Source display
   order cannot define semantic order. Rubric thresholds and grade semantics
   belong in this invariant contract; no removable source may carry the only
   threshold while the surviving state retains a suggestive measurement.
2. Build each original from independent, natural source facts with at least
   two distinct resolution steps. Every source has a same-format,
   same-length-range **counterfactual substitution** that changes its
   decision-bearing fact and changes the correct answer under the invariant
   rule. This is the primary source-discrimination proof and does not depend
   on a document becoming absent. A second oracle enumerates the result from
   the structured facts; author and oracle must agree on original and
   substituted answers.
3. Source withdrawal is a separate robustness test, not an automatic
   `unknown` target. Depending on the frozen rule, the reduced state may have
   a unique different answer or multiple admissible answers. The expected
   outcome set is computed before rendering and sealed privately; an original
   answer that remains uniquely forced fails the withdrawal sensitivity gate.
   If the rule allows multiple answers, the question must request
   insufficient evidence explicitly and consistently. A blind reader must
   reach the same status without guessing the withdrawn source, hidden roster
   or conventional band. Direct original-answer hints fail even when a
   formal proof says `unknown`.
4. Use a mix of natural perturbations: full source withdrawal, field-level
   redaction where such redaction occurs in that document type, and later
   correction or supersession that explicitly withdraws an earlier claim.
   A correction must obey the preregistered temporal precedence rule. No
   variant may introduce a dangling reference or a meta instruction saying
   that a source was removed. The model-facing prompt contains only the
   native task, invariant rule, evidence and question.

## Editorial and blind-review gates

Fresh v12 cases must use at least eight new domains and semantic mechanisms
that do not reuse v9–v11 fact packs or relabel their operations. At least five
evidence forms must serve real decision steps. Long cases are allowed only
when each paragraph changes a scoped fact, reliability assessment,
precedence or competing interpretation; there is no length quota to fill.
Author-side inspections will flag repeated eight-word prose, cross-case
template overlap, grade words or conventional-band cues in evidence,
unnecessary sources and option-position shortcuts. Choice positions and
Boolean values must be balanced among surviving cases, but labels are fixed
by the underlying facts before presentation order is assigned.

Reviewer A receives only salted original IDs and native prompts. After A
seals original answers, reviewer B receives a separately salted, shuffled
variant roster. B must not have seen the originals. The variant roster
contains no `parent_id`, omitted source, perturbation kind, original answer,
target/proof or pairing instruction. The private join map remains inaccessible
until B seals answers and editorial judgments. Both reviewers record direct
answers, ambiguity, unnatural wording and suspected shortcuts. SHA-256
commitments, UTC seal times and receipt ordering are checked before any key
access; a post-key operator reports aggregate results and variant failures.

**Stop rule:** Any ambiguous original, wrong oracle join, false source
necessity, answer cue, dangling reference, grade shortcut, reviewer linkage
leak, formulaic filler, or disagreement between a blind variant answer and
the preregistered admissible outcome set blocks that frozen candidate. Do not
patch a sealed candidate after seeing blind judgments. A failed candidate is
retained with its receipt, and a later version needs a new prospective
protocol. Passing this small DEV pilot would only justify building a larger
independent pool; it is not a release score or model-quality claim.
