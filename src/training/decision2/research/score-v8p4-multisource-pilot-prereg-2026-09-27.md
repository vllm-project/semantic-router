# Score v8.4: diverse source-document pilot (prospective)

**Status: design frozen before generating any v8.4 row or opening a v8.4
selector key.** Earlier v8.3 is a logic smoke with a quality HOLD: four
contradictory workflow states, four copy defects, and short English text drawn
from three repetitive instruction/option templates. This new pilot uses a
fresh seed and case IDs. It does not edit, relabel, filter or promote v8.3.
No optimizer, model selection, protected panel inference or release claim is
authorized by this document.

## Fixed question pool and construction

Construct **six** Score mechanisms with five independent case groups each:
three TRAIN groups and two SELECT groups per mechanism. Each group contains
one level-0, one level-1 and one level-2 counterfactual, giving 54 TRAIN and
36 SELECT rows, 90 rows total. A case group shares its underlying entity and
documents; groups have distinct case facts, source text and IDs. Freeze one
private seed before generation, derive role/mechanism/group streams by HMAC,
and publish only its digest. Do not reroll a seed or replace an unfavorable
group after review. The 90 rows are a **quality pilot**, not a training arm.

Mechanisms and oracle rules are fixed before rendering:

1. **Workflow readiness:** transitive prerequisites, a status board and an
   explicitly timestamped reconciliation note. An unresolved failed ancestor
   gives 0, a queued ancestor with none failed gives 1, all resolved complete
   gives 2. Completed descendants with an incomplete ancestor require an
   explicit stale-status resolution; contradictory live facts are rejected.
2. **Timed connection:** arrival uncertainty, minimum interchange, booking
   cutoff and a cancellation/change bulletin. Earliest arrival after cutoff is
   0, an interval crossing cutoff is 1, latest arrival within cutoff is 2.
   Equality at cutoff is feasible and must be represented in at least one
   group; a nearby service is irrelevant unless it is the named connection.
3. **Stock fulfillment:** a named order, usable on-hand stock, reservations,
   expected replenishment and a dated carrier confirmation. Optimistic stock
   below demand is 0, only unconfirmed units closing the gap is 1, confirmed
   usable units suffice is 2. Units and SKU identity must be explicit.
4. **Versioned policy:** a base eligibility rule, a scoped exception and a
   later amendment with an effective date. The current record fails a
   non-waivable requirement (0), awaits explicitly specified evidence (1),
   or satisfies all current requirements (2). The oracle applies amendment
   precedence and scope rather than a keyword in one section.
5. **Evidence bundle:** two independently attributed documents support or
   contradict a requested factual conclusion, while a third record is a
   near-entity distractor. A direct contradiction is 0, material evidence
   absent is 1, both necessary sources support the conclusion is 2. Absence
   must be a defined rubric outcome, not an unsupported guess.
6. **Service-level obligation:** a dated request, a contract schedule and a
   holiday/service calendar determine whether a deadline is impossible (0),
   conditional on an unconfirmed prerequisite (1), or assured under the
   available confirmed facts (2). The oracle computes dates explicitly and
   distinguishes business days from calendar days.

For each mechanism, render at least three materially different document
genres across its groups, with source attribution and shuffled evidence
ordering. A minimum of one group per mechanism must require joining facts
from two documents; two mechanisms must include a third decisive document.
Across the full pool, at least 24 rows must exceed 700 tokens and at least six
rows 1,500 tokens under the exact candidate tokenizer, with decisive evidence
in both early and late positions. At least 18 rows must be native Chinese
cases written and independently reviewed in Chinese, not sentence-by-sentence
translations of English cases. English and Chinese groups remain disjoint.
Every document must contain realistic irrelevant material without artificial
"other item is unrelated" cues. Vary instructions and ordinal descriptions
within the same oracle, but keep each row's grading rule fully explicit.

## Frozen admission rules

- Compute the oracle from structured facts **before** rendering. Independently
  parse the rendered documents and recompute the answer. Reject any row whose
  structured and rendered oracles disagree, and quarantine its whole case
  group. Check target/decoy identity, option and document order, numbers,
  timestamps, grammar and the absence of one-field label cues. Record all
  excluded groups and causes. At least 27/30 groups and at least four groups
  per mechanism must pass, otherwise the entire pilot is HOLD. Do not backfill
  excluded groups.
- Keep TRAIN and SELECT disjoint by source case and mechanism-specific
  document template family. Compare normalized exact and bounded near matches
  against parent TRAIN/SELECT/CAL, prior Score versions and available
  gold-free protected prompt inventories. A protected overlap or suspicious
  semantic near match is a whole-pilot HOLD pending independent adjudication.
- Before any reviewer sees a key, freeze labeled source rows, keys, gold-free
  blind packets, the generator/code hashes and data audit. A different agent
  reviews every retained original and every complete counterfactual triplet
  for answerability, ambiguity, ordinal coherence, document realism,
  evidence necessity, language quality and shortcut cues. Seal its labels and
  issue list before oracle comparison. Chinese rows need a reviewer able to
  judge their actual text. Any target disagreement, unsupported decisive
  fact, unresolved contradiction or systematic shortcut is HOLD for the
  pilot. Correct labels alone are insufficient.

Only a quality PASS permits a larger source-disjoint data arm and matched
same-checkpoint control/treatment preregistration. That later arm must fix
token/step budget, replay, original 27B starting checkpoint, SELECT and CAL
roles, Score gain and Choice/Noul non-regression criteria before GPU work.
Existing DEV, formal, public and Index numbers may guide the hypothesis but
cannot select v8.4 rows or models post hoc.
