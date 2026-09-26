# Hard multilingual decision pilot: prospective DEV protocol

Status: private DEV candidate, **not approved for model inference or release**.
The existing simple 18-base seven-language pilot was saturated by every tested
model, while the 100-source-ID XNLI/PAWS-X panel omits Score. The new candidate
addresses negation, conflicting evidence, dated state, ordinal arithmetic and
paired language consistency. It does not replace human-translated transfer
data or the sealed JevArena panel.

## Freeze and independence

The private casebook has 18 independently authored base cases, two for each
of nine rule operations. Each is rendered in English, Chinese, Spanish and
Japanese: 72 prompts total, six base cases and 24 language rows per task type
(Choice, Noul, Score). The four versions of a base are **one analysis unit**.
Structured facts feed a separate mechanical oracle. Neither a model response
nor a hand-entered answer is used to derive targets. The private casebook,
gold-free prompts, separate targets, generator and scoring source, and manifest
must be hashed before the first model response. Any edit creates a new candidate
version; old receipts remain available.

Choice uses explicit options with a recorded semantic-ID-to-native-label map;
Noul uses the native boolean probability readout; Score uses levels 0–3.
Missing, malformed or invalid native answers count as failures. Paired language
consistency is computed by base case, including the fraction correct in all
four languages. For model comparisons, freeze exact checkpoints, native
adapters, prompt order and scoring source first; report each type × language
cell and the count of independent bases. The 18-base sample supports failure
discovery, not stable ranking or broad multilingual claims.

## Preregistered acceptance gate before any model run

1. An independent gold-blind reviewer must solve and audit **all 72** localized
   rows without seeing the casebook or targets. Every row must be unambiguous,
   natural enough for its language, and faithful to the same formal rule.
   One mistranslation or shortcut that changes an answer blocks this frozen
   candidate. Machine-generated translation or automatic language ID alone
   does not satisfy this gate. An actual native or qualified bilingual review
   is preferred; absent that, keep `inference_eligible=false`.
2. After the reviewer seals answers and rationale, compare against the private
   mechanical oracle and audit mismatches without revising the frozen packet.
   Detect whether rule-text cues, option positions, or shared numeric syntax
   allow a trivial language-independent shortcut.
3. Run exact and approximate overlap screens against available TRAIN,
   SELECT, CAL and visible DEV/public benchmarks, plus any available gold-free
   sealed source-ID denylist. The current screen is text-based and cannot
   certify semantic non-overlap. Never open sealed FINAL labels.
4. Validate all 72 option layouts with the pinned native parser, including
   Noul boolean and Score level order. A format mismatch blocks inference.

The first accepted candidate, if any, is still **DEV-only**. Do not train on
these prompts, use them to select a release checkpoint, or include their score
in the JevArena release total. Analyze mistakes as hypotheses for new
training data, then evaluate improvements on independent tasks.

## Current candidate boundary

The renderer uses independently written locale templates and one structured
fact source per base. This preserves numbers, codes and boolean states
mechanically, but cannot prove translation quality. Tabular syntax and the
limited operation family can create shortcuts; a ceiling result would trigger
another redesign rather than a multilingual claim. Private prompts and targets
are ignored artifacts and do not enter this repository. This note contains no
source text, model score, secret or private machine identifier.

## Frozen candidate r6: mechanical QA only

Earlier r1–r5 files are retained as pre-inference revisions for locale wording,
style and source-code formatting; **no model was run on any version**. The r6
private casebook SHA-256 is
`6cc11921e6422afba681d64fabd2abd457c54e8a43ff849f802e8a8653ba4911`.
Its prompt, gold-free review, separate target and manifest SHA-256 values are
`d8cd32d61bb79b6cc189acdf90b69b1fba7938020f167fc69918baca08f8a200`,
`9b16c9ba50f51c52108732f0fdb3221580f892758e855f410839503160087824`,
`a18ea8d2262dd058400202547c1af2c9c1867f1301858709ba60269be8b1b989`
and
`ffc31421654ddf685d7a31bbdb5a4d1d30d82da8e571c8f56fa6b75f79c1e218`.
The manifest pins generator/native adapter
`ff1f47a822336cba8e66cd8ceb059f5bf3c1c244f6cc272755e465ef82b1d9ae`
and scorer
`b7a5d45895a862723fbdc8263781d9e1deef248e2fcb3ecff1a2e8871efa1149`.

An oracle-only scorer smoke checked the mechanics, not model quality:
72/72 synthetic native answers were accepted, 18/18 base groups were correct
in four languages, and removing one answer yielded 17/18 all-language groups
with one invalid/missing row. The synthetic prediction SHA-256 is
`08d6284b03d31c5eb884fbe920160524d4dfa487c342de23f9fb26f54f974a25`.
The prior simple 210-row multilingual development source had zero exact and
zero >=0.94 near state matches.

The exact published Eikos native `decision_core.py` source matched its model
package SHA256SUMS entry,
`9ff8d754ce99c6539fc7f5bd88c10b196357c70124f59ed3d1bd1a0d3fdbb7d0`.
CPU-only parser preflight found all 72/72 layouts valid, with Choice/Noul/Score
24 each and the expected native option order. The preflight source and private
receipt SHA-256 values are
`2fd8c9a9de7f5dd36967f0de0be12066c28e60fdbe3dccf766116ca0fb3ea76c`
and
`57770c26ebfdce25c6d2243d8512afaa1d20ae86525e6a44c6f21a717e1d2477`.
No model weights were loaded.

The current rights-clean TRAIN/SELECT/CAL references comprise 8,855 rows and
have zero normalized exact and zero length-filtered >=0.94 near state matches
to r6. Their SHA-256 values are pinned in the private receipt
`14e03d5e68bf35a5ac1f36dc4a3c357a5202473c9ce85444c9eb678e50718b24`.
Five visible development/public reference files added 2,292 state rows and
likewise gave zero exact and zero >=0.94 near matches; private aggregate
receipt SHA-256
`a6c14d294b6e9569952c5955f72addbbbe5aa68c9d9b8eee22db6385cc707e23`.
This text screen does not rule out semantically equivalent paraphrases or
overlap with sealed FINAL. Gold-blind bilingual review remains pending, so the
manifest still has `inference_eligible=false` and no accuracy claim exists.
