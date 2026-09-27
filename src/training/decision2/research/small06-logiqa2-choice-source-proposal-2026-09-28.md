# 0.6B Choice source candidate: LogiQA 2.0

**Status: source research only; no row admitted, no model run.** This proposal
follows the current official-Qwen 0.6B model's severe typed Choice loss. It
does not revise the running fixed-data gradient-projection arm or use its
outcome. A later data ablation needs a separate prospective lock.

The [official LogiQA 2.0 repository](https://github.com/csitfun/LogiQA2.0)
at `955e1d3df6c59d9bfb44d9913da1e1a27ec14e18` supplies original Chinese
and professionally translated English multiple-choice logic questions,
train/dev/test splits, source question IDs, four alternatives in the published
example, and reasoning-type annotations. The authors say three workers checked
label consistency after translation. The repository states CC BY-NC-SA 4.0;
its exact files, data rights and attribution requirements still need an
independent source inventory before private TRAIN admission. The English
translation and Chinese original of one question are one source group, never
two independent examples. The NLI derivative is grouped with its MRC parent
and cannot be used as an independent transfer result.

**Why this source may help:** the user's native System One Choice path asks a
named question over state and alternatives. LogiQA's original human-authored
arguments and answer choices test conditional, necessary/sufficient and other
logical reasoning absent from the mostly short-classification clean-v2
mixture. This is a data-coverage hypothesis, not a prediction of improved
JevArena or real-task transfer. Public exam questions may be present in
foundation-model pretraining, and translated English is not independent
multilingual evidence.

Before any GPU work, on an authorized experiment node pin the official raw
files and SHA-256, count unique original IDs and complete Chinese/English
pairs, check answer validity, option-order distribution, native full-input
tokens, source/license ledger and group-level overlap with TRAIN/SELECT/CAL,
all JevArena prompts and the public 231-item set. Blindly review a fixed
stratified sample for answerability, translation ambiguity and whether the
published answer follows the question wording. Any protected near match or
systematic label ambiguity is a HOLD. Keep source text and review packets
private.

Only a passed CPU/data gate can authorize a **separately frozen** matched-token
Choice replacement against the already archived 0.6B control, with exactly
one predefined model start, row roster, optimizer budget, SELECT rule and
source-disjoint confirmation. Do not append translated duplicates, convert
this four-way task into arbitrary Noul/Score labels, or use its own validation
set as evidence of cross-source transfer. The current formal v3 keys are
already opened, so future v3 scores are same-panel comparisons.
