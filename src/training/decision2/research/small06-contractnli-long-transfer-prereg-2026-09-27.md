# 0.6B long-document transfer screen: prospective preregistration

**Status: protocol only. No ContractNLI labels have been opened for this arm,
no GPU inference or optimizer update has run, and no Decision 2.0 candidate is
selected.** This is a bounded development screen to decide whether a longer
native encoder merits a new training arm. It is not part of sealed JevArena v3,
the public JevBench subset, or the 0.6B release gate.

## Why this experiment comes next

The pinned 486,444,053-parameter English GLiNER2.5-Decide source scored
652/1,600 on typed DEV and 561/1,430 on the three human CSS pilot tasks, but
its native 512-position limit invalidated 56/231 public JevBench questions.
Its completed 64-step, short-input continuation fell to 584/1,600 on typed DEV
and transition fell 21→4/400. Its head-only human-label arm changed zero of
700 SELECT decisions. Kai's rights-clean continuation improved typed DEV
425→587/1,600 but lowered CSS pilot 418→410/1,430. The completed 0.6B Qwen
reranker paired-order arm stopped at SELECT with no gain over its control.
Those negative results do not warrant another 512-position training run.
Laya's 421,293,830-parameter ModernBERT backbone can theoretically handle
8,192 positions, but its pinned published call policy is 1,024; an exploratory
4,096-call ablation is not the published native comparator for this screen.

The pinned multilingual GLiNER2.5 source is smaller (287,355,159 parameters)
but admits 4,096 native positions. On existing exposed panels it scored
579/1,600 typed DEV and 484/1,430 CSS pilot, below the English source. Its
public long-input gain mostly came from fewer overflows; it was worse on
questions both sources could answer. The open question is whether its native
long window retains useful **unseen-source, document-level decision signal**.
Existing SELECT700 and typed DEV1,600 fit within the Qwen reranker's 512-token
window and cannot answer this question. Running another optimizer arm first
would consume GPU time without a valid long-transfer selector.

## Frozen sources, task and data separation

- Candidate architecture: untouched
  [`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
  at revision `6bc1d43d201b0691e733626389af8c57eea3ea68`, measured
  287,355,159 parameters, weight SHA-256
  `9efe0f88c99f2aa794452e9559dc60e98d60d9fa2bf1b60cf2710411b6da5b4e`.
  Use the existing native boundary-classification adapter, not a new prompt
  tuned on these labels. The GLiNER2 library commit is
  `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`.
- Matched longer-window reference: untouched
  [`Qwen/Qwen3-Reranker-0.6B`](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
  at revision `e61197ed45024b0ed8a2d74b80b4d909f1255473`, measured
  595,776,512 parameters, weight SHA-256
  `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`.
  Use its already audited native yes/no candidate scorer with the same
  4,096-token, no-truncation policy. The two parameter counts and different
  pretraining are disclosed; this is architecture triage, not a matched-size
  causal ablation or a JevArena rank.
- New source: the **development** split of the authors'
  [ContractNLI release](https://stanfordnlp.github.io/contract-nli/) under its
  [CC BY 4.0 terms](https://github.com/stanfordnlp/contract-nli/blob/gh-pages/LICENSE).
  Its original full-document text, one of 17 fixed hypotheses, and human
  `Entailment`/`Contradiction`/`NotMentioned` label define one three-way Choice
  question. The published paper reports about 2,254 document tokens on
  average and 86% of documents over 512 BERT tokens, which makes this a
  plausible long-input screen; our two native tokenizer lengths must still
  pass the gates below. The paper describes the task and evidence
  annotations; evidence spans are **not** included in the prompt. Keep the
  exact downloaded archive, split file, license, hash and original source IDs
  in the private audit. Do not use its train or test split for this screen or
  any Decision 2.0 training. Unknown upstream pretraining exposure remains a
  limitation even when our fine-tuning source IDs are disjoint.

Before viewing any ContractNLI answer or starting inference, audit the
rights-clean v2 TRAIN7,455 / SELECT700 / CAL700 source ledger (TRAIN SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`),
all existing Decision 2.0 training candidates and open development prompts
for original source ID, document URL, normalized full-text, exact prompt and
approximate near-duplicate overlap. Remove whole source documents on a
collision; record every exclusion. Check against sealed panels using only
gold-free inputs and commitments. The panel does not enter HF training data.

Use the full original document as state. The question is: “Under this entire
agreement, is the following hypothesis entailed, contradicted, or not
mentioned? <hypothesis>”. The criteria are fixed, in this order:
`entailed`, `contradicted`, `not_mentioned`, with their literal definitions.
No document chunking, retrieval, summaries, evidence annotations, option
deletion or truncation is allowed; otherwise the `NotMentioned` meaning
changes. Record each model's actual native token count for the complete
request. A question is eligible only when **both** native request lengths are
strictly above 1,024 and at most 4,096 positions. This makes the primary
panel a genuine long-input comparison within both original native adapters.

Rank eligible original documents by SHA-256 of
`decision2-small06-contractnli-long-v1:<source-document-id>` and keep the
first 96. Within each document choose at most one hypothesis of each gold
class by SHA-256 of the hypothesis ID with the same seed. Thus each document
contributes at most three correlated questions and never counts as three
independent examples. The audit must show at least **40 distinct documents**
and **20 questions of each gold class**, with at least 20 documents requiring
more than 2,048 native positions for both models. If these population gates
fail, stop without inference or choosing another dataset, split, length band
or render. The resulting fixed prompt and private gold files get SHA-256
receipts; the public branch contains neither raw text nor labels.

## Zero-step runtime check and GPU ceiling

There are **zero optimizer steps**. Before any panel prediction, verify both
source packages against all pinned model files and revision metadata, count
loaded parameters, capture tokenizer/adapter/scorer and image digests, and
verify the original native schema on a 16-question gold-free packet sampled
by fixed ID hash from the frozen panel. Two independent loads in the same
runtime must have zero categorical changes and maximum corresponding option
probability drift at most `1e-4`. Every input must fit within both declared
4,096-token caps; no silently clipped text. A loader, identity,
schema, token-length or stability failure stops this arm **before labels are
scored**. Do not fix a failure by changing a checkpoint, precision, context
cap or prompt after seeing a score.

Run one complete gold-free prediction pass per source model on one authorized
GPU at a time, under a combined ceiling of **0.75 measured GPU-hour**. Before
the full run, estimate cost from the 16-question packet; if it projects above
the ceiling, stop and preregister a separate lower-cost experiment. Do not
select a faster subset based on model outputs. Seal complete prediction files,
native receipts, input IDs and hashes before opening the development labels.
Missing, invalid and over-budget responses count wrong. No CAL temperature,
prompt search, training, public test or protected FINAL/CSS15 label is used.

## Fixed readout and decision

Report the raw number of independent documents, three-way class counts,
native length buckets (1,025–2,048 and 2,049–4,096), all-item coverage,
accuracy, macro-F1, per-class recall, confidence/Brier and failure causes for
both models. Primary contrast is multilingual GLiNER minus Qwen reranker
**macro-F1** on exactly the same questions. Compute a 95% paired percentile
bootstrap with 2,000 resamples of complete document groups, seed `20260927`;
the multiple hypotheses and option-order variants never become independent
samples. Separately report evidence-bearing versus NotMentioned outcomes.
One deterministic reverse option order for 24 document IDs chosen by the
same hash is a robustness view only; it cannot change the primary rank or
sample count.

Advance to a *newly preregistered* GLiNER-multilingual long-structured
training pilot only if the source itself has at least 95% valid answers,
macro-F1 at least `.45`, a nonnegative paired point estimate against the
reranker, and at least `.40` recall for each of the three classes. These are
research-efficiency thresholds, not statistical proof of superiority or a
release claim. If any fails, record the negative result and stop this
multilingual-GLiNER training proposal; consider a different long encoder in a
separate prospective protocol. Even a pass does not show improvement over the
English source's 652/1,600 typed DEV or authorize release. Any subsequent
training must use source-disjoint, group-held-out TRAIN/SELECT/CAL, include
Choice/Noul/Score and structured long-input rows, then clear the same-panel
typed DEV, human-transfer and public-subset gates before a sealed v3 attempt.

The [ContractNLI authors' description](https://stanfordnlp.github.io/contract-nli/)
and [paper](https://aclanthology.org/2021.findings-emnlp.164/) support the
task definition, evidence labels and long-document motivation, not a claim
that either model will work on it. No result is inferred from this protocol.
