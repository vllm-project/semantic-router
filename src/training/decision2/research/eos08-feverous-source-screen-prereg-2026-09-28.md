# Eos 0.8B: FEVEROUS text-evidence Score source screen

**Status: prospective CPU source screen only.** This is a new publisher source,
not an admitted training arm, model result, or evidence of long-input transfer.
No GPU, model prediction, protected answer key, or optimizer is used. This
protocol was frozen before parsing source labels in the sampled rows. A byte
range had been fetched to test HTTP Range support but had not been parsed.

## Fixed question and prior comparison

The eligible initializer is our Decision 1.0 Eos 0.8B package. The previous
same-roster replay failed its development promotion: Score stayed at 85/400.
The exact Eos hard-replay Score block contains 192 rows and 201,359 native
tokens. A same-total-token replacement under the previously frozen ±1% budget
would need at least 155,681 Score tokens across exactly 192 independent rows.
The OCNLI maximum was only 31,789 tokens; VitaminC's sampled revisions were
also short. Those negative screens remain unchanged.

[FEVEROUS](https://fever.ai/dataset/feverous.html) is a distinct human-verified
Wikipedia source with `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` labels,
evidence element IDs and article provenance. The [publisher paper](https://arxiv.org/abs/2106.05707)
reports only about 5% not-enough-info claims and substantial table dependence.
The [publisher terms](https://fever.ai/download/feverous/license.html) follow
article-specific Wikipedia terms, with CC BY-SA 3.0 fallback. This does not
authorize stripping attribution or placing article text in a public model repo.
The [publisher Zenodo record](https://zenodo.org/records/4911508) lists
`train.jsonl` at 175,493,294 bytes with MD5
`d8d4634760dad714b4cc30e43d25e589`; the full Wikipedia archive is about
9.9 GB. Use only publisher TRAIN and do not download that archive at this
stage. A byte-range sample cannot verify the full-file MD5.

The sole contemplated native System One `Score` rubric is ordered **0:
evidence refutes the claim; 1: the supplied evidence is insufficient; 2:
evidence supports the claim**. The claim becomes the question and faithfully
rendered source Wikipedia page text becomes the state. This is an evidence
relation, not a numerical severity scale. No table-dependent row may enter a
text-only student by dropping table evidence. A later explicitly reviewed
textual rendering of entire relevant tables would be a separate protocol.
Publisher `NOT ENOUGH INFO` is a verdict relative to searched Wikipedia, so
its preservation under a finite supplied-page state must be checked blind;
we will not equate it to a generic unknown label automatically.

## Frozen small authentic sample

Use HTTPS byte ranges from publisher Zenodo `train.jsonl?download=1` of exactly
1,048,576 bytes starting at offsets **0, 43,873,323, 87,746,647,
131,619,970**. Require HTTP 206, matching `Content-Range`, total length
175,493,294 and exact byte count; record SHA-256 for each range privately.
For nonzero offsets discard the first partial line; always discard the last
partial line. Parse only complete JSONL records. Do not output source text,
claim IDs, page titles or labels by row. This deterministic four-quartile
sample is a feasibility probe, not a random corpus estimate or training set.

Report aggregate counts by source label and evidence type, plus counts of
distinct referenced Wikipedia page groups. A candidate is **text only** only
if at least one complete annotated evidence set contains exclusively sentence
elements and its required context is available; a mixed or cell-only set is
not converted by omission. Before even considering a full publisher TRAIN
download, require at least **16 distinct referenced pages** for each of
`SUPPORTS` and `REFUTES`, **8** for `NOT ENOUGH INFO`, and at least **8**
text-only NEI records in these four ranges. These are cheap screening floors,
not final admission. If schema, identity, byte-range integrity or floors fail,
stop this source here. A failure is not a statistical upper bound on the whole
corpus.

## Later admission gates, all currently unpassed

Only if the small screen clears its floors, pin/download full publisher TRAIN
and recheck its published MD5. The full Wikipedia archive is a separate cost
decision after full TRAIN metadata. A prospective 192-row treatment must have
64 records per label, 192 independent page/claim components, all required
evidence visible without gold highlighting in native states, no answer text
leakage from labels/metadata, every full state ≤8,192 Eos tokens and total
native Score tokens ≥155,681. Use complete documents rather than padding or
arbitrary evidence-preserving truncation. Group all claims sharing a page or
near-duplicate claim before selection. Verify original article rights per
record and preserve source attribution privately.

Compare exact, near and suspicious semantic matches against rights-clean
TRAIN/SELECT/CAL, gold-free typed DEV/FINAL, CSS pilot/final and public
supplements; quarantine entire linked page/claim groups. A fixed answer-blind
review of 24 stratified source rows must check the three-level rubric, page
completeness and ambiguity before training. A page-free claim-only probe and
evidence-deletion probe on source-disjoint groups must not exceed their
majority controls by more than five accuracy points. Any later model arm must
pre-register the exact Eos native adapter, initializer, full token/step
budget, zero-step parity, selection and stop gates. Protected v3 or public231
scores cannot be used for this source choice.

Passing the small screen only justifies a larger CPU audit. Until every later
gate is met, **0.8B training and release remain HOLD**.
