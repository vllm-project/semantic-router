# Eos 0.8B FEVEROUS metadata screen v2: explicit unlabeled quarantine

**Status: prospective CPU-only follow-up, no training admission.** The
[v1 screen](eos08-feverous-source-screen-v1-hold-2026-09-28.md) failed its
strict schema rule because one of 1,680 sampled publisher TRAIN records has
an empty label. That failure remains immutable. This v2 protocol was written
*after* viewing v1 aggregate labels and text-evidence counts; the four-range
sample is no longer blind evidence. Its purpose is only to decide whether the
full 175 MB TRAIN annotation file warrants a further document and native-token
audit. It is not a new benchmark, student result, or independent validation.

## Frozen source and admissible cleanup

Fetch only publisher Zenodo `train.jsonl` from
<https://zenodo.org/records/4911508/files/train.jsonl?download=1>. Require
exactly 175,493,294 bytes and the Zenodo MD5
`d8d4634760dad714b4cc30e43d25e589`; compute and record SHA-256. Do not
download the approximately 9.9 GB Wikipedia pages at this step. Do not open
publisher dev/test, protected Decision answer keys or model predictions.
Quarantine records whose `label` is the **empty string**; count them and never
use them in any train/eval role. Missing label keys, unknown nonempty labels,
duplicate source IDs, malformed claims or evidence schema fail this v2 screen
rather than being silently repaired. No option/rubric modification is allowed.

The only possible native projection remains **0 refuted / 1 supplied evidence
insufficient / 2 supported** against a fixed claim and article state. A row
is text-eligible if at least one complete publisher evidence set has only
sentence elements with their declared context. Table-bearing or mixed sets
cannot be converted by deleting required cells. NEI meaning with finite
visible pages remains an independent semantic review gate. The [publisher
terms](https://fever.ai/download/feverous/license.html) govern annotations
and article-specific Wikipedia rights; source text and page IDs stay private.

## Metadata-only stop gates

Report label frequencies and text-only counts for all publisher TRAIN records,
the count of empty-label quarantines, and counts of distinct referenced pages
per class. Count duplicate normalized claims and page-linked groups without
printing claims or pages. A stable SHA-256 order of publisher record IDs may
be used to form a **candidate** roster of 64 records per class with 192
distinct claim IDs and no shared referenced page between selected records.
No publisher dev/test record is eligible. If the full TRAIN file does not
contain this balanced page-separated roster, or if source integrity/schema
fails, stop before any wiki archive download. Passing these metadata gates
merely permits a later costly full-page audit; it does not admit TRAIN data.

The full-page audit, if pursued, must confirm that every selected claim's
required publisher evidence remains visibly present in an unhighlighted,
faithful text rendering of complete articles ≤8,192 Eos native tokens, while
all 192 native requests total at least 155,681 tokens. It must check exact,
near and semantic source overlap by original page and claim group against the
control TRAIN/SELECT/CAL, typed DEV/FINAL gold-free prompts, CSS pilot/final
and public supplements. Existing FEVER/VitaminC and HoVer ancestry makes
source-name absence inadequate. Require independent blind review of 24
balanced rows for label/rubric fidelity and answer leakage, and claim-only and
evidence-deleted shortcut probes on held page groups; these must stay within
five accuracy points of their majority baselines. The article-specific
license/attribution ledger, source-ID manifest and protected-set exclusion
must pass before even proposing an Eos GPU arm.

This v2 metadata screen cannot prove long-document exposure or general Score
transfer. Eos 0.8B release remains HOLD until its native development and
formal results pass separately.
