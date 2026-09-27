# Human NLI evidence Score source screen

**Disposition: HOLD.** This is a CPU-only TRAIN-source audit, not a data
admission, model training run, model score, or release claim. GPU use: **0
GPU-hours**. A direct System One three-level Score rubric would ask whether
the shown premise **refutes / does not determine / supports** a claim. The
NLI mapping is respectively contradiction / neutral / entailment. This is a
specific evidence relation, not a substitute for rule precedence or state
updates in the typed Score benchmark.

## Frozen source and rights

| Publisher source | Pinned revision and TRAIN SHA-256 | Original condition relevant to this screen |
| --- | --- | --- |
| [Stanford SNLI 1.0](https://huggingface.co/datasets/stanfordnlp/snli) | HF publisher repo `cdb5c3d5eed6ead6e5a341c8e56e669bb666725b`; `ef9a7b25d97390a62aeda7abe26aec8640600f50b818eaeb9107097d60ac6620` | CC BY-SA 4.0; original caption and crowdworker text; the publisher card describes hypothesis-only artifacts. |
| [Original OCNLI](https://github.com/CLUEbenchmark/OCNLI) | Publisher GitHub `02d55cb3c7dc984682677b8dd81db6a1e4710720`; `cb47a6c00105bfb49bbb791dd537b284d3138f9f1ad5cd1e187afc470ff004e0` | CC BY-NC 2.0; the news-genre premises also originate from LCMC under a separate publisher permission statement. Screen non-news genres separately and do not infer rights for the news text from the general license. |

Only the original OCNLI TRAIN file was read, retaining `genre` and `prem_id`.
Only the SNLI TRAIN parquet was read; publisher development and test files
were not downloaded. Source text, labels, row IDs and private rosters were
kept on the authorized private experiment host. The audit used the existing
official Qwen3.5-4B Base tokenizer for a deterministic 10,000-pair **raw**
token-length sample per source, before native System One wrapping.

| TRAIN source | Raw / eligible pairs | No-consensus / missing provenance removed | Conservative independent premise groups | Contradiction / neutral / entailment | Raw pair tokens median / p90 / p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SNLI | 550,152 / 549,367 | 785 / 0 | 150,687 | 183,187 / 182,764 / 183,416 | 23 / 34 / 48 |
| OCNLI | 50,486 / 50,434 | 49 / 3 | 7,954 | 16,476 / 17,179 / 16,779 | 20 / 33 / 43 |

SNLI has 653 exact normalized duplicate pair rows. OCNLI has 8,037 source
`prem_id` groups, but 99 IDs map to multiple normalized premises and 101
source IDs share a normalized premise with another ID. A connected-component
union of both ID and normalized premise reduces the independent group count
to 7,954; future partitions must use that component, not individual rows or
publisher IDs. OCNLI's 10,672 eligible news rows are quarantined by default.
The remaining four genres contain 39,762 rows / 6,314 conservative groups,
still only **screened**, not admitted.

Both sources are short: full raw pair character median / p99 is 97 / 218
for SNLI and 33 / 68 for OCNLI. Neither constitutes a real long-document
Score training arm.

## Protected-source overlap and shortcuts

The protected inventory was SHA-256 pinned at
`c6d3f497b4385ff48817e9a6f98f63baf529b99b27a132a190164d5d729c18c2`.
It covers 35 role files and 32,269 row comparisons including rights-clean
v2 TRAIN/SELECT/CAL, typed DEV/FINAL gold-free prompts, CSS pilot/15
gold-free prompts and public supplement roles. Only prompt/state text was
consumed; benchmark role files were required to be prompt-only. Roles can
overlap, so 32,269 is not a number of independent questions. The script
rejected answer-bearing protected benchmark entries.

| Source | Exact group matches | Heuristic near group matches | Reading |
| --- | ---: | ---: | --- |
| SNLI vs rights-clean v2 TRAIN | 398 | 788 | Expected preexisting SNLI-related material; quarantine at premise-group level, not by individual pair. |
| SNLI vs CSS15 gold-free | 0 | 90 | Suspicious lexical overlap; not adjudicated, so quarantine conservatively. |
| SNLI vs CSS pilot gold-free | 0 | 6 | Same qualification. |
| SNLI vs Decision Bench text subset | 0 | 1 | Same qualification. |
| OCNLI vs screened roles | 0 | 0 | Zero under this lexical screen is not a semantic-disjointness proof. |

The near scan indexes compact six-character grams from protected textual
leaves up to 800 characters and verifies Jaccard ≥0.72 or shorter-side
containment ≥0.90. It is a **heuristic candidate screen**: 9,444 longer
protected leaves were excluded from this near stage, semantic paraphrases
can be missed, and near candidates can be false positives. Exact and near
group counts can overlap and must not be added. All source groups
in any suspicious match must remain out of future candidate partitions until
private record-level adjudication. No formal answer key or model prediction
was opened.

A deterministic hypothesis-only negative control trained on a group-disjoint
portion of each publisher's TRAIN, using binary character 2–4 grams and a
simple multinomial Naive Bayes classifier. This is an artifact probe, not a
Decision model evaluation:

| Source | Train / test pairs | Majority test accuracy | Hypothesis-only accuracy / balanced accuracy | Original row position modulo 3 / 9 |
| --- | ---: | ---: | ---: | ---: |
| SNLI | 60,000 / 15,000 | 33.29% | **57.77% / 57.76%** | 32.71% / 33.62% |
| OCNLI all genres | 30,000 / 10,000 | 33.98% | **52.18% / 52.17%** | 33.98% / 33.66% |
| OCNLI excluding news | 24,000 / 8,000 | 34.06% | **51.81% / 51.77%** | 34.06% / 34.25% |

These results reproduce a material premise-free cue even after connected
premise-group isolation. They do not show whether a **full** native decision
model relies on that cue, but they preclude blindly importing all labels as
robust evidence reasoning. The official SNLI card separately cites stronger
hypothesis-only baselines as a known corpus limitation.

## Admission gate and next discriminating experiment

Keep both raw corpora on HOLD. A bounded candidate pilot can proceed only
after (1) exact duplicate and all suspected protected-match premise groups
are excluded; (2) OCNLI news remains out pending its distinct source review;
(3) samples are balanced across relation, language and source groups with a
held-out premise-free shortcut diagnostic; (4) a fixed, independently sourced
three-level transfer diagnostic is reserved and overlap checked. A proposed
diagnostic is [ANLI](https://huggingface.co/datasets/facebook/anli)
development rounds, which contain separate adversarially collected NLI pairs;
they were **not downloaded or used** in this screen. It tests cross-source
NLI transfer, while an additional untouched rule/state Score diagnostic is
needed for broader decision transfer. No within-SNLI/OCNLI split alone can
establish that transfer.

If these gates pass, preregister an equal-token substitution arm from the
same own or official-Qwen start, the same native inference path and a fixed
checkpoint rule. Compare the three-level diagnostic, original typed/CSS
development views and Score class calibration. Do not select with keyed
JevArena v3 or public JevBench.

Reproduction is provided by `training/data/audit_nli_evidence_score.py`
and its contract tests. The audited script byte SHA-256 is
`b620e6c6e48baefc66ace722001dc446ea27fb9bf32eaea4339296358f85907d`.
Final private receipt SHA-256:
`d5f1c637a4193a86a1831d6e0375c357295388349a8f1c7a4287c6996485fbeb`.
The first two CPU attempts did not produce valid receipts: the prebuilt
container lacked the initial shortcut-baseline library, then the original
OCNLI file exposed three rows without genre/premise ID. The final receipt
explicitly excludes those rows and uses the frozen source and role bytes.
