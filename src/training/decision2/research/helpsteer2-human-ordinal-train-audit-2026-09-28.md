# HelpSteer2 human ordinal Score source screen

**Disposition: HOLD.** This CPU-only screen did not train a model, evaluate a
model, open the NVIDIA validation split, or read protected benchmark answers.
GPU use was **0 GPU-hours**. It tests a distinct response-quality hypothesis;
it does not join the separately proposed SNLI/OCNLI evidence-score arm.

## Frozen provenance and construct

- Publisher: [NVIDIA HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2/tree/990b2711a36180dd19d9c94b8627844866f8982a), revision
  `990b2711a36180dd19d9c94b8627844866f8982a`, TRAIN archive SHA-256
  `c0d7e91d738d42e8a08070db26c4c09a9c7631308e1f0fd380ff43d130c9f713`.
  Publisher card license: CC BY 4.0, English. The original validation and
  preference add-ons were not accessed. The five fields are human-rated
  0–4 response attributes. The candidate target is **correctness**; it grades
  whether an assistant response includes pertinent facts without errors.
- This is plausibly a System One Score question about a candidate response,
  but it is **not** an annotation of exception precedence, changing state,
  missing evidence, or the project's existing three-level rule Score.
  Converting 0–4 into 0/1/2 would discard source semantics and was not done.
  The pilot's verbal grade criteria are our provisional rendering of the
  publisher's coarse rubric, not publisher-authored definitions for each
  individual level. A blind review must test the mapping.
- The matched tokenizer was the official Qwen3.5-4B Base snapshot used by the
  existing 4B study. Token counts below are raw prompt-plus-response exposure,
  **not** the final native System One wrapped input length or a training budget.

## Full TRAIN aggregate

| Property | Result |
| --- | ---: |
| Rows | 20,324 |
| Normalized prompt groups | 10,146 |
| Group multiplicity | 10,131 × 2; 14 × 4; 1 × 6 |
| Exact normalized prompt-response duplicate rows | 6 |
| Prompt characters, median / p90 / p99 / max | 238 / 2,210 / 3,124 / 4,887 |
| Response characters, median / p90 / p99 / max | 1,342 / 2,817 / 5,088 / 7,698 |
| Raw prompt + response tokens, median / p90 / p99 / max | 413 / 880 / 1,405 / 4,379 |
| Raw token exposure across TRAIN | 9,704,911 |
| Rows over 8,192 raw tokens | 0 |

| Human-rated attribute | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Helpfulness | 1,616 | 1,952 | 2,445 | 5,877 | 8,434 |
| Correctness | 1,459 | 1,979 | 2,115 | 5,080 | 9,691 |
| Coherence | 44 | 235 | 956 | 4,544 | 14,545 |
| Complexity | 622 | 6,831 | 10,866 | 1,900 | 105 |
| Verbosity | 793 | 2,829 | 12,799 | 3,338 | 565 |

Correctness grade 4 is 47.7% of rows; grades 0–2 together are 27.3%.
There are 6,776 two-response groups with unequal correctness grades. In
3,881 of the 6,774 unequal-label, unequal-length pairs, the longer response
has the higher grade (**57.3%**). The grade-0 median response is 684
characters versus 1,411 for grade 4. This is a measurable length cue, though
it is not a sufficient predictor by itself. The data does not contain a
meaningful >8K long-context arm.

## Protected-source leakage audit

The frozen gold-free inventory covered **32** accessible roles, including
rights-sensitive typed FINAL and the 15-task human-transfer prompt rosters,
plus the rights-clean v2 TRAIN, SELECT and CAL rows: **35 role files / 32,269
row comparisons**. Roles overlap and this is not an independent example count.
The role bytes were checked against their frozen SHA-256 values. The audit
read only prompt/state fields for protected rows; no benchmark gold or model
predictions were loaded. Normalized exact matching scanned all HelpSteer2 TRAIN
prompts and responses. Five-word-shingle Jaccard ≥0.60 near matching scanned
all distinct TRAIN prompts using a 16 rare-gram candidate index, followed by
exact Jaccard verification. Candidate retrieval is a heuristic, so a zero
finding is not a proof of all semantic disjointness.

**Finding:** one exact protected state match and four near matching protected
states, all in the CSS15 gold-free role. They correspond to **two normalized
HelpSteer2 prompt groups / six TRAIN responses**. These entire groups must be
quarantined before any possible training partition. No other role matched by
this exact/heuristic near scan. This finding precludes admitting unfiltered
HelpSteer2 TRAIN as source-disjoint data. Response-to-protected *near* overlap,
semantic paraphrases and ShareGPT provenance overlap have not been exhaustively
ruled out.

## Frozen private native Score quality probe

Twelve prompt groups, each with two independently rated responses, were
selected deterministically from the pinned TRAIN split. The six distinct
correctness-grade pairs `(0,4)`, `(1,3)`, `(2,4)`, `(0,2)`, `(1,4)`, `(2,3)`
each contribute one case where the higher-rated response is longer and one
where it is shorter. Prompts are 40–1,200 characters; responses are 100–2,400
characters and within a 1.5× paired length ratio. The blind packet contains
24 native System One Score questions with an explicit five-level rubric,
covering grades **4 / 4 / 6 / 4 / 6** for 0 / 1 / 2 / 3 / 4. One prompt group
uses the publisher's multi-turn delimiters and needs an explicit formatting
review. The private blind packet SHA-256 is
`cfc0df08d7bb12bcc67918f64d1fbc6f29199acb86a9e6e24a21b44505aa00af`;
the separate private key SHA-256 is
`d996c64cbfb68addaacbfbdaba5d3388bb7c7a9db2dda0c5e06a1a15cc65b097`.
Neither packet nor any raw text/record identifier is committed or copied to the
research gist. All 24 pilot inputs had zero ≥0.60 five-gram matches in the 35
protected roles. The pilot is **not admitted TRAIN, SELECT, CAL or independent
transfer evidence** until an answer-blind reviewer checks grade meaning,
ambiguity, factuality, multi-turn handling and shortcuts.

## Separate source-disjoint diagnostic design

[NIST TREC DL 2023 passage judgments](https://trec.nist.gov/data/deep2023.html)
provide a distinct graded-relevance construct. The full official qrels file
has SHA-256 `5a98a95d5714b00c5066593719e5eb7ca93dd13588d0facf7f426a4c9d6b19b8`,
**22,327 judged query-passage rows / 82 query groups**, with grades
0 / 1 / 2 / 3 = **13,866 / 4,372 / 2,259 / 1,830**. Passage level 1 means
related but does not answer; it must not be treated as an ordinary positive
relevance judgment. Unjudged passages are missing, not grade 0. The qrels
include propagated judgments for near-duplicate passages. A development
Score diagnostic should retain native 0–3 criteria, aggregate by query,
obtain the linked MS MARCO v2 passage text under its [noncommercial research
terms](https://microsoft.github.io/msmarco/), and audit exact/near source
identity against TRAIN and all protected roles. The text join, duplicate
classes, rights and overlap are **not yet verified**, so no model score is
permitted. This would test cross-source transfer to relevance, not directly
validate HelpSteer2 correctness labels.

[HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) was pinned for
metadata only at `f6d145777bcbde96137596340fab89793acd1031`; its
relative −3…3 preference labels are not absolute Score grades. Its prompt
origins may also overlap HelpSteer2. It is not the Score diagnostic here.

## Admission decision and next discriminating step

Keep HelpSteer2 on **HOLD**. First, independently review the frozen 12-group
packet without its key and decide whether the native Score mapping describes
human correctness judgments. If the construct passes, quarantine the two
matching source groups, audit the remainder for response/semantic overlap,
then build a source-group-disjoint, balanced candidate and a separately
verified NIST diagnostic. Only after these CPU gates and a matched-token
preregistration should any 0.8B/2B/4B optimizer arm be considered. The
existing control checkpoints and results remain untouched.

Reproduction: `training/data/audit_helpsteer2_score.py` and its CPU tests.
The audit created no HF dataset upload and no model release.
