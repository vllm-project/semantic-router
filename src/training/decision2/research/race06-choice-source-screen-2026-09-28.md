# RACE as a 0.6B native Choice source: bounded CPU screen

**Decision: HOLD; no TRAIN admission, model inference or GPU use.** The
publisher's [RACE dataset page](https://www.cs.cmu.edu/~glai1/data/race/)
permits noncommercial research and restricts commercial reuse of passages and
derived data. The [original paper](https://aclanthology.org/D17-1082/) says
the source questions came from English exams and identifies separate middle-
and high-school splits. This is a human-authored **short-passage** Choice
hypothesis for the 0.6B model's evidence-reading deficit. It cannot repair
the observed Noul/Score weaknesses by itself and cannot establish long-input
transfer.

The direct publisher archive was downloaded from its official URL at SHA-256
`b2769cc9fdc5c546a693300eb9a966cec6870bd349fbc44ed5225f8ad33006e5`.
Only official TRAIN was parsed. The archive has 25,137 TRAIN article files
and 87,866 four-option questions. Two high-school files have no questions and
were excluded; the 25,135 remaining articles contain 25,133 distinct
normalized passage groups. The official TRAIN answer positions are A 19,146,
B 22,726, C 23,891 and D 22,103; any later training sample needs an
option-position control. Source/renderer code is commit `508757dcf`.

| Bounded, deterministic whole-passage sample | Middle | High |
| --- | ---: | ---: |
| Independent articles | 64 | 64 |
| Four-option questions | 254 | 223 |
| Article tokens, median / p90 / max | 250 / 341 / 440 | 365 / 514 / 797 |
| Complete native Choice tokens, median / p90 / p99 / max | 363 / 472 / 566 / 587 | 480 / 636 / 916 / 950 |
| Questions over 8,192-token cap | 0 | 0 |
| Complete article groups within cap | 64 | 64 |

The sample was selected by a fixed hash of the normalized **article group**,
independent of answer labels. Every question in those groups was rendered with
the existing System One segmented Choice prompt and the exact official
Qwen3-0.6B-Base tokenizer (`tokenizer.json` SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`).
The CPU-only renderer was checked byte-for-byte against the production renderer
in a synthetic test and verifies the production function's pinned source hash
at runtime. No article or option was truncated. These 477 sample questions
are **not** an admitted training set or a quality score. The source's short
articles make it unsuitable as the sole replacement for real long-context
evidence data.

The existing rights-clean v2 TRAIN byte hash is
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
Its 7,455-row private provenance check found no RACE name in source,
render-template or original-dataset metadata. This establishes no recorded
direct RACE import; it does **not** prove absence of reused original exam text
or paraphrases. The projected protected-input manifest was verified at SHA-256
`26bbaf82eb1c30c0f2093c70d27718fab6731ea8c80e9a451547b03bd30897e1`:
eight roles and 20,263 input-only records, including TRAIN/SELECT/CAL, typed
DEV/FINAL, human pilot/final, and the public supplement. Against each role,
the **sampled 477 questions** had zero exact raw, normalized, bounded near and
same-ID row-pair matches. The screen is lexical and sampled, not an exhaustive
semantic or original-corpus isolation proof. Twenty-seven optional inventory
roles were not covered. Protected labels were never read.

One crude question-only word-overlap heuristic gave a unique favored option
for 154/477 sample questions and selected the source answer for 36/154. This
neither establishes nor refutes deeper question-only shortcuts. A deterministic
24-pair private blind review packet was therefore created: each chosen question
appears with its article and with that article withheld; its source answer is
in a separately permission-restricted key. Blind packet and key SHA-256 are
`5a7053d6d3b7c81c19f9a4d955744b9fa5876edb38df72badd35144792c3bfe5`
and `50e53c890ac08ab015556aafb2527378c8753329197f02531eeefe54d0cdc8a6`.
The aggregate private receipt SHA-256 is
`386a570f2129142dba82dca6c8e270b97f2ddef042cd169eb2c47f418c20dee6`.
The private files have mode `0600` under a `0700` directory. No source text,
key, prompt, private path or row identifier appears in this note.

## Next bounded decision

1. Have a reviewer who has **not seen the separate key** judge whether each
   article-present prompt has a uniquely supported answer and whether the
   article-removed counterpart lacks it. Treat unclear items as failures.
   An initial triage gate of at least 18/24 genuinely passage-dependent,
   unambiguous pairs was fixed before review. The packet has not been reviewed.
2. If the triage fails, stop this source. If it passes, first resolve passage
   provenance and noncommercial/derived-use obligations, then extend the
   input-only overlap screen to the proposed complete article-group schedule
   and inspect source split/corpus reuse. Keep any restricted passages private.
3. Only after those gates, design a matched-token, source-disjoint Choice
   substitution arm against the existing official-base 0.6B control. Require
   separate Choice, Noul, Score, migration and calibration reporting; do not
   attribute a future total-score gain to this CPU source screen.

Reproduction is `training/data/audit_race06_choice_source.py` with its
synthetic tests. It refuses changed archive, tokenizer, native renderer or
protected inventory bytes and does not overwrite private results. The local
`make check` passed on the code and tests. GPU-hours: **0**.
