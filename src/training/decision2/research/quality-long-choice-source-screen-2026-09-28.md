# QuALITY long-document Choice source: CPU admission screen

**Decision: source candidate only; TRAIN, model-quality and release HOLD.** This
screen did not run inference, train a model, read JevArena labels or select a
checkpoint. It is an independent long-document Choice hypothesis after the
0.6B official-base model regressed on Choice. It does not address its Score
regression, so a future Choice-only gain could still lose the composite.

The [official QuALITY repository](https://github.com/nyu-mll/quality) and
[paper](https://aclanthology.org/2022.naacl-main.391/) document writer-authored,
multi-annotator validated four-option questions over roughly 5,000-token
articles. The pinned repository revision was
`f84977c40dbfef70c9cab48037b7becfc8e45f73`. We used its corrected
`v1.0.1.htmlstripped` files, not earlier HTML-stripping variants. The local
aggregate-only auditor is `training/data/audit_quality_source.py`. Its initial
source-screen revision was SHA-256
`6cc156cf2d41c78162f136cca6a26a4604b462d46746f0bd63fc13b32027e8bf`,
with receipt SHA-256
`19484d676dce75b667c25a83f2dc9d36f62b0f70cd30042ec9672f4fef8f46fe`.
The follow-up native-prompt audit uses auditor SHA-256
`0e4f78bd983ee9ec8151b22ce03c9ee66019fe139b0560423c9929fc72a4119a`
and aggregate receipt SHA-256
`de5e0c230908f8e28a27a1ebf6481d6b58254439bce51df685e973ac4f17c868`.
The production `segments()` renderer SHA-256 was
`dda8eb55a69feba046f1dd385d342d2741d579b2eb27e916064c15a6c6c75e78`;
the remote mirror matched. The tokenizer file hashes are
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`
and `3c04ed3ca964ea2f6b2b5faf0dc4d31aec1cb1e8b4bcf63f402d295046b422b5` for `tokenizer.json` and
`tokenizer_config.json`, respectively. Both audits used the same exact source
files and tokenizer. The newer audit changes no old receipt or score.
No article, question, label or record ID is in this note.

| Official split | Independent articles | Four-option questions | Median article words | Raw token proxy median / p90 | Native Choice median / p90 | Native ≤8,192 | Whole articles native ≤8,192 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRAIN | 150 | 2,523 | 4,752 | 6,607 / 7,650 | 6,689 / 7,731 | 2,440 | 145 |
| DEV | 115 | 2,086 | 4,728 | 6,557 / 7,608 | 6,639 / 7,689 | 2,012 | 111 |
| TEST | 116 | 2,128 | 4,864 | 6,771 / 7,636 | 6,850 / 7,714 | 2,096 | 114 |

The raw proxy tokenizes article, question and four options but omits System One
framing. The follow-up constructs each prospective Choice row from article,
question and four option descriptions, then tokenizes the exact segmented
System One prompt without truncation. It does not read an evaluation target.
At 8,192 tokens, 2,440 of 2,523 TRAIN questions fit; the longest needs 8,685
tokens. Restricting to entire article groups leaves 145 of 150 articles and
2,440 questions before other gates. Only 592 TRAIN questions fit at 4,096.
Those are **length limits, not admitted training examples**. The official test
question labels are withheld; the auditor does not open any.

Across official splits, normalized article IDs, article text,
article–question–options payloads, URLs and title-plus-author keys had zero
exact matches. However, one normalized title alone recurs between TRAIN and
DEV with different authors. We conservatively HOLD its entire TRAIN article
group (16 questions) pending private review. After this title hold and the
8,192-token whole-article rule, the **upper bound** is 144 TRAIN article
groups and 2,424 questions. Exact metadata separation does not establish
semantic independence from other story editions, our protected panels or
pretraining exposure.

TRAIN answers by option position were 624/614/649/636, and the official
TRAIN difficulty flags were 1,272 ordinary / 1,251 hard. Source article
terms differ: 2,000 TRAIN questions appear with Project Gutenberg text,
355 with an ANC/OANC license pointer, and 168 with article-level CC BY 4.0
statements. The [authors' project page](https://nyu-mll.github.io/quality/)
states QuALITY is distributed under CC BY 4.0, covering their dataset release
and annotation terms. Its record-level `license` field separately describes the
underlying article. Those are **review buckets, not one uniform article-rights
clearance**. [Project Gutenberg's terms](https://www.gutenberg.org/policy/license.html)
distinguish works unrestricted by copyright from works shared by permission;
the [OANC license](https://anc.org/OANC/license.txt) carries its own attribution
and modified-text conditions. Before privately storing any article or
redistributing derived examples, confirm the exact work's terms and required
credit. A model's Apache license does not itself license the training text.

## Next decision before GPU

1. Record QuALITY CC BY 4.0 attribution and establish rights per underlying
   article. Exclude unresolved article buckets. Keep complete `article_id`
   groups together; hold the single TRAIN/DEV same-title collision and inspect
   duplicate story editions and near copies, not only exact hashes.
2. Build the complete gold-free protected prompt inventory for TRAIN, SELECT,
   CAL, typed DEV/FINAL, CSS pilot/15 and public231. Quarantine overlap by
   entire article group. The current audit did not locate or scan that
   inventory, so no training row is admitted.
3. Apply the measured full-prompt length gate without cropping. Audit whether
   the answer can be predicted from options/question alone and whether deleting
   the decisive article span changes the answer. Respect the source's
   multi-annotator ambiguity flags. The authors' published question-only
   baseline shows why correct answers need not imply article use.
4. Only then freeze a source-disjoint, article-group-selected substitution
   against the existing official-base control at matched native token and
   optimizer budgets. Report Choice and Score separately, CAL probability
   quality and independent transfer. Do not use official DEV/TEST or opened
   JevArena results to tune the mixture.

This is a viable **research candidate** for long-context Choice supervision,
not evidence that any Decision 2.0 model improved. It leaves the 0.6B first
package and all six release/HOLD states unchanged.
