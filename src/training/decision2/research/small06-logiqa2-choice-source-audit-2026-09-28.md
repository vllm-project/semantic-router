# LogiQA 2.0 Choice source: pinned CPU admission result

**Decision: TRAIN HOLD.** This is a source and tokenizer audit, not a new 0.6B
model result. No model weights, native inference, GPU, JevArena keys, public231
answers or HF private dataset were used. No LogiQA row is admitted to Decision
2.0 TRAIN, SELECT, CAL or JevArena. The existing 0.6B gradient experiment is
separate and unchanged.

## Identity, rights and audit scope

The [authors' LogiQA 2.0 repository](https://github.com/csitfun/LogiQA2.0)
was checked at Git revision `955e1d3df6c59d9bfb44d9913da1e1a27ec14e18`.
Its README (SHA-256
`5ed037a47c3f3aaa7a136dd7884e2416f8176bd9cf68ca8fbcfcf2ef08b81cc3`)
states **CC BY-NC-SA 4.0** and describes Chinese exam/practice questions,
professional English translation and annotation checks. Its `datasource.txt`
lists nine upstream exam/practice websites but supplies no per-question source
or underlying text-rights mapping. Noncommercial research use is a plausible
permitted use under the repository license; attribution, ShareAlike and source
rights still need a per-row ledger before copying a derived corpus elsewhere.
Do not redistribute source text in this repository or the public gist.

The aggregate-only audit code is
[`logiqa2_source_audit.py`](logiqa2_source_audit.py) SHA-256
`624f9309a3d368e9e02736ce4081409cee20ddc8a4c14abfe8365bf817aa27e2`
from signed code commit `012e53b600e08dd6c6545ae5b2530973071dfa40`.
Its exact `segments()` dependency from `decision_model.py` was mirrored at
SHA-256 `dda8eb55a69feba046f1dd385d342d2741d579b2eb27e916064c15a6c6c75e78`.
The CPU container used pinned `Qwen/Qwen3-0.6B-Base` tokenizer revision
`da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; `tokenizer.json` SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
The private aggregate receipt SHA-256 is
`b03b607ee92bc0c2842aae4686cfd07581933aad986ddae3833beb46f5ff336d`.
Source text and the full receipt remain in a task-owned private CPU workspace.

| Official MRC file | Rows / unique source IDs | SHA-256 |
| --- | ---: | --- |
| `train.txt` (English) | 12,567 / 12,229 | `98eb412e8ed53b3d65da5ef75b00b7a0bbdea7970c05ad699291a2a0510922de` |
| `dev.txt` (English) | 1,569 / 1,567 | `bbefb563b7ddc02640ccdc314c1315d5727dba48539d0ecdd126fa351e511b09` |
| `test.txt` (English) | 1,572 / 1,568 | `71940b37ae0184b677c253a148d57ad4e75d6113447b1563c2ca82483e4e4f8d` |
| `train_zh.txt` (Chinese) | 12,751 / 12,750 | `d87a15811cda64cb021d43cb9bc1d282424a8dfb8e35e6a7f6d6a0b36b38a54e` |
| `dev_zh.txt` (Chinese) | 1,593 / 1,593 | `a72a23160c9e12e15ea8c13e57af5032a7c37157573ebdd7e7c8e0ad34aef780` |
| `test_zh.txt` (Chinese) | 1,594 / 1,594 | `7a8db83ccb3ebdc8d5b3886fd0ad9346c7e565722d2d592987b24dd57f251853` |

Publisher DEV/TEST answer indices were read only for aggregate source-integrity
counts below. They cannot be represented as newly unopened blind labels in a
future study designed from this screen. The NLI derivative was not analyzed;
the authors say it was converted from MRC and it cannot be considered an
independent transfer source.

## Source groups, validity and shortcuts

All English rows have four options and answer indices in `0..3`; TRAIN answer
counts are `2,816 / 3,093 / 3,307 / 3,351`. English TRAIN has **338 duplicate
ID groups** containing different prompts, so `id` is not a unique row key.
Chinese TRAIN has one duplicate `example_id` with different text and answer.
Four-option/answer validity is necessary but not sufficient: English TRAIN has
30 rows with normalized duplicate option descriptions; Chinese TRAIN has 33
malformed option rows, 29 normalized duplicate-option rows and two empty
state/question rows. Quarantine these before any mapping.

On Unicode-NFKC, casefolded, whitespace-normalized complete prompts (state,
question, ordered options), English TRAIN has 11,678 groups from 12,567 rows:
883 groups repeat, and two repeated groups contain conflicting answer indices.
Chinese TRAIN has 12,170 groups from 12,751 rows: 284 repeat and ten contain
conflicting answer indices. Exact full-prompt collisions across publisher
splits are substantial:

| Language | TRAIN∩DEV | TRAIN∩TEST | DEV∩TEST |
| --- | ---: | ---: | ---: |
| English | 208 | 205 | 30 |
| Chinese | 93 | 84 | 13 |

Exact passage-only intersections are larger: English TRAIN∩DEV **234** and
TRAIN∩TEST **219**; Chinese **122** and **114**. Some identical cross-split
prompts disagree on the published answer (English TRAIN∩TEST **1**, Chinese
TRAIN∩DEV **7**, TRAIN∩TEST **3**). Thus the publisher's row splits are not
independent by content or article/passage family. A valid future split must
quarantine complete passage/prompt groups and disputed labels, then run a
separate near-duplicate and manual ambiguity check.

The English `id` and Chinese `example_id` fields do **not** constitute a
translation crosswalk. For 9,488 one-to-one same-ID TRAIN pairs, only 2,390
answer positions agree (25.2%, near the four-choice chance rate); the
DEV/TEST figures are 40/149 and 35/163. The authors' translation statement
does not let us identify paired rows from these released fields. Until a
verified original-to-translation mapping is obtained, do not put both
languages into a supposedly independent bilingual arm or count them as
distinct problem groups.

The English TRAIN `type` annotation has five positive labels, which may
co-occur: sufficient conditional 11,930, necessary conditional 5,135,
conjunctive 10,677, disjunctive 3,463 and categorical 7,885. The Chinese
files have no `type` field. These are source annotations, not independent
correctness verification for each task. They support a prospective
type-stratified blind review after group cleanup.

## Native input fit and protected-set HOLD

The exact Qwen3 0.6B segmented native `Choice` prompt contains source
passage as `state`, source question as `instructions` and four options as
candidate descriptions. Untrimmed tokenizer lengths are short:

| TRAIN language | Tokenized rows | Median / p90 / p99 / max | Total native tokens | Above 8,192 |
| --- | ---: | --- | ---: | ---: |
| English | 12,567 | 255 / 322 / 403 / 590 | 3,234,185 | 0 |
| Chinese | 12,718 | 226 / 282 / 343 / 504 | 2,887,775 | 0 |

The Chinese tokenizer count excludes only rows whose four options were
structurally invalid; it **does not** imply that all remaining rows are
admissible. This corpus can test short logical Choice supervision, not the
current model's long-input gap. No answer text, translated rationale or chat
generation was added to the native Decision contract.

Files matching the frozen rights-clean v2 TRAIN/SELECT/CAL byte hashes and a
**complete** gold-free protected prompt inventory were not located in the
available authorized source workspaces. Therefore **no exact or near overlap clearance
against TRAIN, SELECT, CAL, typed FINAL, CSS transfer or public231 is
claimed**. Do not infer safety from the source-internal audit or a zero on a
partial subset. No protected gold file was opened.

## Disposition and one discriminating next step

LogiQA 2.0 is a promising *hypothesis* for the 0.6B Choice deficit, but the
current raw files fail direct admission on identifier integrity, cross-split
duplication, unresolved bilingual pairing and unverified protected overlap.
The next CPU step is one deterministic TRAIN-only roster: quarantine invalid
options, conflicting labels and all source passage/prompt families colliding
with publisher DEV/TEST; retain a single language until crosswalk is verified;
pin the exact kept IDs and native token budget; resolve per-item rights and
check the complete protected prompt inventory with exact, near and blinded
semantic review. A fixed type/answer-position-stratified sample then needs
independent answerability review. If any of those fail, remain HOLD. Only a
new preregistered, token-matched arm against the archived 0.6B control could
test whether Choice improves without sacrificing Noul/Score or real transfer.
