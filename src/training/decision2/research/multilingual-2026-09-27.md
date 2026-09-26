# Decision 2.0 multilingual development diagnosis (2026-09-27)

## Scope and decision

This is a **development diagnostic**, not a final, benchmark rank, or release
claim. No sealed final prompt or label was opened. Four frozen native typed
models were evaluated: Decision 1.0 Nox 4B, pinned source Eikos 4B, and
Decision 2.0 Eikos clean-v1 and clean-v2 4B packages. All predictions retain
model revision, package/calibration identity where applicable, input hashes,
and per-row native answers. The source Eikos and packaged adapters use their
respective recorded native runtimes, so tiny differences deserve runtime
parity caution.

**Finding:** current training and established public panels give little
evidence of broad multilingual decision ability. A simple authored seven
language panel saturated at 100% for all four models. A harder paired
human-translated validation panel exposed transfer gaps, particularly Arabic
and Chinese NLI for source Eikos, and Japanese paraphrase for every 4B model.
Clean-v2 improves Chinese NLI on these 60 base IDs but does not uniformly
beat Decision 1.0 Nox or clean-v1 across languages and tasks. Do not claim
multilingual SOTA or broaden publication score tables from this pilot.

## Corpus and panel language census

The [`multilingual.audit`](../multilingual/audit.py) script audits **all**
model-visible rows, not a sample. It counts declared `language`, Unicode
script, and `langid==1.1.6` top labels for rows with at least 60 Unicode
letters. Significant non-Latin means at least five non-Latin letters and 5%
of a row's visible alphabetic characters. It excludes labels and emits only
aggregate counts and SHA256 fingerprints. The text projection covers
instructions, state and options for training, and state/questions for eval.

| Frozen input | Rows | Declared en / zh | `langid` en / zh | Significant non-Latin rows | Audit SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| rights-clean v2 TRAIN | 7,455 | 6,085 / 1,370 | 5,953 / 1,352 | 1,225 | `002d3b0b3bfbf3f1ed462236d3e1a2e30dadea55e9d94de1e77fb8d7cb647472` |
| rights-clean v2 SELECT | 700 | 588 / 112 | 574 / 112 | 112 | `6dd0524114a94324a49cb25a76e2ce35047420bb8504b59e0c0e777bc8611c3c` |
| rights-clean v2 CAL | 700 | 600 / 100 | 584 / 100 | 100 | `45f685206828f331090baee57508538b20a40e247731f3fcb107839a70823806` |
| human/structured TRAIN | 8,360 | 6,669 / 1,691 | 6,620 / 1,672 | 1,503 | `da07691dbd5da1d47890d1300b051bdc323d52873d482a333955b7dc843bec2b` |
| original balanced TRAIN | 5,824 | 5,231 / 593 | 5,205 / 586 | 525 | `7fb9714ffb6c9f34f6ce8d5696e4fafd1a04ab101340eeb0a44979f98c337720` |

The clean-v2 TRAIN fingerprint is
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`.
Its 1,370 declared Chinese rows (18.38%) arise from programmatic/structured
sources: stage4 general composition 979, stage3 replay 220, targeted
programmatic 120, original programmatic 51. GoEmotions human training
contributes 2,800 English rows; CosmosQA 448, SQuAD 334, SNLI 272 and FLUTE
120 are English. **No clean-v2 row is declared Spanish, French, German,
Japanese or Arabic.** The human/structured 8,360 corpus has Chinese rows in
the same synthetic/structured source families; TweetEval human rows are
declared English.

| Existing public panel | Rows | Full-text `langid` top labels | Significant non-Latin | Audit SHA256 |
| --- | ---: | --- | ---: | --- |
| DEV | 1,600 | en 1,600 | 0 | `ef1d568158b214097d2fecf8863b70cc47b43666444c9f9d11684881781f4605` |
| CSS pilot | 1,430 | en 1,429; is 1 | 0 | `41d45693ecb3c0f971c6355382823e80c84cde7533f455896b859c2a827e6752` |
| JevBench public | 231 | en 231 | 0 | `bca778149985210e60149aa51887e49021cda0f07e935813e7ec6fcfee701397` |
| Decision Bench v4 text-readable | 1,041 | en 1,037; de 3; fr 1 | 0 | `ebe7802b6fb41afedf04e22dacef3a96a2369aa3dfb378c82ab3586536e91cfb` |

Decision Bench has 268 non-Latin alphabetic characters among 2.15 million
visible letters, mainly Greek symbols; no row reaches the significant
non-Latin rule. Four non-English classifier guesses in document material are
code/template candidates, not validated German/French benchmark coverage.
CSS has one Icelandic detector outlier. These panels should be described as
effectively English for multilingual score interpretation.

**Measurement boundary.** These are complete-file counts, so there is no
sampling error for declared language or Unicode character counts. They are
not a semantic-language ground truth: metadata can be wrong, Latin-script
non-English can escape the script census, and English rubric text may mask a
Chinese state. In clean-v2 TRAIN, 112 rows are too short for `langid` and
1,597 have different state-only versus full-visible language labels. The
detector's score margin is not a calibrated probability. We cannot infer a
reliable count of actual Spanish/French/German examples from zero declared
metadata alone. A bilingual manual sample is needed for that error bound.

## Two frozen gold-free development panels

1. [`multilingual.pilot`](../multilingual/pilot.py) creates 210 prompts from
   18 author-controlled base cases, translated into en/zh/es/fr/de/ja/ar,
   with Choice, Noul and Score, native option/label perturbations, protected
   entity and numeric-invariant checks. The unit of independence is 18 base
   cases. Every model scored 210/210 with 0 invalid; this is a **ceiling
   check**, not evidence of broad multilingual robustness. Prompt SHA
   `4a4873ac642dca6bfa22cbe6a89fc9beec8f534d6651603793338094f7d1cef0`;
   separate target SHA
   `9e82c86257286f292dbc2db3cb2666b546db6202e340be7173b305021b749c67`;
   manifest SHA
   `07f82d30335ecfcbaa0332e99091c1f942ef807155a87054f607f6c09daa9d59`.
   Independent native-speaker translation review is pending. The four authored
   score SHA256s (Nox, source, clean-v1, clean-v2) are respectively
   `11ec784403022e7fc235cc3e24925ae1156aada2b0e68a148f2cedd58bc0336c`,
   `f1e32d4bf3ad9284a4b56c3cc878aea4a393594fe259b0fa81703ff53a43c7b5`,
   `df9969349bacf1a47c92060f45c6027e68a88f664148845326bd8927a788b2d2`,
   `410c7b2bc5423f56d4cdee9655371cb8c1b44a4c29ea46cd2a14ad0719e84ec7`.
2. [`multilingual.parallel`](../multilingual/parallel.py) creates 600 prompts
   from 100 distinct validation source IDs: balanced [XNLI](https://arxiv.org/abs/1809.05053)
   60 (Choice; en/ar/de/es/fr/zh) and [PAWS-X](https://arxiv.org/abs/1908.11828)
   40 (Noul; en/de/es/fr/ja/zh). The official [XNLI page](https://cims.nyu.edu/~sbowman/xnli/)
   and [PAWS-X source README](https://github.com/google-research-datasets/paws/blob/master/pawsx/README.md)
   document the validation sets. HF dataset revisions are pinned to
   `facebook/xnli@b8dd5d7af51114dbda02c0e3f6133f332186418e` and
   `google-research-datasets/paws-x@4cd8187c404bda33cb1f62b49b001115862acf37`.
   The builder quarantines 59 of 2,000 PAWS-X IDs whose labels disagree
   across languages, then samples the remaining IDs by deterministic SHA256
   rank within label strata, without model outputs. XNLI language maps and
   all selected PAWS-X labels match. The panel stores source-row digests, not
   source text in Git. Native Eikos parser preflight: 600/600 valid, Choice
   360, Noul 240, at most three options. Prompt SHA
   `d4f83ba17030ddb1591b1ec7da22ae851859fd2d0e2ce6020997a8ba96ffad39`;
   target SHA
   `43eee8ac2add99bf19fe8bf09692334ab88f8d5a3bb793aa7b795b8599740b38`;
   quarantine SHA
   `27dee0665867ac012c21ef77a09c3fdcbbfdd8d08873b9df3b733000c0aabe63`;
   manifest SHA
   `c9f094c3190b5efe91d29552399982a31595be29f06d5f1e2af28016ec4b754d`.
   The panel's language-ID check recovered all 600 declared languages, but
   this does not validate translation meaning. Manual bilingual reannotation
   remains open. This panel has no Score case, so Score transfer remains an
   untested risk beyond the easy authored pilot.

Prompts and targets are separate, predictions are model-bound and hashed,
invalid native answers count wrong. Both scorers refuse input hash/ID drift.
No validation item was used to choose or calibrate these model checkpoints.
Translations of one source ID are **not** independent observations.

## Human-translated validation results

Counts are correct/independent English source IDs in each language, all with
zero invalid native answers. These are conditional on the fixed 60 XNLI and
40 PAWS-X source groups, not population scores. Models share exact prompts.

| XNLI Choice (n=60 per language) | en | ar | de | es | fr | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 1.0 Nox 4B | 51 | 42 | 48 | 50 | 45 | 47 |
| Source Eikos 4B | 51 | 41 | 49 | 49 | 44 | 41 |
| Eikos clean-v1 4B | 49 | 44 | 48 | 46 | 47 | 42 |
| Eikos clean-v2 4B | 49 | 44 | 47 | 47 | 46 | 48 |

| PAWS-X Noul (n=40 per language) | en | de | es | fr | ja | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision 1.0 Nox 4B | 28 | 27 | 27 | 29 | 24 | 29 |
| Source Eikos 4B | 30 | 31 | 31 | 29 | 23 | 27 |
| Eikos clean-v1 4B | 31 | 30 | 28 | 32 | 25 | 25 |
| Eikos clean-v2 4B | 31 | 32 | 29 | 30 | 25 | 26 |

Within the *same* XNLI base IDs, source Eikos scores 10/60 fewer correct in
Arabic and Chinese than in English (paired exact McNemar p=.0129 and .0213).
Clean-v2 scores 5/60 fewer in Arabic and 1/60 fewer in Chinese than in its
English column. Against source Eikos on the *same* Chinese IDs, clean-v2
recovers seven with no newly lost correct answers (48 vs 41; nominal exact
p=.0156); against clean-v1 it gains six (48 vs 42; nominal p=.0313). In
PAWS-X, clean-v2 Japanese is 25/40 versus its English 31/40; all Eikos
variants remain weak. Clean-v2 Chinese PAWS-X 26/40 trails Nox 29/40.

Each language comparison uses only 60 or 40 groups. The exact paired p-values
are exploratory because many model, corpus and language comparisons were
inspected; neither Chinese gain survives a conservative 12-comparison
Bonferroni threshold. Translation and label noise remain. Report **paired
transitions**, not a pooled 600-item accuracy or independent-translation CI.

| Artifact | SHA256 |
| --- | --- |
| Source Eikos parallel score | `ca551aa0de0f2992533732a56f43ab1775cf6b0485efa4e7ed60ac12addfddae` |
| Clean-v1 parallel score | `c6cdeffd46a1cb0b3eb7a652f4484d8e7d81e120bee3fcc46c1583c7c55082b9` |
| Clean-v2 parallel score | `ba65de127376cdfe73d76519fa892b2750029b5e93a7bfff0635d14f7db30c7d` |
| Decision 1.0 Nox parallel score | `2fbb0c87954441f0eb83e10b9a36c63873724e92c35849c1a860335fa9c27bc8` |
| Source Eikos → clean-v2 matched comparison | `d55badaa9c5217a203587c002a85c2b3952df5ddc5ed3502af337ea5bb19eed6` |
| Clean-v1 → clean-v2 matched comparison | `b38854860e8230b5d76ac3ed0f11ef720fc0ee452a0356b90a4de5500bed2add` |
| Decision 1.0 Nox → clean-v2 matched comparison | `b30fb1aadafcd2baa11a937d549593c39aa862bcfd023cf92cac0c328c53b641` |

Source model revisions are `llm-semantic-router/Decision-1.0-Nox-4B@0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` and
`caiovicentino1/Eikos-4B@582ffb13f19a4da3f455e3db198584190bd7755b`.
The clean-v1 package model/calibration SHA256s are
`92ba0384f3dcd94b73e339a46ff865ef3ed94a3e9624a91de045989c19cd0b84` /
`ef1fdc358a0e7fc46783fe5b108586b5e1799da2662e1bbb12ca9e7b92858bab`;
clean-v2 uses
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39` /
`6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`.

## Recommended next training experiment

Run a **bounded, explicitly multilingual continuation arm** rather than
assuming the base model naturally retains each language. Keep the current
clean-v2 adapter as a control. Try multilingual update shares of 10% and 20%,
with the remaining 90%/80% replayed from the existing English/Chinese
training mix and the same total optimizer steps. Within multilingual updates,
start with roughly equal quotas across zh/es/fr/de/ja/ar, each with a matched
English anchor. Count *source groups* rather than translations as independent
rows; prioritize natural Chinese, Arabic and Japanese meaning judgments.
Retain Choice/Noul/Score balance and option/label perturbations.
Avoid training exclusively on translated labels: use bilingual review on a
stratified sample, agreement checks, and contrastive minimal pairs.

The safest initial data source is newly authored or carefully translated
rights-clean-v2 *training* source groups with attribution, semantic checks,
and source-ID deduplication against all validation/final families. Additional
public training splits need a separate rights and overlap audit; **XNLI and
PAWS-X validation rows used here must stay out of training and model
selection**. Freeze independent multilingual SELECT/CAL and a new held-out
test before tuning. Keep existing English DEV/CSS, JevBench and Decision
Bench as regression guardrails. Promote a multilingual arm only if it
improves paired Arabic/Chinese NLI and Japanese paraphrase without material
English, structured, or synthetic regression; recheck native Score on a
harder multilingual set before a broad claim.

## Reproduction and checks

The module [README](../multilingual/README.md) gives local commands.
`multilingual.parallel_score` and `multilingual.compare` verify manifest,
prompt, target, prediction input and native answer identity. Six unit tests
and the 600-row Eikos parser preflight passed; the four model evaluations
returned 0 invalid answers. Raw validation text remains in local private
runtime artifacts, not this repository or a public model bundle.
