# GLiNER multilingual native baseline: frozen development comparison

Status: preregistered before the native 600-prompt comparison. This is an
open-model **development baseline**, not a Decision 2.0 checkpoint, sealed
JevArena result, or independent proof of multilingual ability.

The English source is
[`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide)
at revision `7ee5da4c2415e32259bcdc0b1a7367c32ce8d6f6`, with 486,444,053
measured weights. The new multilingual source is
[`fastino/GLiNER2.5-multi-Decide`](https://huggingface.co/fastino/GLiNER2.5-multi-Decide)
at revision `6bc1d43d201b0691e733626389af8c57eea3ea68`, with 287,355,159
measured weights from safetensors tensor shapes. Both repos are Apache-2.0.
The multilingual checkpoint is a boundary-model mDeBERTa derivative, whereas
the English checkpoint uses the span-model DeBERTa-large backbone. Their
scores will be a reference comparison, **not** a matched-backbone training
ablation. Both source weights and release metadata were downloaded by the HF
CLI to the authorized experiment host; no raw weights enter Git.

The shared test is the previously frozen, human-translated XNLI/PAWS-X
**validation** development panel: 600 prompts from 100 independent English
source IDs, six translations per ID. Its prompt/target/manifest SHA-256 are
`d4f83ba17030ddb1591b1ec7da22ae851859fd2d0e2ce6020997a8ba96ffad39`,
`43eee8ac2add99bf19fe8bf09692334ab88f8d5a3bb793aa7b795b8599740b38`,
and `c9f094c3190b5efe91d29552399982a31595be29f06d5f1e2af28016ec4b754d`.
XNLI uses Choice and PAWS-X uses Noul; this panel has no Score or long-policy
evidence. Neither validation row nor translation may enter training or
checkpoint selection. Labels are held out from native inference.

The pinned GLiNER2 source commit is
`55656fbfa01d3d4a77485e1a1eeeaf682990ccdf` in the same runtime image.
The native adapter source SHA-256 is
`5b52c74978206a5f0036b844933f71815da35fef729a175ed38c8783570e21e0`.
It runs each checkpoint's published schema-conditioned exclusive classifier,
converts the native label probabilities to Choice/Noul with the same mapping,
keeps no-truncation coverage explicit, and scores every missing/invalid
answer as wrong. The English `span` path retains its encoder-declared limit;
the multilingual `boundary` path uses the checkpoint's `max_len=4096`.
A gold-free Chinese smoke returned all three finite label probabilities and
selected the intended action; a 1,045-token synthetic smoke succeeded despite
the boundary encoder's 512-position metadata. This is API compatibility, not
accuracy evidence.

For both source models, run the same 600-item panel and exact
`multilingual.parallel_score`; report each corpus/language with source-ID
pairing, invalids and matched English-language transitions. The
`multilingual.compare` scorer may report source-ID paired deltas, with no
pooled 600-independent-sample claim. Any superiority is exploratory because
the public validation set and model cards are visible, and both models may
have prior exposure. A multilingual training arm is considered only after the
separate MASSIVE TRAIN builder clears source-ID, semantic, rights and overlap
gates; no baseline result can retroactively alter this roster or formula.

## Native development result

Both pinned source checkpoints completed all 600 prompts with zero invalid
answers. The comparison uses the same adapter source SHA above and the same
frozen prompt/target hashes. Each cell is correct / independent source IDs;
the six translations of a source ID are correlated and must not be counted as
six independent experiments.

| XNLI Choice (60 IDs/language) | en | ar | de | es | fr | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLiNER2.5-Decide, 486M | 30 | 22 | 31 | 31 | 32 | 23 |
| GLiNER2.5-multi-Decide, 287M | 23 | 20 | 19 | 21 | 24 | 25 |

| PAWS-X Noul (40 IDs/language) | en | de | es | fr | ja | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLiNER2.5-Decide, 486M | 17 | 20 | 18 | 16 | 20 | 18 |
| GLiNER2.5-multi-Decide, 287M | 17 | 19 | 18 | 16 | 16 | 17 |

The multilingual source improves only Chinese XNLI by 2/60 and ties or
trails the English source in the other eleven corpus/language cells. The
largest paired gap is German XNLI, 19/60 versus 31/60 (nominal exact paired
McNemar p=.029; exploratory after multiple comparisons). Both checkpoints
struggle with these transfer tasks. The English source gets only 2/20 English
XNLI entailment examples, and both checkpoints are at or below chance on
several balanced PAWS-X languages. For context, the existing same-panel 4B
Decision 1.0 Nox and Qwen Eikos baselines score at least 41/60 on each XNLI
language and at least 23/40 on each PAWS-X language, so the apparent compact
model benefit comes with a large observed transfer gap. This does not rule
out a better trained encoder, but it does not support adopting the 287M
checkpoint as a Decision 2.0 candidate without new, same-panel evidence.

The source model may have seen similar public validation data, and its native
schema was not specialized for entailment or paraphrase decisions. The paired
comparison is a development diagnostic, not a sealed generalization claim.
No multilingual training, release candidate selection, or formal JevArena
scoring followed from it.

### Reproduction receipts

- Adapter source: `5b52c74978206a5f0036b844933f71815da35fef729a175ed38c8783570e21e0`;
  scorer: `multilingual.parallel_score` and `multilingual.compare` from the
  Decision 2.0 branch. Library source commit and model revisions are pinned
  above.
- English prediction JSONL SHA-256:
  `df7c5e638f247ca9bf97121770a57491bfed28dfa42c449509776b66b4af84c5`;
  score JSON SHA-256:
  `3cb83bef6a13256e3cda0ee71fb966c5cb788140b5d9ded8295a17cf287414ea`.
- Multilingual prediction JSONL SHA-256:
  `8ab30b2f5fa0a6de1c74e99de4c2f2d6e90a990e1d278fdc47e3ccc930d9447d`;
  score JSON SHA-256:
  `a727b6617d7a180d4b036152784cd620142e7f2e17f2c6d4e45faa5749d9d2b6`.
- Source-ID paired comparison JSON SHA-256:
  `114ad02635189dc87c328f34d77a1af09bb3dc5a371f99398c2720200568c01a`.
  Full prediction and score artifacts remain in the private experiment store.

The boundary loader emitted one compatibility warning for legacy tokenizer
metadata, but all examples ran and passed the native answer schema checks.
No sealed FINAL or authored release gold was opened.

## Follow-up preregistration: broader public development panels

The parallel diagnostic alone is not a comparable decision-model assessment.
Before further inference, freeze a two-model, four-panel run with the same
native adapter SHA-256 `5b52c74978206a5f0036b844933f71815da35fef729a175ed38c8783570e21e0`:

| Development panel | Rows | Prompt SHA-256 |
| --- | ---: | --- |
| Typed Choice/Noul/Score DEV | 1,600 | `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` |
| Three-task human-transfer CSS pilot | 1,430 | `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` |
| JevBench public subset | 231 | `642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd` |
| Decision Bench v4 text-readable | 1,041 | `41c0e4728202d800972edb31375e92e92148906856b2f20396154c8d0a9c80da` |

Rerun both pinned checkpoints on all 4,302 prompts with their respective
native context limits, no truncation, no fitted calibration and no gold in
the collector. The 30 visual-only Decision Bench cases remain N/E. Bind every
prediction file to model/adapter/prompt bytes and score with the already
versioned typed, CSS, public-tier and Decision Bench scorers; missing and
invalid answers are wrong. Report every typed family and CSS task, public
tiers, text task macro, invalids by cause, and actual parameter counts. The
comparison is exploratory public development evidence; do not mix its score
with sealed JevArena or change a release roster based on one slice. Freeze
source and gold hashes before scoring, and preserve previous English runs
instead of overwriting them.

### Four-panel result

Both source models completed all 4,302 predictions; no output file was
resumed or substituted. The native receipt verified every source input hash,
label set, model identity and recorded overflow. These are **development**
results and are not JevArena's six-axis release rank.

| Panel | English 486M | Multilingual 287M | Key validity/caveat |
| --- | ---: | ---: | --- |
| Typed DEV1,600 | 652 correct; Choice 227/800, Noul 208/400, Score 217/400 | 579 correct; Choice 181/800, Noul 190/400, Score 208/400 | 1,600 valid for both; rule and transition weaknesses remain |
| CSS pilot1,430 | 561 correct, median task macro-F1 .31035 | 484 correct, median task macro-F1 .28308 | 1,363 versus 1,427 valid; all three task F1s fall for multilingual |
| JevBench public231 | 116 correct; easy 48/48, standard 46/72, hard 22/111 | 128 correct; easy 48/48, standard 41/72, hard 39/111 | 175 versus 224 valid; this is not the official closed rank |
| Decision Bench v4 text1,041 | 301 correct; task macro .28765 | 502 correct; task macro .48224 | 448 versus 1,018 valid; 30 visual-only N/E for both |

The public-panel raw gain is dominated by coverage. The English span model
records 67 CSS, 56 JevBench and 593 Decision Bench text overflows at its
native 512-position limit; the multilingual boundary model records 3, 7 and
23 at its declared 4,096-token limit. Among the **same valid questions**, the
multilingual model scores 109/175 versus English 116/175 on JevBench, and
246/448 versus English 301/448 on Decision Bench. It recovers 19 correct
JevBench answers among 56 English-invalid questions and 256 among 593
English-invalid Decision Bench questions, yielding the raw gains despite the
common-valid loss. This is a source-architecture/context comparison, not a
matched-window or matched-training causal ablation.

The typed synthetic family detail is also unfavorable: attribute gate
164/400 versus English 206/400; rule precedence 190/400 versus 208/400; set
reconciliation 208/400 versus 217/400; transition table 17/400 versus
21/400. CSS pilot task macro-F1 is .28308 versus .31035 for discourse,
.13211 versus .29967 for implicit hate, and .49407 versus .53144 for
SemEval stance. Thus the small source's long-window coverage is useful for
architecture research, while its measured semantic accuracy and transfer do
not justify selecting it as a Decision 2.0 model. The public Decision Bench
panel is Choice-only and cannot repair this conclusion about Noul/Score.

### Four-panel byte receipts

The source collector SHA-256 remains
`5b52c74978206a5f0036b844933f71815da35fef729a175ed38c8783570e21e0`.
Its receipt implementation SHA-256 is
`c9e0bc2518d4a0109929e60d681f58788fedbe8a2215e4195d96aac79395fdff`.
The scorer source hashes for typed, CSS, public JevBench and Decision Bench
are respectively `d02a3b2bbaa08ec45928fc354532b3c3b5aef80e0a5d8e9ed6348ad6d30e2bcc`,
`cfe199a1826bb89b27c9eb746f808d74f16b46ca6585ac7b0ff7e440d44eeaca`,
`b0f3e77a705dde923e218d92841e333c04b37f8665e324a9760f691784e42557`,
and `0864db1462a671f12a266476053a8fb8e156cecd0db0f3bc99ddbfca20a25b44`.

| Source/panel | Prediction SHA-256 | Native receipt SHA-256 | Score SHA-256 |
| --- | --- | --- | --- |
| English typed | `05f0edfdefa84ae804efb3f7de9c8d96df50367b9af2279b420df6f2c16bbf04` | `5451b3203a7e6b3aa8bb41098df6c7d4023992ea20371d1273fe03a58db1d325` | `e42a9aa2c4b72a7dac9dbfb35c80a15b193aeba5cdb9fa56e025542fb78a63d8` |
| Multilingual typed | `14cdde24415e50134c2139d83aded016857e1dda4de9d0b7ffcc229c87e80e15` | `cc00081d6a542d35d76c2e866e7bf1fe532f46f051753e4ecb18fde7e4fef7f8` | `b6ff1bdc56a82e8390841bbc5d21efe5a175774288ffe4609182474e2feebb84` |
| English CSS | `651d385179f1f796e6f42388403b3d6732af2d1a901609fb63001dea5676bf73` | `7d77941c609e11ca21aeeffe942ca92356916f25c139cf1d2fd8c2c14abf022c` | `dfca77368c7d8ce17bbd9d545fbfb09c247e7c8ae00de8bc8d71a5fa8d23ed9c` |
| Multilingual CSS | `d88612f7ee70255693a6a0deb287665223013d50e9b7f4afce15de109229cc8d` | `9e3ca8f2f9a93836290cb997ff0daa343174c6b4ea04720c265b375bdd439cb9` | `736a50b3c6f8e8bd94c9e557fa92da19cf35928e25ba77b3dea33051c7c31d90` |
| English JevBench public | `a0f16691ae509eedcafcca77db2d3c2f7c1ad516363a9221d1a219be53adce2c` | `3b3d999e1f9c10c53485232bdc41e6d828f5c7479ddff4f2e751ddb1078effd4` | `2f2d6e6a93ab1a79e965a434d2e74300a8b0eb752270bcb3ca42718bca78df5c` |
| Multilingual JevBench public | `9b7797f7dbd3284b6b793d36cf00b74ae0434d1146507b7f4fe5c60ddf5f1a9d` | `7f728cf5261df1d376017bb8db08e57674c9100824fabb6bbed7ea4a7bc292d1` | `f2abd75c820853080e697d1263d13f861cf89a0bbcb583a1ad6157c01a074492` |
| English Decision Bench text | `2c204a2155af7510b34c6045d52caa155f7d07c8951ae4f0df2538fb9acb97ba` | `a71a99ecf84128c59231efb1dcf2528ea4571d301ea23c8202699c9bc2108bd8` | `2392f5f67231b302a32ae1957347acb16d625c869d7c5613f16ca839c7204590` |
| Multilingual Decision Bench text | `3c1299f7c4ea6eaed34858fd5b13481b0ba7bcc58820da407dde1289a1d79424` | `d178e4a8edcbd7a305ba7276a1f44f37928ce288ec33b46eaf721dedbd5488ec` | `2e080475ec0049e712b7873bf4e98641aa1d2c9788e8d08c5cdda134b5973827` |

All full prediction and gold artifacts remain private. No protected FINAL,
CSS15 evaluation, or authored release label was opened for this comparison.
