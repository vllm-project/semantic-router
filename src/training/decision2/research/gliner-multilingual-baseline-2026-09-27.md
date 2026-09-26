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
