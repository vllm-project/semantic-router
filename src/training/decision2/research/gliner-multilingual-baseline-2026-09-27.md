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

Results, model file/score receipts, and any runtime failures will be appended
after the GPU runs. No sealed FINAL or authored release gold is opened.
